# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backend selection: choose how inference runs *before* building a model.

The ordering matters. ``Model(settings)`` immediately calls into Transformers, so
by the time a runtime exists it is already too late to route DeepSeek V4.1 Flash
anywhere else. Everything here therefore happens first, and it reads only the raw
config -- never ``AutoConfig.from_pretrained``, which is precisely the call that
fails for ``deepseek_v41``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import BackendMethod, Settings
from .model_loading import is_deepseek_v41_config
from .runtime import LocalModelRuntime, ModelRuntime, RuntimeCapabilities

if TYPE_CHECKING:
    from .model import Model

__all__ = [
    "RuntimeHandle",
    "create_runtime",
    "read_raw_model_config",
    "select_backend",
]


@dataclass(frozen=True, slots=True)
class RuntimeHandle:
    """The selected runtime, its capabilities, and any Transformers model.

    ``model`` is ``None`` for every backend that does not run inference
    in-process. Code that genuinely needs a Transformers model -- tokenizer and
    processor serialization, hub upload, lm-eval benchmarks -- must go through
    :meth:`require_model`, which fails with an explanation instead of an
    ``AttributeError``.
    """

    runtime: ModelRuntime
    capabilities: RuntimeCapabilities
    model: Model | None

    @property
    def uses_transformers(self) -> bool:
        return self.model is not None

    def require_model(self, action: str) -> Model:
        if self.model is None:
            raise RuntimeError(
                f"{action} needs an in-process transformers model, which the "
                f"{self.capabilities.backend_name!r} backend does not provide. "
                "Tokenizer/processor serialization, hub upload, and lm-eval "
                "benchmarks are transformers-only."
            )
        return self.model


def read_raw_model_config(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Read a model's config without instantiating its architecture.

    A local checkpoint is read straight off disk. A Hub identifier goes through
    ``PretrainedConfig.get_config_dict``, which fetches and parses ``config.json``
    and returns a plain dict -- it never resolves ``model_type`` to a class, so
    it works for architectures Transformers cannot build.
    """

    path = Path(model).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"model config does not exist: {config_path}")
        with config_path.open("rb") as stream:
            data = json.loads(stream.read())
        if not isinstance(data, dict):
            raise TypeError(f"model config must be a JSON object: {config_path}")
        return dict(data)

    # Imported lazily so this module stays importable without Transformers.
    from transformers import (  # noqa: PLC0415 - deliberate lazy import
        PretrainedConfig,
    )

    kwargs: dict[str, Any] = {"revision": revision} if revision else {}
    config, _ = PretrainedConfig.get_config_dict(model, **kwargs)
    return dict(config)


def select_backend(settings: Settings) -> BackendMethod:
    """Resolve the effective backend, reading the raw config only when needed."""

    if settings.model_backend != BackendMethod.AUTO:
        return settings.model_backend

    raw = read_raw_model_config(settings.model, revision=settings.model_commit)
    if is_deepseek_v41_config(raw):
        return BackendMethod.VLLM_DEEPSEEK_V41
    return BackendMethod.TRANSFORMERS


def create_runtime(
    settings: Settings,
    *,
    runtime_factory: Callable[[Model], ModelRuntime] | None = None,
) -> RuntimeHandle:
    """Build the runtime for ``settings``, selecting the backend first.

    For the Transformers backend this is the historical path, unchanged: build
    the model, then wrap it (optionally with the DGX coordinator runtime). For
    DeepSeek V4.1 Flash no Transformers model is constructed at all.
    """

    backend = select_backend(settings)

    if backend == BackendMethod.TRANSFORMERS:
        from .model import Model  # noqa: PLC0415 - lazy, keeps vLLM path clean

        model = Model(settings)
        factory = runtime_factory or LocalModelRuntime
        runtime = factory(model)
        return RuntimeHandle(
            runtime=runtime,
            capabilities=runtime.capabilities,
            model=model,
        )

    if backend == BackendMethod.VLLM_DEEPSEEK_V41:
        from .deepseek_v41_runtime import (  # noqa: PLC0415
            DeepSeekV41Runtime,
        )

        if runtime_factory is not None:
            raise ValueError(
                "the vllm_deepseek_v41 backend owns distributed execution and "
                "does not accept a runtime factory; vLLM runs TP, not Heretic"
            )
        runtime = DeepSeekV41Runtime(settings)
        return RuntimeHandle(
            runtime=runtime,
            capabilities=runtime.capabilities,
            model=None,
        )

    raise ValueError(f"unsupported model backend: {backend!r}")
