# SPDX-License-Identifier: AGPL-3.0-or-later

"""DeepSeek V4.1 Flash backend: Heretic's optimization loop over a vLLM service.

Heretic never loads this model itself. ``transformers`` implements no
``deepseek_v41`` architecture, so this backend hands model loading, tensor
parallelism, prompt encoding, generation, hidden-state extraction and LoRA
application to vLLM, and keeps only what Heretic is actually for: datasets,
refusal-direction computation, Optuna search, abliteration parameter selection,
scorer orchestration, directional-LoRA generation, and export.

The class deliberately does **not** subclass :class:`heretic.model.Model`.

Residual capture point
----------------------
DeepSeek V4.1 carries its residual stream as ``hc_mult = 4`` parallel copies
(Hyper-Connections). From the reference implementation's ``Block.forward``::

    attn_pre, attn_post, attn_comb = self.hc_mixes(x, ...)
    x = self.hc_pre(x, pre_mix)     # collapse [b, s, hc, d] -> [b, s, d]
    x = self.attn(...)              # attention sees the COLLAPSED stream
    x = self.hc_post(x, residual, attn_post, attn_comb)

and inside ``Attention.forward``::

    x = self.wo_b(o.flatten(2))     # -> [b, s, dim]

So ``wo_b`` maps ``(n_groups * o_lora_rank) -> dim`` and writes into the
**collapsed** residual stream of width ``dim = 5120``. The refusal direction that
abliterates ``wo_b`` must therefore live in that same collapsed space, and its
length must equal ``wo_b.out_features = 5120``.

What we capture is the **post-collapse, pre-attention** hidden state: the tensor
immediately after ``hc_pre`` and immediately before the attention sublayer. It is
post-collapse (not the ``hc_mult``-wide stream), it is float32, its
dimensionality is ``(batch, 5120)``, and it matches ``wo_b``'s output dimension
exactly -- which is the requirement, because abliteration subtracts a component
from the projection's *output*.

This is intentionally **not** the generic "hidden state after layer N" that eagle
auxiliary hidden states usually expose, and it is not assumed to be. The runtime
validates the captured width against ``wo_b.out_features`` and fails closed if
they disagree, rather than silently ablating in the wrong basis.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from .abliteration_math import compute_directional_lora, layer_ablation_weight
from .deepseek_v41_targets import (
    V41_COMPONENT,
    TargetCache,
    V41TargetPlan,
    discover_targets,
)
from .runtime import ModelRuntime, RuntimeCapabilities
from .utils import Prompt

if TYPE_CHECKING:
    from .config import Settings
    from .model import AbliterationParameters

BACKEND_NAME = "vllm_deepseek_v41"

#: One conversation in OpenAI chat format.
#:
#: This is not a stylistic choice. Verified against the live deployment: this
#: model's raw ``/v1/completions`` surface returns an empty string with
#: ``finish_reason="stop"`` after one token, because the prompt is never wrapped
#: in the native conversation format. Only the chat surface reaches the
#: deployment's ``tokenizer_mode="deepseek_v41"`` encoder.
ChatMessages = list[dict[str, str]]

#: Layer whose residual width we validate capture against.
_PROBE_LAYER = 0


class BackendUnavailable(RuntimeError):
    """The configured vLLM deployment is not usable."""


class ExactLogitsUnavailable(BackendUnavailable):
    """The deployment cannot return dense full-vocabulary raw logits.

    Raised instead of approximating: a top-K KL divergence is a different
    quantity, and silently substituting it would corrupt every reported score.
    """


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]


class VllmTransport(Protocol):
    """Everything this backend needs from a vLLM deployment.

    Kept as a narrow protocol so the runtime is testable without a GPU, and so
    the HTTP transport can be swapped for an in-process engine or a different
    deployment shape without touching Heretic's optimization logic.
    """

    def generate(
        self,
        messages: list[ChatMessages],
        *,
        max_tokens: int,
        temperature: float,
        lora_name: str | None,
    ) -> list[GenerationResult]: ...

    def raw_logits(
        self, messages: list[ChatMessages], *, lora_name: str | None
    ) -> Tensor: ...

    def hidden_states(
        self,
        messages: list[ChatMessages],
        *,
        layer_ids: tuple[int, ...],
        lora_name: str | None,
    ) -> Tensor: ...

    def load_lora(self, name: str, path: Path) -> None: ...

    def unload_lora(self, name: str) -> None: ...


class HttpVllmTransport:
    """OpenAI-compatible HTTP transport for a vLLM deployment.

    Two capabilities are *not* part of the OpenAI surface and must be served by
    the deployment: dense raw logits (``logprobs=-1`` with
    ``logprobs_mode="raw_logits"``) and per-layer hidden states. Both are
    requested through the same base URL; a deployment that cannot answer them
    fails loudly rather than degrading.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        capture_layer_path: str = "/heretic/hidden_states",
        load_lora_path: str = "/v1/load_lora_adapter",
        unload_lora_path: str = "/v1/unload_lora_adapter",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.capture_layer_path = capture_layer_path
        self.load_lora_path = load_lora_path
        self.unload_lora_path = unload_lora_path

    # -- plumbing ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - configured base URL
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                decoded = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise BackendUnavailable(
                f"vLLM request to {path} failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise BackendUnavailable(
                f"vLLM at {self.base_url} is unreachable: {error.reason}"
            ) from error
        if not isinstance(decoded, dict):
            raise BackendUnavailable(f"vLLM returned a non-object body from {path}")
        return decoded

    @staticmethod
    def _extra_body(lora_name: str | None) -> dict[str, Any]:
        return {"lora_name": lora_name} if lora_name else {}

    # -- protocol ---------------------------------------------------------

    def generate(
        self,
        messages: list[ChatMessages],
        *,
        max_tokens: int,
        temperature: float,
        lora_name: str | None,
    ) -> list[GenerationResult]:
        body = self._post(
            "/v1/chat/completions",
            {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **self._continuation(messages),
                **self._extra_body(lora_name),
            },
        )
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != len(messages):
            raise BackendUnavailable(
                "vLLM returned an unexpected number of completions: "
                f"expected {len(messages)}, got "
                f"{len(choices) if isinstance(choices, list) else 'none'}"
            )
        results: list[GenerationResult] = []
        for choice in sorted(choices, key=lambda item: item.get("index", 0)):
            message = choice.get("message")
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise BackendUnavailable("vLLM chat completion is missing content")
            token_ids: tuple[int, ...] = ()
            token_ids_raw = choice.get("token_ids")
            if isinstance(token_ids_raw, list):
                token_ids = tuple(int(token) for token in token_ids_raw)
            results.append(GenerationResult(text=text, token_ids=token_ids))
        return results

    @staticmethod
    def _continuation(messages: list[ChatMessages]) -> dict[str, Any]:
        """Continue an assistant turn when Heretic supplies a response prefix.

        Heretic appends ``response_prefix`` so that scoring happens at the point
        where responses start to differ. In chat form the faithful equivalent is
        a final assistant message that the model continues, which vLLM exposes as
        ``continue_final_message``.
        """

        if messages and messages[-1] and messages[-1][-1].get("role") == "assistant":
            return {"continue_final_message": True, "add_generation_prompt": False}
        return {}

    def raw_logits(
        self, messages: list[ChatMessages], *, lora_name: str | None
    ) -> Tensor:
        """Dense ``(batch, vocab)`` raw first-token logits.

        ``logprobs=-1`` with ``logprobs_mode="raw_logits"`` is the only
        configuration that yields the full, unprocessed vocabulary. Anything less
        is rejected by :meth:`_reconstruct_logits`.
        """

        body = self._post(
            "/v1/chat/completions",
            {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": -1,
                **self._continuation(messages),
                **self._extra_body(lora_name),
            },
        )
        return self._reconstruct_logits(body, len(messages))

    def _reconstruct_logits(self, body: dict[str, Any], batch: int) -> Tensor:
        """Rebuild a dense ``(batch, vocab)`` tensor in token-ID order.

        The contract this backend requires is one dense raw-logit vector per
        prompt, indexed by token id, covering the *entire* vocabulary. A top-K
        distribution and a processed (post-sampler) distribution are both
        rejected here rather than silently substituted, because the KL divergence
        scorer would otherwise report a different quantity under the same name.

        Whether a given vLLM build actually serves this shape has **not** been
        validated against a live deployment yet; see ``docs/BLOCKERS.md``.
        """

        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != batch:
            raise ExactLogitsUnavailable(
                "vLLM did not return one choice per prompt for a logprobs request"
            )

        vocab_size = body.get("vocab_size")
        rows: list[Tensor] = []
        for choice in sorted(choices, key=lambda item: item.get("index", 0)):
            content = choice.get("logprobs")
            if not isinstance(content, dict):
                raise ExactLogitsUnavailable(
                    "vLLM returned no logprobs. The deployment must expose "
                    "logprobs=-1 with logprobs_mode='raw_logits'."
                )
            dense = content.get("raw_logits")
            if not isinstance(dense, list) or not isinstance(vocab_size, int):
                raise ExactLogitsUnavailable(
                    "the deployment did not return a dense full-vocabulary raw "
                    "logit vector. Configure max_logprobs=-1 and "
                    "logprobs_mode='raw_logits' on the vLLM engine. A top-K "
                    "distribution is not an acceptable substitute for KL "
                    "divergence."
                )
            row = torch.tensor(dense, dtype=torch.float32)
            if row.numel() != vocab_size:
                raise ExactLogitsUnavailable(
                    f"raw logits carry {row.numel()} entries but the vocabulary "
                    f"has {vocab_size} tokens"
                )
            rows.append(row)

        logits = torch.stack(rows)
        if not torch.isfinite(logits).all():
            raise ExactLogitsUnavailable("raw logits contain non-finite values")
        return logits

    def hidden_states(
        self,
        messages: list[ChatMessages],
        *,
        layer_ids: tuple[int, ...],
        lora_name: str | None,
    ) -> Tensor:
        body = self._post(
            self.capture_layer_path,
            {
                "model": self.model_name,
                "messages": messages,
                "layer_ids": list(layer_ids),
                "position": "last",
                **self._extra_body(lora_name),
            },
        )
        states = body.get("hidden_states")
        if not isinstance(states, list) or len(states) != len(messages):
            raise BackendUnavailable(
                "hidden-state endpoint returned an unexpected batch size"
            )
        per_prompt = [torch.tensor(row, dtype=torch.float32) for row in states]
        stacked = torch.stack(per_prompt)
        if stacked.dim() != 3 or stacked.shape[1] != len(layer_ids):
            raise BackendUnavailable(
                "hidden-state endpoint must return (batch, layers, dim); got "
                f"{tuple(stacked.shape)} for {len(layer_ids)} requested layers"
            )
        return stacked

    def load_lora(self, name: str, path: Path) -> None:
        self._post(
            self.load_lora_path,
            {"lora_name": name, "lora_path": str(path)},
        )

    def unload_lora(self, name: str) -> None:
        self._post(self.unload_lora_path, {"lora_name": name})


class DeepSeekV41Runtime(ModelRuntime):
    """Heretic's :class:`ModelRuntime` over a vLLM DeepSeek V4.1 deployment."""

    # vLLM owns distributed execution; Heretic is a single client.
    distributed = False

    def __init__(
        self,
        settings: Settings,
        *,
        transport: VllmTransport | None = None,
        plan: V41TargetPlan | None = None,
    ) -> None:
        self.settings = settings
        self._is_shutdown = False

        checkpoint = settings.vllm_checkpoint_directory or settings.model
        self.plan = plan if plan is not None else discover_targets(checkpoint)
        self._cache = TargetCache(self.plan)

        self._lora_name = settings.vllm_lora_name
        self._lora_directory = Path(settings.vllm_lora_directory) / self._lora_name
        self._active_lora: str | None = None
        self._last_state_dict: dict[str, Tensor] | None = None
        self._last_rank = 1

        if transport is not None:
            self._transport: VllmTransport = transport
        else:
            if not settings.vllm_base_url:
                raise BackendUnavailable(
                    "the vllm_deepseek_v41 backend requires vllm_base_url "
                    "(or model_backend = 'transformers')"
                )
            self._transport = HttpVllmTransport(
                settings.vllm_base_url,
                settings.vllm_model_name or settings.model,
                api_key=settings.vllm_api_key,
                timeout=float(settings.vllm_timeout_seconds),
            )

    # -- capabilities -----------------------------------------------------

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            backend_name=BACKEND_NAME,
            layer_count=self.plan.layer_count,
            abliterable_components=(V41_COMPONENT,),
            distributed=False,
            # The deployment is required to serve dense raw logits; if it cannot,
            # get_logits fails closed rather than approximating.
            supports_exact_logits=True,
            # V1 emits a LoRA adapter. A quantization-preserving standalone
            # checkpoint is not yet validated for this model.
            supports_adapter_export=True,
            supports_merged_export=False,
        )

    # -- lifecycle --------------------------------------------------------

    def _require_active(self) -> None:
        if self._is_shutdown:
            raise RuntimeError("model runtime has been shut down")

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._deactivate_lora()
        self._cache.close()
        self._is_shutdown = True

    def _deactivate_lora(self) -> None:
        if self._active_lora is None:
            return
        name, self._active_lora = self._active_lora, None
        self._transport.unload_lora(name)

    def reset_model(self, model: str | None = None) -> None:
        """Return inference to the exact baseline by deactivating the adapter.

        This never reloads the ~510 GB base checkpoint: unloading a LoRA leaves
        the base weights untouched, so a trial costs one small adapter load
        rather than a full model reload.
        """

        self._require_active()
        if model is not None and model != self.settings.model:
            raise ValueError(
                "the vllm_deepseek_v41 backend cannot switch base models in process"
            )
        self._deactivate_lora()

    # -- abliteration -----------------------------------------------------

    def get_layer_direction(
        self,
        residual_directions: Tensor,
        layer_index: int,
        direction_index: float | None,
    ) -> Tensor:
        """Resolve the refusal direction for one layer, explicitly.

        Heretic's historical convention stores ``layer_count + 1`` residual rows,
        where row 0 is the embedding output and row ``N + 1`` is layer ``N``. That
        convention exists because a Transformers model exposes the embedding
        output as a hidden state. This backend has no such tensor, and fabricating
        one would put a fake vector in the ablation path.

        So the mapping here is explicit and different: ``residual_directions`` is
        indexed **directly by layer**, shaped ``(layer_count, dim)``.

        ``direction_index`` keeps Heretic's meaning -- a continuous position
        along the layer axis. ``None`` uses each layer's own direction; otherwise
        the value is interpolated between the two neighbouring layers and the
        result is L2-normalized, exactly as the Transformers path does after
        accounting for its embedding offset.
        """

        if residual_directions.dim() != 2:
            raise ValueError(
                "residual directions must be (layer_count, dim); got shape "
                f"{tuple(residual_directions.shape)}"
            )
        expected = self.plan.layer_count
        if residual_directions.shape[0] != expected:
            raise ValueError(
                "this backend indexes residual directions directly by layer and "
                f"expects {expected} rows (it has no embedding row); got "
                f"{residual_directions.shape[0]}. Do not pad with a placeholder."
            )
        if residual_directions.shape[1] != self.plan.output_dimension:
            raise ValueError(
                f"residual directions have width {residual_directions.shape[1]}, "
                f"but {V41_COMPONENT} writes into a space of width "
                f"{self.plan.output_dimension}"
            )

        if direction_index is None:
            if not 0 <= layer_index < expected:
                raise ValueError(f"layer index out of range: {layer_index}")
            return residual_directions[layer_index]

        if not 0.0 <= direction_index <= expected - 1:
            raise ValueError(
                f"direction_index must be within [0, {expected - 1}], got "
                f"{direction_index}"
            )

        weight, index = math.modf(direction_index)
        lower = int(index)
        upper = min(lower + 1, expected - 1)
        return F.normalize(
            residual_directions[lower].lerp(residual_directions[upper], weight),
            p=2,
            dim=0,
        )

    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        """Build this trial's adapter and activate it in vLLM.

        Only ``attn.o_proj`` -- V4.1's ``wo_b`` -- is ever touched. Routed
        experts, shared experts, Engram, the vision encoder, the aligner,
        embeddings, MTP and DSpark are not read and not modified.
        """

        self._require_active()

        unsupported = set(parameters) - {V41_COMPONENT}
        if unsupported:
            raise ValueError(
                "the DeepSeek V4.1 backend supports only "
                f"{V41_COMPONENT!r}; got {sorted(unsupported)}. Experts, Engram, "
                "vision and MTP are out of scope."
            )
        if V41_COMPONENT not in parameters:
            raise ValueError(f"missing abliteration parameters for {V41_COMPONENT}")

        params = parameters[V41_COMPONENT]
        rank = self._adapter_rank()

        self._deactivate_lora()

        state_dict: dict[str, Tensor] = {}
        ablated_layers = 0
        for target in self.plan.targets:
            strength = layer_ablation_weight(
                target.layer_index,
                max_weight=params.max_weight,
                max_weight_position=params.max_weight_position,
                min_weight=params.min_weight,
                min_weight_distance=params.min_weight_distance,
            )
            if strength is None or strength == 0:
                continue

            direction = self.get_layer_direction(
                residual_directions, target.layer_index, direction_index
            )
            weight = self._cache.read_dequantized(target)
            factors = compute_directional_lora(
                weight,
                direction,
                strength=strength,
                normalization=self.settings.row_normalization.value,
                rank=rank,
                seed=self.settings.seed,
            )
            # PEFT/vLLM naming for a LoRA on this projection.
            base = f"base_model.model.{target.weight_name[: -len('.weight')]}"
            state_dict[f"{base}.lora_A.weight"] = factors.a.contiguous()
            state_dict[f"{base}.lora_B.weight"] = factors.b.contiguous()
            ablated_layers += 1

        if ablated_layers == 0:
            # Nothing to apply; leaving the baseline active is correct.
            return

        self._last_state_dict = state_dict
        self._last_rank = rank
        self._write_adapter(state_dict, rank)
        self._transport.load_lora(self._lora_name, self._lora_directory)
        self._active_lora = self._lora_name

    def _adapter_rank(self) -> int:
        from .config import RowNormalization

        if self.settings.row_normalization != RowNormalization.FULL:
            return 1
        return self.settings.full_normalization_lora_rank

    def _adapter_config(self, rank: int) -> dict[str, object]:
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "bias": "none",
            # The physical V4.1 projection this component maps to.
            "target_modules": ["wo_b"],
            "base_model_name_or_path": self.settings.model,
        }

    def _write_adapter(self, state_dict: dict[str, Tensor], rank: int) -> None:
        from safetensors.torch import save_file

        self._lora_directory.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(self._lora_directory / "adapter_model.safetensors"))
        (self._lora_directory / "adapter_config.json").write_text(
            json.dumps(self._adapter_config(rank), indent=2),
            encoding="utf-8",
        )

    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
        """Write the selected trial's adapter in PEFT layout.

        This is the export V1 supports for DeepSeek V4.1. It is small -- the
        factors for 40 projections -- and never rewrites the base checkpoint.
        """

        self._require_active()
        if self._last_state_dict is None:
            raise RuntimeError(
                "no abliteration has been applied, so there is no adapter to save"
            )
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        save_file(
            dict(self._last_state_dict),
            str(target / "adapter_model.safetensors"),
        )
        (target / "adapter_config.json").write_text(
            json.dumps(self._adapter_config(self._last_rank), indent=2),
            encoding="utf-8",
        )

    def save_merged(self, directory: str, *, max_shard_size: int | str) -> None:
        """Refused on purpose.

        Producing a merged or standalone V4.1 checkpoint means rewriting the
        released FP8 E4M3 weight with 32x32 ue8m0 block scales, plus every
        related scale tensor, without disturbing routed experts, Engram, vision,
        MTP or DSpark. That has not been validated, and emitting an unvalidated
        510 GB checkpoint is far worse than declining: the corruption would be
        silent and only visible as degraded generations. Export the adapter.
        """

        raise NotImplementedError(
            "the vllm_deepseek_v41 backend does not support merged or standalone "
            "checkpoint export yet. Quantization-preserving FP8 re-quantization "
            "of the 40 attn.wo_b targets has not been validated for this "
            "checkpoint, so emitting one would risk a silently corrupted model. "
            "Use the adapter export strategy instead."
        )

    # -- inference --------------------------------------------------------

    def _render(self, prompts: list[Prompt]) -> list[ChatMessages]:
        """Render prompts as chat messages for the native V4.1 encoder.

        ``apply_chat_template`` is never called here: the checkpoint ships no
        Jinja template. Prompt encoding belongs to the deployment's
        ``tokenizer_mode="deepseek_v41"`` path, which is reachable only through
        the chat-completions surface.

        Heretic's semantics are preserved -- system prompt, then user prompt --
        and ``response_prefix`` becomes a final assistant turn to continue.
        """

        rendered: list[ChatMessages] = []
        for prompt in prompts:
            messages: ChatMessages = []
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.user})
            if self.settings.response_prefix:
                messages.append(
                    {"role": "assistant", "content": self.settings.response_prefix}
                )
            rendered.append(messages)
        return rendered

    def get_responses_once(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self._require_active()
        results = self._transport.generate(
            self._render(prompts),
            max_tokens=1,
            temperature=0.0,
            lora_name=self._active_lora,
        )
        return [result.text for result in results]

    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self._require_active()
        responses: list[str] = []
        for start in range(0, len(prompts), self.settings.batch_size):
            batch = prompts[start : start + self.settings.batch_size]
            results = self._transport.generate(
                self._render(batch),
                max_tokens=self.settings.max_response_length,
                temperature=0.0,
                lora_name=self._active_lora,
            )
            responses.extend(result.text for result in results)
        return responses

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        """Dense ``(batch, vocab)`` raw first-token logits.

        Fails closed if the deployment cannot supply the full raw vocabulary.
        KL divergence over a top-K approximation is a different quantity and is
        never substituted.
        """

        self._require_active()
        blocks = []
        for start in range(0, len(prompts), self.settings.batch_size):
            batch = prompts[start : start + self.settings.batch_size]
            blocks.append(
                self._transport.raw_logits(
                    self._render(batch), lora_name=self._active_lora
                )
            )
        logits = torch.cat(blocks, dim=0)
        if logits.dim() != 2 or logits.shape[0] != len(prompts):
            raise ExactLogitsUnavailable(
                f"expected dense (batch, vocab) logits, got {tuple(logits.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise ExactLogitsUnavailable("raw logits contain non-finite values")
        return logits.float()

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        """Prompt-end hidden states for every layer, in the ``wo_b`` space.

        Shape is ``(prompt, layer, dim)`` with ``dim == wo_b.out_features``. No
        embedding row is included: this backend indexes residuals by layer.
        """

        self._require_active()
        layer_ids = tuple(target.layer_index for target in self.plan.targets)
        blocks = []
        for start in range(0, len(prompts), self.settings.batch_size):
            batch = prompts[start : start + self.settings.batch_size]
            captured = self._transport.hidden_states(
                self._render(batch),
                layer_ids=layer_ids,
                lora_name=self._active_lora,
            )
            if captured.shape[2] != self.plan.output_dimension:
                raise BackendUnavailable(
                    "captured residual width "
                    f"{captured.shape[2]} does not match the dimension "
                    f"{V41_COMPONENT} writes into "
                    f"({self.plan.output_dimension}); the capture point is wrong"
                )
            blocks.append(captured.to(torch.float32))

        residuals = torch.cat(blocks, dim=0)
        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
        return residuals

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")
        running: Tensor | None = None
        count = 0
        for start in range(0, len(prompts), self.settings.batch_size):
            batch = prompts[start : start + self.settings.batch_size]
            block = self.get_residuals(batch)
            block_sum = block.sum(dim=0, dtype=torch.float64).cpu()
            running = block_sum if running is None else running + block_sum
            count += block.shape[0]
        assert running is not None
        return (running / count).to(torch.float32)
