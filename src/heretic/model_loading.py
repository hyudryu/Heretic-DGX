# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .engram_disk import DEFAULT_BLOCK_SIZE, EngramTableLayout

_LAGUNA_DGX_TP_PLAN = {
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.gate_up_proj_scale_inv": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts.down_proj_scale_inv": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
}

# DeepSeek V4.1 Flash's Engram tables are what make the model need the cluster
# in the first place: measured from the released checkpoint they are 189.1 GiB
# on disk (two `rows x 256` fp8 tables plus ue8m0 scales), or 47.3 GiB per rank
# at TP4, on top of a 71.5 GiB-per-rank backbone. They are never ablated, so the
# disk-backed path replaces them with one-row placeholders and reads rows on
# demand (see heretic.engram_disk).
DEEPSEEK_V41_MODEL_TYPE = "deepseek_v41"
DEEPSEEK_V41_TEXT_MODEL_TYPE = "deepseek_v41_text"
_ENGRAM_TENSOR_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")


def complete_laguna_dgx_tp_plan(config: object) -> object:
    """Add the TP entries omitted by Laguna S 2.1's shipped custom config."""

    current = getattr(config, "base_model_tp_plan", None)
    if not isinstance(current, dict):
        raise TypeError("Laguna config must provide a base_model_tp_plan dictionary")
    config.base_model_tp_plan = {**current, **_LAGUNA_DGX_TP_PLAN}
    return config


def _nested_field(value: object, name: str) -> object:
    """Read a field from either a config object or a raw config dict."""

    found = getattr(value, name, None)
    if found is None and isinstance(value, dict):
        found = value.get(name)
    return found


def _engram_source(config: object) -> object:
    """The config object that carries the Engram fields.

    A multimodal checkpoint nests the language model under ``text_config``.
    """

    text_config = _nested_field(config, "text_config")
    return text_config if text_config is not None else config


def is_deepseek_v41_config(config: object) -> bool:
    """Whether a raw config dict or resolved config describes V4.1 Flash.

    The released checkpoint is multimodal, so the language-model type lives
    under ``text_config``. Both the outer and the nested type are checked: the
    outer one is authoritative when it matches, but an outer wrapper type must
    not mask a nested match.
    """

    if _nested_field(config, "model_type") == DEEPSEEK_V41_MODEL_TYPE:
        return True
    return (
        _nested_field(_engram_source(config), "model_type")
        == DEEPSEEK_V41_TEXT_MODEL_TYPE
    )


def is_engram_tensor(name: str) -> bool:
    """Whether a checkpoint tensor belongs to an Engram n-gram table."""

    return name.endswith(_ENGRAM_TENSOR_SUFFIXES)


@dataclass(frozen=True, slots=True)
class EngramOffloadPlan:
    """Per-layer disk layouts for one rank's Engram tables."""

    layouts: tuple[EngramTableLayout, ...]

    @property
    def resident_bytes_avoided(self) -> int:
        return sum(layout.resident_bytes for layout in self.layouts)

    def describe(self) -> str:
        rows = sum(layout.num_rows for layout in self.layouts)
        gib = self.resident_bytes_avoided / 1024**3
        return (
            f"{len(self.layouts)} engram layers, {rows} rows per rank kept on "
            f"disk ({gib:.1f} GiB not allocated)"
        )


def build_engram_offload_plan(
    config: object,
    *,
    rank: int,
    world_size: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> EngramOffloadPlan:
    """Compute rank-local row ranges for every Engram table.

    Fails closed on a checkpoint whose Engram geometry is missing, because the
    reader cannot work out which rows belong to this rank without it.
    """

    source = _engram_source(config)
    layer_ids = tuple(_nested_field(source, "engram_layer_ids") or ())
    num_embeddings = tuple(_nested_field(source, "engram_num_embeddings") or ())
    head_dim = _nested_field(source, "engram_head_dim")

    if not layer_ids:
        return EngramOffloadPlan(layouts=())
    if len(layer_ids) != len(num_embeddings):
        raise ValueError(
            "DeepSeek V4.1 Flash engram_layer_ids and engram_num_embeddings "
            "must have the same length"
        )
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError(
            "DeepSeek V4.1 Flash requires a positive engram_head_dim in its config"
        )

    layouts = tuple(
        EngramTableLayout.for_rank(
            layer_id=int(layer_id),
            rows=int(rows),
            dim=head_dim,
            rank=rank,
            world_size=world_size,
            block_size=block_size,
        )
        for layer_id, rows in zip(layer_ids, num_embeddings, strict=True)
    )
    return EngramOffloadPlan(layouts=layouts)


def build_model_load_kwargs(
    *,
    dtype: object,
    quantization_config: object | None,
    distributed: bool,
    model_commit: str | None,
    device_map: object,
    max_memory: Mapping[str, object] | None,
    trust_remote_code: bool,
    config: object | None = None,
) -> dict[str, Any]:
    """Build mutually exclusive local or fixed DGX TP loader arguments."""

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True if trust_remote_code else None,
    }
    if model_commit is not None:
        kwargs["revision"] = model_commit
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if config is not None:
        kwargs["config"] = config

    if distributed:
        kwargs["tp_plan"] = "auto"
    else:
        kwargs["device_map"] = device_map
        kwargs["max_memory"] = (
            {
                int(key) if key.isdigit() else key: value
                for key, value in max_memory.items()
            }
            if max_memory
            else None
        )
    return kwargs
