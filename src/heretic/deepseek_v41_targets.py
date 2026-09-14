# SPDX-License-Identifier: AGPL-3.0-or-later

"""DeepSeek V4.1 Flash abliteration targets: discovery and a bounded weight cache.

V1 supports exactly one abliteration component, Heretic's ``attn.o_proj``, which
maps to DeepSeek V4.1's physical ``attn.wo_b`` projection. Everything else --
routed experts, shared experts, Engram, the vision encoder, the aligner,
embeddings, MTP and DSpark -- is explicitly out of scope and is never read or
modified here.

Discovery is metadata-only: it reads ``model.safetensors.index.json`` and the
safetensors headers, never the tensor payloads. The cache then reads **only** the
40 target matrices plus their scale tensors, so a trial never touches the other
~510 GB of the checkpoint.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "EXPECTED_V41_LAYER_COUNT",
    "V41_COMPONENT",
    "V41_PHYSICAL_SUFFIX",
    "TargetCache",
    "TargetTensor",
    "V41TargetPlan",
    "discover_targets",
]

#: Heretic's semantic component name. Kept stable across backends so that
#: ``abliteration_components = ["attn.o_proj"]`` means the same thing whether the
#: model is served by Transformers or by vLLM.
V41_COMPONENT = "attn.o_proj"

#: The physical projection it maps to on DeepSeek V4.1 Flash.
V41_PHYSICAL_SUFFIX = "attn.wo_b"

EXPECTED_V41_LAYER_COUNT = 40

#: From the released ``config.json``: ``quantization_config.weight_block_size``.
DEFAULT_BLOCK_SIZE = 32

# Only `layers.N.attn.wo_b` -- `mtp.N.attn.wo_b` must never match.
_WEIGHT_SUFFIX = ".attn.wo_b.weight"
_SCALE_SUFFIX = ".attn.wo_b.scale"

_MAX_HEADER_BYTES = 256 * 1024 * 1024

#: safetensors dtype tags this module knows how to reinterpret. V4.1 Flash needs
#: only the FP8 pair, but the rest are here so a mixed checkpoint fails with a
#: clear message rather than a reshape error.
_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True, slots=True)
class SafetensorsEntry:
    """One tensor's location and geometry inside a shard."""

    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def _read_safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {path.name}")
        length = struct.unpack("<Q", raw_length)[0]
        if length <= 0 or length > _MAX_HEADER_BYTES:
            raise ValueError(
                f"invalid safetensors header length in {path.name}: {length}"
            )
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError(f"truncated safetensors header: {path.name}")
    header = json.loads(encoded)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path.name}")
    return header


def _entry(header: dict[str, object], name: str, shard: str) -> SafetensorsEntry:
    raw = header.get(name)
    if not isinstance(raw, dict):
        raise ValueError(f"tensor {name} not found in shard {shard}")
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if not isinstance(dtype, str):
        raise ValueError(f"tensor {name} has no dtype in shard {shard}")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
    ):
        raise ValueError(f"tensor {name} has an invalid shape in shard {shard}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(value) is not int for value in offsets)
    ):
        raise ValueError(f"tensor {name} has invalid data_offsets in shard {shard}")
    return SafetensorsEntry(
        dtype=dtype,
        shape=tuple(int(dimension) for dimension in shape),
        begin=int(offsets[0]),
        end=int(offsets[1]),
    )


@dataclass(frozen=True, slots=True)
class TargetTensor:
    """One layer's abliteration target and everything needed to read it."""

    layer_index: int
    component: str
    weight_name: str
    scale_name: str | None
    weight_shard: str
    scale_shard: str | None
    weight: SafetensorsEntry
    scale: SafetensorsEntry | None
    block_rows: int
    block_cols: int

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def is_quantized(self) -> bool:
        return self.scale is not None


@dataclass(frozen=True, slots=True)
class V41TargetPlan:
    """The validated set of V4.1 abliteration targets."""

    checkpoint_directory: Path
    targets: tuple[TargetTensor, ...]

    def __post_init__(self) -> None:
        if len(self.targets) != EXPECTED_V41_LAYER_COUNT:
            raise ValueError(
                "DeepSeek V4.1 Flash must expose exactly "
                f"{EXPECTED_V41_LAYER_COUNT} {V41_COMPONENT} targets; "
                f"found {len(self.targets)}"
            )
        seen: set[int] = set()
        for target in self.targets:
            if target.layer_index in seen:
                raise ValueError(
                    f"duplicate {V41_COMPONENT} target for layer {target.layer_index}"
                )
            seen.add(target.layer_index)
        expected = set(range(EXPECTED_V41_LAYER_COUNT))
        if seen != expected:
            missing = sorted(expected - seen)
            raise ValueError(f"missing {V41_COMPONENT} targets for layers {missing}")
        dimensions = {target.out_features for target in self.targets}
        if len(dimensions) != 1:
            raise ValueError(
                f"inconsistent output dimensions across targets: {sorted(dimensions)}"
            )

    @property
    def layer_count(self) -> int:
        return len(self.targets)

    @property
    def output_dimension(self) -> int:
        return self.targets[0].out_features

    @property
    def components(self) -> tuple[str, ...]:
        return (V41_COMPONENT,)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(target.layer_index for target in self.targets)

    def by_layer(self, layer_index: int) -> TargetTensor:
        for target in self.targets:
            if target.layer_index == layer_index:
                return target
        raise KeyError(f"no {V41_COMPONENT} target for layer {layer_index}")

    def describe(self) -> str:
        dims = {target.in_features for target in self.targets}
        return (
            f"{self.layer_count} {V41_COMPONENT} targets "
            f"({V41_PHYSICAL_SUFFIX}) -> [{self.output_dimension}, "
            f"{sorted(dims)}]"
        )


def _discover_block_size(checkpoint_directory: Path) -> int:
    """Read the quantization block size the checkpoint itself declares."""

    config_path = checkpoint_directory / "config.json"
    if not config_path.is_file():
        return DEFAULT_BLOCK_SIZE
    with config_path.open("rb") as stream:
        config = json.loads(stream.read())
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return DEFAULT_BLOCK_SIZE
    block = quantization.get("weight_block_size")
    if (
        not isinstance(block, list)
        or len(block) != 2
        or any(type(value) is not int or value <= 0 for value in block)
    ):
        return DEFAULT_BLOCK_SIZE
    if block[0] != block[1]:
        raise ValueError(f"unsupported non-square weight block size: {block}")
    return int(block[0])


def discover_targets(
    checkpoint_directory: str | Path,
    *,
    expected_layer_count: int = EXPECTED_V41_LAYER_COUNT,
) -> V41TargetPlan:
    """Locate every V4.1 ``attn.wo_b`` target without reading tensor payloads.

    Fails loudly if the checkpoint does not expose exactly one target per
    transformer layer, because a partial set would silently ablate only part of
    the model.
    """

    root = Path(checkpoint_directory)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"checkpoint index does not exist: {index_path} "
            "(a sharded checkpoint is required)"
        )

    with index_path.open("rb") as stream:
        index = json.loads(stream.read())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index weight_map must be a nonempty object")

    block_size = _discover_block_size(root)

    # Only `layers.<int>.attn.wo_b.weight`; `mtp.N.attn.wo_b.weight` is excluded
    # both by the `layers.` prefix and by the strict integer parse below.
    selected: list[tuple[int, str]] = []
    for name in weight_map:
        if not name.endswith(_WEIGHT_SUFFIX):
            continue
        prefix = name[: -len(_WEIGHT_SUFFIX)]
        _, separator, layer_text = prefix.rpartition("layers.")
        if (
            not separator
            or not layer_text.isdigit()
            or prefix != f"layers.{layer_text}"
        ):
            continue
        selected.append((int(layer_text), name))

    if not selected:
        raise ValueError(
            f"checkpoint defines no {V41_PHYSICAL_SUFFIX} tensors; "
            "is this really DeepSeek V4.1 Flash?"
        )

    selected.sort()
    headers: dict[str, dict[str, object]] = {}
    targets: list[TargetTensor] = []

    for layer_index, weight_name in selected:
        shard = weight_map.get(weight_name)
        if not isinstance(shard, str) or not shard:
            raise ValueError(f"tensor {weight_name} has no shard in the index")
        if shard not in headers:
            shard_path = root / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"checkpoint shard does not exist: {shard}")
            headers[shard] = _read_safetensors_header(shard_path)

        weight = _entry(headers[shard], weight_name, shard)
        if len(weight.shape) != 2:
            raise ValueError(
                f"target {weight_name} must be a matrix, got shape {weight.shape}"
            )

        scale_name = weight_name[: -len(".weight")] + ".scale"
        scale: SafetensorsEntry | None = None
        scale_shard: str | None = None
        scale_in_index = weight_map.get(scale_name)
        if isinstance(scale_in_index, str) and scale_in_index:
            if scale_in_index not in headers:
                shard_path = root / scale_in_index
                if not shard_path.is_file():
                    raise FileNotFoundError(
                        f"checkpoint shard does not exist: {scale_in_index}"
                    )
                headers[scale_in_index] = _read_safetensors_header(shard_path)
            scale = _entry(headers[scale_in_index], scale_name, scale_in_index)
            scale_shard = scale_in_index

            expected_scale_shape = (
                -(-weight.shape[0] // block_size),
                -(-weight.shape[1] // block_size),
            )
            if scale.shape != expected_scale_shape:
                raise ValueError(
                    f"scale tensor {scale_name} has shape {scale.shape}, expected "
                    f"{expected_scale_shape} for weight {weight.shape} with "
                    f"block size {block_size}"
                )
            if scale.nbytes != scale.shape[0] * scale.shape[1]:
                raise ValueError(f"scale tensor {scale_name} is not one byte per block")
        else:
            raise ValueError(
                f"target {weight_name} has no quantization scale {scale_name}; "
                "V1 requires the released FP8 layout with ue8m0 block scales"
            )

        targets.append(
            TargetTensor(
                layer_index=layer_index,
                component=V41_COMPONENT,
                weight_name=weight_name,
                scale_name=scale_name,
                weight_shard=shard,
                scale_shard=scale_shard,
                weight=weight,
                scale=scale,
                block_rows=block_size,
                block_cols=block_size,
            )
        )

    plan = V41TargetPlan(checkpoint_directory=root, targets=tuple(targets))
    if plan.layer_count != expected_layer_count:
        raise ValueError(
            f"expected exactly {expected_layer_count} {V41_COMPONENT} targets, "
            f"found {plan.layer_count}"
        )
    return plan


class TargetCache:
    """Reads and dequantizes only the target matrices, once per run.

    The released checkpoint stores these as FP8 E4M3 with ue8m0 block scales.
    ``read_quantized`` returns the raw 1-byte-per-element payload (about 1.7 GiB
    for all 40 layers), which is enough to recompute factors for every trial
    without touching the checkpoint again. ``read_dequantized`` widens one matrix
    to float32 (about 160 MiB) for the abliteration math.
    """

    def __init__(self, plan: V41TargetPlan) -> None:
        self.plan = plan
        self._data_starts: dict[str, int] = {}

    def close(self) -> None:
        self._data_starts.clear()

    def __enter__(self) -> TargetCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _data_start(self, shard: str) -> int:
        """Byte offset of the data section within the shard file.

        safetensors ``data_offsets`` are relative to the start of the data
        buffer, not the file, so every read must be shifted past the 8-byte
        length prefix and the padded JSON header. Getting this wrong reads
        header bytes as tensor data, which decodes to plausible-looking but
        wrong values rather than raising.
        """

        cached = self._data_starts.get(shard)
        if cached is not None:
            return cached
        path = self.plan.checkpoint_directory / shard
        with path.open("rb") as stream:
            raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {shard}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length in {shard}")
        start = 8 + header_length
        self._data_starts[shard] = start
        return start

    def _read_range(self, shard: str, begin: int, end: int) -> bytes:
        if end <= begin:
            raise ValueError(f"invalid tensor byte range in shard {shard}")
        path = self.plan.checkpoint_directory / shard
        with path.open("rb") as stream:
            stream.seek(self._data_start(shard) + begin)
            payload = stream.read(end - begin)
        if len(payload) != end - begin:
            raise ValueError(f"truncated read from shard {shard}")
        return payload

    @staticmethod
    def _to_tensor(payload: bytes, entry: SafetensorsEntry) -> Tensor:
        dtype = _SAFETENSORS_DTYPES.get(entry.dtype)
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype: {entry.dtype}")
        # Read as bytes, then reinterpret, so nothing depends on safetensors'
        # private dtype table.
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        return raw.view(dtype).reshape(entry.shape)

    def read_quantized(self, target: TargetTensor) -> tuple[Tensor, Tensor]:
        """Return the raw FP8 weight and its ue8m0 block scales."""

        if target.scale is None or target.scale_shard is None:
            raise ValueError(f"target {target.weight_name} is not quantized")
        weight = self._to_tensor(
            self._read_range(
                target.weight_shard, target.weight.begin, target.weight.end
            ),
            target.weight,
        )
        scale = self._to_tensor(
            self._read_range(target.scale_shard, target.scale.begin, target.scale.end),
            target.scale,
        )
        return weight, scale

    def read_dequantized(self, target: TargetTensor) -> Tensor:
        """Return the float32 weight matrix, shaped ``(d_out, d_in)``."""

        weight, scale = self.read_quantized(target)
        return dequantize_blocks(
            weight,
            scale,
            block_rows=target.block_rows,
            block_cols=target.block_cols,
        )

    def read_layer(self, layer_index: int) -> Tensor:
        return self.read_dequantized(self.plan.by_layer(layer_index))


def dequantize_blocks(
    weight: Tensor,
    scale: Tensor,
    *,
    block_rows: int,
    block_cols: int,
) -> Tensor:
    """Expand per-block scales onto a block-quantized weight matrix.

    ue8m0 is a pure power-of-two exponent format, so widening it to float32
    yields the multiplier directly and no mantissa work is needed.
    """

    if weight.dim() != 2:
        raise ValueError(f"weight must be a matrix, got shape {tuple(weight.shape)}")
    expected = (-(-weight.shape[0] // block_rows), -(-weight.shape[1] // block_cols))
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match weight "
            f"{tuple(weight.shape)} with block ({block_rows}, {block_cols})"
        )
    expanded = (
        scale.to(torch.float32)
        .repeat_interleave(block_rows, dim=0)
        .repeat_interleave(block_cols, dim=1)
    )
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    if not torch.isfinite(expanded).all():
        raise ValueError("dequantization produced non-finite block scales")
    return weight.to(torch.float32) * expanded
