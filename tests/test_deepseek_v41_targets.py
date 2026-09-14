# SPDX-License-Identifier: AGPL-3.0-or-later

"""Target discovery and cache tests for the DeepSeek V4.1 backend.

These build synthetic checkpoints in the *real* layout -- FP8 E4M3 weights with
ue8m0 32x32 block scales, one ``layers.N.attn.wo_b`` per layer, plus decoy
``mtp.N.attn.wo_b`` and unrelated tensors -- so they run without the 510 GB
checkpoint.
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from heretic.deepseek_v41_targets import (
    DEFAULT_BLOCK_SIZE,
    EXPECTED_V41_LAYER_COUNT,
    V41_COMPONENT,
    TargetCache,
    V41TargetPlan,
    dequantize_blocks,
    discover_targets,
)

BLOCK = DEFAULT_BLOCK_SIZE
D_OUT = 64
D_IN = 128


def _write_shard(
    path: Path,
    tensors: list[tuple[str, str, tuple[int, ...], bytes]],
) -> None:
    """Write a minimal but format-valid safetensors file."""

    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload += data

    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(bytes(payload))


def _fp8_bytes(generator: torch.Generator, shape: tuple[int, ...]) -> bytes:
    """Deterministic finite FP8 E4M3 payload."""

    values = (torch.rand(shape, generator=generator) * 2 - 1).to(torch.float8_e4m3fn)
    return values.view(torch.uint8).numpy().tobytes()


def _scale_bytes(generator: torch.Generator, shape: tuple[int, ...]) -> bytes:
    """Deterministic ue8m0 scales, kept near 1.0 to stay finite.

    ue8m0 encodes a power of two directly, so byte 127 is 1.0. The neighbourhood
    used here spans roughly 2^-7..2^7.
    """

    exponents = torch.randint(120, 135, shape, generator=generator, dtype=torch.int32)
    values = exponents.to(torch.uint8).view(torch.float8_e8m0fnu)
    return values.view(torch.uint8).numpy().tobytes()


def _build_checkpoint(
    root: Path,
    *,
    layers: list[int] | None = None,
    d_out: int = D_OUT,
    d_in: int = D_IN,
    block: int = BLOCK,
    include_scales: bool = True,
    duplicate_layer: bool = False,
    include_decoys: bool = True,
    shard_count: int = 3,
) -> None:
    """Write a synthetic V4.1-shaped checkpoint plus its index."""

    layers = list(range(EXPECTED_V41_LAYER_COUNT)) if layers is None else layers
    generator = torch.Generator().manual_seed(20260913)

    entries: list[tuple[str, str, str, tuple[int, ...], bytes]] = []

    def add(name: str, dtype: str, shape: tuple[int, ...], data: bytes) -> None:
        entries.append((name, dtype, shape, data))

    for layer in layers:
        add(
            f"layers.{layer}.attn.wo_b.weight",
            "F8_E4M3",
            (d_out, d_in),
            _fp8_bytes(generator, (d_out, d_in)),
        )
        if include_scales:
            add(
                f"layers.{layer}.attn.wo_b.scale",
                "F8_E8M0",
                (-(-d_out // block), -(-d_in // block)),
                _scale_bytes(generator, (-(-d_out // block), -(-d_in // block))),
            )
        # A sibling target that must never be selected.
        add(
            f"layers.{layer}.attn.wo_a.weight",
            "F8_E4M3",
            (d_in, d_out),
            _fp8_bytes(generator, (d_in, d_out)),
        )
        add(f"layers.{layer}.attn_norm.weight", "BF16", (d_out,), b"\x00" * (2 * d_out))

    if duplicate_layer:
        add(
            "layers.7.attn.wo_b.dup.weight",
            "F8_E4M3",
            (d_out, d_in),
            _fp8_bytes(generator, (d_out, d_in)),
        )

    if include_decoys:
        # MTP heads carry wo_b too, and are explicitly out of scope for V1.
        for mtp in range(3):
            add(
                f"mtp.{mtp}.attn.wo_b.weight",
                "F8_E4M3",
                (d_out, d_in),
                _fp8_bytes(generator, (d_out, d_in)),
            )
            add(
                f"mtp.{mtp}.attn.wo_b.scale",
                "F8_E8M0",
                (-(-d_out // block), -(-d_in // block)),
                _scale_bytes(generator, (-(-d_out // block), -(-d_in // block))),
            )
        for layer in layers:
            add(
                f"layers.{layer}.engram.embed.weight",
                "F8_E4M3",
                (8, d_out),
                _fp8_bytes(generator, (8, d_out)),
            )

    if duplicate_layer:
        # A genuine duplicate: same layer index, second distinct weight name.
        add(
            "layers.7.attn.wo_b.weight.bak",
            "F8_E4M3",
            (d_out, d_in),
            _fp8_bytes(generator, (d_out, d_in)),
        )

    weight_map: dict[str, str] = {}
    buckets: list[list[tuple[str, str, tuple[int, ...], bytes]]] = [
        [] for _ in range(shard_count)
    ]
    for index, entry in enumerate(entries):
        buckets[index % shard_count].append(entry)

    for shard_index, bucket in enumerate(buckets):
        if not bucket:
            continue
        shard_name = f"model-{shard_index + 1:05d}-of-{shard_count:05d}.safetensors"
        _write_shard(
            root / shard_name,
            [(name, dtype, shape, data) for name, dtype, shape, data in bucket],
        )
        for name, _, _, _ in bucket:
            weight_map[name] = shard_name

    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [block, block],
                    "scale_fmt": "ue8m0",
                },
            }
        ),
        encoding="utf-8",
    )


class TestDiscovery(unittest.TestCase):
    def test_discovers_exactly_forty_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)

            self.assertEqual(plan.layer_count, EXPECTED_V41_LAYER_COUNT)
            self.assertEqual(plan.layer_indices, tuple(range(40)))
            self.assertEqual(plan.components, (V41_COMPONENT,))
            self.assertEqual(plan.output_dimension, D_OUT)

    def test_one_target_per_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            for layer_index in range(40):
                target = plan.by_layer(layer_index)
                self.assertEqual(target.layer_index, layer_index)
                self.assertEqual(target.component, V41_COMPONENT)
                self.assertTrue(target.weight_name.endswith("attn.wo_b.weight"))

    def test_excludes_mtp_heads(self) -> None:
        """MTP carries wo_b too; V1 must never touch it."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root, include_decoys=True)
            plan = discover_targets(root)
            for target in plan.targets:
                self.assertFalse(target.weight_name.startswith("mtp."))
                self.assertTrue(target.weight_name.startswith("layers."))

    def test_scales_are_located_alongside_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            for target in plan.targets:
                self.assertIsNotNone(target.scale)
                self.assertIsNotNone(target.scale_shard)
                self.assertEqual(
                    target.scale.shape,
                    (-(-D_OUT // BLOCK), -(-D_IN // BLOCK)),
                )

    def test_block_size_comes_from_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            self.assertEqual({t.block_rows for t in plan.targets}, {BLOCK})
            self.assertEqual({t.block_cols for t in plan.targets}, {BLOCK})

    def test_shard_numbers_are_not_hardcoded(self) -> None:
        """Discovery must follow the index, not a guessed shard layout."""

        for shard_count in (1, 3, 7):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _build_checkpoint(root, shard_count=shard_count)
                plan = discover_targets(root)
                self.assertEqual(plan.layer_count, EXPECTED_V41_LAYER_COUNT)
                shards = {t.weight_shard for t in plan.targets}
                self.assertTrue(
                    shards.issubset({p.name for p in root.glob("*.safetensors")})
                )


class TestDiscoveryFailsClosed(unittest.TestCase):
    def test_missing_layer_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root, layers=list(range(39)))
            with self.assertRaises(ValueError) as caught:
                discover_targets(root)
            self.assertIn("exactly 40", str(caught.exception))

    def test_missing_scale_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root, include_scales=False)
            with self.assertRaises(ValueError) as caught:
                discover_targets(root)
            self.assertIn("no quantization scale", str(caught.exception))

    def test_missing_index_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                discover_targets(Path(temporary))

    def test_plan_rejects_duplicate_layer_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            duplicated = plan.targets[:-1] + (plan.targets[0],)
            with self.assertRaises(ValueError) as caught:
                V41TargetPlan(checkpoint_directory=root, targets=duplicated)
            self.assertIn("duplicate", str(caught.exception))

    def test_plan_rejects_wrong_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            with self.assertRaises(ValueError):
                V41TargetPlan(checkpoint_directory=root, targets=plan.targets[:-1])

    def test_empty_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"foo": "bar.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as caught:
                discover_targets(root)
            self.assertIn("no attn.wo_b tensors", str(caught.exception))


class TestDequantization(unittest.TestCase):
    def test_block_scales_expand_over_blocks(self) -> None:
        weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float8_e8m0fnu)
        expanded = dequantize_blocks(weight, scale, block_rows=2, block_cols=2)
        expected = torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [4.0, 4.0, 8.0, 8.0],
                [4.0, 4.0, 8.0, 8.0],
            ]
        )
        self.assertTrue(torch.equal(expanded, expected))

    def test_shape_mismatch_fails(self) -> None:
        weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
        scale = torch.ones(3, 3, dtype=torch.float8_e8m0fnu)
        with self.assertRaises(ValueError):
            dequantize_blocks(weight, scale, block_rows=2, block_cols=2)

    def test_row_norms_are_computable_from_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            with TargetCache(plan) as cache:
                matrix = cache.read_layer(0)
            self.assertEqual(matrix.shape, (D_OUT, D_IN))
            self.assertEqual(matrix.dtype, torch.float32)
            self.assertTrue(torch.isfinite(matrix).all())
            norms = torch.linalg.vector_norm(matrix, dim=1)
            self.assertTrue(torch.isfinite(norms).all())
            self.assertTrue((norms > 0).all())

    def test_cache_reads_match_the_shard_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            target = plan.by_layer(3)
            with TargetCache(plan) as cache:
                weight, scale = cache.read_quantized(target)
                dequantized = cache.read_dequantized(target)

            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale.dtype, torch.float8_e8m0fnu)
            self.assertEqual(tuple(weight.shape), (D_OUT, D_IN))
            expected = dequantize_blocks(
                weight, scale, block_rows=BLOCK, block_cols=BLOCK
            )
            self.assertTrue(torch.equal(dequantized, expected))

    def test_description_mentions_the_physical_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            description = plan.describe()
            self.assertIn("40", description)
            self.assertIn("attn.wo_b", description)


class TestSemanticMapping(unittest.TestCase):
    """`attn.o_proj` must be the semantic name; `wo_b` the physical one."""

    def test_component_name_is_heretics(self) -> None:
        self.assertEqual(V41_COMPONENT, "attn.o_proj")

    def test_every_target_carries_the_semantic_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_checkpoint(root)
            plan = discover_targets(root)
            self.assertEqual({t.component for t in plan.targets}, {"attn.o_proj"})
            self.assertEqual(plan.components, ("attn.o_proj",))
            for target in plan.targets:
                self.assertIn(".attn.wo_b.weight", target.weight_name)


if __name__ == "__main__":
    unittest.main()
