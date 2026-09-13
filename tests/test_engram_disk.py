# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline tests for disk-backed Engram tables.

These build synthetic safetensors shards laid out like the real DeepSeek V4.1
Flash checkpoint (one full-table tensor per engram layer, ranks owning
contiguous head ranges) and exercise the reader against a reference
implementation of the dequantization.

The row-offset test is the important one: the upstream patch originally based
every read at the start of the full tensor, so ranks above 0 silently read
rank 0's rows. Wrong rows are still real embedding rows, so the output looks
like plausible text and a smoke test passes. Only a rank-aware comparison
catches it.
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from heretic.engram_disk import (
    DEFAULT_BLOCK_SIZE,
    DiskEngramTable,
    EngramDiskUnavailable,
    EngramTableLayout,
    engram_table_name,
    gather_dequant_many,
)

_DIM = 256
_SCALE_COLUMNS = _DIM // DEFAULT_BLOCK_SIZE
_ROWS_PER_HEAD = 1000
_HEADS = 8
_WORLD_SIZE = 4

# ue8m0 exponent bytes 120..134 are finite fp32 scales (~2^-7 .. 2^7). Values
# outside a much wider band would dequantize to inf/nan and make equality
# assertions meaningless.
_SCALE_MIN = 120
_SCALE_MAX = 134


def _write_checkpoint(
    directory: Path, layer_id: int, rows: int, *, shard: str | None = None
) -> str:
    """Write one engram layer as a shard, merged into any existing index."""

    generator = torch.Generator().manual_seed(layer_id)
    weight = torch.randint(0, 256, (rows, _DIM), dtype=torch.uint8, generator=generator)
    # ue8m0 bytes are fp32 exponent fields. Center them on 127 (scale 1.0) and
    # stay well inside the finite fp32 exponent range: bytes far from 127 give
    # inf, and 0 * inf is nan, which makes comparisons spuriously unequal.
    scale = torch.randint(
        _SCALE_MIN,
        _SCALE_MAX,
        (rows, _SCALE_COLUMNS),
        dtype=torch.uint8,
        generator=generator,
    )

    header = {
        engram_table_name(layer_id, "weight"): {
            "dtype": "F8_E4M3",
            "shape": [rows, _DIM],
            "data_offsets": [0, rows * _DIM],
        },
        engram_table_name(layer_id, "scale"): {
            "dtype": "F8_E8M0",
            "shape": [rows, _SCALE_COLUMNS],
            "data_offsets": [rows * _DIM, rows * _DIM + rows * _SCALE_COLUMNS],
        },
        "__metadata__": {"format": "pt"},
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)

    shard = shard or f"model-{layer_id:05d}-of-00048.safetensors"
    with (directory / shard).open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(weight.numpy().tobytes())
        handle.write(scale.numpy().tobytes())

    index_path = directory / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map.update(existing.get("weight_map", {}))
    weight_map.update({name: shard for name in header if name != "__metadata__"})
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return shard


def _read_layer_tensors(
    directory: Path, shard: str, layer_id: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a shard's engram weight and scale back as uint8 tensors."""

    raw = bytearray((directory / shard).read_bytes())
    header_length = struct.unpack("<Q", bytes(raw[:8]))[0]
    header = json.loads(bytes(raw[8 : 8 + header_length]))
    payload = torch.frombuffer(raw, dtype=torch.uint8)

    def slice_for(suffix: str, columns: int) -> torch.Tensor:
        start = (
            8
            + header_length
            + header[engram_table_name(layer_id, suffix)]["data_offsets"][0]
        )
        return payload[start : start + rows * columns].reshape(rows, columns)

    return slice_for("weight", _DIM), slice_for("scale", _SCALE_COLUMNS)


def _same(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Bit-exact comparison that treats nan as equal to nan.

    With block scales in the finite range the dequantized rows never contain
    nan, but treating nan as matching keeps a stray one from turning into a
    confusing inequality.
    """

    return actual.shape == expected.shape and bool(
        torch.equal(actual, expected)
        or (
            torch.isnan(actual.float()).equal(torch.isnan(expected.float()))
            and torch.equal(actual.float().nan_to_num(), expected.float().nan_to_num())
        )
    )


def _reference_rows(
    weight: torch.Tensor, scale: torch.Tensor, row_ids: torch.Tensor
) -> torch.Tensor:
    selected = weight[row_ids].view(torch.float8_e4m3fn).to(torch.float32)
    selected = selected.view(-1, _SCALE_COLUMNS, DEFAULT_BLOCK_SIZE)
    scales = (scale[row_ids].to(torch.int32) << 23).view(torch.float32)
    return (selected * scales[:, :, None]).reshape(-1, _DIM).to(torch.bfloat16)


class TestEngramTableLayout(unittest.TestCase):
    def test_splits_rows_across_ranks_without_gaps(self) -> None:
        rows = _ROWS_PER_HEAD * _HEADS
        layouts = [
            EngramTableLayout.for_rank(
                layer_id=1,
                rows=rows,
                dim=_DIM,
                rank=rank,
                world_size=_WORLD_SIZE,
            )
            for rank in range(_WORLD_SIZE)
        ]
        self.assertEqual(
            [layout.row_start for layout in layouts], [0, 2000, 4000, 6000]
        )
        self.assertEqual([layout.num_rows for layout in layouts], [2000] * 4)
        self.assertEqual(sum(layout.num_rows for layout in layouts), rows)

    def test_rejects_a_rank_outside_the_world(self) -> None:
        with self.assertRaises(ValueError):
            EngramTableLayout.for_rank(
                layer_id=1, rows=8000, dim=_DIM, rank=4, world_size=4
            )


class TestDiskEngramTable(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.layer_id = 1
        self.rows = _ROWS_PER_HEAD * _HEADS
        self.shard = _write_checkpoint(self.directory, self.layer_id, self.rows)
        self.weight, self.scale = _read_layer_tensors(
            self.directory, self.shard, self.layer_id, self.rows
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _table(self, rank: int) -> DiskEngramTable:
        layout = EngramTableLayout.for_rank(
            layer_id=self.layer_id,
            rows=self.rows,
            dim=_DIM,
            rank=rank,
            world_size=_WORLD_SIZE,
        )
        return DiskEngramTable(self.directory, layout, threads=4, chunk=8)

    def test_every_rank_reads_its_own_rows(self) -> None:
        """The ported offset fix: ranks above 0 must not read rank 0's rows."""

        weight, scale = self.weight, self.scale

        for rank in range(_WORLD_SIZE):
            table = self._table(rank)
            try:
                # Global row ids, as the hash kernel produces them.
                global_ids = torch.tensor(
                    [rank * 2000 + 7, rank * 2000 + 1999], dtype=torch.int64
                )
                rel = global_ids - table.layout.row_start
                owned = torch.ones(rel.numel(), dtype=torch.bool)
                (actual,) = gather_dequant_many(
                    [(table, rel, owned)], threads=2, chunk=4
                )
                expected = _reference_rows(weight, scale, global_ids)
                self.assertTrue(
                    _same(actual, expected),
                    f"rank {rank} read the wrong engram rows",
                )
            finally:
                table.close()

    def test_naive_offsets_really_would_read_the_wrong_rows(self) -> None:
        """Negative control for the test above.

        The point of the offset fix is that a rank-local row id, read from the
        start of the *full* tensor, returns rank 0's data. This asserts that
        premise directly, so the passing test above cannot be vacuous: if the
        shards happened to make rows interchangeable, this would fail.
        """

        rank = 3
        local_row = 7
        global_row = rank * 2000 + local_row

        rank0_rows = _reference_rows(
            self.weight, self.scale, torch.tensor([local_row], dtype=torch.int64)
        )
        real_rows = _reference_rows(
            self.weight, self.scale, torch.tensor([global_row], dtype=torch.int64)
        )
        self.assertFalse(
            _same(rank0_rows, real_rows),
            "shard generation made the offset bug undetectable; "
            "the row-offset test needs distinct rows",
        )

    def test_unowned_rows_are_zeroed(self) -> None:
        table = self._table(1)
        try:
            rel = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
            owned = torch.tensor([True, False, True, False])
            (actual,) = gather_dequant_many([(table, rel, owned)])
            zeros = torch.zeros(_DIM, dtype=torch.bfloat16)
            self.assertTrue(_same(actual[1], zeros))
            self.assertTrue(_same(actual[3], zeros))
            self.assertTrue(torch.count_nonzero(actual[0]) > 0)
        finally:
            table.close()

    def test_repeated_rows_are_read_once_but_returned_in_order(self) -> None:
        table = self._table(0)
        try:
            rel = torch.tensor([5, 5, 9, 5, 9], dtype=torch.int64)
            owned = torch.ones(5, dtype=torch.bool)
            (actual,) = gather_dequant_many([(table, rel, owned)])
            self.assertTrue(_same(actual[0], actual[1]))
            self.assertTrue(_same(actual[0], actual[3]))
            self.assertTrue(_same(actual[2], actual[4]))
        finally:
            table.close()

    def test_two_layers_share_one_batch(self) -> None:
        _write_checkpoint(
            self.directory, 14, self.rows, shard="model-00048-of-00048.safetensors"
        )

        def table_for(layer_id: int) -> DiskEngramTable:
            layout = EngramTableLayout.for_rank(
                layer_id=layer_id,
                rows=self.rows,
                dim=_DIM,
                rank=0,
                world_size=_WORLD_SIZE,
            )
            return DiskEngramTable(self.directory, layout, threads=2, chunk=4)

        first, second = table_for(1), table_for(14)
        try:
            rel = torch.tensor([1, 2, 3], dtype=torch.int64)
            owned = torch.ones(3, dtype=torch.bool)
            outputs = gather_dequant_many([(first, rel, owned), (second, rel, owned)])
            self.assertEqual(len(outputs), 2)
            self.assertEqual(tuple(outputs[0].shape), (3, _DIM))
            self.assertEqual(tuple(outputs[1].shape), (3, _DIM))
        finally:
            first.close()
            second.close()


class TestEngramDiskFailures(unittest.TestCase):
    def test_missing_index_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = EngramTableLayout.for_rank(
                layer_id=1, rows=8000, dim=_DIM, rank=0, world_size=2
            )
            with self.assertRaisesRegex(EngramDiskUnavailable, "index.json"):
                DiskEngramTable(Path(temporary), layout)

    def test_shape_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_checkpoint(directory, 1, _ROWS_PER_HEAD * _HEADS)
            layout = EngramTableLayout.for_rank(
                layer_id=1,
                rows=_ROWS_PER_HEAD * _HEADS * 2,
                dim=_DIM,
                rank=0,
                world_size=2,
            )
            with self.assertRaisesRegex(EngramDiskUnavailable, "shape"):
                DiskEngramTable(directory, layout)


if __name__ == "__main__":
    unittest.main()
