# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read-only, disk-backed Engram (n-gram) tables for DeepSeek V4.1 Flash.

DeepSeek V4.1 Flash carries two Engram n-gram hash tables (layers 1 and 14).
Together they hold 1.573e12 parameters, which is about 1.43 TiB stored in the
checkpoint's fp8 e4m3 form. A DGX Spark has 128 GB of unified memory, so the
tables cannot be resident on any tensor-parallel degree this project targets:
at TP4 the backbone alone consumes the four nodes' memory budget.

This module keeps those tables on the SSD and reads rows from the safetensors
shards on demand. It is a port of the Engram-on-disk patch used by the vLLM
DGX Spark deployment (Tech2Wild/Kai, 2026-09-10), reduced to what Heretic
needs and without the vLLM runtime dependency.

Why this works despite the size
-------------------------------

The naive reading of "1.43 TiB on SSD" is that every lookup becomes a random
disk read, which would be far too slow. Two properties of the real access
pattern make it tractable:

1. **The table is row-sharded by tensor parallelism.** The checkpoint stores
   each layer's full table, but a rank only ever looks up its own head range,
   so rank *r* of *N* reads only ``ceil(rows / N)`` rows -- about 47 GiB at
   TP4, not 1.43 TiB.

2. **Lookups are batched and heavily repeated.** Engram hashes every position
   into ``(max_ngram_size - 1) * n_heads = 24`` bucket ids, but an entire
   forward's ids are gathered in one go, and natural text repeats n-grams
   constantly. Rows are de-duplicated before reading, and the reads for every
   engram layer go out in a single parallel batch across a shared thread pool.

The rows are dequantized on the CPU (fp8 e4m3 against ue8m0 block scales,
matching the in-memory path bit for bit) and copied to the device, so the
tables never occupy device memory.

Correctness note (ported bug fix)
---------------------------------

The original patch based every read at the start of the *full* tensor while
computing rank-local row ids, so ranks above 0 read rank 0's rows. The symptom
was benign-looking (real embedding rows, just the wrong ones), which is why it
survived smoke tests. This port carries the fix: ``row_start``/``num_rows``
shift the weight and scale base offsets into this rank's own range. The
harness in ``tests/test_engram_disk.py`` pins that behavior.
"""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

__all__ = [
    "DiskEngramTable",
    "EngramDiskConfig",
    "EngramTableLayout",
    "engram_table_name",
    "from_rank_environment",
    "gather_dequant_many",
]

# One fp8 scale per 32x32 weight block, matching the checkpoint's quantization
# config (``weight_block_size: [32, 32]``).
DEFAULT_BLOCK_SIZE = 32

# One process-wide pool, so a step's reads for both engram layers go out as a
# single batch instead of 24 serial reads.
_POOL: ThreadPoolExecutor | None = None


class EngramDiskUnavailable(RuntimeError):
    """Raised when disk-backed Engram is requested but cannot be provided."""


@dataclass(frozen=True, slots=True)
class EngramDiskConfig:
    """Resolved disk-offload settings for one rank."""

    enabled: bool
    directory: str | None
    threads: int = 32
    chunk: int = 16

    def __post_init__(self) -> None:
        if self.enabled and not self.directory:
            raise ValueError("engram disk directory is required when enabled")
        if self.threads <= 0:
            raise ValueError("engram disk threads must be positive")
        if self.chunk <= 0:
            raise ValueError("engram disk chunk must be positive")


@dataclass(frozen=True, slots=True)
class EngramTableLayout:
    """How one engram layer's table is sharded across tensor-parallel ranks.

    ``rows`` is the full table's row count. Each rank owns the contiguous row
    range ``[row_start, row_start + num_rows)``, mirroring the in-memory
    loader's ``narrow(0, vocab_start, part_rows)``.
    """

    layer_id: int
    rows: int
    row_start: int
    num_rows: int
    dim: int
    block_size: int = DEFAULT_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("engram layer id must be nonnegative")
        if self.dim <= 0:
            raise ValueError("engram head dim must be positive")
        if self.block_size <= 0 or self.dim % self.block_size:
            raise ValueError("engram block size must divide the head dim")
        if self.num_rows <= 0:
            raise ValueError("engram rank row count must be positive")
        if self.row_start < 0 or self.row_start + self.num_rows > self.rows:
            raise ValueError("engram rank row range must lie inside the table")

    @classmethod
    def for_rank(
        cls,
        *,
        layer_id: int,
        rows: int,
        dim: int,
        rank: int,
        world_size: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> EngramTableLayout:
        """Split ``rows`` across ``world_size`` ranks the way the loader does.

        The in-memory path uses ``ceil(rows / world_size)`` per rank and lets
        the last rank come up short, so the ranges stay contiguous and start at
        ``rank * part_rows``.
        """

        if world_size < 1:
            raise ValueError("engram world size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("engram rank must be in [0, world_size)")
        part_rows = (rows + world_size - 1) // world_size
        row_start = rank * part_rows
        if row_start >= rows:
            raise EngramDiskUnavailable(
                f"engram layer {layer_id} has {rows} rows, too few for "
                f"{world_size} ranks"
            )
        return cls(
            layer_id=layer_id,
            rows=rows,
            row_start=row_start,
            num_rows=min(part_rows, rows - row_start),
            dim=dim,
            block_size=block_size,
        )

    @property
    def scale_columns(self) -> int:
        return self.dim // self.block_size

    @property
    def resident_bytes(self) -> int:
        """Bytes this rank would have held in memory, for reporting only."""
        per_row = self.dim + self.scale_columns
        return self.num_rows * per_row


def engram_table_name(layer_id: int, suffix: str) -> str:
    """Checkpoint tensor name for an engram table component."""

    return f"layers.{layer_id}.engram.embed.{suffix}"


def _thread_pool(threads: int) -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="engram-disk"
        )
    return _POOL


def _pread_exact(descriptor: int, buffer: memoryview, offset: int) -> None:
    """Fill ``buffer`` from ``offset``, retrying short reads.

    ``os.pread`` is POSIX-only. Every DGX Spark deployment target is Linux; on
    other platforms the caller is expected to have refused disk mode already.
    """

    read = getattr(os, "pread", None)
    if read is None:  # pragma: no cover - Windows and other non-POSIX hosts
        raise EngramDiskUnavailable(
            "disk-backed Engram requires os.pread (POSIX); use a Linux host"
        )
    got = 0
    total = len(buffer)
    while got < total:
        chunk = read(descriptor, total - got, offset + got)
        if isinstance(chunk, bytes):
            buffer[got : got + len(chunk)] = chunk
        else:
            break
        if not chunk:
            raise OSError("engram disk table: short read")
        got += len(chunk)
    if got != total:
        raise OSError("engram disk table: short read")


class DiskEngramTable:
    """Random-access reader for one engram layer's weight and scale tensors."""

    def __init__(
        self,
        model_directory: str | os.PathLike[str],
        layout: EngramTableLayout,
        *,
        threads: int = 32,
        chunk: int = 16,
    ) -> None:
        if threads <= 0 or chunk <= 0:
            raise ValueError("engram disk threads and chunk must be positive")

        self.layout = layout
        self.dim = layout.dim
        self.scale_columns = layout.scale_columns
        self.threads = threads
        self.chunk = chunk

        index_path = Path(model_directory) / "model.safetensors.index.json"
        if not index_path.is_file():
            raise EngramDiskUnavailable(
                f"engram disk mode needs {index_path.name} in {model_directory}"
            )
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map")
        if not isinstance(weight_map, dict):
            raise EngramDiskUnavailable(f"{index_path} does not contain a weight_map")

        weight_name = engram_table_name(layout.layer_id, "weight")
        scale_name = engram_table_name(layout.layer_id, "scale")
        opening = self._open(model_directory, weight_map, weight_name)
        self._weight_fd, self._weight_base, weight_shape = opening
        opening = self._open(model_directory, weight_map, scale_name)
        self._scale_fd, self._scale_base, scale_shape = opening

        if weight_shape != (layout.rows, self.dim):
            raise EngramDiskUnavailable(
                f"engram weight {weight_name} has shape {weight_shape}, "
                f"expected {(layout.rows, self.dim)}"
            )
        if scale_shape != (layout.rows, self.scale_columns):
            raise EngramDiskUnavailable(
                f"engram scale {scale_name} has shape {scale_shape}, "
                f"expected {(layout.rows, self.scale_columns)}"
            )

        # Shift the base offsets into this rank's own row range. Without this,
        # every rank above 0 reads rank 0's rows.
        self._weight_base += layout.row_start * self.dim
        self._scale_base += layout.row_start * self.scale_columns

    @staticmethod
    def _open(
        model_directory: str | os.PathLike[str],
        weight_map: dict[str, object],
        tensor_name: str,
    ) -> tuple[int, int, tuple[int, ...]]:
        filename = weight_map.get(tensor_name)
        if not isinstance(filename, str):
            raise EngramDiskUnavailable(
                f"engram tensor {tensor_name} is not in the checkpoint index"
            )
        path = Path(model_directory) / filename
        descriptor = os.open(path, os.O_RDONLY)
        try:
            advise = getattr(os, "posix_fadvise", None)
            if advise is not None:
                advise_range = getattr(os, "POSIX_FADV_RANDOM", None)
                if advise_range is not None:
                    try:
                        advise(descriptor, 0, 0, advise_range)
                    except OSError:
                        pass
            header_length = struct.unpack("<Q", os.pread(descriptor, 8, 0))[0]
            header = json.loads(os.pread(descriptor, header_length, 8))
        except BaseException:
            os.close(descriptor)
            raise
        metadata = header.get(tensor_name)
        if not isinstance(metadata, dict):
            raise EngramDiskUnavailable(
                f"engram tensor {tensor_name} is missing from {filename}"
            )
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise EngramDiskUnavailable(
                f"engram tensor {tensor_name} has no data_offsets"
            )
        if not isinstance(shape, list) or len(shape) != 2:
            raise EngramDiskUnavailable(f"engram tensor {tensor_name} has no shape")
        base = 8 + header_length + int(offsets[0])
        return descriptor, base, (int(shape[0]), int(shape[1]))

    def close(self) -> None:
        for attribute in ("_weight_fd", "_scale_fd"):
            descriptor = getattr(self, attribute, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, attribute, None)

    def read_jobs(
        self, local_rows: list[int], weight: Tensor, scale: Tensor
    ) -> list[tuple[int, int, list[int], int, memoryview]]:
        """Read jobs for this rank's local rows, for batched dispatch."""

        return [
            (
                self._weight_fd,
                self._weight_base,
                local_rows,
                self.dim,
                memoryview(weight.numpy()).cast("B"),
            ),
            (
                self._scale_fd,
                self._scale_base,
                local_rows,
                self.scale_columns,
                memoryview(scale.numpy()).cast("B"),
            ),
        ]

    def dequantize(self, weight: Tensor, scale: Tensor) -> Tensor:
        """fp8 e4m3 rows (as uint8) times ue8m0 block scales -> fp32 rows."""

        rows = weight.shape[0]
        values = (
            weight.view(torch.float8_e4m3fn)
            .to(torch.float32)
            .view(rows, self.scale_columns, self.dim // self.scale_columns)
        )
        # A ue8m0 byte is an fp32 exponent field, so shifting it into place
        # gives 2**(exponent - 127) directly.
        scales = (scale.to(torch.int32) << 23).view(torch.float32)
        return (values * scales[:, :, None]).reshape(rows, self.dim)


def _read_rows(
    descriptor: int,
    base: int,
    rows: list[int],
    lo: int,
    hi: int,
    row_bytes: int,
    buffer: memoryview,
) -> None:
    for index in range(lo, hi):
        offset = base + rows[index] * row_bytes
        view = buffer[index * row_bytes : (index + 1) * row_bytes]
        _pread_exact(descriptor, view, offset)


def _parallel_read(
    jobs: list[tuple[int, int, list[int], int, memoryview]],
    *,
    threads: int,
    chunk: int,
) -> None:
    """Dispatch every row of every job at once across the shared pool."""

    total = sum(len(job[2]) for job in jobs)
    if total == 0:
        return
    if total == 1:
        for descriptor, base, rows, row_bytes, buffer in jobs:
            _read_rows(descriptor, base, rows, 0, len(rows), row_bytes, buffer)
        return
    # A task carries ceil(total / threads) rows, capped for prefill batches.
    chunk = max(1, min(chunk, -(-total // threads)))
    pool = _thread_pool(threads)
    futures = [
        pool.submit(
            _read_rows,
            descriptor,
            base,
            rows,
            lo,
            min(lo + chunk, len(rows)),
            row_bytes,
            buffer,
        )
        for descriptor, base, rows, row_bytes, buffer in jobs
        for lo in range(0, len(rows), chunk)
    ]
    for future in futures:
        future.result()


def gather_dequant_many(
    requests: list[tuple[DiskEngramTable, Tensor, Tensor]],
    *,
    threads: int = 32,
    chunk: int = 16,
) -> list[Tensor]:
    """Gather and dequantize rows for several tables in one parallel batch.

    Each request is ``(table, rel, owned)`` where ``rel`` holds rank-local row
    ids and ``owned`` marks ids that fall inside this rank's range; unowned
    rows read row 0 and are zeroed afterwards, matching the in-memory kernel.
    Rows are de-duplicated per table, which matters because natural text
    repeats n-grams heavily.
    """

    plans: list[tuple[DiskEngramTable, Tensor, Tensor, Tensor, Tensor]] = []
    jobs: list[tuple[int, int, list[int], int, memoryview]] = []
    for table, rel, owned in requests:
        unique, inverse = torch.unique(rel, return_inverse=True)
        weight = torch.empty((unique.numel(), table.dim), dtype=torch.uint8)
        scale = torch.empty((unique.numel(), table.scale_columns), dtype=torch.uint8)
        jobs += table.read_jobs(unique.tolist(), weight, scale)
        plans.append((table, weight, scale, inverse, owned))

    _parallel_read(jobs, threads=threads, chunk=chunk)

    outputs: list[Tensor] = []
    for table, weight, scale, inverse, owned in plans:
        rows = table.dequantize(weight, scale)[inverse]
        rows[~owned] = 0
        outputs.append(rows.to(torch.bfloat16))
    return outputs


def from_rank_environment(environment: object) -> EngramDiskConfig:
    """Build an :class:`EngramDiskConfig` from a ``RankEnvironment``."""

    return EngramDiskConfig(
        enabled=bool(getattr(environment, "engram_disk", False)),
        directory=getattr(environment, "engram_directory", None),
        threads=int(getattr(environment, "engram_threads", 32)),
        chunk=int(getattr(environment, "engram_chunk", 16)),
    )
