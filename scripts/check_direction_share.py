#!/usr/bin/env python3
"""Is the shared expert's output space oriented toward the refusal direction?

STRICTLY READ-ONLY. Never touches the checkpoint's contents, the engine, or the
running study.

THE TEST
--------
A linear map W: R^n -> R^5120 can only write vectors inside its own column
space. The refusal direction d lives in R^5120. So the share of d that W is
*capable* of producing is ||P_W d|| / ||d||, where P_W projects onto span(W).

A random n-dimensional subspace of R^5120 already contains a share
sqrt(n/5120) of any given direction -- for the shared expert (n = 2304) that is
0.671. So the number only says something if it is well above that baseline: it
would mean the learned weight is oriented toward the refusal direction rather
than being indifferent to it.

wo_b is included as a control. Its input dimension (8192) exceeds its output
(5120), so its column space is generically all of R^5120 and the share is 1.0 --
which is why that number alone cannot distinguish anything, and why the shared
expert is the interesting case.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

M = Path("/models/DeepSeek-V4.1-Flash")
DIAG = Path("/tmp/hidden_diag.pt")
LAYERS = 40
E8M0_BIAS = 127
HIDDEN = 5120


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def data_start(path: Path) -> int:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
    return 8 + n


def dequant(path: Path, hdr: dict, base: str) -> torch.Tensor:
    w = hdr[f"{base}.weight"]
    s = hdr[f"{base}.scale"]
    start = data_start(path)
    with path.open("rb") as f:
        f.seek(start + w["data_offsets"][0])
        raw = f.read(w["data_offsets"][1] - w["data_offsets"][0])
        f.seek(start + s["data_offsets"][0])
        raw_s = f.read(s["data_offsets"][1] - s["data_offsets"][0])
    wp = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn)
    wp = wp.to(torch.float32).reshape(w["shape"])
    sp = torch.frombuffer(bytearray(raw_s), dtype=torch.uint8).to(torch.float32)
    sp = torch.pow(2.0, sp - E8M0_BIAS).reshape(s["shape"])
    br, bc = wp.shape[0] // s["shape"][0], wp.shape[1] // s["shape"][1]
    return (wp.reshape(s["shape"][0], br, s["shape"][1], bc) * sp[:, None, :, None]).reshape(
        wp.shape
    )


def share_within_column_space(weight: torch.Tensor, d: torch.Tensor) -> float:
    """||P_W d|| / ||d||, via a thin QR of W (cheaper than an SVD)."""
    q, _ = torch.linalg.qr(weight)  # [out, n], orthonormal columns
    return float((q.t() @ d).norm()) / float(d.norm())


def main() -> int:
    if not DIAG.is_file():
        print(f"missing {DIAG} -- run scripts/diag_direction.py first")
        return 1
    diag = torch.load(DIAG)
    B, G = diag["bad"], diag["good"]

    # row k of the capture == stream entering layer k+1
    dirs: dict[int, torch.Tensor] = {}
    for k in range(B.shape[1]):
        delta = B[:, k].mean(0) - G[:, k].mean(0)
        dirs[k + 1] = torch.nn.functional.normalize(delta, dim=0)

    idx = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, dict] = {}

    print("=" * 88)
    print("Share of the refusal direction d that each projection could write")
    print("(1.0 = its column space contains all of d)")
    print("=" * 88)
    print(f"  {'layer':>5} {'shared_expert':>15} {'random baseline':>16} "
          f"{'vs baseline':>12} {'wo_b (control)':>16}")
    print("-" * 88)

    shared_shares, wo_shares = [], []
    for L in range(LAYERS):
        d = dirs.get(L if L > 0 else 1)
        if d is None:
            continue
        name = f"layers.{L}.ffn.shared_experts.w2.weight"
        shard = idx[name]
        if shard not in by_shard:
            by_shard[shard] = read_header(M / shard)
        hdr = by_shard[shard]
        w2 = dequant(M / shard, hdr, f"layers.{L}.ffn.shared_experts.w2")
        n_in = w2.shape[1]
        baseline = (n_in / HIDDEN) ** 0.5
        sh = share_within_column_space(w2, d)

        wname = f"layers.{L}.attn.wo_b.weight"
        wshard = idx[wname]
        if wshard not in by_shard:
            by_shard[wshard] = read_header(M / wshard)
        wb = dequant(M / wshard, by_shard[wshard], f"layers.{L}.attn.wo_b")
        sw = share_within_column_space(wb, d)

        shared_shares.append(sh)
        wo_shares.append(sw)
        if L % 5 == 0 or L == LAYERS - 1:
            print(f"  {L:>5} {sh:>15.4f} {baseline:>16.4f} "
                  f"{sh/baseline:>11.2f}x {sw:>16.4f}")

    if shared_shares:
        ms = sum(shared_shares) / len(shared_shares)
        mw = sum(wo_shares) / len(wo_shares)
        base = (2304 / HIDDEN) ** 0.5
        print("-" * 88)
        print(f"  mean shared-expert share : {ms:.4f}   (random baseline {base:.4f}, "
              f"ratio {ms/base:.2f}x)")
        print(f"  mean wo_b share          : {mw:.4f}   (expected ~1.0, uninformative)")
        print()
        above = sum(1 for s in shared_shares if s / base > 1.1)
        print(f"  layers where the shared expert's share exceeds baseline by >10%: "
              f"{above}/{len(shared_shares)}")
        print()
        print("  A share near the baseline means the shared expert is indifferent")
        print("  to the refusal direction and ablating it is unlikely to help.")
        print("  Well above baseline means it is oriented toward it and is a real")
        print("  candidate. Either way this bounds capability, not actual use --")
        print("  only a load test and a refusal measurement settle it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
