#!/usr/bin/env python3
"""Are the hyper-connection mixes near-uniform?

STRICTLY READ-ONLY.

The leading hypothesis for the near-zero abliteration effect is that the capture
collapses the hyper-connection state with a plain `.mean(dim=1)`, while the real
forward collapses it with a learned, position-dependent mix
(`hc_attn_fn` / `hc_attn_scale` / `hc_attn_base` + `pre_mix`). If the learned
mixing is close to uniform, a mean is a good approximation and the hypothesis is
weak. If it is far from uniform, the mean-collapse rotates the direction
substantially and the hypothesis is strong.

This does not prove anything on its own -- the mixes are also input-dependent --
but the learned parameters bound how far from uniform the collapse can be.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

M = Path("/models/DeepSeek-V4.1-Flash")
LAYERS = 40
NAMES = ["hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"]


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def data_start(path: Path) -> int:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
    return 8 + n


DT = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F64": torch.float64,
}


def load(path: Path, hdr: dict, name: str) -> torch.Tensor:
    info = hdr[name]
    start = data_start(path)
    with path.open("rb") as f:
        f.seek(start + info["data_offsets"][0])
        raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
    dt = DT.get(info["dtype"])
    if dt is None:
        raise SystemExit(f"unhandled dtype {info['dtype']} for {name}")
    return torch.frombuffer(bytearray(raw), dtype=dt).to(torch.float32).reshape(info["shape"])


def main() -> int:
    idx = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]
    headers: dict[str, dict] = {}

    print("=" * 88)
    print("Hyper-connection mixing parameters")
    print("=" * 88)
    print(f"  {'tensor':>16} {'shape':>16} {'dtype':>8} {'mean':>10} {'std':>10} "
          f"{'min':>10} {'max':>10}")
    print("-" * 88)

    for base in NAMES:
        name = f"layers.0.{base}"
        if name not in idx:
            print(f"  {base:>16}  (absent)")
            continue
        shard = idx[name]
        if shard not in headers:
            headers[shard] = read_header(M / shard)
        hdr = headers[shard]
        info = hdr[name]
        t = load(M / shard, hdr, name)
        print(f"  {base:>16} {str(tuple(info['shape'])):>16} {info['dtype']:>8} "
              f"{float(t.mean()):>10.4f} {float(t.std()):>10.4f} "
              f"{float(t.min()):>10.4f} {float(t.max()):>10.4f}")

    print()
    print("=" * 88)
    print("How far from uniform is the attention mix across layers?")
    print("=" * 88)

    # hc_attn_base is the per-copy bias; if it is uniform across copies the mean
    # collapse is exact in the bias term.
    for base in ("hc_attn_base", "hc_attn_scale"):
        rows = []
        for L in range(LAYERS):
            name = f"layers.{L}.{base}"
            if name not in idx:
                continue
            shard = idx[name]
            if shard not in headers:
                headers[shard] = read_header(M / shard)
            hdr = headers[shard]
            t = load(M / shard, hdr, name).flatten()
            if t.numel() < 2:
                continue
            m = float(t.mean())
            s = float(t.std())
            # coefficient of variation: 0 == perfectly uniform across copies
            rows.append((L, t.numel(), m, s, s / abs(m) if m else float("inf")))
        if not rows:
            continue
        print(f"\n  {base}: {len(rows)} layers, {rows[0][1]} values each")
        print(f"    {'layer':>5} {'mean':>10} {'std':>10} {'std/|mean|':>12}")
        for L, n, m, s, cv in rows[:3] + rows[-2:]:
            print(f"    {L:>5} {m:>10.4f} {s:>10.4f} {cv:>12.4f}")
        avg_cv = sum(r[4] for r in rows) / len(rows)
        print(f"    mean std/|mean| across layers: {avg_cv:.4f}")
        print(f"    (0 would mean identical across the hc copies, so that "
              f"a mean collapse loses nothing)")

    print()
    print("  Interpretation: a large spread means the per-copy coefficients")
    print("  differ substantially, so collapsing with a plain mean produces a")
    print("  different vector than the learned collapse does -- which is what")
    print("  the leading hypothesis predicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
