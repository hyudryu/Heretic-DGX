#!/usr/bin/env python3
"""Feasibility check: can the existing FP8 block-dequant path read the shared
expert down-projection, and what would ablating it cost?

STRICTLY READ-ONLY. Reads tensor headers and one layer's payload (11.25 MiB);
never writes to the checkpoint, never touches the engine, never touches the
running study.

WHY
---
docs/ABLITERATION_EFFECTIVENESS.md establishes that the attention-only search
space is exhausted and ablation strength does nothing. The next candidate
component is the feed-forward down projection. V4.1 has no single
`mlp.down_proj`: there are 384 routed experts per layer plus one shared expert.
This checks whether the shared expert is a viable target on the same terms as
`wo_b`:

  wo_b                  [5120, 8192]  F8_E4M3  scale [160, 256]
  shared_experts.w2     [5120, 2304]  F8_E4M3  scale [160,  72]

Both are [hidden, d_in] with 32x32 block scaling, so both write into the 5120
residual stream where the refusal direction is measured -- meaning the SAME
direction applies to both.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

M = Path("/models/DeepSeek-V4.1-Flash")
LAYERS = 40
SHARED = "ffn.shared_experts.w2"
WO_B = "attn.wo_b"

# e8m0 (ue8m0): power-of-two block scales, exponent in the low 8 bits.
E8M0_BIAS = 127


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def data_start(path: Path) -> int:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
    return 8 + n


def read_range(path: Path, offsets: list[int]) -> bytes:
    start = data_start(path)
    with path.open("rb") as f:
        f.seek(start + offsets[0])
        return f.read(offsets[1] - offsets[0])


def main() -> int:
    idx = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]

    print("=" * 84)
    print("1. DISCOVERY: is there exactly one shared-expert down projection per layer?")
    print("=" * 84)
    found = {}
    for name, shard in idx.items():
        if name.endswith(f"{SHARED}.weight") and name.startswith("layers."):
            rest = name[len("layers.") :]
            head = rest.split(".", 1)[0]
            if head.isdigit():
                found[int(head)] = (name, shard)
    print(f"  found {len(found)} layers, expected {LAYERS}")
    missing = sorted(set(range(LAYERS)) - set(found))
    if missing:
        print(f"  MISSING layers: {missing}")
        return 1
    print("  every layer 0..39 present  ✓")
    # mtp.* is excluded by the `layers.` prefix + digit parse, same as wo_b.
    mtp = [n for n in idx if SHARED in n and not n.startswith("layers.")]
    print(f"  non-layer (mtp) tensors of this form, correctly excluded: {len(mtp)}")

    print()
    print("=" * 84)
    print("2. GEOMETRY: does it match wo_b's conventions?")
    print("=" * 84)
    shapes = set()
    scal = set()
    for L in range(LAYERS):
        name, shard = found[L]
        hdr = read_header(M / shard)
        shapes.add(tuple(hdr[name]["shape"]))
        scales = hdr.get(name.replace(".weight", ".scale"))
        if scales:
            scal.add(tuple(scales["shape"]))
    print(f"  w2.weight shapes across layers : {shapes}")
    print(f"  w2.scale  shapes across layers : {scal}")
    h, dim = next(iter(shapes))
    print(f"  -> output {h} (== hidden size, the residual space)  ✓")
    for s in scal:
        rows, cols = s
        print(f"  -> block scale {rows}x{cols}: {h/rows:.0f} rows x {dim/cols:.0f} cols per block")

    print()
    print("=" * 84)
    print("3. DEQUANT: does the existing block-dequant path work on this tensor?")
    print("=" * 84)
    L = 20
    name, shard = found[L]
    path = M / shard
    hdr = read_header(path)
    w = hdr[name]
    s = hdr[name.replace(".weight", ".scale")]
    raw = read_range(path, w["data_offsets"])
    raw_s = read_range(path, s["data_offsets"])

    # float8_e4m3fn -> float32; ue8m0 -> 2**(b-127)
    wp = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn)
    wp = wp.to(torch.float32).reshape(w["shape"])
    sp = torch.frombuffer(bytearray(raw_s), dtype=torch.uint8).to(torch.float32)
    sp = torch.pow(2.0, sp - E8M0_BIAS).reshape(s["shape"])

    hh, dd = wp.shape
    br, bc = hh // s["shape"][0], dd // s["shape"][1]
    deq = (
        wp.reshape(s["shape"][0], br, s["shape"][1], bc)
        * sp[:, None, :, None]
    ).reshape(hh, dd)

    print(f"  layer {L}: {tuple(deq.shape)}  finite={bool(torch.isfinite(deq).all())}")
    row_norms = deq.norm(dim=1)
    print(f"  row norms: min={float(row_norms.min()):.4f} "
          f"mean={float(row_norms.mean()):.4f} max={float(row_norms.max()):.4f}")
    zero_rows = int((row_norms == 0).sum())
    print(f"  zero rows: {zero_rows}  (wo_b measured 0)")

    print()
    print("=" * 84)
    print("4. COST of adding this as an ablation target (rank 3 LoRA)")
    print("=" * 84)
    r = 3
    per_layer = r * dd + hh * r
    print(f"  per layer : A[{r},{dd}] + B[{hh},{r}] = {per_layer} params")
    print(f"  per trial : {per_layer*LAYERS} params = {per_layer*LAYERS*4/1e6:.1f} MB (fp32)")
    print(f"  read cost : {LAYERS*hh*dd/2**20:.0f} MiB of FP8 (wo_b is "
          f"{LAYERS*5120*8192/2**20:.0f} MiB)")
    print()
    print("  Compare: ablating all 384 routed experts instead would be")
    ex_d = 1152
    ex = (r * ex_d + hh * r) * 384 * LAYERS
    print(f"    {ex} params = {ex*4/1e9:.2f} GB per trial, rsynced to 3 peers every trial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
