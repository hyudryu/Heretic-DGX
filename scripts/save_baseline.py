#!/usr/bin/env python3
"""Preserve the pre-patch direction baseline, while it is still reproducible.

After the hc-collapse patch is deployed, the unpatched engine no longer exists,
so the mean-collapsed directions cannot be regenerated. Without them there is no
A/B: we would see that refusals move (or not) but could not say whether the
directions changed, which is the actual claim being tested.

The captures at /tmp/hidden_diag.pt came from the unpatched engine. Derive the
per-layer directions, record a few quality statistics, and write the result
somewhere durable.

Read-only against the cluster: this reads a local torch file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hidden_diag.pt")
OUT = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else "/home/hyudryu/.cache/huggingface/heretic-baseline-mean-collapse.pt"
)


def main() -> int:
    if not SRC.is_file():
        print(f"missing {SRC} -- run scripts/diag_direction.py first")
        return 1

    d = torch.load(SRC)
    B, G = d["bad"], d["good"]
    nb, L, D = B.shape
    print(f"captures: {nb} harmful, {G.shape[0]} harmless, {L} layers, width {D}")

    dirs = {}
    stats = []
    for k in range(L):
        delta = B[:, k].mean(0) - G[:, k].mean(0)
        dl = torch.nn.functional.normalize(delta, dim=0)
        dirs[k + 1] = dl  # row k == stream entering layer k+1
        stats.append(
            {
                "entering_layer": k + 1,
                "mean_norm": float(B[:, k].mean(0).norm()),
                "delta_norm": float(delta.norm()),
                "rel_sep": float(delta.norm()) / float(
                    max((B[:, k].mean(0).norm() + G[:, k].mean(0).norm()) / 2, 1e-9)
                ),
            }
        )

    payload = {
        "directions": dirs,  # entering layer -> unit direction
        "stats": stats,
        "source": str(SRC),
        "note": (
            "Pre-patch baseline: aux hidden states collapsed with a plain mean "
            "over the hyper-connection copies. Captured from the unpatched "
            "engine before the hc-collapse patch was deployed."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")

    print()
    print("  entering_layer  mean_norm  delta_norm  rel_sep")
    for s in stats:
        if s["entering_layer"] % 5 == 0 or s["entering_layer"] == 1:
            print(
                f"  {s['entering_layer']:>14} {s['mean_norm']:>10.3f} "
                f"{s['delta_norm']:>11.3f} {s['rel_sep']:>8.4f}"
            )

    # Also write a small JSON summary, which is easier to diff by eye later.
    summary = OUT.with_suffix(".json")
    summary.write_text(
        json.dumps(
            {
                "source": str(SRC),
                "layers": L,
                "width": D,
                "n_harmful": nb,
                "n_harmless": int(G.shape[0]),
                "stats": [
                    {k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()}
                    for s in stats
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
