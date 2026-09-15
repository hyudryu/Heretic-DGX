#!/usr/bin/env python3
"""What ranges did Optuna actually search, and what did it converge on?

The parameter/score correlation showed max_weight has no effect on refusals
while coverage (min_weight_distance) does. That only means "more trials will not
help" if the strength range itself is exhausted -- so read the distributions
Optuna recorded, which state the bounds exactly.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "optuna-journal.jsonl")

dists: dict[str, dict] = {}
values: dict[str, list[float]] = defaultdict(list)
trial_values: dict[int, dict[str, float]] = defaultdict(dict)

for line in path.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    name = rec.get("param_name")
    if not name or "param_value_internal" not in rec:
        continue
    tid = rec.get("trial_id")
    v = rec["param_value_internal"]
    values[name].append(v)
    if tid is not None:
        trial_values[tid][name] = v
    d = rec.get("distribution")
    if d:
        try:
            dists[name] = json.loads(d)
        except ValueError:
            dists[name] = {"raw": d}

print("=" * 86)
print("SEARCH SPACE -- bounds Optuna recorded")
print("=" * 86)
print(f"  {'parameter':>26} {'low':>10} {'high':>10} {'observed min':>13} {'observed max':>13}")
print("-" * 86)
for name in sorted(dists):
    attrs = dists[name].get("attributes", {})
    low = attrs.get("low")
    high = attrs.get("high")
    vs = values.get(name, [])
    lo = f"{min(vs):.3f}" if vs else "-"
    hi = f"{max(vs):.3f}" if vs else "-"
    print(f"  {name:>26} {str(low):>10} {str(high):>10} {lo:>13} {hi:>13}")

print()
print("=" * 86)
print("DID THE SEARCH EXHAUST THE STRENGTH RANGE?")
print("=" * 86)
mw = values.get("attn.o_proj.max_weight") or values.get("max_weight") or []
if mw:
    n = len(mw)
    top = sorted(mw)[-max(1, n // 10) :]
    print(f"  max_weight: n={n}  min={min(mw):.3f}  max={max(mw):.3f}")
    print(f"  top decile mean = {sum(top)/len(top):.3f} of an upper bound around 1.5")
    print()
    print("  NOTE: if the bound is ~1.5 and nothing above 1.5 was ever tried,")
    print("  the observed 'no effect' is not evidence that strength is")
    print("  irrelevant -- it is evidence that strength was never pushed.")

print()
print("=" * 86)
print("PARAMETER CO-OCCURRENCE across trials")
print("=" * 86)
print(f"  trials with recorded parameters: {len(trial_values)}")
for name in sorted(values):
    print(f"    {name:>28}: {len(values[name])} values")

# Which trial has the widest coverage and the fewest refusals, if any?
print()
print("=" * 86)
print("EXTREMES")
print("=" * 86)
mwd = values.get("attn.o_proj.min_weight_distance") or []
if mwd:
    print(f"  min_weight_distance: min={min(mwd):.2f} max={max(mwd):.2f}")
mwp = values.get("attn.o_proj.max_weight_position") or []
if mwp:
    print(f"  max_weight_position: min={min(mwp):.2f} max={max(mwp):.2f}")
