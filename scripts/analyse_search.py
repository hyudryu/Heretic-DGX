#!/usr/bin/env python3
"""Does ANY searched parameter move the refusal count? A non-invasive test.

WHY
---
The live study achieves a ~4% refusal reduction while paying KL 0.02-0.05, and
every plumbing explanation has been eliminated by measurement (capture faithful,
directions real, LoRA math correct, engine really applies it). Two architectural
hypotheses remain, and both predict the same observable:

    if the ablation does not reach the refusal mechanism, then NO search
    parameter should correlate with the refusal count.

That is testable from the journal already on disk, with no pause and no engine
load. The search has explored max_weight over a wide range; if refusals are
unaffected by how hard the ablation is applied, strength is not the variable.

Reading the numbers correctly matters: ``Keywords`` is the number of REFUSALS
and is MINIMISED (KeywordRate declares optimization = "minimize").
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "prod-run1.log")
text = LOG.read_text(errors="replace")

TRIAL_RE = re.compile(
    r"Running trial (\d+) of (\d+)\.\.\.(.*?)"
    r"\*\s*Metrics:\s*\n\s*\*\s*Keywords:\s*(\d+)/100\s*\n\s*\*\s*KL divergence:\s*([\d.]+)",
    re.S,
)
PARAM_RE = {
    "direction_index": r"direction_index = ([0-9.]+|per layer)",
    "max_weight": r"attn\.o_proj\.max_weight = ([0-9.]+)",
    "max_weight_position": r"attn\.o_proj\.max_weight_position = ([0-9.]+)",
    "min_weight": r"attn\.o_proj\.min_weight = ([0-9.]+)",
    "min_weight_distance": r"attn\.o_proj\.min_weight_distance = ([0-9.]+)",
}

records = []
for m in TRIAL_RE.finditer(text):
    trial, total, body, kw, kl = m.groups()
    rec: dict[str, object] = {
        "trial": int(trial),
        "refusals": int(kw),
        "kl": float(kl),
    }
    for name, pat in PARAM_RE.items():
        pm = re.search(pat, body)
        if not pm:
            rec[name] = None
        elif pm.group(1) == "per layer":
            rec[name] = None  # categorical, handled separately
        else:
            rec[name] = float(pm.group(1))
    records.append(rec)

baselines = [int(b) for b in re.findall(r"Baseline Keywords: (\d+)/100", text)]

print(f"parsed {len(records)} scored trials from {LOG.name}")
print(f"baselines recorded in the log: {baselines}")
if not records:
    sys.exit("nothing parsed")

ref = [r["refusals"] for r in records]
print(f"\nrefusals: min={min(ref)} max={max(ref)} mean={sum(ref)/len(ref):.2f}  (n={len(ref)})")

base = baselines[0] if baselines else None
if base is not None:
    better = sum(1 for v in ref if v < base)
    equal = sum(1 for v in ref if v == base)
    worse = sum(1 for v in ref if v > base)
    print(f"vs baseline {base}:  better={better}  equal={equal}  worse={worse}")
    print(f"  best improvement: {base - min(ref)} prompts out of {base} "
          f"({100*(base-min(ref))/base:.1f}%)")

print("\n" + "=" * 84)
print("CORRELATION: does any parameter move the refusal count?")
print("=" * 84)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    return pearson(rank(xs), rank(ys))


params = ["max_weight", "max_weight_position", "min_weight", "min_weight_distance"]
print(f"  {'parameter':>22} {'n':>4} {'pearson':>9} {'spearman':>9}   range explored")
print("-" * 84)
for p in params:
    pairs = [(r[p], r["refusals"]) for r in records if r[p] is not None]
    if len(pairs) < 3:
        continue
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    pr, sr = pearson(xs, ys), spearman(xs, ys)
    print(f"  {p:>22} {len(pairs):>4} {pr:>9.3f} {sr:>9.3f}   "
          f"{min(xs):.2f} .. {max(xs):.2f}")

# Same for KL, as a control: KL SHOULD respond to the parameters.
print("\n  control -- the same parameters against KL divergence (which does respond):")
print(f"  {'parameter':>22} {'n':>4} {'pearson':>9} {'spearman':>9}")
print("-" * 84)
for p in params:
    pairs = [(r[p], r["kl"]) for r in records if r[p] is not None]
    if len(pairs) < 3:
        continue
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    print(f"  {p:>22} {len(pairs):>4} {pearson(xs, ys):>9.3f} {spearman(xs, ys):>9.3f}")

print("\n" + "=" * 84)
print("INTERPRETATION")
print("=" * 84)
print("""
  If refusals show near-zero correlation with max_weight while KL shows a clear
  one, that is strong evidence the ablation is being applied with real magnitude
  but is not reaching the refusal mechanism. Strength is then NOT the variable,
  and more trials cannot fix it.

  If refusals DO track max_weight, the mechanism works and merely needs the
  search to push harder.""")

# Split by strength quartile for a coarser, more robust view.
ranked = sorted([r for r in records if r["max_weight"] is not None], key=lambda r: r["max_weight"])
if len(ranked) >= 8:
    q = len(ranked) // 4
    print("\n  by max_weight quartile:")
    for label, chunk in (
        ("weakest 25%", ranked[:q]),
        ("Q2", ranked[q : 2 * q]),
        ("Q3", ranked[2 * q : 3 * q]),
        ("strongest 25%", ranked[3 * q :]),
    ):
        rs = [r["refusals"] for r in chunk]
        kls = [r["kl"] for r in chunk]
        ws = [r["max_weight"] for r in chunk]
        print(f"    {label:>13}  max_weight {min(ws):.2f}-{max(ws):.2f}   "
              f"refusals mean {sum(rs)/len(rs):6.2f}   KL mean {sum(kls)/len(kls):.4f}")
