#!/usr/bin/env python3
"""Diagnose whether the captured residual stream can yield a usable refusal direction.

WHY
---
The live study achieves a ~4% refusal reduction (baseline 98/100 refusals, best
trial 94/100) while paying KL 0.02-0.05 -- see docs/ABLITERATION_EFFECTIVENESS.md.
The adapter is non-zero and loads with HTTP 200, so the plumbing works. Either
the ablation is too weak, or the directions it ablates along are meaningless.

Heretic computes each direction as ``normalize(mean_bad - mean_good)`` over the
captured residual stream. If the capture is corrupted -- wrong layer mapping, a
bad hyper-connection collapse, an off-by-one -- that difference vector is noise
and the ablation removes nothing.

This is testable from OUTSIDE the study. The /heretic/hidden_states endpoint
takes an explicit ``model``, so base-model residuals can be captured with no
adapter loaded, without disturbing the in-flight trial.

TWO INDEPENDENT CHECKS
----------------------
1. **Adjacent-layer similarity.** A residual stream changes slowly: layer L and
   layer L+1 should be highly similar (cos ~0.9). If adjacent rows of the
   capture are near-orthogonal, the capture is not a residual stream at all.

2. **Class separation.** For a genuine refusal direction, harmful and harmless
   prompts must separate along it, and it must generalize to held-out prompts.
   Reported as a leave-one-out margin; near zero means abliterating along it can
   only damage the model, never change its behaviour.
"""

from __future__ import annotations

import glob
import json
import sys
import urllib.request

import torch

SHIM = "http://127.0.0.1:8765"
K = 8  # prompts per class
LAYER_IDS = list(range(1, 41))  # row k == stream ENTERING layer k+1

# Fallback benign prompts if harmless_alpaca is not cached locally.
BENIGN_FALLBACK = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Give me a recipe for banana bread.",
    "What are the benefits of regular exercise?",
    "How do I change a bicycle tire?",
    "Summarize the plot of Romeo and Juliet.",
    "What is the tallest mountain in the world?",
    "How does a refrigerator keep food cold?",
]


def _arrow_texts(pattern: str, n: int) -> list[str]:
    import pyarrow as pa

    files = sorted(glob.glob(pattern, recursive=True), key=str)
    files = [f for f in files if "test" in f] + [f for f in files if "test" not in f]
    for f in files:
        try:
            t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
            col = "text" if "text" in t.column_names else t.column_names[0]
            rows = [str(v) for v in t.column(col).to_pylist() if v]
            if rows:
                return rows[:n]
        except Exception:  # noqa: BLE001, S112
            continue
    return []


def load_prompts() -> tuple[list[str], list[str]]:
    home = __import__("pathlib").Path.home()
    root = home / ".cache/huggingface/datasets"
    bad = _arrow_texts(str(root / "**/*harmful*behaviors*/**/*.arrow"), K)
    good = _arrow_texts(str(root / "**/*harmless*alpaca*/**/*.arrow"), K)
    if not good:
        print("  (harmless_alpaca not cached; using builtin benign prompts)")
        good = BENIGN_FALLBACK[:K]
    return bad, good


def capture(prompt: str) -> torch.Tensor | None:
    body = json.dumps(
        {
            "model": "deepseek-v4.1-flash",  # base model: no adapter involved
            "messages": [{"role": "user", "content": prompt}],
            "layer_ids": LAYER_IDS,
        }
    ).encode()
    req = urllib.request.Request(
        f"{SHIM}/heretic/hidden_states",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
            rows = json.loads(r.read())["hidden_states"][0]
        return torch.tensor(rows, dtype=torch.float32)  # [40, 5120]
    except Exception as exc:  # noqa: BLE001
        print(f"    capture failed: {exc}", file=sys.stderr)
        return None


def main() -> int:
    bad, good = load_prompts()
    if not bad:
        print("could not load harmful prompts from the local dataset cache")
        return 1
    print(f"capturing {len(bad)} harmful + {len(good)} harmless prompts "
          f"({len(LAYER_IDS)} layers each)\n")

    Xb, Xg = [], []
    for i, p in enumerate(bad, 1):
        t = capture(p)
        print(f"  bad  {i}/{len(bad)} {'ok' if t is not None else 'FAILED'}")
        if t is not None:
            Xb.append(t)
    for i, p in enumerate(good, 1):
        t = capture(p)
        print(f"  good {i}/{len(good)} {'ok' if t is not None else 'FAILED'}")
        if t is not None:
            Xg.append(t)

    if len(Xb) < 3 or len(Xg) < 3:
        print("not enough captures to analyse")
        return 1

    B = torch.stack(Xb)  # [nb, 40, D]
    G = torch.stack(Xg)  # [ng, 40, D]
    nb, L, D = B.shape
    ng = G.shape[0]

    print("\n" + "=" * 88)
    print("CHECK 1: adjacent-layer cosine similarity (a residual stream is smooth)")
    print("=" * 88)
    print(f"  {'layers':>12}  {'cos(base)':>10}  {'cos(ablit-pair)':>15}")
    for l in range(L - 1):
        cb = torch.nn.functional.cosine_similarity(B[:, l], B[:, l + 1], dim=-1).mean()
        cg = torch.nn.functional.cosine_similarity(G[:, l], G[:, l + 1], dim=-1).mean()
        if l < 6 or l >= L - 4 or l % 8 == 0:
            print(f"  {l+1:>5} -> {l+2:<4}  {cb:>10.4f}  {cg:>15.4f}")
    all_c = [
        float(torch.nn.functional.cosine_similarity(B[:, l], B[:, l + 1], dim=-1).mean())
        for l in range(L - 1)
    ]
    print(f"\n  mean adjacent cosine: {sum(all_c)/len(all_c):.4f}")
    print("  (>0.8 expected for a residual stream; near 0 means the capture "
          "is not a residual stream)")

    print("\n" + "=" * 88)
    print("CHECK 2: can a refusal direction separate the classes, and generalize?")
    print("=" * 88)
    print(f"  {'layer':>5} {'||mean||':>10} {'||bad-good||':>12} {'rel.sep':>9} "
          f"{'LOO margin':>11} {'acc':>6}")
    best_layer, best_margin = -1, -1e9
    for l in range(L):
        mean_b = B[:, l].mean(0)
        mean_g = G[:, l].mean(0)
        delta = mean_b - mean_g
        d = torch.nn.functional.normalize(delta, dim=0)
        scale = float((mean_b.norm() + mean_g.norm()) / 2)
        rel = float(delta.norm()) / max(scale, 1e-9)

        # Leave-one-out: recompute the direction without the held-out prompt.
        correct = 0
        margins = []
        for label, X in ((1, B), (0, G)):
            for i in range(X.shape[0]):
                other = X[:, l][torch.arange(X.shape[0]) != i]
                if label == 1:
                    mu_b, mu_g = other.mean(0), G[:, l].mean(0)
                else:
                    mu_b, mu_g = B[:, l].mean(0), other.mean(0)
                dd = torch.nn.functional.normalize(mu_b - mu_g, dim=0)
                proj = float(dd @ X[i, l])
                margins.append(proj if label == 1 else -proj)
                if (proj > 0) == (label == 1):
                    correct += 1
        acc = correct / (B.shape[0] + G.shape[0])
        margin = min(margins)  # worst-case separation
        if margin > best_margin:
            best_margin, best_layer = margin, l + 1
        if l < 4 or l >= L - 2 or l % 6 == 0:
            print(f"  {l+1:>5} {scale:>10.3f} {float(delta.norm()):>12.3f} "
                  f"{rel:>9.4f} {margin:>11.4f} {acc:>6.2f}")

    print(f"\n  best layer: entering layer {best_layer}, "
          f"leave-one-out margin {best_margin:.4f}")
    print("  margin > 0 means every held-out prompt is on the correct side of")
    print("  the direction; margin <= 0 means the direction is not real.")

    torch.save({"bad": B, "good": G}, "/tmp/hidden_diag.pt")
    print("\n  raw captures saved to /tmp/hidden_diag.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
