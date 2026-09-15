#!/usr/bin/env python3
"""Does the exported adapter actually encode an ablation along the measured direction?

CONTEXT
-------
Measured on the live run:
  * The captured residual stream is faithful (mean adjacent-layer cosine 0.919).
  * The refusal directions are real: leave-one-out accuracy 1.00 for layers 2-39.
  * Yet the best trial removes ~4% of refusals (baseline 98/100 -> 94/100).

So the direction is good and the adapter is non-zero and loads with HTTP 200.
The remaining question is whether the adapter's weights correspond to the
ablation Heretic intends:

    delta_W  ~=  -strength * (W @ d) outer d

Such a delta is exactly rank 1, and its RIGHT singular vector is ``d``. So if the
exported ``B @ A`` has ``d`` as its dominant right singular vector, Heretic's
construction is correct and the fault lies downstream (in how the engine applies
the adapter). If not, the construction itself is wrong.

This is purely local arithmetic over two files -- it cannot disturb the study.

Usage: verify_adapter_direction.py /tmp/hidden_diag.pt
"""

from __future__ import annotations

import sys

import torch
from safetensors import safe_open

ADAPTER = "/home/hyudryu/.cache/huggingface/heretic-lora/heretic-trial/adapter_model.safetensors"


def main() -> int:
    diag_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hidden_diag.pt"
    diag = torch.load(diag_path)
    B, G = diag["bad"], diag["good"]

    # row k of the capture == stream entering layer k+1
    directions: dict[int, torch.Tensor] = {}
    for k in range(B.shape[1]):
        delta = B[:, k].mean(0) - G[:, k].mean(0)
        directions[k + 1] = torch.nn.functional.normalize(delta, dim=0)

    print("=" * 92)
    print("Does delta_W = B @ A carry `d` as its dominant LEFT singular vector?")
    print("=" * 92)
    print("  delta_W is [d_out=5120, d_in=8192] -- wo_b is NOT square, and `d` is")
    print("  measured in the residual stream (5120), i.e. the OUTPUT space. The")
    print("  intended ablation (I - s d d^T) W has d as its only non-zero LEFT")
    print("  singular direction and W^T d as its right one, so the left side is")
    print("  the correct thing to test.")
    print()
    print(f"  {'layer':>5} {'||delta_W||_F':>13} {'|cos(lsv1, d)|':>15} "
          f"{'|cos(lsv1..3, d)|max':>21}")
    print("-" * 92)

    layers = []
    with safe_open(ADAPTER, framework="pt") as f:
        names = list(f.keys())
        # key form: base_model.model.layers.<L>.attn.wo_b.lora_A.weight
        present = sorted(
            {
                int(n.split(".layers.", 1)[1].split(".", 1)[0])
                for n in names
                if ".layers." in n
            }
        )
        for L in present:
            a = f.get_tensor(f"base_model.model.layers.{L}.attn.wo_b.lora_A.weight").float()
            b = f.get_tensor(f"base_model.model.layers.{L}.attn.wo_b.lora_B.weight").float()
            # peft stores A as [r, in] and B as [out, r]; delta = B @ A
            delta = (b @ a).double()  # [5120, 8192]
            dl = directions.get(L if L > 0 else 1)  # direction for Heretic layer L
            if dl is None:
                continue
            dl = dl.double()

            # top LEFT singular vectors: eigenvectors of delta @ delta^T in the
            # 5120 output space, by orthogonalised power iteration.
            gram = delta @ delta.t()
            vs: list[torch.Tensor] = []
            for _ in range(3):
                w = torch.randn(gram.shape[0], dtype=torch.float64)
                for u in vs:
                    w = w - (w @ u) * u
                for _ in range(80):
                    w = gram @ w
                    for u in vs:
                        w = w - (w @ u) * u
                    w = w / w.norm().clamp_min(1e-30)
                vs.append(w)
            cos1 = abs(float(vs[0] @ dl))
            cosmax = float(torch.stack([v @ dl for v in vs]).abs().max())

            layers.append((L, float(delta.norm()), cos1, cosmax))
            print(f"  {L:>5} {float(delta.norm()):>13.4f} {cos1:>15.4f} {cosmax:>21.4f}")

    if not layers:
        print("  no layer tensors found in the adapter")
        return 1

    cs = [c for _, _, c, _ in layers]
    cm = [m for _, _, _, m in layers]
    print("-" * 92)
    print(f"  mean |cos(top right singular vector, d)| = {sum(cs)/len(cs):.4f}")
    print(f"  mean best over top-3 subspace             = {sum(cm)/len(cm):.4f}")
    print()
    print("  EXPECTED if Heretic's construction is correct: close to 1.0, because")
    print("  a rank-1 outward-product ablation has exactly `d` as its only")
    print("  non-zero right singular direction.")
    print("  A value near 0 means the exported delta is NOT the intended ablation.")

    # Which layers are present at all?
    print()
    print(f"  layers present in the adapter: {len(layers)} of 40 -> "
          f"{sorted(L for L, _, _, _ in layers)[:5]} ... "
          f"{sorted(L for L, _, _, _ in layers)[-3:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
