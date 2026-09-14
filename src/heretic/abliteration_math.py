# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backend-independent directional-ablation math.

This module deliberately knows nothing about Transformers, PEFT, or any
inference engine. It takes an ordinary weight matrix and an ordinary direction
vector and returns LoRA factors, so that every backend -- the in-process
Transformers path and the remote vLLM path -- produces byte-identical
abliteration for the same inputs.

The numerics here are a direct extraction of what ``Model.abliterate`` used to do
inline, with one deliberate difference noted on :func:`layer_ablation_weight`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch import Tensor

from .tp_capabilities import LoraFactors

__all__ = [
    "LoraFactors",
    "RowNormalizationLike",
    "compute_directional_lora",
    "layer_ablation_weight",
    "normalization_names",
]

#: Row-normalization strategies. Mirrors ``config.RowNormalization`` without
#: importing it, so this module stays free of settings dependencies.
RowNormalizationLike = str

_NORMALIZATIONS = frozenset({"none", "pre", "full"})

# ``svd_lowrank`` is randomized; these match the original call exactly.
FULL_SVD_OVERSAMPLING = 4
FULL_SVD_NITER = 6

# Guards ``F.normalize`` against all-zero rows, matching the original inline
# expression's effective behaviour on the values Heretic actually sees.
_NORM_EPSILON = 1e-12


def normalization_names() -> tuple[str, ...]:
    """Every accepted normalization name, sorted."""

    return tuple(sorted(_NORMALIZATIONS))


def _require_normalization(normalization: str) -> str:
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            f"unsupported row normalization: {normalization!r}; "
            f"expected one of {', '.join(normalization_names())}"
        )
    return normalization


def compute_directional_lora(
    weight: Tensor,
    direction: Tensor,
    *,
    strength: float,
    normalization: str,
    rank: int,
    seed: int,
) -> LoraFactors:
    """Compute LoRA factors that subtract ``strength`` along ``direction``.

    ``weight`` is the target projection's weight matrix, shaped
    ``(d_out, d_in)``; ``direction`` is a vector in the projection's *output*
    space, shaped ``(d_out,)``. The returned factors satisfy, for the linear
    layer this ablation targets::

        W_ablated = W + (b @ a).to(W.dtype)

    with the row-norm-preservation semantics selected by ``normalization``:

    ``"none"``
        Plain projection: ``a = v @ W``, ``b = -strength * v``.
    ``"pre"``
        Row-normalize ``W`` before projecting, then restore the original row
        norms on ``b``, so the delta scales with the original row norms.
    ``"full"``
        Apply the projected delta, renormalize, restore the original row norms,
        subtract the original, and keep the best rank-``rank`` approximation
        via a seeded randomized SVD. This is the norm-preserving variant.

    The math runs in float32 regardless of the input dtype, matching the
    original implementation, because the SVD in the ``"full"`` path is not
    stable in reduced precision.

    ``rank`` applies only to ``"full"``. ``"none"`` and ``"pre"`` are each a
    single outer product and are therefore always rank 1; this is preserved
    behaviour, and Heretic pins its own ``lora_rank = 1`` for those modes.
    """

    _require_normalization(normalization)

    if weight.dim() != 2:
        raise ValueError(
            f"directional ablation requires a 2D weight matrix, got shape {tuple(weight.shape)}"
        )
    if direction.dim() != 1:
        raise ValueError(
            f"directional ablation requires a 1D direction vector, got shape {tuple(direction.shape)}"
        )
    if weight.shape[0] != direction.shape[0]:
        raise ValueError(
            "direction vector must match the weight matrix output dimension: "
            f"weight d_out={weight.shape[0]}, direction={direction.shape[0]}"
        )
    if rank < 1:
        raise ValueError(f"LoRA rank must be positive, got {rank}")

    matrix = weight.to(torch.float32)
    vector = direction.to(matrix.dtype)

    # Captured before normalization, because "full" subtracts it at the end and
    # "pre"/"full" both restore its row norms.
    original = matrix if normalization == "full" else None

    row_norms: Tensor | None = None
    if normalization != "none":
        row_norms = LA.vector_norm(matrix, dim=1, keepdim=True)
        matrix = F.normalize(matrix, p=2, dim=1)

    a = (vector @ matrix).view(1, -1)
    b = (-strength * vector).view(-1, 1)

    if normalization == "pre":
        assert row_norms is not None
        b = row_norms * b
    elif normalization == "full":
        assert row_norms is not None
        assert original is not None

        matrix = matrix + b @ a
        matrix = F.normalize(matrix, p=2, dim=1)
        matrix = matrix * row_norms
        matrix = matrix - original

        # Randomized SVD: reseed immediately before the call so the result
        # depends only on ``seed``, never on RNG history.
        torch.manual_seed(seed)
        # Safe to call without CUDA; silently ignored in that case.
        torch.cuda.manual_seed_all(seed)

        u, s, vh = torch.svd_lowrank(
            matrix,
            q=2 * rank + FULL_SVD_OVERSAMPLING,
            niter=FULL_SVD_NITER,
        )
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:, :rank].T

        # Split the singular values evenly so neither factor dominates.
        sqrt_s = torch.sqrt(s)
        b = u @ torch.diag(sqrt_s)
        a = torch.diag(sqrt_s) @ vh

    return LoraFactors(a=a, b=b)


@dataclass(frozen=True, slots=True)
class LayerWeight:
    """A layer's ablation strength under Heretic's triangular schedule."""

    weight: float
    in_range: bool


def layer_ablation_weight(
    layer_index: int,
    *,
    max_weight: float,
    max_weight_position: float,
    min_weight: float,
    min_weight_distance: float,
) -> float | None:
    """Ablation strength for one layer, or ``None`` when the layer is untouched.

    The schedule is Heretic's: ``max_weight`` at ``max_weight_position``, falling
    linearly to ``min_weight`` at ``min_weight_distance`` away, and zero beyond.

    Deliberate divergence from the original inline expression: when
    ``min_weight_distance`` is exactly ``0`` the original evaluated
    ``0 / 0 -> nan`` for the layer at ``max_weight_position`` and multiplied the
    whole delta by ``nan``. Here that degenerate case resolves to its continuous
    limit -- ``max_weight`` at the exact position, untouched everywhere else.
    """

    if min_weight_distance < 0:
        raise ValueError(
            f"min_weight_distance must be nonnegative, got {min_weight_distance}"
        )

    distance = abs(layer_index - max_weight_position)
    if distance > min_weight_distance:
        return None
    if min_weight_distance == 0:
        return max_weight

    return max_weight + (distance / min_weight_distance) * (min_weight - max_weight)
