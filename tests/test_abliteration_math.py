# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parity and contract tests for the extracted abliteration math.

The oracle here is a verbatim copy of what ``Model.abliterate`` used to compute
inline, frozen at the commit that introduced ``abliteration_math``. Every
normalization mode must reproduce it to tight tolerance, so extracting the math
cannot silently change what a trial means.
"""

from __future__ import annotations

import unittest

import torch
import torch.linalg as LA
import torch.nn.functional as F

from heretic.abliteration_math import (
    compute_directional_lora,
    layer_ablation_weight,
    normalization_names,
)
from heretic.tp_capabilities import directional_lora_factors


def _legacy_factors(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    strength: float,
    normalization: str,
    rank: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The original inline math, copied verbatim as a frozen oracle."""

    W = weight.to(torch.float32)
    v = direction.to(W.dtype)

    if normalization == "full":
        W_org = W

    if normalization != "none":
        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
        W = F.normalize(W, p=2, dim=1)

    lora_A = (v @ W).view(1, -1)
    lora_B = (-strength * v).view(-1, 1)

    if normalization == "pre":
        lora_B = W_row_norms * lora_B
    elif normalization == "full":
        W = W + lora_B @ lora_A
        W = F.normalize(W, p=2, dim=1)
        W = W * W_row_norms
        W = W - W_org
        r = rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:, :r].T
        sqrt_S = torch.sqrt(S)
        lora_B = U @ torch.diag(sqrt_S)
        lora_A = torch.diag(sqrt_S) @ Vh

    return lora_A, lora_B


def _fixture(
    d_out: int = 24, d_in: int = 16, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(d_out, d_in, generator=generator)
    direction = torch.randn(d_out, generator=generator)
    return weight, direction


class TestParityWithLegacy(unittest.TestCase):
    """The refactor must not change a single number."""

    def _assert_matches(self, normalization: str, *, rank: int = 3) -> None:
        weight, direction = _fixture()
        strength = 0.75
        seed = 99

        expected_a, expected_b = _legacy_factors(
            weight,
            direction,
            strength=strength,
            normalization=normalization,
            rank=rank,
            seed=seed,
        )
        actual = compute_directional_lora(
            weight,
            direction,
            strength=strength,
            normalization=normalization,
            rank=rank,
            seed=seed,
        )

        self.assertTrue(
            torch.allclose(actual.a, expected_a, atol=1e-6),
            f"{normalization}: lora_A diverged (max abs diff "
            f"{(actual.a - expected_a).abs().max().item():.3e})",
        )
        self.assertTrue(
            torch.allclose(actual.b, expected_b, atol=1e-6),
            f"{normalization}: lora_B diverged (max abs diff "
            f"{(actual.b - expected_b).abs().max().item():.3e})",
        )

    def test_none_matches_legacy(self) -> None:
        self._assert_matches("none")

    def test_pre_matches_legacy(self) -> None:
        self._assert_matches("pre")

    def test_full_matches_legacy(self) -> None:
        self._assert_matches("full")

    def test_full_rank_one_matches_legacy(self) -> None:
        self._assert_matches("full", rank=1)

    def test_full_matches_legacy_on_larger_matrix(self) -> None:
        # Closer to the real wo_b geometry ratio, still cheap.
        weight, direction = _fixture(d_out=256, d_in=128, seed=7)
        expected_a, expected_b = _legacy_factors(
            weight,
            direction,
            strength=1.25,
            normalization="full",
            rank=3,
            seed=1234,
        )
        actual = compute_directional_lora(
            weight,
            direction,
            strength=1.25,
            normalization="full",
            rank=3,
            seed=1234,
        )
        self.assertTrue(torch.allclose(actual.a, expected_a, atol=1e-6))
        self.assertTrue(torch.allclose(actual.b, expected_b, atol=1e-6))


class TestFullNormalizationDeterminism(unittest.TestCase):
    def test_full_svd_is_deterministic_for_fixed_seed(self) -> None:
        weight, direction = _fixture(seed=11)
        first = compute_directional_lora(
            weight,
            direction,
            strength=0.5,
            normalization="full",
            rank=3,
            seed=4242,
        )
        # Churn the global RNG so the result cannot depend on prior state.
        torch.randn(1024)
        second = compute_directional_lora(
            weight,
            direction,
            strength=0.5,
            normalization="full",
            rank=3,
            seed=4242,
        )
        self.assertTrue(torch.equal(first.a, second.a))
        self.assertTrue(torch.equal(first.b, second.b))

    def test_full_svd_changes_with_seed(self) -> None:
        weight, direction = _fixture(seed=13)
        first = compute_directional_lora(
            weight, direction, strength=0.5, normalization="full", rank=3, seed=1
        )
        second = compute_directional_lora(
            weight, direction, strength=0.5, normalization="full", rank=3, seed=2
        )
        self.assertFalse(torch.allclose(first.a, second.a))


class TestAgreementWithTpCapabilities(unittest.TestCase):
    """The unsharded path must agree with the distributed helper.

    ``tp_capabilities.directional_lora_factors`` is the sharded implementation.
    With an identity reduction and a single shard it must produce the same
    factors, otherwise the two backends would disagree about identical inputs.
    """

    def test_none_agrees(self) -> None:
        weight, direction = _fixture()
        pure = compute_directional_lora(
            weight, direction, strength=0.5, normalization="none", rank=1, seed=1
        )
        sharded = directional_lora_factors(
            weight,
            direction,
            strength=0.5,
            normalization="none",
            topology="rowwise",
            sum_across_ranks=lambda tensor: tensor,
        )
        self.assertTrue(torch.allclose(pure.a, sharded.a, atol=1e-6))
        self.assertTrue(torch.allclose(pure.b, sharded.b, atol=1e-6))

    def test_pre_agrees(self) -> None:
        weight, direction = _fixture()
        pure = compute_directional_lora(
            weight, direction, strength=0.5, normalization="pre", rank=1, seed=1
        )
        sharded = directional_lora_factors(
            weight,
            direction,
            strength=0.5,
            normalization="pre",
            topology="rowwise",
            sum_across_ranks=lambda tensor: tensor,
        )
        self.assertTrue(torch.allclose(pure.a, sharded.a, atol=1e-6))
        self.assertTrue(torch.allclose(pure.b, sharded.b, atol=1e-6))


class TestContracts(unittest.TestCase):
    def test_normalization_names(self) -> None:
        self.assertEqual(normalization_names(), ("full", "none", "pre"))

    def test_rejects_unknown_normalization(self) -> None:
        weight, direction = _fixture()
        with self.assertRaises(ValueError):
            compute_directional_lora(
                weight,
                direction,
                strength=1.0,
                normalization="post",
                rank=1,
                seed=1,
            )

    def test_rejects_shape_mismatch(self) -> None:
        weight, _ = _fixture(d_out=24, d_in=16)
        with self.assertRaises(ValueError):
            compute_directional_lora(
                torch.randn(8),
                weight,
                strength=1.0,
                normalization="none",
                rank=1,
                seed=1,
            )

    def test_rejects_non_matrix_weight(self) -> None:
        with self.assertRaises(ValueError):
            compute_directional_lora(
                torch.randn(4, 4, 4),
                torch.randn(4),
                strength=1.0,
                normalization="none",
                rank=1,
                seed=1,
            )

    def test_rejects_nonpositive_rank(self) -> None:
        weight, direction = _fixture()
        with self.assertRaises(ValueError):
            compute_directional_lora(
                weight, direction, strength=1.0, normalization="none", rank=0, seed=1
            )

    def test_output_shapes(self) -> None:
        weight, direction = _fixture(d_out=24, d_in=16)

        # NONE and PRE are inherently rank-1: each is a single outer product, and
        # Heretic pins lora_rank = 1 for them. `rank` governs only the FULL
        # low-rank SVD approximation.
        for normalization in ("none", "pre"):
            factors = compute_directional_lora(
                weight,
                direction,
                strength=1.0,
                normalization=normalization,
                rank=3,
                seed=1,
            )
            self.assertEqual(factors.b.shape, (24, 1), normalization)
            self.assertEqual(factors.a.shape, (1, 16), normalization)

        full = compute_directional_lora(
            weight,
            direction,
            strength=1.0,
            normalization="full",
            rank=3,
            seed=1,
        )
        self.assertEqual(full.b.shape, (24, 3))
        self.assertEqual(full.a.shape, (3, 16))

    def test_rank_is_ignored_for_none_and_pre(self) -> None:
        """Pins preserved behaviour: only FULL consumes `rank`."""

        weight, direction = _fixture()
        for normalization in ("none", "pre"):
            rank_one = compute_directional_lora(
                weight,
                direction,
                strength=0.5,
                normalization=normalization,
                rank=1,
                seed=1,
            )
            rank_eight = compute_directional_lora(
                weight,
                direction,
                strength=0.5,
                normalization=normalization,
                rank=8,
                seed=1,
            )
            self.assertTrue(torch.equal(rank_one.a, rank_eight.a), normalization)
            self.assertTrue(torch.equal(rank_one.b, rank_eight.b), normalization)

    def test_accepts_low_precision_weight(self) -> None:
        """The vLLM backend dequantizes; the math must accept any float dtype."""

        weight, direction = _fixture()
        factors = compute_directional_lora(
            weight.to(torch.bfloat16),
            direction,
            strength=1.0,
            normalization="none",
            rank=1,
            seed=1,
        )
        self.assertEqual(factors.a.dtype, torch.float32)
        self.assertEqual(factors.b.dtype, torch.float32)


class TestLayerAblationWeight(unittest.TestCase):
    """The triangular schedule, shared by every backend."""

    def _schedule(self, layer_index: int) -> float | None:
        return layer_ablation_weight(
            layer_index,
            max_weight=1.0,
            max_weight_position=10.0,
            min_weight=0.25,
            min_weight_distance=2.0,
        )

    def test_peak_at_max_weight_position(self) -> None:
        self.assertAlmostEqual(self._schedule(10), 1.0)

    def test_falls_to_min_weight_at_the_edge(self) -> None:
        self.assertAlmostEqual(self._schedule(12), 0.25)
        self.assertAlmostEqual(self._schedule(8), 0.25)

    def test_interpolates_linearly(self) -> None:
        self.assertAlmostEqual(self._schedule(11), 0.625)
        self.assertAlmostEqual(self._schedule(9), 0.625)

    def test_untouched_beyond_the_distance(self) -> None:
        self.assertIsNone(self._schedule(13))
        self.assertIsNone(self._schedule(7))
        self.assertIsNone(self._schedule(0))
        self.assertIsNone(self._schedule(39))

    def test_matches_legacy_formula_in_bulk(self) -> None:
        max_weight, max_weight_position = 1.3, 17.5
        min_weight, min_weight_distance = 0.2, 6.0
        for layer_index in range(40):
            distance = abs(layer_index - max_weight_position)
            expected = (
                None
                if distance > min_weight_distance
                else max_weight
                + (distance / min_weight_distance) * (min_weight - max_weight)
            )
            actual = layer_ablation_weight(
                layer_index,
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=min_weight,
                min_weight_distance=min_weight_distance,
            )
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertAlmostEqual(actual, expected)

    def test_zero_distance_resolves_instead_of_nan(self) -> None:
        """The original divided 0/0 here and poisoned the whole delta with NaN."""

        self.assertEqual(
            layer_ablation_weight(
                5,
                max_weight=0.9,
                max_weight_position=5.0,
                min_weight=0.1,
                min_weight_distance=0.0,
            ),
            0.9,
        )
        self.assertIsNone(
            layer_ablation_weight(
                6,
                max_weight=0.9,
                max_weight_position=5.0,
                min_weight=0.1,
                min_weight_distance=0.0,
            )
        )

    def test_rejects_negative_distance(self) -> None:
        with self.assertRaises(ValueError):
            layer_ablation_weight(
                0,
                max_weight=1.0,
                max_weight_position=0.0,
                min_weight=0.0,
                min_weight_distance=-1.0,
            )


class TestSemantics(unittest.TestCase):
    """The factors must actually do what abliteration claims."""

    def test_none_removes_the_direction_component(self) -> None:
        weight, direction = _fixture(d_out=24, d_in=16)
        unit = F.normalize(direction, p=2, dim=0)
        factors = compute_directional_lora(
            weight,
            unit,
            strength=1.0,
            normalization="none",
            rank=1,
            seed=1,
        )
        ablated = weight.to(torch.float32) + factors.b @ factors.a
        projection = unit @ ablated
        self.assertLess(projection.abs().max().item(), 1e-5)

    def test_zero_strength_is_identity(self) -> None:
        weight, direction = _fixture()
        factors = compute_directional_lora(
            weight,
            direction,
            strength=0.0,
            normalization="none",
            rank=1,
            seed=1,
        )
        ablated = weight.to(torch.float32) + factors.b @ factors.a
        self.assertTrue(torch.allclose(ablated, weight.to(torch.float32), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
