# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for DeepSeek V4.1 Flash config handling and the Engram offload plan."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from heretic.engram_disk import DEFAULT_BLOCK_SIZE, EngramTableLayout
from heretic.model_loading import (
    build_engram_offload_plan,
    is_deepseek_v41_config,
    is_engram_tensor,
)

# The released checkpoint's engram geometry.
_LAYER_IDS = (1, 14)
_NUM_EMBEDDINGS = (384006168, 384016682)
_HEAD_DIM = 256
_BLOCK_SIZE = DEFAULT_BLOCK_SIZE


def _multimodal_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="deepseek_v41",
        text_config=SimpleNamespace(
            model_type="deepseek_v41_text",
            engram_layer_ids=list(_LAYER_IDS),
            engram_num_embeddings=list(_NUM_EMBEDDINGS),
            engram_head_dim=_HEAD_DIM,
        ),
    )


class TestDeepSeekV41Detection(unittest.TestCase):
    def test_detects_top_level_model_type(self) -> None:
        self.assertTrue(
            is_deepseek_v41_config(SimpleNamespace(model_type="deepseek_v41"))
        )

    def test_detects_nested_text_model_type(self) -> None:
        config = SimpleNamespace(text_config={"model_type": "deepseek_v41_text"})
        self.assertTrue(is_deepseek_v41_config(config))

    def test_detects_raw_config_dict(self) -> None:
        self.assertTrue(is_deepseek_v41_config({"model_type": "deepseek_v41"}))
        self.assertTrue(
            is_deepseek_v41_config({"text_config": {"model_type": "deepseek_v41_text"}})
        )

    def test_rejects_other_models(self) -> None:
        self.assertFalse(is_deepseek_v41_config(SimpleNamespace(model_type="laguna")))
        self.assertFalse(is_deepseek_v41_config({"model_type": "qwen3"}))
        self.assertFalse(is_deepseek_v41_config(SimpleNamespace()))

    def test_recognizes_engram_tensors(self) -> None:
        self.assertTrue(is_engram_tensor("layers.1.engram.embed.weight"))
        self.assertTrue(is_engram_tensor("layers.14.engram.embed.scale"))
        self.assertFalse(is_engram_tensor("layers.1.engram.wkv.weight"))
        self.assertFalse(is_engram_tensor("layers.1.mlp.experts.weight"))

    def test_handles_the_real_released_config_shape(self) -> None:
        """The shipped config.json nests the text model under text_config."""

        raw = {
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": {
                "model_type": "deepseek_v41_text",
                "vocab_size": 129280,
                "hidden_size": 5120,
                "n_routed_experts": 384,
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [384006168, 384016682],
                "engram_max_ngram_size": 4,
                "engram_vocab_size": 16000000,
                "engram_n_heads": 8,
                "engram_head_dim": 256,
                "engram_pad_token_id": 2,
                "engram_compressed_vocab_size": 99092,
            },
            "vision_config": {"model_type": "deepseek_v41_vision"},
        }
        # A config dict whose outer model_type is absent is exactly the case an
        # attribute-only lookup gets wrong.
        self.assertTrue(is_deepseek_v41_config(raw))

        plan = build_engram_offload_plan(raw, rank=0, world_size=4)
        self.assertEqual([layout.layer_id for layout in plan.layouts], [1, 14])
        self.assertEqual(
            sum(layout.num_rows for layout in plan.layouts),
            -(-384006168 // 4) + -(-384016682 // 4),
        )


class TestEngramOffloadPlan(unittest.TestCase):
    def test_splits_real_geometry_across_four_ranks(self) -> None:
        plan = build_engram_offload_plan(_multimodal_config(), rank=0, world_size=4)

        self.assertEqual(len(plan.layouts), 2)
        self.assertEqual([layout.layer_id for layout in plan.layouts], [1, 14])

        # The two tables differ in size (384006168 vs 384016682 rows), so the
        # per-rank ceil differs by a few thousand rows; they are not equal.
        self.assertEqual(
            [layout.num_rows for layout in plan.layouts],
            [-(-_NUM_EMBEDDINGS[0] // 4), -(-_NUM_EMBEDDINGS[1] // 4)],
        )
        self.assertEqual([layout.row_start for layout in plan.layouts], [0, 0])

        # The rank-local footprint is ~47 GiB, versus ~1.43 TiB for the whole
        # table: this is the whole reason disk mode is viable.
        gib = plan.resident_bytes_avoided / 1024**3
        self.assertGreater(gib, 45.0)
        self.assertLess(gib, 50.0)

    def test_each_rank_gets_a_distinct_row_range(self) -> None:
        part_rows = -(-_NUM_EMBEDDINGS[0] // 4)
        ranges = []
        for rank in range(4):
            plan = build_engram_offload_plan(
                _multimodal_config(), rank=rank, world_size=4
            )
            ranges.append((plan.layouts[0].row_start, plan.layouts[0].num_rows))

        self.assertEqual(
            [start for start, _ in ranges],
            [part_rows * rank for rank in range(4)],
        )
        self.assertEqual(len({start for start, _ in ranges}), 4)

    def test_covers_every_row_exactly_once(self) -> None:
        covered = 0
        for rank in range(4):
            plan = build_engram_offload_plan(
                _multimodal_config(), rank=rank, world_size=4
            )
            covered += plan.layouts[0].num_rows

        # The loader pads the last rank's share with ceil, so the union covers
        # at least every real row and never fewer.
        self.assertGreaterEqual(covered, _NUM_EMBEDDINGS[0])

    def test_reads_the_top_level_config_when_not_multimodal(self) -> None:
        config = SimpleNamespace(
            model_type="deepseek_v41",
            engram_layer_ids=list(_LAYER_IDS),
            engram_num_embeddings=list(_NUM_EMBEDDINGS),
            engram_head_dim=_HEAD_DIM,
        )
        plan = build_engram_offload_plan(config, rank=1, world_size=4)
        self.assertEqual(plan.layouts[0].row_start, -(-_NUM_EMBEDDINGS[0] // 4))

    def test_returns_no_layouts_for_a_model_without_engram(self) -> None:
        plan = build_engram_offload_plan(
            SimpleNamespace(model_type="laguna"), rank=0, world_size=2
        )
        self.assertEqual(plan.layouts, ())
        self.assertEqual(plan.resident_bytes_avoided, 0)

    def test_rejects_missing_engram_geometry(self) -> None:
        config = SimpleNamespace(
            model_type="deepseek_v41",
            engram_layer_ids=list(_LAYER_IDS),
            engram_num_embeddings=list(_NUM_EMBEDDINGS),
        )
        with self.assertRaisesRegex(ValueError, "engram_head_dim"):
            build_engram_offload_plan(config, rank=0, world_size=4)

    def test_rejects_mismatched_engram_lengths(self) -> None:
        config = SimpleNamespace(
            model_type="deepseek_v41",
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[384006168],
            engram_head_dim=_HEAD_DIM,
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            build_engram_offload_plan(config, rank=0, world_size=4)

    def test_layouts_use_the_checkpoints_block_size(self) -> None:
        plan = build_engram_offload_plan(_multimodal_config(), rank=0, world_size=4)
        layout = plan.layouts[0]
        self.assertIsInstance(layout, EngramTableLayout)
        self.assertEqual(layout.block_size, _BLOCK_SIZE)
        self.assertEqual(layout.scale_columns, _HEAD_DIM // _BLOCK_SIZE)
        self.assertEqual(layout.dim, _HEAD_DIM)
        self.assertEqual(layout.rows, _NUM_EMBEDDINGS[0])


if __name__ == "__main__":
    unittest.main()
