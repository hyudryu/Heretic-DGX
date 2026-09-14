# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backend selection, runtime capabilities, and the V4.1 runtime contract.

None of these need the 510 GB checkpoint, a GPU, or a live vLLM deployment: the
transport is a fake, and the target plan is fabricated with the same geometry
invariants the real one has.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from heretic.backend import read_raw_model_config, select_backend
from heretic.config import BackendMethod
from heretic.deepseek_v41_runtime import (
    BACKEND_NAME,
    DeepSeekV41Runtime,
    ExactLogitsUnavailable,
    GenerationResult,
)
from heretic.deepseek_v41_targets import (
    EXPECTED_V41_LAYER_COUNT,
    SafetensorsEntry,
    TargetTensor,
    V41TargetPlan,
)
from heretic.model_loading import is_deepseek_v41_config
from heretic.runtime import LocalModelRuntime
from heretic.utils import Prompt

D_OUT = 8
D_IN = 16


def _entry(shape: tuple[int, ...], size: int) -> SafetensorsEntry:
    return SafetensorsEntry(dtype="F8_E4M3", shape=shape, begin=0, end=size)


def _fabricated_plan(layer_count: int = EXPECTED_V41_LAYER_COUNT) -> V41TargetPlan:
    """A plan with the real plan's invariants but toy dimensions."""

    targets = tuple(
        TargetTensor(
            layer_index=layer,
            component="attn.o_proj",
            weight_name=f"layers.{layer}.attn.wo_b.weight",
            scale_name=f"layers.{layer}.attn.wo_b.scale",
            weight_shard="model-00001-of-00001.safetensors",
            scale_shard="model-00001-of-00001.safetensors",
            weight=_entry((D_OUT, D_IN), D_OUT * D_IN),
            scale=_entry((1, 1), 1),
            block_rows=32,
            block_cols=32,
        )
        for layer in range(layer_count)
    )
    return V41TargetPlan(checkpoint_directory=Path("/nonexistent"), targets=targets)


def _settings(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "model": "/models/DeepSeek-V4.1-Flash",
        "model_commit": None,
        "model_backend": BackendMethod.AUTO,
        "row_normalization": SimpleNamespace(value="none"),
        "full_normalization_lora_rank": 3,
        "seed": 1234,
        "batch_size": 2,
        "max_response_length": 8,
        "response_prefix": None,
        "offload_outputs_to_cpu": False,
        "vllm_checkpoint_directory": None,
        "vllm_lora_name": "heretic-trial",
        "vllm_lora_directory": "/tmp/heretic-v41-test-adapters",
        "vllm_base_url": "http://127.0.0.1:8000",
        "vllm_model_name": "deepseek-v4.1-flash",
        "vllm_api_key": None,
        "vllm_timeout_seconds": 5,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeTransport:
    """Records the LoRA lifecycle and serves canned responses."""

    def __init__(self, *, vocab: int = 6) -> None:
        self.vocab = vocab
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.generation_lora_names: list[str | None] = []
        self.logits_lora_names: list[str | None] = []
        self.hidden_lora_names: list[str | None] = []
        self.return_raw_logits = True

    def _active(self) -> str | None:
        return self.loaded[-1] if self.loaded else None

    def generate(self, prompts, *, max_tokens, temperature, lora_name):
        self.generation_lora_names.append(lora_name)
        return [GenerationResult(text="ok", token_ids=(1,)) for _ in prompts]

    def raw_logits(self, prompts, *, lora_name):
        self.logits_lora_names.append(lora_name)
        if not self.return_raw_logits:
            return torch.zeros(len(prompts), self.vocab)
        return torch.randn(len(prompts), self.vocab)

    def hidden_states(self, prompts, *, layer_ids, lora_name):
        self.hidden_lora_names.append(lora_name)
        return torch.randn(len(prompts), len(layer_ids), D_OUT)

    def load_lora(self, name, path):
        self.loaded.append(name)

    def unload_lora(self, name):
        self.unloaded.append(name)
        if name in self.loaded:
            self.loaded.remove(name)


class TestBackendSelection(unittest.TestCase):
    def _write_config(self, root: Path, config: dict) -> None:
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_local_deepseek_v41_config_selects_vllm_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_config(root, {"model_type": "deepseek_v41"})
            self.assertEqual(
                select_backend(_settings(model=str(root))),
                BackendMethod.VLLM_DEEPSEEK_V41,
            )

    def test_multimodal_nested_text_config_selects_vllm_backend(self) -> None:
        """The released checkpoint is multimodal: the type lives in text_config."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_config(
                root,
                {
                    "model_type": "deepseek_v41",
                    "vision_config": {"model_type": "deepseek_vit"},
                    "text_config": {"model_type": "deepseek_v41_text"},
                },
            )
            self.assertEqual(
                select_backend(_settings(model=str(root))),
                BackendMethod.VLLM_DEEPSEEK_V41,
            )

    def test_ordinary_config_selects_transformers_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_config(root, {"model_type": "llama"})
            self.assertEqual(
                select_backend(_settings(model=str(root))),
                BackendMethod.TRANSFORMERS,
            )

    def test_explicit_backend_overrides_detection(self) -> None:
        """An explicit choice must not read the config at all."""

        settings = _settings(
            model="/does/not/exist",
            model_backend=BackendMethod.TRANSFORMERS,
        )
        self.assertEqual(select_backend(settings), BackendMethod.TRANSFORMERS)

    def test_detection_never_instantiates_the_architecture(self) -> None:
        """Selecting V4.1 must not go anywhere near AutoModelForCausalLM."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_config(root, {"model_type": "deepseek_v41"})

            with (
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    side_effect=AssertionError("AutoModel must not be used"),
                ) as forbidden_model,
                patch(
                    "transformers.AutoConfig.from_pretrained",
                    side_effect=AssertionError("AutoConfig must not be used"),
                ) as forbidden_config,
                patch(
                    "heretic.model.Model",
                    side_effect=AssertionError("Model must not be constructed"),
                ) as forbidden_constructor,
            ):
                self.assertEqual(
                    select_backend(_settings(model=str(root))),
                    BackendMethod.VLLM_DEEPSEEK_V41,
                )

            forbidden_model.assert_not_called()
            forbidden_config.assert_not_called()
            forbidden_constructor.assert_not_called()

    def test_local_read_does_not_touch_transformers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_config(root, {"model_type": "deepseek_v41"})
            with patch(
                "transformers.PretrainedConfig.get_config_dict",
                side_effect=AssertionError("local reads must not use transformers"),
            ) as forbidden:
                raw = read_raw_model_config(str(root))
            forbidden.assert_not_called()
            self.assertTrue(is_deepseek_v41_config(raw))

    def test_missing_config_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                read_raw_model_config(temporary)

    def test_v41_detection_is_not_fooled_by_a_lookalike(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_config(root, {"model_type": "deepseek_v4"})
            self.assertEqual(
                select_backend(_settings(model=str(root))),
                BackendMethod.TRANSFORMERS,
            )


class TestCapabilities(unittest.TestCase):
    def test_v41_layer_count_is_forty(self) -> None:
        runtime = DeepSeekV41Runtime(
            _settings(), transport=FakeTransport(), plan=_fabricated_plan()
        )
        self.assertEqual(runtime.capabilities.layer_count, 40)

    def test_v41_supports_only_attn_o_proj(self) -> None:
        runtime = DeepSeekV41Runtime(
            _settings(), transport=FakeTransport(), plan=_fabricated_plan()
        )
        self.assertEqual(runtime.capabilities.abliterable_components, ("attn.o_proj",))

    def test_v41_reports_its_backend_and_export_limits(self) -> None:
        runtime = DeepSeekV41Runtime(
            _settings(), transport=FakeTransport(), plan=_fabricated_plan()
        )
        capabilities = runtime.capabilities
        self.assertEqual(capabilities.backend_name, BACKEND_NAME)
        self.assertFalse(capabilities.distributed)
        self.assertTrue(capabilities.supports_exact_logits)
        self.assertTrue(capabilities.supports_adapter_export)
        # V1 does not claim a quantization-preserving standalone export.
        self.assertFalse(capabilities.supports_merged_export)

    def test_transformers_runtime_reports_its_own_capabilities(self) -> None:
        model = MagicMock()
        model.get_layers.return_value = [object()] * 48
        model.get_abliterable_components.return_value = [
            "attn.o_proj",
            "mlp.down_proj",
        ]
        model.distributed = False

        runtime = LocalModelRuntime(model)
        capabilities = runtime.capabilities
        self.assertEqual(capabilities.backend_name, "transformers")
        self.assertEqual(capabilities.layer_count, 48)
        self.assertEqual(
            capabilities.abliterable_components, ("attn.o_proj", "mlp.down_proj")
        )
        self.assertTrue(capabilities.supports_merged_export)

    def test_capabilities_do_not_call_get_layers_repeatedly(self) -> None:
        """Layer count must be readable without walking the model each time."""

        model = MagicMock()
        model.get_layers.return_value = [object()] * 3
        model.get_abliterable_components.return_value = ["attn.o_proj"]
        model.distributed = False
        runtime = LocalModelRuntime(model)
        self.assertEqual(runtime.capabilities.layer_count, 3)


class TestResidualMapping(unittest.TestCase):
    """`attn.o_proj` is resolved explicitly; there is no embedding row."""

    def setUp(self) -> None:
        self.runtime = DeepSeekV41Runtime(
            _settings(), transport=FakeTransport(), plan=_fabricated_plan()
        )

    def test_per_layer_indexes_directly(self) -> None:
        directions = torch.randn(40, D_OUT)
        for layer_index in (0, 7, 39):
            resolved = self.runtime.get_layer_direction(directions, layer_index, None)
            self.assertTrue(torch.equal(resolved, directions[layer_index]))

    def test_rejects_the_transformers_embedding_row_convention(self) -> None:
        """41 rows would mean someone padded with a placeholder embedding."""

        with self.assertRaises(ValueError) as caught:
            self.runtime.get_layer_direction(torch.randn(41, D_OUT), 0, None)
        self.assertIn("no embedding row", str(caught.exception))

    def test_rejects_wrong_width(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.runtime.get_layer_direction(torch.randn(40, D_OUT + 1), 0, None)
        self.assertIn("writes into a space of width", str(caught.exception))

    def test_rejects_non_matrix_directions(self) -> None:
        with self.assertRaises(ValueError):
            self.runtime.get_layer_direction(torch.randn(40), 0, None)

    def test_interpolation_is_normalized_and_continuous(self) -> None:
        directions = torch.randn(40, D_OUT)
        midpoint = self.runtime.get_layer_direction(directions, 0, 12.5)
        self.assertAlmostEqual(float(midpoint.norm()), 1.0, places=5)
        expected = torch.nn.functional.normalize(
            directions[12].lerp(directions[13], 0.5), p=2, dim=0
        )
        self.assertTrue(torch.allclose(midpoint, expected, atol=1e-6))

    def test_integer_direction_index_hits_that_layer(self) -> None:
        directions = torch.randn(40, D_OUT)
        resolved = self.runtime.get_layer_direction(directions, 0, 5.0)
        expected = torch.nn.functional.normalize(directions[5], p=2, dim=0)
        self.assertTrue(torch.allclose(resolved, expected, atol=1e-6))

    def test_direction_index_range_is_enforced(self) -> None:
        directions = torch.randn(40, D_OUT)
        with self.assertRaises(ValueError):
            self.runtime.get_layer_direction(directions, 0, 40.0)
        with self.assertRaises(ValueError):
            self.runtime.get_layer_direction(directions, 0, -0.5)


class TestResidualShape(unittest.TestCase):
    def test_residuals_have_one_row_per_layer_and_wo_b_width(self) -> None:
        transport = FakeTransport()
        runtime = DeepSeekV41Runtime(
            _settings(), transport=transport, plan=_fabricated_plan()
        )
        residuals = runtime.get_residuals([Prompt(system="s", user="u")])
        self.assertEqual(tuple(residuals.shape), (1, 40, D_OUT))
        self.assertEqual(residuals.dtype, torch.float32)

    def test_capture_width_mismatch_fails_closed(self) -> None:
        class BadWidth(FakeTransport):
            def hidden_states(self, prompts, *, layer_ids, lora_name):
                return torch.randn(len(prompts), len(layer_ids), D_OUT + 3)

        runtime = DeepSeekV41Runtime(
            _settings(), transport=BadWidth(), plan=_fabricated_plan()
        )
        with self.assertRaises(RuntimeError) as caught:
            runtime.get_residuals([Prompt(system="s", user="u")])
        self.assertIn("capture point is wrong", str(caught.exception))


class TestExactLogits(unittest.TestCase):
    def test_logits_are_dense_full_vocabulary(self) -> None:
        transport = FakeTransport(vocab=11)
        runtime = DeepSeekV41Runtime(
            _settings(), transport=transport, plan=_fabricated_plan()
        )
        logits = runtime.get_logits([Prompt(system="s", user="u")])
        self.assertEqual(tuple(logits.shape), (1, 11))
        self.assertEqual(logits.dtype, torch.float32)

    def test_top_k_only_payload_is_rejected(self) -> None:
        """No silent fallback to a top-K KL divergence."""

        from heretic.deepseek_v41_runtime import HttpVllmTransport

        http = HttpVllmTransport("http://127.0.0.1:8000", "m")
        body = {
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "tokens": ["a"],
                        "token_logprobs": [-0.5],
                        "top_logprobs": [{"a": -0.5, "b": -1.0}],
                    },
                }
            ]
        }
        with self.assertRaises(ExactLogitsUnavailable) as caught:
            http._reconstruct_logits(body, 1)
        self.assertIn("top-K", str(caught.exception))

    def test_dense_payload_is_accepted_and_ordered(self) -> None:
        from heretic.deepseek_v41_runtime import HttpVllmTransport

        http = HttpVllmTransport("http://127.0.0.1:8000", "m")
        body = {
            "vocab_size": 3,
            "choices": [
                {"index": 0, "logprobs": {"raw_logits": [1.0, 2.0, 3.0]}},
                {"index": 1, "logprobs": {"raw_logits": [4.0, 5.0, 6.0]}},
            ],
        }
        logits = http._reconstruct_logits(body, 2)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.equal(logits[1], torch.tensor([4.0, 5.0, 6.0])))

    def test_vocabulary_size_mismatch_is_rejected(self) -> None:
        from heretic.deepseek_v41_runtime import HttpVllmTransport

        http = HttpVllmTransport("http://127.0.0.1:8000", "m")
        body = {
            "vocab_size": 5,
            "choices": [{"index": 0, "logprobs": {"raw_logits": [1.0, 2.0]}}],
        }
        with self.assertRaises(ExactLogitsUnavailable):
            http._reconstruct_logits(body, 1)


class TestAdapterLifecycle(unittest.TestCase):
    """baseline -> apply -> reset -> baseline, across two trials."""

    def setUp(self) -> None:
        self.transport = FakeTransport()
        self.settings = _settings()
        self.runtime = DeepSeekV41Runtime(
            self.settings, transport=self.transport, plan=_fabricated_plan()
        )
        self.prompts = [Prompt(system="s", user="u")]

    def _abliterate(self, *, max_weight: float = 1.0) -> None:
        from heretic.model import AbliterationParameters

        noise = torch.randn(D_OUT, D_IN)
        with patch(
            "heretic.deepseek_v41_runtime.TargetCache.read_dequantized",
            return_value=noise,
        ):
            self.runtime.abliterate(
                torch.randn(40, D_OUT),
                None,
                {
                    "attn.o_proj": AbliterationParameters(
                        max_weight=max_weight,
                        max_weight_position=20.0,
                        min_weight=0.0,
                        min_weight_distance=20.0,
                    )
                },
            )

    def test_baseline_uses_no_adapter(self) -> None:
        self.runtime.get_responses(self.prompts)
        self.assertEqual(self.transport.generation_lora_names, [None])

    def test_apply_then_reset_restores_baseline(self) -> None:
        self.runtime.get_responses(self.prompts)
        self._abliterate()
        self.runtime.get_responses(self.prompts)
        self.runtime.reset_model()
        self.runtime.get_responses(self.prompts)

        self.assertEqual(
            self.transport.generation_lora_names,
            [None, "heretic-trial", None],
        )
        self.assertEqual(self.transport.loaded, [])
        self.assertEqual(self.transport.unloaded, ["heretic-trial"])

    def test_second_trial_does_not_inherit_the_first(self) -> None:
        """Each abliterate() must fully replace the previous adapter."""

        self._abliterate(max_weight=1.0)
        self._abliterate(max_weight=0.5)

        # The first adapter is unloaded before the second is loaded, so nothing
        # from trial 1 can leak into trial 2.
        self.assertEqual(len(self.transport.loaded), 1)
        self.assertEqual(self.transport.unloaded, ["heretic-trial"])

    def test_reset_is_idempotent(self) -> None:
        self._abliterate()
        self.runtime.reset_model()
        self.runtime.reset_model()
        self.assertEqual(self.transport.unloaded, ["heretic-trial"])

    def test_logits_follow_the_adapter_too(self) -> None:
        self._abliterate()
        self.runtime.get_logits(self.prompts)
        self.assertEqual(self.transport.logits_lora_names, ["heretic-trial"])

    def test_unsupported_component_is_rejected(self) -> None:
        from heretic.model import AbliterationParameters

        parameters = {
            "mlp.down_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=20.0,
                min_weight=0.0,
                min_weight_distance=20.0,
            )
        }
        with self.assertRaises(ValueError) as caught:
            self.runtime.abliterate(torch.randn(40, D_OUT), None, parameters)
        self.assertIn("attn.o_proj", str(caught.exception))

    def test_shutdown_deactivates_the_adapter(self) -> None:
        self._abliterate()
        self.runtime.shutdown()
        self.assertEqual(self.transport.unloaded, ["heretic-trial"])


class TestPromptRendering(unittest.TestCase):
    def test_no_chat_template_is_used(self) -> None:
        """The release ships no Jinja template; the native encoder owns format."""

        transport = FakeTransport()
        captured: list[list[str]] = []

        class Capturing(FakeTransport):
            def generate(self, prompts, *, max_tokens, temperature, lora_name):
                captured.append(list(prompts))
                return super().generate(
                    prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    lora_name=lora_name,
                )

        runtime = DeepSeekV41Runtime(
            _settings(), transport=Capturing(), plan=_fabricated_plan()
        )
        runtime.get_responses([Prompt(system="You are helpful.", user="What is 1+1?")])
        self.assertEqual(captured, [["You are helpful.\n\nWhat is 1+1?"]])
        del transport

    def test_response_prefix_is_preserved(self) -> None:
        captured: list[list[str]] = []

        class Capturing(FakeTransport):
            def generate(self, prompts, *, max_tokens, temperature, lora_name):
                captured.append(list(prompts))
                return super().generate(
                    prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    lora_name=lora_name,
                )

        runtime = DeepSeekV41Runtime(
            _settings(response_prefix="Answer: "),
            transport=Capturing(),
            plan=_fabricated_plan(),
        )
        runtime.get_responses([Prompt(system="s", user="u")])
        self.assertEqual(captured, [["s\n\nuAnswer: "]])


if __name__ == "__main__":
    unittest.main()
