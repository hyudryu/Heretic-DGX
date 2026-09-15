#!/usr/bin/env python3
"""Build (and optionally deploy) the Heretic abliteration profile for V4.1 TP4.

WHY THIS IS (ALMOST) CONFIG-ONLY
--------------------------------
Heretic needs, per prompt, the residual stream entering every transformer
layer -- that is where refusal directions are measured, and where the
attention output projection (``wo_b``) writes.  The capture itself needs no
fork of vLLM, but it does need the one-line ``pinned-ehs1`` image patch (see
IMAGE below): without it the V2 model runner crashes in ``_dummy_run`` during
memory profiling because ``_mtp_hidden_buffer`` is only allocated for
eagle/draft-model methods, not for ``extract_hidden_states``.

  * ``--speculative-config`` method ``extract_hidden_states`` installs
    ``ExtractHiddenStatesProposer``, which requires ``num_speculative_tokens
    == 1`` and reads ``eagle_aux_hidden_state_layer_ids`` from the *draft*
    hf_config -- which SpeculativeConfig builds from the ``hf_config`` key of
    this very JSON (vllm/config/speculative.py:1202-1228).
  * ``--kv-transfer-config`` with ``ExampleHiddenStatesConnector`` is already
    registered in the v1 connector factory
    (kv_connector/factory.py:158-163), so it is selectable by name alone.
    It writes safetensors to disk and returns the path in
    ``kv_transfer_params["hidden_states_path"]``.

LAYER-ID SEMANTICS (v4.1 is the special case)
---------------------------------------------
vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:50-55:

    # v4.1 reads the attention *inputs* of its target layers, and
    # the target model captures the entry stream of layer L when
    # idx+1 == L, so the ids are used as-is. (v4's ids are in
    # capture-after semantics and keep the +1.)

The capture site (models/deepseek_v4_1/nvidia/model.py:637-645) runs after
``layer(idx)`` returns, so ``idx + 1 == L`` selects the stream *entering*
layer L -- i.e. Heretic's ``resid_pre``.  That is precisely the point at which
``layers.{L}.attn.wo_b`` deposits its output.

Ids are therefore ``1..40``:

    row k (0-based) == id (k+1) == entry stream of layer (k+1)

Mapping onto the 40 ablation targets:

    layer i  <-  row (i - 1)      for i in 1..39     (exact resid_pre)
    layer 0  <-  row 0            (proxy -- see caveat below)
    row 39   ==  id 40 == the final post-layer-39 stream; unused in V1

CAVEAT: layer 0's true resid_pre is the embedding output, which this capture
mechanism cannot express (id 0 never fires).  Row 0 is the closest available
substitute.  One shifted row out of 40 is well inside the noise of a mean
difference over hundreds of prompts -- but it is an approximation, not exact.

CAVEAT: the captured tensor is vLLM's *aux hidden state*: the hyper-connection
state collapsed with ``.mean(dim=1)``, not the mix-weighted
``hc_collapse_triton`` the real forward pass uses.  It is the engine's own
approximation of the layer-entry stream.  It is the only capture point
available without patching vLLM, and it is what EAGLE/DSpark drafters consume.

WHY max-model-len AND max-num-batched-tokens DROP
-------------------------------------------------
Hidden states are stored *in the KV cache* (``HiddenStateCacheSpec``), at
40 * 5120 * block_size * 2 bytes per block -- 52 MiB per block at block_size
128.  The default 300k context would demand an absurd KV pool.  Heretic only
ever sends short prompts, so the Heretic profile trades context for the
capture machinery.

WHY PREFIX CACHING IS OFF
-------------------------
A fully prefix-cached prompt performs no forward pass, so it produces no
hidden states -- it would silently yield an empty sample.  Determinism beats
the speedup for a data-collection profile.
"""

import json
import os
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sparkdeck  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
# pinned-ehs1 = pinned + one-line patch in models/deepseek_v4_1/nvidia/model.py:
# needs_mtp_hidden_states now also covers uses_extract_hidden_states(), because
# the V2 model runner slices get_mtp_target_hidden_states() in _dummy_run
# whenever a speculator exists, and it is None for this method (crash at
# profile_run: "TypeError: 'NoneType' object is not subscriptable").
# pinned-ehs2 = pinned-ehs1 + vl_model.py declares SupportsLoRA on
# DeepseekV41ForCausalLM (the multimodal wrapper; without it --enable-lora
# dies at worker startup with "does not support LoRA yet"). Adapter names
# resolve through the wrapper's existing hf_to_vllm_mapper
# (layers. -> language_model.model.layers.).
# pinned-ehs3-hccollapse = pinned-ehs2 + the hyper-connection collapse patch:
# the aux hidden state is collapsed with the carried pre-mix instead of a plain
# mean (patches/README.md). Heretic-only -- that collapse is wrong for the
# production DSpark drafter, which was trained against mean-collapsed states.
#
# Override with HERETIC_IMAGE when building the recipe for the patched profile.
IMAGE = os.environ.get("HERETIC_IMAGE", "vllm-dsv41:pinned-ehs2")
NODES = [
    "local",
    "fec2d563f9a1424fbb901e323e2fc0c0",
    "b3ae297c31414127b9cdeb21e05da9fa",
    "2cbb72e2b93b4b6da7f2bd54e5636c0b",
]

# Inside the container. Host side is
# /home/hyudryu/.cache/huggingface/heretic-hidden-states -- the only bind mount
# the deployment has. ~19 GB free on node 1, so the collector must stream.
SPOOL = "/root/.cache/huggingface/heretic-hidden-states"

LAYER_IDS = list(range(1, 41))

# ---------------------------------------------------------------- environment
# Copied verbatim from the live production deployment (56ed2c8f0567) so the
# Heretic profile keeps the tuned NCCL/fabric and Engram-on-disk settings.
ENVIRONMENT = {
    "VLLM_CACHE_ROOT": "/cache/clusterops-runtime/vllm",
    "TRITON_CACHE_DIR": "/cache/clusterops-runtime/triton",
    "TILELANG_CACHE_DIR": "/cache/clusterops-runtime/tilelang",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    "VLLM_USE_RUST_FRONTEND": "0",
    "VLLM_HAS_FLASHINFER_CUBIN": "1",
    "TORCH_CUDA_ARCH_LIST": "12.1a",
    "FLASHINFER_CUDA_ARCH_LIST": "12.1a",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "MAX_JOBS": "2",
    "FLASHINFER_NVCC_THREADS": "1",
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is deliberately ABSENT.
    # It is tuned for the serving profile, but vLLM rejects it alongside a KV
    # connector:
    #   ValidationError: 1 validation error for VllmConfig
    #     Value error, KV connector ExampleHiddenStatesConnector is incompatible
    #     with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless
    #     enable_cumem_allocator is also enabled. PyTorch's CUDA VMM allocator can
    #     remap KV cache virtual addresses to different physical pages,
    #     invalidating any pinned/registered KV memory.
    # Unsetting it is the smaller change than enabling the cumem allocator.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # The n-gram stays on the SSD, read-only. Same as production.
    "DSV41_ENGRAM_DISK": "1",
    "DSV41_ENGRAM_DISK_THREADS": "32",
    "DSV41_ENGRAM_DISK_CHUNK": "16",
    "DSV41_ENGRAM_DIR": "/root/.cache/huggingface/engram-local",
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": "rocep1s0f1",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_IGNORE_CPU_AFFINITY": "1",
    "NCCL_DEBUG": "WARN",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CROSS_NIC": "1",
    "NCCL_SOCKET_IFNAME": "enp1s0f1np1",
    "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
    "TP_SOCKET_IFNAME": "enp1s0f1np1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    # Without this, serve/lora/api_router.attach_router is a no-op and
    # /v1/load_lora_adapter returns 404 even with --enable-lora.
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
}

SPEC_CONFIG = {
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,  # asserted == 1 by ExtractHiddenStatesProposer
    "draft_sample_method": "greedy",
    "disable_padded_drafter_batch": False,
    # NOTE: the layer ids must NOT be a top-level key here. SpeculativeConfig is
    # a pydantic model, and `hf_config` is not one of its fields -- passing it at
    # the top level fails with:
    #   ValidationError: 1 validation error for SpeculativeConfig / hf_config
    #     Unexpected keyword argument
    # speculative.py:1213-1219 reads it out of `draft_model_config`, which is
    # declared `SkipValidation[ModelConfig]` and so lets a plain dict through
    # untouched.
    "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": LAYER_IDS}},
}

KV_TRANSFER_CONFIG = {
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "shared_storage_path": SPOOL,
        # Left false deliberately: true lets an API client write hidden states
        # to any path on the server filesystem.
        "allow_custom_save_path": False,
        "use_synchronization_lock": True,
        "num_writer_threads": 8,
    },
}


def extra_args() -> list[str]:
    return [
        "--served-model-name", "deepseek-v4.1-flash",
        "--trust-remote-code",
        "--tokenizer-mode", "deepseek_v41",
        "--block-size", "128",
        "--engram-config", json.dumps({"cpu_offload": False}, separators=(",", ":")),
        "--tool-call-parser", "deepseek_v41",
        "--enable-auto-tool-choice",
        "--reasoning-parser", "deepseek_v41",
        "--limit-mm-per-prompt", json.dumps({"image": 4}, separators=(",", ":")),
        "--mm-processor-cache-gb", "1",
        "--revision", REVISION,
        "--enable-prompt-tokens-details",
        "--max-logprobs", "-1",
        "--logprobs-mode", "raw_logits",
        # Heretic applies abliteration as a per-trial LoRA on attn.wo_b and
        # resets by unloading it. Adapter rank is 3 (row_normalization=full,
        # full_normalization_lora_rank=3); vLLM only accepts power-of-two-ish
        # ranks (1, 8, 16, ...), so 8 is the smallest valid ceiling.
        "--enable-lora",
        "--max-loras", "1",
        "--max-lora-rank", "8",
        # Wrap ONLY the abliteration target: without this the LoRA manager
        # wraps every LinearBase/MoERunner in the model (including the Marlin
        # MoE experts), which is untested surface we don't need.
        # Plain value, NOT JSON: the CLI parses '["wo_b"]' as a literal string
        # (verified in the engine's non-default-args log), which silently
        # matches nothing.
        "--lora-target-modules", "wo_b",
        # Must be EXPLICIT. This vLLM build defaults prefix caching ON, and a
        # fully cached prompt performs no forward pass -> no hidden states
        # (see module docstring). Omitting the flag silently breaks capture.
        "--no-enable-prefix-caching",
        "--compilation-config", json.dumps(
            {"cudagraph_mode": "FULL_AND_PIECEWISE",
             "cudagraph_capture_sizes": [5, 6, 10, 12, 15, 18, 20, 24, 25, 30, 35, 36, 40, 42, 48]},
            separators=(",", ":")),
        # KV pool reality on this build (measured 2026-09-14, util 0.85):
        # 21.0 GiB pool holds 1,511 tokens total (~14 MiB/token -- the packed
        # block-outermost pool strides every group by the widest page, the 52
        # MiB hidden-state page, so the small attention/indexer groups waste
        # most of each stride). 8192 needs 58.35 GiB and fails the startup
        # check; 1024 fits with ~1.5x concurrency. Do not raise this without
        # fixing the pool stride waste in vllm/v1/core/kv_cache_utils.py.
        "--max-model-len", "1024",
        "--max-num-seqs", "8",
        "--default-chat-template-kwargs", json.dumps({"thinking": False}, separators=(",", ":")),
        "-tp", "4",
        "--max-num-batched-tokens", "2048",
        "--speculative-config", json.dumps(SPEC_CONFIG, separators=(",", ":")),
        "--kv-transfer-config", json.dumps(KV_TRANSFER_CONFIG, separators=(",", ":")),
    ]


def recipe() -> dict:
    return {
        "name": "Heretic abliteration - DeepSeek V4.1 Flash TP4",
        "model": MODEL,
        "engine": "vllm",
        "image": IMAGE,
        "extra_args": extra_args(),
        "gpu_memory_utilization": 0.85,
        "gpu_memory_gb": None,
        "environment": ENVIRONMENT,
        "sg_tp_size": None,
        "sg_context_length": None,
        "sg_max_running_requests": None,
        "sg_mem_fraction": None,
        "sg_image": None,
        "deployment_mode": "sharded",
        "node_ids": NODES,
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "show"

    if mode == "show":
        print(json.dumps(recipe(), indent=2))
        return 0

    if mode == "create":
        res = sparkdeck.rpc("tools/call", {
            "name": "create_cluster_recipe",
            "arguments": {"recipe": recipe()},
        })
        if "error" in res:
            print("RPC ERROR:", json.dumps(res["error"], indent=2))
            return 1
        print(json.dumps(sparkdeck.unwrap(res["result"]), indent=2)[:1500])
        return 0

    if mode == "deploy":
        rid = sys.argv[2]
        res = sparkdeck.rpc("tools/call", {
            "name": "deploy_cluster_recipe",
            "arguments": {
                "recipe_id": rid,
                "deployment_name": "Heretic (V4.1 TP4 hidden-states capture)",
            },
        })
        if "error" in res:
            print("RPC ERROR:", json.dumps(res["error"], indent=2))
            return 1
        print(json.dumps(sparkdeck.unwrap(res["result"]), indent=2)[:2500])
        return 0

    print(f"unknown mode: {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
