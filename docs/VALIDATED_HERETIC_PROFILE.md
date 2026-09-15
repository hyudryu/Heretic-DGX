# The validated Heretic profile — DeepSeek V4.1 Flash, TP4

**Verified live on 2026-09-14.** This is the configuration that actually runs a
Heretic abliteration study against the real 552B model across all four Sparks.
It supersedes every earlier attempt in this repository; three earlier profiles
crashed at engine startup and are documented at the bottom so nobody rebuilds
them.

Read this with [`HIDDEN_STATE_CAPTURE.md`](HIDDEN_STATE_CAPTURE.md), which
explains *why* the capture config looks the way it does.

---

## Live state at time of writing

| | |
|---|---|
| Deployment | `8fd087c422c5` — *Heretic (V4.1 TP4 hs-capture, ehs2 ctx1024 lora wo_b, rt-lora)* |
| Status | `ready` / `running`, all four nodes |
| Engine | `vllm-dsv41:pinned-ehs2`, API on rank 0 port **8014** |
| Heretic shim | pid on **127.0.0.1:8765** → forwards to `127.0.0.1:8014` |
| Study | `heretic --model /models/DeepSeek-V4.1-Flash --config ./config.dsv41.toml` |
| Progress | trial **65 of 200**; last scored trial: Keywords 98/100, KL divergence **0.0196** |
| Budget | elapsed 4h18m, estimated remaining ~29h |
| Hidden-state spool | `~/.cache/huggingface/heretic-hidden-states`, growing continuously |

The router on **7878** currently points at this deployment, so
`http://100.97.4.16:7878/v1` serves the Heretic profile *with a trial adapter
loaded*. See "Known tradeoff" below.

---

## Engine configuration

Image `vllm-dsv41:pinned-ehs2`, `gpu_memory_utilization = 0.83`.

```
--served-model-name deepseek-v4.1-flash
--trust-remote-code
--tokenizer-mode deepseek_v41
--block-size 128
--engram-config {"cpu_offload":false}
--tool-call-parser deepseek_v41
--enable-auto-tool-choice
--reasoning-parser deepseek_v41
--limit-mm-per-prompt {"image":4}
--mm-processor-cache-gb 1
--revision dba1be0a40aa45a94ad051997016db3960a90277
--enable-prompt-tokens-details
--max-logprobs -1
--logprobs-mode raw_logits
--enable-lora
--max-loras 1
--max-lora-rank 8
--lora-target-modules wo_b
--no-enable-prefix-caching
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]}
--max-model-len 1024
--max-num-seqs 8
--default-chat-template-kwargs {"thinking":false}
-tp 4
--max-num-batched-tokens 2048
--speculative-config {"method":"extract_hidden_states","num_speculative_tokens":1,"draft_sample_method":"greedy","disable_padded_drafter_batch":false,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[1,2,...,40]}}}
--kv-transfer-config {"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/root/.cache/huggingface/heretic-hidden-states","allow_custom_save_path":false,"use_synchronization_lock":true,"num_writer_threads":8}}
--distributed-executor-backend mp --nnodes 4 --tensor-parallel-size 4 --pipeline-parallel-size 1
```

### The four flags that make or break it

| Flag | Why |
|---|---|
| `--enable-lora --max-loras 1 --max-lora-rank 8 --lora-target-modules wo_b` | Without these the V4.1 multimodal wrapper has no LoRA surface at all, and `/v1/load_lora_adapter` 404s. `wo_b` targets the 40 attention output projections — the V1 abliteration target. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` | Lets a *running* engine swap adapters. This is the whole basis of the Optuna loop: every trial generates a fresh adapter and loads it, no restart. The deployment name records it as `rt-lora`. |
| `--speculative-config … extract_hidden_states` | The only way to get the residual stream out of the engine. Forces `num_speculative_tokens = 1`, so **speculative decoding is off** on this profile. |
| `--no-enable-prefix-caching` | A fully prefix-cached prompt performs no forward pass and produces **no hidden states** — it would silently yield an empty sample. |

### Environment

Identical to production except for two deliberate differences:

- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` — **added** (see above).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — **removed**. vLLM rejects
  it alongside any KV connector:

  ```
  Value error, KV connector ExampleHiddenStatesConnector is incompatible with
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless
  enable_cumem_allocator is also enabled.
  ```

Everything else is carried over: the NCCL/fabric settings
(`NCCL_IB_HCA=rocep1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1`, …), and the
Engram-on-disk settings (`DSV41_ENGRAM_DISK=1`,
`DSV41_ENGRAM_DIR=/root/.cache/huggingface/engram-local`). The n-gram stays on
the SSD and is only ever read.

---

## Study configuration (`/opt/heretic-dgx/config.dsv41.toml`)

```toml
model_backend = "vllm_deepseek_v41"
vllm_base_url = "http://127.0.0.1:8765"
vllm_model_name = "deepseek-v4.1-flash"
vllm_lora_name = "heretic-trial"
vllm_lora_directory = "/home/hyudryu/.cache/huggingface/heretic-lora"

abliteration_components = ["attn.o_proj"]
batch_size = 8
n_trials = 200
n_startup_trials = 60
checkpoint_action = "continue"

[good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:400]"
[bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:400]"
```

`n_trials`/`n_startup_trials` are locked into the Optuna study on first run, so
the 200-trial ceiling is fixed; evaluate at ~100 and resume for the rest.
`checkpoint_action = "continue"` means a restart resumes rather than restarts.

---

## Known tradeoff: the public router serves the abliterated model

The SparkDeck router on 7878 resolves to whatever deployment is running, and
right now that is the Heretic one — with `heretic-trial` loaded as an adapter.
So `http://100.97.4.16:7878/v1` returns the output of a *trial* abliteration,
with `max_model_len` 1024 and speculative decoding disabled.

This is unavoidable while a study runs, because there is one set of four Sparks
and one 552B model: the Heretic profile and the serving profile cannot coexist.
It is worth being explicit about because it degrades interactive use badly
during a 33-hour search. **Weights are safe** — every Heretic path is read-only
against the checkpoint — the cost is availability and output quality, not
correctness.

---

## Superseded attempts (do not rebuild these)

All three crashed at engine startup, before any weights loaded. Each failure is
recorded because each was non-obvious.

| Attempt | Failure |
|---|---|
| `a909bc9f2b52` | `hf_config` passed as a top-level key of `--speculative-config` → `ValidationError: Unexpected keyword argument`. It must nest under `draft_model_config`. |
| `26de9cd3e588` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` inherited from production, rejected with a KV connector. |
| `be707e8875f9` | Correct config, but superseded — the nodes were already occupied by the running profile. |

All three are stopped and left in place rather than deleted, so the failure
history stays inspectable.

---

## Reproducing

`_tools/heretic_profile.py` in the Spark Management workspace builds the recipe
and deploys it:

```sh
python heretic_profile.py show      # full recipe JSON
python heretic_profile.py create    # -> recipe id
python heretic_profile.py deploy <recipe_id>
```

Generated deployments carry `managed_by: sparkdeck-mcp`, so start/stop does not
require `allow_unowned`. The older recipe `5ae70a64` is the ancestor of the live
profile; the live one additionally overrides the image to `pinned-ehs2` and adds
the LoRA flags and `VLLM_ALLOW_RUNTIME_LORA_UPDATING`.
