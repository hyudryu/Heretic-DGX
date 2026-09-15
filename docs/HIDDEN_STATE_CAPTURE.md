# Hidden-state capture on the vLLM V4.1 backend

**Status: verified end to end on 2026-09-14, and now consumed by Heretic
itself.** Current deployment `1fddccd26a01` (image `vllm-dsv41:pinned-ehs2` =
`pinned-ehs1` plus `SupportsLoRA` on the multimodal wrapper; see
DEEPSEEK_V41_BACKEND.md), recipe `5ae70a64`, api port 8014 on gx10-node-1.
Heretic reaches this capture through `_tools/heretic_shim.py` (Spark
Management workspace), an HTTP shim on gx10-node-1 that serves
`POST /heretic/hidden_states` by issuing a 1-token chat completion, reading
the spooled safetensors (as root; spool files are root-owned 0600), returning
the last-token row per requested layer, and deleting the file; a sweeper
thread reaps orphaned spool files (every engine request writes one, only
capture calls consume them) because node 1 has only ~18 GB free.

Original verification: deployment
`8b033da86f5b` ("Heretic (V4.1 TP4 hidden-states capture, ehs1 ctx1024)",
recipe `5ae70a64`, image `vllm-dsv41:pinned-ehs1`, api port 8008 on
gx10-node-1) boots all four TP4 ranks and serves. A chat completion returned
`kv_transfer_params.hidden_states_path`, and the safetensors on gx10-node-1
loaded with `load_hidden_states()` as `hidden_states [T, 40, 5120] bf16` +
`token_ids [T]`, all finite.

**One source change is required — `pinned-ehs1` is NOT config-only.** The
`extract_hidden_states` method never allocates `_mtp_hidden_buffer` in
`models/deepseek_v4_1/nvidia/model.py` (`use_eagle() or uses_draft_model()` is
False for it), but the V2 model runner slices that buffer unconditionally in
`_dummy_run` when a speculator exists, so every rank died in `profile_run`
with `TypeError: 'NoneType' object is not subscriptable`. The patch extends
the predicate with `spec_config.uses_extract_hidden_states()`. Built via
SparkDeck `create_patched_image` (build `71960de1b6fa`), base
`vllm-dsv41:pinned`.

**Two config traps, both now encoded in the recipe:**

1. Prefix caching defaults **ON** in this build. The original recipe omitted
   the flag and got `enable_prefix_caching=True` — a fully cached prompt
   performs no forward pass and yields no hidden states. The recipe now passes
   `--no-enable-prefix-caching` explicitly.
2. The KV pool cannot hold long contexts. At `--max-model-len 8192` the
   startup check demanded **58.35 GiB** against ~14.7 GiB available. At
   `--max-model-len 1024` with `gpu_memory_utilization 0.85` the pool is
   21.0 GiB for **1,511 tokens** (~14 MiB/token; ~1.5x concurrency at 1024).
   The hidden-state page is 52.4 MiB per 128-token block, but the packed
   block-outermost pool strides every cache group by the widest page, so the
   small MLA/indexer groups waste most of each stride — that is where the
   ~14 MiB/token goes. Raising context means fixing the stride waste in
   `vllm/v1/core/kv_cache_utils.py`, not turning a knob.
   `gpu_memory_utilization` must stay ≤ ~0.86: only 104.7/121.63 GiB is free
   at engine start (0.88 was rejected by the startup check).

Everything below is the original mechanism write-up. Where it says the profile
uses `--max-model-len 8192`, read **1024**; the per-block math is unchanged.

This is the thing that unblocks gates 6 and 7. It is written down because
every fact below was paid for with a probe against the real image, and several
of them are counter-intuitive.

---

## Why we need it

Abliteration needs the **residual stream entering every transformer layer**.
That is where the refusal direction is measured, and it is exactly the point at
which `layers.{i}.attn.wo_b` deposits its output. Heretic's own name for it is
`resid_pre`.

vLLM owns the forward pass, so the residual stream has to come back out of the
engine. It does — but only through a debug path that is easy to miss.

---

## The mechanism

Two independent engine features combine:

### 1. `extract_hidden_states` speculative method

`SpeculativeConfig` reads its layer ids out of **`draft_model_config`**
(`vllm/config/speculative.py:1213-1228`):

```python
elif self.method == "extract_hidden_states":
    ...
    if hasattr(self.draft_model_config, "hf_config"):
        hf_config = self.draft_model_config.hf_config.to_dict()
    elif isinstance(self.draft_model_config, dict) and "hf_config" in self.draft_model_config:
        hf_config = self.draft_model_config["hf_config"]
    else:
        hf_config = {}

    self.draft_model_config = copy.copy(self.target_model_config)
    self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
        self.draft_model_config.hf_config, **hf_config
    )
```

So `eagle_aux_hidden_state_layer_ids` — which the capture path requires — is
set purely from the command line. No draft checkpoint on disk, no modified
`config.json`, no `create_patched_image`.

**The nesting is not optional, and getting it wrong is a trap.** Placing
`hf_config` at the top level of `--speculative-config` fails at config
validation:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
hf_config
  Unexpected keyword argument
```

`SpeculativeConfig` is a pydantic model and `hf_config` is not one of its
fields. `draft_model_config` *is* a field, declared
`SkipValidation[ModelConfig] = None` — the `SkipValidation` is what lets a plain
dict through untouched. So it must be:

```json
{"draft_model_config": {"hf_config": {...}}}
```

**It requires `num_speculative_tokens == 1`** — hard `assert` at
`vllm/v1/spec_decode/extract_hidden_states.py:35`. DSpark runs 5, so the
Heretic profile trades speculative decoding speed for visibility. That is fine:
this profile is for data collection, not serving.

### 2. `ExampleHiddenStatesConnector`

Registered by name in the v1 connector factory
(`distributed/kv_transfer/kv_connector/factory.py:158-163`), so it is selectable
from `--kv-transfer-config` with no source change. It reads the aux hidden
states out of the KV cache and writes them to safetensors.

Honest description from its own docstring: *"Simple debug implementation of a
HiddenStatesConnector."* It is a debug path that upstream happens to have
merged, not a supported product feature. That is a risk to note, not to hide.

---

## The exact configuration

```json
{
  "method": "extract_hidden_states",
  "num_speculative_tokens": 1,
  "draft_sample_method": "greedy",
  "disable_padded_drafter_batch": false,
  "draft_model_config": {
    "hf_config": { "eagle_aux_hidden_state_layer_ids": [1, 2, "...", 40] }
  }
}
```

```json
{
  "kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "shared_storage_path": "/root/.cache/huggingface/heretic-hidden-states",
    "allow_custom_save_path": false,
    "use_synchronization_lock": true,
    "num_writer_threads": 8
  }
}
```

### Connector extra-config keys

All read via `get_from_extra_config(key, default)`:

| Key | Default | Meaning |
|---|---|---|
| `shared_storage_path` | `/tmp` | Directory for `{req_id}.safetensors` |
| `num_writer_threads` | `8` | Thread pool size for disk writes |
| `allow_custom_save_path` | `false` | **Leave false.** See below |
| `use_synchronization_lock` | `true` | `flock` protocol so readers never see a partial file |

`allow_custom_save_path: true` lets any API client choose the output path, i.e.
write anywhere the server can write. The connector logs its own warning about
it. It stays **false**.

---

## Layer-id semantics — V4.1 is the special case

This is the part that is easy to get wrong, and vLLM documents it in a comment
(`v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:50-55`):

```python
if getattr(hf_config, "model_type", None) == "deepseek_v41":
    # v4.1 reads the attention *inputs* of its target layers, and
    # the target model captures the entry stream of layer L when
    # idx+1 == L, so the ids are used as-is. (v4's ids are in
    # capture-after semantics and keep the +1.)
    layer_ids = list(dspark_layer_ids)
else:
    layer_ids = [i + 1 for i in dspark_layer_ids]
```

The capture site confirms it
(`models/deepseek_v4_1/nvidia/model.py:622-645`) — it runs **after**
`layer(idx)` returns, so `idx + 1 == L` selects the stream entering layer `L`:

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer):
    hidden_states, residual, post_mix, res_mix, pre_mix = layer(...)
    if idx + 1 in self.aux_hidden_state_layers:
        aux_recon = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        aux_hidden_state = aux_recon.mean(dim=1)
        ...
        aux_hidden_states.append(aux_hidden_state)
```

So:

> **row `k` (0-based) == id `k+1` == the stream entering layer `k+1`** — i.e.
> Heretic's `resid_pre` for layer `k+1`. Width `hidden_size` = **5120**.

There is **no off-by-one to correct** for V4.1. `[1..40]` is the right request.

### Mapping onto the 40 ablation targets

| Target | Source row | Fidelity |
|---|---|---|
| `layers.i.attn.wo_b`, `i` in `1..39` | `row (i - 1)` | **exact** `resid_pre` |
| `layers.0.attn.wo_b` | `row 0` | proxy — see caveat |
| — | `row 39` (id 40) | the final post-layer-39 stream; unused in V1 |

Everything is inside one helper so the mapping has exactly one definition.

**Caveat (layer 0).** Layer 0's true `resid_pre` is the embedding output. This
capture mechanism cannot express it — id 0 would mean `idx == -1`, which never
matches. `row 0` (the entry stream of layer 1, i.e. layer 0's own output) is the
closest available substitute. One shifted row out of 40 sits well inside the
noise of a mean difference over hundreds of prompts, but it is an
approximation, not an identity.

**Caveat (collapse).** The captured tensor is vLLM's *aux hidden state*: the
hyper-connection state collapsed with `.mean(dim=1)` over the `hc_mult=4` copies.
The real forward pass collapses with the mix-weighted `hc_collapse_triton`.
The mean-collapse is the engine's own approximation of the layer-entry stream —
it is what EAGLE/DSpark drafters consume. It is the only capture point
available without patching vLLM. Recorded as a known V1 approximation.

---

## Why the Heretic profile lowers context and batch size

Hidden states are stored **in the KV cache**, via a `HiddenStateCacheSpec` group
— that is the whole design, and it has a brutal per-block cost:

```
num_layers * hidden_size * block_size * dtype_bytes
   40      *    5120     *    128      *      2      = 52.4 MiB per block
```

Against a default `--max-model-len 300000`, that is an absurd KV pool. The
Heretic profile therefore sets:

| Flag | Production | Heretic | Reason |
|---|---|---|---|
| `--max-model-len` | `300000` | `8192` | KV pool above |
| `--max-num-batched-tokens` | `8192` | `2048` | On-device capture buffer is `(max_num_batched_tokens + max_num_seqs) * 40 * 5120 * 2` bytes ≈ 842 MiB at 2048, ≈ 3.4 GiB at 8192 |
| `--enable-prefix-caching` | on | **off** | A fully prefix-cached prompt performs no forward pass and so produces **no hidden states** — it would silently yield an empty sample |

---

## Reading the output

The connector returns the path per request:

```python
return True, {"hidden_states_path": filename}
```

Default filename is `{shared_storage_path}/{request_id}.safetensors`.

Reading it is a one-liner the connector itself exports — and it is
**race-free**, because the writer holds an exclusive `flock` on a companion
`.lock` file and the reader takes a shared lock:

```python
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    load_hidden_states,
)
data = load_hidden_states(path)   # {"hidden_states": [T, 40, 5120], "token_ids": [T]}
```

Two details that matter:

- **Only TP rank 0 writes** (`self._is_tp_rank_zero`, set in
  `register_kv_caches`). Other ranks are no-ops. On this cluster rank 0 is
  `gx10-node-1`, so the spool is written there and nowhere else.
- `request_finished` returns `True` to **delay block freeing** until the
  extraction completes, so blocks are held slightly longer than normal.

`kv_transfer_params["include_output_tokens"]` (default `false`) controls
whether generated tokens are included. Note the deliberate off-by-one in the
code: with output tokens included, the **final** token is dropped, because the
model's last output token was never an input to any forward pass and so has no
hidden state in the cache.

---

## Storage budget

Per token per request, hidden states cost:

```
40 layers * 5120 hidden * 2 bytes (bf16) = 400 KiB
```

So a 200-token prompt is ≈ 82 MB and 800 prompts would be ≈ 64 GB. That does
not fit anywhere convenient:

| Node | Path | Free |
|---|---|---|
| gx10-node-1 (rank 0, the writer) | `/` | **19 GB** |
| gx10-node-2 | `/` | 72 GB |
| gx10-node-3 | `/` | 125 GB |
| spark-node-4 | `/` | 1.5 TB |

The container has exactly **one bind mount**:
`/home/hyudryu/.cache/huggingface` → `/root/.cache/huggingface`. So the only
writable location the engine can see is the 19 GB on node 1. (`/home/hyudryu/t7`
is a second 932 GB SSD with 244 GB free, but it is **not** mounted into the
container, and the container's mount has `rprivate` propagation, so a host-side
bind mount inside it would not appear. Adding it requires a `runtime_file_mounts`
entry, which SparkDeck's config API does not currently expose.)

**Consequence: the collector must stream.** Accumulate per-layer sums and counts
for a chunk of prompts, then delete the safetensors before fetching the next
chunk. Peak footprint is then `chunk_size * prompt_bytes`, which fits comfortably
in 19 GB for chunk sizes in the low hundreds. A per-layer mean is all Heretic
needs, so a streaming accumulation is exactly equivalent to holding everything.

---

## Verified facts worth not re-deriving

- `DeepseekV41ForCausalLM` explicitly declares `SupportsEagle3`
  (`models/deepseek_v4_1/nvidia/vl_model.py:120`), and `DeepseekV4Model` inherits
  `EagleModelMixin` (`.../nvidia/model.py:388`). This matters because
  `_setup_eagle3_aux_hidden_state_outputs` **hard-fails** otherwise:
  `RuntimeError: Model does not support EAGLE3 interface but
  aux_hidden_state_outputs was requested`.
- `extract_hidden_states` turns `use_aux_hidden_state_outputs` on
  (`v1/worker/gpu_model_runner.py:696-700`).
- `eagle_aux_hidden_state_layer_ids` must be **non-empty**, or the proposer
  raises `ValueError` (`v1/spec_decode/extract_hidden_states.py:65-70`).
- `disable_padded_drafter_batch` must be false for this method.
- The engine echoes both configs in its `non-default args` startup log line, so
  confirming they were accepted takes one `docker logs | grep`.

---

## Reproducing the profile

`_tools/heretic_profile.py` in the Spark Management workspace builds and deploys
this as a SparkDeck recipe, separately from the production deployment, so the
serving config is never mutated:

```sh
python heretic_profile.py show      # the full recipe JSON
python heretic_profile.py create    # -> recipe id
python heretic_profile.py deploy <recipe_id>
```

The companion deployment keeps `managed_by: sparkdeck-mcp`, so start/stop does
not require `allow_unowned`.
