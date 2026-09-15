# DeepSeek V4.1 Flash: the native vLLM backend

This is the architecture and status of Heretic's DeepSeek V4.1 support. It exists
because of a hard fact recorded in [`BLOCKERS.md`](BLOCKERS.md): `transformers`
implements no `deepseek_v41` architecture at any version, so the existing
Transformers route cannot load this model and never will without upstream work.

Rather than making DeepSeek V4.1 look like a Transformers model, the backend
removes Heretic's requirement that there *be* one.

---

## The boundary

`ModelRuntime` is now the real seam between Heretic's optimization logic and model
execution:

```
                 settings
                    |
            select_backend(settings)          <-- reads config.json ONLY
                    |
        +-----------+-----------+
        |                       |
  transformers            vllm_deepseek_v41
        |                       |
   Model(settings)      DeepSeekV41Runtime(settings)
        |                       |
  LocalModelRuntime      vLLM service (owns TP4, Engram, kernels)
        |                       |
        +-----------+-----------+
                    |
            ModelRuntime + RuntimeCapabilities
                    |
        Analyzer / Evaluator / Optuna / export
```

Backend selection happens **before** any `Model` is constructed, on both the
coordinator and worker paths. There is no path in which `Model(settings)` is
built first and then discarded.

Selection reads `config.json` directly for a local checkpoint, and
`PretrainedConfig.get_config_dict` for a Hub identifier -- deliberately **not**
`AutoConfig.from_pretrained`, which is the call that fails. This is verified:
selecting the backend for this checkpoint makes zero calls into Transformers.

## Division of labour

| Concern | Owner |
|---|---|
| Model loading, TP4/multi-node, prompt encoding, generation, hidden states, Engram, LoRA application | **vLLM** |
| Datasets, refusal directions, Optuna search, abliteration parameters, scorer orchestration, LoRA factor computation, export | **Heretic** |

Heretic is a single client. vLLM owns distributed execution, so this backend has
`distributed = False` and Heretic does not launch its own PyTorch ranks.

## The one abliteration target

`abliteration_components = ["attn.o_proj"]` keeps Heretic's semantic name and maps
it to DeepSeek V4.1's physical `attn.wo_b`. Measured from the released checkpoint:

| | |
|---|---|
| Physical tensor | `layers.{0..39}.attn.wo_b.weight` |
| Shape / dtype | `[5120, 8192]`, `F8_E4M3` |
| Scale tensor | `layers.{N}.attn.wo_b.scale`, `[160, 256]`, `F8_E8M0` (ue8m0) |
| Block geometry | 32 x 32 (`quantization_config.weight_block_size`) |
| Targets | exactly **40** |
| Total payload | **1.56 GiB** |

Everything else is out of scope for V1 and is never read or modified: routed
experts, shared experts, Engram, the vision encoder, the aligner, embeddings,
`lm_head`, MTP and DSpark. `mtp.{0,1,2}.attn.wo_b` also exists in the checkpoint
and is explicitly excluded.

## Residual capture point

DeepSeek V4.1 carries its residual stream as `hc_mult = 4` parallel copies
(Hyper-Connections). From the reference implementation's `Block.forward`:

```python
attn_pre, attn_post, attn_comb = self.hc_mixes(x, ...)
x = self.hc_pre(x, pre_mix)     # collapse [b, s, hc, d] -> [b, s, d]
x = self.attn(...)              # attention sees the COLLAPSED stream
x = self.hc_post(x, residual, attn_post, attn_comb)
```

and inside `Attention.forward`:

```python
x = self.wo_b(o.flatten(2))     # -> [b, s, dim]
```

**Captured:** the post-collapse, pre-attention hidden state -- the tensor
immediately after `hc_pre` and before the attention sublayer.

- **Pre- or post-collapse:** post-collapse. Not the `hc_mult`-wide stream.
- **Dimensionality:** `(batch, 5120)`, float32.
- **Why it matches `wo_b`:** `wo_b` is `RowParallelLinear(n_groups * o_lora_rank,
  dim)`, so it maps `8192 -> 5120` and writes into the collapsed stream.
  Abliteration subtracts a component from a projection's *output*, so the refusal
  direction must live in that same 5120-wide space. The runtime validates the
  captured width against `wo_b.out_features` and fails closed on a mismatch
  rather than ablating in the wrong basis.

### Indexing: no embedding row

Heretic's historical convention stores `layer_count + 1` residual rows, where row
0 is the embedding output and row `N + 1` is layer `N` -- an artefact of a
Transformers model exposing the embedding output as a hidden state. This backend
has no such tensor, and padding with a placeholder would put a fake vector in the
ablation path.

So the mapping is explicit and different: `residual_directions` is shaped
`(40, 5120)` and indexed **directly by layer**. Passing 41 rows is rejected with
a message naming the embedding-row convention. `get_layer_direction()` is the
single place this is resolved, and `direction_index` keeps Heretic's meaning as a
continuous position along the layer axis.

## Exact logits

The KL scorer needs a dense `(batch, vocab)` tensor of **raw** first-token logits.
The backend requires exactly that and **refuses to approximate**: a top-K
distribution and a processed (post-sampler) distribution are both rejected with
`ExactLogitsUnavailable`, because either would report a different quantity under
the same name. `KLDivergence` itself is unchanged.

The required deployment configuration is `max_logprobs=-1` with
`logprobs_mode="raw_logits"`.

## Trial cost

`reset_model()` deactivates the current adapter; `abliterate()` builds the next
one. The ~510 GB base checkpoint is never reloaded between trials. Factor
computation reads only the cached 40 target matrices.

---

## Status

Verified against the real checkpoint (node `gx10-node-1`, 2026-09-13):

| Gate | Result |
|---|---|
| 1. Read config, select backend, no weight load | **PASS** -- `vllm_deepseek_v41`, 0 Transformers calls |
| 2. Discover exactly 40 `wo_b` targets | **PASS** -- 40 found in 0.12 s, metadata only |
| 3. Cache and dequantize targets, verify dimensions | **PASS** -- `(5120, 8192)` fp32, finite, no zero rows, 3.7 s for all 40, 1.47 GiB peak RSS |
| 4-11 (generate, logits, residuals, LoRA apply/reset, one real trial) | **PASS** -- see 2026-09-14 update below |

Update 2026-09-14 (evening): the full loop has run end to end through
Heretic's own runtime. Deployment `1fddccd26a01` (image
`vllm-dsv41:pinned-ehs2`, api port 8014 on gx10-node-1) plus the HTTP shim
(`_tools/heretic_shim.py` in the Spark Management workspace, running on
gx10-node-1) completed real Optuna trials: per-trial adapter build, 4-node
adapter sync, `/v1/load_lora_adapter`, scorer generation and KLD with the
adapter active, then unload/reset. Early smoke trials (reduced prompt counts:
25/25 residual, 5/5 scorer) showed finite, small KL divergences
(0.038 -> 0.007 over the first completed trials) and ~45-60 s per trial.

Three integration fixes were required, all verified:

1. **Image `pinned-ehs2`**: `DeepseekV41ForCausalLM` (the multimodal wrapper in
   `vl_model.py`) now inherits `SupportsLoRA`. The inner LLM class had it; the
   wrapper the server instantiates did not, and `--enable-lora` died at worker
   startup. The wrapper's existing `hf_to_vllm_mapper` resolves adapter names
   (`base_model.model.layers.N...` -> `language_model.model.layers.N...`).
2. **Recipe**: `--enable-lora --max-loras 1 --max-lora-rank 8
   --lora-target-modules wo_b`, plus env `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`
   (without it the router is never attached and the endpoints 404), and
   `gpu_memory_utilization` 0.83 (startup free-memory check is tight).
   `--lora-target-modules` takes a plain value, not JSON.
3. **Heretic fixes** (this repo): `main.py` `run()` defaulted
   `runtime_factory=LocalModelRuntime`, which `create_runtime` rejects for this
   backend -- the CLI could never reach it (default is now `None`);
   `deepseek_v41_runtime.py` `_post` now tolerates the empty/plain-text 200
   bodies of the LoRA action endpoints and retries transient connection drops.

The shim's role: serves `POST /heretic/hidden_states` by driving a 1-token
chat completion, reading the connector's safetensors spool (as root; the
spool files are root-owned 0600), and deleting files after reading. It also
rewrites `lora_name` request keys to vLLM's `model`-field convention, unloads
a stale same-name adapter before each load, rsyncs adapters to the other
three nodes (no shared FS), and sweeps orphaned spool files (every request
produces one; only capture calls consume them).

`save_merged` refuses outright rather than emitting an unvalidated checkpoint.

### Why this matters for the report

What is established: Heretic can *reach* a V4.1 model without `transformers`,
the 40 physical targets are found and readable, the abliteration math is
backend-independent and proven identical to the pre-refactor implementation,
and (since 2026-09-14) the full loop runs against the live deployment --
dense raw logits serve, the hidden-state capture point answers through the
shim, and LoRA applies to and detaches from this multimodal checkpoint across
all four TP4 ranks. What remains open is *search quality*: whether the
captured residuals (with the documented layer-0 proxy and mean-collapse
approximations) yield good abliterations, which only the staged run answers.

## Configuration

```toml
model_backend = "auto"                     # auto | transformers | vllm_deepseek_v41

vllm_base_url = "http://127.0.0.1:8000"    # excluded from stored settings
vllm_model_name = "deepseek-v4.1-flash"
vllm_timeout_seconds = 600
vllm_lora_name = "heretic-trial"
vllm_lora_directory = ".heretic-v41-adapters"

abliteration_components = ["attn.o_proj"]
batch_size = 8                             # required: no local tokenizer to probe with
```

Connection and topology fields are `exclude=True` so reproduction metadata does
not leak machine layout.
