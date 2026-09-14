# Why Heretic cannot currently run DeepSeek V4.1 Flash

This documents a **hard blocker**, verified empirically on the real nodes on
2026-09-13. It exists so nobody re-derives it, and so the Engram work already in
this repository is not mistaken for something that works end to end.

Read this before `docs/TP4.md`. The runbook is still correct as a runbook; it
just cannot execute until blocker 1 is resolved.

---

## Summary

| # | Blocker | Whose problem | Fixable here? |
|---|---|---|---|
| 1 | `transformers` has no `deepseek_v41` implementation — at any version | upstream | **No** |
| 2 | `transformers` has no Engram support at all | upstream | No |
| 3 | Heretic's Engram disk offload is defined but never wired into the load path | **this repo** | Yes, but pointless until 1 is fixed |
| 4 | The standalone export is hard-coded to Laguna S 2.1 FP8 (48 layers) | **this repo** | Yes, but pointless until 1 is fixed |
| 5 | The checkpoint ships no chat template | upstream | Workaroundable |

Blockers 1 and 2 are fatal and outside this repository's control. Heretic loads
models exclusively through `transformers` (`AutoConfig`,
`AutoModelForCausalLM` / `AutoModelForImageTextToText`, `AutoTokenizer`,
`generate()`), so with no implementation of the architecture there is no load
path at all.

---

## Blocker 1 (fatal): no `transformers` implementation of `deepseek_v41`

The checkpoint declares:

```json
{
  "architectures": ["DeepseekV41ForCausalLM"],
  "model_type": "deepseek_v41",
  "text_config": { "model_type": "deepseek_v41_text", ... },
  "transformers_version": "5.6.0"
}
```

There is **no `auto_map`**, and no `modeling_*.py` / `configuration_*.py` in the
checkpoint, so `trust_remote_code=True` cannot rescue it. The only Python shipped
is `inference/`, which is a standalone reference implementation using its own
`ModelArgs` and `generate.py` — not a Hugging Face module.

Measured on the nodes:

| Probe | Result |
|---|---|
| Pinned transformers (`git@d65f37c`, reports `5.16.0.dev0`) — files containing `deepseek_v41` | **0** |
| Latest released transformers from PyPI (**5.17.0**) — files containing `deepseek_v41` | **0** |
| `transformers` `main` — `src/transformers/models/deepseek_v41` | **HTTP 404** |
| Installed DeepSeek model modules | `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4`, `deepseek_ocr2`, `deepseek_vl`, `deepseek_vl_hybrid` — **no v41** |
| `AutoConfig.from_pretrained(snapshot, trust_remote_code=True)` | `ValueError: The checkpoint you are trying to load has model type 'deepseek_v41' but Transformers does not recognize this architecture.` |
| Same, with latest 5.17.0 in a clean venv | identical failure |

Note that `deepseek_v4` **does** exist in transformers, and V4.1 Flash is
described as a successor to V4-Flash. They are not interchangeable: V4.1 adds the
Causal Encoder-Decoder split, CSA2 attention, mHC, and Engram, and it registers
under a different `model_type`. Renaming the config's `model_type` to
`deepseek_v4` would load *something*, but it would be the wrong architecture with
missing weights — silently wrong, which is worse than failing.

### Confirmed through Heretic's own code path

Running the real entry point on `gx10-node-1`:

```sh
/opt/heretic-dgx/.venv/bin/heretic --model /models/DeepSeek-V4.1-Flash
```

```
* Trying dtype bfloat16...
* Failed: The checkpoint you are trying to load has model type `deepseek_v41`
  but Transformers does not recognize this architecture.
* Trying dtype float32...
* Failed: (same)
...
  File "/opt/heretic-dgx/src/heretic/model.py", line 193, in __init__
    raise Exception("Failed to load model with all configured dtypes.")
Exception: Failed to load model with all configured dtypes.
```

It fails at `Model(settings)`, before any GPU work, for every configured dtype.

---

## Blocker 2: no Engram support in `transformers`

`engram` appears **0 times** (case-insensitive) anywhere in transformers 5.17.0,
and 0 times in the pinned dev build.

So even if a `deepseek_v41` architecture landed tomorrow, the Engram tables would
be ordinary `nn.Parameter`s and the 189.1 GiB would be allocated like any other
weight. Any working disk-offload design has to be carried by this repository, not
by upstream.

---

## Blocker 3: Heretic's Engram offload is not wired in

This is our own gap, and it is the reason `engram_disk = true` currently does
nothing.

`src/heretic/engram_disk.py` defines `DiskEngramTable`, `EngramDiskConfig`,
`from_rank_environment`, and `gather_dequant_many`. `model_loading.py` defines
`is_deepseek_v41_config`, `is_engram_tensor`, and `build_engram_offload_plan`.
All of them are referenced **only by `tests/`** — there is not a single call site
in `src/heretic/` outside their own definitions.

`model.py` contains no reference to Engram at all. Its load path is a direct:

```python
self.model = get_model_class(settings.model).from_pretrained(
    settings.model,
    **build_model_load_kwargs(..., distributed=self.distributed, ...),
)
```

and `build_model_load_kwargs` with `distributed=True` sets `tp_plan="auto"` and
nothing else. No tensor is excluded, no placeholder is substituted, no forward
hook is installed.

**Correcting an earlier claim in this repository.** `docs/TP4.md` section 7 said
the rank logs report a `Engram table DISK-backed: ...` line and that seeing it
means the tables are not resident. That line comes from
`EngramOffloadPlan.describe()`, which nothing calls, so it can never be printed.
The section has been corrected.

---

## Blocker 4: the standalone export is hard-coded to Laguna S 2.1 FP8

`standalone_export.save_runtime_as_standalone` is written for one specific model:

```python
def save_runtime_as_standalone(
    runtime, *, source_directory, destination_directory, max_shard_size,
    expected_identity: CheckpointIdentity = LAGUNA_S_2_1_FP8_IDENTITY,
    expected_layer_count: int = 48,
) -> StandaloneVerification:
```

It calls `verify_checkpoint_identity(source_directory, expected_identity)`, then
`export_standalone_laguna(...)` and `verify_standalone_laguna(...)`. Separately,
`main.preflight_distributed_export` requires `abliteration_components ==
["attn.o_proj"]` and verifies the same Laguna identity.

DeepSeek V4.1 Flash has **40 layers**, not 48, and a different checkpoint
identity, so the export would be rejected. This needs a DeepSeek V4.1 variant of
the exporter (which also has to preserve the Engram tensors untouched).

---

## Blocker 5: no chat template

`tokenizer_config.json` has no `chat_template` key, and the model card states the
release "does not include a Jinja-format chat template" — prompt encoding is
provided as a Python reference in `encoding/encoding.py`.

Heretic's `Model.generate()` calls `tokenizer.apply_chat_template(...)`. A
template would have to be supplied (or the prompt path reworked to use
`encoding.py`). This is the smallest of the five blockers.

---

## How this model *is* run

Two working routes exist, and neither is compatible with Heretic:

1. **The checkpoint's own reference implementation.** `inference/convert.py`
   converts the HF checkpoint into one file per tensor-parallel rank, then
   `generate.py` runs under `torchrun` (single or multi-node). Its own README
   calls it "a readable reference implementation rather than a production
   serving engine."
2. **vLLM**, with its own DeepSeek V4.1 implementation plus an Engram-on-disk
   patch. This is what the working DGX Spark deployment uses.

Heretic needs an in-process `nn.Module` it can run `generate()`,
`output_hidden_states=True`, `output_logits=True`, and PEFT LoRA targeting on. A
serving engine cannot supply that over HTTP.

---

## What would unblock it

In rough order of effort:

1. **Upstream `transformers` support for `deepseek_v41`.** Nothing to do but
   wait, and it does not exist on `main` today. Even then, blockers 2-4 remain.
2. **A `modeling_deepseek_v41.py` written for transformers.** This means
   porting the reference implementation into the HF API: the 20+20 CED split,
   CSA2 with its three attention modes and hierarchical sparse indexer, FP4 main
   KV cache, single-pass mHC, DSpark, the DeepSeek-ViT encoder and aligner, the
   fp8/fp4 MoE kernels — plus a `tp_plan` covering every tensor, and it must
   expose hidden states and logits and accept PEFT LoRA on `attn.o_proj` /
   `mlp.down_proj`. This is a substantial project, not a patch, and it must be
   numerically faithful or the abliteration directions and KLD numbers are
   meaningless.
3. **Decouple Heretic from transformers.** Abliteration itself only needs the
   base weights, the residual stream, and the ability to apply a low-rank
   update. One could compute directions and fit adapters offline against the
   reference implementation (or vLLM), then evaluate candidates with vLLM. That
   is a rearchitecture of Heretic's Optuna loop, and the resulting adapters
   would have to be validated against a real load anyway.

Blockers 3 and 4 are worth fixing regardless, because they are small and they are
prerequisites for any of the above. But neither is worth starting while blocker 1
stands, since nothing can be tested end to end.

---

## What has actually been verified working

- The four-node SSH mesh (fabric and Tailscale).
- `uv sync --frozen` on all four nodes: torch 2.13.0+cu130, CUDA available,
  device `NVIDIA GB10`, capability sm_121.
- Preflight identity agreement across all four ranks on a synthetic checkpoint.
- The cluster config loader accepting a 4-node TOML (`world_size: 4`).
- The hard-link checkpoint farm at `/models/DeepSeek-V4.1-Flash` on all four
  nodes: 55 hard links + 4 copied directories, 0 symlinks, 48 shards, zero
  additional bytes.

The Engram reader and the N-rank generalization remain covered by unit tests
only. They have not executed against real weights.
