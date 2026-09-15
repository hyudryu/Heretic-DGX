# Widening the ablation past `attn.o_proj`

**Status: prepared and mechanically validated, NOT yet run.** Every claim below
was measured against the real checkpoint, read-only; nothing here has been
exercised end to end, and it cannot be without stopping the running study.

Read [`ABLITERATION_EFFECTIVENESS.md`](ABLITERATION_EFFECTIVENESS.md) first. The
short version: the attention-only search space is exhausted, ablation strength
has no measurable effect, and the best trial removes ~4% of refusals. V1 was
scoped to `attn.o_proj` alone by design; that scope is what needs to change.

---

## Why the attention output alone is not enough here

Over 65 scored trials, with `max_weight` pushed edge to edge across its whole
`[0.8, 1.5]` bound:

| parameter | correlation with refusals | range explored |
|---|---|---|
| `max_weight` | **0.028** | 0.80 – 1.50 (exhausted) |
| `min_weight_distance` | −0.471 | 1.70 – 23.34 |

Strength does nothing; coverage does something small. Heretic's own source names
the second component it expects (`main.py:778-783`), including a −0.25 lower
bound that lets the optimizer switch it off:

```python
max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
```

So the natural next component is the feed-forward down projection.

## The catch: V4.1 has no single down projection

```
  15360  layers.N.ffn.experts.E.w2.weight        (384 routed experts x 40 layers)
     40  layers.N.ffn.shared_experts.w2.weight   (one always-active dense path)
```

Ablating all routed experts would be, at rank 3:

```
  289,013,760 params  =  1.16 GB per trial adapter
```

and the shim rsyncs the adapter to three peers **on every trial**. That is
prohibitive at ~13 minutes per trial.

**The shared expert is the viable target**, and it is not a compromise — it is a
dense, always-active path rather than a routed one, so it is a plausible place
for refusal to be written.

## Mechanically validated (read-only, against the real checkpoint)

`scripts/check_shared_expert.py` confirms all four requirements:

| check | result |
|---|---|
| One per layer, discovery-exact | **40/40**, and the 6 `mtp.*` tensors are excluded by the same `layers.<int>.` rule that already excludes `mtp.N.attn.wo_b` |
| Uniform geometry | `[5120, 2304]` weight, `[160, 72]` block scale, all 40 layers |
| Output space | **5120 = hidden size**, i.e. the same residual stream the refusal direction is measured in — so **the same `d` applies to both components** |
| Block convention | 32×32, identical to `wo_b` |
| Dequantizes | finite, row norms 0.82–1.86, **0 zero rows** |
| Cost | **3.6 MB** per trial adapter (vs 2 MB today); **450 MiB** read (vs 1600 MiB) |

Nothing about the shape or the scaling convention differs from `wo_b` beyond the
dimension, so this is a name-and-shape change rather than new arithmetic.

## The exact change

### 1. Engine: add the module to the LoRA targets

vLLM renames the checkpoint's tensor in its own module tree
(`vllm/models/deepseek_v4_1/nvidia/model.py:931`):

```python
".shared_experts.w2": ".shared_experts.down_proj",
```

so the flag must use vLLM's name, **not** the checkpoint's:

```
--lora-target-modules wo_b shared_experts.down_proj     # was: wo_b
```

Everything else in the validated profile stays as it is
(`docs/VALIDATED_HERETIC_PROFILE.md`), including `--max-loras 1`, which is worth
revisiting if both components must be loaded as separate adapters — better to
put both in one adapter if vLLM allows it.

### 2. Target discovery: allow a second physical suffix

`src/heretic/deepseek_v41_targets.py` currently hard-codes one component:

```python
V41_COMPONENT = "attn.o_proj"
V41_PHYSICAL_SUFFIX = "attn.wo_b"
```

and `discover_targets()` selects on `_WEIGHT_SUFFIX` with the prefix check
`prefix != f"layers.{layer_text}"`. Because `layers.0.ffn.shared_experts.w2`
splits the same way — suffix `.ffn.shared_experts.w2.weight`, prefix `layers.0`
— this generalises by parameterising the physical suffix and returning a plan
per component, rather than by rewriting the loop.

`mlp.down_proj` should be the Heretic-facing component name, matching the name
Heretic's own search space already emits, so no change is needed in the
objective function.

### 3. Runtime: ablate both components

`abliteration_components = ["attn.o_proj", "mlp.down_proj"]` in
`config.dsv41.toml`. The V4.1 runtime must build LoRA factors for both. Since
both target projections have a 5120-row output and the direction is measured in
that same 5120 space, `compute_directional_lora` applies unchanged to either —
no new maths.

### 4. Adapter: one file, both modules

The exported PEFT adapter must carry both:
`base_model.model.layers.N.attn.wo_b.*` and
`base_model.model.layers.N.ffn.shared_experts.w2.*` (with the engine receiving
vLLM's name for the latter). Worth confirming vLLM accepts a single adapter
spanning both before committing 33 hours to it.

## Validation order before any long run

Each step is cheap and each one can fail independently, so none should be
skipped:

1. **Discovery against the real checkpoint** — run
   `scripts/check_shared_expert.py` as part of the preflight; it already passes.
2. **Load acceptance** — build one tiny adapter targeting
   `shared_experts.down_proj` and confirm `/v1/load_lora_adapter` returns 200.
   This is the single biggest unknown: if vLLM's target filter or its sharding
   for this module rejects it, nothing downstream matters.
3. **Apply check** — confirm the engine logs `_lora_shrink_kernel` /
   `_lora_expand_kernel` again, and that output changes versus the base model.
4. **Direction check** — confirm the exported adapter's dominant left singular
   vector aligns with the measured direction for the shared-expert layers, using
   `scripts/verify_adapter_direction.py`.
5. **Effect check** — `scripts/measure_refusal_rate.py`, base vs abliterated, on
   real prompts. This is the only measurement that decides anything.
6. **Only then** start a search.

## Cost of getting this wrong

The running study has ~28 hours left. Every trial costs KL 0.02–0.05 against a
baseline of exactly 0 and moves refusals by at most 4 points. A pause to run
steps 2–5 costs one in-flight trial, about 13 minutes, and the study resumes from
its Optuna journal (`checkpoint_action = "continue"`).

The watchdog (`scripts/heretic_watchdog.sh`) restarts the study whenever its
process is missing, so pausing means `sudo pkill -f heretic-watchdog` first and
restarting it afterwards.
