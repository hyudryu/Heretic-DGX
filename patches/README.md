# The hyper-connection collapse patch

**Built and verified on all four nodes. Not deployed.** The running study still
uses `vllm-dsv41:pinned-ehs2`; the patched image sits beside it as
`vllm-dsv41:pinned-ehs3-hccollapse`.

> ## ⚠️ This image must never serve the production profile.
>
> The aux hidden states have **two** consumers, and the mean-collapse is correct
> for one of them.
>
> | consumer | what it does | effect of this patch |
> |---|---|---|
> | `extract_hidden_states` (Heretic) | `torch.stack(target_hidden_states, dim=1)`, then store — no learned projection (`v1/spec_decode/extract_hidden_states.py:127`) | **exactly what we want** |
> | EAGLE3 / DSpark drafter (production) | `main_norm(main_proj(aux_hidden_states))` (`models/deepseek_v4_1/nvidia/dspark.py:148-153`), called from `v1/worker/gpu/spec_decode/dflash/speculator.py:347` | **feeds the drafter out-of-distribution inputs** |
>
> The DSpark draft model was trained against **mean-collapsed** hidden states and
> applies a learned projection to them. `.mean(dim=1)` is therefore *correct for
> vLLM's purpose* — it is the drafter's input distribution, not a bug. It is only
> wrong for abliteration, which needs the residual stream the model actually
> consumes.
>
> Consequence: use `pinned-ehs3-hccollapse` **only** with
> `--speculative-config … extract_hidden_states`. Reusing it with the production
> `dspark` profile would silently degrade speculative decoding rather than fail
> loudly. The production profile must keep `vllm-dsv41:pinned`.

Read [`../docs/ABLITERATION_EFFECTIVENESS.md`](../docs/ABLITERATION_EFFECTIVENESS.md)
for why this was necessary.

---

## The change

One line, in the model's aux-hidden-state emission
(`vllm/models/deepseek_v4_1/nvidia/model.py`):

```diff
             if idx + 1 in self.aux_hidden_state_layers:
                 # Reconstruct the aux hidden state for draft models
                 aux_recon = mhc_post_tilelang(
                     hidden_states, residual, post_mix, res_mix
                 )
-                aux_hidden_state = aux_recon.mean(dim=1)
+                aux_hidden_state = hc_collapse_triton(aux_recon, pre_mix)
```

`hc_collapse_triton` is already imported by this module (line 25) and already
used for exactly this purpose at the model's final collapse (line 674):

```python
# Collapse the hc copies with the pre-mix from the last layer's FFN mixes
hidden_states = hc_collapse_triton(hidden_states, pre_mix)
```

## Why it is correct by construction

The capture site rebuilds a tensor the next layer rebuilds anyway. At the top of
the next iteration the layer does:

```python
residual = mhc_post_tilelang(x, residual, post_mix, res_mix)      # line 337
post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
    residual, self.hc_attn_fn, ..., pre_mix=pre_mix, ...)         # line 347
```

Line 337 is *the same expression* as the capture's `aux_recon`. So the only
difference between the captured value and the real layer input was the collapse:
a plain mean over the `hc_mult` copies, versus the learned mix the model
actually applies. `pre_mix` at the capture site is the mix the current layer
returned for the next layer's attention, which is precisely what that layer's
`mhc_pre_delayed_tilelang` consumes.

The result is the **un-normalized** collapsed stream, which is what
abliteration wants: the attention output is added to the residual, not to the
normalized input.

## The type contract holds

`hc_collapse_triton` asserts three things, and a mismatch would take the engine
down on the first request rather than degrade quietly — so the patch must
satisfy all three:

```python
assert x.ndim == 3 and x.dtype == torch.bfloat16
num_tokens, hc_mult, hidden_size = x.shape
assert pre_mix.shape == (num_tokens, hc_mult)
assert pre_mix.dtype == torch.float32
```

**`x` is `aux_recon`.** `mhc_post_tilelang` returns `torch.empty_like(residual)`,
so it carries the residual's dtype and shape exactly. `residual` is documented as
*"BF16 residual streams of shape (tokens, hc_mult, hidden_size)"* — so
`aux_recon` is bf16, 3-dimensional, `(tokens, 4, 5120)`. ✓

**`pre_mix` is the loop variable**, i.e. the layer's fourth return value
(`ffn_pre` from `mhc_pre_delayed_tilelang`), documented as *"the next FP32
pre-mix, with shapes … (tokens, hc_mult)"*. ✓

`hc_mult` extracted from `x.shape[1]` is therefore the same `hc_mult` the mix's
second dimension carries, so the shapes agree by construction rather than by
coincidence. `pre_mix` is never `None` at the capture site: the layer computes
`ffn_pre` on every path before returning it, including layer 0.

## Why it matters

The learned mixes are far from uniform
(`scripts/check_hc_mixes.py`): across 40 layers `std/|mean|` is 1.93 for
`hc_attn_base` and 0.67 for `hc_attn_scale`, and at layer 0 the per-copy biases
span 39 logit units (−25.49 … +13.44). Through a softmax that makes the real
collapse behave closer to *selecting one copy* than to averaging four, so a
direction measured on the mean can be close to orthogonal to the direction the
model reads.

That single mismatch explains the whole observed picture: refusal directions
that separate harmful from harmless prompts with leave-one-out accuracy 1.00, an
ablation the engine provably applies, and a refusal count that does not move.

## Building it

SparkDeck's `create_patched_image` cannot be used: it requires the base image
identity to match across selected nodes, and it does not. The four nodes carry
content-identical but independently built images — the same `model.py` hash
(`62a6e157…`) under four different image IDs — because each was imported at its
own timestamp, and `spark-node-4`'s is a differently layered 23.5 GB build.
Node 1's build succeeded while the others were refused with *"Base image runtime
configuration, filesystem, or platform differs from the first node"*.

So each node builds the same thin layer from **its own** base:

```sh
scp patches/model.py node:/tmp/dsv41_model_patched.py    # the patched source
BASE=vllm-dsv41:pinned-ehs2 NEW=vllm-dsv41:pinned-ehs3-hccollapse \
    scripts/build_patch_local.sh                          # run on each node
scripts/verify_patch_image.sh                             # run on each node
```

Verified on all four nodes: `model.py` sha256
`2369dc11fcd6863d3904b5ed2ae58eec234afb2c18121703d26f0f4654a470e3`, the patched
line present, the old assignment absent.

## Deploying it

**Prepared, not executed.** `scripts/deploy_hc_collapse_patch.sh` performs the
whole transition; read it before running it, because it stops a run the user
asked for.

Two profiles exist, differing **only** in the engine image — which is what makes
the same script the rollback path:

| recipe | image | role |
|---|---|---|
| `ff09e9a8` | `vllm-dsv41:pinned-ehs3-hccollapse` | patched (mix-weighted collapse) |
| `5ae70a64` | `vllm-dsv41:pinned-ehs2` | rollback (pre-patch, validated) |

Both carry the identical 49 `extra_args` and 34 environment variables, read from
the **live** deployment's own configuration rather than hand-transcribed — the
live profile gained its LoRA flags and `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
through deploy-time overrides, which a hand-written recipe would have missed.
`scripts/make_patched_recipe.py` regenerates either one and refuses to build if
the live profile is not on `extract_hidden_states`, or if `dspark` appears in
its spec — a guard against reusing this image where the mean-collapse is
required.

Deploy the patch:

```sh
scripts/deploy_hc_collapse_patch.sh
```

Roll back:

```sh
RECIPE=5ae70a64 NEW_IMAGE=vllm-dsv41:pinned-ehs2 scripts/deploy_hc_collapse_patch.sh
```

The script's ordering, and why:

1. `sudo pkill -f heretic-watchdog` — otherwise it restarts the study within 60s
2. stop the study (it has no supervisor other than the watchdog)
3. stop deployment `8fd087c422c5` — one 552B model, four Sparks; the old engine
   must go before the new one starts
4. deploy recipe `ff09e9a8`
5. wait for ready (the engine reloads 475 GB, ~10 minutes)
6. restart the shim — it hardcodes its upstream port at startup, so it must be
   pointed at the new rank-0 port
7. archive the old journal
8. start the fresh study
9. **verify** — `scripts/diag_direction.py` then `scripts/measure_refusal_rate.py`
10. restart the watchdog

### A fresh study is required, not a resume

The 69 trials in the existing journal were scored against directions computed
from the **mean-collapsed** capture. After the patch those directions no longer
describe what the engine emits, so `checkpoint_action = "continue"` would resume
a search whose history was measured in a different space. The script archives
the journal instead of continuing it.

### Do not commit to a long search until step 9 passes

A patch that loads is not a patch that works. The cheap check is whether the
refusal count actually moves; the current profile's best result is 94/100
against a baseline of 98/99, and a fix that is real should move that number
substantially rather than by four points.

### The pre-patch baseline is already saved

Once the patch is deployed the unpatched engine no longer exists, so the
mean-collapsed directions cannot be regenerated — and without them there is no
A/B, only a before/after refusal count that cannot say *whether the directions
changed*. That is the actual claim under test, so the baseline was captured
while it was still reproducible:

```
/home/hyudryu/.cache/huggingface/heretic-baseline-mean-collapse.pt    (834 KB)
/home/hyudryu/.cache/huggingface/heretic-baseline-mean-collapse.json
```

It holds a unit direction per entering layer (1…40) derived from the
mean-collapsed capture, plus `mean_norm`, `delta_norm` and `rel_sep` per layer,
and is produced by `scripts/save_baseline.py` from the captures written by
`scripts/diag_direction.py`.

After the patch, re-run `diag_direction.py` and compare the two direction sets —
a cosine well below 1 between the old and new directions *is* the confirmation,
independent of whether refusals move.
