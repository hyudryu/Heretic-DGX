# The hyper-connection collapse patch

**Built and verified on all four nodes. Not deployed.** The running study still
uses `vllm-dsv41:pinned-ehs2`; the patched image sits beside it as
`vllm-dsv41:pinned-ehs3-hccollapse`.

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

**This needs the study stopped**, because the change only takes effect when the
engine restarts:

1. `sudo pkill -f heretic-watchdog` — otherwise it restarts the study within 60s.
2. Stop deployment `8fd087c422c5` and stop the study process.
3. Update the deployment's `image` to `vllm-dsv41:pinned-ehs3-hccollapse`.
4. Start it; wait for ready.
5. Restart the shim, then the study (`checkpoint_action = "continue"` resumes
   from the Optuna journal, losing only the in-flight trial).
6. Restart the watchdog.
7. **Verify before trusting it.** Run `scripts/diag_direction.py` against the
   patched engine and compare the resulting directions with the mean-collapsed
   ones, then `scripts/measure_refusal_rate.py` for the only measurement that
   decides anything.

A fresh study is required rather than a resumed one: the old Optuna study's
trials were scored against directions that no longer describe the capture.
