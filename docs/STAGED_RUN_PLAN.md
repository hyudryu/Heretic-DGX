# Staged run plan: 5 → 50 → 100 → 200 trials

A staged rollout for DeepSeek V4.1 Flash on four DGX Sparks. Start with a
5-trial smoke test, confirm the whole loop works, then scale up only if it does.

> **BLOCKED — do not start stage 1.** Stage 0 has been run against the real
> checkpoint and it **fails**. `transformers` implements no `deepseek_v41`
> architecture at any version (pinned commit, latest release 5.17.0, and `main`
> all lack it), and Heretic loads models only through `transformers`. Details and
> evidence: [`BLOCKERS.md`](BLOCKERS.md).
>
> Everything below remains the correct plan for *once that is resolved*. It is
> not runnable today, and stage 0 is no longer a question this document needs to
> ask.

Before you read the stages, read [§2](#2-the-two-things-that-make-this-different-from-a-normal-run).
Two behaviours of the cluster path dictate the shape of this plan, and the
obvious approach does not work.

---

## 1. What each stage is for

| Stage | Trials | Goal | Timeout | Go/no-go |
|---|---|---|---|---|
| 0 | 0 | Does the model load at all? | 1 h | Must load, or stop and fix that first |
| 1 | 5 | Does the full loop run end to end? | 6 h | 5 trials recorded, KLD sane, no OOM |
| 2 | 50 | Is the search finding good trials? | 72 h | Pareto front improving vs stage 1 |
| 3 | 100 | Does more search help? | 120 h | Best KLD still improving |
| 4 | 200 | Final run | 336 h | — |

Stage 0 is not optional. Nothing below it has ever been run against this model
on real hardware, and a load failure is the single most likely outcome. See
[§6](#6-stage-0-load-check-do-this-first).

---

## 2. The two things that make this different from a normal run

### 2a. You cannot scale up by editing `n_trials` between runs

Resuming a study **replaces your settings with the ones stored in the study**:

```python
if action == "continue":
    settings = Settings.model_validate_json(existing_study.user_attrs["settings"])
```

The tool says so plainly: *"You can continue the previous run from where it
stopped. This will override any specified settings."*

So this does **not** work:

```sh
# Run 1: n_trials = 5     -> 5 trials
# Run 2: edit to n_trials = 50, resume
#   -> stored settings still say 5, so it runs 5 - 5 = 0 more trials
```

There is normally an interactive **"Run additional trials"** option that adds
trials properly (`settings.n_trials = len(study.trials) + n_additional_trials`).
**That option is unreachable in cluster mode**, because every rank is launched
with stdin closed:

```python
subprocess.run(command, cwd=workdir, stdin=subprocess.DEVNULL, ...)
```

The coordinator is rank 0 and is launched this way too, so there is no TTY
anywhere in a cluster run. Any prompt that is reached without a pre-set
config value will fail or spin.

**Therefore: set `n_trials` to the final target (200) on the very first run.**
The trial count is then a *floor* that rises every time you resume, and a stage
boundary is just "how long you let this run go".

`n_startup_trials` is locked in at the same time — set it to 60 now if you want
the default 30% exploration phase. You cannot lower it later without discarding
the study.

### 2b. `timeout_seconds` is the run deadline, and it is the stage gate

Every rank is launched wrapped in coreutils `timeout`, and the launcher enforces
the same deadline:

```python
("timeout", "--kill-after=5s", f"{timeout_seconds}s", "env", ...)
```

So `timeout_seconds` bounds the **entire run**, not startup. That makes it the
natural way to end a stage: let the budget expire, and the study resumes next
time from the trials that completed.

Two consequences:

- `timeout_seconds` lives in the **cluster TOML**, which is *not* part of the
  study settings — so it is safe to change between stages. This is the one knob
  you edit per stage.
- The collective timeout is now a **separate field**,
  `collective_timeout_seconds`. It defaults to `min(1800, timeout_seconds)`, so
  raising the run deadline does not silently make a hung NCCL collective hang
  for a week. Keep it at 1800 unless you have a reason.

When the deadline fires, the in-flight trial is lost but completed trials are
already persisted (the study is an append-only JSONL journal). Worst case you
lose one trial's work.

One residual coupling worth knowing: `timeout_seconds` is *also* used as the
preflight budget, so a preflight that hangs (rather than failing) will sit for
the full deadline. Preflights have `ConnectTimeout=10` and run with
`BatchMode=yes`, so a hang is unlikely — but if a stage appears to do nothing at
all, check the preflight logs before assuming the optimization is running.

---

## 3. One-time configuration

Do this once, before stage 1. After this, **do not edit these values** — they
are captured in the study on the first run, and later edits are ignored.

In your Heretic config (e.g. `config.dsv41.toml`):

```toml
# The FINAL target. Stages stop early by timing out, not by lowering this.
n_trials = 200
n_startup_trials = 60

# Required: there is no TTY to answer the "checkpoint exists" prompt.
checkpoint_action = "continue"

# Keep this at the default. It is excluded from the study settings, so a
# custom value is silently lost on resume and the study would not be found.
study_checkpoint_dir = "checkpoints"

# Optional: slower but lower peak memory, which matters on unified memory.
offload_outputs_to_cpu = true
```

Then, per stage, edit **only** `timeout_seconds` in the cluster TOML.

---

## 4. Running a stage

```sh
cd /opt/heretic-dgx
uv run heretic \
  --cluster ../heretic-cluster-tp4.toml \
  --config ./config.dsv41.toml \
  --model /models/DeepSeek-V4.1-Flash \
  --seed 20250913
```

Pass `--seed` explicitly. Without it each run picks a random seed, which
changes the TPE sampler's behavior between stages and makes the search
needlessly inconsistent.

When the stage's `timeout_seconds` expires the run ends. Check progress and
decide whether to continue.

### Resuming is the same command

Just run it again. `checkpoint_action = "continue"` means no prompt: it finds
the study, restores the stored settings, and runs the remaining trials up to
`n_trials`.

---

## 5. Stage-by-stage

### Stage 0 — load check (no trials)

Do not skip this. Confirm the model loads and the Engram offload engages before
spending hours on optimization.

**Result: FAILED.** Recorded on `gx10-node-1` against the materialised
checkpoint at `/models/DeepSeek-V4.1-Flash`:

```
* Trying dtype bfloat16...
* Failed: The checkpoint you are trying to load has model type `deepseek_v41`
  but Transformers does not recognize this architecture.
* Trying dtype float32...
* Failed: (same)
Exception: Failed to load model with all configured dtypes.
```

It fails inside `Model(settings)`, before any GPU work, for every configured
dtype. The cause is external to this repository: see [`BLOCKERS.md`](BLOCKERS.md).

An earlier version of this section told you to watch for a
`Engram table DISK-backed: ...` line on every rank. **That line cannot appear** —
it is produced by `EngramOffloadPlan.describe()`, which nothing calls, because
the Engram offload is not wired into the load path at all. `docs/TP4.md` §7 has
been corrected, and blocker 3 in `BLOCKERS.md` explains what is missing.


### Stage 1 — 5 trials

```toml
# cluster TOML
timeout_seconds = 21600   # 6 h
```

Expect a large fixed cost before any trial runs: residuals are gathered once
over 800 prompts (400 good + 400 bad) to compute the refusal direction. If 6
hours is not enough to finish 5 trials, raise the timeout and resume — the
residuals are recomputed each run, so it is not wasted, just slow.

Note that with `n_startup_trials = 60`, the first 5 trials are **random
samples**, not TPE-directed. That is exactly what you want here: a smoke test
of the machinery, not a claim about search quality.

Record from the log:

- per-trial wall clock (the run prints `Elapsed time:` and `Estimated remaining time:`)
- the KLD value for each trial
- the baseline KLD (printed before optimization)

**Then size later stages from the measured per-trial time** rather than the
table above — the numbers here are placeholders until stage 1 tells you the
real rate.

Go/no-go:

- 5 trials completed and recorded
- KLD values finite and well under 0.5 (the tool warns above 0.5)
- the refusal-rate scorer is returning something other than 0 or 1 constantly
- memory stable, no OOM, no rank restarts

### Stage 2 — to 50 trials

```toml
timeout_seconds = 259200  # 72 h
```

~45 more trials. TPE starts exploiting after trial 60, so this stage is still
mostly exploration. Expect the Pareto front to be broad rather than deep.

Go/no-go: best KLD at 50 trials is at least as good as at 5, and the refusal
rate has moved in the intended direction.

### Stage 3 — to 100 trials

```toml
timeout_seconds = 432000  # 120 h
```

This is where TPE begins to matter — trials 61-100 are the first directed ones.

Go/no-go: best KLD still improving, or improving much more slowly. If it has
plateaued, stopping at 100 is a defensible result and you can go straight to
export.

### Stage 4 — to 200 trials

```toml
timeout_seconds = 1209600  # 336 h / 14 days
```

This stage reaches `n_trials` and sets `finished = True`. Then it tries to open
the trial-selection menu — which needs a TTY it does not have.

**So do not rely on stage 4 to export.** Run the export as its own step (below).

---

## 6. Exporting afterwards

Export is a separate run, because it needs the fields that prompts would
otherwise ask for. Pass them on the command line: they are **not** part of the
stored settings, so they survive a resume, and the `exclude=True` fields
(`save_directory`, `study_checkpoint_dir`) are exactly the ones you must supply
every time.

```sh
uv run heretic \
  --cluster ../heretic-cluster-tp4.toml \
  --config ./config.dsv41.toml \
  --model /models/DeepSeek-V4.1-Flash \
  --trial-index 0 \
  --export-strategy standalone \
  --model-action save \
  --save-directory /models/DeepSeek-V4.1-Flash-Heretic
```

- `--trial-index 0` is the first entry of the sorted Pareto front.
- `--export-strategy standalone` is what the two-node release validated and
  what the exporter's verification path is built around.
- Omitting `--model-action` or `--save-directory` reintroduces a prompt, which
  cannot be answered in cluster mode.

The exported checkpoint gets a `README.md` model card with the abliterated and
baseline value for every scorer, KLD included — that is the documentation you
asked about.

---

## 7. Cost expectations

The per-trial cost is dominated by the refusal-rate scorer, which generates text
(`max_response_length = 100` tokens) for 100 prompts. The KLD scorer only needs
one forward pass per prompt.

If a full 200-trial run is too expensive, the honest levers in order:

1. **Reduce the scorer prompt counts** in `[scorer.KeywordRate.prompts]` and
   `[scorer.KLDivergence.prompts]` (e.g. `test[:25]`). This cuts per-trial cost
   linearly and directly changes what KLD means, so record the values you use.
2. **Reduce `max_response_length`**, if your refusals are detectable early.
3. **Stop at 100.** A 100-trial result is a legitimate output.
4. Reduce `n_trials` — but only *before* the first run, since it is locked into
   the study afterwards.

---

## 8. What this plan cannot tell you yet

Everything above assumes the loop works. One of the two open questions has since
been answered, and the answer is no:

- ~~**Whether Heretic can load DeepSeek V4.1 Flash at all.**~~ **Answered: it
  cannot.** `transformers` has no `deepseek_v41` implementation at the pinned
  commit, at the latest release (5.17.0), or on `main`, and Heretic loads models
  only through `transformers`. See [`BLOCKERS.md`](BLOCKERS.md). The blocked
  command and its exact error are recorded in stage 0 above.
- **The real per-trial time.** The timeout values in the stage table are
  estimates. Replace them with measured values after stage 1.

Neither the N-node generalization nor the Engram disk path has been exercised on
physical four-node hardware. See `docs/TP4.md` §10.

