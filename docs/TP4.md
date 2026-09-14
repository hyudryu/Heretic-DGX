# Running Heretic on four DGX Sparks (TP4)

This is the setup and runbook for DeepSeek V4.1 Flash on a four-node DGX Spark
cluster. It covers what changed relative to the original two-node release, why
the Engram tables live on disk, and how to run and troubleshoot it.

Everything below is written to be used later, from scratch, on the real nodes.

---

## 1. The short version

```sh
# On all four nodes, at the same absolute path:
git clone https://github.com/hyudryu/Heretic-DGX.git /opt/heretic-dgx
cd /opt/heretic-dgx
git checkout main
uv sync --frozen

# On the coordinator only:
cp cluster.tp4.example.toml ../heretic-cluster-tp4.toml
$EDITOR ../heretic-cluster-tp4.toml        # node list + fabric interface

uv run heretic \
  --cluster ../heretic-cluster-tp4.toml \
  --config ./config.default.toml \
  --model /models/DeepSeek-V4.1-Flash
```

---

## 2. Why four nodes, and why the Engram tables are on disk

DeepSeek V4.1 Flash is a 552B-parameter MoE. Two things about it drive the
whole design:

**The transformer weights need more than two Sparks.** At TP4 the backbone
fits across four 128 GB unified-memory nodes. This is what the N-node change
in this repository enables.

**The Engram n-gram tables are too large to co-reside.** V4.1 Flash adds Engram
layers at layer 1 and layer 14. From the released `config.json`:

| Field | Value |
|---|---|
| `engram_layer_ids` | `[1, 14]` |
| `engram_num_embeddings` | `[384006168, 384016682]` |
| `engram_head_dim` | `256` |
| `engram_n_heads` | `8` |
| `engram_max_ngram_size` | `4` |

Measuring the released checkpoint directly, each table is `rows x head_dim` in
fp8 e4m3 plus one ue8m0 scale per 32 columns:

| Tensor | Shape | Bytes |
|---|---|---|
| `layers.1.engram.embed.weight` | `[384006168, 256]` | 91.6 GiB |
| `layers.14.engram.embed.weight` | `[384016682, 256]` | 91.6 GiB |
| `layers.{1,14}.engram.embed.scale` | `[rows, 8]` | 2.9 GiB each |

That is **189.1 GiB** of tables, and **47.3 GiB per rank at TP4**. The MoE
backbone is a further 307.2 GB, or **71.5 GiB per rank at TP4**.

So a single Spark would need roughly `71.5 + 47.3 = 118.8 GiB` resident out of
128 GB of unified memory -- which is shared with the host and has to also hold
activations, the CUDA context, and Heretic's own residual tensors. There is no
realistic headroom at that margin.

So the tables stay on the SSD. With `engram_disk = true`, only the transformer
weights are loaded into memory; the Engram tables are read row-by-row from the
safetensors shards on demand.

### Why that is fast enough

The naive reading -- "189 GiB on SSD means a disk read per lookup" -- would be
far too slow. Two properties of the real access pattern make it work:

1. **Only this rank's rows are ever read.** The checkpoint stores each layer's
   full table, but a rank looks up only its own row range, so rank *r* of 4
   reads ~47 GiB of rows, not 189 GiB.

2. **Lookups are batched and repeat heavily.** Engram hashes each position into
   `(max_ngram_size - 1) x n_heads = 24` bucket ids, but the whole forward's
   ids are gathered at once, rows are de-duplicated before reading, and the
   reads for both Engram layers go out in a single parallel batch over a shared
   32-thread pool.

This is the same mechanism used by the working vLLM DGX Spark deployment
([`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)),
which measured ~7.7 ms serial versus ~3.1 ms parallel per step at concurrency 1.
This port reads the tables read-only and does not depend on vLLM.

### What "read-only" means here

The tables are opened with `os.open(path, os.O_RDONLY)` and given
`POSIX_FADV_RANDOM`. Nothing in the ablation path writes to them, and Heretic
never trains them -- they are frozen. The exported checkpoint preserves them
untouched, exactly as the standalone exporter already does for other non-target
artifacts.

---

## 3. Prerequisites

Per node:

- Ubuntu (or equivalent) on a DGX Spark, 128 GB unified memory.
- CUDA + NCCL with working node-to-node GPU collectives.
- Python 3.10+ and `uv`.
- The **same** checkout revision at the same absolute path on all four nodes.
- The **same** extracted model checkpoint at the same absolute path on all four.

SSH:

- Key-based SSH from the coordinator to every worker. The launcher uses
  `ssh -o BatchMode=yes`, so password auth will fail. Verify:

  ```sh
  for h in spark2 spark3 spark4; do ssh -o BatchMode=yes "$h" true && echo "$h ok"; done
  ```

- The launcher connects to the **coordinator locally** (no SSH to itself) and
  to ranks 1..N-1 over SSH. If you list the coordinator as a remote hostname it
  will still be launched locally -- `rank 0` never goes over SSH.

---

## 4. Cluster configuration

Copy `cluster.tp4.example.toml` outside the repository and fill it in.

```toml
python = "/opt/heretic-dgx/.venv/bin/python"
workdir = "/opt/heretic-dgx"
backend = "nccl"
master_port = 29500

# Bounds the ENTIRE run (every rank is wrapped in coreutils `timeout`).
timeout_seconds = 604800          # 7 days, for a full 200-trial run

# Separate process-group collective timeout, so a long run still fails fast
# when a collective genuinely hangs. Must not exceed timeout_seconds.
collective_timeout_seconds = 1800

nccl_socket_ifname = "enp1s0f0np0"

engram_disk = true
engram_disk_path = "/models/DeepSeek-V4.1-Flash"
engram_disk_threads = 32
engram_disk_chunk = 16

[[nodes]]                       # rank 0, the coordinator
host = "spark1"
rank_address = "100.97.4.16"

[[nodes]]                       # rank 1
host = "spark2"
rank_address = "100.66.93.123"

[[nodes]]                       # rank 2
host = "spark3"
rank_address = "100.101.49.94"

[[nodes]]                       # rank 3
host = "spark4"
rank_address = "100.82.66.54"
```

Key points:

- **Order matters.** The first `[[nodes]]` entry is rank 0 (coordinator) and is
  launched locally. Its `rank_address` becomes `MASTER_ADDR`.
- **`python` and `workdir` must be absolute** and identical on every node. Both
  are validated at load time.
- **`host` is the SSH destination**, not the rank address. An `~/.ssh/config`
  alias is usually the cleanest choice.
- **`rank_address` must be reachable from every other node** and must be the
  address you want NCCL to use. See the warning below.
- `nccl_socket_ifname` must name an interface present on all four nodes.
  Verify with `ip -br link` on each.
- Replace every placeholder. The loader rejects duplicate hosts or duplicate
  rank addresses.

### A warning about Tailscale addresses

The four addresses above are Tailscale (100.64.0.0/10) mesh addresses. If that
is the only path between your nodes, TP4 will work but collectives will be slow
-- each MoE layer all-reduces across that mesh, and every token pays for it.

If your Sparks have a direct high-speed interconnect, use **those** addresses
for `rank_address` and keep Tailscale for the SSH `host` path. The split is
deliberate: the config separates the management path (`host`) from the rank
traffic path (`rank_address`) exactly so you can do this.

---

## 5. Checkpoint layout

`engram_disk_path` must point at a directory containing:

```
model.safetensors.index.json
model-00001-of-00048.safetensors
model-00002-of-00048.safetensors
...
config.json
tokenizer.json
```

The Engram reader resolves tensor locations from
`model.safetensors.index.json` -> `weight_map`, then reads the tensor's
`data_offsets` from that shard's safetensors header. It expects:

```
layers.1.engram.embed.weight     # [384006168, 256],  F8_E4M3, 91.6 GiB
layers.1.engram.embed.scale      # [384006168, 8],    F8_E8M0,  2.9 GiB
layers.14.engram.embed.weight    # [384016682, 256],  F8_E4M3, 91.6 GiB
layers.14.engram.embed.scale     # [384016682, 8],    F8_E8M0,  2.9 GiB
```

### Every entry must be a regular file, not a symlink

This is the trap when the model is already in the Hugging Face cache. A cache
`snapshots/<commit>/` directory contains **only symlinks** into `blobs/`, and
Heretic's preflight rejects them:

- `checkpoint_identity._read_json_object` uses `lstat()` and requires
  `S_ISREG`, so `config.json` fails with
  *"checkpoint metadata must be a regular file"*.
- `_hash_regular_file` additionally opens with `O_NOFOLLOW`.

That check is deliberate -- it stops a symlink being swapped between the
identity hash and the load -- so the fix is to materialise the layout rather
than to relax it.

**Hard links are the right fix**, because they are real directory entries
(`S_ISREG`) sharing the same inode, so they cost no additional disk space:

```sh
REPO="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash"
SNAP="$REPO/snapshots/$(cat "$REPO/refs/main")"
sudo mkdir -p /models/DeepSeek-V4.1-Flash
sudo chown "$USER:$USER" /models/DeepSeek-V4.1-Flash
for f in "$SNAP"/*; do
  ln -f "$(readlink -f "$f")" "/models/DeepSeek-V4.1-Flash/$(basename "$f")"
done
```

This works because the cache and `/models` are on the same filesystem, which is
true on all four nodes here (`/dev/nvme0n1p2`). Hard-linking across filesystems
fails, and copying is not an option: the checkpoint is 475 GiB and the smallest
node has ~19 GB free.

Verify before running:

```sh
find /models/DeepSeek-V4.1-Flash -maxdepth 1 -type l | wc -l   # expect 0
ls /models/DeepSeek-V4.1-Flash/config.json                     # regular file
```

**Status on this cluster: done.** The farm exists at
`/models/DeepSeek-V4.1-Flash` on all four nodes, built from commit
`dba1be0a40aa45a94ad051997016db3960a90277` of
`models--deepseek-ai--DeepSeek-V4.1-Flash`:

| Check | Node 1 | Node 2 | Node 3 | Node 4 |
|---|---|---|---|---|
| hard links created | 55 | 55 | 55 | 55 |
| directories copied | 4 | 4 | 4 | 4 |
| symlinks remaining | 0 | 0 | 0 | 0 |
| `*.safetensors` regular files | 48 | 48 | 48 | 48 |
| `config.json` | regular, `links 2` | regular, `links 2` | regular, `links 2` | regular, `links 2` |

`links 2` is the confirmation that the hard link worked: the same inode is now
reachable from both `blobs/` and `/models/`. Free space was unchanged on every
node, because a hard link adds a directory entry and nothing else.

The script that did it (idempotent, safe to re-run) is in the operations
workspace as `_tools/farm.sh`.

### Other notes

- This is the **Hugging Face** checkpoint, not a `convert.py`-style
  `model{rank}-mp4.safetensors` TP4 conversion. The reader derives each rank's
  row range itself from the full-table tensor.
- Keep this directory on local NVMe on every node. If it lives on NFS, every
  rank's row reads cross the network, and the page cache that makes this fast
  (on GB10, unified memory) is shared with the GPU pool.
- Row reads are positional (`preadv`) and page-cache friendly; the working set
  is small because of the de-duplication.
- **Expect the preflight to be slow, and now know how slow.** It hashes every
  payload file, so each run reads the full 475 GiB checkpoint once per node, and
  ranks are preflighted sequentially over SSH. Measured against this checkpoint
  by running the gate directly (`python -m heretic.checkpoint_identity`):

  | Node | Wall clock |
  |---|---|
  | `gx10-node-1` | 556 s |
  | `gx10-node-2` | 590 s |
  | `gx10-node-3` | 544 s |
  | `spark-node-4` | 702 s |

  That is roughly **9-12 minutes per node**, and about **37 minutes** for a
  full four-rank preflight run sequentially over SSH — on *every* run, including
  each staged resume. Budget for it, and do not mistake it for a hang. All four
  nodes produce the same identity:

  ```
  digest      = 3a7f5ce2b986c2300e8381b1f1c089e7342abe0fa95a16b4102f87077b3a56c7
  file_count  = 50
  total_bytes = 510304181917   (475.3 GiB)
  ```

---

## 6. Running

From the coordinator:

```sh
cd /opt/heretic-dgx
uv run heretic \
  --cluster ../heretic-cluster-tp4.toml \
  --config ./config.default.toml \
  --model /models/DeepSeek-V4.1-Flash
```

What happens, in order:

1. The cluster TOML is loaded and validated (node count, distinct hosts and
   addresses, absolute paths, Engram settings).
2. A **preflight** runs on every rank over SSH: it hashes the source tree and
   the checkpoint payload. All ranks must agree exactly, or the run aborts
   before any GPU work starts.
3. A **CPU-only Gloo collective probe** runs on all four ranks to prove the
   process-group wiring before CUDA is involved.
4. The rank applications launch: rank 0 locally, ranks 1-3 over SSH, all with
   `WORLD_SIZE=4` and their own `RANK`.
5. Optimization runs with rank 0 driving and every rank executing each
   operation in lockstep; the exported checkpoint is written by rank 0.

Useful environment variables (forwarded to ranks):

| Variable | Effect |
|---|---|
| `HERETIC_LOG_DIR` | Where per-rank stdout/stderr logs are written. Default `~/.local/state/heretic/rank-logs`. |
| `HF_HOME`, `HF_HUB_CACHE` | Forwarded to ranks. |

Useful cluster-file fields:

| Field | Default | Meaning |
|---|---|---|
| `timeout_seconds` | `900` | Wall-clock budget for the **entire** rank application, enforced by a `timeout` wrapper. This is the run deadline, not a startup budget. |
| `collective_timeout_seconds` | `min(1800, timeout_seconds)` | Process-group collective timeout. Kept separate so a multi-day run still fails fast on a hung collective. Must not exceed `timeout_seconds`. |
| `engram_disk_threads` | `32` | Size of the shared Engram read pool. |
| `engram_disk_chunk` | `16` | Rows per read task, capped for prefill-sized batches. |

For a staged rollout that starts at 5 trials and scales to 200, see
[`STAGED_RUN_PLAN.md`](STAGED_RUN_PLAN.md). It covers the per-stage timeout
values and two non-obvious constraints of the cluster path (settings are locked
into the study on the first run, and there is no TTY to answer prompts).

### Where the exported model lands, and why rank 0 is node 4

**Only rank 0 writes an export.** `LocalModelRuntime.save_adapter` and
`save_merged` read:

```python
sink = directory if rank == 0 else tempfile.mkdtemp(prefix=f"heretic-...-rank-{rank}-")
...
is_main_process=rank == 0
```

so every non-zero rank writes into a throwaway temp directory that is deleted
afterwards. And only rank 0 reaches the export menu at all — worker ranks enter
`run_dgx_worker` and loop on `receive()` instead. The exported model therefore
lands on **whatever filesystem rank 0 is running on**.

Since rank 0 is launched **locally**, "save the model on node 4" means "make
node 4 rank 0, and launch from node 4". There is no shared filesystem on this
cluster (NFS is inactive, and there are no `nfs`/`sshfs`/`cifs` mounts), so a
mount-based redirect is not available.

Free space is what makes this necessary rather than merely tidy:

| Node | Root filesystem | Free |
|---|---|---|
| `spark-node-4` | 3.7T | **1.5T** |
| `gx10-node-3` | 916G | 125G |
| `gx10-node-2` | 916G | 71G |
| `gx10-node-1` | 916G | **19G** |

Node 1, the original coordinator, cannot hold an export.

The cluster file therefore lists node 4 first:

```toml
[[nodes]]
host = "10.100.58.4"          # spark-node-4, rank 0, launched locally
rank_address = "10.100.58.4"

[[nodes]]
host = "10.100.58.1"          # gx10-node-1
rank_address = "10.100.58.1"

[[nodes]]
host = "10.100.58.3"          # gx10-node-2
rank_address = "10.100.58.3"

[[nodes]]
host = "10.100.58.2"          # gx10-node-3
rank_address = "10.100.58.2"
```

and the run is launched **from node 4**:

```sh
ssh hyudryu@10.100.58.4
cd /opt/heretic-dgx
uv run heretic \
  --cluster /opt/heretic-cluster-tp4.toml \
  --config ./config.default.toml \
  --model /models/DeepSeek-V4.1-Flash \
  --model-action save \
  --export-strategy standalone \
  --trial-index 0 \
  --save-directory /models/abliterated/deepseek-v4.1-flash-heretic
```

Three things to watch:

- **Do not run this from node 1.** The launcher starts rank 0 *locally*, so a
  config that lists node 4 first but is invoked on node 1 would give node 1 rank
  0's identity while node 4 is also launched as rank 0 — a silent rank
  misassignment. Node 1's copy of the config has been parked as
  `/opt/heretic-cluster-tp4.toml.stale-node1-coordinator` to prevent this.
- **`--save-directory` must be passed every time.** It is `exclude=True` in
  `Settings`, so it is never persisted into the study config.
- **Rank order is not free.** Row sharding for the Engram tables is
  `EngramTableLayout.for_rank(rank=..., world_size=...)`, so moving a node
  changes which rows it owns. That is fine and symmetric, but a study resumed
  across a rank-order change is not comparable to an earlier stage.


---

## 7. Verifying the Engram offload is actually active

> **This does not work yet.** The mechanism described in this section is
> implemented and unit-tested but is not wired into the model load path, so
> `engram_disk = true` currently has no effect. See
> [`BLOCKERS.md`](BLOCKERS.md) — the load cannot even be attempted until
> transformers supports the architecture.

The intended signal is a line in each rank's log at startup:

```
2 engram layers, 96001542 rows per rank kept on disk (47.3 GiB not allocated)
```

That text comes from `EngramOffloadPlan.describe()`. **Nothing calls it**, so the
line is never printed today. If you are looking for evidence that the offload is
active, this is not it — there is currently no such evidence to find.

The status today:

- `heretic.engram_disk` (`DiskEngramTable`, `EngramDiskConfig`,
  `from_rank_environment`, `gather_dequant_many`) is referenced only by
  `tests/test_engram_disk.py`.
- `heretic.model_loading.build_engram_offload_plan`, `is_deepseek_v41_config`,
  and `is_engram_tensor` are referenced only by
  `tests/test_model_loading_deepseek.py`.
- `heretic.model.Model.__init__` goes straight to
  `from_pretrained(..., tp_plan="auto")` with no Engram handling, no excluded
  tensors, and no forward hook.

So if you run it as-is, the Engram tables are ordinary parameters and the
process will try to allocate them like any other weight.

What *is* real and verifiable without launching a cluster: `EngramTableLayout`
computes per-rank row ranges from the real checkpoint geometry, pinned by
`tests/test_engram_disk.py` (including a negative control proving the
row-offset test can actually detect the upstream bug), and
`tests/test_model_loading_deepseek.py` checks the plan against the released
config's `engram_*` fields.


---

## 8. Troubleshooting

**"DGX cluster must define at least two nodes"** -- the TOML has fewer than two
`[[nodes]]` entries.

**"engram_disk_path is required when engram_disk is true"** -- set the path or
turn `engram_disk` off.

**Preflight fails with a source or checkpoint mismatch** -- the four nodes are
not running the same revision, or their checkpoints differ. This is intentional
and is checked before any GPU work. Fix by syncing the checkout and re-copying
the checkpoint.

**Collective probe fails / hangs** -- `MASTER_ADDR` or the fabric is wrong, or
the nodes cannot reach each other on `rank_address`. Confirm every
`rank_address` is reachable from every node, not just from the coordinator.

**Out-of-memory during load** -- either the Engram offload is not active (see
section 7), or something else is resident. On GB10, host page cache competes
with the GPU pool, so a cold Engram cache is reclaimable but not free.

**Very low tokens/second** -- check whether rank traffic is crossing Tailscale
rather than a direct fabric link (section 4). This write-up's earlier note on
the 100.x addresses applies.

**A rank fails and the others hang** -- they should not: a failure cancels its
peers. If you do see a hang, capture `HERETIC_LOG_DIR` output from all ranks;
the failure detail names the rank.

---

## 9. What changed in this repository

Relative to the original two-node implementation:

- The cluster config, launch planner, rank environment, preflight, collective
  probe, command channel, and application runner are all generalized from
  exactly two ranks to N ranks. Two-node configurations still work unchanged.
- `TorchDistributedCommandChannel` broadcasts from rank 0 to all workers and
  gathers errors from every rank, instead of assuming a single peer.
- `tp_capabilities` accepts an N-rank device mesh instead of requiring size 2.
- New `heretic.engram_disk` module: the read-only, disk-backed Engram tables
  described above, including the rank row-offset fix.
- New `model_loading` support for detecting DeepSeek V4.1 Flash and computing
  the per-rank Engram row ranges.

### The row-offset bug worth knowing about

The upstream vLLM patch originally based every read at the start of the **full**
tensor while computing rank-local row ids, so every rank above 0 read rank 0's
rows. The output still looked like text, because the rows are real embedding
rows -- just the wrong ones -- so smoke tests passed. This port carries the fix
and pins it with a test (`test_every_rank_reads_its_own_rows`) plus a negative
control that proves the test can actually detect the bug.

---

## 10. Validation status and limits

**Blocker: the model cannot be loaded at all yet.** `transformers` does not
implement the `deepseek_v41` architecture — not at the pinned commit, not in the
latest PyPI release (5.17.0), and not on `main`. Heretic loads models only
through `transformers`, so there is no load path.

This was verified by running the real entry point on `gx10-node-1` against the
materialised checkpoint:

```
* Trying dtype bfloat16...
* Failed: The checkpoint you are trying to load has model type `deepseek_v41`
  but Transformers does not recognize this architecture.
...
Exception: Failed to load model with all configured dtypes.
```

It fails inside `Model(settings)`, before any GPU work. The full analysis, with
evidence and the four further blockers behind it, is in
[`BLOCKERS.md`](BLOCKERS.md). Read that first.

The planned validation order is unchanged, and step 1 has now been *attempted*
and failed for reasons outside this repository:

1. Load the model on all four ranks with the Engram offload on, and confirm the
   "DISK-backed" lines and that memory stays within budget. — **FAILS: no
   transformers implementation of the architecture.**
2. Run a short optimization pass on a small prompt set.
3. Export standalone, reload, and generate. — **also blocked: the standalone
   exporter is hard-coded to Laguna S 2.1 FP8 (48 layers); this model has 40.**
4. Measure KL divergence as a sanity check.

Per the original project's own release discipline, a new topology or model is
not "supported" until it has been through load, optimize, export, reload, and
generation. **DeepSeek V4.1 Flash on TP4 is not supported.**

