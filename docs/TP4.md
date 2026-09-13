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

**The Engram n-gram tables do not fit at all.** V4.1 Flash adds Engram layers
at layer 1 and layer 14. From the released `config.json`:

| Field | Value |
|---|---|
| `engram_layer_ids` | `[1, 14]` |
| `engram_num_embeddings` | `[384006168, 384016682]` |
| `engram_head_dim` | `256` |
| `engram_n_heads` | `8` |
| `engram_max_ngram_size` | `4` |

That is `(384006168 + 384016682) x 256 x 8` = **1.573e12 parameters**. At the
checkpoint's fp8 e4m3 storage that is **~1.43 TiB**. Per rank at TP4 it would
still be **~366 GiB** -- more than twice a single node's entire memory.

So the tables stay on the SSD. With `engram_disk = true`, only the transformer
weights are loaded into memory; the Engram tables are read row-by-row from the
safetensors shards on demand.

### Why that is fast enough

The naive reading -- "1.43 TiB on SSD means a disk read per lookup" -- would be
far too slow. Two properties of the real access pattern make it work:

1. **Only this rank's rows are ever read.** The checker stores each layer's
   full table, but a rank looks up only its own head range, so rank *r* of 4
   reads ~47 GiB of rows, not 1.43 TiB.

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
timeout_seconds = 3600
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

`engram_disk_path` must point at the **extracted Hugging Face checkpoint
directory**, i.e. the one containing:

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
`data_offsets` from that shard's safetensors header. It expects the two tensors:

```
layers.1.engram.embed.weight     # rows x head_dim, F8_E4M3
layers.1.engram.embed.scale      # rows x (head_dim / 32), F8_E8M0
layers.14.engram.embed.weight
layers.14.engram.embed.scale
```

Notes:

- This is the **Hugging Face** checkpoint, not a `convert.py`-style
  `model{rank}-mp4.safetensors` TP4 conversion. The reader derives each rank's
  row range itself from the full-table tensor.
- Keeping this directory on local NVMe on every node matters a great deal. If
  it lives on NFS, every rank's row reads cross the network, and the page cache
  that makes this fast (on GB10, unified memory) is shared with the GPU pool.
- Row reads are positional (`preadv`) and page-cache friendly; the working set
  is small because of the de-duplication.

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
| `timeout_seconds` | `900` | Per-rank launch and preflight budget. Raise it: a TP4 load from disk is slow. |
| `engram_disk_threads` | `32` | Size of the shared Engram read pool. |
| `engram_disk_chunk` | `16` | Rows per read task, capped for prefill-sized batches. |

---

## 7. Verifying the Engram offload is actually active

The rank logs report it at startup. On each rank you should see a line like:

```
Engram table DISK-backed: 96001542 rows x 256 per rank stay on disk (47.9 GiB not allocated)
```

If you see that line, the tables are **not** resident and are being read from
`engram_disk_path`. If instead the process allocates hundreds of GiB or dies
with unified-memory exhaustion, disk mode is not active -- check that
`engram_disk = true` and `engram_disk_path` are set, and that the path exists
on that node.

A quick standalone check that the reader works against your checkpoint, without
launching the cluster: see `tests/test_engram_disk.py`, which builds synthetic
shards in the real layout and verifies the row offsets per rank.

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

Be clear-eyed about this: the N-node generalization and the Engram reader are
covered by the unit tests in `tests/` (76 tests, green on Linux), but **they
have not yet been run end to end on the four physical Sparks.** The plan is to
validate TP4 in this order:

1. Load the model on all four ranks with the Engram offload on, and confirm the
   "DISK-backed" lines and that memory stays within budget.
2. Run a short optimization pass on a small prompt set.
3. Export standalone, reload, and generate.
4. Measure KL divergence as a sanity check.

Per the original project's own release discipline, a new topology or model is
not "supported" until it has been through load, optimize, export, reload, and
generation. Treat this document as the plan for that validation, not as a claim
that it is already done.
