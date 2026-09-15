#!/usr/bin/env bash
# Deploy the hyper-connection collapse patch and start a FRESH study.
#
# THIS IS DESTRUCTIVE. It stops the running study and the running deployment.
# It is written out rather than executed because it stops a run the user asked
# for. Read it end to end before running it.
#
# WHY A FRESH STUDY, NOT A RESUME
# -------------------------------
# The 69 trials in the existing Optuna journal were scored against directions
# computed from the mean-collapsed capture. After the patch those directions no
# longer describe what the engine emits, so resuming would continue a search
# whose history was measured in a different space. The journal is archived, not
# continued -- set checkpoint_action to "continue" only if you deliberately want
# to keep the old trials.
#
# WHAT EACH STEP IS FOR
#  1  the watchdog restarts the study within 60s of it disappearing
#  2  stop the study cleanly; it has no supervisor other than the watchdog
#  3  one 552B model, four Sparks: the old engine must go before the new starts
#  4  deploy recipe ff09e9a8 -- same 49 flags and 34 env vars as the live
#     profile, only the image differs (vllm-dsv41:pinned-ehs3-hccollapse)
#  5  the engine reloads 475 GB; this takes ~10 minutes
#  6  the shim hardcodes the upstream port at startup, so it must be restarted
#     with the new one
#  7  archive the old journal so the fresh study starts clean
#  8  start the study
#  9  verify before trusting it -- a patch that loads is not a patch that works
# 10  restore supervision
set -eu

# RECIPE selects the profile. The two differ ONLY in the engine image, which is
# what makes this script also the rollback path:
#
#   ff09e9a8  vllm-dsv41:pinned-ehs3-hccollapse   patched (mix-weighted collapse)
#   5ae70a64  vllm-dsv41:pinned-ehs2              rollback (pre-patch, validated)
#
# Roll back with:  RECIPE=5ae70a64 NEW_IMAGE=vllm-dsv41:pinned-ehs2 \
#                      scripts/deploy_hc_collapse_patch.sh
RECIPE=${RECIPE:-ff09e9a8}
NEW_IMAGE=${NEW_IMAGE:-vllm-dsv41:pinned-ehs3-hccollapse}
OLD_DEPLOY=${OLD_DEPLOY:-8fd087c422c5}
RUN=/opt/heretic-dgx
STAMP=$(date +%Y%m%d-%H%M%S)
JOURNAL="$RUN/checkpoints/--models--DeepSeek-V4--1-Flash.jsonl"

say() { printf '\n=== %s ===\n' "$*"; }

say "0. preconditions"
echo "  recipe   : $RECIPE"
echo "  image    : $NEW_IMAGE"
docker image inspect "$NEW_IMAGE" >/dev/null 2>&1 || {
    echo "FATAL: $NEW_IMAGE is not present on this node"; exit 1; }
docker run --rm --entrypoint grep -q 'hc_collapse_triton(aux_recon, pre_mix)' \
    "$NEW_IMAGE" /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py \
    || { echo "FATAL: patched line missing from $NEW_IMAGE"; exit 1; }
echo "  $NEW_IMAGE present and patched"

say "1. stop the watchdog (else it restarts the study in 60s)"
sudo pkill -f heretic-watchdog || true
sleep 2
pgrep -f heretic-watchdog >/dev/null && { echo "FATAL: watchdog still alive"; exit 1; }
echo "  watchdog stopped"

say "2. stop the study"
STUDY=$(pgrep -f '^/opt/heretic-dgx/.venv/bin/python .venv/bin/heretic' | head -1 || true)
if [ -n "${STUDY:-}" ]; then
    kill "$STUDY"
    for _ in $(seq 1 30); do
        pgrep -f '^/opt/heretic-dgx/.venv/bin/python .venv/bin/heretic' >/dev/null || break
        sleep 1
    done
    pgrep -f '^/opt/heretic-dgx/.venv/bin/python .venv/bin/heretic' >/dev/null \
        && { echo "  still running; sending SIGKILL"; pkill -9 -f 'bin/heretic --model' || true; }
fi
echo "  study stopped"

say "3. stop the old deployment ($OLD_DEPLOY)"
python3 - "$OLD_DEPLOY" <<'PY'
import json, sys, urllib.request
d = sys.argv[1]
body = json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
    "name":"stop_cluster_deployment","arguments":{"deployment_id":d,"allow_unowned":True}}}).encode()
req = urllib.request.Request("http://100.97.4.16:7878/mcp", data=body,
    headers={"Content-Type":"application/json","Accept":"application/json"})
print(urllib.request.urlopen(req, timeout=300).read().decode()[:300])
PY

say "4. deploy the patched recipe ($RECIPE)"
python3 - "$RECIPE" <<'PY'
import json, sys, urllib.request
r = sys.argv[1]
body = json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
    "name":"deploy_cluster_recipe","arguments":{"recipe_id":r,
    "deployment_name":"Heretic (V4.1 TP4 hc-collapse patch)"}}}).encode()
req = urllib.request.Request("http://100.97.4.16:7878/mcp", data=body,
    headers={"Content-Type":"application/json","Accept":"application/json"})
print(urllib.request.urlopen(req, timeout=900).read().decode()[:400])
PY

say "5. wait for the engine"
python3 <<'PY'
import json, time, urllib.request
# find the new deployment and its rank-0 port
def rpc(method, params=None):
    body = json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params or {}}).encode()
    req = urllib.request.Request("http://100.97.4.16:7878/mcp", data=body,
        headers={"Content-Type":"application/json","Accept":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
res = rpc("tools/call", {"name":"list_cluster_deployments","arguments":{}})
rows = res["result"].get("structuredContent", {}).get("result", [])
mine = [r for r in rows if "hc-collapse" in str(r.get("name",""))]
if not mine:
    raise SystemExit("could not find the new deployment")
d = mine[-1]
did = d["id"]
port = d.get("api_port")
print(f"  deployment {did}, rank0 port {port}")
deadline = time.time() + 1800
while time.time() < deadline:
    res = rpc("tools/call", {"name":"get_cluster_deployment","arguments":{"deployment_id":did}})
    r = res["result"].get("structuredContent", {})
    st = r.get("status")
    print(f"  status={st}")
    if st == "ready":
        print(f"READY port={r.get('port') or port}")
        break
    if st in ("failed","stopped"):
        raise SystemExit(f"engine did not come up: {st}")
    time.sleep(20)
else:
    raise SystemExit("timed out waiting for the engine")
PY

say "6. restart the shim against the new engine port"
# Set ENGINE_PORT to the rank-0 port printed in step 5.
: "${ENGINE_PORT:?set ENGINE_PORT to the rank-0 port printed above}"
sudo pkill -f heretic_shim.py || true
sleep 2
sudo -S env HERETIC_SHIM_UPSTREAM="http://127.0.0.1:${ENGINE_PORT}" setsid nohup \
    /opt/heretic-dgx/.venv/bin/python /home/hyudryu/heretic_shim.py \
    >>/home/hyudryu/heretic_shim.log 2>&1 &
sleep 5
curl -s -o /dev/null -w '  shim health: %{http_code}\n' --max-time 10 http://127.0.0.1:8765/health

say "7. archive the old journal (fresh study)"
if [ -f "$JOURNAL" ]; then
    mkdir -p "$RUN/checkpoints/archive-$STAMP"
    mv "$JOURNAL" "$RUN/checkpoints/archive-$STAMP/"
    echo "  archived to $RUN/checkpoints/archive-$STAMP/"
fi
mv "$RUN/prod-run1.log" "$RUN/prod-run1.log.$STAMP" 2>/dev/null || true

say "8. start the fresh study"
cd "$RUN"
CUDA_VISIBLE_DEVICES= setsid nohup .venv/bin/heretic \
    --model /models/DeepSeek-V4.1-Flash --config ./config.dsv41.toml --seed 20250915 \
    </dev/null >>prod-run1.log 2>&1 &
sleep 20
pgrep -f '^/opt/heretic-dgx/.venv/bin/python .venv/bin/heretic' >/dev/null \
    && echo "  study running" || echo "  WARNING: study did not start; see prod-run1.log"

say "9. VERIFY -- a patch that loads is not a patch that works"
echo "  Watching the first trial. Baseline Keywords should still be ~98/100."
echo "  Then run, on this node:"
echo "    python3 /tmp/diag_direction.py        # directions from the patched capture"
echo "    python3 /tmp/measure_refusal_rate.py  # base vs abliterated refusals"
echo "  Compare the new directions against the mean-collapsed ones, and do not"
echo "  commit to a long search until the refusal count actually moves."

say "10. restore supervision"
sudo install -m 0755 "$RUN/../heretic_watchdog.sh" /usr/local/bin/heretic-watchdog 2>/dev/null || true
sudo setsid nohup /usr/local/bin/heretic-watchdog </dev/null >/dev/null 2>&1 &
sleep 3
pgrep -f heretic-watchdog >/dev/null && echo "  watchdog running" || echo "  WARNING: watchdog not running"

say "done"
grep -E 'Running trial|Baseline Keywords|Elapsed' "$RUN/prod-run1.log" | tail -5
