#!/usr/bin/env bash
# Build the Heretic patch image ON THIS NODE, from this node's own base image.
#
# Why not SparkDeck's create_patched_image: it requires the base image identity
# to match across all selected nodes, and it does not here. The four nodes carry
# content-identical but independently built images -- same model.py hash,
# different image IDs, because each was imported at its own timestamp and
# spark-node-4's is a differently layered 23.5 GB build. Node 1's build
# therefore succeeded while nodes 2-4 were refused with
#   "Base image runtime configuration, filesystem, or platform differs from the
#    first node"
#
# Building locally sidesteps that entirely: each node layers the one file onto
# its own base, which is exactly what the successful node-1 build did.
#
# Additive only -- creates a new tag, touches no running container.
set -u

BASE=${BASE:-vllm-dsv41:pinned-ehs2}
NEW=${NEW:-vllm-dsv41:pinned-ehs3-hccollapse}
TARGET=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py
WORK=/tmp/heretic-patch-build

echo "host: $(hostname)"
echo "base: $BASE -> $NEW"

echo ""
echo "########## base image present? ##########"
if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  echo "  BASE IMAGE MISSING -- aborting"
  exit 1
fi
docker image inspect "$BASE" --format '  id={{.Id}}'

echo ""
echo "########## the file we are replacing, BEFORE ##########"
BEFORE=$(docker run --rm --entrypoint sha256sum "$BASE" "$TARGET" 2>/dev/null | awk '{print $1}')
echo "  sha256: ${BEFORE:-<unreadable>}"

echo ""
echo "########## build context ##########"
rm -rf "$WORK"; mkdir -p "$WORK"
cp /tmp/dsv41_model_patched.py "$WORK/model.py" || { echo "  patched source missing"; exit 1; }

# Assert the patch is present and the old line is gone, guarding on the
# assignment rather than the bare expression (the explanatory comment quotes it).
grep -q 'hc_collapse_triton(aux_recon, pre_mix)' "$WORK/model.py" || { echo "  patch not present"; exit 1; }
if grep -q 'aux_hidden_state = aux_recon.mean(dim=1)' "$WORK/model.py"; then
  echo "  old mean-collapse still present"; exit 1
fi
python3 -m py_compile "$WORK/model.py" && echo "  py_compile: OK"
echo "  bytes: $(stat -c %s "$WORK/model.py")"

cat > "$WORK/Dockerfile" <<EOF
FROM $BASE
COPY model.py $TARGET
EOF
cat "$WORK/Dockerfile"

echo ""
echo "########## build ##########"
docker build -t "$NEW" "$WORK" 2>&1 | tail -6

echo ""
echo "########## verify the patched file is in the new image ##########"
AFTER=$(docker run --rm --entrypoint sha256sum "$NEW" "$TARGET" 2>/dev/null | awk '{print $1}')
echo "  after  sha256: ${AFTER:-<unreadable>}"
echo "  before sha256: ${BEFORE:-<unreadable>}"
if [ -n "${AFTER:-}" ] && [ "$AFTER" != "${BEFORE:-}" ]; then
  echo "  OK: file changed"
else
  echo "  PROBLEM: file unchanged or unreadable"
fi

echo ""
echo "########## confirm the patched line reads back from the image ##########"
docker run --rm --entrypoint grep "$NEW" -n 'aux_hidden_state = hc_collapse_triton(aux_recon, pre_mix)' "$TARGET" \
  || echo "  patched line NOT found"

echo ""
echo "########## disk after ##########"
df -h / | tail -1
rm -rf "$WORK"
