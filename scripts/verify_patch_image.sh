#!/usr/bin/env bash
# Verify the patched image on this node.
set -u
IMG=vllm-dsv41:pinned-ehs3-hccollapse
T=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py
EXPECT=2369dc11fcd6863d3904b5ed2ae58eec234afb2c18121703d26f0f4654a470e3

echo "host: $(hostname)"
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "  MISSING: $IMG"; exit 1
fi
echo "  image id : $(docker image inspect "$IMG" --format '{{.Id}}')"
H=$(docker run --rm --entrypoint sha256sum "$IMG" "$T" 2>/dev/null | awk '{print $1}')
echo "  model.py : $H"
echo "  expected : $EXPECT"
if [ "$H" = "$EXPECT" ]; then echo "  MATCH"; else echo "  MISMATCH"; fi

echo "  patched line:"
docker run --rm --entrypoint grep "$IMG" -n 'hc_collapse_triton(aux_recon, pre_mix)' "$T" 2>/dev/null | sed 's/^/    /'
if docker run --rm --entrypoint grep "$IMG" -q 'aux_hidden_state = aux_recon.mean(dim=1)' "$T" 2>/dev/null; then
  echo "  WARNING: old mean-collapse assignment still present"
else
  echo "  old mean-collapse assignment absent (good)"
fi
