#!/usr/bin/env bash
# Lab 04 — Intra-node NVLink / NVSwitch on the HGX H100 baseboard.
#
# Captures the full 8-GPU NVLink topology and the single-node NCCL all-reduce
# bandwidth curve — the intra-node ceiling that Part II's inter-node numbers
# are measured against.
#
# GPU SAFETY: this needs all 8 GPUs on one node. In this environment that node
# is normally held by a DWS capacity holder, so the run is performed inside a
# pre-existing 8-GPU "workbench" pod (see manifests/nccl-workbench-a.yaml).
# Point LAB04_POD at whatever 8-GPU pod you have; the script only execs into it.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-04"; mkdir -p "$OUT"
POD="${LAB04_POD:-nccl-workbench-a}"

NODE="$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}')"
echo "[lab-04] using 8-GPU pod $POD on node $NODE"

# 1. Full 8-GPU NVLink mesh + per-link rate + fabric state.
cap_run "topo-8gpu"    "${OUT}/topo-8gpu.txt"     -- kubectl exec "$POD" -- nvidia-smi topo -m
cap_run "nvlink-status" "${OUT}/nvlink-status.txt" -- kubectl exec "$POD" -- nvidia-smi nvlink --status -i 0
cap_run "fabric-state" "${OUT}/fabric-state.txt"   -- \
  kubectl exec "$POD" -- bash -lc 'nvidia-smi -q -i 0 | grep -iA3 "^    Fabric"'

# 2. Single-node 8-GPU NCCL all-reduce sweep (nccl-tests, prebuilt in NGC image).
cap_run "all-reduce-8gpu" "${OUT}/all_reduce_8gpu.txt" -- \
  kubectl exec "$POD" -- bash -lc 'NCCL_DEBUG=WARN all_reduce_perf -b 8 -e 8G -f 2 -g 8'

cap_verify_provenance "lab-04" "assets/lab-04" "$NODE" \
  "intra-node NVLink: 8-GPU NV18 mesh, 18x26.562GB/s links; single-node all-reduce peak busbw"
echo "[lab-04] done. See assets/lab-04/"
