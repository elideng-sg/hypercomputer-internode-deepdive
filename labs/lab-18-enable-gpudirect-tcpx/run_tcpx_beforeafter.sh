#!/usr/bin/env bash
# run_tcpx_beforeafter.sh — the lab-18 "close the cliff" capture: gVNIC BEFORE → TCPX AFTER.
#
# STATUS (honest): the BEFORE half is already captured live on the existing single-gVNIC
# cluster (labs/lab-06 2-node + labs/lab-12 8/16/24-GPU curve — NET/Socket, ~28 GB/s). The
# AFTER half requires the TCPX cluster from scripts/provision_tcpx_pool.sh, which is PENDING:
# the existing cluster lacks Dataplane V2 (a create-time-only gate), so TCPX needs a new
# cluster + scarce A3 Flex capacity. This runner is therefore STAGED — it lays out the exact
# steps the AFTER capture will run once that environment exists. It does NOT fabricate numbers.
#
# When the TCPX pool is live, run:
#   scripts/provision_tcpx_pool.sh up          # stand up the TCPX cluster/pool/plugin
#   bash labs/lab-18-enable-gpudirect-tcpx/run_tcpx_beforeafter.sh after
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/scripts/lib_capture.sh"
OUT="$REPO/assets/lab-18"; mkdir -p "$OUT"

before(){
  # Reuse the already-captured gVNIC baseline — do NOT re-run; just point at the evidence.
  {
    echo "# BEFORE = single-gVNIC / TCP (captured live, labs 06 & 12)"
    echo "# transport (lab-06): NET/IB No device found -> NET/Socket eth0 -> Using network Socket"
    echo "# 2-node 16-GPU busbw : ~28.6 GB/s peak (assets/lab-06/allreduce_2node.txt)"
    echo "# 8/16/24-GPU busbw   : 465 -> 23.7 -> 14.95 GB/s (assets/lab-12/allreduce_*.txt)"
    echo "# GPU Direct RDMA     : Disabled for HCA 0 'eth0'"
  } > "$OUT/before_gvnic_summary.txt"
  echo "wrote $OUT/before_gvnic_summary.txt (evidence: assets/lab-06, assets/lab-12)"
}

after(){
  : "${KUBE_CONTEXT:?set KUBE_CONTEXT to the TCPX cluster context}"
  # 1) TCPX transport read — expect NET/GPUDirectTCPX (contrast the BEFORE NET/Socket)
  cap_run "$OUT/after_tcpx_transport.txt" \
    kubectl --context "$KUBE_CONTEXT" logs tcpx-workbench-0 -c nccl-bench
  # 2) TCPX all-reduce busbw sweep (same message sizes as lab-06/lab-12 for a fair before/after)
  cap_run "$OUT/after_tcpx_allreduce.txt" \
    kubectl --context "$KUBE_CONTEXT" logs tcpx-workbench-0 -c nccl-bench --tail=200
  # 3) NIC/MTU proof on a node
  cap_run "$OUT/after_tcpx_mtu.txt" \
    kubectl --context "$KUBE_CONTEXT" exec tcpx-workbench-0 -c nccl-bench -- bash -c \
      'for i in 1 2 3 4; do ip -o link show eth$i | sed "s/\\\\/ /"; done'
  cap_verify_provenance "$OUT"
}

case "${1:-before}" in
  before) before ;;
  after)  after ;;
  *) echo "usage: $0 {before|after}"; exit 2 ;;
esac
