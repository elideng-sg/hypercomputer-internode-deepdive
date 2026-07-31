#!/usr/bin/env bash
# run_tcpx_beforeafter.sh — the lab-18 "close the cliff" capture: gVNIC BEFORE → TCPX AFTER.
#
# STATUS: BOTH halves are captured live (2026-07-31).
#   BEFORE — reused from labs/lab-06 (2-node) + labs/lab-12 (8/16/24-GPU curve): NET/Socket,
#            23.70 GB/s @512MB. Not re-run; this script only writes the evidence pointer.
#   AFTER  — captured on hypercomputer-a3-tcpx / asia-east1-c, 2x a3-highgpu-8g (16x H100):
#            Using network GPUDirectTCPX_v7, 0x NET/Socket, 83.27 GB/s busbw peak @2GB.
#
# WHAT THIS SCRIPT DOES vs WHAT IT DOESN'T. The `after` phase is a CAPTURE step, not a
# provisioning or launch step. The benchmark itself is a 2-rank torchrun that must be started
# by hand inside both Pods (they're a Pod PAIR, not a Job — see PREREQUISITES). Trying to
# wrap the launch here would hide the one thing a reader needs to see: the rendezvous.
#
# PREREQUISITES for `after` — in order:
#   1. kubectl apply -f manifests/tcpx/network-crds.yaml
#   2. kubectl apply -f manifests/tcpx/nccl-tcpx-installer.yaml
#      Wait for 2/2 READY. If it sticks at 0 ready, check WHICH container is stuck before
#      touching labels: the docs' pause:3.9 image 404s (G27), and the plugin may already be
#      installed. `bash scripts/verify_gpu_fabric.sh` now tells these apart (G30).
#   3. Scale the capacity holder DOWN so the Flex GPUs free up:
#        kubectl scale deploy gpu-holder-tcpx --replicas=0
#   4. Apply the workbench PAIR (rank 0 + rank 1; podAntiAffinity splits them across nodes):
#        kubectl apply -f manifests/tcpx/workbench-tcpx.yaml            # tcpx-wb-0
#        sed 's/tcpx-wb-0/tcpx-wb-1/' manifests/tcpx/workbench-tcpx.yaml | kubectl apply -f -
#      Wait for BOTH 2/2 Running.
#   5. Launch the benchmark. Copy the SHARED harness into both Pods so the numbers stay
#      comparable with lab-06/lab-12, then start each rank (rank 0's IP is the master):
#        M=$(kubectl get pod tcpx-wb-0 -o jsonpath='{.status.podIP}')
#        for p in tcpx-wb-0 tcpx-wb-1; do
#          kubectl cp labs/lab-06-2node-nccl-collectives/allreduce_bench.py $p:/work/ -c bench
#        done
#        # in EACH pod (R=0 on tcpx-wb-0, R=1 on tcpx-wb-1), from /work:
#        #   BENCH_MAX_EXP=31 BENCH_ITERS=20 BENCH_WARMUP=5 \
#        #   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$R \
#        #     --master_addr=$M --master_port=29600 allreduce_bench.py 2>&1 \
#        #     | tee /proc/1/fd/1 > /work/run.log
#        # `tee /proc/1/fd/1` matters: it mirrors into the container log so the evidence
#        # survives the exec session dying (lab-22's rule).
#   6. THEN run:  KUBE_CONTEXT=<tcpx ctx> bash $0 after
#
# HOLDER DISCIPLINE (standing rule: never leave a DWS GPU node idle). When done capturing,
# hand the GPUs BACK gap-free — scale the holder UP first so its Pods are already Pending,
# and only THEN delete the workbench Pods:
#     kubectl scale deploy gpu-holder-tcpx --replicas=2
#     kubectl delete pod tcpx-wb-0 tcpx-wb-1
# Measured on 2026-07-31: the holder claimed both nodes within 30 s.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/scripts/lib_capture.sh"
OUT="$REPO/assets/lab-18"; mkdir -p "$OUT"

RANK0="${RANK0:-tcpx-wb-0}"
RANK1="${RANK1:-tcpx-wb-1}"
RUNLOG="${RUNLOG:-/work/run.log}"

before(){
  # Reuse the already-captured gVNIC baseline — do NOT re-run; just point at the evidence.
  {
    echo "# BEFORE = single-gVNIC / TCP (captured live, labs 06 & 12)"
    echo "# transport (lab-06): NET/IB No device found -> NET/Socket eth0 -> Using network Socket"
    echo "# 2-node 16-GPU busbw : ~28.6 GB/s peak (assets/lab-06/allreduce_2node.txt)"
    echo "# 8/16/24-GPU busbw   : 465 -> 23.7 -> 14.95 GB/s (assets/lab-12/allreduce_*.txt)"
    echo "# same-size row used for the lab-18 comparison: 512MB = 23.70 GB/s"
    echo "#                       (assets/lab-12/allreduce_16gpu.txt)"
    echo "# GPU Direct RDMA     : Disabled for HCA 0 'eth0'"
  } > "$OUT/before_gvnic_summary.txt"
  echo "wrote $OUT/before_gvnic_summary.txt (evidence: assets/lab-06, assets/lab-12)"
}

after(){
  : "${KUBE_CONTEXT:?set KUBE_CONTEXT to the TCPX cluster context}"
  # Fail early and loudly rather than writing half-empty assets: the Pods must still be
  # ALIVE. On a Flex pool the node takes its NIC list, plugin libs and NCCL logs with it
  # when it goes away — there is no back-filling this later (lab-22's capture rule).
  for p in "$RANK0" "$RANK1"; do
    kubectl get pod "$p" >/dev/null 2>&1 || {
      echo "ERROR: pod $p not found in context $KUBE_CONTEXT." >&2
      echo "       The AFTER capture reads from the LIVE Pods; see PREREQUISITES in this file's header." >&2
      exit 1; }
  done
  kubectl exec "$RANK0" -c bench -- test -s "$RUNLOG" 2>/dev/null || {
    echo "ERROR: $RUNLOG is missing/empty in $RANK0 — has the benchmark been launched (step 5)?" >&2
    exit 1; }

  # 1) TRANSPORT — the decisive read. Which net plugin did NCCL actually load, and did the
  #    socket fallback happen? Counting NET/Socket is the point: it must be 0.
  cap_run lab-18 after_tcpx_transport.txt -- \
    kubectl exec "$RANK0" -c bench -- bash -c "
      echo '## transport lines by kind:'
      grep -ohE 'NET/(GPUDirectTCPX|Socket)|Using network [A-Za-z0-9_]+' $RUNLOG | sort | uniq -c | sort -rn
      echo; echo '## the plugin banner + version:'
      grep -m1 'Using network GPUDirectTCPX' $RUNLOG
      echo; echo '## rails actually carrying traffic (flow-steer + connect lines):'
      grep -ohE '192\.168\.[0-3]' $RUNLOG | sort | uniq -c
      echo; echo '## registration errors (MUST be zero — see G29):'
      grep -c -E 'gpu_tx_reg_mr|dma_buf frags|p2pdma' $RUNLOG"

  # 2) BUSBW SWEEP — same harness/message sizes as lab-06 and lab-12 for a fair before/after.
  cap_run lab-18 after_tcpx_allreduce.txt -- \
    kubectl exec "$RANK0" -c bench -- bash -c \
      "awk '/algbw_GBps/,/# done/' $RUNLOG"

  # 3) IN-POD FABRIC — 4 rails at jumbo MTU, the rail IPs, the installed plugin libs, and the
  #    two TIER-SPECIFIC absences that catch people out: no env-profile script (G28) and no
  #    /dev/aperture_devices (TCPXO-only). The pytorch image has no iproute2, so use sysfs.
  cap_run lab-18 after_tcpx_inpod_fabric.txt -- \
    kubectl exec "$RANK0" -c bench -- bash -c '
      echo "## NICs and MTU inside the Pod netns (sysfs; no iproute2 in this image):"
      for i in /sys/class/net/eth*; do n=$(basename $i); echo "  $n mtu=$(cat $i/mtu) mac=$(cat $i/address)"; done
      echo; echo "## NCCL plugin libraries installed by the DaemonSet:"
      ls -l /usr/local/nvidia/lib64/libnccl* 2>/dev/null | sed "s|/usr/local/nvidia/lib64/||"
      echo; echo "## env-profile script on this tier (G28 — TCPXO ships one, TCPX does NOT):"
      echo "  matches: $(ls /usr/local/nvidia/lib64/*env* 2>/dev/null | wc -l)"
      echo; echo "## /dev/aperture_devices (TCPXO-only; absence is CORRECT for TCPX):"
      [ -e /dev/aperture_devices ] && echo "  PRESENT" || echo "  ABSENT (expected for TCPX)"'

  # 4) MONITORING — does the traffic show up? Unlike TCPXO, on TCPX the netdev counters DO
  #    see it, which makes them a valid rail-balance check on this tier (narrows G25).
  cap_run lab-18 after_tcpx_monitoring.txt -- \
    kubectl exec "$RANK0" -c bench -- python3 -c '
import os
t=r=0
for i in ["eth0","eth1","eth2","eth3","eth4"]:
    p=f"/sys/class/net/{i}/statistics/"
    if not os.path.exists(p): continue
    a=int(open(p+"tx_bytes").read()); b=int(open(p+"rx_bytes").read())
    if i!="eth0": t+=a; r+=b
    print("  %-5s tx=%15d (%8.2f GB)  rx=%15d (%8.2f GB)"%(i,a,a/1e9,b,b/1e9))
print("  GPU-rail totals: tx=%.2f GB rx=%.2f GB  (balance across rails is the signal)"%(t/1e9,r/1e9))'

  cap_verify_provenance "lab-18" "after_tcpx_allreduce.txt" \
    "$(kubectl get pod "$RANK0" "$RANK1" -o jsonpath='{range .items[*]}{.spec.nodeName}{" "}{end}')" \
    "TCPX enabled: Using network GPUDirectTCPX_v7, 0x NET/Socket, 83.27 GB/s busbw peak @2GB (16 GPUs)"
}

case "${1:-before}" in
  before) before ;;
  after)  after ;;
  *) echo "usage: $0 {before|after}"; exit 2 ;;
esac
