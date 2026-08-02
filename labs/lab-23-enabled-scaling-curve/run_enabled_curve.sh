#!/usr/bin/env bash
# lab-23 — the ENABLED scaling curve, and what tuning is worth.
#
# WHY THIS LAB EXISTS
#   Every scaling curve in this guide (doc-15, lab-12) is measured on the
#   single-gVNIC TCP fabric: 465 -> 23.7 -> 14.95 GB/s across 8/16/24 GPUs.
#   lab-18 (TCPX, 83.27 GB/s) and lab-22 (TCPXO, 317.84 GB/s) then proved the
#   cliff is an architecture choice — but BOTH measured only ONE inter-node
#   point (2 nodes / 16 GPUs). So the guide currently has a 3-point curve on
#   the fabric nobody should use, and a 2-point curve on the fabric everyone
#   should. This lab measures the third point on the ENABLED fabric.
#
#   It also closes the guide's two standing "floor, not a ceiling" disclaimers
#   (lab-18 §"honesty note", lab-22 §5.1 item 3) — but NOT the way it set out to.
#   The intent was untuned-vs-tuned at the same size; the platform refuses (14
#   POLICY_ENFORCED NCCL vars abort init, G34). So the disclaimer is closed by
#   proving there is no env-tuning headroom, and the one knob left outside the
#   policy file (NCCL_ALGO) is measured instead. See §4 of the README.
#
# WHAT IT DOES NOT DO
#   It does not re-litigate whether the fabric is engaged. Layer 8 (NET/FasTrak)
#   is asserted BEFORE any sweep runs and the script ABORTS if the transport is
#   sockets — an unnoticed fallback would silently turn this into a rerun of
#   lab-12 and quietly corrupt the curve. See doc-25 §3 for why config checks
#   alone are insufficient.
#
# FABRIC / FLEX SAFETY
#   Needs all 24 GPUs, so it takes the whole cluster: holder 3->0, three TCPXO
#   workbenches occupy the nodes, EXIT trap restores holder to 3. No node is
#   ever drained, cordoned or deleted (Flex-start pools do not come back).
#   Per the TCPXO pod contract (manifests/tcpxo/workbench-tcpxo.yaml) the pods
#   use nodeSelector + podAntiAffinity and NEVER nodeName — nodeName makes
#   kubelet reject a GPU Pod outright (doc-25 §4.11) instead of queueing it,
#   which is exactly what breaks a gap-free handover from the holder.
#
# USAGE
#   bash labs/lab-23-enabled-scaling-curve/run_enabled_curve.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../../scripts/lib_capture.sh
source "${REPO_ROOT}/scripts/lib_capture.sh"

CLUSTER="${CLUSTER:-hypercomputer-a3-tcpxo}"
ZONE="${ZONE:-asia-southeast1-c}"
OUT="${ASSETS}/lab-23"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"
PORT="${PORT:-29520}"
HOLDER="${HOLDER:-gpu-holder-tcpxo}"
PODS=(tcpxo-wb-0 tcpxo-wb-1 tcpxo-wb-2)
ACCEL_LABEL="cloud.google.com/gke-accelerator=nvidia-h100-mega-80gb"
mkdir -p "$OUT"

# Resolve the cluster to a context ONCE, then pin every kubectl call to it.
# This is checker bug #3 from lab-22 §4.3, and it cost a report that spliced two
# clusters together: a target parameter that only steers SOME backends will
# describe the wrong system with total confidence. Fail loudly if unresolvable.
CTX="$(command kubectl config get-contexts -o name 2>/dev/null \
        | grep -E "_${ZONE}_${CLUSTER}$" | head -1)"
if [ -z "$CTX" ]; then
  echo "[lab-23] FATAL: no kube context for ${CLUSTER}/${ZONE}." >&2
  echo "  gcloud container clusters get-credentials ${CLUSTER} --location=${ZONE}" >&2
  exit 2
fi
export KUBE_CONTEXT="$CTX"
_kubectl_ctx_args=(--context "$CTX")   # re-arm lib_capture's wrapper post-source
echo "[lab-23] context: ${CTX}"

# --- Enumerate the 3 A3-Mega nodes (stable order) ----------------------------
mapfile -t NODES < <(kubectl get nodes -l "$ACCEL_LABEL" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort)
if [ "${#NODES[@]}" -lt 3 ]; then
  echo "[lab-23] FATAL: need 3 A3-Mega nodes, found ${#NODES[@]}." >&2
  echo "  A Flex-start pool that scaled to 0 takes its NICs and logs with it." >&2
  exit 1
fi
echo "[lab-23] nodes: ${NODES[*]}"

# --- Safety net: restore the holder on ANY exit path -------------------------
cleanup() {
  echo "[lab-23] cleanup: deleting workbenches, restoring ${HOLDER} to 3"
  kubectl delete pod "${PODS[@]}" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy "$HOLDER" --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

# --- Preflight: the fabric prerequisites must already be in place ------------
# All three are node-level DaemonSets; a missing one fails OPEN (doc-25 §4.10),
# so check before borrowing rather than debugging a dead fabric mid-window.
echo "[lab-23] preflight: installer + device-injector readiness"
for ds in nccl-tcpxo-installer device-injector; do
  ready="$(kubectl get ds "$ds" -n kube-system \
            -o jsonpath='{.status.numberReady}' 2>/dev/null || echo 0)"
  if [ "${ready:-0}" -lt 3 ]; then
    echo "[lab-23] FATAL: DaemonSet ${ds} ready=${ready:-0}, need 3." >&2
    echo "  kubectl apply -f manifests/tcpxo/  (see lab-22 §2)" >&2
    exit 1
  fi
  echo "[lab-23]   ${ds}: ${ready}/3 ready"
done

# --- Hold handoff -----------------------------------------------------------
echo "[lab-23] scaling ${HOLDER} 3->0 to free all 24 GPUs"
kubectl scale deploy "$HOLDER" --replicas=0
kubectl wait --for=delete pod -l app=gpu-holder-tcpxo --timeout=240s || true

# Three copies of the PROVEN 2-rank workbench spec (manifests/tcpxo/
# workbench-tcpxo.yaml), differing only in name. The antiAffinity on
# app=tcpxo-workbench is what spreads them one-per-node.
gen_workbench() { # gen_workbench <pod-name>
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $1
  labels: {app: tcpxo-workbench}
  annotations:
    devices.gke.io/container.tcpxo-daemon: |+
      - path: /dev/nvidia0
      - path: /dev/nvidia1
      - path: /dev/nvidia2
      - path: /dev/nvidia3
      - path: /dev/nvidia4
      - path: /dev/nvidia5
      - path: /dev/nvidia6
      - path: /dev/nvidia7
      - path: /dev/nvidiactl
      - path: /dev/nvidia-uvm
      - path: /dev/dmabuf_import_helper
    networking.gke.io/default-interface: 'eth0'
    networking.gke.io/interfaces: |
      [
        {"interfaceName":"eth0","network":"default"},
        {"interfaceName":"eth1","network":"gpu-net-0"},
        {"interfaceName":"eth2","network":"gpu-net-1"},
        {"interfaceName":"eth3","network":"gpu-net-2"},
        {"interfaceName":"eth4","network":"gpu-net-3"},
        {"interfaceName":"eth5","network":"gpu-net-4"},
        {"interfaceName":"eth6","network":"gpu-net-5"},
        {"interfaceName":"eth7","network":"gpu-net-6"},
        {"interfaceName":"eth8","network":"gpu-net-7"}
      ]
spec:
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-h100-mega-80gb
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels: {app: tcpxo-workbench}
        topologyKey: kubernetes.io/hostname
  restartPolicy: Never
  tolerations:
  - operator: Exists
  volumes:
  - name: nvidia
    hostPath: {path: /home/kubernetes/bin/nvidia/lib64}
  - name: shared-memory
    emptyDir: {medium: Memory, sizeLimit: 16Gi}
  - name: sys
    hostPath: {path: /sys}
  - name: proc-sys
    hostPath: {path: /proc/sys}
  - name: aperture-devices
    hostPath: {path: /dev/aperture_devices}
  - name: workdir
    emptyDir: {}
  containers:
  - name: tcpxo-daemon
    image: us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/tcpgpudmarxd-dev:v1.0.22
    imagePullPolicy: Always
    command: ["/bin/sh", "-c"]
    args:
    - |
      set -ex
      chmod 755 /fts/entrypoint_rxdm_container.sh
      /fts/entrypoint_rxdm_container.sh --num_hops=2 --num_nics=8 --uid= --alsologtostderr
    securityContext:
      capabilities:
        add: ["NET_ADMIN","NET_BIND_SERVICE"]
    volumeMounts:
    - {name: nvidia, mountPath: /usr/local/nvidia/lib64}
    - {name: sys, mountPath: /hostsysfs}
    - {name: proc-sys, mountPath: /hostprocsysfs}
    env:
    - {name: LD_LIBRARY_PATH, value: /usr/local/nvidia/lib64}
  - name: bench
    image: nvcr.io/nvidia/pytorch:24.10-py3
    imagePullPolicy: IfNotPresent
    securityContext:
      capabilities: {add: ["IPC_LOCK"]}
    resources:
      limits: {nvidia.com/gpu: 8}
    volumeMounts:
    - {name: nvidia, mountPath: /usr/local/nvidia/lib64}
    - {name: shared-memory, mountPath: /dev/shm}
    - {name: aperture-devices, mountPath: /dev/aperture_devices}
    - {name: workdir, mountPath: /work}
    env:
    - {name: LD_LIBRARY_PATH, value: /usr/local/nvidia/lib64}
    - {name: NCCL_LIB_DIR, value: /usr/local/nvidia/lib64}
    - {name: NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY, value: /dev/aperture_devices}
    command: ["/bin/bash","-c","sleep infinity"]
EOF
}

for P in "${PODS[@]}"; do gen_workbench "$P"; done
echo "[lab-23] waiting for 3 workbenches (2 containers each) to be Ready..."
kubectl wait --for=condition=Ready pod "${PODS[@]}" --timeout=600s

# Stage the UNMODIFIED lab-06 harness — same bench as the gVNIC curve, so the
# enabled numbers are apples-to-apples with doc-15/lab-12 by construction.
#
# Staged into /workspace, NOT the /work emptyDir: launch_node.sh hardcodes
# `/workspace/w_<rank>.log` for its per-rank logs (line 30). Staging elsewhere
# runs the benchmark fine but writes every log to a directory that does not
# exist, so the sweep "succeeds" with zero recoverable output (G31). Keeping the
# launcher unmodified is deliberate — it is the same file lab-06/12 measured
# with, and forking it would break the apples-to-apples claim.
for P in "${PODS[@]}"; do
  kubectl exec "$P" -c bench -- bash -lc 'mkdir -p /workspace'
  kubectl cp "${LAB06}/allreduce_bench.py" "${P}:/workspace/allreduce_bench.py" -c bench
  kubectl cp "${LAB06}/launch_node.sh"     "${P}:/workspace/launch_node.sh"     -c bench
done

# --- The env that engages the fabric ----------------------------------------
# Source the vendor profile; never hand-write NCCL_FASTRAK_IFNAME (lab-22 §5).
# On TCPXO the profile discovers eth1..eth8 on the node it runs on.
PROFILE='source /usr/local/nvidia/lib64/nccl-env-profile.sh'
BASE_ENV='NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET BENCH_MAX_EXP=31 BENCH_ITERS=15'

# run_sweep <nnodes> <tag> [extra_env]
#   Writes assets/lab-23/allreduce_<tag>.txt (+ _full.log). Extra env is applied
#   AFTER the profile so a tuning knob can override a profile default.
run_sweep() {
  local nnodes="$1" tag="$2" extra="${3:-}"
  local master_ip node_r pod pids=() rc=0
  # MUST be eth0's address, not `hostname -I | awk '{print $1}'`. On a 9-NIC
  # TCPXO pod `hostname -I` lists the GPU RAIL addresses first (192.169.0.x),
  # and the rails are isolated per-rail /24s with no route between nodes — so
  # rank 1 hangs dialing a c10d store it can never reach, then every rank dies
  # at init with no fabric error anywhere. The control plane is eth0; only the
  # NCCL data path rides eth1-8 (G32). Read eth0 explicitly.
  #
  # Resolved via python3's ioctl rather than `ip`/`ifconfig`: neither binary
  # exists in nvcr.io/nvidia/pytorch (G33), and a plain `command -v ip` guard
  # would have turned this into a silent fallback to the wrong NIC.
  master_ip="$(kubectl exec "${PODS[0]}" -c bench -- python3 -c \
"import fcntl,socket,struct
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'eth0'))[20:24]))")"
  if [ -z "$master_ip" ]; then
    echo "[lab-23] FATAL: could not resolve eth0 address on ${PODS[0]}" >&2
    return 1
  fi
  echo "[lab-23] === ${tag}: ${nnodes} node(s) / $((nnodes*8)) GPUs, master=${master_ip} ==="
  for node_r in $(seq 0 $((nnodes - 1))); do
    pod="${PODS[$node_r]}"
    kubectl exec "$pod" -c bench -- bash -lc \
      "cd /workspace && ${PROFILE} && NNODES=${nnodes} ${BASE_ENV} ${extra} \
       bash launch_node.sh ${node_r} ${master_ip} ${PORT} 8" &
    pids+=($!)
    # rank0 must stand up the c10d store before the others dial in. Written as a
    # full if/fi, NOT `[ x ] && sleep 8`: that form returns 1 on the non-rank0
    # iterations, and as the last statement in the loop body it aborts the whole
    # script under the `set -e` inherited from lib_capture.sh (G1).
    if [ "$node_r" -eq 0 ]; then sleep 8; fi
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done

  kubectl exec "${PODS[0]}" -c bench -- bash -lc 'cat /workspace/w_0.log' \
    > "${OUT}/allreduce_${tag}_full.log" 2>/dev/null || true
  grep -E '^#|^\s+[0-9]' "${OUT}/allreduce_${tag}_full.log" \
    > "${OUT}/allreduce_${tag}.txt" || true
  # A sweep that produced no data rows is a FAILED sweep, not a slow one. Say so
  # here — otherwise it silently becomes a missing row in the curve CSV, which
  # reads as "not measured" rather than "broken".
  if ! grep -qE '^\s+[0-9]' "${OUT}/allreduce_${tag}.txt" 2>/dev/null; then
    echo "[lab-23] WARNING: ${tag} produced NO data rows (rc=${rc}) — see ${tag}_full.log" >&2
  fi
  kubectl exec "${PODS[0]}" -c bench -- bash -lc 'rm -f /workspace/w_*.log' || true
  echo "[lab-23] ${tag} rc=${rc}"
  return 0
}

# --- LAYER 8 GATE: prove FasTrak before trusting a single number ------------
# A 2-node sweep whose transport is NET/Socket is not a slow enabled fabric, it
# is a silent fallback (doc-25 §3) — and it would look like a plausible curve.
# Assert first, abort loudly, never publish an unverified rung.
echo "[lab-23] layer-8 gate: asserting NET/FasTrak on a 16-GPU probe"
run_sweep 2 "gate_probe"
GATE="${OUT}/allreduce_gate_probe_full.log"
grep -iE 'Using network|NET/FasTrak|NET/Socket|GPU Direct' "$GATE" \
  | head -40 > "${OUT}/transport_gate.txt" || true
sock="$(grep -c 'NET/Socket' "$GATE" 2>/dev/null || true)"; sock="${sock:-0}"
fast="$(grep -ciE 'FasTrak' "$GATE" 2>/dev/null || true)"; fast="${fast:-0}"
echo "[lab-23]   FasTrak lines=${fast}  NET/Socket lines=${sock}"
if [ "$fast" -eq 0 ] || [ "$sock" -ne 0 ]; then
  echo "[lab-23] FATAL: fabric is NOT engaged (FasTrak=${fast}, Socket=${sock})." >&2
  echo "  This would silently reproduce lab-12's TCP curve. See doc-25 §3/§4." >&2
  echo "  Evidence: ${OUT}/transport_gate.txt" >&2
  exit 1
fi
echo "[lab-23] layer-8 gate PASS — fabric engaged, numbers are meaningful"

# --- 1. The enabled curve: 8 / 16 / 24 GPUs ---------------------------------
# SKIP_CURVE=1 reuses already-captured curve assets and re-runs only the tuning
# comparison — the borrow window is a scarce Flex resource, so don't re-measure
# three good rungs to retry a fourth.
if [ "${SKIP_CURVE:-0}" != "1" ]; then
  run_sweep 1 "8gpu"    # intra-node NVLink ceiling (no fabric involved)
  run_sweep 2 "16gpu"   # one inter-node hop — comparable to lab-22's 317.84
  run_sweep 3 "24gpu"   # THE NEW POINT: ring crosses three node boundaries
else
  echo "[lab-23] SKIP_CURVE=1 — reusing existing 8/16/24-GPU assets"
fi

# --- 2. What tuning is worth (same 16-GPU config, tuned) --------------------
# Closes the standing disclaimers in lab-18 and lab-22 §5.1. Reported as a delta
# against the 16gpu run above, never as an absolute claim.
#
# The env is NOT tunable, and that is the finding. The plugin loads a shim
# ("Guest Config Checker", a3plus_guest_config.textproto) that compares 14
# variables against vendor-expected values and *refuses to initialize* on any
# mismatch, aborting before a single collective runs. Both obvious tuning
# attempts died there (G34):
#   NCCL_FASTRAK_NUM_FLOWS=4 -> "mismatch enforced ... (expected 2)"
#   NCCL_MIN_NCHANNELS=8     -> "mismatch enforced ... (expected 4)"
# Evidence: failure_shim_enforced_num_flows.txt,
#           failure_shim_enforced_nchannels.txt,
#           guest_config_enforced.txt (the policy file itself).
# ENFORCED covers the whole throughput surface — NCCL_PROTO, NCCL_BUFFSIZE,
# NCCL_MIN_NCHANNELS, NCCL_CROSS_NIC, NET_GDR_LEVEL, all four *CHUNKSIZE vars,
# and six NCCL_FASTRAK_*. The vendor profile is not a baseline to improve on;
# it is a contract. So "317.84 is an untuned floor" was the wrong frame for
# TCPXO: there is no env-tuning headroom to find.
#
# What IS left is algorithm selection: NCCL_ALGO is absent from the policy file.
# That makes it the one guide claim worth re-testing here — lab-12 found Tree
# beat Ring at every size at 3 nodes on the TCP fabric (~11x mid-range), and
# doc-15 concluded "NCCL's default under-picks". Whether that survives on a
# fabric 12x faster is a real question, and it is the honest replacement for a
# tuning sweep that the platform forbids.
TUNED_ENV="${TUNED_ENV:-NCCL_ALGO=Tree}"
run_sweep 3 "24gpu_tree" "$TUNED_ENV"

# --- Transport + rail evidence for the NEW 24-GPU rung ----------------------
grep -iE 'NET/IB|NET/Socket|NET/FasTrak|Using network|GPU Direct|NCCL version|Channel|Trees|Rings|via NET' \
  "${OUT}/allreduce_24gpu_full.log" | head -60 > "${OUT}/transport_24gpu.txt" || true

# Rail balance across all 8 NICs at 3 nodes: an uneven spread is doc-25 §4.7
# (throughput ~1/N of expected) and would explain a disappointing 24-GPU point.
for r in 1 2 3 4 5 6 7 8; do
  printf 'eth%s %s\n' "$r" \
    "$(grep -c "eth${r}" "${OUT}/allreduce_24gpu_full.log" 2>/dev/null || echo 0)"
done > "${OUT}/rail_balance_24gpu.txt"

# --- The curve, and the enabled-vs-gVNIC comparison -------------------------
# gVNIC reference values are lab-12's measured peaks (assets/lab-12/
# scaling_curve.csv), quoted here so the CSV is self-describing.
{
  echo "gpus,nodes,peak_busbw_GBps,latency_floor_ms,gvnic_peak_busbw_GBps,speedup_x"
  for spec in "8gpu 1 465.43" "16gpu 2 23.70" "24gpu 3 14.95"; do
    set -- $spec; tag="$1"; nn="$2"; gv="$3"
    f="${OUT}/allreduce_${tag}.txt"; [ -s "$f" ] || continue
    peak=$(awk '/^ / {if ($5+0>m) m=$5} END{printf "%.2f", m+0}' "$f")
    floor=$(awk '/^ / {print $3; exit}' "$f")
    sp=$(awk -v p="$peak" -v g="$gv" 'BEGIN{if(g>0) printf "%.2f", p/g; else print "NA"}')
    echo "$((nn*8)),${nn},${peak},${floor},${gv},${sp}"
  done
} > "${OUT}/enabled_scaling_curve.csv"

if [ -s "${OUT}/allreduce_24gpu_tree.txt" ]; then
  {
    echo "config,gpus,algo,peak_busbw_GBps"
    printf 'default,24,ring(auto),%s\n' \
      "$(awk '/^ / {if ($5+0>m) m=$5} END{printf "%.2f", m+0}' "${OUT}/allreduce_24gpu.txt")"
    printf 'NCCL_ALGO=Tree,24,tree,%s\n' \
      "$(awk '/^ / {if ($5+0>m) m=$5} END{printf "%.2f", m+0}' "${OUT}/allreduce_24gpu_tree.txt")"
  } > "${OUT}/algo_delta.csv"
else
  echo "[lab-23] NOTE: no NCCL_ALGO=Tree sweep captured — algo_delta.csv not written." >&2
  echo "  The curve above stands on its own." >&2
fi

echo "[lab-23] enabled_scaling_curve.csv:"; cat "${OUT}/enabled_scaling_curve.csv"
if [ -s "${OUT}/algo_delta.csv" ]; then
  echo "[lab-23] algo_delta.csv:"; cat "${OUT}/algo_delta.csv"
fi

cap_verify_provenance "lab-23" "assets/lab-23/enabled_scaling_curve.csv" \
  "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "TCPXO ENABLED 8/16/24-GPU scaling curve (layer-8 gated on NET/FasTrak) + 24-GPU ring-vs-tree delta; env tuning proven impossible (14 POLICY_ENFORCED vars, G34); same lab-06 harness as the gVNIC curve"

echo "[lab-23] done. Assets in ${OUT}/ (holder restored by EXIT trap)"
