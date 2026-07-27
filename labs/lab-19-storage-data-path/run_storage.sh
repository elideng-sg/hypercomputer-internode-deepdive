#!/usr/bin/env bash
# Lab 19 — storage & the data path (Flex-safe, single-node borrow).
#
# WHY THIS LAB: every training lab so far fed the GPUs from synthetic in-memory
# tensors. Real jobs read from a bucket. This lab makes the DATA PATH visible as
# a GPU signal: mount a GCS bucket with GCSFuse and run the SAME fixed compute
# step two ways —
#   STARVED : read shards off the bucket every step  → GPU idles waiting on I/O
#   FED     : bytes already resident (compute-bound)  → GPU stays busy
# then read the GCSFuse sequential-read throughput (the ceiling), write a
# checkpoint back to the bucket (the write half), and cross-check the GPU-busy
# signal on the monitoring pipeline (GMP/DCGM) exactly as lab-17 did — so
# "training is slow" is diagnosed as a STORAGE symptom, not a GPU one.
#
# AUTH / why userspace GCSFuse (gotcha G20): the managed GCSFuse CSI driver needs
# Workload Identity at the NODE level (workloadMetadataConfig=GKE_METADATA), and
# switching that on this pre-WIF pool RECREATES the nodes — unacceptable for scarce
# Flex A3 capacity. So this lab mounts GCSFuse in USERSPACE inside a privileged pod,
# authenticated by a GSA key Secret — zero node changes, real numbers, same data
# path. The managed CSI+WIF manifest ships as the production reference in
# manifests/storage/gcsfuse-workbench.yaml.
#
# GPU SAFETY / Flex-safe: scale gpu-holder 3->2 (frees ONE node's 8 GPUs), occupy
# that node with a workbench requesting all 8 GPUs (node stays FULLY held — the
# bench drives only cuda:0). The EXIT trap deletes the workbench and restores
# gpu-holder=3. No device-state change, no node drain/cordon/delete.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
set +e   # phases run bounded / best-effort; manage errors explicitly (gotcha G1)

LAB="lab-19"
OUT="${ASSETS}/${LAB}"; mkdir -p "$OUT"
HERE="${REPO_ROOT}/labs/lab-19-storage-data-path"

POD="${LAB19_POD:-gcs-data-wb}"
IMAGE="${LAB19_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
SECRET="${LAB19_SECRET:-lab19-gcs-key}"     # Secret holding key.json for lab19-gcs GSA
BUCKET="${LAB19_BUCKET:-hdlab-elideng-lab-data-asiaeast1}"
PROJECT="${LAB19_PROJECT:-hdlab-elideng}"
GMP="https://monitoring.googleapis.com/v1/projects/${PROJECT}/location/global/prometheus/api/v1"

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-19] FATAL: need the 3-node pool, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-19] pool nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-19] cleanup: free workbench, restore gpu-holder=3"
  kubectl delete pod "$POD" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

X() { kubectl exec "$POD" -c bench -- bash -lc "$1"; }   # in-pod shell

# --- borrow ONE node: holder 3->2, find the node with no holder pod ----------
echo "[lab-19] scaling gpu-holder 3->2 (frees one node's 8 GPUs)"
kubectl scale deploy gpu-holder --replicas=2
for _ in $(seq 1 60); do
  n=$(kubectl get pods -l app=gpu-holder -o name 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = "2" ] && break; sleep 3
done
held="$(kubectl get pods -l app=gpu-holder -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)"
FREE_NODE=""
for node in "${NODES[@]}"; do grep -qx "$node" <<<"$held" || { FREE_NODE="$node"; break; }; done
[ -n "$FREE_NODE" ] || { echo "[lab-19] FATAL: could not identify a freed node (held=[$held])" >&2; exit 1; }
FREE_SHORT="${FREE_NODE##*-}"
echo "[lab-19] borrowing free node: $FREE_NODE (short=${FREE_SHORT})"

# --- workbench: 8 GPUs (node fully held) + key Secret + /dev/fuse ------------
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: ${POD}, labels: { app: gcs-data-wb } }
spec:
  nodeName: ${FREE_NODE}
  restartPolicy: Never
  tolerations: [{ operator: Exists }]
  containers:
  - name: bench
    image: ${IMAGE}
    command: ["sleep","infinity"]
    securityContext: { privileged: true }         # userspace FUSE mount needs /dev/fuse + CAP_SYS_ADMIN
    resources: { limits: { nvidia.com/gpu: "8" } }
    env:
    - { name: GOOGLE_APPLICATION_CREDENTIALS, value: /var/secrets/key.json }
    volumeMounts:
    - { name: key,  mountPath: /var/secrets, readOnly: true }
    - { name: fuse, mountPath: /dev/fuse }
    - { name: dshm, mountPath: /dev/shm }
  volumes:
  - { name: key,  secret: { secretName: ${SECRET} } }
  - { name: fuse, hostPath: { path: /dev/fuse } }
  - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
EOF
kubectl wait --for=condition=Ready "pod/${POD}" --timeout=300s \
  || { echo "[lab-19] FATAL: pod not ready" >&2; kubectl describe pod "$POD" | tail -30; exit 1; }

# --- install userspace gcsfuse + mount the bucket at /data -------------------
echo "[lab-19] installing userspace gcsfuse + mounting gs://${BUCKET} at /data"
X "set -e; export DEBIAN_FRONTEND=noninteractive
   apt-get update -qq >/dev/null 2>&1
   apt-get install -y -qq lsb-release curl gnupg fuse >/dev/null 2>&1
   echo \"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-\$(lsb_release -c -s) main\" > /etc/apt/sources.list.d/gcsfuse.list
   curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null
   apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq gcsfuse >/dev/null 2>&1
   gcsfuse --version; mkdir -p /data
   gcsfuse --implicit-dirs ${BUCKET} /data
   pip -q install pynvml >/dev/null 2>&1 || true" \
  || { echo "[lab-19] FATAL: gcsfuse mount failed" >&2; exit 1; }
kubectl cp "${HERE}/dataloader_bench.py" "${POD}:/workspace/dataloader_bench.py" -c bench

# --- Phase 0: mount proof ----------------------------------------------------
cap_run "$LAB" "mount_proof.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'echo "## fuse mount:"; mount | grep -i fuse; echo "## shards:"; ls /data/shards/shard_*.bin | wc -l'

# --- Phase A: GCSFuse sequential read throughput -----------------------------
echo "[lab-19] === Phase A: GCSFuse sequential read throughput ==="
cap_run "$LAB" "gcsfuse_read_throughput.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'BYTES=$(cat /data/shards/shard_*.bin | wc -c); t0=$(date +%s.%N); cat /data/shards/shard_*.bin > /dev/null; t1=$(date +%s.%N);
   python3 - "$BYTES" "$t0" "$t1" <<PY
import sys; b=int(sys.argv[1]); dt=float(sys.argv[3])-float(sys.argv[2])
print(f"read {b/2**30:.2f} GiB in {dt:.2f}s -> {b/2**20/dt:.1f} MiB/s sequential (GCSFuse userspace, fileCache off)")
PY'

# --- Phase B/C: the data-path contrast (identical compute, STARVED vs FED) ---
gmp_engine_range() { # <outfile> <start> <end> <tag>
  local tok; tok="$(gcloud auth print-access-token 2>/dev/null)"
  { echo "# ${4}  DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu=0} node …${FREE_SHORT}  [${2}..${3} step 15s]";
    curl -s -G "${GMP}/query_range" -H "Authorization: Bearer ${tok}" \
      --data-urlencode 'query=DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0"}' \
      --data-urlencode "start=${2}" --data-urlencode "end=${3}" --data-urlencode "step=15" \
    | python3 -c 'import json,sys,datetime
d=json.load(sys.stdin)
for s in d.get("data",{}).get("result",[]):
    h=s["metric"].get("Hostname","?")
    if "'"${FREE_SHORT}"'" not in h: continue
    out=[datetime.datetime.fromtimestamp(float(t),datetime.timezone.utc).strftime("%H:%M:%S")+"="+format(float(v),".3f") for t,v in s.get("values",[])]
    print("  "+h[-4:]+" gpu0: "+" ".join(out))'; } >> "${OUT}/${1}"
}
: > "${OUT}/dcgm_crosscheck.txt"

echo "[lab-19] === STARVED (identical compute + per-step shard reads) ==="
S0=$(date -u +%s)
cap_run "$LAB" "dataloader_starved.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'DATA_DIR=/data/shards MODE=starved STARVE_SHARDS=8 STEPS=200 python3 /workspace/dataloader_bench.py'
S1=$(date -u +%s); sleep 45; gmp_engine_range dcgm_crosscheck.txt "$S0" "$((S1+30))" "STARVED window"

echo "[lab-19] === FED (identical compute, bytes resident) ==="
F0=$(date -u +%s)
cap_run "$LAB" "dataloader_fed.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'DATA_DIR=/data/shards MODE=fed STEPS=1200 python3 /workspace/dataloader_bench.py'
F1=$(date -u +%s); sleep 45; gmp_engine_range dcgm_crosscheck.txt "$F0" "$((F1+30))" "FED window"

# --- Phase E: checkpoint WRITE back to the bucket ----------------------------
echo "[lab-19] === Phase E: checkpoint write to GCS via GCSFuse ==="
cap_run "$LAB" "checkpoint_write.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'python3 - <<PY
import torch, time, os
os.makedirs("/data/checkpoints", exist_ok=True)
sd={f"layer{i}.weight": torch.randn(4096,4096) for i in range(8)}   # ~0.5 GiB fp32
p="/data/checkpoints/ckpt_lab19.pt"; t0=time.time(); torch.save(sd,p); dt=time.time()-t0
sz=os.path.getsize(p); print(f"wrote {sz/2**30:.2f} GiB checkpoint in {dt:.2f}s -> {sz/2**20/dt:.1f} MiB/s (GCSFuse write path)")
PY'
cap_run "$LAB" "checkpoint_verify.txt" -- bash -lc "gcloud storage ls -l gs://${BUCKET}/checkpoints/ 2>&1"

# --- provenance --------------------------------------------------------------
cap_verify_provenance "$LAB" "assets/lab-19/gcsfuse_read_throughput.txt" "$FREE_NODE" "GCSFuse seq read MiB/s"
cap_verify_provenance "$LAB" "assets/lab-19/dataloader_starved.txt"      "$FREE_NODE" "starved data path (GPU idle on I/O)"
cap_verify_provenance "$LAB" "assets/lab-19/dataloader_fed.txt"          "$FREE_NODE" "fed data path (GPU compute-bound)"
cap_verify_provenance "$LAB" "assets/lab-19/checkpoint_write.txt"        "$FREE_NODE" "checkpoint write to GCS"

echo "[lab-19] DONE — assets in ${OUT}. (trap restores gpu-holder=3)"
