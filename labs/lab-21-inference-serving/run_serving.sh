#!/usr/bin/env bash
# Lab 21 — inference serving & autoscale (Flex-safe, single-node borrow).
#
# WHY THIS LAB: lab-20 showed TRAINING — throughput-bound, one long job. SERVING
# is the opposite workload: latency-bound, bursty, many small requests, and the
# architectural answer to bursty load is AUTOSCALING. This lab MEASURES the two
# facts that autoscaling is built on, on real H100s:
#   A. the SATURATION KNEE  — sweep client concurrency against a real GPU model
#      server (ResNet-50 + dynamic batching); throughput rises then plateaus while
#      latency climbs — the curve an autoscaler reacts to.
#   B. HORIZONTAL SCALING   — run the same inference on 1..8 GPUs; aggregate
#      throughput scales ~linearly, so N replicas ≈ N× QPS. This is the thing a
#      single GPU cannot show and the reason replica autoscaling works.
#   C. DCGM under load       — the GPU signal (engine-active) an HPA scales on,
#      read off GMP exactly as lab-17/19/20.
#
# The production autoscale topology (HPA-on-DCGM, cluster-autoscaler node scale-up,
# Inference Gateway, Vertex AI) ships as the REFERENCE manifest
# manifests/serving/inference-autoscale.yaml — NOT applied live, because the
# always-hold rule + Flex cap-of-3 leave zero spare GPU capacity to scale into
# (the same measured-rung / reference-rung split as lab-18 and lab-19).
#
# GPU SAFETY / Flex-safe: scale gpu-holder 3->2 (frees ONE node's 8 GPUs), occupy
# that node with a single pod requesting all 8 GPUs (node stays FULLY held). All
# serving work runs inside that one pod; the EXIT trap deletes it and restores
# gpu-holder=3. No device-state change, no node drain/cordon/delete.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
set +e

LAB="lab-21"
OUT="${ASSETS}/${LAB}"; mkdir -p "$OUT"
HERE="${REPO_ROOT}/labs/lab-21-inference-serving"
POD="${LAB21_POD:-infer-serve-wb}"
IMAGE="${LAB21_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
PROJECT="${LAB21_PROJECT:-hdlab-elideng}"
GMP="https://monitoring.googleapis.com/v1/projects/${PROJECT}/location/global/prometheus/api/v1"

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-21] FATAL: need the 3-node pool, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-21] pool nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-21] cleanup: free workbench, restore gpu-holder=3"
  kubectl delete pod "$POD" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

X() { kubectl exec "$POD" -c bench -- bash -lc "$1"; }

# --- borrow ONE node ---------------------------------------------------------
echo "[lab-21] scaling gpu-holder 3->2 (frees one node's 8 GPUs)"
kubectl scale deploy gpu-holder --replicas=2
for _ in $(seq 1 60); do
  n=$(kubectl get pods -l app=gpu-holder --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = "2" ] && break; sleep 3
done
held="$(kubectl get pods -l app=gpu-holder -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)"
FREE_NODE=""; for node in "${NODES[@]}"; do grep -qx "$node" <<<"$held" || { FREE_NODE="$node"; break; }; done
[ -n "$FREE_NODE" ] || { echo "[lab-21] FATAL: could not identify a freed node" >&2; exit 1; }
FREE_SHORT="${FREE_NODE##*-}"
echo "[lab-21] borrowing free node: $FREE_NODE (short=${FREE_SHORT})"

# --- workbench: 8 GPUs (node fully held) -------------------------------------
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: ${POD}, labels: { app: infer-serve-wb } }
spec:
  nodeName: ${FREE_NODE}
  restartPolicy: Never
  tolerations: [{ operator: Exists }]
  containers:
  - name: bench
    image: ${IMAGE}
    command: ["sleep","infinity"]
    resources: { limits: { nvidia.com/gpu: "8" } }
    volumeMounts: [{ name: dshm, mountPath: /dev/shm }]
  volumes:
  - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
EOF
kubectl wait --for=condition=Ready "pod/${POD}" --timeout=300s \
  || { echo "[lab-21] FATAL: pod not ready" >&2; kubectl describe pod "$POD" | tail -30; exit 1; }
kubectl cp "${HERE}/serve.py"             "${POD}:/workspace/serve.py" -c bench
kubectl cp "${HERE}/loadgen.py"           "${POD}:/workspace/loadgen.py" -c bench
kubectl cp "${HERE}/throughput_scale.py"  "${POD}:/workspace/throughput_scale.py" -c bench

# --- Phase A: the serving saturation knee (latency vs concurrency) -----------
echo "[lab-21] === Phase A: start server + sweep concurrency (saturation knee) ==="
X 'cd /workspace; BATCH_MAX=16 BATCH_DELAY_MS=5 nohup python3 serve.py >/workspace/serve.log 2>&1 &
   for i in $(seq 1 60); do curl -s http://127.0.0.1:8080/ >/dev/null 2>&1 && break; sleep 2; done
   echo "server up:"; head -1 /workspace/serve.log'
cap_run "$LAB" "latency_vs_concurrency.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'for C in 1 2 4 8 16 32 64; do CONC=$C DURATION=8 python3 /workspace/loadgen.py; done'

# --- Phase C (during a sustained load): DCGM engine-active off GMP ------------
echo "[lab-21] === Phase C: DCGM engine-active under sustained serving load ==="
X 'CONC=64 DURATION=90 python3 /workspace/loadgen.py >/workspace/sustain.log 2>&1 &'
S0=$(date -u +%s); sleep 90; S1=$(date -u +%s); sleep 45
tok="$(gcloud auth print-access-token 2>/dev/null)"
{ echo "# DCGM_FI_PROF_GR_ENGINE_ACTIVE gpu0 node …${FREE_SHORT} under CONC=64 serving load  [${S0}..$((S1+30))]";
  curl -s -G "${GMP}/query_range" -H "Authorization: Bearer ${tok}" \
    --data-urlencode 'query=DCGM_FI_PROF_GR_ENGINE_ACTIVE{gpu="0"}' \
    --data-urlencode "start=${S0}" --data-urlencode "end=$((S1+30))" --data-urlencode "step=15" \
  | python3 -c 'import json,sys,datetime
d=json.load(sys.stdin)
for s in d.get("data",{}).get("result",[]):
    h=s["metric"].get("Hostname","?")
    if "'"${FREE_SHORT}"'" not in h: continue
    vals=s.get("values",[]); avg=sum(float(v) for _,v in vals)/max(1,len(vals))
    out=[datetime.datetime.fromtimestamp(float(t),datetime.timezone.utc).strftime("%H:%M:%S")+"="+format(float(v),".3f") for t,v in vals]
    print(f"  {h[-4:]} gpu0 avg={avg:.3f}: "+" ".join(out))'; } | tee "${OUT}/dcgm_under_load.txt"

# --- Phase B: horizontal throughput scaling across the 8 GPUs ----------------
echo "[lab-21] === Phase B: horizontal throughput scaling 1..8 GPUs ==="
X 'pkill -f serve.py 2>/dev/null; sleep 3'   # free GPUs for the scaling probe
cap_run "$LAB" "throughput_scaling.txt" -- kubectl exec "$POD" -c bench -- bash -lc \
  'for W in 1 2 4 8; do WORKERS=$W BATCH=16 DURATION=8 python3 /workspace/throughput_scale.py; done'

# --- reference manifest (validate it parses; NOT applied — see header) --------
echo "[lab-21] === reference autoscale topology (dry-run only; not applied) ==="
cap_run "$LAB" "reference_autoscale_dryrun.txt" -- bash -lc \
  "echo '# NOT applied live (always-hold + Flex leave no capacity to scale into).';
   echo '# Server-side validation that the production topology is well-formed:';
   kubectl --context ${KUBE_CONTEXT} apply --dry-run=server -f ${REPO_ROOT}/manifests/serving/inference-autoscale.yaml 2>&1"

# --- provenance --------------------------------------------------------------
cap_verify_provenance "$LAB" "assets/lab-21/latency_vs_concurrency.txt" "$FREE_NODE" "serving saturation knee (latency vs concurrency)"
cap_verify_provenance "$LAB" "assets/lab-21/throughput_scaling.txt"      "$FREE_NODE" "horizontal throughput scaling 1..8 GPUs"
cap_verify_provenance "$LAB" "assets/lab-21/dcgm_under_load.txt"         "$FREE_NODE" "GMP DCGM engine-active under serving load"

echo "[lab-21] DONE — assets in ${OUT}. (trap restores gpu-holder=3)"
