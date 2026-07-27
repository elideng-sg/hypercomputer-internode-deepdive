#!/usr/bin/env bash
# Lab 20 — the end-to-end training PIPELINE (Flex-safe, 2-node borrow).
#
# WHY THIS LAB: labs 12/13 proved the GANG (Kueue + JobSet placing a multi-node
# job atomically); lab-19 proved the DATA PATH (GCS via GCSFuse). This lab ties
# them into ONE real training run and adds the two pieces the earlier labs faked:
#   data (GCS)  ->  distributed training (2 nodes / 16 GPUs, DDP)  ->  checkpoints (GCS)  ->  metrics (GMP/DCGM)
# The dataset is LEARNABLE (y = X·w* + b* + noise) so the loss genuinely drops —
# an honest training curve, not a fixed synthetic tensor. Code AND data both come
# from the bucket; rank 0 writes checkpoints back to the bucket; DCGM engine-active
# on GMP confirms the 16 GPUs were actually training.
#
# AUTH / userspace GCSFuse (gotcha G20): the managed CSI+WIF path needs the node
# pool on GKE_METADATA, which RECREATES nodes — unsafe for scarce Flex A3 capacity.
# So each worker mounts GCSFuse in userspace (privileged + /dev/fuse) with the
# lab19-gcs-key GSA-key Secret. Zero node changes.
#
# GPU SAFETY / Flex-safe: scale gpu-holder 3->1 (frees TWO nodes / 16 GPUs), the
# 16-GPU JobSet occupies both freed nodes (they stay FULLY held — by the training
# job now), and the EXIT trap deletes the JobSet and restores gpu-holder=3. No
# device-state change, no node drain/cordon/delete.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
set +e   # phases run bounded / best-effort; manage errors explicitly (gotcha G1)

LAB="lab-20"
OUT="${ASSETS}/${LAB}"; mkdir -p "$OUT"
HERE="${REPO_ROOT}/labs/lab-20-training-pipeline"

BUCKET="${LAB20_BUCKET:-hdlab-elideng-lab-data-asiaeast1}"
PROJECT="${LAB20_PROJECT:-hdlab-elideng}"
JOBSET="train-pipeline"
GMP="https://monitoring.googleapis.com/v1/projects/${PROJECT}/location/global/prometheus/api/v1"

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-20] FATAL: need the 3-node pool, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-20] pool nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-20] cleanup: delete JobSet, restore gpu-holder=3"
  kubectl delete jobset "$JOBSET" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

# --- 0. stage the training CODE into the bucket (pipeline reads code from GCS) --
echo "[lab-20] staging train_pipeline.py -> gs://${BUCKET}/code/"
gcloud storage cp "${HERE}/train_pipeline.py" "gs://${BUCKET}/code/train_pipeline.py" \
  || { echo "[lab-20] FATAL: could not stage code to bucket" >&2; exit 1; }
cap_run "$LAB" "code_and_data_staged.txt" -- bash -lc \
  "echo '## code:'; gcloud storage ls gs://${BUCKET}/code/; echo '## train shards:'; gcloud storage ls gs://${BUCKET}/train/ | grep -c train_ ; gcloud storage ls gs://${BUCKET}/train/"

# --- 1. Kueue quota objects (idempotent) -------------------------------------
echo "[lab-20] applying Kueue queues (gpu-cq-24 / gpu-lq-24)"
kubectl apply -f "${REPO_ROOT}/manifests/kueue-gpu-queues-24.yaml" >/dev/null 2>&1

# --- 2. borrow TWO nodes: holder 3->1 (frees 16 GPUs on 2 nodes) -------------
echo "[lab-20] scaling gpu-holder 3->1 (frees two nodes' 16 GPUs for the gang)"
kubectl scale deploy gpu-holder --replicas=1
for _ in $(seq 1 60); do
  n=$(kubectl get pods -l app=gpu-holder --field-selector=status.phase=Running -o name 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = "1" ] && break; sleep 3
done
held="$(kubectl get pods -l app=gpu-holder -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)"
FREE_NODES=(); for node in "${NODES[@]}"; do grep -qx "$node" <<<"$held" || FREE_NODES+=("$node"); done
echo "[lab-20] holder now on: [${held//$'\n'/ }] ; freed nodes for gang: ${FREE_NODES[*]}"
[ "${#FREE_NODES[@]}" -ge 2 ] || { echo "[lab-20] FATAL: expected 2 freed nodes, got ${#FREE_NODES[@]}" >&2; exit 1; }
FREE_A="${FREE_NODES[0]##*-}"; FREE_B="${FREE_NODES[1]##*-}"

# --- 3. launch the training JobSet (gang) ------------------------------------
echo "[lab-20] applying JobSet ${JOBSET} (2 nodes / 16 GPUs, gang via gpu-lq-24)"
kubectl apply -f "${REPO_ROOT}/manifests/pipeline/jobset-train-pipeline.yaml"

# capture gang admission (Kueue Workload admitted for the JobSet)
sleep 8
cap_run "$LAB" "gang_admission.txt" -- bash -lc \
  "kubectl --context ${KUBE_CONTEXT} get jobset ${JOBSET} -o wide 2>&1;
   echo '## Kueue workloads:'; kubectl --context ${KUBE_CONTEXT} get workloads.kueue.x-k8s.io 2>&1 | grep -i ${JOBSET} || kubectl --context ${KUBE_CONTEXT} get workloads.kueue.x-k8s.io 2>&1"

# --- 4. wait for the rank-0 pod, mark the training window --------------------
echo "[lab-20] waiting for worker pods to schedule + run ..."
RANK0=""
for _ in $(seq 1 100); do
  RANK0=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET} \
            -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -- '-worker-0-0-' | head -1)
  [ -n "$RANK0" ] && [ "$(kubectl get pod "$RANK0" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ] && break
  sleep 6
done
[ -n "$RANK0" ] || { echo "[lab-20] FATAL: rank-0 pod never appeared" >&2; kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET} 2>&1 | tail; exit 1; }
echo "[lab-20] rank-0 pod: $RANK0 (training window opens now)"
T0=$(date -u +%s)

# --- 5. stream + capture the training log (loss curve, throughput, ckpts) ----
echo "[lab-20] streaming training log from rank-0 (loss should decrease) ..."
cap_run "$LAB" "training_log.txt" -- bash -lc \
  "kubectl --context ${KUBE_CONTEXT} logs -f ${RANK0} 2>&1 | grep -E 'pipeline|mounted|node_rank' "
T1=$(date -u +%s)

# capture a worker-1 tail as multi-node rendezvous proof
W1=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET} \
       -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -- '-worker-1-0-' | head -1)
[ -n "$W1" ] && cap_run "$LAB" "worker1_rendezvous.txt" -- bash -lc \
  "kubectl --context ${KUBE_CONTEXT} logs ${W1} 2>&1 | grep -E 'mounted|node_rank|host=' | head -20"

# --- 6. GMP/DCGM: engine-active across the 16 GPUs during the training window -
echo "[lab-20] waiting 45s for GMP ingest, then querying DCGM engine-active over the run window"
sleep 45
tok="$(gcloud auth print-access-token 2>/dev/null)"
{ echo "# DCGM_FI_PROF_GR_ENGINE_ACTIVE across the 2 training nodes (…${FREE_A}, …${FREE_B})  [${T0}..$((T1+30))]";
  echo "# a training run keeps all 16 GPUs active; this is the GMP-side proof.";
  curl -s -G "${GMP}/query_range" -H "Authorization: Bearer ${tok}" \
    --data-urlencode 'query=avg by (Hostname)(DCGM_FI_PROF_GR_ENGINE_ACTIVE)' \
    --data-urlencode "start=${T0}" --data-urlencode "end=$((T1+30))" --data-urlencode "step=15" \
  | python3 -c 'import json,sys,datetime
d=json.load(sys.stdin)
rows=d.get("data",{}).get("result",[])
keep=[s for s in rows if any(x in s["metric"].get("Hostname","") for x in ["'"${FREE_A}"'","'"${FREE_B}"'"])]
if not keep: print("  (no samples for training nodes in window)")
for s in keep:
    h=s["metric"].get("Hostname","?")
    vals=s.get("values",[])
    avg=sum(float(v) for _,v in vals)/max(1,len(vals))
    peak=max((float(v) for _,v in vals), default=0.0)
    out=[datetime.datetime.fromtimestamp(float(t),datetime.timezone.utc).strftime("%H:%M:%S")+"="+format(float(v),".3f") for t,v in vals]
    print(f"  {h[-4:]}  avg={avg:.3f} peak={peak:.3f}  "+ " ".join(out))'; } \
  | tee "${OUT}/dcgm_training_active.txt"

# --- 7. verify checkpoints landed in the bucket ------------------------------
echo "[lab-20] verifying checkpoints written to GCS"
cap_run "$LAB" "checkpoints_in_gcs.txt" -- bash -lc \
  "gcloud storage ls -l gs://${BUCKET}/checkpoints/pipeline/ 2>&1"

# --- 8. final JobSet status (completed) --------------------------------------
cap_run "$LAB" "jobset_final.txt" -- bash -lc \
  "kubectl --context ${KUBE_CONTEXT} get jobset ${JOBSET} -o wide 2>&1;
   kubectl --context ${KUBE_CONTEXT} get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET} 2>&1"

# --- provenance --------------------------------------------------------------
cap_verify_provenance "$LAB" "assets/lab-20/training_log.txt"       "${FREE_NODES[*]}" "DDP training loss curve (2 nodes/16 GPUs, data+code+ckpt on GCS)"
cap_verify_provenance "$LAB" "assets/lab-20/dcgm_training_active.txt" "${FREE_NODES[*]}" "GMP DCGM engine-active during training"
cap_verify_provenance "$LAB" "assets/lab-20/checkpoints_in_gcs.txt"  "${FREE_NODES[*]}" "checkpoints written to GCS"

echo "[lab-20] DONE — assets in ${OUT}. (trap restores gpu-holder=3)"
