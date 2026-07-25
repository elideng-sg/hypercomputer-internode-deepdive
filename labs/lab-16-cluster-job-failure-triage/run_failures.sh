#!/usr/bin/env bash
# Lab 16 — cluster & job failure triage (Flex-safe, NO GPU-borrow window).
#
# "My job won't run / crashed / is stuck." The scheduler/quota/framework layer
# of doc-16's stack — the failures that live in kubectl/Kueue/JobSet state and
# Cloud Logging, not in nvidia-smi. Five faults, each captured live, each read
# with the SAME triage loop (get -> describe -> events -> logs -> root cause):
#   A. OOMKilled (exit 137)      — a container that exceeds its memory limit.
#   B. CrashLoopBackOff          — a container that keeps exiting non-zero.
#   C. Unschedulable (Pending)   — a pod the scheduler can never place.
#   D. Kueue-inadmissible gang   — a 32-GPU JobSet gated by a 24-GPU quota
#      (QuotaReserved=False, ZERO pods) — the multi-node/quota signature.
#   E. Job retry -> BackoffLimitExceeded — the retry-then-give-up mechanism
#      (and the JobSet failurePolicy.maxRestarts analog for gangs).
#
# WHY THIS LAB: every prior lab walks the healthy path; when a job DOESN'T run,
# the signature is in cluster state, not on the GPU. Scenario D genuinely needs
# the 3-node cluster + Kueue quota (a 32-GPU/4-pod gang vs a 24-GPU quota); the
# others are the process-lifecycle signatures the healthy-path labs never emit,
# collected here so the triage-layer is complete.
#
# GPU SAFETY / Flex-safe: this lab needs NO borrow window at all.
#   - A/B/E run as 0-GPU pods (they use only spare CPU/mem on the GPU nodes).
#   - C requests 9 GPUs (> 8/node) so it can NEVER be scheduled AND the
#     autoscaler can never satisfy it -> guaranteed Pending, no scale-up.
#   - D is gated on QUOTA (32 > 24) before scheduling -> zero pods, GPUs never
#     touched.
# The gpu-holder is never scaled; it stays 3/3 throughout. No node is drained,
# cordoned, or deleted, and no ProvisioningRequest is created (nothing scales
# the Flex pool). Every object is deleted by the EXIT trap.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
set +e   # this lab submits intentionally-failing / never-scheduling objects; manage errors explicitly (gotcha G1)

LAB="lab-16"
OUT="${ASSETS}/${LAB}"; mkdir -p "$OUT"
QUEUES="${REPO_ROOT}/manifests/kueue-gpu-queues-24.yaml"   # gpu-cq-24 (24-GPU quota) + gpu-lq-24 + a3-high-flex flavor
IMAGE="${LAB16_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"   # already cached on these nodes (no pull)

# Names we create (and must clean up).
OBJS_PODS=(oom-fault oom-fixed crashloop-fault unschedulable-fault)
OBJS_JOBS=(retry-fault)
OVERQUOTA_JS=lab16-overquota-32

# --- provenance / holder sanity (informational; we never scale the holder) ----
holder="$(kubectl get deploy gpu-holder -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null)"
echo "[lab-16] gpu-holder at start: ${holder} (this lab does NOT borrow GPUs; it should stay 3/3 throughout)"
mapfile -t NODES < <(cap_nodes | sort)
echo "[lab-16] pool nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-16] cleanup: deleting all lab-16 objects (holder was never touched)"
  for p in "${OBJS_PODS[@]}"; do kubectl delete pod "$p" --wait=false --ignore-not-found 2>/dev/null || true; done
  for j in "${OBJS_JOBS[@]}"; do kubectl delete job "$j" --wait=false --ignore-not-found 2>/dev/null || true; done
  kubectl delete jobset "$OVERQUOTA_JS" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl delete -f "$QUEUES" --wait=false --ignore-not-found 2>/dev/null || true
}
trap cleanup EXIT

# small helper: wait until a pod reaches one of the given phase/reason strings (bounded)
wait_pod_state() { # <pod> <grep-regex> <max-seconds>
  local pod="$1" want="$2" max="${3:-60}"
  local end=$((SECONDS + max))
  while [ "$SECONDS" -lt "$end" ]; do
    local s; s="$(kubectl get pod "$pod" -o jsonpath='{.status.phase} {.status.containerStatuses[*].state.*.reason} {.status.containerStatuses[*].lastState.*.reason}' 2>/dev/null)"
    echo "$s" | grep -qiE "$want" && return 0
    sleep 3
  done
  return 1
}

# =============================================================================
# Phase A — OOMKilled (exit 137). A container that allocates past its mem limit.
# =============================================================================
echo "[lab-16] === Phase A: OOMKilled (exit 137) ==="
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: oom-fault, labels: { app: lab16 } }
spec:
  restartPolicy: Never
  tolerations: [{ operator: Exists }]
  containers:
  - name: hog
    image: ${IMAGE}
    command: ["python3","-c","import time; print('# allocating 4 GiB under a 256Mi limit', flush=True); x=bytearray(4*1024**3); time.sleep(5)"]
    resources: { limits: { memory: "256Mi", cpu: "500m" }, requests: { memory: "256Mi", cpu: "500m" } }
EOF
wait_pod_state oom-fault 'OOMKilled|Error|Completed' 90
cap_run "$LAB" "oom_get.txt"      -- kubectl get pod oom-fault -o wide >/dev/null
cap_run "$LAB" "oom_describe.txt" -- kubectl describe pod oom-fault >/dev/null
grep -EiA2 'Last State|Reason:|Exit Code|OOMKilled' "${OUT}/oom_describe.txt" 2>/dev/null | head -n 20 > "${OUT}/oom_signature.txt"
echo "[lab-16] OOM signature:"; cat "${OUT}/oom_signature.txt"

# The fix: raise the memory limit so the same workload fits -> Completed.
echo "[lab-16] --- fix: same pod with a 6Gi limit (expect Completed) ---"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: oom-fixed, labels: { app: lab16 } }
spec:
  restartPolicy: Never
  tolerations: [{ operator: Exists }]
  containers:
  - name: hog
    image: ${IMAGE}
    command: ["python3","-c","print('# allocating 4 GiB under a 6Gi limit -> fits', flush=True); x=bytearray(4*1024**3); print('# ok, allocated', flush=True)"]
    resources: { limits: { memory: "6Gi", cpu: "500m" }, requests: { memory: "2Gi", cpu: "500m" } }
EOF
wait_pod_state oom-fixed 'Completed|Error|OOMKilled' 90
cap_run "$LAB" "oom_fixed_get.txt" -- kubectl get pod oom-fixed -o wide >/dev/null
echo "[lab-16] fixed pod status:"; sed -n '1,4p' "${OUT}/oom_fixed_get.txt"

# =============================================================================
# Phase B — CrashLoopBackOff. A container that keeps exiting non-zero.
# =============================================================================
echo "[lab-16] === Phase B: CrashLoopBackOff ==="
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: crashloop-fault, labels: { app: lab16 } }
spec:
  restartPolicy: Always
  tolerations: [{ operator: Exists }]
  containers:
  - name: flaky
    image: ${IMAGE}
    command: ["bash","-lc","echo '# starting, will exit 1 in 2s (simulated startup crash)'; sleep 2; exit 1"]
    resources: { limits: { memory: "256Mi", cpu: "250m" }, requests: { memory: "128Mi", cpu: "250m" } }
EOF
# let it accumulate a few restarts / enter backoff
wait_pod_state crashloop-fault 'CrashLoopBackOff' 90
cap_run "$LAB" "crashloop_get.txt"      -- kubectl get pod crashloop-fault -o wide >/dev/null
cap_run "$LAB" "crashloop_describe.txt" -- kubectl describe pod crashloop-fault >/dev/null
{ echo "# get (note RESTARTS climbing, STATUS CrashLoopBackOff):"; sed -n '1,3p' "${OUT}/crashloop_get.txt";
  echo; echo "# describe — last state + backoff event:";
  grep -EiA1 'Last State|Reason:|Exit Code|Back-off|restarting' "${OUT}/crashloop_describe.txt" 2>/dev/null | head -n 16; } > "${OUT}/crashloop_signature.txt"
echo "[lab-16] crashloop signature:"; cat "${OUT}/crashloop_signature.txt"

# =============================================================================
# Phase C — Unschedulable (Pending forever). Requests 9 GPUs (> 8/node): the
# scheduler can't place it and the autoscaler can never satisfy it (no scale-up).
# =============================================================================
echo "[lab-16] === Phase C: unschedulable pod (Insufficient nvidia.com/gpu) ==="
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: unschedulable-fault, labels: { app: lab16 } }
spec:
  restartPolicy: Never
  nodeSelector: { cloud.google.com/gke-nodepool: ${LAB_NODEPOOL} }
  tolerations: [{ operator: Exists }]
  containers:
  - name: greedy
    image: ${IMAGE}
    command: ["bash","-lc","echo should-never-run; sleep 30"]
    resources: { limits: { nvidia.com/gpu: "9" } }   # > 8 GPUs/node -> unschedulable AND unscalable
EOF
sleep 20   # give the scheduler time to emit FailedScheduling
cap_run "$LAB" "unschedulable_get.txt"      -- kubectl get pod unschedulable-fault -o wide >/dev/null
cap_run "$LAB" "unschedulable_describe.txt" -- kubectl describe pod unschedulable-fault >/dev/null
{ echo "# get (STATUS should be Pending):"; sed -n '1,3p' "${OUT}/unschedulable_get.txt";
  echo; echo "# FailedScheduling event(s):";
  grep -EiA1 'FailedScheduling|Insufficient|didn.t match|nodeSelector|Unschedulable|NotTriggerScaleUp' "${OUT}/unschedulable_describe.txt" 2>/dev/null | head -n 12; } > "${OUT}/unschedulable_signature.txt"
echo "[lab-16] unschedulable signature:"; cat "${OUT}/unschedulable_signature.txt"
kubectl delete pod unschedulable-fault --wait=false --ignore-not-found 2>/dev/null

# =============================================================================
# Phase D — Kueue-inadmissible gang: a 32-GPU (4x8) JobSet against a 24-GPU
# ClusterQueue. Gated on QUOTA before scheduling: QuotaReserved=False, 0 pods.
# The holder keeps all 24 GPUs the whole time (nothing is scheduled).
# =============================================================================
echo "[lab-16] === Phase D: Kueue-inadmissible 32-GPU gang (24-GPU quota) ==="
kubectl apply -f "$QUEUES"; sleep 3
cap_run "$LAB" "quota_clusterqueue.txt" -- bash -c "echo '### the 24-GPU ClusterQueue (nominalQuota nvidia.com/gpu = 24):'; kubectl --context ${KUBE_CONTEXT} get clusterqueue gpu-cq-24 -o wide; kubectl --context ${KUBE_CONTEXT} get clusterqueue gpu-cq-24 -o jsonpath='{.spec.resourceGroups[0].flavors[0].resources}'; echo"
cat <<EOF | kubectl apply -f -
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: ${OVERQUOTA_JS}
  namespace: default
  labels: { kueue.x-k8s.io/queue-name: gpu-lq-24 }   # admitted (or gated) by Kueue as ONE Workload
spec:
  replicatedJobs:
  - name: worker
    replicas: 4                                       # 4 x 8 GPU = 32 > 24 quota -> gang-gated
    template:
      spec:
        parallelism: 1
        completions: 1
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            nodeSelector: { cloud.google.com/gke-nodepool: ${LAB_NODEPOOL} }
            tolerations: [{ operator: Exists }]
            containers:
            - name: worker
              image: ${IMAGE}
              command: ["bash","-lc","echo should-never-run; sleep 30"]
              resources: { limits: { nvidia.com/gpu: "8" } }
EOF
sleep 12
cap_run "$LAB" "gang_inadmissible.txt" -- bash -c "echo '### Kueue Workload for the 32-GPU gang (expect QuotaReserved/Admitted = False):'; kubectl --context ${KUBE_CONTEXT} get workloads -n default -o wide; echo; kubectl --context ${KUBE_CONTEXT} get workloads -n default -o json | python3 -c 'import json,sys
for w in json.load(sys.stdin)[\"items\"]:
    n=w[\"metadata\"][\"name\"]
    if \"${OVERQUOTA_JS}\" in n:
        conds=[(c[\"type\"],c[\"status\"],c.get(\"reason\",\"\"),c.get(\"message\",\"\")[:90]) for c in w.get(\"status\",{}).get(\"conditions\",[])]
        print(\"  workload:\", n)
        [print(\"   \",c) for c in conds]'; echo; echo '### pods for the gated JobSet (expect NONE created):'; kubectl --context ${KUBE_CONTEXT} get pods -l jobset.sigs.k8s.io/jobset-name=${OVERQUOTA_JS} 2>&1 | tail -2"
echo "[lab-16] inadmissible-gang signature:"; sed -n '1,20p' "${OUT}/gang_inadmissible.txt"
echo "[lab-16] (the at-quota 24-GPU gang IS admitted and runs — captured live in lab-13a; not re-run here to keep the holder at 3/3)"
kubectl delete jobset "$OVERQUOTA_JS" --wait=false --ignore-not-found 2>/dev/null

# =============================================================================
# Phase E — Job retry -> BackoffLimitExceeded (the retry-then-give-up mechanism).
# =============================================================================
echo "[lab-16] === Phase E: Job retry -> BackoffLimitExceeded ==="
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata: { name: retry-fault, labels: { app: lab16 } }
spec:
  backoffLimit: 2          # retry the pod up to 2x (3 attempts) then give up
  template:
    spec:
      restartPolicy: Never
      tolerations: [{ operator: Exists }]
      containers:
      - name: flaky
        image: ${IMAGE}
        command: ["bash","-lc","echo '# attempt: failing (exit 1)'; sleep 2; exit 1"]
        resources: { limits: { memory: "256Mi", cpu: "250m" }, requests: { memory: "128Mi", cpu: "250m" } }
EOF
# wait for the Job to exhaust retries and go Failed
end=$((SECONDS + 180))
while [ "$SECONDS" -lt "$end" ]; do
  c="$(kubectl get job retry-fault -o jsonpath='{range .status.conditions[*]}{.type}={.status}({.reason}) {end}' 2>/dev/null)"
  case "$c" in *Failed=True*|*Complete=True*) break;; esac
  sleep 5
done
cap_run "$LAB" "retry_job.txt"     -- bash -c "echo '### Job status (expect Failed, reason BackoffLimitExceeded, 3 failed attempts):'; kubectl --context ${KUBE_CONTEXT} get job retry-fault -o wide; echo; kubectl --context ${KUBE_CONTEXT} get job retry-fault -o jsonpath='{range .status.conditions[*]}{.type}={.status} reason={.reason} msg={.message}{\"\n\"}{end}'; echo; echo '### one pod per attempt (all Error):'; kubectl --context ${KUBE_CONTEXT} get pods -l job-name=retry-fault -o wide"
cap_run "$LAB" "retry_describe.txt" -- kubectl describe job retry-fault >/dev/null
grep -EiA1 'BackoffLimitExceeded|failed|Warning|Pods Statuses|Reason' "${OUT}/retry_describe.txt" 2>/dev/null | head -n 14 > "${OUT}/retry_signature.txt"
echo "[lab-16] retry signature:"; sed -n '1,20p' "${OUT}/retry_job.txt"

# =============================================================================
# Timeline + provenance
# =============================================================================
holder_end="$(kubectl get deploy gpu-holder -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null)"
{
  echo "# lab-16 cluster & job failure triage — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# cluster asia-east1-c   gpu-holder start=${holder} end=${holder_end}   (NO borrow window: holder untouched)"
  echo ""
  echo "## A. OOMKilled (exit 137):";       sed -n '1,10p' "${OUT}/oom_signature.txt" 2>/dev/null
  echo ""; echo "## B. CrashLoopBackOff:";  sed -n '1,10p' "${OUT}/crashloop_signature.txt" 2>/dev/null
  echo ""; echo "## C. Unschedulable:";     sed -n '1,8p'  "${OUT}/unschedulable_signature.txt" 2>/dev/null
  echo ""; echo "## D. Kueue-inadmissible gang (32>24):"; sed -n '1,14p' "${OUT}/gang_inadmissible.txt" 2>/dev/null
  echo ""; echo "## E. Job retry -> BackoffLimitExceeded:"; sed -n '1,8p' "${OUT}/retry_job.txt" 2>/dev/null
} > "${OUT}/failures_timeline.txt"
echo "[lab-16] ============ timeline summary ============"; cat "${OUT}/failures_timeline.txt"

cap_verify_provenance "lab-16" "assets/lab-16" "${NODES[0]:-asia-east1-c}" \
  "asia-east1-c cluster/job failure triage (NO GPU borrow; holder ${holder}->${holder_end}): OOMKilled exit137 + fix, CrashLoopBackOff, unschedulable Insufficient-GPU, Kueue-inadmissible 32-GPU gang vs 24-quota (0 pods), Job retry->BackoffLimitExceeded"
echo "[lab-16] done. (all lab-16 objects deleted by EXIT trap; gpu-holder never scaled)"
