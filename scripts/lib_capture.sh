#!/usr/bin/env bash
# Shared capture helpers. Source this from every lab's run.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/assets"
VERIF="$REPO_ROOT/VERIFICATION.md"

# --- Cluster/pool targeting (env overrides, backward-compatible defaults) ----
# LAB_NODEPOOL : GKE node-pool label to enumerate GPU nodes from.
#   Default a3-h100-dws-pool = the 2-node us-central1 lab cluster (labs 01-11).
#   Set to a3-high-flex-pool for the 3-node asia-east1-c scaling cluster (lab-12/13).
# KUBE_CONTEXT : kube context to target. Empty = current context (labs 01-11
#   were captured on the current/us-central1 context, so the default is unchanged).
LAB_NODEPOOL="${LAB_NODEPOOL:-a3-h100-dws-pool}"
KUBE_CONTEXT="${KUBE_CONTEXT:-}"

# Wrapper so EVERY kubectl call in every lab (including those passed through
# cap_run "$@") transparently honors KUBE_CONTEXT. With KUBE_CONTEXT unset this
# is a no-op and behavior is identical to calling kubectl directly.
_kubectl_ctx_args=()
[ -n "$KUBE_CONTEXT" ] && _kubectl_ctx_args=(--context "$KUBE_CONTEXT")
kubectl() { command kubectl "${_kubectl_ctx_args[@]}" "$@"; }

cap_nodes() {
  kubectl get nodes -l "cloud.google.com/gke-nodepool=$LAB_NODEPOOL" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}

cap_run() { # cap_run <label> <outfile> -- <cmd...>
  local label="$1" outfile="$2"; shift 2; [ "$1" = "--" ] && shift
  mkdir -p "$ASSETS/$label"
  local path="$ASSETS/$label/$outfile"
  { printf '# cmd:'; printf ' %q' "$@"; printf '\n'; } | tee "$path"
  "$@" 2>&1 | tee -a "$path"
  return "${PIPESTATUS[0]}"
}

cap_gpu_exec() { # cap_gpu_exec <pod> -- <cmd...>
  local pod="$1"; shift; [ "$1" = "--" ] && shift
  kubectl exec "$pod" -- bash -lc "$*"
}

cap_verify_provenance() { # <lab> <artifact> <nodes> <note>
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '| %s | %s | %s | %s | %s |\n' "$ts" "$1" "$2" "$3" "$4" >> "$VERIF"
}
