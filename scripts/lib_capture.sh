#!/usr/bin/env bash
# Shared capture helpers. Source this from every lab's run.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/assets"
VERIF="$REPO_ROOT/VERIFICATION.md"

cap_nodes() {
  kubectl get nodes -l cloud.google.com/gke-nodepool=a3-h100-dws-pool \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}

cap_run() { # cap_run <label> <outfile> -- <cmd...>
  local label="$1" outfile="$2"; shift 2; [ "$1" = "--" ] && shift
  mkdir -p "$ASSETS/$label"
  local path="$ASSETS/$label/$outfile"
  echo "# cmd: $*" | tee "$path"
  "$@" 2>&1 | tee -a "$path"
}

cap_gpu_exec() { # cap_gpu_exec <pod> -- <cmd...>
  local pod="$1"; shift; [ "$1" = "--" ] && shift
  kubectl exec "$pod" -- bash -lc "$*"
}

cap_verify_provenance() { # <lab> <artifact> <nodes> <note>
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '| %s | %s | %s | %s | %s |\n' "$ts" "$1" "$2" "$3" "$4" >> "$VERIF"
}
