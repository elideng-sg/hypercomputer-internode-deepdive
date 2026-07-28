#!/usr/bin/env bash
# collect_fabric_bundle.sh — package everything a support engineer needs to diagnose a
# GPU-fabric problem, in one pass, without a second round-trip to the customer.
#
# WHY: the expensive part of a fabric escalation is not the analysis, it is the three days
# of "please also send us X". Every item below has been needed at least once to close a
# real silent-fallback ticket, and several of them are UNRECOVERABLE after the fact —
# a Flex-start node that scales back to 0 takes its dmesg, its NIC list and its NCCL
# topology with it. Capture while the node is alive.
#
# USAGE:
#   scripts/collect_fabric_bundle.sh                                  # cluster-level only
#   NAMESPACE=training POD=trainer-0 scripts/collect_fabric_bundle.sh  # + in-pod capture
#   OUTDIR=/tmp/mybundle scripts/collect_fabric_bundle.sh
#
# Produces  fabric-bundle-<cluster>-<utc>.tar.gz  plus a SUMMARY.txt you can paste into
# the ticket. Read-only: creates one debug Pod only if ALLOW_DEBUG_POD=1.
#
# SAFE TO SEND? The bundle contains node names, internal IPs, VPC/subnet names and NCCL
# logs. It does NOT read Secrets, ConfigMap payloads, or workload data. Review SUMMARY.txt
# before sending outside your org.

set -uo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
CLUSTER="${CLUSTER:-}"
ZONE="${ZONE:-}"
NAMESPACE="${NAMESPACE:-}"
POD="${POD:-}"
ALLOW_DEBUG_POD="${ALLOW_DEBUG_POD:-0}"

# CLUSTER/ZONE must steer the kubectl half of the bundle too, not just the gcloud
# half — otherwise a bundle labelled "$CLUSTER" contains another cluster's Pods,
# DaemonSets and logs, and support triages the wrong system. See the same fix and
# the mis-read it prevents in verify_gpu_fabric.sh.
KCTX=""
if [ -z "$CLUSTER" ]; then
  CTX="$(kubectl config current-context 2>/dev/null)"
  CLUSTER="$(echo "$CTX" | awk -F_ '{print $NF}')"
  ZONE="${ZONE:-$(echo "$CTX" | awk -F_ '{print $(NF-1)}')}"
else
  WANT="gke_${PROJECT}_${ZONE}_${CLUSTER}"
  if kubectl config get-contexts -o name 2>/dev/null | grep -qx "$WANT"; then
    KCTX="$WANT"
  else
    KCTX="$(kubectl config get-contexts -o name 2>/dev/null | grep -E "_${CLUSTER}\$" | head -1)"
  fi
  if [ -z "$KCTX" ]; then
    echo "FATAL: CLUSTER=$CLUSTER requested but no kubeconfig context matches it — the"
    echo "       bundle would mix in a different cluster's kubectl output. Run:"
    echo "       gcloud container clusters get-credentials $CLUSTER --zone ${ZONE:-<zone>} --project ${PROJECT:-<project>}"
    exit 2
  fi
fi

# Every later kubectl routes through here, including the ones run via cap() —
# bash resolves `"$@"` against shell functions, so the pin survives that hop.
kubectl(){
  if [ -n "$KCTX" ]; then command kubectl --context "$KCTX" "$@"
  else command kubectl "$@"; fi
}
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="${OUTDIR:-${TMPDIR:-/tmp}/fabric-bundle-${CLUSTER}-${STAMP}}"
mkdir -p "$OUTDIR" || { echo "FATAL: cannot create $OUTDIR"; exit 2; }

say(){ echo "[collect] $*"; }
# cap(): run a command, tee to a file, never abort the bundle on failure. A missing
# artifact is itself a diagnostic signal, so we record the failure instead of dying.
cap(){ local f="$1"; shift; say "$f"; { echo "\$ $*"; "$@" 2>&1; } > "$OUTDIR/$f" || echo "(command failed — see output above)" >> "$OUTDIR/$f"; }

# ---------------------------------------------------------------------------
# 0. The verdict first. Support reads SUMMARY.txt before anything else.
# ---------------------------------------------------------------------------
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SELF_DIR/verify_gpu_fabric.sh" ] || [ -f "$SELF_DIR/verify_gpu_fabric.sh" ]; then
  say "00-verify-gpu-fabric.txt (the PASS/FAIL verdict)"
  PROJECT="$PROJECT" CLUSTER="$CLUSTER" ZONE="$ZONE" NAMESPACE="$NAMESPACE" POD="$POD" \
    bash "$SELF_DIR/verify_gpu_fabric.sh" > "$OUTDIR/00-verify-gpu-fabric.txt" 2>&1
  VERIFY_RC=$?
else
  echo "verify_gpu_fabric.sh not found next to this script" > "$OUTDIR/00-verify-gpu-fabric.txt"
  VERIFY_RC=2
fi

# ---------------------------------------------------------------------------
# 1. Cluster-level create-time gates. These decide whether the fabric is even
#    POSSIBLE, and they cannot be changed later — so they bound the whole ticket.
# ---------------------------------------------------------------------------
cap 01-cluster-describe.txt gcloud container clusters describe "$CLUSTER" --zone "$ZONE" --project "$PROJECT"
cap 02-nodepools.txt        gcloud container node-pools list --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" --format=yaml
cap 03-networks-vpc.txt     gcloud compute networks list --project "$PROJECT" --format='table(name,mtu,subnet_mode)'
cap 04-subnets.txt          gcloud compute networks subnets list --project "$PROJECT" --format='table(name,region,network,ipCidrRange)'

# Per-pool additionalNodeNetworkConfigs — the create-time-only GPU NIC attachment.
{
  echo "=== additionalNodeNetworkConfigs per node pool (GPU NIC attachment) ==="
  for p in $(gcloud container node-pools list --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" --format='value(name)' 2>/dev/null); do
    echo "--- pool: $p"
    gcloud container node-pools describe "$p" --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
      --format='yaml(config.machineType,config.flexStart,networkConfig,autoscaling)' 2>&1
  done
} > "$OUTDIR/05-pool-network-config.txt"

# ---------------------------------------------------------------------------
# 2. Kubernetes-level fabric objects.
# ---------------------------------------------------------------------------
cap 10-networks-crd.yaml   kubectl get networks.networking.gke.io -o yaml
cap 11-paramsets-crd.yaml  kubectl get gkenetworkparamsets.networking.gke.io -o yaml
cap 12-nodes.yaml          kubectl get nodes -o yaml
cap 13-ds-kube-system.txt  kubectl get ds -A -o wide
cap 14-gpu-pods.txt        kubectl get pods -A -o wide
# Events expire (default 1h TTL) — grab them before they vanish.
cap 15-events.txt          kubectl get events -A --sort-by=.lastTimestamp

# Node annotations that report the REALISED NIC set. This is the ground truth
# for "were the GPU networks actually attached to the VM?".
kubectl get nodes -o json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for n in d.get("items",[]):
    m=n["metadata"]; lbl=m.get("labels",{}); ann=m.get("annotations",{})
    if not lbl.get("cloud.google.com/gke-accelerator"): continue
    print("="*70); print("node:", m["name"])
    print("  machine    :", lbl.get("node.kubernetes.io/instance-type"))
    print("  accelerator:", lbl.get("cloud.google.com/gke-accelerator"))
    for k in ("networking.gke.io/nic-info","networking.gke.io/north-interfaces",
              "networking.gke.io/default-interface"):
        if k in ann: print(f"  {k}:\n    {ann[k]}")
    al=n.get("status",{}).get("allocatable",{})
    print("  allocatable:", {k:v for k,v in al.items() if "gpu" in k.lower() or "networking" in k})
' > "$OUTDIR/16-node-nic-annotations.txt" 2>&1

# Plugin installer logs: if the plugin never installed, this says why.
{
  for ds in $(kubectl get ds -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -iE 'nccl|tcpx|fastrak'); do
    ns="${ds%%/*}"; nm="${ds##*/}"
    echo "############ DaemonSet $ns/$nm ############"
    kubectl -n "$ns" describe ds "$nm" 2>&1 | sed -n '1,60p'
    for p in $(kubectl -n "$ns" get pods -l "k8s-app=$nm" -o name 2>/dev/null | head -3); do
      echo "---- logs $p (all containers) ----"
      kubectl -n "$ns" logs "$p" --all-containers --tail=200 2>&1
    done
  done
} > "$OUTDIR/17-plugin-installer-logs.txt" 2>&1

# ---------------------------------------------------------------------------
# 3. In-pod capture — the decisive evidence. Only this proves which transport
#    NCCL chose. Everything above is circumstantial.
# ---------------------------------------------------------------------------
if [ -n "$NAMESPACE" ] && [ -n "$POD" ]; then
  cap 20-pod-describe.txt kubectl describe pod -n "$NAMESPACE" "$POD"
  cap 21-pod-logs.txt     kubectl logs -n "$NAMESPACE" "$POD" --all-containers --tail=5000
  cap 22-pod-logs-prev.txt kubectl logs -n "$NAMESPACE" "$POD" --all-containers --previous --tail=2000

  # Grep out the lines that decide the ticket, so support does not have to.
  {
    echo "=== TRANSPORT DECISION (the single most important lines in this bundle) ==="
    grep -nE 'NET/(Socket|IB|FasTrak|GPUDirectTCPX)|Using network|GPU Direct RDMA|NCCL INFO Bootstrap' \
      "$OUTDIR/21-pod-logs.txt" 2>/dev/null | head -40
    echo
    echo "=== INTERPRETATION ==="
    if grep -q 'NET/Socket' "$OUTDIR/21-pod-logs.txt" 2>/dev/null; then
      echo "!! NET/Socket present => SILENT FALLBACK to plain TCP. Inter-node collectives"
      echo "   are running at roughly 1/17th of the intra-node rate. This is the bug."
    elif grep -qE 'NET/(FasTrak|GPUDirectTCPX|IB)' "$OUTDIR/21-pod-logs.txt" 2>/dev/null; then
      echo "OK GPUDirect transport engaged. If throughput is still low the cause is ABOVE"
      echo "   the fabric (data loading, NCCL_ALGO, batch/topology), not the network."
    else
      echo "?? No NCCL transport line found. Re-run the workload with:"
      echo "     NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
      echo "   Without these, the transport choice is simply not logged and the ticket"
      echo "   cannot be resolved from logs alone."
    fi
    echo
    echo "=== NCCL_* environment actually seen by the container ==="
    kubectl exec -n "$NAMESPACE" "$POD" -- env 2>/dev/null | grep -E '^NCCL_|^LD_LIBRARY_PATH' | sort
  } > "$OUTDIR/23-transport-verdict.txt" 2>&1

  # Inside the pod: which NICs and which plugin .so are actually visible.
  kubectl exec -n "$NAMESPACE" "$POD" -- sh -c '
    echo "=== ip link (expect nic0 + one per GPU NIC) ==="; (ip -o link show 2>/dev/null || echo "ip unavailable")
    echo; echo "=== MTU per interface (want 8244 on GPU NICs) ==="
    (ip -o link show 2>/dev/null | sed -n "s/^[0-9]*: \([^:]*\).*mtu \([0-9]*\).*/\1 mtu=\2/p" || true)
    echo; echo "=== NCCL net plugin .so present? ==="
    for d in /usr/local/nvidia/lib64 /var/lib/tcpx/lib64 /var/lib/tcpxo/lib64 /usr/local/tcpx/lib64; do
      [ -d "$d" ] && { echo "-- $d"; ls -la "$d" 2>/dev/null | grep -iE "nccl|net" ; }
    done
    echo; echo "=== nvidia-smi topo (GPU<->NIC affinity; NODE/SYS on a GPU-NIC pair = wrong rail) ==="
    (nvidia-smi topo -m 2>/dev/null || echo "nvidia-smi unavailable")
  ' > "$OUTDIR/24-in-pod-nic-and-plugin.txt" 2>&1
else
  cat > "$OUTDIR/20-NO-POD-CAPTURED.txt" <<'EOF'
No NAMESPACE/POD was given, so the DECISIVE evidence is missing from this bundle.

Everything else here is circumstantial: it proves the fabric is *configured*, not that
NCCL *used* it. Re-run against a live GPU pod:

    NAMESPACE=<ns> POD=<pod> scripts/collect_fabric_bundle.sh

If the workload has already exited, restart it with NCCL_DEBUG=INFO and
NCCL_DEBUG_SUBSYS=INIT,NET before capturing. On Flex-start pools do this while the
node is still up — when the pool scales back to 0 the evidence is gone for good.
EOF
fi

# ---------------------------------------------------------------------------
# 4. SUMMARY.txt — the paste-into-the-ticket file.
# ---------------------------------------------------------------------------
{
  echo "GPU FABRIC ESCALATION BUNDLE"
  echo "collected : $STAMP (UTC)"
  echo "project   : $PROJECT"
  echo "cluster   : $CLUSTER (zone $ZONE)"
  echo "workload  : ${NAMESPACE:-<none>}/${POD:-<none>}"
  echo "verify_gpu_fabric.sh exit: $VERIFY_RC   (0=healthy 1=degraded 2=undeterminable)"
  echo
  echo "---------------- VERDICT ----------------"
  sed -n '/RESULT LAYER/,$p' "$OUTDIR/00-verify-gpu-fabric.txt" 2>/dev/null
  echo
  if [ -f "$OUTDIR/23-transport-verdict.txt" ]; then
    echo "---------------- TRANSPORT ----------------"
    sed -n '/=== INTERPRETATION ===/,/=== NCCL_\* environment/p' "$OUTDIR/23-transport-verdict.txt" 2>/dev/null
  else
    echo "---------------- TRANSPORT ----------------"
    echo "NOT CAPTURED — no pod supplied. See 20-NO-POD-CAPTURED.txt."
  fi
  echo
  echo "---------------- FILES ----------------"
  ls -la "$OUTDIR"
} > "$OUTDIR/SUMMARY.txt" 2>&1

TARBALL="${OUTDIR}.tar.gz"
tar czf "$TARBALL" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")" 2>/dev/null

echo
echo "=============================================================="
cat "$OUTDIR/SUMMARY.txt" | sed -n '1,30p'
echo "=============================================================="
echo "bundle : $TARBALL"
echo "raw    : $OUTDIR"
echo
echo "Attach the tarball to the ticket. Paste SUMMARY.txt into the description."
[ "$VERIFY_RC" = "1" ] && echo "NOTE: verify reports a DEGRADED fabric — see the VERDICT section for fix order."
exit 0
