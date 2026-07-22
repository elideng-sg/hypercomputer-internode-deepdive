#!/usr/bin/env bash
# Fault (b): show a noisy neighbour collapsing cross-node bandwidth on the shared gVNIC.
# Two hostNetwork iperf3 probes (0 GPU) on the two GPU nodes, over the SAME ~200 Gbit/s
# gVNIC that carries all inter-node NCCL (doc-05). hv7m is DWS-held, so this is a
# GPU-free proxy for a cross-node collective, mapped onto the lab-06 busbw floor.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib_capture.sh
LAB=lab-10
SRV=hhp6; CLI=hv7m   # server on the free node, client on the held node's HOST net (no GPU)

# iperf3 server pod (hostNetwork, pinned to hhp6, 0 GPU)
kubectl run iperf-srv --image=networkstatic/iperf3 --restart=Never \
  --overrides='{"spec":{"hostNetwork":true,"nodeSelector":{"kubernetes.io/hostname":"'"$SRV"'"}}}' \
  -- -s >/dev/null 2>&1
for i in $(seq 1 30); do [ "$(kubectl get pod iperf-srv --no-headers 2>/dev/null|grep -c Running)" = 1 ] && break; sleep 3; done
SRVIP=$(kubectl get pod iperf-srv -o jsonpath='{.status.podIP}')

runclient () { # $1=label $2=streams $3=duration
  kubectl run "iperf-cli-$1" --image=networkstatic/iperf3 --restart=Never --rm -i \
    --overrides='{"spec":{"hostNetwork":true,"nodeSelector":{"kubernetes.io/hostname":"'"$CLI"'"}}}' \
    -- -c "$SRVIP" -P "$2" -t "$3" 2>/dev/null | grep -E 'SUM|sender|receiver'
}

{
  echo "### victim flow ALONE (8 parallel TCP streams):"; runclient victim1 8 8
  echo; echo "### background 32-stream load — pegs the gVNIC at its ceiling:"
  runclient bg 32 25 &
  sleep 3
  echo; echo "### the SAME victim flow, now DURING saturation:"; runclient victim2 8 8
  wait
} | tee "$ASSETS/$LAB/fault-nic-saturation.live.txt"

kubectl delete pod iperf-srv --ignore-not-found --grace-period=0 --force >/dev/null 2>&1
echo "nic-saturation done; probes deleted (0 GPU used, hv7m host-net only, holder untouched)."
