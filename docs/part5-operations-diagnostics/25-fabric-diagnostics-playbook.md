# Doc 25 — GPU Fabric Diagnostics Playbook (TCPX / TCPXO)

> **Audience:** local and remote support teams taking a first-line ticket that reads
> *"distributed training is slow"*, *"multi-node is no faster than single-node"*, or
> *"the job hangs at NCCL init"*. No prior knowledge of the customer's cluster is assumed.
>
> **Companion tooling:** [`scripts/verify_gpu_fabric.sh`](../../scripts/verify_gpu_fabric.sh)
> (verdict in one command) and [`scripts/collect_fabric_bundle.sh`](../../scripts/collect_fabric_bundle.sh)
> (escalation bundle in one command).

---

## 0. The one thing to understand before you touch a ticket

**The GPU fabric fails open.** Every other layer of the stack fails loudly — a bad image
gives you `ImagePullBackOff`, a bad PVC gives you `FailedMount`, an OOM gives you a
`137`. The fabric does not. If any layer of the GPUDirect stack is missing or
misconfigured, NCCL silently selects its built-in TCP socket transport, the job runs to
completion, every exit code is `0`, and nothing anywhere reports an error.

The only symptom is **throughput**:

| Path | Measured on this platform | Notes |
|---|---|---|
| Intra-node (NVLink/NVSwitch) | **~480 GB/s** | unaffected by fabric misconfiguration — this is why single-node benchmarks look fine |
| Inter-node, GPUDirect engaged | ~100–200 GB/s class | TCPX (A3 High, 4 NICs) / TCPXO (A3 Mega, 8 NICs) |
| Inter-node, silent TCP fallback | **~28.6 GB/s** | ~**17×** worse than intra-node; **no error is emitted** |

> **Consequence for support:** "the job completed successfully" is *not* evidence that the
> fabric is healthy, and "there are no errors in the logs" is *not* evidence either. A
> silent 17× regression on scarce H100 capacity is the single most expensive failure mode
> on this platform. It must be checked explicitly, every time.

---

## 1. First response — one command

```bash
CLUSTER=<cluster> ZONE=<zone> scripts/verify_gpu_fabric.sh
# with a live workload pod (strongly preferred — see §3):
CLUSTER=<cluster> ZONE=<zone> NAMESPACE=<ns> POD=<pod> scripts/verify_gpu_fabric.sh
```

Exit codes: `0` healthy for its tier · `1` degraded (silent fallback likely) · `2` cannot determine.

It checks the eight layers that must **all** hold for GPUDirect to engage, prints a
PASS/FAIL table, and orders the probable root causes. Real output from this project's
production A3 Mega cluster:

```
 [FAIL] 1. Dataplane V2          '<empty>' (legacy) — GPUDirect IMPOSSIBLE on this cluster
 [FAIL] 2. Multi-networking      '<empty>' — GPU NICs cannot be attached
 [FAIL] 3. GPU Network CRDs      0 NetDevice GPU nets (want 8; 0 total)
 [FAIL] 4. Pool GPU NICs         0 additional node networks (want 8)
 [PASS] 5. Jumbo MTU 8244        all gpu-net VPCs >= 8244 (or none present)
 [FAIL] 6. NCCL plugin DS        not found
 tier=TCPXO  pass=1  fail=5  warn=1
```

That cluster runs real workloads at the 28.6 GB/s floor and has never emitted an error
about it.

---

## 2. The eight layers, and what each one costs you

Order matters: layers 1–2 are **create-time-only**, so if they fail, nothing downstream is
worth investigating until a new cluster exists.

| # | Layer | How it fails | Fixable in place? |
|---|---|---|---|
| 1 | Dataplane V2 (`ADVANCED_DATAPATH`) | multi-networking impossible | ❌ **create-time only** |
| 2 | Multi-networking | Pods cannot attach GPU NICs | ❌ **create-time only** |
| 3 | `Network`/`GKENetworkParamSet` CRDs, `deviceMode: NetDevice`, one pair per NIC | Pod gets no GPU NIC | ✅ `kubectl apply` |
| 4 | Node pool `--additional-node-network` × N | VM has no GPU NICs | ❌ **recreate the pool** |
| 5 | Jumbo MTU 8244 on GPU VPCs | large BW loss; mismatch → hangs | ❌ set at VPC create |
| 6 | NCCL plugin installer DaemonSet (correct flavour) | no `libnccl-net`, socket fallback | ✅ fix + re-apply |
| 7 | Realised per-node NIC count | pool built wrong | ❌ recreate the pool |
| 8 | **NCCL's actual transport choice** | *this is the verdict* | — |

**Layers 1–7 are necessary. Only layer 8 is sufficient.** A cluster can pass 1–7 and still
fall back to sockets (wrong env family, missing rxdm sidecar). Never close a ticket on
1–7 alone.

### Tier expectations

| Machine | Tier | GPU NICs | Plugin | NCCL env family | Good transport line |
|---|---|---|---|---|---|
| `a3-highgpu-8g` | TCPX | 4 | `nccl-plugin-gpudirecttcpx` | `NCCL_GPUDIRECTTCPX_*` | `NET/GPUDirectTCPX` |
| `a3-megagpu-8g` | TCPXO | 8 | `nccl-plugin-gpudirecttcpx-dev` (see §4.1) | `NCCL_FASTRAK_*` | `NET/FasTrak` |
| `a3-ultragpu-8g`, `a4-highgpu-8g`, `a4x-highgpu-4g` | RDMA/RoCE | 8 | native (no installer) | `NCCL_IB_*` | `NET/IB` |

---

## 3. The decisive check (layer 8) — do this before escalating

Everything else is circumstantial. Only the workload's own NCCL log proves which
transport was used. The workload **must** run with:

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
```

Without these two variables the transport choice is *never logged* and the ticket cannot
be resolved from logs at all. Then:

```bash
kubectl logs -n <ns> <pod> | grep -E 'NET/(Socket|IB|FasTrak|GPUDirectTCPX)|GPU Direct RDMA'
```

| What you see | Verdict |
|---|---|
| `NET/FasTrak` / `NET/GPUDirectTCPX` / `NET/IB` | ✅ GPUDirect engaged. Slowness is **above** the fabric — data loading, `NCCL_ALGO`, batch size, job topology. |
| `NET/Socket` | ❌ **Silent fallback.** This is the bug. Work §2 top-down. |
| `GPU Direct RDMA Disabled` | ❌ Usually a missing **rxdm sidecar** (§4.3). |
| nothing | ⚠️ Debug env not set. Re-run the workload; you have no evidence yet. |

> **Flex-start warning.** On a Flex-start / DWS pool the node returns to the pool when the
> workload ends, taking its NIC list, `dmesg` and NCCL logs with it. **Capture while the
> node is alive** — run `scripts/collect_fabric_bundle.sh` before letting the pool scale to 0.

---

## 4. Failure-signature catalogue

Each entry: the observable signature, the mechanism, and the fix. Signatures marked
**OBSERVED** were hit live while building this platform — they are not hypotheticals.

### 4.1 `Init:ErrImagePull` on the TCPXO installer — **OBSERVED 2026-07-28**

```
nccl-tcpxo-installer-4d4xd   0/1   Init:ErrImagePull
Back-off pulling image ".../gpudirect-tcpxo/nccl-plugin-gpudirecttcpxo:latest"
```

**Mechanism.** The obvious image name does not exist. The TCPXO plugin lives in the
`gpudirect-tcpxo` *repository* but under the `nccl-plugin-gpudirecttcpx-dev` *image* name
— repo says `tcpxo`, image says `tcpx`, and the one is **not** a typo for the other.
There is also no usable `:latest` contract: tags are `vN.N.N`, and the repo additionally
carries `deprecated-public-image-*` and `no-new-use-public-image-*` prefixes.

**Why it is expensive.** This fails on an already-provisioned Flex-start GPU node, so it
burns scarce H100 time while you debug an image name.

**Fix.** Pin an explicit version and enumerate tags rather than guessing:

```bash
gcloud artifacts docker images list \
  us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev \
  --include-tags --format='value(tags)'
# plugin was v1.0.17 and rxdm tcpgpudmarxd-dev was v1.0.23 on 2026-07-28 —
# the two images are versioned SEPARATELY. Do not assume a shared version.
```

### 4.2 Installer DaemonSet `DESIRED 0` / `0 ready` — wrong accelerator label

**Signature.** `kubectl get ds -n kube-system` shows the installer with `DESIRED 0`, or
`0 ready` while GPU nodes exist. No error anywhere; NCCL then uses sockets.

**Mechanism.** `nodeSelector: cloud.google.com/gke-accelerator` must match **exactly**.
A3 High is `nvidia-h100-80gb`; A3 Mega is `nvidia-h100-mega-80gb`. A one-word mismatch
means the DaemonSet never schedules and the plugin is never installed.

**Fix.** Correct the selector and re-apply. Distinguish this from the benign case below.

> **Not a bug:** `0 ready` with **0 GPU nodes** on a Flex-start pool is correct — a
> DaemonSet has nowhere to land while capacity is pending. `verify_gpu_fabric.sh`
> deliberately reports `WARN` (not `FAIL`) for this, so support is not sent chasing a
> non-bug on every stocked-out Flex cluster.

### 4.3 Plugin installed, but `GPU Direct RDMA Disabled` — missing rxdm sidecar

**Mechanism.** The receive-datapath manager (`tcpgpudmarxd` for TCPX, tcpxo-daemon for
TCPXO) must run **as a sidecar in the workload Pod** so it shares the Pod's network
namespace. A node-level DaemonSet cannot see the workload Pod's netns. This is the #2
cause of "plugin loaded but GPUDirect still off".

**Fix.** Add the rxdm container to the workload Pod spec (not to a DaemonSet), with the
shared-memory and device mounts from the GKE recipe.

### 4.4 Copied the TCPX recipe onto A3 Mega — wrong env family

**Mechanism.** TCPXO is *not* "TCPX with more NICs". Three things differ: the plugin
image, the rxdm flavour, and the NCCL env family — `NCCL_FASTRAK_*` (TCPXO) vs
`NCCL_GPUDIRECTTCPX_*` (TCPX). Setting the TCPX variables on A3 Mega leaves the FasTrak
plugin unconfigured; it fails open to sockets. **This is the most common TCPXO bring-up
mistake.**

**Fix.** Use the TCPXO env block. Confirm with `kubectl exec <pod> -- env | grep NCCL_`
and check the family matches the machine type per §2.

### 4.5 Wrong `*_SOCKET_IFNAME` — job hangs at NCCL init

**Mechanism.** The control-plane interface list must name the real interfaces. On a
correctly built A3 Mega node the NICs are `eth0` (control) + `eth1..eth8` (GPU). Naming a
nonexistent interface leaves NCCL waiting for a bootstrap connection that can never
complete — the classic "hangs at init, no error" report.

**Verify** (this is real captured output from a live TCPXO node — 9 NICs, eth0 + 8 GPU):

```
NIC COUNT: 9
  eth0 10.148.0.17 0000:00:0c.0      <- control plane
  eth1 192.169.0.2 0000:06:00.0      <- GPU NIC 0
  eth2 192.169.1.2 0000:07:00.0
  ...
  eth8 192.169.7.2 0000:8e:00.0      <- GPU NIC 7
```

```bash
kubectl get node <node> -o jsonpath='{.metadata.annotations.networking\.gke\.io/nic-info}'
```

**Fix.** Set the control interface to `eth0` and the data interfaces to the GPU NIC range
per the tier's recipe. Never hand-guess names — read the annotation.

### 4.6 MTU mismatch — hangs and retransmits, not a clean error

**Mechanism.** GPU VPCs must be MTU **8244**. A default 1460 VPC silently costs a large
fraction of achievable bandwidth; a *mismatch* between peers causes hangs and
retransmissions rather than an error. MTU is fixed at VPC creation.

**Fix.** `gcloud compute networks describe <gpu-net> --format='value(mtu)'` for every GPU
VPC; recreate any that are undersized.

### 4.7 Rail imbalance — throughput ~1/N of expected

**Mechanism.** Traffic concentrates on one NIC instead of spreading across all rails.
Expected symptom: aggregate inter-node throughput near single-NIC line rate while the
transport line correctly shows GPUDirect.

**Verify.** Per-NIC byte counters should be roughly equal across `eth1..ethN` (§5). In
`nvidia-smi topo -m`, a `NODE`/`SYS` hop on a GPU↔NIC pair that should be `PIX`/`PXB`
indicates the GPU is crossing the host bridge to reach its NIC.

**Fix.** Check the GPU↔NIC affinity mapping in the workload env against the topology.

### 4.8 Cluster gates missing — the unfixable one

**Signature.** Layers 1–2 FAIL (`datapathProvider` empty, `enableMultiNetworking` empty).

**Mechanism.** Both are **create-time-only**. No `gcloud container clusters update` can
add them.

**Fix.** A **new cluster** is required (`--enable-dataplane-v2 --enable-multi-networking`).
Tell the customer this early and plainly: it is a migration, not a config change. Set
expectations before spending a week on downstream layers.

> **Also create-time:** the node pool's `--additional-node-network` flags (layer 4). An
> existing pool cannot gain GPU NICs — it must be recreated.

### 4.9 Flex-start pool provisioning errors that look like fabric bugs

Two real ones from this project, both easily misread as fabric problems:

| Error | Reality |
|---|---|
| `flex start node pools require autoscaling enabled` / `flex start node pools don't support reservations` | API-level: a flex pool needs `--enable-autoscaling --total-min-nodes=0 --total-max-nodes=N --reservation-affinity=none`. Pending Pods drive the scale-up. |
| `FailedScaleUp: GCE out of resources` | Genuine zonal stockout, **not** misconfiguration. Retry; do not "fix" anything. |
| `Requested CIDR ... is not available in network default` | Pod CIDR collides with another cluster in the shared VPC. Survey with `gcloud container clusters list --format='value(name,clusterIpv4Cidr)'`. Fails *late*, after VPCs are built. |

---

## 5. Monitoring — catching silent fallback before a human notices

The point of monitoring here is to make a *silent* failure *loud*. Three signals:

### 5.1 Per-NIC throughput and rail balance

```promql
# Per-interface receive rate on GPU nodes (expect traffic on ALL GPU NICs, roughly equal)
sum by (instance, device) (
  rate(node_network_receive_bytes_total{device=~"eth[1-8]"}[5m])
)

# Rail imbalance ratio: max NIC / mean NIC. Sustained > 2 during a collective-heavy
# phase means traffic is concentrating on one rail (§4.7).
max by (instance) (rate(node_network_receive_bytes_total{device=~"eth[1-8]"}[5m]))
  /
avg by (instance) (rate(node_network_receive_bytes_total{device=~"eth[1-8]"}[5m]))
```

### 5.2 The silent-fallback alert (the important one)

There is no metric named "NCCL fell back to sockets", so alert on the *shape* of the
failure: GPUs busy, and GPU-NIC traffic essentially absent.

```yaml
- alert: GPUFabricSilentFallbackSuspected
  # GPUs are working hard but the dedicated GPU NICs are idle => inter-node traffic is
  # almost certainly going over eth0/TCP instead of GPUDirect. This is the ONLY
  # automated warning you will get for a 17x regression.
  expr: |
    (avg by (instance) (DCGM_FI_DEV_GPU_UTIL) > 80)
    and
    (sum by (instance) (rate(node_network_transmit_bytes_total{device=~"eth[1-8]"}[10m])) < 1e8)
  for: 15m
  labels: {severity: critical}
  annotations:
    summary: "GPU fabric may have silently fallen back to TCP on {{ $labels.instance }}"
    runbook: "docs/part5-operations-diagnostics/25-fabric-diagnostics-playbook.md — run scripts/verify_gpu_fabric.sh"

- alert: GPUNCCLPluginNotReady
  # Catches §4.1/§4.2 (bad image, wrong accelerator label) — but only when GPU nodes
  # actually exist, so stocked-out Flex pools do not page anyone.
  expr: |
    kube_daemonset_status_number_ready{daemonset=~"nccl-.*(tcpx|tcpxo).*"} == 0
    and on() (count(kube_node_labels{label_cloud_google_com_gke_accelerator!=""}) > 0)
  for: 10m
  labels: {severity: warning}
```

### 5.3 DCGM signals worth a panel

| Metric | Reads on |
|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | GPU busy — pair with NIC idle to detect fallback (§5.2) |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | intra-node path; stays healthy even when the fabric is broken (this is *why* single-node tests mislead) |
| `DCGM_FI_PROF_PCIE_TX_BYTES` / `_RX_BYTES` | host-bounce traffic; high values with GPUDirect "engaged" hint at a wrong rail (§4.7) |
| `DCGM_FI_DEV_XID_ERRORS` | rules out a GPU fault before blaming the network |

**Suggested Grafana layout** — one row, four panels, readable in ten seconds:
1. per-NIC TX/RX stacked by `device` (all rails visible, or the imbalance is obvious)
2. rail-imbalance ratio (§5.1) with a threshold line at 2
3. GPU util vs GPU-NIC bytes on a shared time axis (the fallback signature)
4. plugin DaemonSet ready count, annotated with GPU node count

---

## 6. Escalation — one command, no round-trips

```bash
NAMESPACE=<ns> POD=<pod> scripts/collect_fabric_bundle.sh
```

Produces `fabric-bundle-<cluster>-<utc>.tar.gz` plus a `SUMMARY.txt` written to be pasted
straight into a ticket. It captures the verdict, cluster create-time gates, per-pool
network config, the CRDs, node NIC annotations, installer logs, workload logs (current
**and** previous), the in-pod NIC/plugin/topology view, and a pre-computed transport
verdict so the reader does not have to grep.

**Before you escalate, confirm you have:**

- [ ] `verify_gpu_fabric.sh` output (exit code stated)
- [ ] Machine type and expected tier (§2)
- [ ] **A workload NCCL log with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`** — without this, expect the ticket to bounce
- [ ] The transport line, verbatim
- [ ] `NCCL_*` env as seen inside the container (`kubectl exec <pod> -- env | grep NCCL_`)
- [ ] Node NIC annotation (`nic-info`), captured **while the node is alive**
- [ ] Whether the pool is Flex-start (evidence expires when it scales to 0)

**Bundle contents are internal-ish:** node names, internal IPs, VPC/subnet names, NCCL
logs. No Secrets, ConfigMap payloads, or workload data are read. Review `SUMMARY.txt`
before sending outside your organisation.

---

## 7. Two-minute triage card

```
"multi-node training is slow"
  |
  1. scripts/verify_gpu_fabric.sh
  |     exit 1 + layer 1/2 FAIL  -> create-time gate missing. NEW CLUSTER REQUIRED (§4.8). Say so now.
  |     exit 1 + layer 4/7 FAIL  -> pool built without GPU NICs. RECREATE THE POOL.
  |     exit 1 + layer 3 FAIL    -> kubectl apply the Network/ParamSet CRDs. Cheap fix.
  |     exit 1 + layer 6 FAIL    -> installer image or accelerator label (§4.1, §4.2).
  |     exit 0                   -> config is right; go to 2. DO NOT STOP HERE.
  |
  2. Is the workload running with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET?
  |     no  -> no evidence exists yet. Re-run it. Nothing else is actionable.
  |     yes -> grep the transport line.
  |
  3. Transport line says...
        NET/Socket                -> silent fallback. Check env family (§4.4), rxdm sidecar (§4.3), SOCKET_IFNAME (§4.5).
        GPU Direct RDMA Disabled  -> rxdm sidecar missing from the WORKLOAD pod (§4.3).
        NET/FasTrak | TCPX | IB    -> fabric is fine. Look ABOVE it: data loading, NCCL_ALGO, topology.
                                      If throughput ~1/N of expected, check rail balance (§4.7).
  |
  4. Still unresolved -> NAMESPACE=.. POD=.. scripts/collect_fabric_bundle.sh, attach tarball, paste SUMMARY.txt.
```

---

## 8. Why this platform's own clusters fail the check

Honest self-assessment, because support will encounter these exact clusters:

| Cluster | Machine | Tier possible | Fabric status |
|---|---|---|---|
| `hypercomputer-a3-asiaeast1` | `a3-highgpu-8g` | TCPX | ❌ single-gVNIC — no Dataplane V2. **Create-time; unfixable in place.** |
| `hypercomputer-a3-asiasoutheast1` | `a3-megagpu-8g` | TCPXO | ❌ single-gVNIC — no Dataplane V2. Runs real workloads at the 28.6 GB/s floor. |
| `hypercomputer-a3-tcpx` | `a3-highgpu-8g` | TCPX | ✅ gates cleared: DPv2 + multi-networking + 4 GPU nets + CRDs live. Capacity-gated only. |
| `hypercomputer-a3-tcpxo` | `a3-megagpu-8g` | TCPXO | ✅ gates cleared: DPv2 + multi-networking + 8 GPU nets + CRDs live; **9 NICs realised on-node**. |

The two production clusters are the reason this playbook exists: they were built before
the create-time gates were understood, they have never emitted a single error about it,
and the cost is a permanent ~17× inter-node penalty that only an explicit check reveals.
