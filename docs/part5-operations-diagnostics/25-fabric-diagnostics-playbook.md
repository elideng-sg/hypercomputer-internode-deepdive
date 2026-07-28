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
| Inter-node, **GPUDirect-TCPXO engaged** | **317.84 GB/s** | measured, 2 × `a3-megagpu-8g` / 16 GPUs, `NET/FasTrak` ([lab-22 §5.1](../../labs/lab-22-fabric-diagnostics/README.md)) |
| Inter-node, silent TCP fallback | **23.7 GB/s** (also ~28.6 on another cluster) | **≈13.4×** worse than the same hardware with the fabric on; **no error is emitted** |

Both numbers come from the **same benchmark on the same GPU model**. The only difference is
whether the eight layers below are all in place.

> **Consequence for support:** "the job completed successfully" is *not* evidence that the
> fabric is healthy, and "there are no errors in the logs" is *not* evidence either. A
> silent **13×** regression on scarce H100 capacity is the single most expensive failure mode
> on this platform. It must be checked explicitly, every time.
>
> Put in customer terms: a 10-hour distributed training run on a silently-degraded fabric is
> not 10% slower. Its collective phases are running at ~7% of the bandwidth they should have.

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
 [FAIL] 6c. rxdm devices        /dev/dmabuf_import_helper MISSING on gke-...-44c95a12-5w60
 [FAIL] 7. Node NICs            only 0 GPU NICs (want 8)
 tier=TCPXO  pass=1  fail=6  warn=1
```

That cluster runs real workloads at the socket-fallback floor and has never emitted an error
about it.

Contrast — the same script against a cluster where the fabric **is** working
(`hypercomputer-a3-tcpxo`, verified 2026-07-28):

```
 [PASS] 1..5 ...
 [PASS] 6. NCCL plugin DS       installed and ready (2)
 [PASS] 6c. rxdm devices        /dev/dmabuf_import_helper present, 8 aperture devices (want 8)
 [PASS] 7. Node NICs            8 GPU NICs + 1 control NIC on gke-...-03hg (want 8)
 [PASS] 8. NCCL transport       NET/FasTrak — GPUDirect ENGAGED
 tier=TCPXO  pass=8  fail=0  warn=0
```

Both runs verbatim: [`assets/lab-22/verify_layer8_fix_controls.txt`](../../assets/lab-22/verify_layer8_fix_controls.txt).
**Always have a known-good and a known-bad output in front of you** — a PASS table means
nothing if you have never seen the script fail.

> **The `pass=8 fail=0 warn=0` above is the only output that fully closes a ticket**, because
> it includes layer 8. Note what it took to get there: that same cluster reported
> `[WARN] 8. NCCL transport — no NCCL transport line found` while its workload log contained
> **153** `NET/FasTrak` lines. Two bugs in the checker, both worth knowing as habits:
> `kubectl logs <pod>` silently defaults to the **first** container (the rxdm sidecar, which
> never prints a transport line — pass `--all-containers`), and `echo "$LOG" | grep -q PAT`
> under `set -o pipefail` returns **141** (SIGPIPE) instead of 0 once the log outgrows the
> pipe buffer, so the `if` takes the false branch *even though the pattern matched*. Use
> `grep -q PAT <<<"$LOG"`. Both defects made the tool report **"no evidence"** when the
> evidence was in hand — the one failure mode a diagnostic tool must never have.

> **A third bug, and the reason the two `CLUSTER=`/`ZONE=` invocations above are safe to
> trust.** Those variables originally steered only the **`gcloud`** layers (1, 2, 4, 5); the
> **`kubectl`** layers (tier detection, 3, 6, 6c, 7, 8) kept reading the *current context*.
> Targeting cluster A while your kubeconfig pointed at cluster B produced a single table
> spliced from both — layer 1 `FAIL` from A, layer 6c `PASS` from B, and the tier taken from
> B's machine family, which silently changed the expected NIC count and transport pattern for
> every check. The output looked entirely plausible. **If you run this script against a
> cluster that is not your current context, confirm the `-- kubectl pinned to context:` line
> names the cluster you asked for**; if the script cannot resolve one it now exits `2` rather
> than answering about the wrong system. The same fix applies to
> `collect_fabric_bundle.sh` — an unpinned bundle is *named* for one cluster and *contains*
> another's Pods and logs, which is how a ticket gets triaged against the wrong system.
> Generalised: any tool that takes a target as a parameter must honour it in **every** backend
> it queries — pin once at a wrapper, never per call site.

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
| 6c | **`/dev/dmabuf_import_helper` + `/dev/aperture_devices/<BDF>` on the node** (TCPXO) | rxdm sidecar dies **exit 0**; Pod stuck `1/2`; fabric moves nothing | ✅ apply the *official* installer + `nri-device-injector` |
| 7 | Realised per-node NIC count | pool built wrong | ❌ recreate the pool |
| 8 | **NCCL's actual transport choice** | *this is the verdict* | — |

Layer **6c** was added after a bring-up that passed layers 1–7 and still could not move a
byte (§4.6). If you only remember one addition to this playbook, remember that layer 6
("plugin DaemonSet is Ready") and layer 6c ("the devices the plugin needs actually exist")
are **different layers** and fail independently.

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
# --all-containers is NOT optional: without it kubectl reads only the FIRST container,
# which on a TCPXO pod is the rxdm sidecar — a container that never logs a transport line.
kubectl logs -n <ns> <pod> --all-containers --tail=4000 \
  | grep -E 'NET/(Socket|IB|FasTrak|GPUDirectTCPX)|GPU Direct RDMA'
```

| What you see | Verdict |
|---|---|
| `NET/FasTrak` / `NET/GPUDirectTCPX` / `NET/IB` | ✅ GPUDirect engaged. Slowness is **above** the fabric — data loading, `NCCL_ALGO`, batch size, job topology. |
| `NET/Socket` | ❌ **Silent fallback.** This is the bug. Work §2 top-down. |
| `GPU Direct RDMA Disabled` | ❌ Usually a missing **rxdm sidecar** (§4.3). |
| nothing | ⚠️ Debug env not set, **or the log never reached pod stdout** — see below. You have no evidence yet. |

> **A third reason for "nothing", easy to miss.** If the customer launched the job with
> `kubectl exec` (or any interactive shell), NCCL's output went to *that session*, not to the
> container's stdout — so `kubectl logs` is legitimately empty even though the run was
> perfectly healthy. Ask how the job was started. When driving a benchmark by hand, tee it
> into PID 1's stdout so the evidence survives the session:
>
> ```bash
> ... torchrun ... 2>&1 | tee /proc/1/fd/1 > /work/run.log
> ```
>
> This is also why "capture while the node is alive" matters more than it sounds: on Flex
> capacity you get one shot at the evidence.

> **Flex-start warning.** On a Flex-start / DWS pool the node returns to the pool when the
> workload ends, taking its NIC list, `dmesg` and NCCL logs with it. **Capture while the
> node is alive** — run `scripts/collect_fabric_bundle.sh` before letting the pool scale to 0.

### 3.1 What a healthy TCPXO log looks like

Captured on `hypercomputer-a3-tcpxo`, 2026-07-28
([`tcpxo_transport_fastrak.txt`](../../assets/lab-22/tcpxo_transport_fastrak.txt)) — this is
the reference to diff a customer's log against:

```
NCCL INFO Using network FasTrak
NCCL INFO Initializing network FasTrak, version: 1.0.17
# and the count that closes the ticket:
$ grep -c 'NET/Socket' rank0.log
0
```

Two further checks worth making in the same pass:

```bash
# 1. the NIC list must be AUTO-DISCOVERED, not hand-written
kubectl exec -n <ns> <pod> -- env | grep -E 'FASTRAK_IFNAME|CTRL_DEV|SOCKET_IFNAME'
#   NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
#   NCCL_FASTRAK_CTRL_DEV=eth0        NCCL_SOCKET_IFNAME=eth0
# 2. jumbo MTU must be realised INSIDE the pod, not just on the VPC
kubectl exec -n <ns> <pod> -- sh -c 'for i in /sys/class/net/eth*; do echo "$i $(cat $i/mtu)"; done'
#   eth0 1460   eth1..eth8 8244
```

The env block must come from **sourcing the vendor's `nccl-env-profile.sh`**, which discovers
the interfaces on the node it runs on. A hand-written `NCCL_FASTRAK_IFNAME` is a latent bug —
correct until the machine shape changes. (`ip` is often absent from CUDA images; read
`/sys/class/net/` instead of reaching for `ip link`.)

> **Do not triage by counting "error" lines.** A healthy TCPXO run on this platform produced
> **63** lines matching `-i 'error|abort'` and every one was benign: 47 `abort` + 16 `Abort`
> are normal communicator teardown (`Abort START`), and the `NCCL WARN` lines are the config
> checker's advisories (`NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY ... expected unset`,
> `NCCL_LIB_DIR`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, CPU affinity). Escalating on an error
> *count* is a false-positive generator. Quote the **transport line** instead.

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
```

**And the version is not free choice — it is pinned to the node's GPU driver.** The plugin
and the rxdm daemon are versioned separately *from each other* and *both* track the driver
release train:

| Node GPU driver | rxdm `tcpgpudmarxd-dev` | plugin `nccl-plugin-gpudirecttcpx-dev` |
|---|---|---|
| **580.65.06+** (R580) | **`v1.0.22`** | **`v1.0.16`** |
| 595.71.05+ (R595) | `v1.0.23` | `v1.0.17` |

Read the driver first, then choose the row:

```bash
kubectl get node <node> -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-gpu-driver-version}{"\n"}'
```

A mismatch here does **not** fail at image pull — it fails at the device layer with §4.10's
signature, which is far harder to read. This is the most expensive version trap on the
platform.

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

### 4.10 rxdm dmabuf importer failure — **OBSERVED 2026-07-28** · the worst fail-open

**Read this one first on any TCPXO ticket.** It defeats layers 1–7 entirely: a cluster in
this state passes every configuration check in §2 and cannot move a byte over the fabric.

**Signature.** The workload Pod sits at **`READY 1/2`** with `STATUS Running`, forever. No
restarts, no Warning events, no `CrashLoopBackOff`. `kubectl describe pod` on the sidecar:

```
    State:          Terminated
      Reason:       Completed        <-- NOT Error
      Exit Code:    0                <-- NOT 1
      Restart Count: 0
```

Only the sidecar's own logs disagree with every Kubernetes-level signal:

```
Failed to create dmabuf importer context
Memory importer: failed to initialize addr translator on NIC IP 192.169.0.3
Exiting with result:1                <-- logs say 1; the process exits 0
```

**Mechanism.** rxdm registers GPU memory with the NICs through two node-level device nodes:

| Device | What it is |
|---|---|
| `/dev/dmabuf_import_helper` | misc char device `10:260`, registered by the out-of-tree **`import-helper`** kernel module |
| `/dev/aperture_devices/<BDF>` | one bind-mount per GPU-NIC PCI function (vendor `1ae0:0084`) — 8 on A3 Mega |

Both are created by the **official** `nccl-tcpxo-installer` DaemonSet's `pre-installation`
**initContainer**, which does `nsenter -at 1 -- modprobe import-helper` and then the aperture
bind-mount loop. An installer that runs only

```bash
/scripts/container_entry.sh install --install-nccl
```

installs `libnccl-net.so` correctly — layer 6 goes **green** — and never creates the devices.
rxdm then cannot set up address translation, gives up, and the fabric is dead while every
dashboard is clean.

Reproduced identically on rxdm `v1.0.17` **and** `v1.0.23`, which is why this is easy to
misdiagnose as a version mismatch. It is not. (Version pairing still matters — §4.1.)

**Verify** — two commands on the node, or just run layer 6c of the checker:

```bash
ls -l /dev/dmabuf_import_helper        # want: crw------- 1 root root 10, 260
ls /dev/aperture_devices/ | wc -l      # want: 8 on A3 Mega (4 on A3 High)
```

> **Do not use `kubectl debug node/<n> -- ls /dev` in a script.** Without a TTY it returns
> no stdout at all, which reads as "device missing" and yields a confident false FAIL. Use an
> ordinary Pod with `/dev` hostPath-mounted read-only and read `kubectl logs`.

**Fix.** Apply the **official** manifests, unmodified — both of them:

```bash
kubectl apply -f manifests/tcpxo/nccl-tcpxo-installer.yaml   # creates the devices
kubectl apply -f manifests/tcpxo/nri-device-injector.yaml    # injects them into the sidecar
```

The injector is what honours the workload Pod annotation that gives the rxdm sidecar the GPUs
and the helper device:

```yaml
metadata:
  annotations:
    devices.gke.io/container.tcpxo-daemon: |+
      - path: /dev/nvidia0        # ... through /dev/nvidia7
      - path: /dev/nvidiactl
      - path: /dev/nvidia-uvm
      - path: /dev/dmabuf_import_helper
```

Workload containers additionally need `NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices`
and the matching `hostPath` mount. Working reference:
[`manifests/tcpxo/workbench-tcpxo.yaml`](../../manifests/tcpxo/workbench-tcpxo.yaml).
Full signature: [`assets/lab-22/tcpxo_failure_dmabuf_importer.txt`](../../assets/lab-22/tcpxo_failure_dmabuf_importer.txt).

**Support takeaway.** On this platform, **`READY 1/2` on a GPU Pod is a fabric symptom**, and
**exit code `0` does not mean success** for the rxdm sidecar. Read container logs, never just
container state.

### 4.11 `UnexpectedAdmissionError` on a GPU Pod — `nodeName` pinning, not a capacity bug

```
UnexpectedAdmissionError: Allocate failed due to requested number of devices
unavailable for nvidia.com/gpu. Requested: 8, Available: 0
```

**Mechanism.** The Pod used `nodeName:` to target a node directly. `nodeName` **bypasses the
scheduler**, so the Pod is never queued: kubelet admits it at once, finds the GPUs already
allocated, and **rejects** it terminally. The same Pod with a `nodeSelector` would simply sit
`Pending` until GPUs free up.

**Why support cares.** The message says "devices unavailable", so it reads as a stockout or a
device-plugin fault. It is neither — it is a Pod-spec bug, and it is 100% reproducible.

**Fix.** Use scheduler placement (`nodeSelector` on
`cloud.google.com/gke-accelerator`, plus `podAntiAffinity` on `kubernetes.io/hostname` for
one-rank-per-node). This also makes handovers gap-free: the new Pods queue *before* the
incumbent releases its GPUs.

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
  |     exit 1 + layer 6c FAIL   -> hand-rolled installer; /dev/dmabuf_import_helper missing (§4.10).
  |     exit 0                   -> config is right; go to 2. DO NOT STOP HERE.
  |
  1b. TCPXO pod stuck at READY 1/2, sidecar "Completed" exit 0?
  |     -> §4.10. This is a FABRIC failure, not a scheduling one. Highest-yield check on TCPXO.
  |  Pod in UnexpectedAdmissionError "devices unavailable"?
  |     -> §4.11. nodeName bypasses the scheduler. Pod-spec bug, not a stockout.
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
                                      Reference: 317.84 GB/s busbw, 16 GPU, 2x a3-megagpu-8g (§0).
  |
  4. Still unresolved -> NAMESPACE=.. POD=.. scripts/collect_fabric_bundle.sh, attach tarball, paste SUMMARY.txt.
```

---

## 8. Why this platform's own clusters fail the check

Honest self-assessment, because support will encounter these exact clusters:

| Cluster | Machine | Tier possible | Fabric status |
|---|---|---|---|
| `hypercomputer-a3-asiaeast1` | `a3-highgpu-8g` | TCPX | ❌ single-gVNIC — no Dataplane V2. **Create-time; unfixable in place.** |
| `hypercomputer-a3-asiasoutheast1` | `a3-megagpu-8g` | TCPXO | ❌ single-gVNIC — no Dataplane V2. Runs real workloads at the socket-fallback floor. |
| `hypercomputer-a3-tcpx` | `a3-highgpu-8g` | TCPX | ✅ gates cleared: DPv2 + multi-networking + 4 GPU nets + CRDs live. Capacity-gated; **no throughput figure claimed**. |
| `hypercomputer-a3-tcpxo` | `a3-megagpu-8g` | TCPXO | ✅ **fully proven end-to-end**: 8 GPU nets, 9 NICs on-node, layer 6c devices present, `NET/FasTrak`, **317.84 GB/s busbw @ 16 GPU**. |

The two production clusters are the reason this playbook exists: they were built before
the create-time gates were understood, they have never emitted a single error about it,
and the cost is a permanent **≈13×** inter-node penalty that only an explicit check reveals.
The last row is the control that makes that number real — same GPU model, same benchmark,
same region, **317.84 GB/s vs 23.7 GB/s**, and the only difference is the eight layers.

> **What is still not proven here.** The TCPX (A3 High) row has never carried traffic — its
> gates are verified, its bandwidth is not. And 317.84 GB/s was measured **without** NUMA/rail
> CPU pinning (the run logged `CPU affinity ... not a subset`), so treat it as a floor for a
> healthy fabric, not a target to certify hardware against.
