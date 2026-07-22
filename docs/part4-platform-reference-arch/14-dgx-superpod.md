# 14: DGX SuperPOD — Reference Architecture, Scalable Units, and Orchestration

**Knowledge-first / reference architecture.** This describes NVIDIA's DGX SuperPOD reference design; our environment is a 2-node GCP A3 cluster, not a SuperPOD. Nothing here was run on SuperPOD hardware; the contrast to our A3 cluster is the takeaway.

---

## Introduction

A **DGX SuperPOD** is NVIDIA's validated reference architecture for large-scale AI infrastructure, designed to deliver turnkey, production-ready GPU clusters at datacenter scale. Unlike ad-hoc GPU deployments or cloud instances where the operator assembles components from multiple vendors, a DGX SuperPOD is a **pre-integrated, validated system** with defined hardware configurations (compute, storage, networking), management software (Base Command Manager, Run:ai, or Slurm), and acceptance testing criteria that NVIDIA guarantees will achieve published performance benchmarks (e.g., MLPerf results).

The SuperPOD architecture scales from tens to thousands of GPUs using a **scalable unit (SU)** design — a repeatable building block (typically 32 DGX H100 nodes, ~256 H100 GPUs, plus leaf switches, storage, and management infrastructure) that can be replicated to grow the cluster. Each SU is connected via a **spine-leaf fabric** (Quantum InfiniBand or Spectrum-X Ethernet), and the entire pod is managed as a single orchestrated resource pool. SuperPODs are deployed by enterprises, research labs, and cloud providers (including GCP's internal infrastructure for AI services) where predictable, validated, high-utilization AI workloads justify the capital investment and operational overhead of a purpose-built platform.

This document explains the SuperPOD architecture, contrasts it with the GCP A3 cluster we've used for the live labs (Parts I–III), and maps each SuperPOD component to its cloud equivalent. The goal is to make the **concepts portable** — whether you're debugging NCCL collectives on a GKE A3 cluster or a DGX SuperPOD, the mechanisms (NVLink, RDMA, gang scheduling, DCGM diagnostics) are identical; only the infrastructure layer (compute fabric, orchestration, management) differs.

---

## The Scalable Unit (SU): Repeatable Building Block

The **scalable unit (SU)** is the fundamental design element of a DGX SuperPOD. An SU is a self-contained cluster slice with a fixed number of DGX systems, leaf switches, and storage nodes, sized to fit within a single spine-leaf network tier. The specific SU configuration depends on the GPU generation and fabric choice, but the principle is consistent: **replicate the SU to scale the pod without re-engineering the network topology or management layer**.

### Typical SU Configuration (DGX H100 SuperPOD)

A common SU for a DGX H100 SuperPOD (as documented in NVIDIA's reference architecture) comprises:

- **32 DGX H100 systems** (8 H100 GPUs per node = 256 H100 GPUs per SU).
- **Quantum InfiniBand fabric** (rail-optimized; discussed below) or **Spectrum-X Ethernet** fabric.
- **InfiniBand leaf switches** (e.g., 8 Quantum-2 QM9700 switches for a 32-node IB SU; each DGX connects to multiple switches for rail optimization).
- **Storage fabric** (separate high-bandwidth network to parallel filesystem; e.g., VAST, DDN, WekaIO, or NFS on RDMA).
- **Management network** (in-band or out-of-band Ethernet for Base Command Manager, OS provisioning, monitoring, and BMC/IPMI access).

The SU is the **unit of replication**: a 4-SU SuperPOD has 128 DGX H100 nodes (1024 H100 GPUs), built by deploying 4 identical SUs and connecting their leaf switches to a shared spine layer. NVIDIA publishes validated topologies for 1-SU (32 nodes), 2-SU (64 nodes), 4-SU (128 nodes), and larger configurations, each with documented acceptance criteria (NCCL bandwidth, HPL Linpack results, MLPerf training time).

### Why Scalable Units Matter

The SU design solves a critical problem in large-scale GPU clusters: **bisection bandwidth**. As you add nodes, the aggregate bandwidth between arbitrary subsets of nodes (the bisection bandwidth) determines whether multi-node training jobs can scale efficiently. A naive flat topology (all nodes connected to a single switch tier) saturates the uplink bandwidth once the cluster exceeds a few dozen nodes. The SU + spine-leaf model provides **predictable, non-blocking bandwidth** within an SU (all nodes in an SU can communicate at full link rate via leaf switches) and **controlled oversubscription** between SUs (spine switches provide a known oversubscription ratio, typically 2:1 or 4:1, depending on the fabric).

**Contrast with GCP A3:** GCP's Jupiter fabric is a **custom multi-stage Clos network** with high bisection bandwidth across the entire availability zone (not just a single pod). The fabric is shared across all tenant workloads and dynamically routes GPU traffic using Google's in-network congestion control (not rail-optimized IB or Spectrum-X). There is no tenant-visible "SU" — GKE schedules pods across any available A3 nodes in the pool. The **Dynamic Workload Scheduler (DWS)** provisions capacity on-demand and holds nodes for 7 days, but the fabric topology and oversubscription ratios are opaque to the tenant. From the tenant's perspective, all nodes in the GKE cluster are "one hop away" (modulo GKE topology hints for rack/zone affinity). This is **elastic by design** — you pay for what you provision, and GCP handles the underlay fabric — whereas a SuperPOD SU is a **fixed capital asset** optimized for maximum utilization (you own all 32 nodes in an SU whether you're running jobs on them or not).

---

## Three Fabrics: Compute, Storage, and Management

A DGX SuperPOD is not a single network; it is **three overlapping fabrics**, each optimized for a different traffic pattern:

### 1. Compute Fabric (GPU-to-GPU)

The **compute fabric** is the rail-optimized, ultra-low-latency network that carries NCCL collective traffic between GPUs across nodes. This is where **Quantum InfiniBand** or **Spectrum-X Ethernet** (with RoCE/GPUDirect-RDMA) appears. The compute fabric is designed for **lossless, in-order delivery** of GPU memory reads and writes, with hardware congestion control (PFC/ECN on Ethernet, credit-based flow control on InfiniBand) and adaptive routing to avoid hotspots.

Key properties of the compute fabric:

- **400–800 Gbps per port** (depending on IB generation: HDR = 200 Gbps, NDR = 400 Gbps, XDR = 800 Gbps per lane × 4 lanes = 3.2 Tbps theoretical; Spectrum-X = 400 Gbps RoCE per NIC).
- **Rail-optimized topology** (forward link: `./13-spectrum-x-and-fabrics.md`): each GPU NIC or GPU-attached network link connects to a **dedicated rail** (a non-overlapping tree of switches). A DGX H100 has 8 GPUs and 8 network ports (one per GPU); each port connects to a different rail. This design ensures that all-to-all NCCL collectives distribute traffic evenly across rails, maximizing bisection bandwidth and avoiding single-rail congestion. The rail-optimized model is the **key differentiator** of InfiniBand and Spectrum-X fabrics for AI workloads.
- **In-network computing** (InfiniBand SHARP): Quantum InfiniBand switches support **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)**, an in-network aggregation engine that offloads all-reduce operations to the switch fabric. Instead of GPUs sending data to a root node for reduction and broadcasting the result, the switches perform the reduction **in flight**, cutting the effective latency and freeing GPU/CPU cycles. SHARP is **not available on Spectrum-X or GCP TCPX/RoCE**; it is unique to InfiniBand.

**Contrast with GCP A3:** The GCP compute fabric for A3 High/Mega uses **GPUDirect-TCPX** (a GCP-proprietary NCCL plugin that offloads GPU memory DMA to the **Titanium** network offload ASIC; discussed in Part II). TCPX is designed for low latency and high aggregate bandwidth on the multi-NIC A3 GPU network, but it is **not rail-optimized** in the tenant-visible sense — the gVNIC appears as a single 200 Gbps pipe, and traffic striping across multiple rails (if present in the underlay) is handled by Titanium and the Jupiter fabric controller, not by NCCL topology hints. A3 Ultra/A4 (with ConnectX-7 RoCE) is **closer to the SuperPOD model** — the NIC exposes multiple RDMA-capable virtual functions (VFs), and NCCL can stripe traffic across them — but the fabric is still Google-managed, not tenant-configured.

### 2. Storage Fabric (High-Throughput to Parallel Filesystem)

The **storage fabric** is a separate high-bandwidth network (often InfiniBand or 100 Gbps Ethernet) connecting DGX nodes to a parallel filesystem (VAST, DDN EXAScaler, WekaIO, or NFS/RDMA). Large training jobs write checkpoints, logs, and profiling data to shared storage at multi-GB/s rates; the storage fabric must sustain this bandwidth without starving the compute fabric. In many SuperPODs, the storage fabric is also RDMA-capable (RDMA over InfiniBand or RoCE for NFS), but it is **physically separate** from the compute fabric (different switches, different subnets) to prevent storage bursts from congesting GPU collectives.

**Contrast with GCP A3:** GKE nodes mount **Google Cloud Storage (GCS)** (via gcsfuse or Filestore) or **Persistent Disks** (block storage). There is no tenant-visible "storage fabric" — GCS/Filestore traffic shares the same gVNIC and Jupiter fabric as GPU traffic, but Google's congestion control prioritizes latency-sensitive GPU traffic (NCCL) over bulk storage writes. Checkpointing to GCS is asynchronous (written to local disk or memory buffer first, then uploaded in the background) to avoid blocking training. For high-performance storage, tenants deploy **parallel filesystems in the cluster** (e.g., Lustre on Persistent Disk, or third-party appliances like WekaIO on GCP Compute instances with local NVMe), approximating the SuperPOD storage fabric model.

### 3. Management and In-Band/Out-of-Band Fabrics

The **management fabric** is a low-bandwidth Ethernet network (typically 1–10 Gbps) for:

- **Base Command Manager (BCM)** or **Run:ai** control plane (job submission, scheduling, health monitoring).
- **OS provisioning** (PXE boot, image deployment, firmware updates).
- **IPMI/BMC** (out-of-band access to server management controllers for power cycling, BIOS configuration, sensor data).
- **NVSM (NVIDIA System Management)** telemetry (GPU, NVLink, NVSwitch health; logs aggregated to BCM).

The management fabric is **separate from the compute fabric** to ensure that management traffic (firmware updates, log collection, DCGM metrics scraping) does not interfere with NCCL collectives. In some SuperPODs, the management network is **out-of-band** (a physically separate Ethernet network with no route to the compute fabric), while in others it is **in-band** (shares the same Ethernet switches but uses a different VLAN or subnet). The choice depends on security and availability requirements (out-of-band management allows you to diagnose a node whose OS has crashed or whose compute NIC is down, but requires duplicate cabling and switches).

**Contrast with GCP A3:** GKE's control plane (API server, scheduler, controller-manager, kubelet) runs on **GCP-managed infrastructure** (not on the GPU nodes themselves). The GPU nodes connect to the GKE control plane over Google's internal network (opaque to tenants). **Cloud Logging** and **Cloud Monitoring** collect logs and metrics (DCGM via `dcgm-exporter` DaemonSet, or GKE-native GPU metrics) over the same network. There is no tenant-visible IPMI/BMC access — if a node fails, GKE marks it as `NotReady`, DWS provisions a replacement, and the user never sees the hardware layer. This **abstracts away the management fabric** — convenient for operations, but opaque for deep debugging (you can't power-cycle a node or check BMC sensors; you file a GCP support ticket instead).

---

## Rail-Optimized Topology and Multi-Rail NCCL

The **rail-optimized topology** is the signature design pattern of DGX SuperPOD compute fabrics (both Quantum InfiniBand and Spectrum-X Ethernet). Each GPU or GPU-attached NIC connects to a **dedicated rail** — a non-overlapping tree of leaf and spine switches. For a DGX H100 with 8 GPUs and 8 network ports (or 4 dual-port ConnectX-7 NICs, each port mapped to 2 GPUs), each port connects to a different rail. When NCCL performs an all-reduce, it stripes the data across all 8 rails (each rail carries 1/8 of the traffic), distributing the load evenly and maximizing bisection bandwidth.

### How Rail-Optimization Works

Consider a 32-node DGX H100 SU with 8 rails:

1. Each DGX has 8 network ports (one per GPU or per GPU pair, depending on NIC configuration).
2. Each port connects to a **separate rail** (Rail 0 through Rail 7).
3. Each rail is a **separate leaf-spine tree**: Rail 0 has its own leaf switch(es) and spine switch(es), Rail 1 has its own, etc.
4. The 8 rails are **non-overlapping** — traffic on Rail 0 never competes with traffic on Rail 1 for the same switch port or link.
5. When NCCL calls `ncclAllReduce` on a 32-node job, it **stripes the collective across all 8 rails** — each rail sees 1/8 of the total traffic, so the aggregate bandwidth is 8 × (single-rail bandwidth).

This is why InfiniBand and Spectrum-X fabrics achieve **near-linear multi-node NCCL scaling** (bus bandwidth ≈ algo bandwidth for large all-reduce; see `nccl-tests` results in NVIDIA SuperPOD whitepapers). The rail-optimized model is also why **NCCL topology files** (generated by `nvidia-smi topo` or by BCM/NVSM on DGX systems) are critical — NCCL needs to know which NIC port corresponds to which rail to stripe traffic correctly.

**Contrast with GCP A3:** GCP A3 High/Mega (TCPX) does **not expose multiple rails to NCCL**. The gVNIC appears as a single 200 Gbps interface; traffic is striped across the underlay Jupiter fabric by **Titanium** (Google's offload ASIC) and the fabric controller, not by NCCL. The user cannot provide a NCCL topology file to manually control rail affinity (the topology is fixed and opaque). A3 Ultra/A4 (with ConnectX-7 RoCE) is **closer to the rail model** — the NIC exposes multiple virtual functions (VFs), and NCCL can discover them via `ibv_devinfo` — but the fabric is still **cloud-managed**: you don't configure the switch topology or VLAN mappings; GCP provisions the network, and NCCL auto-discovers what's available. The **result is comparable** (A3 Ultra/A4 RoCE all-reduce bandwidth is within 10–20% of DGX H100 InfiniBand for large messages), but the **operational model differs** (cloud-elastic vs fixed SU capacity).

---

## NVLink Switch System and GB200 NVL72: Scale-Up Domains

In addition to **scale-out** (connecting multiple DGX nodes via network fabrics), DGX SuperPODs increasingly rely on **scale-up** — extending the NVLink domain beyond a single HGX baseboard (8 GPUs) to **72 GPUs or more in a single coherent NVLink fabric**. This is the **NVLink Switch System** (also called **NVSwitch 2.0** or **external NVSwitch**) and its flagship deployment: the **GB200 NVL72** (72 Blackwell GPUs in one NVLink domain).

### What is an NVLink Switch System?

The **NVLink Switch System** is a set of external NVLink switch trays (not to be confused with the NVSwitch chips **inside** an HGX baseboard) that connect multiple HGX baseboards (or Grace-Blackwell superchips) into a single NVLink domain. Each tray contains multiple NVSwitch chips; the trays are cabled to the GPU baseboards via high-speed copper or fiber NVLink cables. The system operates as a **two-tier NVLink fabric**:

1. **Tier 1 (intra-baseboard):** 8 GPUs on an HGX baseboard connect via NVSwitch chips on the baseboard (same as a DGX H100 or GCP A3 node).
2. **Tier 2 (inter-baseboard):** Multiple baseboards connect to the external NVLink Switch System, which forwards NVLink traffic between baseboards.

The result is a **single flat NVLink address space**: GPU 0 on baseboard A can perform a direct NVLink read from GPU 15 on baseboard B at full NVLink bandwidth (~900 GB/s for Hopper, ~1.8 TB/s for Blackwell) without traversing an InfiniBand or Ethernet network. This is **scale-up**, not scale-out — the GPUs are not networked; they are **coherent peers** in the same NVLink domain.

### GB200 NVL72: 72 GPUs, One Domain

The **GB200 NVL72** is NVIDIA's flagship NVLink scale-up product:

- **72 Blackwell GPUs** (or 36 Grace-Blackwell superchips, each with 2 GPUs + 1 Grace CPU).
- All 72 GPUs connected via **NVLink Switch System** into a single NVLink domain.
- **~130 TB/s aggregate bisection bandwidth** within the domain (72 GPUs × ~1.8 TB/s per GPU ÷ 2).
- **Grace CPU coherency:** Each Grace-Blackwell superchip's CPUs share a coherent memory space with their 2 attached GPUs via NVLink + Coherent Hub Interface (CHI). The CPUs can directly read/write GPU memory (HBM) at NVLink bandwidth, and vice versa, with cache coherence.

The NVL72 domain is typically deployed as **one rack or two racks** of Grace-Blackwell systems with NVLink Switch trays. A single NVL72 domain can run a 72-GPU model-parallel or pipeline-parallel training job **without NCCL** — the framework (Megatron-LM, DeepSpeed, JAX/GSPMD) issues direct GPU-to-GPU memory copies via NVLink, bypassing the network stack entirely. For collectives, NCCL can use the NVLink domain as a **single "node"** (intra-domain collectives use NVLink; inter-domain collectives use InfiniBand or Ethernet).

**Contrast with GCP A3/A4:** GCP A3/A4 nodes use **single HGX baseboards** (8 GPUs per node). There is no tenant-visible NVLink Switch System. However, **GCP A4X** (announced for GA in 2025) will offer **GB200 NVL domains** — the architecture mirrors NVL72 (Google provisions a multi-node NVLink domain via their NVLink Switch infrastructure), but the domain is **Google-managed** (tenants request "N GPUs in an NVL domain" via GKE, and GCP provisions the underlying NVLink Switch fabric). The **user-visible API** is the same (PyTorch/JAX see 72 GPUs as CUDA devices), but the **operational boundary** differs (on-prem, you cable and configure the NVLink Switch trays yourself; on GCP, you request the domain size, and GCP provisions it). See [`./11-dgx-hgx-systems.md`](./11-dgx-hgx-systems.md) for HGX architecture details and [`../toolkit/T6-portability-matrix.md`](../toolkit/T6-portability-matrix.md) for the GCP ↔ NVIDIA product mapping.

---

## Management and Orchestration: Base Command Manager, Run:ai, and Validation

A DGX SuperPOD is not just hardware; it includes **management software** that provisions nodes, schedules jobs, monitors health, updates firmware, and validates performance. The two primary management platforms are **Base Command Manager (BCM)** and **Run:ai** (both NVIDIA products), though many research SuperPODs also deploy **Slurm** for HPC-style job scheduling.

### Base Command Manager (BCM)

**Base Command Manager** is NVIDIA's flagship DGX orchestration platform. BCM is a centralized control plane (deployed on dedicated management nodes or VMs) that:

- **Provisions DGX nodes:** Deploys OS images (DGX OS, Ubuntu, RHEL), installs drivers/CUDA, configures network interfaces, and joins nodes to the BCM cluster.
- **Job scheduling:** Accepts job submissions (via CLI, API, or GUI), queues jobs, allocates GPU resources, and launches workloads (Slurm-compatible API, or native BCM jobs).
- **Health monitoring:** Aggregates DCGM metrics, NVSM system health, NVLink/NVSwitch status, and BMC sensor data from all DGX nodes; raises alerts on GPU XID errors, thermal throttling, NVLink degradation, or BMC/IPMI faults.
- **Firmware and driver updates:** Centrally pushes firmware (BIOS, BMC, NIC, GPU) and driver updates to the fleet; validates compatibility and rolls back on failure.
- **User management and quotas:** Integrates with LDAP/AD for user authentication; enforces GPU quotas (user X can use up to 64 GPUs; team Y has priority on 256 GPUs).
- **Telemetry and logging:** Collects logs from all DGX nodes, containers, and jobs; exports metrics to Prometheus/Grafana or third-party monitoring systems.

BCM is **Kubernetes-native** (runs atop K8s, uses custom resources for jobs) but can also integrate with **Slurm** (BCM becomes the Slurm cluster manager, and users submit jobs via `sbatch`). BCM is designed for **enterprise-scale operations** — a single BCM instance can manage thousands of DGX nodes across multiple datacenters.

### Run:ai

**Run:ai** is a Kubernetes-native GPU scheduling and orchestration platform (acquired by NVIDIA in 2025). Run:ai focuses on **multi-tenancy, fairness, and utilization**:

- **Gang scheduling:** Ensures all pods in a multi-node training job are scheduled atomically (same as GKE JobSet + Kueue).
- **Fair-share scheduling:** Allocates GPUs based on quotas, priorities, and historical usage (team A gets 50% of GPUs, team B gets 30%, etc.).
- **GPU fractions:** Allows over-subscription of GPUs (multiple pods sharing one GPU via time-slicing or MIG); useful for inference workloads or small training runs.
- **Preemption:** Lower-priority jobs can be paused or evicted to free GPUs for higher-priority jobs.
- **Telemetry and cost tracking:** Tracks GPU utilization per user, team, and project; generates reports for chargeback/showback.

Run:ai is **lighter-weight than BCM** (no OS provisioning, firmware updates, or IPMI/BMC integration; focuses on scheduling and quota enforcement) and integrates with standard Kubernetes clusters (you install Run:ai as a Helm chart on an existing K8s cluster with GPU nodes). Many DGX SuperPODs deploy **both BCM and Run:ai**: BCM manages the physical infrastructure (node provisioning, health, firmware), and Run:ai manages the workload scheduling and user quotas.

### Slurm

Many academic and research SuperPODs use **Slurm** (the HPC job scheduler) instead of BCM or Run:ai. Slurm is battle-tested for large-scale batch workloads, has deep integration with HPC tools (e.g., MPI launchers, parallel filesystems), and is familiar to HPC users. Slurm's `gres` (Generic RESource) mechanism exposes GPUs to jobs (`sbatch --gres=gpu:8`), and Slurm's gang scheduling (via `salloc` or `sbatch --exclusive`) ensures multi-node jobs get all their nodes at once. Slurm **does not provide GPU health monitoring, firmware updates, or K8s-style orchestration** — operators pair it with DCGM, Prometheus/Grafana, and custom prolog/epilog scripts for health checks.

**Contrast with GCP A3:** GKE is the orchestrator (managed Kubernetes control plane). **JobSet** (multi-pod jobs) and **Kueue** (gang scheduling, queue management) replace BCM/Run:ai's job scheduling. **Dynamic Workload Scheduler (DWS)** replaces BCM's node provisioning (DWS queues GPU capacity requests, auto-provisions A3 nodes, and holds them for 7 days). **Cloud Monitoring** and **dcgm-exporter** replace BCM's telemetry. GCP does **not expose firmware update APIs** (Google manages firmware), **IPMI/BMC access** (you can't power-cycle a node), or **Slurm** (you use GKE, or you deploy Slurm yourself on Compute Engine VMs). The **functional mapping** is:

| DGX SuperPOD | GCP A3 / GKE |
|:---|:---|
| Base Command Manager (job scheduling, node provisioning) | GKE + JobSet + Kueue + DWS |
| Run:ai (multi-tenancy, fair-share scheduling) | Kueue (quota, priority, preemption) |
| Slurm (`gres` GPU scheduling) | GKE device-plugin (`nvidia.com/gpu` extended resource) |
| BCM health monitoring (NVSM, DCGM, BMC telemetry) | Cloud Monitoring + dcgm-exporter DaemonSet |
| BCM firmware updates | GCP-managed (opaque to tenant) |
| IPMI/BMC out-of-band access | Not available (GCP support handles hardware failures) |

See [`../part3-clustering-execution/07-gke-scheduling-topology.md`](../part3-clustering-execution/07-gke-scheduling-topology.md) and [`../part3-clustering-execution/08-job-frameworks-jobset-kueue.md`](../part3-clustering-execution/08-job-frameworks-jobset-kueue.md) (if these docs exist) for GKE scheduling details, or the toolkit doc [`../toolkit/T6-portability-matrix.md`](../toolkit/T6-portability-matrix.md) for the full cross-platform mapping.

---

## Validation and Acceptance Testing

Every DGX SuperPOD must pass **acceptance testing** before NVIDIA certifies it as production-ready. The acceptance suite validates that the system meets published performance benchmarks and that all hardware (GPUs, NVLink, NICs, switches, storage) is functioning correctly. Typical tests include:

### 1. NCCL Bandwidth Tests (`nccl-tests`)

Run `nccl-tests` (all-reduce, all-gather, reduce-scatter) on all nodes in the SU (e.g., 32 nodes, 256 GPUs) to verify that:

- **Bus bandwidth** (raw network bandwidth) matches the expected value (e.g., ~400 GB/s per DGX H100 for 8×50 GB/s NICs).
- **Algo bandwidth** (effective bandwidth after NCCL algorithm overhead) is ≥90% of bus bandwidth for large messages (≥1 GB all-reduce).
- **No stragglers:** All nodes achieve similar bandwidth (no single node bottlenecked by a bad NIC, cable, or switch port).

### 2. HPL Linpack (FP64 Compute)

Run **HPL (High-Performance Linpack)** — a dense matrix factorization benchmark — to verify peak FP64 FLOPS and check for GPU, memory, or NVLink errors under sustained load. HPL is the benchmark used for the TOP500 supercomputer list. A DGX H100 SU (32 nodes, 256 H100 GPUs) should achieve **~15 PFLOPS** sustained (256 GPUs × ~60 TFLOPS FP64 per GPU × ~90% efficiency).

### 3. MLPerf Training

Run **MLPerf Training** benchmarks (e.g., ResNet-50, BERT, GPT-3) to validate end-to-end training performance, including data loading, preprocessing, multi-node collectives, and checkpointing. MLPerf results are **public** (NVIDIA submits DGX SuperPOD results to the MLPerf leaderboard), so customers can compare their acceptance test results to the published baseline.

### 4. DCGM Diagnostics

Run **DCGM diagnostics** (`dcgmi diag -r 3` or higher) on all GPUs to check for:

- ECC errors (correctable or uncorrectable).
- Thermal throttling (GPUs exceeding thermal limits under load).
- NVLink degradation (reduced bandwidth or link-down errors).
- Memory errors (HBM or SRAM faults).

DCGM diagnostics must pass on all GPUs before the system is released to users.

### 5. Fabric Validation (InfiniBand or Spectrum-X)

Run **fabric bandwidth tests** (`perftest` suite: `ib_write_bw`, `ib_send_bw`, etc.) on all NIC ports to verify that:

- Each NIC achieves the expected bandwidth (e.g., 400 Gbps for HDR InfiniBand, 400 Gbps for RoCE on ConnectX-7).
- No packet loss or retransmits (lossless fabric).
- Latency is within spec (e.g., <2 µs for small InfiniBand messages).

**Contrast with GCP A3:** GCP does **not provide acceptance test results** to individual tenants (you cannot request "show me the HPL score for my 2-node A3 cluster"). However, GCP publishes **aggregate MLPerf results** for its A3 and A4 clusters (under the "Google" submitter; see MLPerf Training v4.0 results for A3 Mega ResNet-50 and GPT-3 scores). Tenants can run **their own validation** (NCCL-tests, DCGM diagnostics, custom benchmarks) on their GKE clusters to verify that performance matches expectations. See `VERIFICATION.md` in this repo for the acceptance-style tests recorded so far on the A3 cluster (single-GPU arch inspection, driver/CUDA + DCGM health, single-GPU GEMM and profiling); the multi-GPU and 2-node NCCL sweeps are added to that log as they run.

---

## SuperPOD vs GCP A3: Side-by-Side Comparison

The table below contrasts a DGX H100 SuperPOD (reference architecture) with the GCP A3 cluster used in this guide's labs.

| Dimension | DGX H100 SuperPOD | GCP A3 Cluster (This Guide) |
|:---|:---|:---|
| **Compute Nodes** | 32–256+ DGX H100 systems (256–2048+ H100 GPUs per pod) | 2 × `a3-highgpu-8g` (16 × H100 80GB total) |
| **GPU Fabric (Intra-Node)** | HGX H100 baseboard (8 GPUs, NVSwitch, NVLink 4.0; ~900 GB/s bisection BW) | Same (HGX H100 baseboard; Google-managed, no tenant-visible Fabric Manager) |
| **GPU Fabric (Inter-Node)** | **Quantum InfiniBand** (rail-optimized HDR/NDR; 400–800 Gbps per NIC; 8 rails; SHARP in-network all-reduce) or **Spectrum-X Ethernet** (RoCE; 400 Gbps ConnectX-7 or BlueField-3 SuperNIC; adaptive routing) | **GPUDirect-TCPX** (gVNIC 200 Gbps; rxdm offload to Titanium; NCCL plugin; no rail-optimization exposed to tenant) |
| **Network Topology** | Spine-leaf with scalable units (SU); each SU = 32 nodes + leaf switches; multiple SUs connected via spine; rail-optimized (8 rails per node; each GPU NIC on a separate rail) | Jupiter fabric (Google Clos network; opaque topology; dynamic routing; shared across tenant VMs; no rail-optimization visible to NCCL) |
| **Storage Fabric** | Separate high-BW network (InfiniBand or 100G Ethernet) to parallel filesystem (VAST, DDN, WekaIO, NFS/RDMA) | GCS (via gcsfuse) or Filestore (NFS); shares gVNIC with GPU traffic; or tenant-deployed parallel FS (Lustre, WekaIO on Compute instances) |
| **Management Network** | Dedicated Ethernet (in-band or OOB) for Base Command Manager, IPMI/BMC, OS provisioning, firmware updates | GKE control plane (GCP-managed; opaque network); Cloud Logging/Monitoring (in-band); no tenant IPMI access |
| **Orchestration** | **Base Command Manager** (centralized job scheduling, node provisioning, health monitoring, firmware updates) or **Run:ai** (K8s-native GPU scheduler) or **Slurm** (HPC job scheduler) | **GKE** (managed K8s) + **JobSet** + **Kueue** (gang scheduling) + **DWS** (auto-provisioning with 7-day holds) |
| **Driver & Device Management** | **DGX OS** (pre-installed drivers, NVSM, Fabric Manager; managed by BCM) or **GPU Operator** (on non-DGX K8s) | GKE device-plugin DaemonSet (auto-installs drivers; pinned version per GKE release; no tenant firmware control) |
| **Health Monitoring** | **NVSM** (DGX fleet health GUI; BMC/IPMI sensors; NVLink/NVSwitch health via Fabric Manager) + **DCGM** + BCM telemetry | **dcgm-exporter** DaemonSet → Prometheus → Cloud Monitoring; GKE node health checks; no BMC access |
| **Validation / Acceptance** | NVIDIA-certified acceptance tests (NCCL bandwidth, HPL Linpack, MLPerf, DCGM diagnostics, fabric tests); published results per SU | GCP publishes aggregate MLPerf results (A3 Mega, A4); tenant runs own validation (NCCL-tests, DCGM diag); no per-cluster cert |
| **Capital Model** | Fixed CapEx (you own all nodes, switches, storage in the SU; optimize for utilization) | OpEx (pay per provisioned GPU-hour; DWS holds nodes for 7 days; elastic scale-down when unused) |
| **Operational Model** | On-prem (you manage datacenter, power, cooling, cabling, firmware, OS, network config, BCM/Slurm deployment) | Cloud-managed (Google handles hardware, firmware, datacenter, underlay network; tenant manages GKE workloads) |
| **NVLink Scale-Up (NVL Domains)** | Available (GB200 NVL72 = 72 Blackwell GPUs in one NVLink domain via NVLink Switch System; DGX B200 SuperPODs) | Not available on A3 (8 GPUs per node, no inter-node NVLink); **A4X** will offer GB200 NVL domains (Google-managed) |
| **In-Network Computing** | SHARP (on Quantum IB; offloads all-reduce to switches) | Not available (TCPX uses rxdm offload to Titanium, but no in-network reduction; RoCE on A3 Ultra/A4 has no SHARP equivalent) |

**Key takeaway:** The **mechanisms** (GPU architecture, NVLink/NVSwitch, NCCL algorithms, DCGM diagnostics, Nsight profiling) are **identical** — what you learn debugging NCCL hangs or profiling kernels on A3 applies directly to DGX SuperPODs. The **differences are in the infrastructure layer**: SuperPODs use rail-optimized InfiniBand/Spectrum-X fabrics with SHARP, tenant-configurable topology, and BCM/Slurm orchestration; GCP uses Titanium+TCPX or RoCE, Jupiter fabric with opaque topology, and GKE+DWS orchestration. Both are **production-grade**; the choice depends on whether you value **cloud elasticity and managed operations** (GCP) or **fixed capacity, rail-optimized fabrics, and deep hardware control** (SuperPOD).

---

## Tools and Cross-References

The NVIDIA tool stack is **platform-agnostic** — the same tools (`nvidia-smi`, DCGM, Nsight Systems, NCCL-tests, `perftest`) work identically on DGX SuperPODs, GCP A3, bare-metal Kubernetes, and Slurm clusters. The only differences are in **access patterns** (e.g., on GCP you run DCGM diagnostics inside a Kubernetes pod; on DGX you run it directly on the node or via BCM) and **visibility** (e.g., DGX exposes NVSM and Fabric Manager logs; GCP does not).

See the toolkit references for hands-on usage:

- **[T1: Monitoring & Inventory](../toolkit/T1-monitoring-inventory.md)** — `nvidia-smi`, DCGM, NVML, device topology.
- **[T2: Health Diagnostics](../toolkit/T2-health-diagnostics.md)** — DCGM diagnostics, XID codes, thermal/power monitoring.
- **[T3: Profiling & Tracing](../toolkit/T3-profiling-tracing.md)** — Nsight Systems, Nsight Compute, PyTorch Profiler, NVTX.
- **[T4: Benchmarking](../toolkit/T4-benchmarking.md)** — NCCL-tests, nvbandwidth, perftest (IB/RoCE), GPU microbenchmarks.
- **[T5: Networking & Fabric Tools](../toolkit/T5-networking-fabric-tools.md)** — `perftest`, `ethtool`, `ibstat`/`ibv_devinfo`, NCCL debug flags, topology files.
- **[T6: Portability Matrix](../toolkit/T6-portability-matrix.md)** — Full cross-platform mapping (GCP ↔ DGX ↔ Slurm ↔ bare-metal K8s).

For related architecture docs:

- **[`./11-dgx-hgx-systems.md`](./11-dgx-hgx-systems.md)** — DGX H100/H200 system architecture, HGX baseboard, NVSM, Fabric Manager (if this doc exists).
- **[`./13-spectrum-x-and-fabrics.md`](./13-spectrum-x-and-fabrics.md)** — Spectrum-X Ethernet vs Quantum InfiniBand, rail-optimized topology, SHARP, RoCE (if this doc exists).
- **[`./12-bluefield-dpu-doca.md`](./12-bluefield-dpu-doca.md)** — BlueField DPU architecture, DOCA, offload capabilities.

---

## Summary

A **DGX SuperPOD** is NVIDIA's validated reference architecture for large-scale AI infrastructure, built from repeatable scalable units (SUs) of 32 DGX systems, compute/storage/management fabrics, and Base Command Manager or Run:ai orchestration. The architecture is optimized for **maximum GPU utilization** (fixed capacity, you own all nodes), **rail-optimized fabrics** (InfiniBand SHARP or Spectrum-X RoCE), and **deep hardware control** (NVSM, Fabric Manager, IPMI/BMC access, firmware updates).

In contrast, **GCP A3** clusters are **cloud-elastic**: you provision GPU nodes on-demand (DWS holds them for 7 days), the Jupiter fabric is shared and opaque, and GKE+JobSet+Kueue replace BCM/Run:ai. The **core GPU stack** (H100, HGX, NVLink, NCCL) is identical, so **what you learn on A3 transfers directly to SuperPODs** — the mechanisms are the same, only the infrastructure boundary differs.

The **NVLink Switch System / GB200 NVL72** represents the next frontier: scale-up domains of 72 GPUs in a single coherent NVLink fabric, where all GPUs communicate at NVLink bandwidth without touching the network. This architecture will appear on GCP as A4X (Google-managed NVL domains) and on DGX SuperPODs as GB200 NVL72 (tenant-configured).

**Compare → lab-11 (A3 tenant vs DGX SuperPOD table)**: For hands-on exercises contrasting the A3 cluster with DGX SuperPOD concepts (observing HGX topology, running NCCL tests, exporting NCCL topology, comparing orchestration models), see `labs/lab-11-platform-compare/` (if available).
