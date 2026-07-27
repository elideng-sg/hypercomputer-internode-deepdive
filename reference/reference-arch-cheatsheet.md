# Reference Architecture Cheat Sheet

This quick reference defines the key NVIDIA purpose-built platform components and reference architectures discussed in Part IV of the guide. Each entry provides a concise one-line definition; detailed explanations, system diagrams, and contrasts with the GCP A3 lab are in the Part IV docs.

---

## DGX System

A purpose-built AI system integrating GPUs, NVSwitch fabric, high-speed networking (InfiniBand or Ethernet), DGX OS, and system management software (NVSM, Fabric Manager) in a single rack-mountable unit optimized for AI workloads (e.g., DGX H100, DGX H200).

---

## HGX Baseboard

The GPU compute module (8 GPUs connected via NVSwitch in a fully connected NVLink topology) that serves as the foundation for DGX systems and is also used in OEM servers and cloud instances (e.g., GCP A3 nodes).

---

## BlueField DPU

A data processing unit (DPU) that offloads networking, storage, and security tasks from the host CPU, often integrated with high-speed RDMA-capable NICs, programmable via the DOCA software framework.

---

## Spectrum-X

NVIDIA's AI-optimized Ethernet networking platform combining Spectrum-4 switches and BlueField-3 SuperNICs, providing adaptive routing, congestion control, and RDMA over Converged Ethernet (RoCE) for GPU clusters.

---

## Quantum InfiniBand

NVIDIA's high-performance InfiniBand fabric with in-network computing (SHARP for collective offload), rail-optimized topologies, and ultra-low latency, designed for large-scale AI and HPC clusters.

---

## DGX SuperPOD

NVIDIA's validated reference architecture for large-scale AI infrastructure, built from scalable units (SUs) of DGX systems with compute, storage, and management fabrics, managed by Base Command Manager or Run:ai, and validated via acceptance tests.

---

## NVLink Switch System / GB200 NVL72

A large-scale NVLink domain (e.g., 72 GPUs or 36 GB200 Grace-Blackwell superchips) connected via NVLink switches, enabling all-to-all GPU communication at NVLink bandwidth across multiple nodes within a single scale-up domain.

---

## GCP GPU-workload patterns (Part VI)

The design patterns for architecting a GPU workload on GKE, each instantiated by a Part VI lab. The recurring rule: **the safe Flex cluster can host the measured rung; the production-scale rung ships as a server-validated reference** (nodes are never recreated/drained, GPUs are never left unheld).

| Concern | Pattern | GCP mechanism | Measured live (lab) | Reference / next rung |
| :--- | :--- | :--- | :--- | :--- |
| **Fabric** | move off plain TCP | multi-network pool + GPUDirect-TCPX/TCPXO DaemonSet | single-gVNIC ~28.6 GB/s floor ([lab-12](../labs/lab-12-multinode-scaling/)) | TCPX pool provisioning + reference ([lab-18](../labs/lab-18-enable-gpudirect-tcpx/), [doc-21](../docs/part6-architecture-gcp-integration/21-gke-network-design.md)) |
| **Storage / data path** | feed the GPU from object storage | **GCSFuse CSI + Workload Identity** (userspace + GSA-key when node-WIF gated) | starved 11.8% vs fed 100% GPU-busy; ~4.9 GiB/s read ([lab-19](../labs/lab-19-storage-data-path/), [doc-22](../docs/part6-architecture-gcp-integration/22-storage-and-data-path.md)) | managed CSI+WIF (needs node `GKE_METADATA`); Filestore/Lustre/Hyperdisk ML tiers |
| **Training pipeline** | data → gang → checkpoints → metrics | **JobSet + Kueue** gang, GCSFuse data+code, checkpoints to GCS, DCGM on GMP | 2-node/16-GPU DDP, loss 263→0.008, 264 MiB ckpts, engine-active plateau ([lab-20](../labs/lab-20-training-pipeline/), [doc-23](../docs/part6-architecture-gcp-integration/23-training-pipeline-jobset.md)) | Artifact Registry image build; async/sharded checkpointing at scale |
| **Inference serving** | hold a latency SLO at min cost | dynamic batching + **HPA-on-DCGM** + cluster autoscaler + Inference Gateway | serving knee (~1.1k req/s, p99→1s); 1→8-GPU 7.24× scaling; GPU only ~17% at the knee ([lab-21](../labs/lab-21-inference-serving/), [doc-24](../docs/part6-architecture-gcp-integration/24-inference-serving-autoscale.md)) | full autoscale topology + Gateway API (`manifests/serving/inference-autoscale.yaml`); Vertex AI managed |
