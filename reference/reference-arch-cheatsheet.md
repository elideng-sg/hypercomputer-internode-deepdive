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
