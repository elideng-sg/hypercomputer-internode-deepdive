# Glossary

This glossary defines key terms used throughout the guide. Terms are listed alphabetically.

---

## Terms

**Algo bandwidth:** The effective bandwidth of a collective operation as measured by the application, accounting for the reduction in data movement achieved by the collective algorithm (e.g., ring, tree) compared to naive point-to-point transfers.

**Bus bandwidth:** The raw hardware bandwidth available on the interconnect (e.g., NVLink, PCIe, RDMA fabric), independent of the collective algorithm used.

**DPU (Data Processing Unit):** A specialized processor (e.g., NVIDIA BlueField) that offloads networking, storage, and security tasks from the host CPU, often integrated with high-speed NICs.

**DWS (Dynamic Workload Scheduler):** GKE's queued provisioning system for managing GPU node pools with multi-day capacity holds and on-demand scaling.

**Gang scheduling:** A scheduling policy that ensures all pods in a multi-pod job are scheduled simultaneously (atomically) on available nodes, preventing partial scheduling and deadlock.

**GPUDirect:** A family of technologies that enable direct memory access between GPUs and other devices (e.g., NICs, storage, peer GPUs) without involving the CPU, reducing latency and freeing CPU cycles.

**HBM (High Bandwidth Memory):** Stacked DRAM technology providing significantly higher bandwidth and lower power consumption than traditional GDDR memory, used in modern datacenter GPUs (e.g., H100, H200).

**NCCL (NVIDIA Collective Communications Library):** NVIDIA's library for implementing multi-GPU and multi-node collective communication primitives (all-reduce, all-gather, reduce-scatter, etc.) optimized for GPU interconnects.

**NVLink:** NVIDIA's high-bandwidth, low-latency GPU-to-GPU interconnect technology, providing significantly higher bandwidth than PCIe for intra-node GPU communication.

**NVSwitch:** A high-speed switch fabric chip that connects multiple GPUs via NVLink in a fully connected topology, enabling all-to-all GPU communication at full NVLink bandwidth (used in HGX and DGX systems).

**NVL72:** A 72-GPU NVLink scale-up domain (as in GB200 NVL72 / Grace-Blackwell), where all 72 GPUs are linked through NVLink Switch trays into one coherent, all-to-all NVLink-bandwidth domain.

**Rail-optimized:** A network topology design (common in NVIDIA SuperPOD and Quantum InfiniBand fabrics) that segregates traffic into multiple independent "rails" or paths to maximize bisection bandwidth and avoid congestion.

**RDMA (Remote Direct Memory Access):** A networking protocol that allows direct memory access from one host to another without involving the operating system or CPU, minimizing latency and CPU overhead.

**Ring (collective algorithm):** A collective communication algorithm where GPUs are logically arranged in a ring and data is passed sequentially around the ring, efficient for bandwidth-limited scenarios.

**RoCE (RDMA over Converged Ethernet):** An RDMA protocol that runs over Ethernet, enabling low-latency, high-bandwidth communication on standard Ethernet fabrics with lossless extensions (Priority Flow Control, ECN).

**Scalable unit:** A building block in a DGX SuperPOD reference architecture, typically consisting of a fixed number of DGX systems (e.g., 32 DGX H100 nodes) with a well-defined compute, storage, and networking configuration that can be replicated to scale the cluster.

**SHARP (Scalable Hierarchical Aggregation and Reduction Protocol):** An in-network computing technology in NVIDIA Quantum InfiniBand switches that offloads collective reduction operations (e.g., all-reduce) to the network fabric, reducing latency and freeing GPU/CPU cycles.

**SIMT (Single Instruction, Multiple Threads):** NVIDIA's execution model where a single instruction is broadcast to multiple threads (a warp) that execute it in lockstep on different data, enabling massive parallelism.

**SM (Streaming Multiprocessor):** The fundamental execution unit in an NVIDIA GPU, containing CUDA cores, Tensor Cores, shared memory, and scheduling logic; a GPU comprises many SMs operating in parallel.

**SuperNIC:** NVIDIA's next-generation high-performance network adapter (e.g., BlueField-3 SuperNIC) combining RDMA-capable NIC functionality with programmable DPU offload capabilities.

**TCPX / TCPXO (GPUDirect-TCPX / TCPXO):** GCP's GPU networking stacks for A3, both requiring a
cluster created with Dataplane V2 + multi-networking. **TCPX** (A3 High, `a3-highgpu-8g`) gives
NCCL **4 dedicated GPU NICs** at MTU 8244 and DMAs payload straight into GPU HBM via dmabuf — but
the transport is still **host TCP sockets**, so kernel NIC counters *do* see the traffic
(*measured: 83.27 GB/s busbw @ 16 GPU, lab-18*). **TCPXO / FasTrak** (A3 Mega, `a3-megagpu-8g`)
uses **8 NICs** and replaces the transport with a **userspace datapath**, so it is genuinely
kernel-bypass and netdev counters read ~zero under full load (*measured: 317.84 GB/s, lab-22*).
They are not interchangeable: different plugin images, different NCCL env families
(`NCCL_GPUDIRECTTCPX_*` vs `NCCL_FASTRAK_*`), different transport lines
(`NET/GPUDirectTCPX` vs `NET/FasTrak`), and different observability. Both fail **open** — a
misconfigured fabric silently falls back to `NET/Socket` (~23.7 GB/s) with no error anywhere.

**Tree (collective algorithm):** A collective communication algorithm where GPUs are logically arranged in a tree structure and data is aggregated/broadcast hierarchically, efficient for latency-sensitive scenarios and small message sizes.

**Warp:** A group of 32 threads in an NVIDIA GPU that execute together in SIMT fashion; the fundamental scheduling unit of a Streaming Multiprocessor (SM).
