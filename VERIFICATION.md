# Verification Log

Provenance for every live run. Appended automatically by `scripts/lib_capture.sh`.

| UTC timestamp | lab | artifact | nodes | note |
| :--- | :--- | :--- | :--- | :--- |
| 2026-07-21T09:52:01Z | lab-01 | assets/lab-01 | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6 | arch inspect |
| 2026-07-21T10:47:33Z | lab-02 | driver/cuda/dcgm/gpu-burn diagnostics | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6 | driver 535.309.01, CUDA 12.2 |
| 2026-07-21T11:03:00Z | lab-03 | gemm.csv,nsys-stats.txt,nvbandwidth.txt,ncu-output.txt | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6 | single-GPU-profile |
| 2026-07-22T03:07:26Z | lab-11 | assets/lab-11 | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6 | platform-compare: HGX baseboard confirmed; gVNIC only; no FM/DPU/SuperNIC; 4-GPU NV18 mesh |
| 2026-07-22T05:23:12Z | lab-04 | assets/lab-04 | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hv7m | intra-node NVLink: 8-GPU NV18 mesh, 18x26.562GB/s links; single-node all-reduce peak busbw ~479.9 GB/s @ 8GiB |
| 2026-07-22T05:23:12Z | lab-06 | assets/lab-06 | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hv7m,gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6 | 2-node 16-GPU all-reduce; transport NET/Socket eth0, NET/IB none, GPU Direct RDMA Disabled; peak busbw ~28.6 GB/s @ 1GB |
| 2026-07-22T05:23:12Z | lab-09 | assets/lab-09 | gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hv7m,gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6 | 2-node 16-GPU DDP+FSDP; DDP 390.26 ms/step, FSDP 537.56 ms/step; profiler: all-reduce 89.63% CUDA time |
