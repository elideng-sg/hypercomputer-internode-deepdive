# NCCL Tunable Environment Variables

This reference catalogs the NCCL environment variables that control collective communication behavior — debug output, transport selection, topology, algorithm/protocol choice, and buffering. These are the knobs used for **tuning** (squeezing bandwidth out of a given fabric) and **troubleshooting** (forcing a path, or making NCCL tell you what it chose).

> **How to read this table.** *Effect* is the mechanism (what NCCL does). *When to set* is the practical trigger. The **Seen in** column points to the lab where the variable is exercised on the live cluster. Measured before/after deltas (e.g. the Ring-vs-Tree crossover, socket-thread scaling) are recorded in those labs' captured assets, not asserted here — this file is the knowledge map, the labs hold the numbers.
>
> **First rule of NCCL debugging:** set `NCCL_DEBUG=INFO` and *read what NCCL actually chose* (transport, algorithm, protocol, channel count) before changing anything. Almost every "NCCL is slow" question is answered by the `NET/…` and `Channel …` lines, not by guessing a tunable.

---

## Debug & observability

| Env var | Effect | When to set | Seen in |
| :--- | :--- | :--- | :--- |
| `NCCL_DEBUG` | Log verbosity: `WARN` (errors only), `INFO` (init, transport, algo, ring topology), `TRACE` (per-op). | Always start at `INFO` to see the chosen transport/algo; `WARN` in steady state. | lab-06, lab-12, lab-15 |
| `NCCL_DEBUG_SUBSYS` | Filter `INFO` to subsystems: `INIT,NET,GRAPH,TUNING,ENV,COLL`. | Narrow the firehose — `GRAPH` for ring/tree construction, `NET` for transport, `ENV` to confirm which vars NCCL actually read. | lab-12, lab-15 |
| `NCCL_DEBUG_FILE` | Write debug to a per-rank file (`%h`/`%p` expand to host/pid) instead of stderr. | Multi-rank jobs where interleaved stderr is unreadable; capture one file per rank. | lab-12, lab-15 |
| `NCCL_TOPO_DUMP_FILE` | Dump the detected intra-node topology (XML) NCCL built. | Verifying NVLink/PCIe discovery, or feeding a hand-tuned topology back via `NCCL_TOPO_FILE`. | lab-04, lab-15 |

## Transport & network selection

| Env var | Effect | When to set | Seen in |
| :--- | :--- | :--- | :--- |
| `NCCL_SOCKET_IFNAME` | Restrict the TCP transport to specific host interface(s) (prefix match, `^` to exclude). | Multi-NIC nodes where NCCL picks the wrong interface (e.g. a management NIC); the classic "one slow rail" fix. | lab-06, lab-15 |
| `NCCL_SOCKET_NTHREADS` | Number of helper threads per socket connection (TCP transport). | Single TCP flow can't saturate a fast NIC (see lab-05's 1-stream vs 8-stream gap); raise with `NSOCKS_PERTHREAD` to parallelize. | lab-06, lab-12 |
| `NCCL_NSOCKS_PERTHREAD` | Sockets opened per helper thread; total flows = `NTHREADS × NSOCKS_PERTHREAD`. | Same as above — more concurrent TCP flows over a single gVNIC to approach line rate. | lab-06, lab-12 |
| `NCCL_NET_PLUGIN` | Select/disable the external net plugin (`libnccl-net.so`): `none`, or a named plugin (e.g. `gpudirecttcpx`). | Confirming whether the GPUDirect plugin is loaded; `none` forces the built-in socket transport, which is how lab-22 produced a same-Pods A/B of a healthy vs degraded fabric with **one env var**. | lab-05, lab-18, lab-22 |
| `NCCL_NET_GDR_LEVEL` | Threshold controlling GPUDirect RDMA use (GPU↔NIC DMA without host bounce). | RDMA/RoCE fabrics (A3 Ultra/A4); has no effect on the plain-TCP gVNIC path (no GDR there). | (reference — RDMA fabric) |
| `NCCL_IB_HCA` / `NCCL_IB_DISABLE` | Select the InfiniBand/RoCE HCA(s), or disable the IB transport entirely. | RoCE/IB fabrics only; `IB_DISABLE=1` to force the socket path for comparison. | (reference — RDMA fabric) |
| `NCCL_P2P_LEVEL` | Max topological distance for intra-node GPU↔GPU P2P (NVLink/PCIe). | Isolating whether a slow intra-node path (P2P disabled → staged through host) is the culprit. | lab-04 |
| `NCCL_SHM_DISABLE` | Disable the shared-memory transport (host-staged intra-node fallback). | Diagnosing intra-node transport issues; almost never set in production. | (diagnostic) |
| `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME` / `..._CTRL_DEV` | (TCPX plugin) the four GPU-NIC data rails (`eth1,eth2,eth3,eth4`) and the control NIC (`eth0`) for GPUDirect-TCPX. | Only on a multi-network TCPX pool — and **mandatory** there: unlike TCPXO, the TCPX plugin ships **no** `nccl-env-profile.sh` to derive it (G28), so the manifest must carry the list. Cross-check it against the node's `networking.gke.io/nic-info` rather than trusting it. | lab-18 (measured: 4 rails, 264 NCCL refs each) |
| `NCCL_FASTRAK_IFNAME` / `NCCL_FASTRAK_CTRL_DEV` / `NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY` | (TCPXO/FasTrak plugin) the eight GPU-NIC rails, the control NIC, and the aperture-device directory. | A3 Mega only. **Source the vendor's `nccl-env-profile.sh`** instead of hand-writing these — on this tier it exists and discovers the NICs on the node it runs on. `LLCM_DEVICE_DIRECTORY=/dev/aperture_devices` also needs the matching hostPath mount. | lab-22 |
| `NCCL_USE_DMA_BUF` | Register GPU memory with the NIC via **dmabuf** (core NCCL setting; default on). | Diagnosing TCPX registration failures. **This is the real control** — the plausible-looking `NCCL_GPUDIRECTTCPX_USE_DMABUF` is *silently ignored*, so "disabling dmabuf" via that spelling changes nothing. Setting `0` on TCPX does not rescue a driver-incompatible plugin either; it just swaps the error for `p2pdma api won't work with only RegMr, due to alignment issue` (G29). | lab-18 (failure path) |

## Algorithm, protocol & channels

| Env var | Effect | When to set | Seen in |
| :--- | :--- | :--- | :--- |
| `NCCL_ALGO` | Force the collective algorithm: `Ring`, `Tree`, `CollnetChain`, `NVLS`, … (default: NCCL auto-selects by size/topology). | The Ring-vs-Tree study: at ≥3 nodes tree depth grows and the small-message crossover becomes measurable. Force each to see the crossover. | lab-12b |
| `NCCL_PROTO` | Force the wire protocol: `Simple`, `LL` (low-latency), `LL128`. | `LL`/`LL128` cut latency for small messages at a bandwidth cost; pair with `NCCL_ALGO` when mapping the small-message regime. | lab-12b |
| `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS` | Cap/floor the number of parallel rings ("channels"). | Bounding channel count to isolate per-channel bandwidth, or working around a topology that spawns too few/many. | lab-12 |
| `NCCL_BUFFSIZE` | Per-channel transport buffer size (bytes). | Large-message bandwidth tuning; bigger buffers amortize per-chunk overhead on high-latency links. | lab-12 |
| `NCCL_CROSS_NIC` | Allow a ring to cross between NICs/rails (0/1/2). | Multi-rail fabrics (TCPX/RDMA) — controls rail alignment of the ring. Note the underlay rail→switch topology is **not** tenant-visible on GCP, so this tunes NIC choice, not switch affinity. | lab-18 |

## Timeouts & error handling (PyTorch-side)

These live on the framework side (`torch.distributed`) but govern NCCL collective behavior, so they belong in the same triage kit.

| Env var | Effect | When to set | Seen in |
| :--- | :--- | :--- | :--- |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | Make the watchdog tear down the process group on a collective error/timeout instead of hanging forever. | Always, for training jobs — turns a silent hang into a fast, diagnosable crash. | lab-09, lab-15, lab-16 |
| `TORCH_NCCL_BLOCKING_WAIT` | Make collectives block (and honor the timeout) rather than return early. | Debugging hangs where you need a deterministic timeout stack trace. | lab-15 |
| `NCCL_TIMEOUT` / PG `timeout` | Rendezvous/collective timeout budget. | Large gangs with slow init (see lab-13's 24-GPU rendezvous); raise if init legitimately takes longer than the default. | lab-13, lab-15 |

---

## The busbw vs algbw distinction (used everywhere the number appears)

nccl-tests reports two bandwidths and they mean different things:

- **algbw** (algorithm bandwidth) = `data_size / time`. What the *application* sees.
- **busbw** (bus bandwidth) = `algbw × 2(n−1)/n` for all-reduce. Normalizes out the fact that a ring all-reduce moves each byte `2(n−1)/n` times, so busbw is comparable **across GPU counts** and against the hardware link ceiling.

The `2(n−1)/n` factor is why busbw — not algbw — is the right axis for a scaling curve: it isolates fabric efficiency from the collective's inherent data movement. This is applied identically in lab-04, lab-06, and lab-12 so the 8/16/24-GPU points are comparable.

---

**Related:** [tool-cheatsheets.md](tool-cheatsheets.md) (nccl-tests invocations) · [doc-06](../docs/part2-inter-node/06-nccl-collectives.md) (ring vs tree, bus vs algo) · lab-06 / lab-12 (measured sweeps) · lab-15 (comms troubleshooting).
