#!/usr/bin/env python3
# dataloader_bench.py — make the data path visible as GPU-busy time.
#
# Runs the SAME fixed compute step two ways and reports the GPU-busy fraction:
#   mode=starved : each step reads SHARDS off the mounted bucket (GCSFuse) THEN computes.
#                  If the per-step read takes longer than the compute, the GPU idles on I/O.
#   mode=fed     : the bytes are already resident, so the identical compute runs
#                  back-to-back and the GPU stays busy.
# The COMPUTE IS IDENTICAL in both modes — the ONLY difference is whether the step
# waits on storage. That is the whole point of doc-22: a "slow training" symptom
# that is not the GPU's fault — the data path is starving it.
#
# HONEST metric: the headline number is GPU-busy fraction = compute_time / wall_time,
# i.e. the fraction of wallclock the GPU actually spent computing (what a training
# loop truly achieves). NVML instantaneous utilization is reported too, but it is
# NOISY for short kernels, so the verdict keys off the wall-fraction. run_storage.sh
# also cross-checks DCGM_FI_PROF_GR_ENGINE_ACTIVE on the managed pipeline.
#
# env: DATA_DIR (mounted path), MODE=starved|fed, STEPS, MATMUL_N, COMPUTE_ITERS, STARVE_SHARDS
import os, time, glob, sys, threading
import torch

DATA_DIR      = os.environ.get("DATA_DIR", "/data")
MODE          = os.environ.get("MODE", "starved")
STEPS         = int(os.environ.get("STEPS", "30"))
MATMUL_N      = int(os.environ.get("MATMUL_N", "8192"))
COMPUTE_ITERS = int(os.environ.get("COMPUTE_ITERS", "40"))   # ~100 ms of GEMM on an H100
# how many shards STARVED reads per step (-1 = all of them). Reading a large volume
# per step is what makes I/O dominate even at multi-GB/s GCSFuse throughput.
STARVE_SHARDS = int(os.environ.get("STARVE_SHARDS", "-1"))

try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _h = None

def gpu_util():
    if pynvml is None:
        return -1.0
    return float(pynvml.nvmlDeviceGetUtilizationRates(_h).gpu)

# background NVML sampler — instantaneous util is only meaningful when sampled often
_util_samples, _sampling = [], True
def _sampler():
    while _sampling:
        u = gpu_util()
        if u >= 0:
            _util_samples.append(u)
        time.sleep(0.05)

def load_shards(paths):
    n = 0
    for p in paths:
        with open(p, "rb") as f:
            n += len(f.read())
    return n

def main():
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda:0")
    shards = sorted(glob.glob(os.path.join(DATA_DIR, "shard_*.bin")))
    if not shards:
        print(f"NO SHARDS in {DATA_DIR} — populate the bucket first", file=sys.stderr); sys.exit(2)
    read_set = shards if STARVE_SHARDS < 0 else shards[:STARVE_SHARDS]

    a = torch.randn(MATMUL_N, MATMUL_N, device=dev, dtype=torch.float16)
    b = torch.randn(MATMUL_N, MATMUL_N, device=dev, dtype=torch.float16)

    def compute():
        for _ in range(COMPUTE_ITERS):
            c = a @ b
        torch.cuda.synchronize()
        return c

    for _ in range(3):      # warmup
        compute()

    # In "fed" mode the bytes are already resident (read once, reused) — compute-bound.
    resident = load_shards(read_set) if MODE == "fed" else None

    t = threading.Thread(target=_sampler, daemon=True); t.start()
    io_s, comp_s, read_bytes = 0.0, 0.0, 0
    t0 = time.time()
    for i in range(STEPS):
        t_io = time.time()
        if MODE == "starved":
            read_bytes += load_shards(read_set)      # fresh read every step (storage-bound)
        io_s += time.time() - t_io

        t_c = time.time()
        compute()                                     # IDENTICAL compute in both modes
        comp_s += time.time() - t_c
    wall = time.time() - t0
    global _sampling; _sampling = False; t.join(timeout=1)

    busy_frac = 100.0 * comp_s / wall                 # honest GPU-busy fraction
    nvml_mean = (sum(_util_samples) / len(_util_samples)) if _util_samples else -1.0
    gib = read_bytes / 2**30
    print(f"MODE={MODE} steps={STEPS} matmul_n={MATMUL_N} iters/step={COMPUTE_ITERS} read_set={len(read_set)} shards")
    print(f"  wall={wall:.2f}s  io={io_s:.2f}s ({100*io_s/wall:.0f}%)  compute={comp_s:.2f}s ({100*comp_s/wall:.0f}%)  read={gib:.1f} GiB")
    print(f"  GPU-busy fraction (compute/wall) = {busy_frac:.1f}%   NVML sampled util (secondary) = {nvml_mean:.1f}%")
    print(f"  VERDICT: {'STARVED (GPU idle on the data path)' if busy_frac < 50 else 'FED (GPU compute-bound)'}")

if __name__ == "__main__":
    main()
