#!/usr/bin/env python3
"""
Sustained single-GPU load for health/throttle triage (lab-14).

Runs continuous large fp16 GEMMs on cuda:0 for LOAD_SECONDS, driving the GPU to
near-TDP power and boost clocks so that `nvidia-smi -q -d PERFORMANCE,CLOCK,
POWER,TEMPERATURE` and `nvidia-smi dmon` show a real idle->load delta (and, when
the power limit is capped, a real "SW Power Cap" throttle reason going Active).

Which physical GPU it hits is chosen by CUDA_VISIBLE_DEVICES (so the caller can
launch one process per device to load a whole node). This is a benign, fully
reversible compute load -- no config or node-level change of its own.

Env:
  LOAD_SECONDS  wall-clock seconds to sustain the load (default 60)
  MATMUL_N      square matrix dimension (default 8192 -- large enough to be
                compute-bound and hold the SMs busy)
"""
import os
import time

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available")

    seconds = float(os.environ.get("LOAD_SECONDS", "60"))
    n = int(os.environ.get("MATMUL_N", "8192"))
    dev = torch.device("cuda:0")  # cuda:0 == the device CUDA_VISIBLE_DEVICES exposes

    a = torch.randn(n, n, dtype=torch.float16, device=dev)
    b = torch.randn(n, n, dtype=torch.float16, device=dev)
    c = torch.empty(n, n, dtype=torch.float16, device=dev)

    # Warm the kernels/clocks up before the timed loop.
    for _ in range(5):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()

    name = torch.cuda.get_device_name(0)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"# load start dev(CUDA_VISIBLE_DEVICES={vis})={name} n={n} secs={seconds}",
          flush=True)

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < seconds:
        # Chain matmuls to keep the SMs saturated without host round-trips.
        for _ in range(50):
            torch.mm(a, b, out=c)
            a = c * 0.0 + a  # keep values bounded, avoid NaN/Inf blow-up
        torch.cuda.synchronize()
        iters += 50

    print(f"# load done iters={iters} elapsed={time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
