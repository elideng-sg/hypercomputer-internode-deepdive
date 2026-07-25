#!/usr/bin/env python3
"""Profiled multi-GPU DDP step loop -> Kineto traces for HTA (lab-17, Phase D).

Runs a handful of DDP training steps on one node's GPUs under
`torch.profiler`, then exports one Chrome/Kineto trace per rank into
TRACE_DIR. Holistic Trace Analysis (HTA) reads that directory to compute the
**temporal breakdown** (compute vs. communication vs. idle) and the
**communication/computation overlap** — the day-2 view of *where the step time
actually goes* that a throughput number alone can't show.

The model is deliberately small but with a large-enough parameter tensor that
DDP's gradient all-reduce (a real NCCL collective) shows up in the trace next
to the fp16 GEMMs, so comm/compute overlap is meaningful.

Launched by run_monitoring.sh via torchrun --nproc_per_node=N (single node, so
RANK==LOCAL_RANK and MASTER_ADDR=127.0.0.1). No node/config change of its own.

Env:
  TRACE_DIR     directory for per-rank traces        (default /workspace/traces)
  STEPS         profiled DDP steps                    (default 8)
  WARMUP        un-profiled warmup steps              (default 3)
  HID           hidden dim (sets the all-reduce size) (default 8192)
"""
import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=120),
        rank=rank, world_size=world,
    )

    trace_dir = os.environ.get("TRACE_DIR", "/workspace/traces")
    steps = int(os.environ.get("STEPS", "8"))
    warmup = int(os.environ.get("WARMUP", "3"))
    hid = int(os.environ.get("HID", "8192"))

    # A few big linear layers -> a sizeable gradient all-reduce per step.
    model = nn.Sequential(
        nn.Linear(hid, hid), nn.ReLU(),
        nn.Linear(hid, hid), nn.ReLU(),
        nn.Linear(hid, hid),
    ).to(dev).half()
    ddp = DDP(model, device_ids=[local_rank])
    opt = torch.optim.SGD(ddp.parameters(), lr=1e-4)
    x = torch.randn(256, hid, dtype=torch.float16, device=dev)
    tgt = torch.randn(256, hid, dtype=torch.float16, device=dev)
    loss_fn = nn.MSELoss()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        out = ddp(x)
        loss = loss_fn(out, tgt)
        loss.backward()          # DDP triggers gradient all-reduce here
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"# profiling {steps} DDP steps, world={world}, hid={hid} -> {trace_dir}",
              flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()

    os.makedirs(trace_dir, exist_ok=True)
    # HTA reads rank from the trace's distributedInfo metadata; a rank-tagged
    # filename keeps the directory human-readable too.
    prof.export_chrome_trace(f"{trace_dir}/rank-{rank}.pt.trace.json")
    if rank == 0:
        print("# trace export complete", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
