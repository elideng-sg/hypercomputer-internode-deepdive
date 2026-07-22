#!/usr/bin/env python3
"""Minimal 2-node distributed-training driver: DDP and FSDP.

Launched by torchrun (--nnodes=2 --nproc_per_node=8). Trains a small
transformer-ish MLP stack on synthetic data so the run is deterministic and
fast; the point is the *communication pattern*, not model quality:

  * DDP   -> gradient all-reduce every step (bucketed).
  * FSDP  -> parameters + grads sharded across all ranks; all-gather on
             forward, reduce-scatter on backward.

Emits per-step timing and, on rank 0, a short summary. Wrap with
torch.profiler externally (see run.sh) to capture a trace for HTA/nsys.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.net(x))


class Model(nn.Module):
    def __init__(self, d=4096, layers=8):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(d) for _ in range(layers)])
        self.head = nn.Linear(d, d)

    def forward(self, x):
        return self.head(self.blocks(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ddp", "fsdp"], default="ddp")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--profile", action="store_true",
                    help="wrap the loop in torch.profiler; rank 0 exports a "
                         "Chrome trace + prints the top ops by CUDA time")
    ap.add_argument("--trace-out", default="/workspace/trace_rank0.json")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    torch.manual_seed(0)
    model = Model(args.dim, args.layers).cuda()
    if args.mode == "ddp":
        model = DDP(model, device_ids=[local_rank])
    else:
        model = FSDP(model, device_id=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"# mode={args.mode} world={world} dim={args.dim} layers={args.layers} "
              f"batch={args.batch} params/rank~{n_params/1e6:.1f}M")
        print(f"# {'step':>5} {'loss':>10} {'step_ms':>10} {'img_per_s':>10}")

    def train_step():
        x = torch.randn(args.batch, args.dim, device="cuda")
        y = torch.randn(args.batch, args.dim, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        return loss, time.perf_counter() - t0

    prof = None
    if args.profile:
        from torch.profiler import profile, ProfilerActivity, schedule
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=3, active=5, repeat=1),
            record_shapes=False, with_stack=False)
        prof.start()

    step_times = []
    for step in range(args.steps):
        loss, dt = train_step()
        if prof is not None:
            prof.step()
        if step >= 5:  # skip warmup/compile
            step_times.append(dt)
        if rank == 0:
            ips = (args.batch * world) / dt
            print(f"  {step:>5} {loss.item():>10.4f} {dt*1e3:>10.2f} {ips:>10.1f}")

    if prof is not None:
        prof.stop()
        if rank == 0:
            prof.export_chrome_trace(args.trace_out)
            print(f"# trace written to {args.trace_out}")
            print(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=15))

    if rank == 0 and step_times:
        avg = sum(step_times) / len(step_times)
        print(f"# steady-state: {avg*1e3:.2f} ms/step  "
              f"{(args.batch*world)/avg:.1f} samples/s (global)")
        print("# done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
