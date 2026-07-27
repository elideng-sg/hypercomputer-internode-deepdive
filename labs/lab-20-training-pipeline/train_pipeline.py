#!/usr/bin/env python3
# train_pipeline.py — the end-to-end training half of lab-20.
#
# A REAL (small) distributed training job that ties the pipeline together:
#   data (GCS via GCSFuse)  ->  DDP training across N nodes  ->  checkpoints (GCS)  ->  metrics (GMP/DCGM)
#
# It trains an MLP regressor on a synthetic-but-LEARNABLE dataset (y = X·w* + b* + noise),
# so the loss genuinely decreases — an honest training curve, not a fixed synthetic tensor.
# Data shards (train_*.npz) are read from the mounted bucket at DATA_DIR; checkpoints are
# written back to CKPT_DIR (also on the bucket). Ranks are launched manually (RANK/LOCAL_RANK/
# WORLD_SIZE env) and rendezvous via c10d — the same robust multi-pod pattern as lab-13.
#
# env: DATA_DIR, CKPT_DIR, STEPS, BATCH, LOG_EVERY, CKPT_EVERY, MASTER_ADDR, MASTER_PORT,
#      RANK, LOCAL_RANK, WORLD_SIZE
import os, time, glob, datetime, numpy as np, torch, torch.nn as nn, torch.distributed as dist

DATA_DIR   = os.environ.get("DATA_DIR", "/data/train")
CKPT_DIR   = os.environ.get("CKPT_DIR", "/data/checkpoints/pipeline")
STEPS      = int(os.environ.get("STEPS", "800"))
BATCH      = int(os.environ.get("BATCH", "8192"))
HIDDEN     = int(os.environ.get("HIDDEN", "8192"))   # wide enough to keep the H100s genuinely busy
LOG_EVERY  = int(os.environ.get("LOG_EVERY", "50"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "200"))

def main():
    rank  = int(os.environ["RANK"]); lrank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(lrank)
    dev = torch.device("cuda", lrank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=600))

    # --- data: each rank owns shard (rank % num_shards), sliced by (rank // num_shards) ----
    shards = sorted(glob.glob(os.path.join(DATA_DIR, "train_*.npz")))
    assert shards, f"no train_*.npz in {DATA_DIR}"
    ns = len(shards)
    mine = shards[rank % ns]
    t_read = time.time()
    with np.load(mine) as z:
        X = z["X"]; y = z["y"]
    grp = rank // ns                                  # 0 or 1 when world=16, ns=8
    ngrp = max(1, (world + ns - 1) // ns)
    X = X[grp::ngrp]; y = y[grp::ngrp]                # disjoint rows across ranks sharing a shard
    read_s = time.time() - t_read
    Xg = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
    yg = torch.from_numpy(np.ascontiguousarray(y)).to(dev).unsqueeze(1)
    D = Xg.shape[1]; n = Xg.shape[0]

    model = nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(),
                          nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                          nn.Linear(HIDDEN, 1)).to(dev)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[lrank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    os.makedirs(CKPT_DIR, exist_ok=True) if rank == 0 else None

    if rank == 0:
        print(f"[pipeline] world={world} shards={ns} D={D} rows/rank={n} read={read_s:.1f}s "
              f"steps={STEPS} batch={BATCH}", flush=True)
    dist.barrier()

    g = torch.Generator(device=dev); g.manual_seed(rank)
    seen = 0; t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, n, (BATCH,), device=dev, generator=g)
        xb, yb = Xg[idx], yg[idx]
        opt.zero_grad(set_to_none=True)
        out = model(xb); loss = lossf(out, yb)
        loss.backward(); opt.step()
        seen += BATCH * world
        if step % LOG_EVERY == 0 or step == 1:
            lt = loss.detach().clone(); dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            torch.cuda.synchronize()
            if rank == 0:
                dt = time.time() - t0
                print(f"[pipeline] step {step:4d}/{STEPS}  loss={lt.item():.4f}  "
                      f"{seen/dt/1e6:.2f}M samples/s  ({seen/1e6:.1f}M seen)", flush=True)
        if step % CKPT_EVERY == 0 and rank == 0:
            p = os.path.join(CKPT_DIR, f"step_{step:04d}.pt")
            tc = time.time(); torch.save(model.module.state_dict(), p); wr = time.time() - tc
            print(f"[pipeline] CKPT step {step} -> {p}  ({wr:.2f}s, GCSFuse write)", flush=True)
    if rank == 0:
        print(f"[pipeline] DONE {STEPS} steps in {time.time()-t0:.1f}s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
