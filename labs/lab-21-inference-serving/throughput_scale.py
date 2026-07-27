#!/usr/bin/env python3
# throughput_scale.py — the measurement a single GPU can't make: does serving
# throughput scale HORIZONTALLY when you add GPUs? Spawn W worker processes, each
# pinned to its own GPU (cuda:0..W-1), each running a tight ResNet-50 inference
# loop at fixed batch for DURATION seconds; sum the per-GPU throughput. Linear
# scaling is the empirical basis for autoscaling: N replicas ≈ N× the QPS, so an
# autoscaler can add replicas to track load. Weights random (no download).
#
# env: WORKERS(8), BATCH(16), DURATION(10)
import os, time, torch, torchvision
import multiprocessing as mp

BATCH    = int(os.environ.get("BATCH", "16"))
DURATION = float(os.environ.get("DURATION", "10"))

def worker(gpu, out):
    torch.backends.cudnn.benchmark = True
    dev = torch.device(f"cuda:{gpu}")
    m = torchvision.models.resnet50(weights=None).eval().half().to(dev, memory_format=torch.channels_last)
    x = torch.randn(BATCH, 3, 224, 224, device=dev).half().contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(5): _ = m(x)
        torch.cuda.synchronize()
        n = 0; stop = time.time() + DURATION
        while time.time() < stop:
            _ = m(x); torch.cuda.synchronize(); n += 1
    out.put((gpu, n * BATCH / DURATION))   # images/s on this GPU

if __name__ == "__main__":
    mp.set_start_method("spawn")
    W = int(os.environ.get("WORKERS", "8"))
    out = mp.Queue(); procs = [mp.Process(target=worker, args=(g, out)) for g in range(W)]
    for p in procs: p.start()
    res = [out.get() for _ in range(W)]
    for p in procs: p.join()
    res.sort()
    total = sum(r for _, r in res)
    per = total / W
    print(f"WORKERS={W:2d}  aggregate={total:9.1f} img/s  per-GPU={per:8.1f} img/s  "
          f"(batch={BATCH})", flush=True)
