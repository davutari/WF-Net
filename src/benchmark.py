"""
Computational cost.

  run_flops()          params + FLOPs, PyTorch FlopCounterMode (hardware independent)
  gpu_busy()           idle-GPU guard (nvidia-smi samples)
  run_gpu_timing()     latency (median / p95 / mean, batch 1) and throughput
                       (batch 32), FP32 and AMP -- needs an IDLE GPU
  run_batch_scaling()  throughput & peak memory vs batch size
  run_cpu_scaling()    strong scaling of CPU inference vs number of threads
"""
import os
import platform
import subprocess
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from src.viz import save_fig
from src.models import ABLATION_MODELS, WFNet

N_CLASSES = 9
IMG = config.IMG_SIZE


def all_models():
    return list(config.BASELINE_MODELS) + list(ABLATION_MODELS) + [config.PROPOSED_NAME]


def build(name):
    """Architecture only (no pretrained download): cost does not depend on weights."""
    if name == config.PROPOSED_NAME:
        return WFNet(N_CLASSES, pretrained=False)
    if name in ABLATION_MODELS:
        return WFNet(N_CLASSES, pretrained=False, **ABLATION_MODELS[name])
    import timm
    return timm.create_model(config.BASELINE_MODELS[name], pretrained=False,
                             num_classes=N_CLASSES)


# --------------------------------------------------------------------------- #
def count_flops(model):
    # Not under no_grad: FlopCounterMode's module tracker needs autograd metadata
    # on views of parameters (e.g. an expanded learnable query). Only the forward
    # pass is executed and counted.
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, IMG, IMG)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9          # GFLOPs (1 MAC = 2 FLOPs)


def run_flops(out_dir, models=None, log=print):
    rows = []
    for name in models or all_models():
        m = build(name).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        gf = count_flops(m)
        rows.append({"model": name, "params_M": round(params, 2), "GFLOPs": round(gf, 2)})
        log(f"  {name:36s} {params:7.2f} M  {gf:7.2f} GFLOPs")
        del m
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "flops.csv"), index=False)


# --------------------------------------------------------------------------- #
def gpu_busy(samples=6, interval=0.5, util_thr=80, mem_thr_mib=4000):
    # An idle workstation can show up to ~47% utilisation / ~2.3 GB, a training job
    # ~92% / ~7.8 GB -> memory is the decisive signal.
    """(busy, description) from nvidia-smi; (False, ...) if nvidia-smi is missing."""
    utils, mems = [], []
    for _ in range(samples):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            u, mu = (float(v) for v in out.split(","))
            utils.append(u)
            mems.append(mu)
        except Exception as e:
            return False, f"nvidia-smi unavailable ({e})"
        time.sleep(interval)
    u, mu = float(np.median(utils)), float(np.median(mems))
    return (u >= util_thr or mu >= mem_thr_mib), \
        f"GPU median utilisation {u:.0f}%, memory used {mu:.0f} MiB"


class _fast_cudnn:
    """Deployment-style timing: cuDNN autotuning on, determinism off; restored after."""
    def __enter__(self):
        self.prev = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False

    def __exit__(self, *exc):
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = self.prev
        return False


@torch.no_grad()
def _time(model, x, iters, amp):
    dev = x.device
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(10):                       # warm-up (cuDNN autotune, caches)
            model(x)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t0)
    return np.array(ts)


def _timing_row(name, dev, iters_lat, iters_thr, batch_thr):
    m = build(name).to(dev).eval()
    r = {"model": name}
    for amp in (False, True):
        tag = "amp" if amp else "fp32"
        t1 = _time(m, torch.randn(1, 3, IMG, IMG, device=dev), iters_lat, amp) * 1e3
        tb = _time(m, torch.randn(batch_thr, 3, IMG, IMG, device=dev), iters_thr, amp)
        r.update({f"lat_median_ms_{tag}": round(float(np.median(t1)), 2),
                  f"lat_p95_ms_{tag}": round(float(np.percentile(t1, 95)), 2),
                  f"lat_mean_ms_{tag}": round(float(t1.mean()), 2),
                  f"lat_std_ms_{tag}": round(float(t1.std()), 2),
                  f"throughput_ips_{tag}": round(batch_thr / float(np.median(tb)), 1)})
    r["throughput_ips_amp_with_tta"] = round(r["throughput_ips_amp"] / 2, 1)
    del m
    torch.cuda.empty_cache()
    return r


def run_gpu_timing(out_dir, models=None, iters_lat=200, iters_thr=30, batch_thr=32, log=print):
    dev = torch.device("cuda")
    rows = []
    with _fast_cudnn():
        for name in models or all_models():
            r = _timing_row(name, dev, iters_lat, iters_thr, batch_thr)
            rows.append(r)
            log(f"  {name:36s} lat {r['lat_median_ms_fp32']:6.2f}/{r['lat_median_ms_amp']:6.2f}"
                f" ms (fp32/amp)  thr {r['throughput_ips_fp32']:7.1f}/"
                f"{r['throughput_ips_amp']:7.1f} img/s")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "latency_throughput.csv"), index=False)


def _scaling_rows(name, dev, batches, iters, log):
    m = build(name).to(dev).eval()
    rows = []
    for amp in (False, True):
        for b in batches:
            try:
                torch.cuda.reset_peak_memory_stats(dev)
                t = _time(m, torch.randn(b, 3, IMG, IMG, device=dev), iters, amp)
                rows.append({"model": name, "precision": "amp" if amp else "fp32",
                             "batch": b, "throughput_ips": b / float(np.median(t)),
                             "latency_ms_per_batch": float(np.median(t)) * 1e3,
                             "peak_mem_mib": torch.cuda.max_memory_allocated(dev) / 2**20})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  {name} {'amp' if amp else 'fp32'}: OOM at batch {b}")
                break
    del m
    torch.cuda.empty_cache()
    return rows


def run_batch_scaling(out_dir, models, batches=(1, 2, 4, 8, 16, 32, 64, 128), iters=20,
                      log=print):
    dev = torch.device("cuda")
    rows = []
    with _fast_cudnn():
        for name in models:
            rows += _scaling_rows(name, dev, batches, iters, log)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "batch_scaling.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (name, prec), g in df.groupby(["model", "precision"]):
        ax[0].plot(g.batch, g.throughput_ips, "o-", label=f"{name} ({prec})")
        ax[1].plot(g.batch, g.peak_mem_mib, "o-", label=f"{name} ({prec})")
    for a, yl in zip(ax, ["throughput (images/s)", "peak GPU memory (MiB)"]):
        a.set_xscale("log", base=2)
        a.set_xlabel("batch size")
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, "batch_scaling.png"))


@torch.no_grad()
def run_cpu_scaling(out_dir, name, batch=8, iters=5, max_threads=None, log=print):
    m = build(name).eval()
    x = torch.randn(batch, 3, IMG, IMG)
    max_t = max_threads or os.cpu_count() or 1
    threads = sorted({t for t in [1, 2, 4, 8, 12, 16, 24, 32, max_t] if t <= max_t})
    prev = torch.get_num_threads()
    rows = []
    try:
        for t in threads:
            torch.set_num_threads(t)
            m(x)                                    # warm-up
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                m(x)
                ts.append(time.perf_counter() - t0)
            rows.append({"threads": t, "sec_per_batch": float(np.median(ts))})
            log(f"  threads={t:3d}  {np.median(ts):.3f} s/batch")
    finally:
        torch.set_num_threads(prev)
    df = pd.DataFrame(rows)
    df["throughput_ips"] = batch / df.sec_per_batch
    df["speedup"] = df.sec_per_batch.iloc[0] / df.sec_per_batch
    df["efficiency"] = df.speedup / df.threads
    df.to_csv(os.path.join(out_dir, "cpu_scaling.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df.threads, df.speedup, "o-", label="measured")
    ax[0].plot(df.threads, df.threads, "--", color="grey", label="ideal (linear)")
    ax[0].set_xlabel("CPU threads")
    ax[0].set_ylabel("speed-up")
    ax[0].legend()
    ax[1].plot(df.threads, df.efficiency, "o-")
    ax[1].set_xlabel("CPU threads")
    ax[1].set_ylabel("parallel efficiency")
    for a in ax:
        a.grid(alpha=0.3)
    plt.suptitle(f"Strong scaling of CPU inference - {name} (batch {batch})")
    plt.tight_layout()
    save_fig(os.path.join(out_dir, "cpu_scaling.png"))


def run_loader_scaling(out_dir, data_dir, workers=(0, 1, 2, 4, 8, 16), batches=30,
                       batch_size=16, log=print):
    """Throughput of the training input pipeline (JPEG decode + augmentation) vs
    the number of parallel DataLoader worker processes (CPU parallelism)."""
    from torch.utils.data import DataLoader
    from src.data import WasteDataset, discover_dataset
    paths, labels, _ = discover_dataset(data_dir)
    ds = WasteDataset(paths, labels, train=True)
    max_w = os.cpu_count() or 1
    rows = []
    for w in [w for w in workers if w <= max_w]:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=w,
                            generator=torch.Generator().manual_seed(0))
        it = iter(loader)
        next(it)                                    # worker start-up excluded
        t0, n = time.perf_counter(), 0
        for _ in range(batches):
            try:
                x, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, _ = next(it)
            n += x.shape[0]
        dt = time.perf_counter() - t0
        rows.append({"workers": w, "images_per_s": n / dt})
        log(f"  workers={w:2d}  {n / dt:8.1f} images/s")
        del it, loader
    df = pd.DataFrame(rows)
    df["speedup_vs_0"] = df.images_per_s / df.images_per_s.iloc[0]
    df.to_csv(os.path.join(out_dir, "loader_scaling.csv"), index=False)
    plt.figure(figsize=(5.5, 3.8))
    plt.plot(df.workers, df.images_per_s, "o-")
    plt.xlabel("DataLoader worker processes")
    plt.ylabel("training images / s (decode + augment)")
    plt.title("Parallel input pipeline scaling")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, "loader_scaling.png"))


def write_meta(out_dir, extra):
    lines = [f"date      : {datetime.now():%Y-%m-%d %H:%M}",
             f"torch     : {torch.__version__}  (CUDA {torch.version.cuda})",
             f"cpu       : {platform.processor()}  ({os.cpu_count()} logical cores)",
             f"platform  : {platform.platform()}",
             f"image     : {IMG}x{IMG}, {N_CLASSES} classes"]
    if torch.cuda.is_available():
        lines.append(f"gpu       : {torch.cuda.get_device_name(0)}")
    with open(os.path.join(out_dir, "benchmark_meta.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines + extra) + "\n\n")
