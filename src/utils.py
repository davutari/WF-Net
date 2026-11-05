"""Reproducibility and crash-safe file helpers."""
import copy
import json
import os
import platform
import random
import sys
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Crash-safe writes: write to <path>.tmp, flush + fsync to the physical disk,
# then atomically rename. After a power cut a file is either the complete old
# version, the complete new version, or absent -- never half-written.
# --------------------------------------------------------------------------- #
def durable_write(path, writer):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def durable_json(path, obj):
    durable_write(path, lambda f: f.write(
        json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8")))


def durable_torch_save(obj, path):
    durable_write(path, lambda f: torch.save(obj, f))


def durable_savez(path, **arrays):
    durable_write(path, lambda f: np.savez_compressed(f, **arrays))


def rng_state(generator=None):
    """Snapshot of every RNG used in training (for exact resume)."""
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
          "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        st["generator"] = generator.get_state()
    return st


def set_rng_state(st, generator=None):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    if generator is not None and "generator" in st:
        generator.set_state(st["generator"])


def write_run_snapshot(config_module, out_path) -> None:
    """Persist the full run configuration + environment for reproducibility.

    Dumps every UPPER_CASE constant in the config module plus interpreter,
    platform, device and key package versions into a text file inside the run
    folder -- so a versioned run can always be traced back to exact settings.
    """
    lines = ["===== RUN CONFIGURATION SNAPSHOT =====", ""]
    lines.append("[environment]")
    lines.append(f"python       : {sys.version.split()[0]}")
    lines.append(f"platform     : {platform.platform()}")
    lines.append(f"torch        : {torch.__version__}")
    lines.append(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"gpu          : {torch.cuda.get_device_name(0)}")
    for pkg in ("timm", "torchvision", "numpy", "pandas", "sklearn",
                "scikit_posthocs"):
        try:
            mod = __import__(pkg)
            lines.append(f"{pkg:13s}: {getattr(mod, '__version__', '?')}")
        except Exception:
            lines.append(f"{pkg:13s}: (not installed)")

    lines += ["", "[config]"]
    for name in sorted(dir(config_module)):
        if name.isupper():
            lines.append(f"{name} = {getattr(config_module, name)!r}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding (used together with a seeded generator)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ModelEMA:
    """Exponential moving average of model weights.

    Keeps a shadow copy of the model whose parameters track the trained weights
    with decay ``d`` (``ema = d*ema + (1-d)*model``). Integer/boolean buffers
    (e.g. BN ``num_batches_tracked``) are copied verbatim.

    Decay warm-up (as in timm's ModelEma): d_n = min(decay, (1+n)/(10+n)).
    Without it, a fixed 0.999 on a ~33-steps-per-epoch dataset would keep ~52% of
    the *initial* (randomly initialised head) weights after 20 epochs and ~27%
    after 40, so the evaluated model would lag far behind training.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for e, m in zip(self.ema.state_dict().values(),
                        model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach().to(e.dtype), alpha=1.0 - d)
            else:
                e.copy_(m)
