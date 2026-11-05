"""
run_experiments.py -- single entry point of the WF-Net experiments.

All outputs go to  <repo>/results/  only (the datasets in <repo>/data are only read).

Usage (from the repository root):
    python run_experiments.py              run every pending step
    python run_experiments.py --status     show progress
    python run_experiments.py --step main  run selected step(s)
    python run_experiments.py --quick      smoke test (separate folder)

Interruption-safe: finished steps and finished (model, fold) jobs are skipped,
and a job that was cut off continues from its last saved epoch (<= 10 min of
work is repeated). Just start the same command again after a power cut.
A step that fails with an unexpected error (e.g. a transient GPU/driver error)
is retried automatically up to STEP_RETRIES times after a short pause.
"""
import os
import subprocess
import sys
import warnings


def _relaunch_in_project_venv():
    """If started with any other Python (IDE default, Anaconda, plain `python`),
    re-run this script with the project's own environment <repo>/.venv -- the
    only one that has the tested torch/timm versions -- BEFORE importing torch."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, ".venv")
    py = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv, "bin", "python")
    if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.abspath(venv)):
        return                                   # already the right environment
    if not os.path.exists(py) or os.environ.get("WFNET_RELAUNCHED") == "1":
        return                                   # check_environment() will report it
    print(f"[run_experiments] Started with Python: {sys.executable}\n"
          f"[run_experiments] Restarting with the project environment: {py}", flush=True)
    env = dict(os.environ, WFNET_RELAUNCHED="1")
    child = subprocess.Popen([py, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    while True:
        try:
            sys.exit(child.wait())
        except KeyboardInterrupt:
            continue          # the child gets the same Ctrl+C and stops cleanly


if __name__ == "__main__":
    _relaunch_in_project_venv()

# --- quiet third-party output (must happen before these libraries are imported) ---
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

import argparse
import json
import time
import traceback
from collections import OrderedDict
from datetime import datetime

import torch

import config
from src.runner import fmt_hms

# ------------------------------------------------------------------------------ #
# Steps (order = execution order):  key -> (folder, description, estimate)
# ------------------------------------------------------------------------------ #
STEPS = OrderedDict([
    ("cost",                     ("01_cost",                     "Params/FLOPs, latency, throughput, batch/CPU/data-loading scaling", "~40 min")),
    ("main",                     ("02_main_seed42",              "Main 5-fold CV: 19 models + Garbage12 external test + learning curves", "~26-28 h")),
    ("branch_weighting",         ("03_branch_weighting",         "Branch weighting of the fusion + two-branch Grad-CAM", "~15 min")),
    ("few_shot",                 ("04_few_shot",                 "Few-shot adaptation to Garbage12 (k=1,5,10,20)", "~15 min")),
    ("multi_object",             ("05_multi_object",             "Multi-object scene test + class-specific Grad-CAM", "~10 min")),
    ("stats_main",               ("06_stats_main",               "Bootstrap CI, McNemar, Wilcoxon, corrected t (all models)", "~5 min")),
    ("repeated_cv",              ("07_repeated_cv",              "Repeated CV: seeds 1337 and 2024, 7 key models", "~15 h")),
    ("stats_repeated",           ("08_stats_repeated",           "Statistics, 3x5 repeated CV", "~5 min")),
    ("distillation_sensitivity", ("09_distillation_sensitivity", "Distillation sensitivity sweep (7 settings x 3 folds)", "~4.5 h")),
    ("garbage12",                ("10_garbage12_training",       "Training on Garbage12 (3-fold, 4 models)", "~8-15 h")),
])
DEPENDS = {"branch_weighting": ["main"], "few_shot": ["main"], "multi_object": ["main"],
           "stats_main": ["main"], "stats_repeated": ["main", "repeated_cv"]}
GPU_STEPS = {"cost", "main", "branch_weighting", "few_shot", "multi_object", "repeated_cv",
             "distillation_sensitivity", "garbage12"}
TRAIN_STEPS = {"main", "repeated_cv", "distillation_sensitivity", "garbage12"}
STEP_RETRIES = 2            # extra attempts for a step that raised an unexpected error
RETRY_WAIT_S = 60

# Smoke-test settings (--quick): tiny data, 1 epoch, few models, separate folder.
QUICK = {"models": [config.PROPOSED_NAME, "Ablation: Dual (concat)", "MobileNetV2"],
         "per_class": 20, "folds": 2, "epochs": 1, "cross_limit": 40, "seeds": [1337]}


class Tee:
    """Mirror stdout into a UTF-8 log file (progress bars stay on the console only)."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Plan:
    def __init__(self, quick):
        self.quick = quick
        self.root = os.path.join(config.OUT_ROOT, "_quick_test") if quick else config.OUT_ROOT
        self.dirs = {k: os.path.join(self.root, v[0]) for k, v in STEPS.items()}
        seeds = QUICK["seeds"] if quick else config.REPEAT_SEEDS
        self.repeat_dirs = [os.path.join(self.dirs["repeated_cv"], f"seed{s}") for s in seeds]
        self.state_path = os.path.join(self.root, "status.json")

    def load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass             # damaged -> rebuilt; finished jobs are still skipped
        return {"steps": {}}

    def save(self, st):
        from src.utils import durable_json
        durable_json(self.state_path, st)

    def report_dirs(self):
        d = dict(self.dirs)
        d["repeated_runs"] = self.repeat_dirs
        return d


def log(msg=""):
    print(msg, flush=True)


# ------------------------------------------------------------------------------ #
# Step implementations
# ------------------------------------------------------------------------------ #
def all_main_models():
    from src.models import ABLATION_MODELS
    return list(config.BASELINE_MODELS) + list(ABLATION_MODELS) + [config.PROPOSED_NAME]


def step_cost(plan, args):
    from src import benchmark as B
    q = plan.quick
    out = plan.dirs["cost"]
    os.makedirs(out, exist_ok=True)
    models = QUICK["models"] if q else None
    B.write_meta(out, [f"mode      : {'quick' if q else 'full'}"])
    log("FLOPs / parameters (CPU):")
    B.run_flops(out, models, log=log)
    if torch.cuda.is_available():
        log("GPU latency / throughput (FP32 and AMP):")
        B.run_gpu_timing(out, models, iters_lat=20 if q else 200, iters_thr=5 if q else 30,
                         log=log)
        log("Batch scaling:")
        B.run_batch_scaling(out, [config.PROPOSED_NAME] if q else
                            [config.PROPOSED_NAME, "Ablation: WF-Net-Lite", "ViT-B/16", "Swin-T"],
                            batches=(1, 4, 16) if q else (1, 2, 4, 8, 16, 32, 64, 128),
                            iters=3 if q else 20, log=log)
    else:
        log("No CUDA -> GPU timing measurements skipped (FLOPs only).")
    log("CPU inference thread scaling:")
    B.run_cpu_scaling(out, config.PROPOSED_NAME, batch=2 if q else 8, iters=1 if q else 5,
                      max_threads=4 if q else None, log=log)
    log("Data-loading pipeline parallel scaling:")
    B.run_loader_scaling(out, config.DATA_DIR, workers=(0, 2) if q else (0, 1, 2, 4, 8, 16),
                         batches=3 if q else 30, log=log)


def _cv_kwargs(plan):
    if not plan.quick:
        return {}
    return {"n_folds": QUICK["folds"], "epochs": QUICK["epochs"],
            "debug_per_class": QUICK["per_class"], "cross_limit": QUICK["cross_limit"]}


def step_main(plan, args):
    from src.runner import run_cv
    models = QUICK["models"] if plan.quick else all_main_models()
    run_cv(plan.dirs["main"], models, seed=config.MAIN_SEED, cross=True,
           ckpt_models=[config.PROPOSED_NAME], log=log, **{"n_folds": 5, **_cv_kwargs(plan)})


def step_branch_weighting(plan, args):
    from src.branch_weighting import run_branch_weighting
    run_branch_weighting(plan.dirs["main"], plan.dirs["branch_weighting"], cross=True,
                         cross_limit=QUICK["cross_limit"] if plan.quick else 2000,
                         gallery=1 if plan.quick else 4, log=log)


def step_few_shot(plan, args):
    from src.adapt import run_adaptation
    if plan.quick:
        run_adaptation(plan.dirs["main"], plan.dirs["few_shot"], shots=(1, 5), draws=2,
                       limit=300, log=log)
    else:
        run_adaptation(plan.dirs["main"], plan.dirs["few_shot"], log=log)


def step_multi_object(plan, args):
    from src.multiobject import run_multiobject
    run_multiobject(plan.dirs["main"], plan.dirs["multi_object"],
                    pairs_per_fold=10 if plan.quick else 200,
                    gallery=1 if plan.quick else 4, log=log)


def step_stats_main(plan, args):
    from src.posthoc import run_posthoc
    run_posthoc([plan.dirs["main"]], plan.dirs["stats_main"],
                boot=200 if plan.quick else config.BOOTSTRAP_RESAMPLES, log=log)


def step_repeated_cv(plan, args):
    from src.runner import run_cv
    models = QUICK["models"] if plan.quick else config.KEY_MODELS
    seeds = QUICK["seeds"] if plan.quick else config.REPEAT_SEEDS
    for s, d in zip(seeds, plan.repeat_dirs):
        log(f"--- Repeated CV, seed {s} ---")
        run_cv(d, models, seed=s, cross=False, ckpt_models=[], log=log,
               **{"n_folds": 5, **_cv_kwargs(plan)})


def step_stats_repeated(plan, args):
    from src.posthoc import run_posthoc
    run_posthoc([plan.dirs["main"]] + plan.repeat_dirs, plan.dirs["stats_repeated"],
                boot=200 if plan.quick else config.BOOTSTRAP_RESAMPLES, log=log)


def step_distillation_sensitivity(plan, args):
    from src.sweep import run_sweep
    if plan.quick:
        run_sweep(plan.dirs["distillation_sensitivity"], config.MAIN_SEED, [0], config.SWEEP_LEVELS,
                  epochs=QUICK["epochs"], debug_per_class=QUICK["per_class"],
                  max_configs=2, log=log)
    else:
        run_sweep(plan.dirs["distillation_sensitivity"], config.MAIN_SEED, config.SWEEP_FOLDS,
                  config.SWEEP_LEVELS, log=log)


def step_garbage12(plan, args):
    from src.runner import run_cv
    if plan.quick:
        run_cv(plan.dirs["garbage12"], ["MobileNetV2"], seed=config.MAIN_SEED,
               data_dir=config.GARBAGE12_DIR, cross=False, ckpt_models=[], num_workers=2,
               log=log, **_cv_kwargs(plan))
    else:
        run_cv(plan.dirs["garbage12"], config.GARBAGE12_MODELS, seed=config.MAIN_SEED,
               n_folds=config.GARBAGE12_FOLDS, epochs=config.GARBAGE12_EPOCHS,
               data_dir=config.GARBAGE12_DIR, cross=False, ckpt_models=[],
               num_workers=config.GARBAGE12_WORKERS, log=log)


STEP_FN = {k: globals()[f"step_{k}"] for k in STEPS}


# ------------------------------------------------------------------------------ #
# Pre-flight checks
# ------------------------------------------------------------------------------ #
def show_status(plan):
    st = plan.load()["steps"]
    log(f"Results folder: {plan.root}")
    log(f"{'#':>2}  {'step':24s} {'status':11s} {'time':>12s}  {'estimate':>10s}  description")
    for i, (k, (folder, desc, est)) in enumerate(STEPS.items(), 1):
        s = st.get(k, {})
        status = s.get("status", "pending")
        dur = fmt_hms(s["seconds"]) if "seconds" in s else ""
        log(f"{i:>2}  {k:24s} {status:11s} {dur:>12s}  {est:>10s}  {desc}")


REQUIRED_PACKAGES = ["torch", "torchvision", "timm", "numpy", "pandas", "scipy", "sklearn",
                     "matplotlib", "seaborn", "scikit_posthocs", "PIL", "tqdm"]


def check_environment(args):
    """The study must run in THIS project's virtual environment (the one every
    test was done with) and with all required packages present."""
    import importlib
    expected = os.path.join(config.PROJECT_DIR, ".venv")
    here = os.path.normcase(os.path.abspath(sys.prefix))
    correct_cmd = (f"{os.path.join(expected, 'Scripts', 'python.exe')} "
                   f"{os.path.join(config.ROOT_DIR, 'run_experiments.py')}")
    if here != os.path.normcase(os.path.abspath(expected)) and not args.force:
        log("STOPPED: wrong Python environment.")
        log(f"  in use  : {sys.executable}")
        log(f"  required: {os.path.join(expected, 'Scripts', 'python.exe')}")
        log(f"  command : {correct_cmd}")
        return False
    missing = []
    for m in REQUIRED_PACKAGES:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    if missing:
        log(f"STOPPED: missing package(s): {', '.join(missing)} (environment: {sys.prefix})")
        return False
    return True


def check_gpu(args, steps):
    if args.force or not (set(steps) & GPU_STEPS):
        return True
    if not torch.cuda.is_available():
        log("WARNING: CUDA not found; training on the CPU is very slow. "
            "Add --force to run on the CPU on purpose.")
        return bool(args.quick)          # smoke test may run on CPU
    from src.benchmark import gpu_busy
    busy, desc = gpu_busy()
    log(desc)
    if busy:
        log("STOPPED: the GPU is used by another job. Two trainings on one GPU cause "
            "out-of-memory errors and wrong timings. Start again when the GPU is free "
            "(--force overrides this check).")
        return False
    return True


def preflight_weights(plan, steps):
    """Make sure every pretrained backbone is available BEFORE a multi-day run
    (cached or downloadable). Fails early instead of hours later."""
    if not (set(steps) & TRAIN_STEPS):
        return True
    import timm
    from src.models import ABLATION_MODELS
    names = set()
    models = QUICK["models"] if plan.quick else all_main_models()
    for m in models:
        if m in config.BASELINE_MODELS:
            names.add(config.BASELINE_MODELS[m])
        else:
            names.update(ABLATION_MODELS.get(m, {}).get("backbones",
                                                        config.PROPOSED_BACKBONES))
    failed = []
    log(f"Pre-flight check: are the {len(names)} pretrained backbones available (downloaded if needed)...")
    for n in sorted(names):
        try:
            timm.create_model(n, pretrained=True, num_classes=0)
        except Exception as e:
            failed.append(f"{n}: {type(e).__name__}: {str(e)[:120]}")
    if failed:
        log("STOPPED: these pretrained weights could not be loaded (check the internet "
            "connection; once downloaded they stay in the cache):")
        for f in failed:
            log("   - " + f)
        return False
    log("Pre-flight check passed: all backbones are available.")
    return True


def main():
    ap = argparse.ArgumentParser(description="WF-Net experiments")
    ap.add_argument("--step", nargs="+", choices=list(STEPS), help="run only these steps")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--quick", action="store_true", help="smoke test (separate folder)")
    ap.add_argument("--rerun", nargs="+", choices=list(STEPS), default=[],
                    help="run finished step(s) again (finished jobs are still skipped)")
    ap.add_argument("--force", action="store_true",
                    help="skip the busy-GPU and disk-space checks")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):               # never crash on non-ASCII output such as ±
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    plan = Plan(args.quick)
    os.makedirs(os.path.join(plan.root, "logs"), exist_ok=True)
    if args.status:
        show_status(plan)
        return 0
    if not check_environment(args):
        return 6

    from src.safety import RunLock, free_gb, keep_awake
    lock = RunLock(plan.root)
    ok, msg = lock.acquire()
    if not ok:
        log("STOPPED: " + msg + " A second copy would corrupt the results.")
        return 2
    sys.stdout = Tee(sys.stdout, os.path.join(
        plan.root, "logs", f"run_{datetime.now():%Y%m%d_%H%M%S}.log"))

    state = plan.load()
    for k in args.rerun:
        state["steps"].pop(k, None)
    wanted = args.step or list(STEPS)
    todo = [k for k in wanted if state["steps"].get(k, {}).get("status") != "done"]
    interrupted = [k for k in todo if state["steps"].get(k, {}).get("status")
                   in ("running", "interrupted", "error")]
    log("=" * 78)
    log(f"WF-Net experiments | {datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"{'QUICK TEST' if args.quick else 'FULL RUN'}")
    log(f"Results : {plan.root}")
    log(f"Device  : {config.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    log(f"Steps   : {', '.join(todo) if todo else '(all finished)'}")
    if interrupted:
        log(f"Resume  : interrupted step(s) continue where they stopped: {', '.join(interrupted)}")
    log("=" * 78)
    if not todo:
        show_status(plan)
        return 0

    gb = free_gb(plan.root)
    log(f"Free disk: {gb:.1f} GB")
    if gb < config.MIN_FREE_DISK_GB and not args.force:
        log(f"STOPPED: at least {config.MIN_FREE_DISK_GB} GB of free space is required.")
        return 3
    if not check_gpu(args, todo):
        return 4
    if not preflight_weights(plan, todo):
        return 5
    if keep_awake():
        log("System sleep is blocked during this run (the display may still turn off).")

    from src.report import build_report
    failed = []
    for k in todo:
        missing = [d for d in DEPENDS.get(k, [])
                   if state["steps"].get(d, {}).get("status") != "done"]
        if missing:
            log(f"\n[{k}] skipped: required step(s) not finished: {', '.join(missing)}")
            continue
        folder, desc, est = STEPS[k]
        log(f"\n{'#' * 78}\n# STEP: {k}: {desc}  (estimated {est})\n{'#' * 78}")
        t0 = time.time()
        prev = state["steps"].get(k, {})
        state["steps"][k] = {"status": "running",
                             "started": prev.get("started",
                                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                             "seconds_before": prev.get("seconds_before", 0.0)
                             + prev.get("session_seconds", 0.0)}
        plan.save(state)
        ok = False
        for attempt in range(1 + STEP_RETRIES):
            try:
                STEP_FN[k](plan, args)
                ok = True
                break
            except KeyboardInterrupt:
                state["steps"][k].update(status="interrupted", session_seconds=time.time() - t0)
                plan.save(state)
                log(f"\n[{k}] stopped by the user. The same command continues where it "
                    "stopped.")
                raise
            except Exception:
                log(f"\n[{k}] ERROR (attempt {attempt + 1}/{1 + STEP_RETRIES}):\n"
                    f"{traceback.format_exc()}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt < STEP_RETRIES:
                    log(f"[{k}] retrying in {RETRY_WAIT_S} s from where it stopped "
                        "(finished jobs are skipped).")
                    time.sleep(RETRY_WAIT_S)
        if not ok:
            state["steps"][k].update(status="error", session_seconds=time.time() - t0)
            plan.save(state)
            failed.append(k)
            log(f"[{k}] failed in all {1 + STEP_RETRIES} attempts. Moving on to the other steps; "
                "this step is retried on the next start.")
            continue
        total = state["steps"][k].get("seconds_before", 0.0) + (time.time() - t0)
        state["steps"][k].update({"status": "done", "seconds": total,
                                  "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        state["steps"][k].pop("session_seconds", None)
        plan.save(state)
        log(f"[{k}] finished ({fmt_hms(total)})")
        try:
            build_report(plan.root, plan.report_dirs(), log=log)
        except Exception as e:                       # report is a convenience only
            log(f"[report] could not be built: {e}")
    log("")
    show_status(plan)
    if failed:
        log(f"\nFailed step(s): {', '.join(failed)}; details above and in the logs folder.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
