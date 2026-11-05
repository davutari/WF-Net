"""
followup_a4000.py -- follow-up experiments of WF-Net on the RTX A4000. Same code, folds,
job seeds and GPU as run_experiments.py; only the listed settings change. Nothing outside
the result folder results/11_followup is written.

Needs results/02_main_seed42 and results/10_garbage12_training from a prior
run_experiments.py run (folds, teacher checkpoints).

Usage (from the repository root):
    python followup_a4000.py --quick     smoke test (a few minutes)
    python followup_a4000.py             full queue
    python followup_a4000.py --status    progress and results so far

Interruption-safe: finished jobs are skipped and a job that was cut off continues from
its last saved epoch. After a power cut or Ctrl+C, start the same command again.

Queue (order = priority; summary.txt is refreshed after every job):
  1. g12/nomix       WF-Net on Garbage12, MixUp/CutMix off                  3 x ~22 min
  2. g12/eased       MixUp/CutMix off, only crop + flip augmentation, no
                     label smoothing, fusion dropout 0.1, no stochastic depth 3 x ~22 min
  3. lite/kd         WF-Net-Lite on WasteSnap with offline distillation from
                     the WF-Net model of the same fold                        5 x ~18 min
  4. g12/nosd        Garbage12, self-distillation and auxiliary heads off     3 x ~22 min
  5. g12/default     reproduction control of the published run (fold 0 only)  1 x ~22 min
  6. g12/default40   published settings with 40 instead of 20 epochs          3 x ~45 min
"""
import os
import subprocess
import sys
import warnings

CODE_VERSION = 1


def _relaunch_in_project_venv():
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, ".venv")
    py = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv, "bin", "python")
    if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.abspath(venv)):
        return
    if not os.path.exists(py) or os.environ.get("FOLLOWUP_RELAUNCHED") == "1":
        return
    env = dict(os.environ, FOLLOWUP_RELAUNCHED="1")
    child = subprocess.Popen([py, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    while True:
        try:
            sys.exit(child.wait())
        except KeyboardInterrupt:
            continue


if __name__ == "__main__":
    _relaunch_in_project_venv()

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

import argparse
import collections
import csv
import json
import time
from datetime import datetime

import torch            # torch first
import numpy as np

import config

SEED = 42
KD_T = 3.0              # temperature of the self-distillation in the paper; not tuned
KD_LAMBDA = 1.0         # weight of the self-distillation in the paper; not tuned

EXPS = {
    "g12": dict(folds=3, epochs=20, workers=4, model=config.PROPOSED_NAME,
                data=config.GARBAGE12_DIR, pub="10_garbage12_training"),
    "lite": dict(folds=5, epochs=40, workers=0, model="Ablation: WF-Net-Lite",
                 data=config.DATA_DIR, pub="02_main_seed42"),
}
PUB_FOLDER = "WF_Net__Proposed_"        # same split for every model of a run

QUEUE = ([("g12", "nomix", f) for f in range(3)]
         + [("g12", "eased", f) for f in range(3)]
         + [("lite", "kd", f) for f in range(5)]
         + [("g12", "nosd", f) for f in range(3)]
         + [("g12", "default", 0)]
         + [("g12", "default40", f) for f in range(3)])
QUICK_QUEUE = [("g12", "nomix", 0), ("g12", "eased", 0), ("lite", "kd", 0), ("g12", "nosd", 0)]


def root_dir(quick):
    return os.path.join(config.PROJECT_DIR, "results", "11_followup",
                        "_quick_test" if quick else "")


def job_dir(root, exp, arm, fold):
    return os.path.join(root, exp, arm, f"fold{fold}")


# --------------------------------------------------------------------------- #
# One job (runs in its own process)
# --------------------------------------------------------------------------- #
def rebuild_order(exp):
    """Dataset order of the published run, rebuilt from its saved test folds, so that
    the folds are identical to the published ones on any file system."""
    spec = EXPS[exp]
    classes = sorted(d for d in os.listdir(spec["data"])
                     if os.path.isdir(os.path.join(spec["data"], d)))
    c2i = {c: i for i, c in enumerate(classes)}
    idx2rel, pub = {}, []
    for f in range(spec["folds"]):
        d = np.load(os.path.join(config.OUT_ROOT, spec["pub"], "folds", PUB_FOLDER, f"fold{f}",
                                 "preds.npz"), allow_pickle=True)
        pub.append(np.asarray(d["test_idx"]))
        for i, p in zip(d["test_idx"], d["test_paths"]):
            parts = str(p).replace("\\", "/").split("/")
            idx2rel[int(i)] = (parts[-2], parts[-1])
    n = len(idx2rel)
    assert sorted(idx2rel) == list(range(n)), "the published test folds do not partition the data"
    paths, labels = [], []
    for i in range(n):
        c, fn = idx2rel[i]
        fp = os.path.join(spec["data"], c, fn)
        assert os.path.exists(fp), f"missing image: {fp}"
        paths.append(fp)
        labels.append(c2i[c])
    return paths, labels, classes, pub


def apply_arm(exp, arm):
    if exp != "g12":
        return
    if arm == "nomix":
        config.USE_MIXUP = False
    elif arm == "eased":
        import src.data
        import torchvision.transforms as T
        config.USE_MIXUP = False
        config.LABEL_SMOOTHING = 0.0
        config.FUSION_DROPOUT = 0.1
        config.FUSION_DROP_PATH = 0.0

        def eased_tf(train):
            if train:
                return T.Compose([
                    T.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
                    T.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(config.MEAN, config.STD),
                ])
            return T.Compose([
                T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(config.MEAN, config.STD),
            ])
        src.data.build_transforms = eased_tf
    elif arm == "nosd":
        config.DEEP_SUPERVISION = False
    elif arm == "default40":
        EXPS["g12"]["epochs"] = 40
    else:
        assert arm == "default", arm


_TEACHER = {}


def patch_kd_student(fold, n_classes):
    """build_model returns WF-Net-Lite with one extra loss term: distillation toward the
    WF-Net model of the same fold, which was trained only on that fold's training split."""
    import torch.nn.functional as F
    import src.engine
    from src.models import WFNet, ABLATION_MODELS

    t = WFNet(n_classes, pretrained=False)
    ck = os.path.join(config.OUT_ROOT, "02_main_seed42", "checkpoints",
                      f"ckpt_WF_Net__Proposed__fold{fold}.pt")
    t.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    t = t.to(config.DEVICE).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    _TEACHER["m"] = t                    # module-level: never registered in the student

    class KDStudent(WFNet):
        def forward(self, x):
            if self.training:
                with torch.no_grad():
                    self._t_logits = _TEACHER["m"](x)     # same (mixed) input as the student
            return super().forward(x)

        def distillation_loss(self, ce_fn):
            base = super().distillation_loss(ce_fn)
            kd = F.kl_div(F.log_softmax(self._fused_cache / KD_T, dim=1),
                          F.softmax(self._t_logits / KD_T, dim=1),
                          reduction="batchmean") * (KD_T * KD_T)
            return base + KD_LAMBDA * kd

    def build(display_name, num_classes):
        assert display_name == "Ablation: WF-Net-Lite"
        return KDStudent(num_classes, **ABLATION_MODELS[display_name])

    src.engine.build_model = build
    return t


def run_job(exp, arm, fold, quick):
    spec = EXPS[exp]
    config.SEED = SEED
    config.N_FOLDS = spec["folds"]
    config.NUM_WORKERS = spec["workers"]

    root = root_dir(quick)
    jdir = job_dir(root, exp, arm, fold)
    os.makedirs(jdir, exist_ok=True)
    mp = os.path.join(jdir, "metrics.json")
    if os.path.exists(mp):
        return

    apply_arm(exp, arm)
    config.EPOCHS = spec["epochs"]

    from src.data import make_folds
    from src.engine import train_one_fold, _evaluate, _loader
    from src.runner import job_seed
    from src.utils import seed_everything, durable_json, durable_savez, write_run_snapshot

    write_run_snapshot(config, os.path.join(jdir, "run_config.txt"))
    paths, labels, classes, pub = rebuild_order(exp)
    folds = list(make_folds(labels))
    for f, (_, _, te) in enumerate(folds):
        assert np.array_equal(np.sort(te), np.sort(pub[f])), f"fold {f} differs from the published split"
    tr, va, te = folds[fold]
    print(f"[split verified] {exp}: {len(paths)} images, {len(classes)} classes, "
          f"train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    counts = collections.Counter(labels)
    class_counts = [counts[i] for i in range(len(classes))]
    epochs = spec["epochs"]
    if quick:
        tr, va, te, epochs = tr[:64], va[:64], te[:64], 2

    teacher_f1 = None
    if arm == "kd":
        t = patch_kd_student(fold, len(classes))
        tm, *_ = _evaluate(t, _loader(paths, labels, te, False), config.DEVICE, len(classes),
                           tta=config.USE_TTA)
        teacher_f1 = float(tm["macro_f1"])
        print(f"[teacher] fold{fold} test macro-F1 = {teacher_f1:.4f}", flush=True)

    # seeded last, so the student starts exactly as in the published WF-Net-Lite job
    seed_everything(job_seed(SEED, spec["model"], fold), config.DETERMINISTIC)
    t0 = time.time()
    state_path = os.path.join(jdir, "train_state.pt")
    metrics, y_true, y_pred, y_prob, _, _ = train_one_fold(
        spec["model"], paths, labels, tr, va, te, class_counts, len(classes), fold,
        epochs=epochs, cross_targets=None, state_path=state_path)
    durable_savez(os.path.join(jdir, "preds.npz"), y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                  test_idx=np.asarray(te), test_paths=np.array([paths[i] for i in te]))
    rec = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
    rec.update(exp=exp, arm=arm, fold=fold, code=CODE_VERSION, history=metrics["history"],
               wall_s=time.time() - t0, teacher_macro_f1=teacher_f1, device=str(config.DEVICE),
               finished=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    durable_json(mp, rec)                          # done marker (last)
    if os.path.exists(state_path):
        os.remove(state_path)
    print(f"[done] {exp}/{arm}/fold{fold}: macro-F1={rec['macro_f1']:.4f} "
          f"acc={rec['accuracy']:.4f} wall={rec['wall_s']/60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
# Queue driver
# --------------------------------------------------------------------------- #
def published(exp, name):
    out = {}
    p = os.path.join(config.OUT_ROOT, EXPS[exp]["pub"], "per_fold_metrics.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["model"] == name:
                    out[int(r["fold"])] = float(r["macro_f1"])
    return out


def write_summary(root, queue):
    lines = [f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    done = 0
    arms = []
    for exp, arm, _ in queue:
        if (exp, arm) not in arms:
            arms.append((exp, arm))
    pub_g = published("g12", config.PROPOSED_NAME)
    pub_l = published("lite", "Ablation: WF-Net-Lite")
    lines.append("Published: g12/WF-Net  " + " ".join(f"f{k}={v:.4f}" for k, v in sorted(pub_g.items()))
                 + (f"  mean={np.mean(list(pub_g.values())):.4f}" if pub_g else ""))
    lines.append("Published: lite/control " + " ".join(f"f{k}={v:.4f}" for k, v in sorted(pub_l.items()))
                 + (f"  mean={np.mean(list(pub_l.values())):.4f}" if pub_l else ""))
    lines.append("")
    for exp, arm in arms:
        vals = {}
        for e, a, f in queue:
            if (e, a) == (exp, arm):
                mp = os.path.join(job_dir(root, e, a, f), "metrics.json")
                if os.path.exists(mp):
                    with open(mp, encoding="utf-8") as fh:
                        vals[f] = json.load(fh)["macro_f1"]
                    done += 1
        n_all = sum(1 for e, a, _ in queue if (e, a) == (exp, arm))
        txt = " ".join(f"f{k}={v:.4f}" for k, v in sorted(vals.items()))
        mean = f"  mean={np.mean(list(vals.values())):.4f}" if vals else ""
        lines.append(f"{exp}/{arm:10s} [{len(vals)}/{n_all}] {txt}{mean}")
    lines += ["", f"Finished jobs: {done}/{len(queue)}"]
    with open(os.path.join(root, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test in a separate folder")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--job", dest="job", nargs=3, metavar=("EXP", "ARM", "FOLD"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.job:
        run_job(a.job[0], a.job[1], int(a.job[2]), a.quick)
        return 0

    from src.safety import RunLock, keep_awake, free_gb
    root = root_dir(a.quick)
    os.makedirs(root, exist_ok=True)
    queue = QUICK_QUEUE if a.quick else QUEUE
    if a.status:
        print(write_summary(root, queue)[0])
        return 0
    lock = RunLock(root)
    ok, msg = lock.acquire()
    if not ok:
        print(msg)
        return 2
    if free_gb(root) < config.MIN_FREE_DISK_GB:
        print(f"Stopped: at least {config.MIN_FREE_DISK_GB} GB of free disk space is required.")
        return 3
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'}", flush=True)
    keep_awake()
    failed = []
    for k, (exp, arm, fold) in enumerate(queue, 1):
        if os.path.exists(os.path.join(job_dir(root, exp, arm, fold), "metrics.json")):
            print(f"[{k}/{len(queue)}] {exp}/{arm}/fold{fold}: already finished, skipped", flush=True)
            continue
        print(f"\n[{k}/{len(queue)}] {exp}/{arm}/fold{fold} starting ({datetime.now():%H:%M})", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--job", exp, arm, str(fold)]
        if a.quick:
            cmd.append("--quick")
        for attempt in range(3):
            if subprocess.call(cmd) == 0:
                break
            print(f"[{exp}/{arm}/fold{fold}] error (attempt {attempt + 1}/3); retrying in 60 s", flush=True)
            time.sleep(60)
        else:
            failed.append((exp, arm, fold))
        print(write_summary(root, queue)[0], flush=True)
    if failed:
        print(f"\nFailed jobs: {failed}. The same command retries them.", flush=True)
        return 1
    print(f"\nQUEUE FINISHED. Results folder: {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
