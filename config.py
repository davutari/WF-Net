"""
Configuration of the WF-Net experiments.

Defines the complete experimental protocol. The two datasets are read from
<repo>/data (read-only) and every output is written to <repo>/results.
"""
import os
import torch

# --------------------------------------------------------------------------- #
# Paths  (fixed on purpose: no environment-variable overrides, so a stray env var
#         can never redirect a run to other data or other folders)
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))           # repository root
PROJECT_DIR = ROOT_DIR
DATA_ROOT = os.path.join(PROJECT_DIR, "data")                    # datasets (read-only)
DATA_DIR = os.path.join(DATA_ROOT, "WasteSnap_Dataset")
GARBAGE12_DIR = os.path.join(DATA_ROOT, "garbage_classification")

OUT_ROOT = os.path.join(PROJECT_DIR, "results")                  # every output goes here
# Per-step output folders are assigned at runtime by the runner:
RESULTS_DIR = OUT_ROOT
FIG_DIR = os.path.join(OUT_ROOT, "figures")
CKPT_DIR = os.path.join(OUT_ROOT, "checkpoints")
RESULTS_ROOT = OUT_ROOT          # kept for modules that refer to it

# --------------------------------------------------------------------------- #
# Reproducibility  (fixed seed => identical results across runs)
# --------------------------------------------------------------------------- #
SEED = 42
DETERMINISTIC = True

# --------------------------------------------------------------------------- #
# Data / cross-validation protocol
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
N_FOLDS = 5              # 5-fold stratified CV
VAL_SPLIT = 0.15        # carved out of the training folds for early stopping
NUM_WORKERS = 0         # 0 is safest on Windows; raise on Linux
BATCH_SIZE = 16         # small dataset -> small batch

# ImageNet statistics (all backbones are ImageNet-pretrained)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
EPOCHS = 40
FREEZE_EPOCHS = 5       # phase-1: train head only, then unfreeze backbone
LR_HEAD = 1e-3          # phase-1 head learning rate
LR_BACKBONE = 1e-5      # fine-tuning learning rate for the backbone
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0       # focal loss focusing parameter
EARLY_STOP_PATIENCE = 10
MONITOR = "macro_f1"    # early stopping / checkpoint selection metric

USE_AMP = True          # mixed precision if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# 10 baselines (all timm names, all ImageNet-pretrained, identical protocol).
BASELINE_MODELS = {
    "VGG16":            "vgg16",
    "ResNet50":         "resnet50",
    "InceptionV3":      "inception_v3",
    "Xception":         "xception",
    "DenseNet121":      "densenet121",
    "MobileNetV2":      "mobilenetv2_100",
    "EfficientNetB0":   "efficientnet_b0",
    "InceptionResNetV2":"inception_resnet_v2",
    "ViT-B/16":         "vit_base_patch16_224",
    "Swin-T":           "swin_tiny_patch4_window7_224",
}

# --------------------------------------------------------------------------- #
# Proposed model: WF-Net (Waste Fusion Network)
#   Two backbones (CNN + hierarchical Transformer) with in-branch SE refinement,
#   fused by CATF: a standard Transformer encoder layer over the joint token
#   sequence of both branches.
# --------------------------------------------------------------------------- #
PROPOSED_NAME = "WF-Net (Proposed)"
PROPOSED_BACKBONES = ("convnext_tiny", "swin_tiny_patch4_window7_224")
# Lightweight CNN + ViT pair for the backbone-choice ablation
LITE_BACKBONES = ("efficientnet_b0", "deit_tiny_patch16_224")
FUSION_DIM = 384          # common token dimension both branches are projected to
FUSION_HEADS = 6          # multi-head attention heads in the fusion transformer
FUSION_LAYERS = 1         # number of Transformer encoder layers in CATF
FUSION_DROPOUT = 0.3
FUSION_DROP_PATH = 0.1    # stochastic depth on the fusion residual branches
# CATF fusion mechanism: "self" = standard self-attention over the concatenated
# tokens (default, the proposed model); "cross" = explicit bidirectional
# cross-attention between the branches (ablation variant).
CATF_MODE = "self"

# In-branch feature refinement attention: "se" | "cbam" | "none".
# SE is channel-only (no 7x7 spatial conv) and has fewer parameters than CBAM.
BRANCH_ATTENTION = "se"

# Deep self-distillation (training only, zero inference overhead):
# each branch gets an auxiliary head that (a) is supervised directly and
# (b) is distilled towards the detached fused teacher prediction.
DEEP_SUPERVISION = True
AUX_WEIGHT = 0.3          # weight of the auxiliary CE/focal loss per branch
KD_WEIGHT = 1.0           # weight of the self-distillation (KL) term
KD_TEMPERATURE = 3.0      # softening temperature for distillation

# --------------------------------------------------------------------------- #
# Small-data regularization (applied uniformly to every model for fairness)
# --------------------------------------------------------------------------- #
# MixUp / CutMix batch augmentation.
USE_MIXUP = True
MIXUP_ALPHA = 0.2         # Beta(a, a) for MixUp (lower -> milder blends)
CUTMIX_ALPHA = 1.0        # Beta(a, a) for CutMix patch area
MIXUP_PROB = 0.5          # probability of mixing a given training batch
MIXUP_SWITCH_PROB = 0.5   # given a mix, probability of CutMix over MixUp

# EMA of weights: evaluate an exponential moving average of the trained weights.
USE_EMA = True
EMA_DECAY = 0.999

# Test-time augmentation (horizontal-flip averaging) applied to ALL models at
# the final test evaluation.
USE_TTA = True

# --------------------------------------------------------------------------- #
# Artifact saving (checkpoints / predictions / reproducibility snapshot)
# --------------------------------------------------------------------------- #
# Persist the best (EMA) weights of each fold so the trained model can be reused
# or deployed. SAVE_ALL_CHECKPOINTS also stores the 10 baselines + ablations
# (uses a lot of disk); by default only the proposed model is saved.
SAVE_CHECKPOINTS = True
SAVE_ALL_CHECKPOINTS = False
# Dump per-fold predictions (y_true / y_pred / y_prob) so any figure or metric
# can be regenerated without re-training the multi-hour benchmark.
SAVE_PREDICTIONS = True

# --------------------------------------------------------------------------- #
# Cross-dataset generalization  (train on WasteSnap -> test on EXTERNAL sets)
#
# Multiple targets can be evaluated at once; each produces its own table/figure. Because
# label taxonomies differ, only the *shared* classes are scored, via a per-set
# name mapping. Every fold-trained model is evaluated on every external set ->
# mean +/- std cross-dataset macro-F1 per target.
#
# To enable a target: set its directory (edit CROSS_DATASETS) to the
# external dataset root (class-subfolder layout). Unset/missing dirs are skipped.
# --------------------------------------------------------------------------- #
CROSS_RESTRICT_TO_SHARED = True   # mask logits to shared classes (closed-set)
RUN_CROSS = True

# External target: Garbage12.
CROSS_DATASETS = {"Garbage12": GARBAGE12_DIR}

# Per-target label maps: EXTERNAL folder name -> WasteSnap source class name.
# Keys are matched case-/punctuation-insensitively; unmapped folders are skipped.
CROSS_LABEL_MAPS = {
    "Garbage12": {           # Kaggle: mostafaabla/garbage-classification (12 cls)
        "battery":     "Hazardous waste",
        "biological":  "Organic waste",
        "brown-glass": "Glass waste",
        "green-glass": "Glass waste",
        "white-glass": "Glass waste",
        "cardboard":   "Paper and cardboard waste",
        "paper":       "Paper and cardboard waste",
        "metal":       "Metal waste",
        "plastic":     "Plastic waste",
        "trash":       "General waste",
        # clothes, shoes -> unmapped, skipped
    },
}

# Include the ablation variants of WF-Net in the benchmark.
RUN_ABLATION = True

# --------------------------------------------------------------------------- #
# Study plan (used by run_experiments.py)
# --------------------------------------------------------------------------- #
MAIN_SEED = 42                       # main 5-fold CV (all 19 models + Garbage12 zero-shot)
REPEAT_SEEDS = [1337, 2024]          # extra seeds -> 3x5 repeated CV
KEY_MODELS = [                       # competitors re-run in the repeated CV
    PROPOSED_NAME, "ViT-B/16", "Swin-T", "VGG16",
    "Ablation: Dual (concat)", "Ablation: CNN-only",
    "Ablation: WF-Net w/o self-distill",
]
SWEEP_FOLDS = [0, 1, 2]              # distillation sensitivity
SWEEP_LEVELS = {"AUX_WEIGHT": [0.1, 0.3, 0.5],
                "KD_WEIGHT": [0.5, 1.0, 2.0],
                "KD_TEMPERATURE": [2.0, 3.0, 4.0]}
GARBAGE12_MODELS = [PROPOSED_NAME, "ViT-B/16", "Swin-T", "Ablation: Dual (concat)"]
GARBAGE12_FOLDS = 3                  # training on the larger dataset
GARBAGE12_EPOCHS = 20
GARBAGE12_WORKERS = 4                # 15.5k images: parallel JPEG decoding
BOOTSTRAP_RESAMPLES = 2000           # post-hoc statistics

# --------------------------------------------------------------------------- #
# Crash safety (power cut, reboot, Ctrl+C)
# --------------------------------------------------------------------------- #
# The full training state of the running (model, fold) is saved crash-safely
# at the end of an epoch every TRAIN_STATE_EVERY_MIN minutes and when training
# ends -> after an interruption at most this much training is repeated.
TRAIN_STATE_EVERY_MIN = 10
MIN_FREE_DISK_GB = 10                # refuse to start below this (states ~1.2 GB)
