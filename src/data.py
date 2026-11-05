"""
Dataset discovery, transforms and Stratified K-Fold splitting.

Guarantees implemented here:
  * The test fold is NEVER seen during training/validation (no data leakage).
  * Augmentation is applied to the training split ONLY; val/test use a clean
    deterministic transform.
  * Splits are stratified so every fold preserves the class distribution.
"""
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T

import config


def discover_dataset(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """Scan the class-subfolder layout and return (paths, labels, class_names)."""
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(data_dir, cname)
        for fname in sorted(os.listdir(cdir), key=str.upper):   # same order on every file system
            if os.path.splitext(fname)[1].lower() in exts:
                paths.append(os.path.join(cdir, fname))
                labels.append(idx)
    return paths, labels, class_names


def build_transforms(train: bool):
    """Strong augmentation for training; deterministic pipeline for eval."""
    if train:
        return T.Compose([
            T.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            T.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(20),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.RandAugment(num_ops=2, magnitude=7),
            T.ToTensor(),
            T.Normalize(config.MEAN, config.STD),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
    ])


class WasteDataset(Dataset):
    def __init__(self, paths, labels, train: bool):
        self.paths = paths
        self.labels = labels
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def make_folds(labels: List[int]):
    """Yield (train_idx, val_idx, test_idx) for every stratified fold."""
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                          random_state=config.SEED)
    for trainval_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        # carve a stratified validation set out of the training portion
        tr_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=config.VAL_SPLIT,
            stratify=labels[trainval_idx],
            random_state=config.SEED,
        )
        yield tr_idx, val_idx, test_idx
