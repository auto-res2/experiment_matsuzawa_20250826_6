"""
preprocess.py
=============
Data preparation utilities.  For reproducibility and small memory footprint the
script downloads CIFAR-10 (60k × 32² RGB) into ``data/`` and performs a
five-class subset selection so that the quick experiment ends in seconds.

The returned DataLoaders are used by ``src.train`` and ``src.evaluate``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

# -----------------------------------------------------------------------------

def _cifar_subset(root: Path, train: bool = True, n_class: int = 5):
    ds = CIFAR10(root, train=train, download=True,
                 transform=T.Compose([
                     T.ToTensor(),
                     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 ]))
    # select only the first n_class categories to reduce run-time
    idx = [i for i, (_, y) in enumerate(ds) if y < n_class]
    return Subset(ds, idx)

# -----------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 64, n_class: int = 5,
                    num_workers: int = 2) -> Tuple[Dict[str, DataLoader], int]:
    root = Path("data")
    train_set = _cifar_subset(root, True, n_class)
    val_set   = _cifar_subset(root, False, n_class)

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True),
    }
    return loaders, n_class
