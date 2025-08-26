#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/preprocess.py
Synthetic data preparation and shared utilities.
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_fig_defaults():
    import matplotlib as mpl
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['figure.dpi'] = 200


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def autocast_if_cuda():
    return torch.cuda.amp.autocast(enabled=torch.cuda.is_available())


def get_scaler():
    return torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def max_mem_allocated_mib() -> Optional[float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
    return None


# =============================================================================
# Synthetic dataset
# =============================================================================

class SyntheticImages(Dataset):
    def __init__(self, n: int = 2048, size: int = 32, pattern: str = 'blobs', num_classes: int = 10, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.images = []
        self.labels = []
        for _ in range(n):
            label = rng.randint(0, num_classes)
            img = self.make_image(size, pattern, rng, label)
            self.images.append(img.astype(np.float32))
            self.labels.append(label)
        self.images = np.stack(self.images, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)

    @staticmethod
    def normalize(img):
        img = img / 127.5 - 1.0
        return img

    def make_image(self, size: int, pattern: str, rng: np.random.RandomState, label: int) -> np.ndarray:
        img = np.zeros((3, size, size), dtype=np.float32)
        if pattern == 'blobs':
            cx = (label % 5 + 0.5) / 5.0 * size
            cy = (label // 5 + 0.5) / 5.0 * size
            xv, yv = np.meshgrid(np.arange(size), np.arange(size))
            d = ((xv - cx) ** 2 + (yv - cy) ** 2) / (2.0 * (size * 0.08 + rng.rand() * 2) ** 2)
            blob = np.exp(-d)
            color = rng.rand(3, 1, 1) * 255.0
            img = color * blob[None, :, :]
        elif pattern == 'checkerboard':
            freq = 2 + (label % 4)
            p = (np.add.outer(np.arange(size), np.arange(size)) % (size // freq) < (size // (2*freq))).astype(np.float32)
            img = np.stack([p*255.0, np.roll(p, 1, axis=0)*255.0, np.roll(p, 1, axis=1)*255.0], axis=0)
        elif pattern == 'stripes':
            stripe_w = max(1, size // (4 + (label % 3)))
            p = (np.arange(size)[None, :] // stripe_w) % 2
            img = np.stack([p*255.0, np.roll(p, 1, axis=1)*255.0, np.roll(p, 2, axis=1)*255.0], axis=0)
        else:
            img = rng.rand(3, size, size).astype(np.float32) * 255.0
        img = self.normalize(img)
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx])
        y = int(self.labels[idx])
        return x, y


def make_dataloaders(batch_size: int = 16, size: int = 32, n: int = 1024, seed: int = 0):
    ds1 = SyntheticImages(n=n, size=size, pattern='blobs', seed=seed)
    ds2 = SyntheticImages(n=n, size=size, pattern='checkerboard', seed=seed+1)
    ds3 = SyntheticImages(n=n, size=size, pattern='stripes', seed=seed+2)
    def loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    return loader(ds1), loader(ds2), loader(ds3)
