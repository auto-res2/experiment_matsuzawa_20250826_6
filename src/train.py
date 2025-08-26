"""
train.py
========
Training utilities for RevDR-VM v2 toy experiments.
This file purposefully keeps the implementation *minimal* so that the full
experiment pipeline can finish in < 1 min on a single NVIDIA Tesla-T4 or on CPU
for CI.

The trainer receives ready-made DataLoaders from ``src.preprocess`` and trains a
very small toy network (≈45 k parameters).  The same Trainer can later be
plugged into the real RevDR-VM kernels because the API is identical.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
#  Tiny RevDR-VM toy backbone (identical to the one shown in the proposal)
# -----------------------------------------------------------------------------
class RevCouple(nn.Module):
    def __init__(self, dim: int, stripes: int = 4):
        super().__init__()
        inner = dim // stripes // 2
        self.F = nn.Sequential(
            nn.Conv2d(dim // 2, inner, 1), nn.GELU(),
            nn.Conv2d(inner, dim // 2, 1), nn.SiLU(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        y1 = x2
        gated = torch.sigmoid(self.alpha) * x2
        y2 = x1 + self.F(gated)
        return y1, y2

    def inverse(self, y1: torch.Tensor, y2: torch.Tensor):
        x2 = y1
        gated = torch.sigmoid(self.alpha) * x2
        x1 = y2 - self.F(gated)
        return x1, x2


class TinyRevDRVM(nn.Module):
    """A very small CNN-style network enhanced with a reversible block."""

    def __init__(self, n_class: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.rev = RevCouple(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_class)

    def forward(self, x):
        x = self.stem(x)
        c = x.shape[1] // 2
        y1, y2 = self.rev(x[:, :c], x[:, c:])
        x = torch.cat([y1, y2], 1)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, device: torch.device, epochs: int = 2, lr: float = 3e-4):
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.model = TinyRevDRVM().to(device)
        self.opt = AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def _step(self, batch, train: bool = True):
        imgs, labels = [t.to(self.device) for t in batch]
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        if train:
            loss.backward()
            self.opt.step(); self.opt.zero_grad()
        pred = logits.argmax(1)
        acc = (pred == labels).float().mean().item()
        return loss.item(), acc

    # ------------------------------------------------------------------
    def fit(self, loaders: Dict[str, torch.utils.data.DataLoader]) -> List[Dict]:
        history: List[Dict] = []
        for ep in range(1, self.epochs + 1):
            for phase in ("train", "val"):
                self.model.train(phase == "train")
                losses, accs = [], []
                loop = tqdm(loaders[phase], leave=False)
                for batch in loop:
                    l, a = self._step(batch, train=(phase == "train"))
                    losses.append(l); accs.append(a)
                    loop.set_description(f"{phase} e{ep}")
                log = {
                    "epoch": ep,
                    "phase": phase,
                    "loss": sum(losses) / len(losses),
                    "acc":  sum(accs) / len(accs),
                }
                history.append(log)
                print(f"{phase.upper()}  ep {ep:02d} | loss {log['loss']:.3f} | "
                      f"acc {log['acc']*100:5.2f} %")
        # ----------- save checkpoint -----------------
        Path("models").mkdir(exist_ok=True)
        ckpt_path = Path("models") / "tiny_revdrvm.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Model saved → {ckpt_path.relative_to(Path.cwd())}")
        return history
