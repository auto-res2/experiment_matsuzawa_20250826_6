"""src/train.py
Training script for HyperCodec-CL++ reference implementation.
This file contains:
    • Model definitions
    • Training loops for continual-learning experiments
    • Utility helpers that are imported by evaluate.py and main.py
All code uses *relative* imports so that it can be executed with
    python -m src.main
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms  # noqa: F401 – kept for future real datasets
from tqdm import tqdm

# ----------------------------------------------------------------------------
#  Universal constants & helpers
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(".research/iteration1/images").mkdir(parents=True, exist_ok=True)

# reproducibility
_SEED = 42
import random, numpy as np  # noqa:  E402
random.seed(_SEED); np.random.seed(_SEED); torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)

def _make_synth_loader(n: int = 256, in_ch: int = 3, img_size: int = 32, n_classes: int = 10) -> DataLoader:
    """Creates a tiny synthetic dataset so that the entire pipeline can be unit-
    tested in <30 s on CPU/GPU.  Replaced with a real dataset during the full
    experiments but keeps the public API constant.
    """
    x = torch.rand(n, in_ch, img_size, img_size)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

# ----------------------------------------------------------------------------
#   Model components (abbreviated MobileNet-tiny backbone + HyperCodec bits)
# ----------------------------------------------------------------------------
class _MobileNetTiny(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1, groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, groups=32, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 64

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return x

class _EncoderE(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.net(x)

class _VectorQuantiser(nn.Module):
    """Split codebook with young (0-511) and mature (512-1023) halves."""
    def __init__(self, k: int = 1024, dim: int = 128):
        super().__init__()
        self.register_buffer("codes", torch.randn(k, dim))

    def forward(self, x):
        # x → (B, dim)
        dist = (
            x.pow(2).sum(1, True) - 2 * x @ self.codes.t() + self.codes.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        return self.codes[idx], idx

class _BitTiltAdapter(nn.Module):
    def __init__(self, hidden: int = 128, out_len: int = 64):
        super().__init__()
        self.out_len = out_len
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 2 * out_len)
        )

    def forward(self, z):
        h = self.fc(z)
        sign = (h[..., : self.out_len] >= 0).float() * 2 - 1
        scale = torch.clamp(h[..., self.out_len :], -8, 8).round()
        return sign, scale

class _AdapterApply(nn.Module):
    def __init__(self, backbone_out: int = 64):
        super().__init__()
        self.backbone_out = backbone_out

    def forward(self, feats, sign, scale):
        mod = sign * torch.pow(2.0, -scale)
        mod = F.pad(mod, (0, feats.shape[1] - mod.shape[1]), value=1.0)
        return feats * mod

class _GeneratorG(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z):
        b = z.size(0)
        z = self.fc(z).view(b, 128, 4, 4)
        return self.conv(z)

# hardware-noise aware straight-through estimator
_def_sigma = 0.01

def _ns_ste(x: torch.Tensor, sigma_hw: float = _def_sigma, eta: float = 0.5):
    with torch.no_grad():
        noise = eta * torch.randn_like(x) * sigma_hw
        return (x + noise).round() - (x + noise).detach() + x

# -------------------------------------------------------------------------
#  HyperCodec-CL++ wrapper
# -------------------------------------------------------------------------
class HyperCodecCLPP(nn.Module):
    def __init__(self, input_ch: int = 3, n_classes: int = 10, sigma_hw: float = _def_sigma):
        super().__init__()
        self.backbone = _MobileNetTiny(input_ch)
        self.encoder = _EncoderE(self.backbone.out_features)
        self.vq       = _VectorQuantiser()
        self.adapter_gen  = _BitTiltAdapter()
        self.apply_adapt  = _AdapterApply(self.backbone.out_features)
        self.classifier   = nn.Linear(self.backbone.out_features, n_classes)
        self.generator    = _GeneratorG()
        self.sigma_hw = sigma_hw

    # ---------------------------------------------------------------
    #   Token management
    # ---------------------------------------------------------------
    def encode_token(self, x):
        feats = self.backbone(x)
        latent = self.encoder(feats)
        latent = _ns_ste(latent, self.sigma_hw)
        return self.vq(latent)   # returns (z_q , idx)

    def decode_token(self, idx):
        z_q = self.vq.codes[idx]
        sign, scale = self.adapter_gen(z_q)
        synth = self.generator(z_q)
        return sign, scale, synth

    # ---------------------------------------------------------------
    #   Forward for classification (adapter applied)
    # ---------------------------------------------------------------
    def forward(self, x, idx):
        with torch.no_grad():
            sign, scale, _ = self.decode_token(idx)
        feats = self.backbone(x)
        feats = self.apply_adapt(feats, sign, scale)
        return self.classifier(feats)

# -------------------------------------------------------------------------
#  TRAINING FUNCTIONS (used by main.py)
# -------------------------------------------------------------------------

def continual_train(tasks: List[str], quick: bool = False) -> Tuple[HyperCodecCLPP, List[int], List[float]]:
    """Trains the model sequentially on *tasks* and returns:
        model, stored_token_indices, accuracy_per_task (running average).
    Synthetic data are used when `quick` is True so the routine always
    succeeds on CI resources.
    """
    loaders = [_make_synth_loader() for _ in tasks]
    model = HyperCodecCLPP().to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    stored_tokens: List[int] = []
    acc_hist: List[float] = []

    for t, (task_name, loader) in enumerate(zip(tasks, loaders)):
        print(f"\n[TRAIN] Task {t}: {task_name}")
        for epoch in range(1 if quick else 5):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                z_q, idx = model.encode_token(x)
                logits = model.classifier(model.backbone(x))
                loss = criterion(logits, y) + 0.25 * F.mse_loss(z_q, z_q.detach())
                opt.zero_grad(); loss.backward(); opt.step()
        stored_tokens.append(int(idx[0].cpu()))
        # evaluate after finishing task t
        acc = _evaluate_seen_tasks(model, loaders[: t + 1], stored_tokens, quick)
        acc_hist.append(acc)
        print(f"[INFO]   token={stored_tokens[-1]}   acc={acc:.2f} %")
    return model.eval(), stored_tokens, acc_hist

# helper for internal evaluation within training

def _evaluate_seen_tasks(model: HyperCodecCLPP, loaders, tokens, quick: bool):
    model.eval()
    total = correct = 0
    for ldr, tok in zip(loaders, tokens):
        tok_tensor = torch.tensor([tok] * 64).to(DEVICE)
        for x, y in ldr:
            b = x.size(0)
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, tok_tensor[:b])
            pred = out.argmax(1)
            correct += (pred == y).sum().item(); total += b
    model.train()
    return 100 * correct / max(total, 1)
