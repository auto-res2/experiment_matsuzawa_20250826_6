#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/train.py
Training utilities and model definitions for RevMoDiff experiments.

Notes:
- Uses only relative imports within src per project rules.
- Figures saved as PDF into .research/iteration1/images.
- Designed to run a quick toy experiment on CPU or a single NVIDIA T4 GPU (16 GB).
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from .preprocess import (
    get_device,
    autocast_if_cuda,
    get_scaler,
    reset_peak_mem,
    max_mem_allocated_mib,
    ensure_dir,
    set_fig_defaults,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Diffusion schedule helper
# =============================================================================

class DDPMHelper:
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device=None):
        self.device = device if device is not None else get_device()
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register(betas, alphas, alphas_cumprod)

    def register(self, betas, alphas, alphas_cumprod):
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        # Ensure buffers are on the same device as the inputs for safe indexing and math
        buf_dev = t.device
        sqrt_ac = self.sqrt_alphas_cumprod.to(buf_dev)[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod.to(buf_dev)[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def t_embedding(self, t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device)
        )
        args = t.float().unsqueeze(1) / self.T * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =============================================================================
# Parameter composer (FiLM-like modulation) with shared weights
# =============================================================================

class FiLMComposer(nn.Module):
    def __init__(self, cond_dim: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(),
            nn.Linear(128, 2 * out_channels)
        )
        self.register_buffer('scale', torch.tensor(0.1))

    def forward(self, cond: torch.Tensor):
        x = self.net(cond)
        gamma, beta = x.chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma)
        beta = self.scale * torch.tanh(beta)
        return gamma, beta


class SharedConv2d(nn.Module):
    """Simple shared Conv2d used across blocks. Parameters are shared by reference."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# Reversible affine coupling with group slicing
# =============================================================================

class STNet(nn.Module):
    """
    Processes half-channels in G groups. Each group uses the same shared convs.
    in_half = ch // 2. We slice into groups of gsz = in_half // groups.
    shared_conv1: [gsz -> hidden_group], shared_conv2: [hidden_group -> 2*gsz].
    """
    def __init__(self, ch: int, groups: int, hidden_group: int,
                 shared_conv1: SharedConv2d, shared_conv2: SharedConv2d,
                 composer: FiLMComposer, block_id: int, column_id: int):
        super().__init__()
        assert ch % 2 == 0, "ch must be even"
        self.in_half = ch // 2
        assert self.in_half % groups == 0, "(ch//2) must be divisible by groups"
        self.groups = groups
        self.gsz = self.in_half // groups
        self.hidden_group = hidden_group
        # GroupNorm per-group
        self.gn = nn.GroupNorm(num_groups=min(8, self.gsz), num_channels=self.gsz)
        self.act = nn.SiLU()
        self.shared_conv1 = shared_conv1  # in: gsz, out: hidden_group
        self.shared_conv2 = shared_conv2  # in: hidden_group, out: 2*gsz
        self.composer = composer
        self.block_id = block_id
        self.column_id = column_id

    def forward(self, x1: torch.Tensor, cond_vec: torch.Tensor):
        # x1: [B, in_half, H, W]
        B, C, H, W = x1.shape
        assert C == self.in_half
        gamma, beta = self.composer(cond_vec)  # [B, 2*base_ch]
        # split FiLM across s and t of total size C
        gamma_s, gamma_t = gamma.chunk(2, dim=-1)
        beta_s, beta_t = beta.chunk(2, dim=-1)
        # But gamma_s/t expect [B, C]; composer produced size base_ch (= ch), and C = ch//2.
        # So we take the first C for s and next C for t via chunk already.
        gamma_s = gamma_s[:, :C]
        gamma_t = gamma_t[:, :C]
        beta_s = beta_s[:, :C]
        beta_t = beta_t[:, :C]

        y_parts = []
        for g in range(self.groups):
            xs = x1[:, g*self.gsz:(g+1)*self.gsz]
            ys = self.gn(xs)
            ys = self.act(ys)
            ys = self.shared_conv1(ys)
            ys = self.act(ys)
            ys = self.shared_conv2(ys)
            y_parts.append(ys)  # [B, 2*gsz, H, W]
        y = torch.cat(y_parts, dim=1)  # [B, 2*C, H, W]
        s, t = y.chunk(2, dim=1)  # each [B, C, H, W]
        # Apply FiLM
        s = s * gamma_s.view(B, -1, 1, 1) + beta_s.view(B, -1, 1, 1)
        t = t * gamma_t.view(B, -1, 1, 1) + beta_t.view(B, -1, 1, 1)
        s = torch.tanh(s)
        return s, t


class AffineCouplingRev(nn.Module):
    def __init__(self, ch: int, groups: int, hidden_group: int,
                 shared_conv1: SharedConv2d, shared_conv2: SharedConv2d,
                 composer: FiLMComposer, block_id: int, column_id: int):
        super().__init__()
        self.stnet = STNet(ch=ch, groups=groups, hidden_group=hidden_group,
                           shared_conv1=shared_conv1, shared_conv2=shared_conv2,
                           composer=composer, block_id=block_id, column_id=column_id)
        self.ch = ch

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.stnet(x1, cond_vec)
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        return y

    def inverse(self, y: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=1)
        s, t = self.stnet(y1, cond_vec)
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)
        return x


class _RevFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, cond_vec):
        with torch.no_grad():
            y = module.forward(x, cond_vec)
        ctx.module = module
        ctx.cond_vec = cond_vec
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        module = ctx.module
        cond_vec = ctx.cond_vec
        with torch.no_grad():
            x = module.inverse(y, cond_vec)
        x.requires_grad_(True)
        with torch.enable_grad():
            y_re = module.forward(x, cond_vec)
            dx = torch.autograd.grad(y_re, x, dy, retain_graph=False, create_graph=False)[0]
        return dx, None, None


class ReversibleWrapper(nn.Module):
    def __init__(self, module: AffineCouplingRev):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        return _RevFunction.apply(x, self.module, cond_vec)


# =============================================================================
# Progressive columns + Rev blocks model (RevMoDiff)
# =============================================================================

class Column(nn.Module):
    def __init__(self, ch: int, num_blocks: int, groups: int, hidden_group: int,
                 composer: FiLMComposer, column_id: int, global_block_offset: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_half = ch // 2
        assert in_half % groups == 0
        gsz = in_half // groups
        # Shared convs per column operating on per-group channels
        shared1 = SharedConv2d(gsz, hidden_group)
        shared2 = SharedConv2d(hidden_group, 2 * gsz)
        for i in range(num_blocks):
            block_id = global_block_offset + i
            block = AffineCouplingRev(
                ch=ch,
                groups=groups,
                hidden_group=hidden_group,
                shared_conv1=shared1,
                shared_conv2=shared2,
                composer=composer,
                block_id=block_id,
                column_id=column_id,
            )
            self.blocks.append(ReversibleWrapper(block))

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, cond_vec)
        return x


class ColumnGater(nn.Module):
    def __init__(self, num_cols: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(), nn.Linear(128, num_cols)
        )

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        logits = self.net(cond_vec)
        return torch.sigmoid(logits)


class RevMoDiff(nn.Module):
    def __init__(self, img_ch: int = 3, base_ch: int = 32, hidden_group: int = 32,
                 num_cols: int = 2, num_blocks_per_col: int = 2, groups: int = 2,
                 timesteps: int = 100, text_dim: int = 0, use_gates: bool = False):
        super().__init__()
        self.img_ch = img_ch
        self.base_ch = base_ch
        self.timesteps = timesteps
        self.text_dim = text_dim
        self.groups = groups
        # Stem and head
        self.stem = nn.Conv2d(img_ch, base_ch, 3, padding=1)
        self.head = nn.Conv2d(base_ch, img_ch, 3, padding=1)
        # Composer for FiLM (produces size 2*base_ch to split into s,t)
        self.cond_dim = 64 + 2 + (text_dim if text_dim > 0 else 0)  # t_emb + layer idx + column idx + optional text
        self.composer = FiLMComposer(cond_dim=self.cond_dim, out_channels=base_ch)
        # Columns
        self.columns = nn.ModuleList()
        global_offset = 0
        for k in range(num_cols):
            col = Column(ch=base_ch, num_blocks=num_blocks_per_col, groups=groups, hidden_group=hidden_group,
                         composer=self.composer, column_id=k, global_block_offset=global_offset)
            global_offset += num_blocks_per_col
            self.columns.append(col)
        # Gater
        self.use_gates = use_gates
        self.gater = ColumnGater(num_cols=num_cols, cond_dim=self.cond_dim) if use_gates else None
        # Diffusion helper for embeddings
        self.diff = DDPMHelper(timesteps=timesteps)

    def build_cond_vec(self, t: torch.Tensor, layer_idx: torch.Tensor, col_idx: torch.Tensor, text: Optional[torch.Tensor]):
        t_emb = self.diff.t_embedding(t, dim=64)
        li = layer_idx.float().unsqueeze(1)
        ci = col_idx.float().unsqueeze(1)
        if text is not None and self.text_dim > 0:
            cond = torch.cat([t_emb, li, ci, text], dim=1)
        else:
            cond = torch.cat([t_emb, li, ci], dim=1)
        return cond

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, text: Optional[torch.Tensor] = None, return_gates: bool = False):
        B = x_t.size(0)
        h = self.stem(x_t)
        layer_idx = torch.zeros(B, device=x_t.device)
        out = h
        gates_to_return = None
        if self.use_gates and self.gater is not None:
            cond_g = self.build_cond_vec(t, torch.zeros_like(layer_idx), torch.zeros_like(layer_idx), text)
            gates = self.gater(cond_g)  # [B, K]
            gates_to_return = gates.detach()
        for k, col in enumerate(self.columns):
            col_idx = torch.full((B,), float(k), device=x_t.device)
            cond = self.build_cond_vec(t, layer_idx, col_idx, text)
            if self.use_gates and self.gater is not None:
                gk = gates[:, k].view(B, 1, 1, 1)
                out_k = col(out, cond)
                out = out + gk * (out_k - out)
            else:
                out = col(out, cond)
        eps_hat = self.head(out)
        if return_gates:
            return eps_hat, gates_to_return
        return eps_hat


# =============================================================================
# Baseline UNet with activation checkpointing
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, img_ch=3, base_ch=32, timesteps=100):
        super().__init__()
        self.enc1 = ConvBlock(img_ch, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_ch, base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out = nn.Conv2d(base_ch, img_ch, 3, padding=1)
        self.diff = DDPMHelper(timesteps=timesteps)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, text: Optional[torch.Tensor] = None):
        h1 = ckpt(self.enc1, x_t)
        d1 = self.down1(h1)
        h2 = ckpt(self.enc2, d1)
        u1 = self.up1(h2)
        h = torch.cat([u1, h1], dim=1)
        h = ckpt(self.dec1, h)
        out = self.out(h)
        return out


# =============================================================================
# Training helpers and plotting
# =============================================================================

@dataclass
class TrainConfig:
    steps: int = 20
    batch_size: int = 8
    lr: float = 2e-3
    img_size: int = 32
    timesteps: int = 100


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_model(model: nn.Module, loader, cfg: TrainConfig, label: str) -> Dict[str, List[float]]:
    device = get_device()
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = get_scaler()
    diff = DDPMHelper(cfg.timesteps, device=device)

    losses: List[float] = []
    iters = 0
    start_time = time.time()
    reset_peak_mem()
    for step, (x0, y) in enumerate(loader):
        if iters >= cfg.steps:
            break
        x0 = x0.to(device, non_blocking=torch.cuda.is_available())
        b = x0.size(0)
        t = torch.randint(0, cfg.timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        x_t = diff.q_sample(x0, t, noise)
        with autocast_if_cuda():
            pred = model(x_t, t, None)
            loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if (iters % 5) == 0:
            print(f"[{label}] step={iters:03d}, loss={loss.item():.4f}")
        iters += 1
    wall = time.time() - start_time
    peak = max_mem_allocated_mib()
    if peak is None:
        peak = float('nan')
    print(f"[{label}] done. steps={iters}, wall={wall:.2f}s, peak_mem(MiB)={peak if not math.isnan(peak) else 'N/A'}")
    return {"loss": losses, "wall": [wall], "peak_mem_mib": [peak]}


def save_loss_plot(losses: List[float], title: str, filepath_pdf: str):
    ensure_dir(os.path.dirname(filepath_pdf))
    set_fig_defaults()
    plt.figure(figsize=(4,3))
    sns.lineplot(x=np.arange(len(losses)), y=losses)
    plt.xlabel('Step')
    plt.ylabel('MSE (noise pred)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {filepath_pdf}")


def save_bar_plot(categories: List[str], values: List[float], title: str, ylabel: str, filepath_pdf: str):
    ensure_dir(os.path.dirname(filepath_pdf))
    set_fig_defaults()
    plt.figure(figsize=(4,3))
    sns.barplot(x=categories, y=values)
    plt.ylabel(ylabel)
    plt.title(title)
    for i, v in enumerate(values):
        try:
            plt.text(i, v, f"{v:.1f}", ha='center', va='bottom')
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(filepath_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {filepath_pdf}")


def save_image_grid_pdf(images: torch.Tensor, filepath_pdf: str, nrow: int = 4):
    """Save a grid of images (in [-1,1]) to a PDF using matplotlib."""
    ensure_dir(os.path.dirname(filepath_pdf))
    set_fig_defaults()
    x = images.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0  # [0,1]
    B = x.size(0)
    ncol = nrow
    nrow_eff = int(np.ceil(B / ncol))
    fig, axes = plt.subplots(nrow_eff, ncol, figsize=(2*ncol, 2*nrow_eff))
    axes = np.array(axes).reshape(nrow_eff, ncol)
    idx = 0
    for r in range(nrow_eff):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis('off')
            if idx < B:
                img = x[idx].permute(1,2,0).numpy()
                ax.imshow(img)
            idx += 1
    plt.tight_layout()
    plt.savefig(filepath_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved {filepath_pdf}")
