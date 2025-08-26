#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/evaluate.py
Evaluation and sampling utilities for RevMoDiff experiments.
"""

import os
import time
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch

from .train import RevMoDiff, DDPMHelper, save_bar_plot, save_image_grid_pdf
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


@torch.no_grad()
def simple_sampler(model: RevMoDiff, shape: Tuple[int, int, int, int], steps: int = 20,
                   text: Optional[torch.Tensor] = None, use_gates: bool = False) -> torch.Tensor:
    device = next(model.parameters()).device
    B, C, H, W = shape
    x = torch.randn(B, C, H, W, device=device)
    betas = model.diff.betas
    alphas = model.diff.alphas
    alphas_cum = model.diff.alphas_cumprod
    for i in reversed(range(steps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        with autocast_if_cuda():
            orig_use = model.use_gates
            model.use_gates = use_gates
            eps = model(x, t, text)
            model.use_gates = orig_use
        a_t = alphas[t].view(B, 1, 1, 1)
        b_t = betas[t].view(B, 1, 1, 1)
        ac_t = alphas_cum[t].view(B, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - ac_t) * eps) / torch.sqrt(ac_t)
        mean = torch.sqrt(a_t) * x + (1 - torch.sqrt(a_t)) * x0_hat
        if i > 0:
            x = mean + torch.sqrt(b_t) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1, 1)


def measure_latency_and_memory(model: RevMoDiff, shape: Tuple[int, int, int, int], steps: int = 20,
                               use_gates: bool = False) -> Dict[str, float]:
    text = None
    if model.text_dim > 0:
        text = torch.randn(shape[0], model.text_dim, device=next(model.parameters()).device)
    reset_peak_mem()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    _ = simple_sampler(model, shape, steps=steps, text=text, use_gates=use_gates)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.time() - t0
    peak = max_mem_allocated_mib()
    peak = peak if peak is not None else float('nan')
    return {"latency_s": latency, "peak_mem_mib": peak}


def plot_gating_usage(model: RevMoDiff, shape: Tuple[int, int, int, int], steps: int, filepath_pdf: str):
    ensure_dir(os.path.dirname(filepath_pdf))
    set_fig_defaults()
    device = next(model.parameters()).device
    B = shape[0]
    gates_over_t: List[np.ndarray] = []
    for i in range(steps):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        cond = model.build_cond_vec(t, torch.zeros(B, device=device), torch.zeros(B, device=device), None)
        gates = model.gater(cond) if (model.gater is not None) else torch.ones(B, len(model.columns), device=device)
        gates_over_t.append(gates.mean(dim=0).detach().cpu().numpy())
    gates_over_t = np.stack(gates_over_t, axis=0)  # [steps, K]
    plt.figure(figsize=(4,3))
    for k in range(gates_over_t.shape[1]):
        plt.plot(np.arange(steps), gates_over_t[:, k], label=f'col{k}')
    plt.xlabel('Timestep')
    plt.ylabel('Mean gate')
    plt.title('Column gates over timesteps')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {filepath_pdf}")


def can_train_resolution(model_builder, H: int = 1024, W: int = 1024, timesteps: int = 50) -> bool:
    device = get_device()
    if device.type != 'cuda':
        print("High-res feasibility check: CUDA not available; skipping.")
        return False
    try:
        model = model_builder().to(device)
        model.train()
        diff = DDPMHelper(timesteps, device=device)
        x0 = torch.randn(1, 3, H, W, device=device)
        t = torch.randint(0, timesteps, (1,), device=device)
        noise = torch.randn_like(x0)
        x_t = diff.q_sample(x0, t, noise)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = get_scaler()
        reset_peak_mem()
        with autocast_if_cuda():
            pred = model(x_t, t, None)
            loss = torch.nn.functional.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        peak = max_mem_allocated_mib()
        if peak is not None:
            print(f"High-res {H}x{W} success. Peak mem(MiB)={peak:.1f}")
        else:
            print(f"High-res {H}x{W} success.")
        return True
    except RuntimeError as e:
        print("High-res feasibility failed:", str(e))
        return False
