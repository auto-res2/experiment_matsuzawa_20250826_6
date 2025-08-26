#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/main.py
Entry point to run RevMoDiff experiments from project root:
    python -m src.main
"""

import os
import time
import yaml
import math

import torch

from .preprocess import seed_all, make_dataloaders, ensure_dir, get_device
from .train import (
    RevMoDiff, TinyUNet, TrainConfig, train_one_model, count_parameters,
    save_loss_plot, save_bar_plot, save_image_grid_pdf,
)
from .evaluate import (
    simple_sampler, measure_latency_and_memory, plot_gating_usage, can_train_resolution
)


FIG_DIR = os.path.join('.research', 'iteration3', 'images')
ensure_dir(FIG_DIR)
MODEL_DIR = os.path.join('models')
ensure_dir(MODEL_DIR)


def load_config(path: str = 'config/experiment.yaml'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    # Fallback defaults
    return {
        'seed': 123,
        'timesteps': 100,
        'steps_toy': 20,
        'batch_size': 8,
        'img_size': 32,
        'use_gates': True,
    }


def experiment_1_toy(cfg_dict):
    print("\n==== Experiment 1 (toy) — Training-time memory & loss parity ====")
    seed_all(cfg_dict.get('seed', 123))
    loader_blobs, loader_check, _ = make_dataloaders(batch_size=cfg_dict.get('batch_size', 8), size=cfg_dict.get('img_size', 32), n=256)

    cfg = TrainConfig(steps=cfg_dict.get('steps_toy', 20), batch_size=cfg_dict.get('batch_size', 8), lr=2e-3, img_size=cfg_dict.get('img_size', 32), timesteps=cfg_dict.get('timesteps', 100))

    rev = RevMoDiff(img_ch=3, base_ch=32, hidden_group=32, num_cols=2, num_blocks_per_col=2, groups=2,
                    timesteps=cfg.timesteps, text_dim=0, use_gates=False)
    base = TinyUNet(img_ch=3, base_ch=32, timesteps=cfg.timesteps)

    print(f"RevMoDiff params: {count_parameters(rev)/1e6:.3f} M")
    print(f"Baseline  params: {count_parameters(base)/1e6:.3f} M")

    res_rev_blobs = train_one_model(rev, loader_blobs, cfg, label='RevMoDiff/blobs')
    res_base_blobs = train_one_model(base, loader_blobs, cfg, label='Baseline/blobs')

    # Also a short training on a different pattern to sample memory variation
    res_rev_check = train_one_model(rev, loader_check, cfg, label='RevMoDiff/check')
    res_base_check = train_one_model(base, loader_check, cfg, label='Baseline/check')

    # Plots
    save_loss_plot(res_rev_blobs['loss'], 'Training loss (RevMoDiff, blobs)', os.path.join(FIG_DIR, 'training_loss_revmodeff.pdf'))
    save_loss_plot(res_base_blobs['loss'], 'Training loss (Baseline, blobs)', os.path.join(FIG_DIR, 'training_loss_baseline.pdf'))

    peak_rev = float(min([x for x in res_rev_blobs['peak_mem_mib'] + res_rev_check['peak_mem_mib'] if not math.isnan(x)] + [float('inf')]))
    peak_base = float(min([x for x in res_base_blobs['peak_mem_mib'] + res_base_check['peak_mem_mib'] if not math.isnan(x)] + [float('inf')]))
    if peak_rev == float('inf'): peak_rev = float('nan')
    if peak_base == float('inf'): peak_base = float('nan')
    cats = ['Baseline', 'RevMoDiff']
    vals = [peak_base if not math.isnan(peak_base) else 0.0, peak_rev if not math.isnan(peak_rev) else 0.0]
    save_bar_plot(cats, vals, 'Peak training memory (MiB)', 'MiB', os.path.join(FIG_DIR, 'peak_memory_baseline_vs_revmodeff.pdf'))

    # Save a tiny checkpoint
    torch.save({'revmodeff': rev.state_dict()}, os.path.join(MODEL_DIR, 'revmodeff_toy.pt'))


def experiment_2_toy(cfg_dict):
    print("\n==== Experiment 2 (toy) — Adaptive columns: compute & inference ====")
    seed_all(cfg_dict.get('seed', 123) + 1)
    from .preprocess import make_dataloaders
    loader_blobs, _, _ = make_dataloaders(batch_size=cfg_dict.get('batch_size', 8), size=cfg_dict.get('img_size', 32), n=128)

    timesteps = cfg_dict.get('timesteps', 60)
    steps_warm = max(10, cfg_dict.get('steps_toy', 20)//2)

    rev_gated = RevMoDiff(img_ch=3, base_ch=32, hidden_group=32, num_cols=3, num_blocks_per_col=2, groups=2,
                          timesteps=timesteps, text_dim=0, use_gates=True)
    device = get_device()
    rev_gated.to(device)

    opt = torch.optim.AdamW(rev_gated.parameters(), lr=2e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    diff = rev_gated.diff
    print("Training RevMoDiff (with gates) briefly to warm up...")
    rev_gated.train()
    step = 0
    for x0, _ in loader_blobs:
        if step >= steps_warm:
            break
        x0 = x0.to(device)
        b = x0.size(0)
        t = torch.randint(0, timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        x_t = diff.q_sample(x0, t, noise)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred, gates = rev_gated(x_t, t, None, return_gates=True)
            loss = torch.nn.functional.mse_loss(pred, noise) + 1e-3 * gates.mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (step % 5) == 0:
            print(f"[gated warmup] step={step:03d}, loss={loss.item():.4f}, mean_gate={gates.mean().item():.3f}")
        step += 1

    rev_gated.eval()
    shape = (4, 3, cfg_dict.get('img_size', 32), cfg_dict.get('img_size', 32))

    # Full-depth (gates off)
    rev_gated.use_gates = False
    m_full = measure_latency_and_memory(rev_gated, shape, steps=20, use_gates=False)
    print(f"Full-depth: latency={m_full['latency_s']:.3f}s, peak_mem(MiB)={m_full['peak_mem_mib'] if not math.isnan(m_full['peak_mem_mib']) else 'N/A'}")

    # Gated
    rev_gated.use_gates = True
    m_gated = measure_latency_and_memory(rev_gated, shape, steps=20, use_gates=True)
    print(f"Gated:      latency={m_gated['latency_s']:.3f}s, peak_mem(MiB)={m_gated['peak_mem_mib'] if not math.isnan(m_gated['peak_mem_mib']) else 'N/A'}")

    # Plots
    from .train import save_bar_plot
    save_bar_plot(['full', 'gated'], [m_full['latency_s'], m_gated['latency_s']], 'Inference latency (s)', 'seconds', os.path.join(FIG_DIR, 'inference_latency_gated_vs_full.pdf'))

    if rev_gated.gater is not None:
        plot_gating_usage(rev_gated, shape, steps=timesteps, filepath_pdf=os.path.join(FIG_DIR, 'column_usage_revmodeff.pdf'))

    # Sample a small grid with and without gates
    imgs_full = simple_sampler(rev_gated, shape=(8, 3, cfg_dict.get('img_size', 32), cfg_dict.get('img_size', 32)), steps=20, use_gates=False)
    save_image_grid_pdf(imgs_full, os.path.join(FIG_DIR, 'samples_full_depth.pdf'), nrow=4)
    imgs_gated = simple_sampler(rev_gated, shape=(8, 3, cfg_dict.get('img_size', 32), cfg_dict.get('img_size', 32)), steps=20, use_gates=True)
    save_image_grid_pdf(imgs_gated, os.path.join(FIG_DIR, 'samples_gated.pdf'), nrow=4)


def experiment_3_toy(cfg_dict):
    print("\n==== Experiment 3 (toy) — High-resolution feasibility harness ====")
    def builder():
        return RevMoDiff(img_ch=3, base_ch=32, hidden_group=32, num_cols=2, num_blocks_per_col=2, groups=4, timesteps=50, text_dim=0, use_gates=False)
    ok = can_train_resolution(builder, H=1024, W=1024, timesteps=50)
    print(f"1024x1024 feasibility: {'SUCCESS' if ok else 'SKIPPED/FAILED'}")


def reversibility_sanity_check():
    print("\n==== Reversibility sanity check ====")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = RevMoDiff(img_ch=3, base_ch=32, hidden_group=32, num_cols=1, num_blocks_per_col=1, groups=2, timesteps=10).to(device)
    model.eval()
    x = torch.randn(2, 3, 16, 16, device=device)
    t = torch.randint(0, 10, (2,), device=device)
    col = model.columns[0]
    cond = model.build_cond_vec(t, torch.zeros(2, device=device), torch.zeros(2, device=device), None)
    h = model.stem(x)
    y = col(h, cond)
    z = y.clone().detach()
    for b in reversed(col.blocks):
        z = b.module.inverse(z, cond)
    rel_err = (z - h).norm() / (h.norm() + 1e-8)
    print(f"Reconstruction relative error: {rel_err.item():.6e}")


if __name__ == '__main__':
    # Allow running this module directly for a quick smoke test
    cfg = load_config('config/experiment.yaml')
    reversibility_sanity_check()
    experiment_1_toy(cfg)
    experiment_2_toy(cfg)
    experiment_3_toy(cfg)
