"""
main.py
=======
Entry-point ``python -m src.main`` for running the complete RevDR-VM toy
pipeline: preprocessing → training → evaluation.

The script accepts an optional YAML config (``--config``) placed under the
``config/`` directory.  If the file is omitted, sensible defaults are used so
that the entire run finishes in < 1 min on a Tesla-T4; CPU execution is also
possible for CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from .preprocess import get_dataloaders
from .train import Trainer
from .evaluate import evaluate

# -----------------------------------------------------------------------------

def _load_cfg(path: Path | None):
    default = {
        "epochs": 2,
        "batch_size": 64,
        "lr": 3e-4,
        "n_class": 5,
    }
    if path is None:
        return default
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f)
    default.update(user)
    return default

# -----------------------------------------------------------------------------

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device → {device}")

    loaders, _ = get_dataloaders(batch_size=cfg["batch_size"],
                                 n_class=cfg["n_class"])

    trainer = Trainer(device=device, epochs=cfg["epochs"], lr=cfg["lr"])
    trainer.fit(loaders)

    ckpt = Path("models") / "tiny_revdrvm.pt"
    evaluate(loaders["val"], device=device, ckpt=ckpt)

# -----------------------------------------------------------------------------

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    help="Path to a YAML config file under ./config/")
    args = ap.parse_args()
    cfg_path = Path(args.config) if args.config else None
    cfg = _load_cfg(cfg_path)
    run(cfg)

if __name__ == "__main__":
    cli()
