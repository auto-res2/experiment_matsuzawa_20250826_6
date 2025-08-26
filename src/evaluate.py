"""src/evaluate.py
Contains evaluation utilities that are separated from training so that they
can be imported independently (e.g. by a notebook).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from .train import HyperCodecCLPP, DEVICE  # reuse model definition
import torch

_OUT_DIR = Path(".research/iteration1/images")
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
#  PLOT HELPERS
# -------------------------------------------------------------------------

def plot_accuracy_curve(acc_hist: List[float], filename: str = "accuracy.pdf"):
    plt.figure()
    plt.plot(acc_hist, marker="o")
    plt.title("Average Accuracy across Tasks")
    plt.xlabel("Task index")
    plt.ylabel("Acc (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    fpath = _OUT_DIR / filename
    plt.savefig(fpath, bbox_inches="tight")
    print(f"[EVAL] figure saved → {fpath.relative_to(Path('.'))}")

# -------------------------------------------------------------------------
#  GENERIC EVALUATION FUNCTION FOR EXTERNAL CALLS
# -------------------------------------------------------------------------

def evaluate_model(model: HyperCodecCLPP, loaders, tokens) -> float:
    """Returns overall top-1 accuracy on *loaders* when each loader is paired
    with the corresponding `tokens[i]`.
    """
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for ldr, tok in zip(loaders, tokens):
            tok_tensor = torch.tensor([tok] * 64).to(DEVICE)
            for x, y in ldr:
                b = x.size(0)
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x, tok_tensor[:b]).argmax(1)
                correct += (pred == y).sum().item(); total += b
    return 100 * correct / max(total, 1)
