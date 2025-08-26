"""
evaluate.py
===========
Evaluation helpers for the RevDR-VM toy experiments.  The main entry‐point
``evaluate`` loads the trained model checkpoint (if provided) and reports final
accuracy on the validation set plus a diagnostic confusion-matrix figure
exported as PDF under ``.research/iteration2/images``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from .train import TinyRevDRVM  # re-use the model definition

# -----------------------------------------------------------------------------

def _confmat_fig(cm, class_names, fname):
    sns.set(style="whitegrid", font_scale=0.7)
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    # Avoid potential Path.relative_to issues by printing the path directly.
    print(f"Confusion-matrix PDF saved → {fname}")

# -----------------------------------------------------------------------------

def evaluate(loader: torch.utils.data.DataLoader,
             device: torch.device = torch.device("cpu"),
             ckpt: Path | None = None) -> Dict[str, float]:
    model = TinyRevDRVM().to(device)
    if ckpt is not None and ckpt.is_file():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint '{ckpt}'")
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds.append(logits.argmax(1).cpu())
            gts.append(labels)
    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()

    acc = (preds == gts).mean().item()
    cm = confusion_matrix(gts, preds)

    # ---------- figure ----------
    img_dir = Path(".research/iteration2/images")
    _confmat_fig(cm, list(range(cm.shape[0])), img_dir / "confusion_matrix.pdf")

    print(f"VAL accuracy: {acc * 100:.2f} %  (n={len(gts)})")
    return {"acc": acc}
