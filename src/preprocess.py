"""src/preprocess.py
Placeholder for data-pre-processing.
In this minimal example we do not download real datasets to keep CI time low.
The module still exposes the public API that full experiments expect.
"""
from __future__ import annotations
from torch.utils.data import DataLoader

# In a future iteration this file would contain code that downloads and
# preprocesses SVHN/CIFAR/Audio/IMU datasets.  The training pipeline already
# falls back to synthetic data when these functions return `None`.

def get_task_loaders(tasks):
    """Returns a list of DataLoader objects, one per task.  For now `None` is
    returned so that `train.py` switches to synthetic loaders automatically.
    """
    return [None for _ in tasks]
