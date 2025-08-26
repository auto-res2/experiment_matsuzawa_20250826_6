"""src/main.py
Entry-point that orchestrates preprocessing, training and evaluation.
Run with
    python -m src.main               # full experiment (slow)
    python -m src.main --quick       # 30-second smoke test
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

from . import preprocess  # noqa
from .train import continual_train
from .evaluate import plot_accuracy_curve


def _run_experiment(quick: bool):
    tasks = ["SVHN", "CIFAR10", "CIFAR20", "TinyIN20"]
    print(f"Running experiment with tasks: {tasks} | quick={quick}")
    model, tokens, acc_hist = continual_train(tasks, quick=quick)
    plot_accuracy_curve(acc_hist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Run smoke test only")
    args = p.parse_args()

    t0 = time.time()
    _run_experiment(args.quick)
    dt = time.time() - t0
    print(f"Finished in {dt:.1f} s")


if __name__ == "__main__":
    main()
