"""
src/main.py
-----------
Command-line entry-point that wires together preprocessing, training and
evaluation so that the whole pipeline can be executed with
    python -m src.main
as required by the task specification.
"""
from __future__ import annotations

import argparse, sys

# Always use **relative** imports inside the src package
from . import preprocess as _pre
from . import train      as _train
from . import evaluate   as _eval


def main() -> None:
    parser = argparse.ArgumentParser(description="ATaS-Diff research runner")
    parser.add_argument("--stage", choices=["all","prep","train","eval","test"], default="all",
                        help="which part of the pipeline to execute")
    parser.add_argument("--exp",   choices=["1","2","3","all"], default="all",
                        help="which evaluation experiment to run (only with --stage eval)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 1. Pre-processing (idempotent, virtually cost-free for our synthetic
    #    example but keeps the interface future-proof).
    # ---------------------------------------------------------------------
    if args.stage in {"all", "prep"}:
        prompts = _pre.load_prompts()
        print(f"[main] loaded {sum(len(v) for v in prompts.values())} prompts.")
        if args.stage == "prep":
            return

    # ---------------------------------------------------------------------
    # 2. Training (can be skipped if weights are already saved).
    # ---------------------------------------------------------------------
    if args.stage in {"all", "train"}:
        _train.train()
        if args.stage == "train":
            return

    # ---------------------------------------------------------------------
    # 3. Evaluation / Experiments -----------------------------------------
    # ---------------------------------------------------------------------
    if args.stage in {"all", "eval"}:
        if args.exp in {"1","all"}: _eval.run_experiment_1()
        if args.exp in {"2","all"}: _eval.run_experiment_2()
        if args.exp in {"3","all"}: _eval.run_experiment_3()
    elif args.stage == "test":
        _eval.fast_sanity_test()


if __name__ == "__main__":
    # `python -m src.main` triggers this path.
    # Make sure the package root is on sys.path (needed when cwd==project root).
    if "" not in sys.path:  # pragma: no cover – safety-net for some envs
        sys.path.insert(0, "")
    main()
