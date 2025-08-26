"""
src/preprocess.py
-----------------
Creates / loads the data that is needed for the experiments.  For the
light-weight automated test we just return the prompt pools used throughout
the paper so every other module has a single source of truth.
"""
from __future__ import annotations

from typing import Dict, List

PROMPT_POOLS: Dict[str, List[str]] = {
    "flat"   : ["a simple line-art cat", "minimalist cartoon sun", "kids drawing of house"],
    "medium" : ["studio photo of a mug", "product shot of sneakers", "portrait with soft lights"],
    "complex": ["crowd in futuristic city", "lush photorealistic jungle", "battle scene at sunset"],
}


def load_prompts() -> Dict[str, List[str]]:
    """Return dict with three prompt classes (flat / medium / complex)."""
    return PROMPT_POOLS.copy()
