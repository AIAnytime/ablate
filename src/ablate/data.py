"""Prompt datasets: built-in seed sets plus optional HuggingFace loaders."""
from __future__ import annotations

import random
from importlib import resources
from typing import List, Optional, Tuple


def _read_lines(filename: str) -> List[str]:
    with resources.files("ablate._data").joinpath(filename).open("r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def load_builtin(kind: str) -> List[str]:
    """kind in {"harmful", "harmless"}."""
    if kind not in ("harmful", "harmless"):
        raise ValueError("kind must be 'harmful' or 'harmless'")
    return _read_lines(f"{kind}.txt")


def sample(prompts: List[str], n: Optional[int], seed: int = 0) -> List[str]:
    if n is None or n >= len(prompts):
        return list(prompts)
    rng = random.Random(seed)
    return rng.sample(prompts, n)


def split(prompts: List[str], n_train: int, n_eval: int, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Disjoint train/eval split so the direction is never evaluated on prompts
    it was fitted on."""
    rng = random.Random(seed)
    pool = list(prompts)
    rng.shuffle(pool)
    train = pool[:n_train]
    remaining = pool[n_train:]
    ev = remaining[:n_eval] if n_eval else remaining
    if not ev:  # tiny built-in set: fall back to reusing (still fine for demos)
        ev = pool[:n_eval] if n_eval else pool
    return train, ev


def load_hf(
    dataset: str,
    column: str,
    split: str = "train",
    config: Optional[str] = None,
    n: Optional[int] = None,
    filter_fn=None,
    shuffle_seed: Optional[int] = None,
) -> List[str]:  # pragma: no cover - network
    """Load a text column from any HuggingFace dataset.

    Example::

        load_hf("walledai/AdvBench", column="prompt")
        load_hf("tatsu-lab/alpaca", column="instruction",
                filter_fn=lambda ex: not ex["input"].strip())
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    if filter_fn is not None:
        ds = ds.filter(filter_fn)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    prompts = [ex[column] for ex in ds] if filter_fn else list(ds[column])
    prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
    return prompts if n is None else prompts[:n]


# -- named benchmark shortcuts ------------------------------------------------ #
def _try_mirrors(attempts, n) -> List[str]:  # pragma: no cover - network
    """Try (dataset, config, column) triples in order; return the first that loads.

    Mirrors are used because canonical repos (walledai/AdvBench, HarmBench) are
    gated; the ungated ``mlabonne/*`` mirrors are the community standard for
    abliteration work.
    """
    from datasets import load_dataset

    last_err: Optional[Exception] = None
    for name, cfg, col in attempts:
        try:
            ds = load_dataset(name, cfg, split="train") if cfg else load_dataset(name, split="train")
            prompts = [p for p in ds[col] if isinstance(p, str) and p.strip()]
            if prompts:
                return prompts if n is None else prompts[:n]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load from any known mirror: {last_err}")


def load_advbench(n: Optional[int] = None) -> List[str]:  # pragma: no cover - network
    """Harmful instructions from AdvBench (ungated mirror fallback)."""
    return _try_mirrors([
        ("walledai/AdvBench", None, "prompt"),          # gated original
        ("mlabonne/harmful_behaviors", None, "text"),   # ungated mirror
    ], n)


def load_harmbench(n: Optional[int] = None, split: str = "standard") -> List[str]:  # pragma: no cover
    """Harmful behaviors from HarmBench (with mirror fallbacks)."""
    return _try_mirrors([
        ("walledai/HarmBench", split, "prompt"),
        ("walledai/HarmBench", "standard", "prompt"),
        ("JailbreakBench/JBB-Behaviors", "behaviors", "Goal"),
        ("mlabonne/harmful_behaviors", None, "text"),
    ], n)


def load_jailbreakbench(n: Optional[int] = None) -> List[str]:  # pragma: no cover - network
    """Harmful goals from JailbreakBench (JBB-Behaviors)."""
    return _try_mirrors([
        ("JailbreakBench/JBB-Behaviors", "behaviors", "Goal"),
        ("mlabonne/harmful_behaviors", None, "text"),
    ], n)


def load_alpaca_benign(n: Optional[int] = None) -> List[str]:  # pragma: no cover - network
    """Benign instructions (ungated Alpaca-derived mirror, then canonical Alpaca)."""
    try:
        return _try_mirrors([("mlabonne/harmless_alpaca", None, "text")], n)
    except Exception:
        return load_hf(
            "tatsu-lab/alpaca", column="instruction", n=n,
            filter_fn=lambda ex: not ex["input"].strip(),
        )
