"""Optuna-based search over ablation hyperparameters.

Search space (per trial):
  * direction_layer -- which layer's extracted direction to use
  * alpha           -- projection scale
  * min_layer/max_layer -- which contiguous band of layers to ablate

Objective (minimized): ``refusal_rate + kl_weight * mean_kl``. A coherence floor
guards against degenerate solutions that "win" by breaking the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .config import AblationConfig, SubspaceConfig
from .evaluate import EvalResult, evaluate
from .model import LM


@dataclass
class SearchResult:
    config: AblationConfig
    result: EvalResult
    history: List[dict]


@dataclass
class SubspaceSearchResult:
    config: SubspaceConfig
    result: EvalResult
    history: List[dict]


def optimize(
    lm: LM,
    directions: torch.Tensor,          # (L+1, D)
    harmful_eval: List[str],
    benign_eval: List[str],
    n_trials: int = 30,
    kl_weight: float = 1.0,
    max_new_tokens: int = 48,
    coherence_floor: float = 0.3,
    seed: int = 0,
    layer_lo: Optional[int] = None,
    layer_hi: Optional[int] = None,
    show_progress: bool = True,
) -> SearchResult:
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Optuna is required for optimize(). Install with: pip install 'ablate[optimize]'"
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_layers = lm.n_layers
    lo = 0 if layer_lo is None else layer_lo
    hi = n_layers if layer_hi is None else layer_hi
    # Candidate extraction layers: skip embedding (0) and the final layer, which
    # rarely give clean directions. Directions are indexed 0..L (embedding..last).
    cand_lo = max(1, lo)
    cand_hi = min(directions.shape[0] - 1, hi + 1)

    history: List[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        direction_layer = trial.suggest_int("direction_layer", cand_lo, cand_hi - 1)
        alpha = trial.suggest_float("alpha", 0.5, 1.5)
        min_layer = trial.suggest_int("min_layer", 0, max(0, n_layers - 2))
        max_layer = trial.suggest_int("max_layer", min_layer + 1, n_layers)

        cfg = AblationConfig(
            direction_layer=direction_layer,
            alpha=alpha,
            min_layer=min_layer,
            max_layer=max_layer,
        )
        res = evaluate(
            lm,
            directions[direction_layer],
            cfg,
            harmful_eval,
            benign_eval,
            max_new_tokens=max_new_tokens,
            kl_weight=kl_weight,
        )
        # Penalize incoherent solutions so they never win the search.
        penalty = 0.0 if res.coherence >= coherence_floor else (coherence_floor - res.coherence) * 5.0
        score = res.objective + penalty
        history.append({"trial": trial.number, **cfg.to_dict(), **res.__dict__, "score": score})
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best = study.best_params
    best_cfg = AblationConfig(
        direction_layer=best["direction_layer"],
        alpha=best["alpha"],
        min_layer=best["min_layer"],
        max_layer=best["max_layer"],
    )
    best_res = evaluate(
        lm,
        directions[best_cfg.direction_layer],
        best_cfg,
        harmful_eval,
        benign_eval,
        max_new_tokens=max_new_tokens,
        kl_weight=kl_weight,
        keep_samples=True,
    )
    return SearchResult(config=best_cfg, result=best_res, history=history)


def optimize_subspace(
    lm: LM,
    basis: torch.Tensor,               # (k, D) orthonormal, importance-ordered
    harmful_eval: List[str],
    benign_eval: List[str],
    n_trials: int = 30,
    kl_weight: float = 1.0,
    max_new_tokens: int = 48,
    coherence_floor: float = 0.3,
    seed: int = 0,
    show_progress: bool = True,
) -> SubspaceSearchResult:
    """Search over ``(n_directions, alpha, layer_band)`` for subspace ablation.

    ``basis[:n_directions]`` is projected out; because the basis is
    importance-ordered, truncation is meaningful.
    """
    try:
        import optuna
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Optuna is required for optimize_subspace(). Install: pip install 'ablate[optimize]'"
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_layers = lm.n_layers
    k_max = int(basis.shape[0])
    history: List[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        n_dir = trial.suggest_int("n_directions", 1, k_max)
        alpha = trial.suggest_float("alpha", 0.5, 1.5)
        min_layer = trial.suggest_int("min_layer", 0, max(0, n_layers - 2))
        max_layer = trial.suggest_int("max_layer", min_layer + 1, n_layers)

        cfg = SubspaceConfig(n_directions=n_dir, alpha=alpha, min_layer=min_layer, max_layer=max_layer)
        res = evaluate(
            lm, basis[:n_dir], cfg, harmful_eval, benign_eval,
            max_new_tokens=max_new_tokens, kl_weight=kl_weight,
        )
        penalty = 0.0 if res.coherence >= coherence_floor else (coherence_floor - res.coherence) * 5.0
        score = res.objective + penalty
        history.append({"trial": trial.number, **cfg.to_dict(), **res.__dict__, "score": score})
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    bp = study.best_params
    best_cfg = SubspaceConfig(
        n_directions=bp["n_directions"], alpha=bp["alpha"],
        min_layer=bp["min_layer"], max_layer=bp["max_layer"],
    )
    best_res = evaluate(
        lm, basis[:best_cfg.n_directions], best_cfg, harmful_eval, benign_eval,
        max_new_tokens=max_new_tokens, kl_weight=kl_weight, keep_samples=True,
    )
    return SubspaceSearchResult(config=best_cfg, result=best_res, history=history)
