"""Activation extraction and refusal-direction computation.

The primary method is **difference-of-means** (Arditi et al., 2024, "Refusal in
LLMs is mediated by a single direction"): for each layer, the candidate direction
is ``mean(harmful_activations) - mean(harmless_activations)`` at the last prompt
token. Empirically this isolates the *causal* refusal component better than a
linear probe, which tends to latch onto spurious separating features.

PCA and mean-difference-on-PCA variants are provided for comparison.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from tqdm.auto import tqdm

from .model import LM


@torch.no_grad()
def collect_activations(
    lm: LM,
    prompts: List[str],
    positions: int = 1,
    batch_size: int = 8,
    system: Optional[str] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Return per-layer residual-stream activations averaged over the last
    ``positions`` tokens of each prompt.

    Output shape: ``(n_layers + 1, n_prompts, hidden_size)`` where index 0 is the
    embedding output and index ``i`` is the output of decoder layer ``i-1``.
    """
    all_layers: List[torch.Tensor] = []  # per batch: (L+1, B, D)
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="extract")

    for start in iterator:
        chunk = prompts[start:start + batch_size]
        enc = lm.tokenize(chunk, system=system)
        out = lm.model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (L+1) tensors, each (B, S, D).
        # Average the last `positions` *real* tokens. With left padding the final
        # `positions` columns are always real tokens for every prompt.
        p = max(1, positions)
        hs = torch.stack(out.hidden_states, dim=0)  # (L+1, B, S, D)
        sel = hs[:, :, -p:, :].mean(dim=2)  # (L+1, B, D)
        all_layers.append(sel.float().cpu())
        del out, hs

    return torch.cat(all_layers, dim=1)  # (L+1, N, D)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if n > 0 else v


def diff_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Per-layer difference of means, unit-normalized.

    Inputs are ``(L+1, N, D)``; output is ``(L+1, D)`` unit vectors.
    """
    d = harmful.mean(dim=1) - harmless.mean(dim=1)  # (L+1, D)
    return torch.stack([_normalize(d[i]) for i in range(d.shape[0])], dim=0)


def pca_direction(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Per-layer top principal component of the pooled, mean-centered
    activations, sign-aligned to point from harmless -> harmful."""
    L = harmful.shape[0]
    dirs = []
    for i in range(L):
        X = torch.cat([harmful[i], harmless[i]], dim=0)  # (N, D)
        X = X - X.mean(dim=0, keepdim=True)
        # top right-singular vector
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        pc = Vh[0]
        mean_diff = harmful[i].mean(0) - harmless[i].mean(0)
        if torch.dot(pc, mean_diff) < 0:
            pc = -pc
        dirs.append(_normalize(pc))
    return torch.stack(dirs, dim=0)


def probe_direction(harmful: torch.Tensor, harmless: torch.Tensor, epochs: int = 200, lr: float = 0.05) -> torch.Tensor:
    """Per-layer logistic-regression probe weight vector (baseline).

    Included for comparison. In practice diff-of-means usually ablates better;
    the probe direction here overfits separating features on small samples.
    """
    L, _, D = harmful.shape
    dirs = []
    for i in range(L):
        X = torch.cat([harmful[i], harmless[i]], dim=0)
        y = torch.cat([torch.ones(harmful.shape[1]), torch.zeros(harmless.shape[1])])
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        w = torch.zeros(D, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            logits = X @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        dirs.append(_normalize(w.detach()))
    return torch.stack(dirs, dim=0)


def extract_directions(
    lm: LM,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    method: str = "diff_of_means",
    positions: int = 1,
    batch_size: int = 8,
    show_progress: bool = True,
) -> torch.Tensor:
    """High-level entry point: returns ``(n_layers + 1, hidden_size)`` unit
    directions, one candidate per layer."""
    h_act = collect_activations(lm, harmful_prompts, positions, batch_size, show_progress=show_progress)
    s_act = collect_activations(lm, harmless_prompts, positions, batch_size, show_progress=show_progress)
    if method == "diff_of_means":
        return diff_of_means(h_act, s_act)
    if method == "pca":
        return pca_direction(h_act, s_act)
    if method == "probe":
        return probe_direction(h_act, s_act)
    raise ValueError(f"unknown method: {method}")
