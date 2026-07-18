"""Multi-direction (subspace) ablation.

A single refusal direction is often insufficient: safety behaviour is
redundantly encoded across layers, heads, and multiple residual directions.
Here we extract an *orthonormal basis* ``Q`` of shape ``(k, D)`` spanning a
refusal **subspace**, and project the whole subspace out of the residual stream:

    h' = h - alpha * (h Qᵀ) Q          # removes the component of h in span(Q)

``k == 1`` recovers ordinary single-direction ablation.

Two extraction strategies:

* ``"band"`` — the diff-of-means direction from each of the ``k`` strongest
  layers, orthonormalized (order-preserving Gram-Schmidt). Directions from
  different layers live in the same residual space, so their span is a
  meaningful multi-layer refusal subspace. Truncating ``Q[:j]`` keeps the ``j``
  strongest.
* ``"pca"`` — the top-``k`` principal components of the pooled, mean-centered
  activations at one layer, sign-aligned to the harmful side. Components are
  variance-ordered, so truncation is principled.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .extract import collect_activations
from .model import LM


def as_basis(x: torch.Tensor) -> torch.Tensor:
    """Coerce a direction ``(D,)`` or basis ``(k, D)`` to a 2-D basis."""
    return x.unsqueeze(0) if x.dim() == 1 else x


def orthonormalize(vecs: torch.Tensor) -> torch.Tensor:
    """Order-preserving modified Gram-Schmidt.

    Input ``(k, D)``; output ``(r, D)`` with orthonormal rows, ``r <= k`` (near
    linearly-dependent rows are dropped). Row order is preserved so that
    ``basis[:j]`` keeps the ``j`` most important directions.
    """
    out: List[torch.Tensor] = []
    for v in vecs.float():
        w = v.clone()
        for u in out:
            w = w - (w @ u) * u
        n = w.norm()
        if n > 1e-6:
            out.append(w / n)
    if not out:
        return vecs[:0]
    return torch.stack(out, dim=0)


def project_subspace(h: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """``h - alpha * (h Qᵀ) Q`` for orthonormal ``Q`` ``(k, D)`` (or a ``(D,)`` vector)."""
    Q = as_basis(basis).to(dtype=h.dtype, device=h.device)
    coeffs = h @ Q.T            # (..., k)
    return h - alpha * (coeffs @ Q)


def extract_subspace(
    lm: LM,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    method: str = "band",
    n_directions: int = 4,
    layer: Optional[int] = None,
    positions: int = 1,
    batch_size: int = 8,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """Return ``(basis (k, D), info)``. ``basis`` rows are orthonormal and
    ordered by importance."""
    h_act = collect_activations(lm, harmful_prompts, positions, batch_size, show_progress=show_progress)
    s_act = collect_activations(lm, harmless_prompts, positions, batch_size, show_progress=show_progress)
    raw = h_act.mean(dim=1) - s_act.mean(dim=1)  # (L+1, D) per-layer diff-of-means
    norms = raw.norm(dim=-1)

    if method == "band":
        # Candidate layers exclude embedding (0) and the final layer.
        cand = list(range(1, raw.shape[0] - 1))
        cand.sort(key=lambda i: float(-norms[i]))
        chosen = cand[:n_directions]
        vecs = torch.stack([raw[i] for i in chosen], dim=0)  # ordered strongest-first
        basis = orthonormalize(vecs)
        return basis, {"method": "band", "layers": chosen, "n_directions": int(basis.shape[0])}

    if method == "pca":
        L = int(layer) if layer is not None else int(norms[1:-1].argmax().item()) + 1
        X = torch.cat([h_act[L], s_act[L]], dim=0)
        Xc = X - X.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        comps = Vh[:n_directions]  # (k, D), variance-ordered, already orthonormal
        dm = h_act[L].mean(0) - s_act[L].mean(0)
        comps = torch.stack([c if torch.dot(c, dm) >= 0 else -c for c in comps], dim=0)
        basis = orthonormalize(comps)
        return basis, {"method": "pca", "layer": L, "n_directions": int(basis.shape[0])}

    raise ValueError(f"unknown subspace method: {method}")
