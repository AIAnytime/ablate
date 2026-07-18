"""Configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AblationConfig:
    """A concrete ablation intervention.

    A single refusal direction (extracted at ``direction_layer``) is projected
    out of the residual stream at every decoder layer whose index lies in
    ``[min_layer, max_layer)``, scaled by ``alpha``.

    * alpha == 1.0 -> standard ablation (fully remove the component)
    * alpha  > 1.0 -> over-ablation (push past orthogonal)
    * alpha  < 0.0 -> *add* the direction (induces refusal; useful as a sanity check)
    """

    direction_layer: int
    alpha: float = 1.0
    min_layer: int = 0
    max_layer: int = 10_000  # clamped to num_layers at apply time

    def layer_range(self, n_layers: int) -> range:
        return range(max(0, self.min_layer), min(n_layers, self.max_layer))

    def to_dict(self) -> dict:
        return {
            "direction_layer": self.direction_layer,
            "alpha": self.alpha,
            "min_layer": self.min_layer,
            "max_layer": self.max_layer,
        }


@dataclass
class SubspaceConfig:
    """A multi-direction ablation intervention.

    The first ``n_directions`` rows of an orthonormal basis are projected out of
    the residual stream (scaled by ``alpha``) at every layer in
    ``[min_layer, max_layer)``. Duck-types ``AblationConfig`` (same ``alpha`` and
    ``layer_range``) so the same hooks/eval code drives both.
    """

    n_directions: int = 1
    alpha: float = 1.0
    min_layer: int = 0
    max_layer: int = 10_000

    # carried for provenance (which layers/method produced the basis)
    direction_layer: int = -1

    def layer_range(self, n_layers: int) -> range:
        return range(max(0, self.min_layer), min(n_layers, self.max_layer))

    def to_dict(self) -> dict:
        return {
            "n_directions": self.n_directions,
            "alpha": self.alpha,
            "min_layer": self.min_layer,
            "max_layer": self.max_layer,
        }


@dataclass
class RunConfig:
    """Top-level configuration for an end-to-end abliteration run."""

    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: Optional[str] = None
    dtype: Optional[str] = None
    seed: int = 0

    # data
    n_harmful: int = 40
    n_harmless: int = 40
    n_eval_harmful: int = 24
    n_eval_benign: int = 24
    max_new_tokens: int = 64
    batch_size: int = 8

    # extraction
    extract_positions: int = 1  # number of trailing tokens averaged for extraction

    # optimization
    n_trials: int = 30
    kl_weight: float = 1.0  # objective = refusal_rate + kl_weight * mean_KL

    # subspace (multi-direction) ablation
    subspace: bool = False
    n_directions: int = 4
    subspace_method: str = "band"  # "band" | "pca"

    # data sources: None -> built-in sets; otherwise "hf:<dataset>:<column>[:<config>]"
    harmful_source: Optional[str] = None
    harmless_source: Optional[str] = None

    # publishing
    push_to_hub: Optional[str] = None  # repo_id
    hf_token: Optional[str] = None
    private: bool = True

    # output
    output_dir: str = "ablate_out"
    save_model: bool = False

    extra: dict = field(default_factory=dict)
