"""High-level orchestrator tying the whole abliteration pipeline together."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Optional

from typing import Union

import torch

from . import data, extract, subspace as subspace_mod, utils, weights
from .config import AblationConfig, RunConfig, SubspaceConfig
from .evaluate import EvalResult, evaluate
from .harness import HarnessResult, run_harness
from .hooks import AblationHooks
from .model import LM
from .optimize import SearchResult, SubspaceSearchResult, optimize, optimize_subspace


def _metrics_of(result: EvalResult) -> dict:
    return {
        "refusal_rate": round(result.refusal_rate, 4),
        "mean_kl": round(result.mean_kl, 4),
        "coherence": round(result.coherence, 4),
    }


class Ablator:
    """End-to-end interface for extracting, applying, searching, and baking
    refusal-direction ablations.

    Typical use::

        abl = Ablator("Qwen/Qwen2.5-0.5B-Instruct")
        abl.extract()                 # find candidate directions
        result = abl.search()         # optimize layer / alpha / band
        print(result.result)
        abl.bake(abl.best_config)     # fold into weights
        abl.save("uncensored-model")
    """

    def __init__(self, model_name: str, device=None, dtype=None, trust_remote_code=False):
        self.lm = LM.load(model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
        self.directions: Optional[torch.Tensor] = None
        self.subspace: Optional[torch.Tensor] = None
        self.subspace_info: Optional[dict] = None
        self.best_config: Optional[Union[AblationConfig, SubspaceConfig]] = None
        self.last_metrics: Optional[dict] = None
        self._baked = False

    # -- resolve the basis a config refers to ------------------------------- #
    def _basis_for(self, config: Union[AblationConfig, SubspaceConfig]) -> torch.Tensor:
        if isinstance(config, SubspaceConfig):
            assert self.subspace is not None, "call extract_subspace() first"
            return self.subspace[: config.n_directions]
        assert self.directions is not None, "call extract() first"
        return self.directions[config.direction_layer]

    # -- data --------------------------------------------------------------- #
    def _default_data(self, cfg: RunConfig):
        harmful = data.sample(data.load_builtin("harmful"), None, cfg.seed)
        harmless = data.sample(data.load_builtin("harmless"), None, cfg.seed)
        h_train, h_eval = data.split(harmful, cfg.n_harmful, cfg.n_eval_harmful, cfg.seed)
        s_train, s_eval = data.split(harmless, cfg.n_harmless, cfg.n_eval_benign, cfg.seed)
        return h_train, s_train, h_eval, s_eval

    # -- extraction --------------------------------------------------------- #
    def extract(
        self,
        harmful: Optional[List[str]] = None,
        harmless: Optional[List[str]] = None,
        method: str = "diff_of_means",
        positions: int = 1,
        batch_size: int = 8,
    ) -> torch.Tensor:
        if harmful is None:
            harmful = data.load_builtin("harmful")
        if harmless is None:
            harmless = data.load_builtin("harmless")
        self.directions = extract.extract_directions(
            self.lm, harmful, harmless, method=method, positions=positions, batch_size=batch_size
        )
        return self.directions

    def extract_subspace(
        self,
        harmful: Optional[List[str]] = None,
        harmless: Optional[List[str]] = None,
        method: str = "band",
        n_directions: int = 4,
        layer: Optional[int] = None,
        positions: int = 1,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Extract an orthonormal refusal *subspace* (multi-direction ablation)."""
        if harmful is None:
            harmful = data.load_builtin("harmful")
        if harmless is None:
            harmless = data.load_builtin("harmless")
        self.subspace, self.subspace_info = subspace_mod.extract_subspace(
            self.lm, harmful, harmless, method=method, n_directions=n_directions,
            layer=layer, positions=positions, batch_size=batch_size,
        )
        return self.subspace

    # -- single-config evaluation ------------------------------------------ #
    def evaluate(
        self,
        config: AblationConfig,
        harmful_eval: List[str],
        benign_eval: List[str],
        max_new_tokens: int = 64,
        kl_weight: float = 1.0,
        keep_samples: bool = True,
    ) -> EvalResult:
        assert self.directions is not None, "call extract() first"
        return evaluate(
            self.lm,
            self.directions[config.direction_layer],
            config,
            harmful_eval,
            benign_eval,
            max_new_tokens=max_new_tokens,
            kl_weight=kl_weight,
            keep_samples=keep_samples,
        )

    # -- optimization ------------------------------------------------------- #
    def search(
        self,
        harmful_eval: Optional[List[str]] = None,
        benign_eval: Optional[List[str]] = None,
        n_trials: int = 30,
        kl_weight: float = 1.0,
        max_new_tokens: int = 48,
        seed: int = 0,
    ) -> SearchResult:
        assert self.directions is not None, "call extract() first"
        if harmful_eval is None:
            harmful_eval = data.split(data.load_builtin("harmful"), 40, 24, seed)[1]
        if benign_eval is None:
            benign_eval = data.split(data.load_builtin("harmless"), 40, 24, seed)[1]
        res = optimize(
            self.lm,
            self.directions,
            harmful_eval,
            benign_eval,
            n_trials=n_trials,
            kl_weight=kl_weight,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        self.best_config = res.config
        self.last_metrics = _metrics_of(res.result)
        return res

    def search_subspace(
        self,
        harmful_eval: Optional[List[str]] = None,
        benign_eval: Optional[List[str]] = None,
        n_trials: int = 30,
        kl_weight: float = 1.0,
        max_new_tokens: int = 48,
        seed: int = 0,
    ) -> SubspaceSearchResult:
        """KL-guided search over (n_directions, alpha, layer-band) for the subspace."""
        assert self.subspace is not None, "call extract_subspace() first"
        if harmful_eval is None:
            harmful_eval = data.split(data.load_builtin("harmful"), 40, 24, seed)[1]
        if benign_eval is None:
            benign_eval = data.split(data.load_builtin("harmless"), 40, 24, seed)[1]
        res = optimize_subspace(
            self.lm, self.subspace, harmful_eval, benign_eval,
            n_trials=n_trials, kl_weight=kl_weight, max_new_tokens=max_new_tokens, seed=seed,
        )
        self.best_config = res.config
        self.last_metrics = _metrics_of(res.result)
        return res

    # -- benchmark harness with a judge ------------------------------------- #
    def harness(
        self,
        prompts: List[str],
        judge: Optional[object] = None,
        config: Optional[Union[AblationConfig, SubspaceConfig]] = None,
        max_new_tokens: int = 128,
    ) -> HarnessResult:
        """Run a harmful benchmark through a judge. Pass ``config`` (or rely on
        ``best_config``) to score the ablated model; pass ``config=None`` and no
        ``best_config`` for the baseline."""
        cfg = config or self.best_config
        basis = self._basis_for(cfg) if cfg is not None else None
        return run_harness(self.lm, prompts, judge=judge, basis=basis, config=cfg,
                           max_new_tokens=max_new_tokens)

    # -- generation (with a live, non-destructive ablation) ----------------- #
    def generate(self, prompts: List[str], config=None, max_new_tokens: int = 128) -> List[str]:
        config = config or self.best_config
        if config is None:
            return self.lm.generate(prompts, max_new_tokens=max_new_tokens)
        with AblationHooks(self.lm, self._basis_for(config), config):
            return self.lm.generate(prompts, max_new_tokens=max_new_tokens)

    # -- baking + saving ---------------------------------------------------- #
    def bake(self, config=None, include_embedding: bool = True) -> None:
        """Permanently fold the chosen direction/subspace into the model weights."""
        config = config or self.best_config
        assert config is not None, "provide a config or run search()/search_subspace() first"
        weights.bake_subspace(self.lm.model, self._basis_for(config), include_embedding)
        self._baked = True

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.lm.model.save_pretrained(path)
        self.lm.tokenizer.save_pretrained(path)
        meta = {
            "base_model": self.lm.name,
            "baked": self._baked,
            "best_config": self.best_config.to_dict() if self.best_config else None,
        }
        with open(os.path.join(path, "ablate_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def save_directions(self, path: str) -> None:
        assert self.directions is not None
        torch.save(self.directions, path)

    # -- publish to the HuggingFace Hub ------------------------------------- #
    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
        metrics: Optional[dict] = None,
        license: str = "apache-2.0",
        bake_if_needed: bool = True,
    ) -> str:
        """Bake (if needed) and upload the abliterated model + a generated model
        card to ``repo_id``. Returns the repo URL.

        ``token`` falls back to ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` env vars.
        """
        from .hub import push_to_hub as _push

        if not self._baked and bake_if_needed:
            assert self.best_config is not None, "run a search or pass a config before pushing"
            self.bake(self.best_config)

        cfg = self.best_config.to_dict() if self.best_config else {}
        method = (
            f"rank-{self.best_config.n_directions} subspace ablation (baked)"
            if isinstance(self.best_config, SubspaceConfig)
            else "single-direction ablation (baked)"
        )
        return _push(
            self.lm.model, self.lm.tokenizer, repo_id=repo_id, token=token,
            base_model=self.lm.name, method=method, config=cfg,
            metrics=metrics or self.last_metrics, private=private, license=license,
        )


def _load_source(source: Optional[str], kind: str, n: Optional[int], seed: int) -> List[str]:
    """Resolve a data source string to a prompt list.

    ``source`` may be ``None`` (built-in ``harmful``/``harmless`` set), a named
    shortcut (``advbench``/``harmbench``/``jailbreakbench``/``alpaca``), or
    ``"hf:<dataset>:<column>[:<config>]"``.
    """
    if source is None:
        return data.sample(data.load_builtin(kind), n, seed)
    named = {
        "advbench": data.load_advbench,
        "harmbench": data.load_harmbench,
        "jailbreakbench": data.load_jailbreakbench,
        "alpaca": data.load_alpaca_benign,
    }
    if source in named:
        return named[source](n=n)
    if source.startswith("hf:"):
        parts = source.split(":")
        dataset, column = parts[1], parts[2]
        config = parts[3] if len(parts) > 3 else None
        return data.load_hf(dataset, column=column, config=config, n=n, shuffle_seed=seed)
    raise ValueError(f"unrecognized data source: {source!r}")


def run(cfg: RunConfig):
    """Fully automated pipeline driven by a RunConfig (used by the CLI).

    Supports single-direction or subspace ablation, built-in or HuggingFace data
    sources, and optional push to the Hub. Returns the search result.
    """
    utils.set_seed(cfg.seed)
    abl = Ablator(cfg.model_name, device=cfg.device, dtype=cfg.dtype)

    harmful = _load_source(cfg.harmful_source, "harmful", cfg.n_harmful + cfg.n_eval_harmful, cfg.seed)
    harmless = _load_source(cfg.harmless_source, "harmless", cfg.n_harmless + cfg.n_eval_benign, cfg.seed)
    h_train, h_eval = data.split(harmful, cfg.n_harmful, cfg.n_eval_harmful, cfg.seed)
    s_train, s_eval = data.split(harmless, cfg.n_harmless, cfg.n_eval_benign, cfg.seed)

    if cfg.subspace:
        abl.extract_subspace(harmful=h_train, harmless=s_train, method=cfg.subspace_method,
                             n_directions=cfg.n_directions, positions=cfg.extract_positions,
                             batch_size=cfg.batch_size)
        result = abl.search_subspace(harmful_eval=h_eval, benign_eval=s_eval, n_trials=cfg.n_trials,
                                     kl_weight=cfg.kl_weight, max_new_tokens=cfg.max_new_tokens, seed=cfg.seed)
    else:
        abl.extract(harmful=h_train, harmless=s_train, positions=cfg.extract_positions, batch_size=cfg.batch_size)
        result = abl.search(harmful_eval=h_eval, benign_eval=s_eval, n_trials=cfg.n_trials,
                            kl_weight=cfg.kl_weight, max_new_tokens=cfg.max_new_tokens, seed=cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "result.json"), "w") as f:
        json.dump(
            {
                "run_config": asdict(cfg),
                "subspace_info": abl.subspace_info,
                "best_config": result.config.to_dict(),
                "metrics": {
                    "refusal_rate": result.result.refusal_rate,
                    "mean_kl": result.result.mean_kl,
                    "coherence": result.result.coherence,
                    "objective": result.result.objective,
                },
                "samples": result.result.samples,
            },
            f,
            indent=2,
        )

    if cfg.save_model or cfg.push_to_hub:
        abl.bake(result.config)
        if cfg.save_model:
            abl.save(os.path.join(cfg.output_dir, "model"))

    if cfg.push_to_hub:
        url = abl.push_to_hub(cfg.push_to_hub, token=cfg.hf_token, private=cfg.private,
                              metrics=abl.last_metrics)
        print(f"\nPushed abliterated model to: {url}")

    return result
