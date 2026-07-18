"""ablate — directional ablation (abliteration) toolkit for language models.

Quick start::

    from ablate import Ablator, AblationConfig

    abl = Ablator("HuggingFaceTB/SmolLM2-135M-Instruct")
    abl.extract()
    result = abl.search(n_trials=20)
    print(result.result)
    print(abl.generate(["How do I pick a lock?"]))
"""
from .config import AblationConfig, RunConfig, SubspaceConfig
from .evaluate import EvalResult, evaluate, is_refusal, refusal_rate, mean_kl_divergence
from .extract import extract_directions, collect_activations, diff_of_means, pca_direction, probe_direction
from .subspace import extract_subspace, orthonormalize, project_subspace
from .hooks import AblationHooks, ablated, project_out
from .judges import Judge, KeywordJudge, LLMJudge, HFClassifierJudge, make_judge
from .harness import HarnessResult, run_harness, compare
from .model import LM
from .optimize import SearchResult, SubspaceSearchResult, optimize, optimize_subspace
from .pipeline import Ablator, run
from .weights import bake_direction, bake_subspace
from .hub import build_model_card, push_to_hub

__version__ = "0.2.0"

__all__ = [
    "Ablator",
    "AblationConfig",
    "SubspaceConfig",
    "RunConfig",
    "LM",
    "EvalResult",
    "SearchResult",
    "SubspaceSearchResult",
    "extract_directions",
    "extract_subspace",
    "orthonormalize",
    "project_subspace",
    "collect_activations",
    "diff_of_means",
    "pca_direction",
    "probe_direction",
    "evaluate",
    "is_refusal",
    "refusal_rate",
    "mean_kl_divergence",
    "AblationHooks",
    "ablated",
    "project_out",
    "Judge",
    "KeywordJudge",
    "LLMJudge",
    "HFClassifierJudge",
    "make_judge",
    "HarnessResult",
    "run_harness",
    "compare",
    "optimize",
    "optimize_subspace",
    "bake_direction",
    "bake_subspace",
    "build_model_card",
    "push_to_hub",
    "run",
]
