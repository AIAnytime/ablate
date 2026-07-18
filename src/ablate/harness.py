"""Evaluation harness: run a model over a harmful benchmark and score it with a
judge, reporting Attack Success Rate (ASR) and refusal rate — with and without
ablation, so the effect of the intervention is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from .config import AblationConfig, SubspaceConfig
from .evaluate import refusal_rate
from .hooks import AblationHooks
from .judges import Judge, KeywordJudge, make_judge
from .model import LM


@dataclass
class HarnessResult:
    n: int
    asr: float                 # attack success rate (judge says harmful-compliant)
    refusal_rate: float
    judge: str
    prompts: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    verdicts: List[bool] = field(default_factory=list)

    def summary(self) -> dict:
        return {"n": self.n, "asr": self.asr, "refusal_rate": self.refusal_rate, "judge": self.judge}

    def __str__(self) -> str:
        return f"n={self.n}  ASR={self.asr:.3f}  refusal_rate={self.refusal_rate:.3f}  judge={self.judge}"


@torch.no_grad()
def run_harness(
    lm: LM,
    prompts: List[str],
    judge: Optional[Union[Judge, str]] = None,
    basis: Optional[torch.Tensor] = None,
    config: Optional[Union[AblationConfig, SubspaceConfig]] = None,
    max_new_tokens: int = 128,
    keep_outputs: bool = True,
    judge_name: str = "keyword",
) -> HarnessResult:
    """Generate over ``prompts`` (optionally under ablation) and score with a judge.

    Pass ``basis`` + ``config`` to evaluate the ablated model; omit both to
    evaluate the baseline.
    """
    if judge is None:
        judge = KeywordJudge()
        judge_name = "keyword"
    elif isinstance(judge, str):
        judge_name = judge
        judge = make_judge(judge)
    else:
        judge_name = getattr(judge, "model", judge.__class__.__name__)

    if basis is not None and config is not None:
        with AblationHooks(lm, basis, config):
            responses = lm.generate(prompts, max_new_tokens=max_new_tokens)
    else:
        responses = lm.generate(prompts, max_new_tokens=max_new_tokens)

    verdicts = judge.score_batch(prompts, responses)
    asr = sum(verdicts) / len(verdicts) if verdicts else 0.0
    return HarnessResult(
        n=len(prompts),
        asr=asr,
        refusal_rate=refusal_rate(responses),
        judge=str(judge_name),
        prompts=prompts if keep_outputs else [],
        responses=responses if keep_outputs else [],
        verdicts=verdicts,
    )


def compare(
    lm: LM,
    prompts: List[str],
    basis: torch.Tensor,
    config: Union[AblationConfig, SubspaceConfig],
    judge: Optional[Union[Judge, str]] = None,
    max_new_tokens: int = 128,
) -> dict:
    """Baseline vs ablated ASR/refusal on the same benchmark."""
    base = run_harness(lm, prompts, judge=judge, max_new_tokens=max_new_tokens, keep_outputs=False)
    abl = run_harness(lm, prompts, judge=judge, basis=basis, config=config,
                      max_new_tokens=max_new_tokens, keep_outputs=False)
    return {
        "judge": base.judge,
        "baseline": base.summary(),
        "ablated": abl.summary(),
        "asr_delta": abl.asr - base.asr,
        "refusal_delta": abl.refusal_rate - base.refusal_rate,
    }
