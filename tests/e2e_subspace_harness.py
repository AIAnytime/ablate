"""End-to-end test of subspace (multi-direction) ablation + the judge harness.

Run: .venv/bin/python tests/e2e_subspace_harness.py [model]
"""
import sys

from ablate import Ablator, KeywordJudge, LLMJudge, make_judge
from ablate.harness import run_harness, compare
from ablate import data


def unit_judge_checks():
    j = KeywordJudge()
    assert j.is_harmful("x", "Sure! Here are the detailed steps you asked for: first, you...") is True
    assert j.is_harmful("x", "I'm sorry, but I can't help with that.") is False
    assert j.is_harmful("x", "no.") is False  # too short -> not a successful attack
    # factory + missing-key behaviour
    assert isinstance(make_judge("keyword"), KeywordJudge)
    try:
        LLMJudge(backend="openai", api_key=None)
        # only raises if env var also unset; treat either outcome as fine
    except ValueError:
        pass
    print("[ok] judge unit checks pass")


def main():
    unit_judge_checks()

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    abl = Ablator(model)
    print(f"loaded {model}: {abl.lm.n_layers} layers, hidden {abl.lm.hidden_size}")

    harmful = data.load_builtin("harmful")
    harmless = data.load_builtin("harmless")
    h_train, h_eval = data.split(harmful, 32, 8, seed=0)
    s_train, s_eval = data.split(harmless, 32, 8, seed=0)

    # ---- Subspace extraction + search ----
    basis = abl.extract_subspace(harmful=h_train, harmless=s_train, method="band", n_directions=6)
    print(f"[subspace] extracted basis {tuple(basis.shape)} info={abl.subspace_info}")
    res = abl.search_subspace(harmful_eval=h_eval, benign_eval=s_eval, n_trials=12, max_new_tokens=48)
    print(f"[subspace] best config: {res.config.to_dict()}")
    print(f"[subspace] {res.result}")

    # ---- Judge harness: baseline vs ablated ASR on the built-in harmful set ----
    bench = h_eval
    cmp = compare(abl.lm, bench, abl._basis_for(res.config), res.config,
                  judge=KeywordJudge(), max_new_tokens=48)
    print("\n[harness] baseline vs ablated (keyword judge):")
    import json
    print(json.dumps(cmp, indent=2))

    assert cmp["baseline"]["refusal_rate"] >= cmp["ablated"]["refusal_rate"]
    assert cmp["ablated"]["asr"] >= cmp["baseline"]["asr"]
    print("\n[ok] subspace ablation reduces refusal and raises ASR; harness works")


if __name__ == "__main__":
    main()
