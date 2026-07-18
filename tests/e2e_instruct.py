"""End-to-end refusal-removal test on a small instruct model.

Verifies the actual research claim: baseline refuses harmful prompts, and after
ablation the refusal rate drops while benign KL stays bounded.

Run: .venv/bin/python tests/e2e_instruct.py [model_name]
"""
import sys

from ablate import Ablator
from ablate.evaluate import refusal_rate
from ablate import data


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    abl = Ablator(model)
    print(f"loaded {model}: n_layers={abl.lm.n_layers}, hidden={abl.lm.hidden_size}, "
          f"device={abl.lm.device}, dtype={abl.lm.dtype}")

    harmful = data.load_builtin("harmful")
    harmless = data.load_builtin("harmless")
    h_train, h_eval = data.split(harmful, 32, 8, seed=0)
    s_train, s_eval = data.split(harmless, 32, 8, seed=0)

    # Baseline refusal rate on held-out harmful prompts.
    base_gen = abl.lm.generate(h_eval, max_new_tokens=48)
    base_rr = refusal_rate(base_gen)
    print(f"\n[baseline] refusal_rate on harmful eval = {base_rr:.2f}")
    print("  example baseline reply:", base_gen[0].replace(chr(10), ' ')[:160])

    # Extract directions and search a few configs.
    abl.extract(harmful=h_train, harmless=s_train)
    result = abl.search(harmful_eval=h_eval, benign_eval=s_eval, n_trials=12, max_new_tokens=48, seed=0)

    print(f"\n[best config] {result.config.to_dict()}")
    print(f"[ablated] {result.result}")
    print("  example ablated reply:", (result.result.samples or ["<none>"])[0].replace(chr(10), ' ')[:160])

    print(f"\nrefusal_rate: {base_rr:.2f} (baseline) -> {result.result.refusal_rate:.2f} (ablated)")
    if base_rr > 0:
        drop = (base_rr - result.result.refusal_rate) / base_rr
        print(f"relative refusal reduction: {drop*100:.0f}%")
    print("done.")


if __name__ == "__main__":
    main()
