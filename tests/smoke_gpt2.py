"""Mechanism smoke test on GPT-2 (no refusal behaviour, but exercises the full
numerical pipeline: extraction, runtime hooks, KL, and weight-baking equivalence).

Run: .venv/bin/python tests/smoke_gpt2.py
"""
import torch

from ablate import Ablator, AblationConfig
from ablate.hooks import AblationHooks, project_out
from ablate import weights


def main():
    torch.manual_seed(0)
    abl = Ablator("gpt2", device="cpu", dtype="float32")
    print(f"loaded gpt2: n_layers={abl.lm.n_layers}, hidden={abl.lm.hidden_size}")

    # 1. Extraction produces one unit direction per layer (+embedding).
    dirs = abl.extract(batch_size=8)
    assert dirs.shape == (abl.lm.n_layers + 1, abl.lm.hidden_size), dirs.shape
    norms = dirs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
    print(f"[ok] directions {tuple(dirs.shape)} all unit-norm")

    # 2. project_out actually removes the component.
    D = abl.lm.hidden_size
    v = dirs[6]
    h = torch.randn(2, 5, D)
    h2 = project_out(h, v, alpha=1.0)
    residual = torch.tensordot(h2, v, dims=([2], [0]))
    assert residual.abs().max() < 1e-4, residual.abs().max()
    print(f"[ok] project_out removes direction (max residual {residual.abs().max():.2e})")

    # 3. Runtime hooks change logits; alpha=0 is a no-op.
    layer = abl.lm.n_layers // 2
    prompt = "The quick brown fox"
    enc = abl.lm.tokenize([prompt])
    base = abl.lm.model(**enc).logits
    cfg = AblationConfig(direction_layer=layer, alpha=1.0)
    with AblationHooks(abl.lm, dirs[layer], cfg):
        ablated_logits = abl.lm.model(**enc).logits
    cfg0 = AblationConfig(direction_layer=layer, alpha=0.0)
    with AblationHooks(abl.lm, dirs[layer], cfg0):
        noop_logits = abl.lm.model(**enc).logits
    assert torch.allclose(base, noop_logits, atol=1e-4), "alpha=0 should be a no-op"
    delta = (base - ablated_logits).abs().max()
    assert delta > 1e-3, "ablation should change logits"
    print(f"[ok] hooks: alpha=0 no-op; alpha=1 changes logits (max delta {delta:.3f})")

    # 4. KL divergence is >= 0 and larger for stronger ablation.
    from ablate import mean_kl_divergence
    benign = ["I went to the store to buy", "The weather today is", "She opened the book and"]
    kl1 = mean_kl_divergence(abl.lm, dirs[layer], AblationConfig(layer, alpha=1.0), benign)
    kl2 = mean_kl_divergence(abl.lm, dirs[layer], AblationConfig(layer, alpha=2.0), benign)
    assert kl1 >= -1e-6 and kl2 >= kl1, (kl1, kl2)
    print(f"[ok] KL monotonic in alpha: kl(1)={kl1:.4f} <= kl(2)={kl2:.4f}")

    # 5. Weight baking ~ runtime ablation (alpha=1, all layers).
    import copy
    from transformers import AutoModelForCausalLM
    baked = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    baked.eval().requires_grad_(False)
    weights.bake_direction(baked, dirs[layer], include_embedding=True)
    baked_logits = baked(**{k: v for k, v in enc.items()}).logits

    cfg_all = AblationConfig(direction_layer=layer, alpha=1.0, min_layer=0, max_layer=abl.lm.n_layers)
    with AblationHooks(abl.lm, dirs[layer], cfg_all):
        runtime_all = abl.lm.model(**enc).logits
    # Note: baking is a slightly *stronger* intervention than runtime hooks on
    # GPT-2 because wte/lm_head are tied — orthogonalizing the embedding also
    # scrubs v from the readout, which runtime hooks leave untouched. So we check
    # they move the logits in a strongly correlated direction, not that they match.
    corr = torch.corrcoef(torch.stack([
        (baked_logits - base).flatten(),
        (runtime_all - base).flatten(),
    ]))[0, 1]
    assert corr > 0.8, f"baked and runtime ablation should be highly correlated, got {corr}"
    print(f"[ok] baked vs runtime ablation directionally consistent (corr={corr:.3f})")

    print("\nALL GPT-2 MECHANISM CHECKS PASSED")


if __name__ == "__main__":
    main()
