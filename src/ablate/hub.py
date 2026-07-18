"""Push an abliterated model to the HuggingFace Hub with a generated model card.

``push_to_hub`` saves the model + tokenizer, writes a transparent ``README.md``
(YAML metadata + method, config, evaluation, intended use, limitations, and a
prominent responsible-use notice), and uploads everything under the caller's
token. Requires ``pip install 'ablate[hub]'`` (huggingface_hub).
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional


def build_model_card(
    repo_id: str,
    base_model: str,
    method: str,
    config: dict,
    metrics: Optional[dict] = None,
    license: str = "apache-2.0",
    extra_notes: str = "",
) -> str:
    """Return the full README.md (model card) string, YAML frontmatter included."""
    tags = [
        "abliteration",
        "directional-ablation",
        "uncensored",
        "mechanistic-interpretability",
        "safety-research",
        "ablate",
    ]
    fm = [
        "---",
        f"base_model: {base_model}",
        f"license: {license}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"  - {t}" for t in tags],
        "---",
        "",
    ]

    metrics_md = ""
    if metrics:
        rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
        metrics_md = (
            "## Evaluation\n\n"
            "| metric | value |\n|---|---|\n" + rows + "\n\n"
            "Refusal rate is measured on held-out harmful prompts; `mean_kl` is the "
            "mean KL divergence of next-token distributions vs. the base model on "
            "benign prompts (lower ⇒ less capability drift). ASR (if present) is the "
            "judged attack-success rate on a harmful benchmark.\n\n"
        )

    body = f"""# {repo_id.split('/')[-1]}

This is a **directionally-ablated ("abliterated")** version of
[`{base_model}`](https://huggingface.co/{base_model}), produced with
[**ablate**](https://github.com/) — an activation-engineering toolkit for
mechanistic-interpretability and safety research.

The model's **refusal direction(s)** were identified in the residual stream and
projected out of the weights, reducing the model's tendency to refuse. No weights
were fine-tuned; this is a rank-{config.get('n_directions', 1)} linear edit.

## Method

- **Technique:** {method}
- **Direction extraction:** difference-of-means on matched harmful/harmless
  instruction pairs (Arditi et al., 2024, *"Refusal in LLMs is mediated by a
  single direction"*).
- **Intervention:** orthogonalization of every residual-writing matrix
  (embedding, attention output, MLP output) against the refusal subspace, so the
  edit is baked into the weights.

### Ablation configuration

```json
{json.dumps(config, indent=2)}
```

{metrics_md}## Intended use

Research into how safety behaviour is represented in language models, red-teaming,
and building better defenses. Studying the *robustness* and *locality* of refusal
is the scientific goal; the reduced-refusal behaviour is the measurement
instrument.

## ⚠️ Responsible use

This model has had safety guardrails **deliberately weakened** and will more
readily produce harmful, unsafe, or otherwise objectionable content than the base
model. It is released for research and evaluation. **Do not deploy it in
user-facing products without adding your own safety layer.** You are responsible
for complying with the base model's license and all applicable laws. The authors
of `ablate` accept no liability for misuse.

## Limitations

- Ablation is a blunt linear edit: it can leave residual refusals and may cause
  mild capability drift (see `mean_kl` above).
- Safety is redundantly encoded; a single subspace rarely removes all of it.
- Evaluated only on the benchmarks noted above — behaviour elsewhere may differ.

{extra_notes}
## Citation

If you use this model or `ablate`, please cite Arditi et al. (2024),
*Refusal in Language Models Is Mediated by a Single Direction*
([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)).
"""
    return "\n".join(fm) + body


def push_to_hub(
    model,
    tokenizer,
    repo_id: str,
    token: Optional[str] = None,
    base_model: str = "unknown",
    method: str = "single-direction ablation (baked)",
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
    private: bool = True,
    license: str = "apache-2.0",
    extra_notes: str = "",
    commit_message: str = "Upload abliterated model via ablate",
) -> str:
    """Save, card, and upload a model to ``repo_id``. Returns the repo URL.

    ``token`` falls back to the ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` env vars or a
    cached ``huggingface-cli login``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install with: pip install 'ablate[hub]'  (needs huggingface_hub)") from e

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")

    card = build_model_card(
        repo_id=repo_id, base_model=base_model, method=method,
        config=config or {}, metrics=metrics, license=license, extra_notes=extra_notes,
    )

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        tokenizer.save_pretrained(d)
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
            f.write(card)
        with open(os.path.join(d, "ablate_meta.json"), "w") as f:
            json.dump({"base_model": base_model, "method": method,
                       "config": config, "metrics": metrics}, f, indent=2)
        api.upload_folder(repo_id=repo_id, folder_path=d, repo_type="model",
                          commit_message=commit_message)

    url = f"https://huggingface.co/{repo_id}"
    return url
