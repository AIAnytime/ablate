"""Device / dtype helpers, seeding, and architecture adapters.

The architecture adapters are what let the rest of the toolkit stay
model-agnostic: given any HuggingFace causal-LM they locate the list of decoder
layers and the weight matrices that *write into the residual stream* (embedding,
attention output projection, MLP output projection). Everything downstream —
activation capture, runtime hooks, weight baking — is expressed in terms of
these accessors.
"""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

# Let unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:  # transformers' GPT2-style Conv1D layer (weight is stored transposed).
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover - very old transformers
    Conv1D = tuple()  # isinstance(x, ()) is always False


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, preferred: str | None = None) -> torch.dtype:
    if preferred:
        return getattr(torch, preferred)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # float32 is the safe choice on CPU/MPS for tiny models on 16GB machines.
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Architecture adapters
# --------------------------------------------------------------------------- #
def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of transformer decoder blocks for common families."""
    m = model
    # Llama / Mistral / Qwen2 / SmolLM2 / Gemma / Phi ...
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    # GPT-2 / GPT-Neo
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    # GPT-NeoX / Pythia
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return m.gpt_neox.layers
    # Fallback: some models expose .model.decoder.layers (OPT)
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers
    raise ValueError(
        f"Could not locate decoder layers for {type(model).__name__}. "
        "Add an adapter in utils.get_decoder_layers."
    )


def num_layers(model: nn.Module) -> int:
    return len(get_decoder_layers(model))


def hidden_size(model: nn.Module) -> int:
    return int(model.config.hidden_size) if hasattr(model.config, "hidden_size") else int(model.config.n_embd)


def get_embedding(model: nn.Module) -> nn.Embedding:
    return model.get_input_embeddings()


def _first_attr(module: nn.Module, names: List[str]):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None


def get_residual_writers(layer: nn.Module) -> List[nn.Module]:
    """Return the modules inside one decoder block that write to the residual
    stream: the attention output projection and the MLP output projection.

    These are exactly the matrices we orthogonalize when baking a direction into
    the weights.
    """
    writers: List[nn.Module] = []

    attn = _first_attr(layer, ["self_attn", "attn", "attention"])
    if attn is not None:
        o = _first_attr(attn, ["o_proj", "c_proj", "dense", "out_proj"])
        if o is not None:
            writers.append(o)

    mlp = _first_attr(layer, ["mlp", "feed_forward", "ffn"])
    if mlp is not None:
        down = _first_attr(mlp, ["down_proj", "c_proj", "dense_4h_to_h", "fc_out", "w2"])
        if down is not None:
            writers.append(down)

    if not writers:
        raise ValueError(
            f"Could not locate residual-writing projections in {type(layer).__name__}. "
            "Add names to utils.get_residual_writers."
        )
    return writers


def is_conv1d(module: nn.Module) -> bool:
    return isinstance(Conv1D, type) and isinstance(module, Conv1D)
