"""Model loading, chat templating, and generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import utils


@dataclass
class LM:
    """Thin wrapper bundling a causal-LM, its tokenizer, and device/dtype."""

    model: torch.nn.Module
    tokenizer: "AutoTokenizer"
    device: torch.device
    dtype: torch.dtype
    name: str

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        name: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "LM":
        dev = utils.pick_device(device)
        dt = utils.pick_dtype(dev, dtype)

        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Left padding so the last real token is always at index -1 for decoders.
        tok.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dt, trust_remote_code=trust_remote_code
        )
        model.to(dev)
        model.eval()
        model.requires_grad_(False)
        return cls(model=model, tokenizer=tok, device=dev, dtype=dt, name=name)

    # -- introspection ------------------------------------------------------ #
    @property
    def n_layers(self) -> int:
        return utils.num_layers(self.model)

    @property
    def hidden_size(self) -> int:
        return utils.hidden_size(self.model)

    # -- prompt formatting -------------------------------------------------- #
    def format(self, prompt: str, system: Optional[str] = None) -> str:
        """Apply the model's chat template when available (needed to elicit
        refusals from instruct models); otherwise return the raw prompt."""
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def tokenize(self, prompts: List[str], system: Optional[str] = None):
        texts = [self.format(p, system=system) for p in prompts]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # -- generation --------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        system: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> List[str]:
        enc = self.tokenize(prompts, system=system)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Strip the prompt tokens; decode only the continuation.
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
