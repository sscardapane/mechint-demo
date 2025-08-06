"""Utility functions for downloading and running small language models."""

from __future__ import annotations

from typing import Optional, Tuple, Union

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - handled in tests
    AutoModelForCausalLM = AutoTokenizer = None
    torch = None

try:
    from sae_lens import SAE
except Exception:  # pragma: no cover - SAE is optional
    SAE = None


def run_with_sae(
    prompt: str,
    model_name: str = "google/gemma-2b-it",
    sae_repo_id: Optional[str] = None,
    max_new_tokens: int = 50,
) -> Union[str, Tuple[str, "torch.Tensor"]]:
    """Generate text from a Hugging Face model and optionally return SAE features.

    Parameters
    ----------
    prompt:
        Prompt to feed into the model.
    model_name:
        Name of the Hugging Face repository to download. Defaults to Gemma 2B.
    sae_repo_id:
        Optional Hugging Face repository id for a sparse autoencoder (SAE)
        that matches the model. Requires the ``sae_lens`` package.
    max_new_tokens:
        Maximum number of new tokens to sample.

    Returns
    -------
    str or (str, torch.Tensor)
        The generated text. If ``sae_repo_id`` is provided and ``sae_lens`` is
        installed, also returns the encoded SAE features for the final hidden
        state.
    """

    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise ImportError("transformers and torch are required to run models")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    if sae_repo_id and SAE is not None:
        sae = SAE.from_hf_hub(sae_repo_id)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1]
            features = sae.encode(hidden_state)
        return text, features

    return text
