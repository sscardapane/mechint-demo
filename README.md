# mechint-demo

This repository demonstrates working with small language models. It now includes a
utility function to download and run models such as [Gemma 2B](https://huggingface.co/google/gemma-2b-it),
optionally loading sparse autoencoders (SAEs) when available.

## Running a model

```
from llm_runner import run_with_sae

text = run_with_sae("Hello", model_name="google/gemma-2b-it")
print(text)
```

Use the `sae_repo_id` argument to load an SAE from the Hugging Face hub and return
its encoded features alongside the generated text.
