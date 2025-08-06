# mechint-demo

This repository demonstrates working with small language models. It now includes a
utility function to download and run models such as [Gemma 2B](https://huggingface.co/google/gemma-2b-it),
optionally loading sparse autoencoders (SAEs) when available.

## Installation

Install the utilities directly from the repository. Required dependencies
(`torch`, `transformers`, and `accelerate`) will be installed automatically:

```
pip install git+https://github.com/sscardapane/mechint-demo.git
```

To enable optional SAE features, include the `sae` extra:

```
pip install "git+https://github.com/sscardapane/mechint-demo.git#egg=mechint-demo[sae]"
```

## Usage in Google Colab

Open the example notebook directly in Google Colab. The first cell installs
the package so the notebook runs even when imported on its own.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sscardapane/mechint-demo/blob/main/notebooks/usage.ipynb)

## Running a model

```
from llm_runner import run_with_sae

text = run_with_sae("Hello", model_name="google/gemma-2b-it")
print(text)
```

Use the `sae_repo_id` argument to load an SAE from the Hugging Face hub and return
its encoded features alongside the generated text.
