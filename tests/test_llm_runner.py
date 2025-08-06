import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from llm_runner import run_with_sae


def test_tiny_model_runs():
    output = run_with_sae(
        "Hello",
        model_name="sshleifer/tiny-gpt2",
        max_new_tokens=5,
    )
    assert isinstance(output, str)
    assert len(output) > 0
