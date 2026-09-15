import os
import subprocess
import sys
import textwrap
import pytest


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_random_llama_matches_unpatched_transformers(backend):
    # A separate process allows collecting the reference before global monkey
    # patching, independently of pytest module order and earlier fixtures.
    script = textwrap.dedent("""
        import os
        import random
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        import lade

        device = os.environ.get("LADE_TEST_DEVICE", "cpu")
        references = []
        for seed in (1, 7, 42):
            torch.manual_seed(seed)
            config = LlamaConfig(vocab_size=32, hidden_size=64,
                intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2,
                eos_token_id=None, pad_token_id=31, attention_dropout=0.0)
            config._attn_implementation = os.environ["LADE_TEST_BACKEND"]
            model = LlamaForCausalLM(config).eval().to(device)
            prompt = torch.tensor([[1, 8, 5, 3, 2, 6]], device=device)
            expected = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=20)
            with torch.no_grad():
                expected_forward = model(prompt, output_attentions=True)
            references.append((config, model.state_dict(), expected, expected_forward.logits,
                               expected_forward.attentions))

        lade.augment_all()
        lade.augment_all()
        for config, weights, expected, expected_logits, expected_attentions in references:
            model = LlamaForCausalLM(config).eval().to(device)
            model.load_state_dict(weights)
            with torch.no_grad():
                actual_forward = model(prompt, output_attentions=True)
            torch.testing.assert_close(actual_forward.logits, expected_logits)
            for actual_attention, expected_attention in zip(actual_forward.attentions, expected_attentions):
                torch.testing.assert_close(actual_attention, expected_attention)
            for enabled in (0, 1):
                os.environ["USE_LADE"] = str(enabled)
                random.seed(0)
                lade.config_lade(LEVEL=3, WINDOW_SIZE=4, GUESS_SET_SIZE=8, POOL_FROM_PROMPT=1)
                actual = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=20)
                assert torch.equal(actual, expected), (enabled, actual, expected)
        print("unpatched reference matches augmented AR and lookahead: 3 seeds, device=" + device)
    """)
    result = subprocess.run([sys.executable, "-c", script], env={**os.environ, "LADE_TEST_BACKEND": backend},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
