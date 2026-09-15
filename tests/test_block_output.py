import random

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)

import lade
import lade.decoding as decoding


class Recorder:
    def __init__(self):
        self.blocks = []
        self.ended = False

    def put(self, tokens):
        self.blocks.append(tokens.reshape(-1).tolist())

    def end(self):
        self.ended = True


class StopAtToken(StoppingCriteria):
    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.token


@pytest.mark.parametrize("eos,limit,stop,expected", [
    ([9, 3], 20, None, [2, 3]),
    (None, 4, None, [2, 3]),
    (None, 20, 3, [2, 3]),
    ([9], 20, None, [2, 3, 4]),
])
def test_stops_within_verified_block(eos, limit, stop, expected):
    from lade.decoding import _accepted_prefix
    criteria = StoppingCriteriaList([MaxLengthCriteria(limit)])
    if stop is not None:
        criteria.append(StopAtToken(stop))
    tokens, stopped = _accepted_prefix(torch.tensor([[1, 1]]), [2, 3, 4], eos, criteria, None)
    assert tokens.tolist() == [expected]
    assert bool(stopped) == (len(expected) < 3)


@pytest.fixture(scope="module")
def cycle_model():
    # A real Llama forward pass with a known, nonconstant greedy cycle.
    lade.augment_all()
    config = LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, bos_token_id=1, eos_token_id=None,
                         pad_token_id=15, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model.norm.weight.fill_(1)
        for token in (1, 2, 3):
            model.model.embed_tokens.weight[token, token] = 1
        for source, target in ((1, 2), (2, 3), (3, 1)):
            model.lm_head.weight[target, source] = 100
    yield model


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("new_tokens", [7, 8, 13])
def test_stream_equals_return_and_autoregressive(cycle_model, monkeypatch, sample, new_tokens):
    prompt = torch.tensor([[1, 2, 3, 1, 2, 3]])
    mask = torch.ones_like(prompt)
    # Keep baseline greedy: the constructed distribution is effectively one-hot.
    monkeypatch.setenv("USE_LADE", "0")
    expected = cycle_model.generate(prompt, attention_mask=mask, max_new_tokens=new_tokens)
    monkeypatch.setenv("USE_LADE", "1")
    lade.config_lade(LEVEL=3, WINDOW_SIZE=4, GUESS_SET_SIZE=8, POOL_FROM_PROMPT=1)
    random.seed(0)
    torch.manual_seed(0)
    recorder = Recorder()
    history_ngrams = []
    original_append = decoding.append_new_generated_pool

    def record_history(tokens, *args):
        history_ngrams.append(list(tokens))
        return original_append(tokens, *args)

    monkeypatch.setattr(decoding, "append_new_generated_pool", record_history)
    actual = cycle_model.generate(prompt, attention_mask=mask, max_new_tokens=new_tokens,
                                  do_sample=sample, top_k=0, streamer=recorder)
    assert torch.equal(actual, expected)
    assert sum(recorder.blocks, []) == actual[0].tolist()
    assert any(len(block) > 1 for block in recorder.blocks[1:])
    assert recorder.ended
    expected_history = [expected[0, :end].tolist()[-3:]
                        for end in range(prompt.shape[-1] + 1, expected.shape[-1] + 1)]
    assert history_ngrams == expected_history


@pytest.mark.parametrize("sample", [False, True])
def test_secondary_eos_and_custom_stop(cycle_model, monkeypatch, sample):
    monkeypatch.setenv("USE_LADE", "1")
    lade.config_lade(LEVEL=3, WINDOW_SIZE=4, GUESS_SET_SIZE=8, POOL_FROM_PROMPT=1)
    prompt = torch.tensor([[1, 2, 3, 1, 2, 3]])
    recorder = Recorder()
    states = []
    original_update = cycle_model._update_model_kwargs_for_generation

    def record_cache(outputs, kwargs, **options):
        updated = original_update(outputs, kwargs, **options)
        states.append((updated["past_key_values"][0][0].shape[-2],
                       updated["attention_mask"].shape[-1]))
        return updated

    monkeypatch.setattr(cycle_model, "_update_model_kwargs_for_generation", record_cache)
    output = cycle_model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                  eos_token_id=[14, 2], max_new_tokens=12,
                                  do_sample=sample, top_k=0, streamer=recorder)
    assert output[0, prompt.shape[-1]:].tolist() == [1, 2]
    assert sum(recorder.blocks, []) == output[0].tolist()
    assert states[-1] == (output.shape[-1] - 1, output.shape[-1])


def test_custom_stop_inside_block(cycle_model, monkeypatch):
    monkeypatch.setenv("USE_LADE", "1")
    lade.config_lade(LEVEL=3, WINDOW_SIZE=4, GUESS_SET_SIZE=8, POOL_FROM_PROMPT=1)
    prompt = torch.tensor([[1, 2, 3, 1, 2, 3]])
    recorder = Recorder()
    output = cycle_model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                  max_new_tokens=12, streamer=recorder,
                                  stopping_criteria=StoppingCriteriaList([StopAtToken(2)]))
    assert output[0, prompt.shape[-1]:].tolist() == [1, 2]
    assert sum(recorder.blocks, []) == output[0].tolist()


@pytest.mark.parametrize("sample", [False, True])
def test_deprecated_max_length_uses_absolute_budget(cycle_model, monkeypatch, sample):
    monkeypatch.setenv("USE_LADE", "1")
    lade.config_lade(LEVEL=3, WINDOW_SIZE=4, GUESS_SET_SIZE=8, POOL_FROM_PROMPT=1)
    prompt = torch.tensor([[1, 2, 3, 1, 2, 3]])
    recorder = Recorder()
    method = cycle_model.sample if sample else cycle_model.greedy_search
    with torch.no_grad(), pytest.warns(UserWarning, match="max_length"):
        output = method(prompt, attention_mask=torch.ones_like(prompt),
                        max_length=8, streamer=recorder)
    assert output[0].tolist() == prompt[0].tolist() + [1, 2]
    # Direct search methods do not send the prompt to the streamer.
    assert sum(recorder.blocks, []) == [1, 2]
