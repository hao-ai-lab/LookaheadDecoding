from types import SimpleNamespace

import torch
from transformers import GenerationMixin

from lade.decoding import FUNC_MAP, greedy_search_proxy, sample_proxy
from lade.utils import augment_generate, inject_module


def test_injection_preserves_inherited_methods_constructors_and_descriptors():
    class Original(torch.nn.Module):
        def forward(self, value):
            return value + 1

    class Extension(torch.nn.Module):
        __module__ = "lade.tests"

        def __init__(self):
            raise AssertionError("extension constructor must not be transplanted")

        @staticmethod
        def static_helper(value):
            return value * 2

        @classmethod
        def class_helper(cls):
            return cls

        def extension(self):
            return 42

    inject_module(SimpleNamespace(Model=Extension), SimpleNamespace(Model=Original))
    instance = Original()
    assert instance(3) == 4
    assert instance.extension() == 42
    assert instance.static_helper(3) == 6
    assert Original.static_helper(3) == 6
    assert instance.class_helper() is Original


def test_augmentation_is_idempotent_and_disabled_sampling_uses_sampling(monkeypatch):
    monkeypatch.setattr(GenerationMixin, "greedy_search", lambda *a, **k: "greedy")
    monkeypatch.setattr(GenerationMixin, "sample", lambda *a, **k: "sample")
    saved = FUNC_MAP.copy()
    try:
        augment_generate()
        augment_generate()
        monkeypatch.setenv("USE_LADE", "0")
        assert greedy_search_proxy(None) == "greedy"
        assert sample_proxy(None) == "sample"
    finally:
        FUNC_MAP.clear()
        FUNC_MAP.update(saved)
