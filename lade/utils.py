import os 
from transformers import GenerationMixin
from transformers.models.llama import modeling_llama 

from .decoding import greedy_search_proxy, sample_proxy, FUNC_MAP, CONFIG_MAP
from .models import llama
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch 

def config_lade(WINDOW_SIZE=None, LEVEL=None, DEBUG=None, GUESS_SET_SIZE=None, ALWAYS_FWD_ONE=None, SPLIT_FLAG=None):
    if WINDOW_SIZE != None:
        CONFIG_MAP["WINDOW_SIZE"] = WINDOW_SIZE
    if LEVEL != None:
        CONFIG_MAP["LEVEL"] = LEVEL
    if GUESS_SET_SIZE != None:
        CONFIG_MAP["GUESS_SET_SIZE"] = GUESS_SET_SIZE
    if ALWAYS_FWD_ONE != None:
        CONFIG_MAP["ALWAYS_FWD_ONE"] = ALWAYS_FWD_ONE
    if DEBUG != None:
        CONFIG_MAP["DEBUG"] = DEBUG
    if SPLIT_FLAG != None:
        CONFIG_MAP["SPLIT_FLAG"] = SPLIT_FLAG
    

def augment_llama():
    modeling_llama.LlamaForCausalLM.jforward_multilevel = llama.jforward_multilevel
    modeling_llama.LlamaModel.LlamaModeljforward = llama.LlamaModeljforward
    modeling_llama.LlamaModel.j_prepare_decoder_attention_mask = llama.j_prepare_decoder_attention_mask    

def augment_generate():
    FUNC_MAP["greedy_search"] = GenerationMixin.greedy_search
    GenerationMixin.greedy_search = greedy_search_proxy
    FUNC_MAP["sample"] = GenerationMixin.sample
    # consider renaming or combining greedy/sample proxy; but the only difference is in the Gumbel noise & temperature config
    GenerationMixin.sample = sample_proxy 

    
def augment_all():
    augment_llama()
    augment_generate()


def get_hf_model(model_path, quant, dtype, device, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
    model_config = AutoConfig.from_pretrained(model_path)
    assert quant is None or len(quant) == 0

    model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device, cache_dir=cache_dir if len(cache_dir) > 0 else None)
    model = model.eval()
    model.tokenizer = tokenizer
    
    return model, tokenizer

def get_model(model_path, quant, dtype, device, cache_dir, use_ds, native_offload = False):
    return get_hf_model(model_path, quant, dtype, device, cache_dir)