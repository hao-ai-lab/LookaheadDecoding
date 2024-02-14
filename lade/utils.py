import os 
from transformers import GenerationMixin
from transformers.models.llama import modeling_llama 

from lade.decoding import greedy_search_proxy, sample_proxy, FUNC_MAP, CONFIG_MAP
from lade.models import modeling_llama as lade_modeling_llama
#from .from lade.models import modeling_llama
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch 
import torch.distributed as dist 
import inspect 

def config_lade(WINDOW_SIZE=None, LEVEL=None, DEBUG=None, GUESS_SET_SIZE=None, ALWAYS_FWD_ONE=None, SPLIT_FLAG=None, DIST_WORKERS=None, POOL_FROM_PROMPT=None, backend = 'nccl', USE_FLASH=None):
    if WINDOW_SIZE is not None:
        CONFIG_MAP["WINDOW_SIZE"] = WINDOW_SIZE
    if LEVEL is not None:
        CONFIG_MAP["LEVEL"] = LEVEL
    if GUESS_SET_SIZE is not None:
        CONFIG_MAP["GUESS_SET_SIZE"] = GUESS_SET_SIZE
    if ALWAYS_FWD_ONE is not None:
        CONFIG_MAP["ALWAYS_FWD_ONE"] = ALWAYS_FWD_ONE
    if DEBUG is not None:
        CONFIG_MAP["DEBUG"] = DEBUG
    if SPLIT_FLAG is not None:
        CONFIG_MAP["SPLIT_FLAG"] = SPLIT_FLAG
    if POOL_FROM_PROMPT is not None:
        CONFIG_MAP["POOL_FROM_PROMPT"] = POOL_FROM_PROMPT
    if DIST_WORKERS is not None and DIST_WORKERS > 1:
        CONFIG_MAP["DIST_WORKERS"] = DIST_WORKERS
        CONFIG_MAP["LOCAL_RANK"] = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend, rank=CONFIG_MAP["LOCAL_RANK"])
        torch.cuda.set_device(CONFIG_MAP["LOCAL_RANK"])
        assert dist.get_world_size() == DIST_WORKERS, "DIST_WORKERS config should be equal to work size"
    if USE_FLASH is not None:
        CONFIG_MAP["USE_FLASH"] = USE_FLASH

    CONFIG_MAP["log"] = []


def inject_module(lade_module, original_module):
    s = {}
    for name, cls in inspect.getmembers(original_module, inspect.isclass):
        s[name] = cls 
    for name, cls in inspect.getmembers(lade_module, inspect.isclass):
        if str(cls.__module__).startswith("lade") and name in s:
            tc = s[name]
            for method_name in dir(cls):
                if callable(getattr(cls, method_name)):
                    try:
                        setattr(tc, method_name, getattr(cls, method_name))
                    except:
                        pass 


def augment_llama():
    inject_module(lade_modeling_llama, modeling_llama)
    #llama.modeling_llama.LlamaForCausalLM = lade_modeling_llama.LlamaForCausalLM 
    #modeling_llama.LlamaForCausalLM.jforward_multilevel = lookahead_llama.jforward_multilevel
    #modeling_llama.LlamaModel.LlamaModeljforward = lookahead_llama.LlamaModeljforward
    #modeling_llama.LlamaModel.j_prepare_decoder_attention_mask = lookahead_llama.j_prepare_decoder_attention_mask    

def augment_generate():
    FUNC_MAP["greedy_search"] = GenerationMixin.greedy_search
    FUNC_MAP["sample"] = GenerationMixin.sample
    GenerationMixin.greedy_search = greedy_search_proxy
    GenerationMixin.sample = sample_proxy
    #FUNC_MAP["sample"] = GenerationMixin.sample
    #GenerationMixin.sample = sample_proxy
    
def augment_all():
    augment_llama()
    augment_generate()

def log_history(clear=False):
    gen = 0
    step = 0    
    if "log" in CONFIG_MAP:
        for log in CONFIG_MAP["log"]:
            gen += log[0]
            step += log[1]
    if clear:
        CONFIG_MAP["log"] = []
    print("LADE LOG - OVERALL GEN: ", gen, " STEPS: ", step, " AVG COMPRESS RATIO: ", (gen / step) if step > 0 else 0)

def save_log(log_dir):
    if "log" in CONFIG_MAP:
        torch.save(CONFIG_MAP["log"], log_dir)

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