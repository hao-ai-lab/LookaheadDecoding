from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 
import os 
if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=1)

assert torch.cuda.is_available()

torch_device = "cuda"

model_name = "PY007/TinyLlama-1.1B-Chat-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=torch_device)
model.tokenizer = tokenizer
prompt = "How do you fine tune a large language model?"
input_text = (
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)

model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

#warm up
greedy_output = model.generate(**model_inputs, max_new_tokens=1)
#end warm up

# generate 256 new tokens
torch.cuda.synchronize()
t0 = time.time()
greedy_output = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
torch.cuda.synchronize()
t1 = time.time()

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=False))
print("Generated Tokens:", (greedy_output.numel() - model_inputs['input_ids'].numel()) ,"Generation Speed: ", (greedy_output.numel() - model_inputs['input_ids'].numel()) / (t1 - t0), " tokens/s")

#python minimal.py #44 tokens/s
#LOAD_LADE=1 USE_LADE=1 python minimal.py #74 tokens/s, 1.6x throughput without changing output distribution!

