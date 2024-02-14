from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 
import os 
if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 
    lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=1, USE_FLASH=True, POOL_FROM_PROMPT=True)

assert torch.cuda.is_available()

torch_device = "cuda"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=torch_device, attn_implementation="flash_attention_2")
model.tokenizer = tokenizer
prompt = "How do you fine tune a large language model?"
input_text = (
    f"<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
)


model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

#warm up
greedy_output = model.generate(**model_inputs, max_new_tokens=1)
#end warm up

# generate 256 new tokens
torch.cuda.synchronize()
t0s = time.time()
sample_output = model.generate(**model_inputs, max_new_tokens=256, do_sample=True, temperature=0.7,
                        top_k=50, top_p=0.9)
torch.cuda.synchronize()
t1s = time.time()

torch.cuda.synchronize()
t0g = time.time()
greedy_output = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
torch.cuda.synchronize()
t1g = time.time()

print("Output:\n" + 100 * '-')
print("Greedy output: ", tokenizer.decode(greedy_output[0], skip_special_tokens=False))
print("Sample output: ", tokenizer.decode(sample_output[0], skip_special_tokens=False))

print("Greedy Generated Tokens:", (greedy_output.numel() - model_inputs['input_ids'].numel()) ,"Generation Speed: ", (greedy_output.numel() - model_inputs['input_ids'].numel()) / (t1g - t0g), " tokens/s")
print("Sample Generated Tokens:", (sample_output.numel() - model_inputs['input_ids'].numel()) ,"Generation Speed: ", (sample_output.numel() - model_inputs['input_ids'].numel()) / (t1s - t0s), " tokens/s")

#python minimal.py #44 tokens/s
#LOAD_LADE=1 USE_LADE=1 python minimal.py #74 tokens/s, 1.6x throughput without changing output distribution!

