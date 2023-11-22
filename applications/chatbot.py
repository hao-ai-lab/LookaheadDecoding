import torch
import argparse
import subprocess
import lade
from lade.utils import get_model
import time, os

if __name__ == "__main__":
    lade.augment_all()
    #lade.config_lade(LEVEL=8, WINDOW_SIZE=80, GUESS_SET_SIZE=80, DEBUG=1)

    lade.config_lade(LEVEL=5, WINDOW_SIZE=15, GUESS_SET_SIZE=15, DEBUG=1) #A100
    #lade.config_lade(LEVEL=5, WINDOW_SIZE=10, GUESS_SET_SIZE=10, DEBUG=1) #Game GPU like 3090
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0) 
    parser.add_argument("--model_path", type=str, help="model path", default="meta-llama/Llama-2-7b-chat-hf") #tiiuae/falcon-7b-instruct #"TheBloke/Falcon-180B-Chat-GPTQ" 
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--quant", type=str, default="")
    parser.add_argument("--use_ds", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    
    if args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    
    #if args.use_ds:
    model, tokenizer = get_model(args.model_path, args.quant, args.dtype, args.device, args.cache_dir, args.use_ds, False)

    user_input = ""
    num_rounds = 0
    if args.model_type == "llama":  
        roles = ("[INST]", "[/INST]") #support llama2 only
    else:
        assert False 

    user_input = ""
    if args.model_type == "llama":  
        system_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"

    first_time = True
    while True:
        num_rounds += 1
        if args.chat:
            model_input = input("User: ")
        else:
            model_input = '''Which methods did Socrates employ to challenge the prevailing thoughts of his time?'''
            print("User: " + model_input)
        if system_prompt is not None and first_time:
            if args.model_type == "llama":  
                new_inputs = system_prompt + f"{model_input}\n {roles[1]} "
            new_inputs = "[INST]" + f"{model_input}\n {roles[1]} "
            first_time = False
        else:
            new_inputs = f"{roles[0]}: {model_input}\n {roles[1]}: "
        user_input += new_inputs

        generate_kwargs = dict(max_new_tokens=1024, do_sample=False, stop_token=None, top_p=1.0, temperature=1.0) #greedy

        print("Assistant: " , flush=True, end="")
        input_ids = tokenizer(user_input, return_tensors="pt",
                          max_length=1024, truncation=True).input_ids.to(args.device)

        if not args.chat:
            lade.config_lade(DEBUG=0)
            tmp_kwargs = dict(max_new_tokens=1, do_sample=False, stop_token=None, top_p=1.0, temperature=1.0) 
            tmp_greedy_output = model.generate(input_ids=input_ids, **tmp_kwargs).tolist() #warmup
            lade.config_lade(DEBUG=1)

        os.environ["CHAT"] = "1"
        t0 = time.time()
        greedy_output = model.generate(input_ids=input_ids, **generate_kwargs).tolist()
        
        t1 = time.time()
        os.environ["CHAT"] = "0"
        output = tokenizer.decode(greedy_output[0], skip_special_tokens=False)

        user_input = f"{output}\n\n"
        
        if args.debug:
            generated_tokens = len(greedy_output[0]) - input_ids.numel()
            print()
            print("======================================SUMMARY=========================================")
            print("Input tokens: ", input_ids.numel() ,"Generated tokens: ", generated_tokens,"Time: ", round(t1 - t0, 2), "s Throughput: ", round(generated_tokens / (t1 - t0), 2), "tokens/s")
            print("======================================================================================")
            #print("\n\n\n\n")
        if not args.chat:
            break
