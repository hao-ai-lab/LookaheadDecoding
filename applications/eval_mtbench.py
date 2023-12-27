"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
#adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm
from typing import Dict, List, Optional
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype
import time
import lade

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    debug,
    cache_dir,
    cpu_offloading,
    use_pp,
    use_tp_ds,
    use_flash
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    ###not shuffle
    #random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                debug=debug,
                cache_dir=cache_dir,
                cpu_offloading=cpu_offloading,
                use_pp=use_pp,
                use_tp_ds=use_tp_ds,
                use_flash=use_flash
            )
        )


from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from fastchat.model.model_adapter import Llama2Adapter, raise_warning_for_incompatible_cpu_offloading_configuration

def load_model(
    model_path: str,
    device: str = "cuda",
    device_map: str= "",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    revision: str = "main",
    debug: bool = False,
    use_flash:bool = False
):
    """Load a model from Hugging Face."""
    # get model adapter
    adapter = Llama2Adapter()
    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
        if CPU_ISA in ["avx512_bf16", "amx"]:
            try:
                import intel_extension_for_pytorch as ipex

                kwargs = {"torch_dtype": torch.bfloat16}
            except ImportError:
                warnings.warn(
                    "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                )
    elif device.startswith("cuda"):
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )
            if debug:
                print(model)
            return model, tokenizer
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype
    if use_flash:
        kwargs["use_flash_attention_2"] = use_flash
    if len(device_map) > 0:
        kwargs["device_map"] = device_map
    # Load model
    model, tokenizer = adapter.load_model(model_path, kwargs)
     
    if len(device_map) > 0:
        return model, tokenizer

    if (
        device == "cpu"
        and kwargs["torch_dtype"] is torch.bfloat16
        and CPU_ISA is not None
    ):
        model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

    if (device.startswith("cuda") and num_gpus == 1 and not cpu_offloading) or device in (
        "mps",
        "xpu",
        "npu",
    ):
        model.to(device)

    if device == "xpu":
        model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    if debug:
        print(model)

    return model, tokenizer

#@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    debug,
    cache_dir,
    cpu_offloading,
    use_pp,
    use_tp_ds,
    use_flash
):  
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    
    print("configuration: ", "flash attn: ", use_flash, " HF PP: ",  use_pp, " DS TP: ", use_tp_ds, " GPUS: ", devices)

    ds_local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if use_pp:
        model, tokenizer = load_model(
        model_path,
        use_flash=use_flash,
        device=f"cuda",
        device_map="balanced",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=cpu_offloading,
        debug=debug,
        )
    
    elif use_tp_ds:
        import deepspeed
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        model, tokenizer = load_model(
        model_path,
        use_flash=use_flash,
        device_map="cpu",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=cpu_offloading,
        debug=debug,
        )
        model = deepspeed.init_inference(
            model,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            dtype=torch.half
        )
    else:
        model, tokenizer = load_model(
        model_path,
        use_flash=use_flash,
        device=f"cuda:{lade.get_device()}",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=cpu_offloading,
        debug=debug,
        )
        #model = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, torch_dtype=torch.float16, device_map=lade.get_device())
        model.tokenizer = tokenizer

    overall_time = 0
    overall_tp = 0
    overall_gen = 0
    count_gen = 0
    stats = {}
    for question_idx, question in enumerate(tqdm(questions)):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        temperature = 0.0 #force greedy
        
        stats[question_idx] = {} #
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            prompts = []

            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                
                # some models may error out when generating long outputs
                if True:
                    start_time = time.time()
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                    )
                    end_time = time.time()
                    gap_time = end_time - start_time 
                    tokens = output_ids.numel() - len(input_ids[0])
                    overall_time += gap_time
                    overall_gen += tokens
                    overall_tp += tokens / gap_time
                    count_gen += 1

                    stats[question_idx][j] = [gap_time, tokens]
                    if lade.get_device() == 0 and ds_local_rank == 0:
                        print([f"step {i} turn {j} time: ", gap_time, " generated tokens: ", tokens, " throughput: " , tokens / gap_time])
                    
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                '''
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"
                '''
                turns.append(output)
                conv.messages[-1][-1] = output

            choices.append({"index": i, "turns": turns, "prompts" : prompts})

        if lade.get_device() == 0 and ds_local_rank == 0:
            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
        #if question_idx == 1:
        #    break

    if lade.get_device() == 0 and ds_local_rank == 0:
        torch.save(stats[question_idx], answer_file + ".pt")
        print("LOG SAVE TO ", answer_file + ".pt")
        print(f"AVERAGE THROUGHPUT1 {overall_tp / count_gen} AVERAGE THROUGHPUT2 {overall_gen / overall_time} STAT {[overall_tp, count_gen, overall_gen, overall_time]}")
        lade.log_history()
        lade.save_log(answer_file + "-lade-log.pt")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--cpu_offloading", action="store_true"
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--guess",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--use-pp",
        type=int,
        default=0,
    )    
    parser.add_argument(
        "--use-tp-ds",
        type=int,
        default=0,
    )    
    parser.add_argument(
        "--use-flash",
        type=int,
        default=0,
    )  

    args = parser.parse_args()
    if int(os.environ.get("USE_LADE", 0)):
        
        lade.augment_all()
        lade.config_lade(LEVEL=args.level, WINDOW_SIZE=args.window, GUESS_SET_SIZE=args.guess, DEBUG=1, USE_FLASH=args.use_flash, DIST_WORKERS=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")))
        print("lade activated config: ",  lade.decoding.CONFIG_MAP)

    question_file = f"mtbench.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        debug=args.debug,
        cache_dir=args.cache_dir,
        cpu_offloading=args.cpu_offloading,
        use_pp=args.use_pp,
        use_tp_ds=args.use_tp_ds,
        use_flash=args.use_flash
    )
    ds_local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if lade.get_device() == 0 and ds_local_rank == 0:
        reorg_answer_file(answer_file)
