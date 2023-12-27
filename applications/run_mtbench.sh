#download data 
wget https://raw.githubusercontent.com/lm-sys/FastChat/v0.2.31/fastchat/llm_judge/data/mt_bench/question.jsonl -O mtbench.jsonl 

export CUDA=0
export LADE=0
export LEVEL=0
export WIN=0
export GUESS=0
export FLASH=0
export PP=0
CUDA_VISIBLE_DEVICES=$CUDA USE_LADE=$LADE python eval_mtbench.py \
    --model-path meta-llama/Llama-2-7b-chat-hf --model-id \
    llama-2-7b-level-$LEVEL-win-$WIN-guess-$GUESS-f$FLASH-pp$CUDA \
    --level $LEVEL --window $WIN --guess $GUESS --use-flash $FLASH --use-pp $PP

export CUDA=0
export LADE=1
export LEVEL=5
export WIN=15
export GUESS=15
export FLASH=0
export PP=0
CUDA_VISIBLE_DEVICES=$CUDA USE_LADE=$LADE python eval_mtbench.py \
    --model-path meta-llama/Llama-2-7b-chat-hf --model-id \
    llama-2-7b-level-$LEVEL-win-$WIN-guess-$GUESS-f$FLASH-pp$CUDA \
    --level $LEVEL --window $WIN --guess $GUESS --use-flash $FLASH --use-pp $PP

export GPUS=1
export LEVEL=0
export WIN=0
export GUESS=0
export FLASH=0
deepspeed --num_gpus $GPUS eval_mtbench.py --model-path meta-llama/Llama-2-7b-chat-hf \
    --model-id llama-2-7b-level-$LEVEL-win-$WIN-guess-$GUESS-f$FLASH-ds$GPUS \
    --level $LEVEL --window $WIN --guess $GUESS --use-flash $FLASH --use-tp-ds 1
