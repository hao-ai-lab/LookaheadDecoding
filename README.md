## Introduction 
We introduce Lookahead Decoding, 

## Contents
- [Introduction](#introduction)
- [Contents](#contents)
- [Installation](#installation)
  - [Install From The Source](#install-from-source)
  - [Inference](#inference)
  - [Use In Your Own Code](#inference-plugin)
- [Citation](#citation)
- [Guidance](#guidance)


## Installation
### Install from the source
```bash
git clone https://github.com/hao-ai-lab/ParallelDecoding.git
cd ParallelDecoding
pip install -r requirements.txt
pip install -e .
```

### Inference With Lookahead decoding
You can run the minimal example to see the speedup that Lookahead decoding brings.
```bash
python minimal.py #no Lookahead decoding
USE_LADE=1 LOAD_LADE=1 python minimal.py #use Lookahead decoding, 1.6x speedup
```

You can also enjoy chatting with your own chatbots with Lookahead decoding.
```bash
USE_LADE=1 python applications/chatbot.py  --model_path meta-llama/Llama-2-7b-chat-hf --debug --chat #chat, with lookahead 
USE_LADE=0 python applications/chatbot.py  --model_path meta-llama/Llama-2-7b-chat-hf --debug --chat #chat, without lookahead


USE_LADE=1 python applications/chatbot.py  --model_path meta-llama/Llama-2-7b-chat-hf --debug #no chat, with lookahead
USE_LADE=0 python applications/chatbot.py  --model_path meta-llama/Llama-2-7b-chat-hf --debug #no chat, without lookahead
```

### Use Lookahead decoding in your own code
You can import and use Lookahead decoding in your own code in three LoCs. Note that Lookahead decoding only support LLaMA and Greedy Search yet.

```python
import lade
lade.augment_all()
lade.config_pading(LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7, DEBUG=0)
```
## Citation

## Guidance
The core implementation is in decoding.py. Lookahead decoding requires an adaptation for each specific model. An example is in models/llama.py.

