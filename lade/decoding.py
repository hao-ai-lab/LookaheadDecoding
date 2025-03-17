from transformers import GenerationMixin
import torch
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput, SampleOutput, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper
import torch.distributed as dist
import os, time, random
FUNC_MAP = {}
CONFIG_MAP = {}
COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))

def greedy_search_proxy(self, *args, **kwargs):
    USE_LADE = int(os.environ.get("USE_LADE", 0))
    CHAT = int(os.environ.get("CHAT", 0))
    if CHAT and USE_LADE:
        return jacobi_greedy_search_multilevel(self, chat=True, *args, **kwargs)
    elif CHAT:
        return greedy_search_chat(self, chat=True, *args, **kwargs)
    
    if USE_LADE:
        return jacobi_greedy_search_multilevel(self, chat=False, *args, **kwargs)
    else:
        return FUNC_MAP["greedy_search"](self, *args, **kwargs)

def sample_proxy(self, *args, **kwargs):
    USE_LADE = int(os.environ.get("USE_LADE", 0))
    
    if USE_LADE:
        return jacobi_sample_multilevel(self, chat=int(os.environ.get("CHAT", 0)), *args, **kwargs)
    else:
        return FUNC_MAP["greedy_search"](self, *args, **kwargs)


def update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE):
    if GUESS_SET_SIZE != -1: #limited guess set size for each key, lru policy  
        if lst_token not in token_map:
            token_map[lst_token] = []
        tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
        if tup in token_map[lst_token]:
            token_map[lst_token].remove(tup)
            token_map[lst_token].append(tup)
        elif len(token_map[lst_token]) < GUESS_SET_SIZE:
            token_map[lst_token].append(tup) 
        else:
            assert len(token_map[lst_token]) == GUESS_SET_SIZE
            token_map[lst_token] = token_map[lst_token][1:] + [tup]

        for i in range(1, WINDOW_SIZE):
            if past_tokens[0][i - 1] not in token_map:
                token_map[past_tokens[0][i - 1]] = []
            tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)

            if tup in token_map[past_tokens[0][i - 1]]:
                token_map[past_tokens[0][i - 1]].remove(tup)
                token_map[past_tokens[0][i - 1]].append(tup)
            elif len(token_map[past_tokens[0][i - 1]]) < GUESS_SET_SIZE:
                token_map[past_tokens[0][i - 1]].append(tup) 
            else:
                assert len(token_map[past_tokens[0][i - 1]]) == GUESS_SET_SIZE
                token_map[past_tokens[0][i - 1]] = token_map[past_tokens[0][i - 1]][1:] + [tup]

    else: #unlimited guess set size for each key 
        #first add 
        if lst_token not in token_map:
            token_map[lst_token] = set()
        #build tuple
        tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
        #add tuple
        token_map[lst_token].add(tup) 

        for i in range(1, WINDOW_SIZE):
            if past_tokens[0][i - 1] not in token_map:
                token_map[past_tokens[0][i - 1]] = set()
            tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)
            token_map[past_tokens[0][i - 1]].add(tup) 

def append_new_generated_pool(tokens, token_map, LEVEL, GUESS_SET_SIZE):
    if len(tokens) != LEVEL:
        return 
    lst_token = tokens[0]
    tup = tuple(tokens[1:])

    if GUESS_SET_SIZE != -1: #limited guess set size for each key, lru policy  
        if lst_token not in token_map:
            token_map[lst_token] = []
        if tup in token_map[lst_token]:
            token_map[lst_token].remove(tup)
            token_map[lst_token].append(tup)
        elif len(token_map[lst_token]) < GUESS_SET_SIZE:
            token_map[lst_token].append(tup) 
        else:
            assert len(token_map[lst_token]) == GUESS_SET_SIZE
            token_map[lst_token] = token_map[lst_token][1:] + [tup]
    else: #unlimited guess set size for each key 
        #first add 
        if lst_token not in token_map:
            token_map[lst_token] = set()
        token_map[lst_token].add(tup) 


def fill_pool_with_prompt(prompts, token_map, LEVEL, GUESS_SET_SIZE):
    for start_idx in range(len(prompts) - LEVEL + 1):
        lst_token = prompts[start_idx]
        tup = tuple(prompts[start_idx+1:start_idx+LEVEL])
        
        if len(tup) != LEVEL - 1:
            return 
        
        if GUESS_SET_SIZE != -1: #limited guess set size for each key, lru policy  
            if lst_token not in token_map:
                token_map[lst_token] = []
            if tup in token_map[lst_token]:
                token_map[lst_token].remove(tup)
                token_map[lst_token].append(tup)
            elif len(token_map[lst_token]) < GUESS_SET_SIZE:
                token_map[lst_token].append(tup) 
            else:
                assert len(token_map[lst_token]) == GUESS_SET_SIZE
                token_map[lst_token] = token_map[lst_token][1:] + [tup]
        else: #unlimited guess set size for each key 
            #first add 
            if lst_token not in token_map:
                token_map[lst_token] = set()
            token_map[lst_token].add(tup) 



def filter_window(level_window, eos_token_id, reset_func):
    
    for idx in range(len(level_window)):
        if level_window[idx] == eos_token_id:
            level_window[idx] = reset_func()

def jacobi_sample_multilevel(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    chat: bool = False,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
    For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> model.generation_config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model.sample(
    ...     input_ids,
    ...     logits_processor=logits_processor,
    ...     logits_warper=logits_warper,
    ...     stopping_criteria=stopping_criteria,
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    WINDOW_SIZE = CONFIG_MAP.get("WINDOW_SIZE", 60)
    GUESS_SET_SIZE = CONFIG_MAP.get("GUESS_SET_SIZE", 60)
    ALWAYS_FWD_ONE = CONFIG_MAP.get("ALWAYS_FWD_ONE", 1)
    LEVEL = CONFIG_MAP.get("LEVEL", 8)
    DEBUG = CONFIG_MAP.get("DEBUG", 0)
    DIST_WORKERS = CONFIG_MAP.get("DIST_WORKERS", 1)
    LOCAL_RANK = CONFIG_MAP.get("LOCAL_RANK", 0)
    USE_FLASH = CONFIG_MAP.get("USE_FLASH", 0) #not use flash by default
    POOL_FROM_PROMPT = CONFIG_MAP.get("POOL_FROM_PROMPT", 0)
    USE_AWQ = False #not support AWQ
    #IN FLASH ATTENTION WE REORDERED LOOKAHEAD WINDOW 

    GUESS_SIZE = LEVEL - 1
    NOT_SEQ = 0
    CONTINUE_ALL = 0
    TEMP_FOR_GUESS = 0.0
    
    assert TEMP_FOR_GUESS == 0
    #assert LEVEL <= 8
    def random_set():
        return random.randint(0,self.vocab_size - 1)

    all_old_tokens = input_ids[0].tolist()
    init_len = len(all_old_tokens)
    #print("original: ", init_len, input_ids.numel())

    def copy_from():
        return random.choice(all_old_tokens)
    
    order_copy_from_idx = [0]

    def order_copy_from():
        if order_copy_from_idx[0] >= len(all_old_tokens):
            order_copy_from_idx[0] = 0
        ret = all_old_tokens[order_copy_from_idx[0]]
        order_copy_from_idx[0] = 1 + order_copy_from_idx[0]
        return ret

    def copy_from_last():
        return all_old_tokens[-1]

    set_token = copy_from

    past_tokens = [[set_token() for _ in range(WINDOW_SIZE + LEVEL - 3)]] + [None for _ in range(LEVEL - 2)]

    if DIST_WORKERS > 1:
        dist.broadcast_object_list(past_tokens, src=0) #keep past_tokens always the same on different GPUs
    
    ###############end Init methods
    fill_level = 0
    guess_tokens = None
    token_map = {}
    steps = 0
    guess_skip_dist = 0

    if POOL_FROM_PROMPT:
        fill_pool_with_prompt(all_old_tokens, token_map, LEVEL, GUESS_SET_SIZE)

    if chat:
        init = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                   spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
        prev = len(init)

    for warper in logits_warper:
        #assert type(warper) == TemperatureLogitsWarper or type(warper) == TopPLogitsWarper or type(warper) == TopKLogitsWarper,  f"please set top_k=0 {warper}"
        assert type(warper) == TemperatureLogitsWarper or type(warper) == TopKLogitsWarper or type(warper) == TopPLogitsWarper,  f"please set top_k=0.0 and top_p=1.0 {warper}"

    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        #this only support llama, check compatibility with other models
        past_key_values = model_kwargs.pop("past_key_values", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        else:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1 - guess_skip_dist:]
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -1 - guess_skip_dist:]
        model_inputs["past_key_values"] = past_key_values

        if past_tokens[LEVEL - 2] is not None and lst_token in token_map and GUESS_SET_SIZE > 0:  
            guess_tokens_ = token_map[lst_token]
            guess_tokens = []
            for tok in list(guess_tokens_):
                guess_tokens += list(tok)
        else:
            guess_tokens = None

        #not support logits_processor yet
        assert return_dict_in_generate == False
        assert len(logits_processor) == 0

        # forward pass to get next token
        outputs = self.jforward_multilevel(
            **model_inputs,
            past_tokens=past_tokens,
            guess_tokens=guess_tokens,
            return_dict=True,
            not_seq=NOT_SEQ,
            continue_all=CONTINUE_ALL,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            level=LEVEL,
            WINDOWS_SIZE=WINDOW_SIZE,
            guess_size=GUESS_SIZE,
            fill_level=fill_level,
            dist_workers=DIST_WORKERS,
            la_mask_offset=0,
            local_rank=LOCAL_RANK,
            use_flash=USE_FLASH
        )

        steps += 1

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.out_logits #outputs.logits[:, -1, :]

        #not support logits_processor and only support temperature w/o top-p top-k, I will support these two later
        # pre-process distribution
        next_token_scores = logits_warper(input_ids, next_token_logits)

        #delete return_dict_in_generate here, we set it to False
        # Store scores, attentions and hidden_states when required
        
        # finished sentences should have their next token be a padding token
        #if eos_token_id is not None:
        #    if pad_token_id is None:
        #        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        #    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        #for bs > 1, so I comment these out

        #handling output
        max_hit = 0

        if past_tokens[1] is None:
            #first fill, not use verification branch
            assert fill_level == 0
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            hits = [next_tokens.item()] 

            past_tokens[0] = past_tokens[0][1:] 
            past_tokens[1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist() #fill window with argmax

            fill_level += 1
        elif past_tokens[LEVEL - 2] is None: 
            #fill other levels, not use verification branch
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            hits = [next_tokens.item()] 

            for level in range(fill_level + 1):
                past_tokens[level] = past_tokens[level][1:] 

            past_tokens[fill_level + 1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()[1:] #fill window with argmax
            
            fill_level += 1
        else:
            

            if guess_tokens is not None:
                probs_next = torch.nn.functional.softmax(next_token_scores, dim=-1)[0]
                hits = []
                #= original model output
                guess_logits = logits_warper(input_ids, outputs.guess_logits[0])
                guess_probs = torch.nn.functional.softmax(guess_logits, dim=-1) #
                #guess_results = torch.argmax(outputs.guess_logits, dim=-1)[0].tolist()
                guess_indices = list(range(outputs.guess_logits.size(1) // GUESS_SIZE))
                #algorithm modified from specinfer
                for idx_in_ngram in range(GUESS_SIZE):
                    
                    g_idx = 0
                    is_accept = False
                    #print("gues: ", guess_indices)
                    
                    while g_idx < len(guess_indices):
                        guess_idx = guess_indices[g_idx]
                        guess_offset = guess_idx * GUESS_SIZE


                        #draft_guess is draft model (by lookahead) generation
                        draft_guess = guess_tokens[guess_offset + idx_in_ngram]
                        prob_accept = min(1, probs_next[draft_guess].item()) #min(1, prob_llm/prob_draft) #use argmax, prob_draft is 1
                        sample_prob = random.random()

                        if sample_prob < prob_accept:
                            #accept
                            hits.append(draft_guess)
                            is_accept = True 
                            max_hit_idx = guess_idx
                            new_guess_indices = []
                            for guess_idx_n in guess_indices:
                                guess_offset_n = guess_idx_n * GUESS_SIZE
                                new_draft_guess = guess_tokens[guess_offset_n + idx_in_ngram]
                                if new_draft_guess == draft_guess:
                                    new_guess_indices.append(guess_idx_n)
                            guess_indices = new_guess_indices
                            break 
                        else:
                            #not accept
                            #max norm (argmax)
                            probs_next[draft_guess] = 0
                            probs_next = probs_next / probs_next.sum()
                            g_idx += 1         
                    
                    if is_accept:
                        probs_next = guess_probs[guess_offset + idx_in_ngram]
                        continue 
                    else:
                        new_token_gen = torch.multinomial(probs_next, num_samples=1).item()
                        #print("non accept: ", probs_next.size(), new_token_gen)
                        hits.append(new_token_gen)
                        break

                #hits.append(new_token_gen)

                max_hit = len(hits) - 1
                        
            else:
                probs_next = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs_next, num_samples=1).squeeze(1)
                hits = [next_tokens.item()]


            #new window level, use argmax to generate
            new_results = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            
            assert len(past_tokens[LEVEL - 2]) == WINDOW_SIZE and len(new_results) == WINDOW_SIZE

            update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE)

            #update windows when max_hit > 1
            if ALWAYS_FWD_ONE:
                past_tokens[0] = past_tokens[1][1:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][:]

                past_tokens[LEVEL - 2] = new_results             
            else:
                past_tokens[0] = past_tokens[1][1 + max_hit:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][max_hit:]

                past_tokens[LEVEL - 2] = new_results[max_hit:]
            

            if max_hit > 0:
                if not ALWAYS_FWD_ONE:
                    for level in range(LEVEL - 1):
                        past_tokens[level] = past_tokens[level] + [set_token() for _ in range(max_hit)]

                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat((attention_mask, torch.ones(1, max_hit, device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            if eos_token_id is not None:
                #filter <EOS> (we find too many <EOS> in window lead to numerical error)
                filter_window(past_tokens[LEVEL - 2], eos_token_id[0], set_token)

        #update kv cache of correctly speculated tokens
        past_key_values = []
        for idx, kv in enumerate(outputs.past_key_values):
            for hh in range(max_hit):
                assert outputs.step_len == kv[0].size(2)
                kv[0][:,:,outputs.kvcache_len + hh,:] = kv[0][:,:,outputs.step_len-len(guess_tokens)+max_hit_idx * GUESS_SIZE + hh,:]
                kv[1][:,:,outputs.kvcache_len + hh,:] = kv[1][:,:,outputs.step_len-len(guess_tokens)+max_hit_idx * GUESS_SIZE + hh,:]
            past_key_values.append( (kv[0][:,:,:outputs.kvcache_len + max_hit,:], kv[1][:,:,:outputs.kvcache_len + max_hit,:]) )
        outputs.past_key_values = past_key_values

        lst_token = hits[max_hit]

        for hit_ids in range(max_hit + 1):
            if eos_token_id is not None and hits[hit_ids] == eos_token_id[0]:
                all_old_tokens.append(hits[hit_ids])
                next_tokens = eos_token_id_tensor
                max_hit = hit_ids
                break
            else:
                all_old_tokens.append(hits[hit_ids])
                if POOL_FROM_PROMPT:
                    append_new_generated_pool(all_old_tokens[-LEVEL:], token_map, LEVEL, GUESS_SET_SIZE)

        if chat:

            all_str = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                    spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
            if COLOR_PRINT:
                from termcolor import colored
                if max_hit > 1:
                    not_hit = self.tokenizer.decode(all_old_tokens[:-max_hit + 1], skip_special_tokens=True, \
                                    spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,) 
                    pt = colored(not_hit[prev:],"blue") +  colored(all_str[len(not_hit):], "blue")
                else:
                    pt = all_str[prev:]                    
                print(pt,  flush=True, end="")
            else:
                print(all_str[prev:],  flush=True, end="")
            prev = len(all_str)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, torch.tensor(hits[:max_hit + 1], device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)], dim=-1)

        #input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)


        ###not change codes below
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
    
    #if predict more tokens than max_length, remove them
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            all_old_tokens = all_old_tokens[:criteria.max_length]
            input_ids = input_ids[:,:criteria.max_length]

    if max_length is not None:
        all_old_tokens = all_old_tokens[:init_len + max_length]
        input_ids = input_ids[:][:init_len + max_length]
    #end handling 
    if DEBUG and LOCAL_RANK == 0:
        print("\n==========================ACCELERATION===SUMMARY======================================")
        print("Generated tokens: ", len(all_old_tokens) - init_len, "Total steps: ", steps, " Compression ratio: ", round((len(all_old_tokens) - init_len) / steps, 2))
        print("======================================================================================", end="")
        CONFIG_MAP["log"].append([len(all_old_tokens) - init_len, steps, round((len(all_old_tokens) - init_len) / steps, 2)])

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids




def jacobi_greedy_search_multilevel(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    
    chat: bool = False, 
    stop_token: Optional[str]= None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    ############### configurations 
    WINDOW_SIZE = CONFIG_MAP.get("WINDOW_SIZE", 60)
    GUESS_SET_SIZE = CONFIG_MAP.get("GUESS_SET_SIZE", 60)
    ALWAYS_FWD_ONE = CONFIG_MAP.get("ALWAYS_FWD_ONE", 1)
    LEVEL = CONFIG_MAP.get("LEVEL", 8)
    DEBUG = CONFIG_MAP.get("DEBUG", 0)
    DIST_WORKERS = CONFIG_MAP.get("DIST_WORKERS", 1)
    LOCAL_RANK = CONFIG_MAP.get("LOCAL_RANK", 0)
    USE_FLASH = CONFIG_MAP.get("USE_FLASH", 0) #not use flash by default
    POOL_FROM_PROMPT = CONFIG_MAP.get("POOL_FROM_PROMPT", 0)
    USE_AWQ = False #not support AWQ
    #IN FLASH ATTENTION WE REORDERED LOOKAHEAD WINDOW 

    GUESS_SIZE = LEVEL - 1
    NOT_SEQ = 0
    CONTINUE_ALL = 0
    TEMP_FOR_GUESS = 0.0
    USE_AWQ = False 
    import random
    assert TEMP_FOR_GUESS == 0
    assert ALWAYS_FWD_ONE == 1
    assert USE_AWQ == False 

    ############### Init methods
    #random.seed(10) #unset this random seed later

    all_old_tokens = input_ids[0].tolist()
    init_len = len(all_old_tokens)
    order_copy_from_idx = [0]


    def random_set():
        return random.randint(0,self.vocab_size - 1)

    def copy_from():
        return random.choice(all_old_tokens)

    def order_copy_from():
        if order_copy_from_idx[0] >= len(all_old_tokens):
            order_copy_from_idx[0] = 0
        ret = all_old_tokens[order_copy_from_idx[0]]
        order_copy_from_idx[0] = 1 + order_copy_from_idx[0]
        return ret

    def copy_from_last():
        return all_old_tokens[-1]

    set_token = copy_from

    past_tokens = [[set_token() for _ in range(WINDOW_SIZE + LEVEL - 3)]] + [None for _ in range(LEVEL - 2)]
    #past_tokens is the lookahead window. Current we initialize it with random copy from prompts

    if DIST_WORKERS > 1:
        dist.broadcast_object_list(past_tokens, src=0) #keep past_tokens always the same on different GPUs
    
    ###############end Init methods
    fill_level = 0
    guess_tokens = None
    token_map = {}
    steps = 0
    guess_skip_dist = 0

    if POOL_FROM_PROMPT:
        fill_pool_with_prompt(all_old_tokens, token_map, LEVEL, GUESS_SET_SIZE)
        
    if chat:
        init = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                   spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
        prev = len(init)

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break
        
        # prepare model inputs
        #this only support llama, check compatibility with other models
        past_key_values = model_kwargs.pop("past_key_values", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        else:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1 - guess_skip_dist:]
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -1 - guess_skip_dist:]
        model_inputs["past_key_values"] = past_key_values

        ori_guess = None
        #set up guess_tokens for verification branch 
        # past_tokens[LEVEL - 2] is not None means we are still in warmup stage filling multi-level window
        if past_tokens[LEVEL - 2] is not None and lst_token in token_map and GUESS_SET_SIZE > 0:  
            ###############NOT ENTER CURRENTLY
            guess_tokens_ = token_map[lst_token]
            guess_tokens = []
            for tok in list(guess_tokens_):
                guess_tokens += list(tok)
            ori_guess = guess_tokens
            #shards guess_tokens on different GPUs
            if DIST_WORKERS > 1 and guess_tokens is not None:
                assert len(guess_tokens) % GUESS_SIZE == 0
                cnt_guess = (len(guess_tokens) // GUESS_SIZE + DIST_WORKERS - 1) // DIST_WORKERS
                guess_base = cnt_guess * LOCAL_RANK
                guess_end = cnt_guess * (LOCAL_RANK + 1)
                guess_tokens = guess_tokens[GUESS_SIZE * guess_base: GUESS_SIZE * guess_end]
                if len(guess_tokens) == 0:
                    guess_tokens = None
        else:
            guess_tokens = None

        assert return_dict_in_generate == False
        assert len(logits_processor) == 0
        # forward pass to get next token        
        #if LOCAL_RANK == 0:
        #    print("position: ", model_inputs["input_ids"], model_inputs["position_ids"], ori_guess, guess_tokens)
        #forward 
        if DIST_WORKERS > 1:
            window_len = len(past_tokens[0]) + 1
            split_window_len = (window_len + DIST_WORKERS - 1) // DIST_WORKERS
            window_start = min(split_window_len * LOCAL_RANK, window_len)
            window_end = min(split_window_len * (LOCAL_RANK + 1), window_len)

            if LOCAL_RANK == DIST_WORKERS - 1:
                assert len(past_tokens[0]) == window_end - 1
            past_tokens_inp = [past_tokens[0][: window_end - 1]]
            for l in range(1, len(past_tokens)):
                tokens = past_tokens[l]
                past_tokens_inp.append(tokens[window_start: window_end] if tokens is not None else None)
        else:
            past_tokens_inp = past_tokens
            
        outputs = self.jforward_multilevel(
            **model_inputs,
            past_tokens=past_tokens_inp,
            guess_tokens=guess_tokens,
            return_dict=True,
            not_seq=NOT_SEQ,
            continue_all=CONTINUE_ALL,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            level=LEVEL,
            WINDOWS_SIZE=WINDOW_SIZE,
            guess_size=GUESS_SIZE,
            fill_level=fill_level,
            dist_workers=DIST_WORKERS,
            la_mask_offset=0,
            local_rank=LOCAL_RANK,
            use_flash=USE_FLASH
        )
        
        steps += 1

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        if past_tokens[LEVEL - 2] is None: #prefill  
            next_token_logits = outputs.out_logits
        else:
            next_token_logits = outputs.out_logits #outputs.logits[:, -1, :]

        # pre-process distribution
        #next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens_scores = next_token_logits
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if DIST_WORKERS > 1:
            torch.distributed.broadcast(next_tokens, src=0)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        first_guess = next_tokens.item()
        max_hit = 0 
        hits = [first_guess] + [0] * GUESS_SIZE

        new_results = []

        if past_tokens[1] is None: #filling multi-level window, the very first step is different
            assert fill_level == 0
            past_tokens[0] = past_tokens[0][1:] 
            past_tokens[1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            
            if DIST_WORKERS > 1:
                nn_past_tokens = [copy.deepcopy(past_tokens[1])]
                torch.distributed.broadcast_object_list(nn_past_tokens, src=DIST_WORKERS - 1)
                past_tokens[1] = nn_past_tokens[0]

            fill_level += 1
        elif past_tokens[LEVEL - 2] is None: #filling multi-level window
            for level in range(fill_level + 1):
                past_tokens[level] = past_tokens[level][1:] 
            current_past_tokens = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()
            
            
            if DIST_WORKERS > 1:
                nn_past_tokens = [None] * DIST_WORKERS
                torch.distributed.all_gather_object(nn_past_tokens, current_past_tokens)
                current_past_tokens = sum(nn_past_tokens, [])

            
            #time.sleep(10000)
            past_tokens[fill_level + 1] = current_past_tokens[1:]
            #print("new past: ", (LOCAL_RANK, past_tokens))


            fill_level += 1
        else: 
            #time.sleep(10000)
            #multi-level window is filled
            #match guess tokens 
            if guess_tokens is not None:
                guess_results = torch.argmax(outputs.guess_logits, dim=-1)[0].tolist()
                for eg in range(len(guess_results) // GUESS_SIZE):
                    egx = eg * GUESS_SIZE
                    correct = [first_guess] + guess_results[egx:egx + GUESS_SIZE]
                    myguess = guess_tokens[egx:egx + GUESS_SIZE]
                    gg = 0
                    # if all tokens in myguess equal to correct, gg should be len(myguess)
                    for gg in range(len(myguess) + 1):
                        if gg >= len(myguess) or myguess[gg] != correct[gg]:
                            break 
                    if gg > max_hit:
                        max_hit = gg 
                        max_hit_idx = eg 
                        hits[:max_hit + 1] = correct[:max_hit + 1]
            #max_hit is the length of longest accepted sequence in verification branch 

            #sync max_hit if we have multi-GPUs
            if DIST_WORKERS > 1:
                max_hit_all_ranks = [0] * DIST_WORKERS
                torch.distributed.all_gather_object(max_hit_all_ranks, max_hit)
                max_hit = max(max_hit_all_ranks)
                max_hit_rank = max_hit_all_ranks.index(max_hit)

                if max_hit > 0:
                    hit_info = [hits]
                    torch.distributed.broadcast_object_list(hit_info, src=max_hit_rank)
                    hits = hit_info[0]
                    #print("rank: ", [hits, torch.distributed.get_rank(), max_hit, LOCAL_RANK, max_hit_rank])
            #if LOCAL_RANK == 0:
            #    print("rank: ",hits, max_hit)
            #sync new_results
            new_results = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()

            if DIST_WORKERS > 1:
                nn_past_tokens = [None] * DIST_WORKERS
                torch.distributed.all_gather_object(nn_past_tokens, new_results)
                new_results = sum(nn_past_tokens, [])
            #else:
            #    current_past_tokens = new_results
            #print("brand new past: ", (LOCAL_RANK, past_tokens, new_results))

            #time.sleep(1000)

            assert len(past_tokens[LEVEL - 2]) == WINDOW_SIZE and len(new_results) == WINDOW_SIZE

            update_token_map(token_map, lst_token, past_tokens, new_results, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE)


            if ALWAYS_FWD_ONE:
                past_tokens[0] = past_tokens[1][1:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][:]

                past_tokens[LEVEL - 2] = new_results             
            else:
                past_tokens[0] = past_tokens[1][1 + max_hit:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][max_hit:]

                past_tokens[LEVEL - 2] = new_results[max_hit:]



        if max_hit > 0:
            if not ALWAYS_FWD_ONE:
                for level in range(LEVEL - 1):
                    past_tokens[level] = past_tokens[level] + [set_token() for _ in range(max_hit)]

            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat((attention_mask, torch.ones(1, max_hit, device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
        
        #not support awq
        assert USE_AWQ == False  

        past_key_values = []

        #plan to remove kv-cache copy and set tokens into next input when dist_workers > 1, as communication is costly
        if DIST_WORKERS > 1 and max_hit > 0:

            guess_skip_dist = max_hit
            for idx, kv in enumerate(outputs.past_key_values):
                past_key_values.append( (kv[0][:,:,:outputs.kvcache_len,:], kv[1][:,:,:outputs.kvcache_len,:]) )
            outputs.past_key_values = past_key_values
        else:
            guess_skip_dist = 0
            offset_kv_cache = outputs.step_len-len(guess_tokens)+max_hit_idx * GUESS_SIZE if max_hit > 0 else 0
            for idx, kv in enumerate(outputs.past_key_values):
                #update kv-cache from verification branch  
                if max_hit > 0:
                    kv[0][:,:,outputs.kvcache_len:outputs.kvcache_len+max_hit,:] = kv[0][:,:,offset_kv_cache:offset_kv_cache+max_hit,:]
                    kv[1][:,:,outputs.kvcache_len:outputs.kvcache_len+max_hit,:] = kv[1][:,:,offset_kv_cache:offset_kv_cache+max_hit,:]
                past_key_values.append( (kv[0][:,:,:outputs.kvcache_len + max_hit,:], kv[1][:,:,:outputs.kvcache_len + max_hit,:]) )
            outputs.past_key_values = past_key_values

        lst_token = hits[max_hit]

        #stopping condition
        for hit_idx in range(max_hit + 1):
            if eos_token_id is not None and hits[hit_idx] == eos_token_id[0]:
                all_old_tokens.append(hits[hit_idx])
                next_tokens = eos_token_id_tensor
                max_hit = hit_idx
                break
            else:
                all_old_tokens.append(hits[max_hit])
                if POOL_FROM_PROMPT:
                    append_new_generated_pool(all_old_tokens[-LEVEL:], token_map, LEVEL, GUESS_SET_SIZE)


        if chat and LOCAL_RANK == 0:
            all_str = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                    spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
            if COLOR_PRINT:
                from termcolor import colored
                if max_hit > 1:
                    not_hit = self.tokenizer.decode(all_old_tokens[:-max_hit + 1], skip_special_tokens=True, \
                                    spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,) 
                    pt = colored(not_hit[prev:],"blue") +  colored(all_str[len(not_hit):], "blue")
                else:
                    pt = all_str[prev:]                
                print(pt,  flush=True, end="")
            else:
                print(all_str[prev:],  flush=True, end="")
            prev = len(all_str)
        
        input_ids = torch.cat([input_ids, torch.tensor(hits[:max_hit + 1], device=next_tokens.device, dtype=next_tokens.dtype).unsqueeze(0)], dim=-1)
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
    
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            #print("steop: ",  criteria.max_length, init_len, len(all_old_tokens), input_ids.size())
            all_old_tokens = all_old_tokens[:criteria.max_length]
            input_ids = input_ids[:,:criteria.max_length]
    if max_length is not None:
        #print("max : ", max_length, init_len)
        all_old_tokens = all_old_tokens[:init_len + max_length]
        input_ids = input_ids[:][:init_len + max_length]

    if DEBUG and LOCAL_RANK == 0:
        print("\n==========================ACCELERATION===SUMMARY======================================")
        print("Generated tokens: ", len(all_old_tokens) - init_len, "Total steps: ", steps, " Compression ratio: ", round((len(all_old_tokens) - init_len) / steps, 2))
        print("======================================================================================", end="")
        CONFIG_MAP["log"].append([len(all_old_tokens) - init_len, steps, round((len(all_old_tokens) - init_len) / steps, 2)])

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids






def greedy_search_chat(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    chat: int=True,
    stop_token: Optional[str] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    
    assert input_ids.size(0) == 1
    all_old_tokens = input_ids[0].tolist()
    init_len = len(all_old_tokens)
    init = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                   spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
    prev = len(init)
    steps = 0
    this_peer_finished = False  # used by synced_gpus only
    while True:
        steps += 1
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        all_old_tokens.append(next_tokens.item())
        all_str = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                   spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
        if chat:
            print(all_str[prev:],  flush=True, end="")
        prev = len(all_str)


        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
    DEBUG = CONFIG_MAP.get("DEBUG", 0)
    if DEBUG:
        #print("===DEBUG INFO===", " generated tokens: ", len(all_old_tokens) - init_len, "total step: ", steps, len(token_map.keys()), sum(len(value) for value in token_map.values()), input_ids.numel(), reps)

        print("\n==========================ACCELERATION===SUMMARY======================================")
        print("Generated tokens: ", len(all_old_tokens) - init_len, "Total steps: ", steps, " Compression ratio: N/A ")
        print("======================================================================================", end="")

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
