import torch, math, time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, CausalLMOutputWithPast, _expand_mask

def j_make_causal_mask_multilevel(
    level_sizes: list,is_prefill:bool, WINDOW_SIZE: int, guess : list, guess_size: int, not_seq:bool, continue_all:bool,input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

    if is_prefill:
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        assert past_key_values_length == 0
        assert guess is None
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    tiny_mask_size = level_sizes[0] + 1
    mask_cond = torch.arange(tiny_mask_size, device=device)
    hm = mask_cond < (mask_cond + 1).view(tiny_mask_size, 1)

    if guess is not None:
        mask[:,0] = 0
        lguess = len(guess)
        if guess_size == 2:
            small_m = torch.tensor([0, torch.finfo(dtype).min]).repeat(lguess // 2)[:-1]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m, -1)
        elif guess_size == 3:
            small_m1 = torch.tensor([0, 0, torch.finfo(dtype).min]).repeat(lguess // 3)[:-1]
            small_m2 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 3)[:-2]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2)
        elif guess_size == 4:
            small_m1 = torch.tensor([0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 4)[:-1]
            small_m2 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-2]
            small_m3 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-3]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3)
        elif guess_size == 5:
            small_m1 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 5)[:-1]
            small_m2 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-2]
            small_m3 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-3]
            small_m4 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-4]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4)
        elif guess_size == 6:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 6)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-2]
            small_m3 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-3]
            small_m4 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-4]
            small_m5 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-5]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5)
        elif guess_size == 7:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 7)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-2]
            small_m3 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-3]
            small_m4 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-4]
            small_m5 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-5]
            small_m6 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-6]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5).diagonal_scatter(small_m6, -6)

        else:
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0)
            for i in range(guess_size - 1): #7 : 0 - 5
                small_l = [0] * (guess_size - i - 1)  + [torch.finfo(dtype).min] * (i + 1)
                small_m = torch.tensor(small_l).repeat(lguess // guess_size)[:-1 - i]
                mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].diagonal_scatter(small_m, -1 - i)
            #assert False 

    else:
        assert tgt_len == sum(level_sizes) + 1

    #print("level: ", level_sizes)
    for ll in range(len(level_sizes)):
        if ll > 0:
            assert level_sizes[ll] == tiny_mask_size
        mask[tiny_mask_size*ll:tiny_mask_size*(ll+1),:tiny_mask_size].masked_fill_(hm, 0)
        for row in range(1, ll + 1):
            mask[tiny_mask_size*ll:tiny_mask_size*(ll+1),tiny_mask_size*row:tiny_mask_size*(row+1)].fill_diagonal_(0)


    #mask.masked_fill_(, 0)
    mask = mask.to(dtype)
    

    #lm[0] += 1
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)




def j_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, others):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    WINDOW_SIZE, is_prefill, guess, guess_size, not_seq, continue_all, level_sizes = others
    combined_attention_mask = None
    #print("size: ", input_shape, past_key_values_length)
    if input_shape[-1] > 1:
        combined_attention_mask = j_make_causal_mask_multilevel(
            level_sizes,
            is_prefill,            
            WINDOW_SIZE,
            guess,
            guess_size,
            not_seq,
            continue_all,
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        #print("shape: ", expanded_attn_mask.size(), combined_attention_mask.size())
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def LlamaModeljforward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    WINDOWS_SIZE: int=0,
    is_prefill: bool=False,
    level_sizes: Optional[List[int]] =None,
    guess_size: int=2,
    not_seq: bool=False,
    continue_all: bool=False,
    guess: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    #print("seq: ", seq_length, input_ids.shape, past_key_values[0][0].shape[2] if past_key_values is not None else None, attention_mask.shape, use_cache)
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()
    
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self.j_prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, (WINDOWS_SIZE, is_prefill, guess, guess_size, not_seq, continue_all, level_sizes), 
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
        else:
            layer_outputs = decoder_layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def jforward_multilevel(
    self,
    input_ids: torch.LongTensor = None,
    past_tokens: Optional[List[torch.FloatTensor]] = None,
    guess_tokens: Optional[List[torch.FloatTensor]] = None,
    guess_size: int = 2,
    not_seq: bool = False,
    continue_all: bool=False,
    level: int = 3,
    fill_level: int=-1,
    WINDOWS_SIZE: int=-1,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    

    #assert attention_mask.all().item(), " Mask Must All Be One "
    assert labels is None, " Inference Mode "
    assert input_ids.size(0) == 1, " single batch only "
    if level is not None:
        assert level == len(past_tokens) + 1
        assert guess_size == level - 1
    
    if past_key_values is not None:
        past_size = past_key_values[0][0].size(2)
        #assert past_size == attention_mask.size(1) - 1
    else:
        past_size = 0
    #print("past: ", past_size, )
   # assert past_size == attention_mask.size(1)

    prefill_size = input_ids.size(1) 
    for layer in self.model.layers:
        layer.self_attn.cur_len = prefill_size
    
    import time 
    level_sizes = []

    assert continue_all == False
    lst_id = position_ids[0][-1].item()

    all_past = []
    ids_list = []
    attn_size = 0
    for ll in range(fill_level + 1):
        #print("past size: ", len(past_tokens[ll]))
        all_past += past_tokens[ll]
        attn_size += len(past_tokens[ll])
        level_sizes.append(len(past_tokens[ll]))
        if ll == 0:
            ids_list += list(range(lst_id + 1, lst_id + 1 + len(past_tokens[ll])))
        else:
            ids_list += list(range(lst_id + ll, lst_id + ll + len(past_tokens[ll])))

    if guess_tokens is not None:
        input_ids = torch.cat((input_ids, torch.tensor(all_past + guess_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        guess_ids = list(range(lst_id + 1, lst_id + 1 + guess_size)) * (len(guess_tokens) // guess_size)
        position_ids = torch.cat((position_ids, torch.tensor(ids_list + guess_ids, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones(1, attn_size + len(guess_tokens), \
                device=input_ids.device, dtype=input_ids.dtype)), dim=1)
    
    else:
    #print("original size: ", input_ids.size(), position_ids.size(), attention_mask.size())
        input_ids = torch.cat((input_ids, torch.tensor(all_past, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        position_ids = torch.cat((position_ids, torch.tensor(ids_list, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones(1, attn_size, \
                device=input_ids.device, dtype=input_ids.dtype)), dim=1)
        #print("input new: ", input_ids.size(), attention_mask.size(), position_ids.size(), attn_size, past_key_values[0][0].size(2))
    step_len = attention_mask.size(1)
    
    #assert attention_mask.all().item()
    #attention_mask = None 
    #print("setting: is_prefill", past_tokens[1] is None)
    outputs = self.model.LlamaModeljforward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        WINDOWS_SIZE=WINDOWS_SIZE,
        is_prefill=past_tokens[1] is None,
        level_sizes=level_sizes,
        guess_size=guess_size,
        not_seq=not_seq,
        guess=guess_tokens
    )
    #print("done fwd")
    hidden_states = outputs[0]

    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    
    
    logits = logits.float()

    loss = None
    if labels is not None: #train
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    ret = CausalLMOutputWithPast(
        loss=loss,
        logits=logits.to(input_ids.device),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    ret.kvcache_len = prefill_size + past_size
    ret.step_len = step_len

    if guess_tokens is not None:
        lguess = len(guess_tokens)
    else:
        lguess = 0
    
    ret.out_logits = ret.logits[:,prefill_size - 1,:].to(input_ids.device)
    assert fill_level != -1
    if lguess > 0:
        ret.inp_logits = ret.logits[:,-len(past_tokens[fill_level])-lguess:-lguess,:].to(input_ids.device)
        ret.guess_logits = ret.logits[:,-lguess:,:].to(input_ids.device)
    else:
        ret.inp_logits = ret.logits[:,-len(past_tokens[fill_level]):,:].to(input_ids.device)

    return ret

