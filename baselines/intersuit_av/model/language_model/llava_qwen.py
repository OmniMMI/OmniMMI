#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from intersuit_av.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)

from torch.nn.attention.flex_attention import create_mask


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        speeches: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[List[List[int]]] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            # (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, speeches, speech_lengths)
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal_av(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, speeches, speech_lengths)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        speeches: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_av(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, speeches=speeches, speech_lengths=speech_lengths)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
    
    @torch.no_grad()
    def generate_parallel(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        streamer = kwargs.pop("streamer", None)
        tokenizer = kwargs.pop("tokenizer", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        new_query = kwargs.pop("new_query", None)
        if new_query is not None:
            new_inputs_embeds = self.get_model().embed_tokens(new_query)
        else:
            new_inputs_embeds = None
        bsz, seq = inputs_embeds.shape[:-1]

        # get channel ids 10: system; 14: 13: system+template
        prefix_length = 13 + 32 * 144 + 1 # system + video
        # channel = [0] * prefix_length + [1] * (seq - prefix_length)
        
        
        past_key_values = DynamicCache()
        max_cache_length = past_key_values.get_max_length()
        cache_position = torch.arange(seq, dtype=torch.int64, device=inputs_embeds.device)
        max_new_tokens = kwargs.pop("max_new_tokens", 64)
        for idx in range(max_new_tokens):
            
            # discern noise
            if idx == 4:
                # # encounter new query 
                # # construct new attention masks by flexattention
                # def prefix_mask(b, h, q_idx, kv_idx):
                #     return kv_idx <= prefix_length
                # def causal_mask(b, h, q_idx, kv_idx):
                #     return q_idx >= kv_idx
                # def transition_attention_mask(b, h, q_idx, kv_idx):
                #     prefix_mask = kv_idx <= prefix_length
                #     causal_mask = q_idx >= kv_idx
                #     block_mask = channel[q_idx] == channel[kv_idx]
                #     return prefix_mask | (causal_mask & block_mask)
                # new_attention_mask = create_mask(transition_attention_mask, 1, 1, bsz, seq)
                outputs_ids = torch.cat([outputs_ids, new_query], dim=-1)
                
                # print(new_inputs_embeds.shape)
                attention_mask[:, prefix_length:] = 0 # mask prev attention
                cur_length = attention_mask.shape[1] - prefix_length
                # print(attention_mask.shape)
                attention_mask = torch.cat([attention_mask[:,:-1], attention_mask.new_ones((new_inputs_embeds.shape[0], new_inputs_embeds.shape[1]))], dim=-1)
                cache_position = torch.arange(cache_position[-1], cache_position[-1]+new_inputs_embeds.shape[1], dtype=torch.int64, device=inputs_embeds.device)
                
                outputs = super().forward(
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    inputs_embeds=new_inputs_embeds,
                    use_cache=True,
                )
                
                
                
                temperature = 1
                next_prob = torch.softmax(outputs.logits[:,-1]/temperature, dim=-1)
                next_entropy = -torch.sum(next_prob*torch.log(next_prob+1e-5), dim=-1)
                next_threshold = 0.09
                next_alpha = 0.3
                noise_threshold = torch.minimum(
                    torch.ones_like(next_entropy) * next_threshold,
                    torch.exp(-next_entropy) * next_alpha
                )
                noise_prob = next_prob[:, 151644] # |<im_start>|
                # HACK batchsize must be 1
                if noise_prob[0] > noise_threshold[0]:
                    # if noise: continue generate, give up the query, revert attention
                    attention_mask[:, prefix_length:prefix_length+cur_length] = 1 # revert prev attention
                    attention_mask[:, prefix_length+cur_length:0] = 0 # mask noise attention
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                    cache_position = cache_position[-1:] + 1
                else:
                    # if not noise: change channel
                    print("*******start new topic*******")
                    next_token_ids = outputs.logits[:, -1:].argmax(-1)
                    outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                    if tokenizer is not None:
                        print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                    if streamer is not None:
                        streamer.stream(output_ids=next_token_ids)
                    inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                    cache_position = cache_position[-1:] + 1
                    
            
            else:
                outputs = super().forward(
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
                
                print("*"*50)
                print(past_key_values.key_cache[-1].shape)
                print("*"*50)
                
                # greedy sample
                next_token_ids = outputs.logits[:, -1:].argmax(-1)
                if idx == 0:
                    outputs_ids = next_token_ids
                else:
                    outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
                if tokenizer is not None:
                    print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
                if streamer is not None:
                    streamer.stream(output_ids=next_token_ids)
                inputs_embeds = self.get_model().embed_tokens(next_token_ids)
                # print(inputs_embeds.shape)
                # inputs_embeds = torch.cat([inputs_embeds, next_inputs_embeds], dim=1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
                cache_position = cache_position[-1:] + 1
                # cache_position = torch.cat([cache_position, cache_position.new_])
                
            if next_token_ids.squeeze(0)[0].item() == tokenizer.eos_token_id:
                print("="*20,"COMPLETE","="*20)
                break

        return outputs_ids

    @torch.no_grad()
    def generate_streaming(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        streamer = kwargs.pop("streamer", None)
        tokenizer = kwargs.pop("tokenizer", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        max_new_tokens = kwargs.pop("max_new_tokens", 64)
        for idx in range(max_new_tokens):
            
            outputs = super().forward(
                # attention_mask=attention_mask,
                past_key_values=past_key_values,
                # cache_position=cache_position,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            
            # greedy sample
            next_token_ids = outputs.logits[:, -1:].argmax(-1)
            if idx == 0:
                outputs_ids = next_token_ids
            else:
                outputs_ids = torch.cat([outputs_ids, next_token_ids], dim=-1)
            # if tokenizer is not None:
            #     print(tokenizer.decode(next_token_ids.squeeze(0), skip_special_tokens=True), end='', flush=True)
            # if streamer is not None:
            #     streamer.stream(output_ids=next_token_ids)
            inputs_embeds = self.get_model().embed_tokens(next_token_ids)
            # print(inputs_embeds.shape)
            # inputs_embeds = torch.cat([inputs_embeds, next_inputs_embeds], dim=1)
            # attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
            # cache_position = cache_position[-1:] + 1
            # cache_position = torch.cat([cache_position, cache_position.new_])
            
            if next_token_ids.squeeze(0)[0].item() == tokenizer.eos_token_id:
                # print("="*20,"COMPLETE","="*20)
                break

        return outputs_ids, past_key_values


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
