import os
import torch, torchvision, transformers, collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# torchvision.set_video_backend('video_reader')
from dataclasses import asdict
# from torchvision.io import read_video
from decord import VideoReader, cpu
import torchaudio

from transformers.cache_utils import (
    DynamicCache,
    OffloadedCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)


logger = transformers.logging.get_logger('liveinfer')

from intersuit_av.conversation import conv_templates, SeparatorStyle
from intersuit_av.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from intersuit_av.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, tokenizer_image_speech_token, KeywordsStoppingCriteria
from intersuit_av.vid_utils import load_video
from intersuit_av.model.builder import load_pretrained_model
from intersuit_av.arguments_live import parse_args
from intersuit_av.inference_util import MaxHeapDict

import ChatTTS
chat = ChatTTS.Chat()
chat.load(compile=True)
from num2words import num2words
import re
def clean_text(text):
    def replace(match):
        num = match.group(0)
        return num2words(num)
    text = text.replace("-", "to").replace("\n", " ")
    return re.sub(r'\b\d+\b', replace, text)
import whisper


class InterSuitOnlineAV:
    def __init__(self, model_args) -> None:
        args = parse_args()
        
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        model_path = model_args['model_path']
        device = model_args['device']
        self.device = f"cuda:{device}"
        
        model_name = get_model_name_from_path(model_path)
        self.model_name = model_name
        llava_model_args = {"multimodal": True}
        if args.attn_implementation is not None:
            llava_model_args["attn_implementation"] = args.attn_implementation
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "average"
        llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(model_path, None, model_name, device_map=self.device, **llava_model_args)
        if 'qwen' in model_path.lower():
            conv_mode = 'qwen_1_5'
        elif 'llama3' in model_path.lower():
            conv_mode = 'llava_llama_3'
        elif 'mistral' in model_path.lower():
            conv_mode = 'mistral_instruct'
        else:
            conv_mode = "llava_v1"
        # if args.conv_mode is not None and conv_mode != args.conv_mode:
        #     print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        # else:
        #     args.conv_mode = conv_mode
        args.conv_mode = conv_mode
        self.conv_mode = conv_mode
        self.conv = conv_templates[args.conv_mode].copy()
        
        
        
        
        # visual
        # self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = args.frame_resolution
        # self.frame_num_tokens = self.model.config.frame_num_tokens
        # self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        # self.frame_token_interval_id = self.model.config.frame_token_interval_id
        # self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(self.model.config.frame_num_tokens).reshape(1,-1)
        
        # generation
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device=self.device, dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.tokenizer.eos_token_id
        # self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        # self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to(self.device)
        # self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to(self.device)
        
        # app
        self.reset()
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @torch.no_grad()
    def _call_for_response(self, video_time, query):
        if query is not None:
            # if self.current_output:
            #     self.conv.messages[-1][-1] = self.current_output
            #     self.current_output = None
            self.conv.append_message(self.conv.roles[0], '\n'+query)
            self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        # print(prompt)
        # inputs_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)
        inputs_ids = tokenizer_image_speech_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.model.device)
        # TODO: support batch inference
        # input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
        # pad_token_ids = (
        #     self.tokenizer.pad_token_id
        #     if self.tokenizer.pad_token_id is not None
        #     else self.tokenizer.eos_token_id
        # )
        # input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_ids).to(self.device)
        # attention_masks = input_ids.ne(pad_token_ids).to(self.device)
        # stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # TODO: support interleaved multi-turn
        # split input query: 
        # single turn: ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', None, 'query', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ']
        # mutlip turn: ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', None, 'Ċ', 'query', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ', 'yes', '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', 'query2', '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ']
        # print(inputs_ids)
        # print("##before: ", self.tokenizer.decode(inputs_ids))
        inputs_ids = inputs_ids[self.vis_index:]
        # print("##after: ", self.tokenizer.decode(inputs_ids))
        
        inputs_embeds = self.model.get_model().embed_tokens(inputs_ids.unsqueeze(0))
        output_ids, self.past_key_values = self.model.generate_streaming(
            inputs_embeds,
            past_key_values=self.past_key_values,
            tokenizer=self.tokenizer
        )
        
        post_length = inputs_embeds.shape[-2] + output_ids.shape[-1] - 1
        self.past_key_values.crop(self.past_key_values.get_seq_length() - post_length)
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        self.current_output = output_text
        
        self.query_inputs_ids = inputs_ids
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{output_text}'
        return query, response
    
    @torch.no_grad()
    def _call_for_streaming(self, ):
        while self.frame_embeds_queue:
            # 1. if query is before next frame, input query
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                query = query[-1]
                # if self.current_output is not None:
                #     self.conv.messages[-1][-1] = self.current_output
                #     self.current_output = None
                self.conv.append_message(self.conv.roles[0], '\n'+query)
                self.conv.append_message(self.conv.roles[1], None)
                # return video_time, query
            
            video_time, frame_embeds = self.frame_embeds_queue.popleft() # frame_embeds, N_patch, H
            
            if self.past_key_values is None: # initialize
                # prefix_prompt
                # qwen -> CHATML
                prefix_prompt = "" if self.conv.system == "" else self.conv.system + self.conv.sep + "\n"
                prefix_prompt += self.conv.roles[0] + "\n"
                prefix_inputs_ids = self.tokenizer(prefix_prompt).input_ids
                prefix_inputs_ids = torch.tensor(prefix_inputs_ids, dtype=torch.long).to(self.device)
                self.vis_index = prefix_inputs_ids.shape[-1]
                # print("prefix: ")
                # print(self.tokenizer.decode(prefix_inputs_ids))
                # post_prompt + query: "\n" + query + post_prompt
                speech_features, query = self.query_queue[-1][1]
                nl_input_ids = self.tokenizer("\n"+query).input_ids
                post_prompt_input_ids = self.tokenizer(self.conv.sep + "\n" + self.conv.roles[1] + "\n").input_ids
                self.nl_input_ids = torch.tensor(nl_input_ids, dtype=torch.long).to(self.device)
                self.post_prompt_input_ids = torch.tensor(post_prompt_input_ids, dtype=torch.long).to(self.device)
                self.query_inputs_embeds = torch.cat([
                    self.model.get_model().embed_tokens(self.nl_input_ids.unsqueeze(0)),
                    speech_features.unsqueeze(0),
                    self.model.get_model().embed_tokens(self.post_prompt_input_ids.unsqueeze(0))
                ],dim=1)
                self.past_key_values = DynamicCache()
                inputs_embeds = torch.cat([
                    self.model.get_model().embed_tokens(prefix_inputs_ids.unsqueeze(0)),
                    self.query_inputs_embeds
                ], dim=1)
            else:
                inputs_embeds = torch.cat([
                    frame_embeds.unsqueeze(0),
                    self.query_inputs_embeds
                ], dim=1)
            self.frame_count += 1
            # TODO interleaved multiple turn
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=self.past_key_values,
                output_attentions=True,
                return_dict=True
            )
            attentions = outputs.attentions[-1].cpu()
            del outputs
            self.past_key_values.crop(self.past_key_values.get_seq_length() - self.query_inputs_embeds.shape[-2])
            
            # 2. if the same time, input query after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                query = query[-1]
                if self.current_output is not None:
                    self.conv.messages[-1][-1] = self.current_output
                    self.current_output = None
                self.conv.append_message(self.conv.roles[0], '\n'+query)
                self.conv.append_message(self.conv.roles[1], None)
                # return video_time, query
            
            # 3. grounding 
            if self.frame_count > 2:
                attentions = attentions.squeeze(0)
                # print("nan shape: ", torch.nonzero(torch.isnan(attentions), as_tuple=False))
                non_nan_mask = ~torch.isnan(attentions)
                attentions = torch.where(non_nan_mask, attentions, torch.tensor(0.0))
                
                # attentions = attentions.mean(0)[-3, :]
                # ground_attention = attentions[self.vis_index:self.vis_index+144*self.frame_count]
                # ground_score = ground_attention.reshape(-1, 144).mean(dim=-1)
                attentions = attentions.mean(0)[-self.query_inputs_embeds.shape[-2]:, :]
                attentions = attentions[:, self.vis_index:self.vis_index+144*self.frame_count]
                attentions = attentions.reshape(self.query_inputs_embeds.shape[-2], -1, 144).mean(dim=-1)
                if "qwen" in self.model_name.lower():
                    ground_score = attentions[-3]
                elif "llama" in self.model_name.lower():
                    ground_score = attentions[-4]
                
                # # ablate
                # ground_score = attentions.mean(-1)
                
                
                
                # grounding algorithm: when looking forward, when a frame always larger than expectation + variance, we tag it as a hit 
                std, mean = torch.std_mean(ground_score)
                threshold = mean + 2*std
                salients = ground_score > threshold
                salients = salients.nonzero().squeeze(-1).tolist()
                # # topk
                # salients = torch.topk(ground_score, 1).indices.sort()[0].tolist()
                # # print(salients)
                
                for sa in salients:
                    self.salient.add_or_update(sa, -self.salient.entry_finder.get(sa, (0, -1, -1))[0]+1)
                if self.salient.heap:
                    sa, cnt = self.salient.peek_max()
                    # print(self.salient.entry_finder)
                    # print(sa, cnt)
                    if cnt > 4 and sa not in self.highlight_points: # forward step
                        self.highlight_points.append(sa)
                        return sa/self.frame_fps, None
        
        return None, None
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.salient = MaxHeapDict()
        self.highlight_points = collections.deque()
        self.conv = conv_templates[self.conv_mode].copy()
        self.current_output = None
        self.frame_count = 0
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None

    def input_query_stream(self, query, history=None, video_time=None):
        
        # encode query
        audio_path = self.video_path.split("/")[-1]+".wav"
        if not os.path.exists(audio_path):
            wav = chat.infer(query)
            try:
                torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
        speech = whisper.load_audio(audio_path)
        speech = whisper.pad_or_trim(speech)
        # speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).to(device=self.model.device, dtype=torch.float16)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).to(device=self.model.device).to(torch.bfloat16)
        speech_length = torch.LongTensor([speech.shape[0]]).to(self.model.device)
        os.remove(audio_path)
        
        # # # encode speech
        # speech_features = self.model.encode_speech(speech.unsqueeze(0), speech_length).to(torch.bfloat16)
        speech = speech.unsqueeze(0)
        speech_lengths = speech_length
        speech_encoder_type = self.model.config.speech_encoder_type
        speech_encoder = self.model.get_speech_encoder()
        speech_encoder = speech_encoder.to(self.model.dtype)
        if "whisper" in speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f'Unknown speech encoder: {speech_encoder}')
        speech_projector_type = self.model.config.speech_projector_type
        speech_projector = self.model.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        
        
        if video_time is None:
            self.query_queue.append((self.video_time, (speech_features[0], query)))
        else:
            self.query_queue.append((video_time, (speech_features[0], query)))
        
        
        # if video_time is None:
        #     self.query_queue.append((self.video_time, query))
        # else:
        #     self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            # print('frame_idx ', frame_idx)
            # print('last_frame_idx ', self.last_frame_idx)
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            
            # encode video
            video_idx_in_batch = [0]
            # small trick to save GPU Mem
            images_list = [self.video_tensor[ranger].to(self.device)] # self.video_tensor: L, H, W, C
            concat_images = torch.cat([image for image in images_list], dim=0) # L, H, W, C
            split_sizes = [image.shape[0] for image in images_list]
            frames_embeds = self.model.encode_multimodals(concat_images, video_idx_in_batch, split_sizes) #[ (L, N, H)
            # unires + video = flat
            # frames_embeds = [x.flatten(0, 1) for x in image_features] # [L*N, H]
            # TODO unires image
            
            frames_embeds = frames_embeds[0] # L, N, H
            self.frame_embeds_queue.extend([(r/self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)]) # 
            
            
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    
    def load_video(self, video_path):
        if os.path.isdir(video_path):
            video_tensor = load_video(video_path, video_decode_backend='frame', fps=self.frame_fps)
        elif video_path.endswith(".gif"):
            video_tensor = load_video(video_path, video_decode_backend='gif', fps=self.frame_fps)
        else:
            # video_tensor = load_video(video_path, fps=self.frame_fps, max_frames=512) # T H W C
            video_tensor = load_video(video_path, fps=self.frame_fps) # T H W C
        if "qwen" in self.model_name.lower():
            self.video_tensor = self.processor.preprocess(video_tensor, return_tensors="pt")['pixel_values'].to(torch.bfloat16)
        else:
            self.video_tensor = self.processor.preprocess(video_tensor, return_tensors="pt")['pixel_values'].half()
        
        # print(self.video_tensor.shape)
        self.num_video_frames = self.video_tensor.shape[0]
        self.video_duration = self.video_tensor.shape[0] / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

        self.video_path = video_path

    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            # query, response = self._call_for_response(video_time, query)
            # HACK: for evaluation
            query, response = query, "TEMPLATE"
        return query, response, video_time
    