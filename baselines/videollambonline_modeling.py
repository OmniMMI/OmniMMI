import os, subprocess
import numpy as np
import torch, torchvision, transformers, collections
from dataclasses import asdict


logger = transformers.logging.get_logger('liveinfer')


from videollamb.conversation import conv_templates, SeparatorStyle
from videollamb.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from videollamb.utils import disable_torch_init
from videollamb.mm_utils import get_model_name_from_path, tokenizer_x_token, KeywordsStoppingCriteria
from videollamb.vid_utils import load_video
from videollamb.model.builder import load_pretrained_model
from videollamb.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollamb.train.train import smart_tokenizer_and_embedding_resize
from videollamb.model.multimodal_projector.self_segment import segment, segment_left
from videollamb.arguments_live import parse_args

class VideoLLaMBOnline:
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
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(model_path, None, model_name, args.num_frames, args.load_8bit, args.load_4bit, device=self.device)
        if 'mistral' in model_path:
            conv_mode = 'mistral_instruct'
        else:
            conv_mode = "llava_v1"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
        self.conv = conv_templates[args.conv_mode].copy()
        
        
        # self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        # self.model.to('cuda')

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
        self.inplace_output_ids = torch.zeros(1, 100, device='cuda', dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        # self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        # self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')
        
        # app
        self.reset()

    def _call_for_response(self, video_time, query):
        if query is not None and len(self.conv.messages) == 0:
            self.conv.append_message(self.conv.roles[0], DEFAULT_X_TOKEN['VIDEO'] + '\n' + query)
            self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        # print("*"*20)
        # print(prompt)
        # print("*"*20)
        self.last_ids = tokenizer_x_token(prompt, self.tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to('cuda')
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.last_ids)
        # print('frame_queue ', list(self.frame_embeds_queue))
        # get all embedding
        all_frame_tensors = torch.stack([x[1] for x in self.all_frame_tensors_queue]) # TCHW
        
        # languagebind
        num_select = max(8, all_frame_tensors.shape[0]-all_frame_tensors.shape[0]%8)
        selected_idx = np.linspace(0, all_frame_tensors.shape[0]-1, num_select, dtype=int)
        all_frame_tensors = all_frame_tensors[selected_idx, :, :, :]
        
        # print('all tensor ', all_frame_tensors.shape)
        output_ids = self.model.generate(
            self.last_ids,
            X=[all_frame_tensors.permute(1,0,2,3)], # languagebind
            # X=[all_frame_tensors],
            X_modalities=["VIDEO"],
            X_sizes=[None],
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True
        )
        self.past_key_values = None
        self.last_ids = None
        query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()}'
        return query, response
    
    def _call_for_streaming(self, ):
        
        while self.frame_embeds_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            # 3. --> if there is left segment index, then response
            cls_embeds = torch.stack([x[1] for x in self.cls_embeds_queue])
            # print(cls_embeds.shape)
            boundaries = segment(cls_embeds)
            if boundaries[-1] not in self.boundaries and len(boundaries) > 2 and video_time - boundaries[-1] < 3 and boundaries[-1] - self.boundaries[-1] > 1:
                for bd in boundaries:
                    if bd not in self.boundaries:
                        self.boundaries.append(bd)
                return video_time, None
            
        return None, None

        
    
    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.cls_embeds_queue = collections.deque()
        self.all_frame_tensors_queue = collections.deque()
        self.boundaries = [0]
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            # print('frame_idx ', frame_idx)
            # print('last_frame_idx ', self.last_frame_idx)
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            
            # languagebind
            num_select = max(8, len(ranger)-len(ranger)%8)
            new_ranger = np.linspace(0, len(ranger)-1, num_select, dtype=int)
            # frames_embeds = self.model.encode_video_features(self.video_tensor[new_ranger].permute(1, 0, 2, 3).unsqueeze(0)) # B, L, N, D
            frames_embeds = self.model.encode_image_features(self.video_tensor[ranger].unsqueeze(0)) # B, L, N, D
            self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embed) for r, frame_embed in zip(ranger, frames_embeds.squeeze(0))])
            # print(frames_embeds.shape)
            cls_embeds = frames_embeds[:, :, 0, :] # B, L, D
            self.cls_embeds_queue.extend([(r / self.frame_fps, cls_embed) for r, cls_embed in zip(ranger, cls_embeds.squeeze(0))])
            self.all_frame_tensors_queue.extend([(r/ self.frame_fps, video_tensor) for r, video_tensor in zip(ranger, self.video_tensor[ranger])])
            
        self.last_frame_idx = frame_idx
        self.video_time = video_time

        
    def load_video(self, video_path):
        # LanguageBind
        self.video_tensor = self.processor["VIDEO"](video_path, return_tensors="pt", fps=self.frame_fps)["pixel_values"][0].half().to(self.device).permute(1, 0, 2, 3) # CTHW -> TCHW
        
        # print(self.video_tensor.shape)
        self.num_video_frames = self.video_tensor.shape[0]
        self.video_duration = self.video_tensor.shape[0] / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')
        



    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        return query, response, video_time
    