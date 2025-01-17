import os, shutil, cv2
import requests
from PIL import Image
from io import BytesIO

import torch
from transformers import TextStreamer

from videollamb.conversation import conv_templates, SeparatorStyle
from videollamb.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from videollamb.mm_utils import get_model_name_from_path, tokenizer_x_token, KeywordsStoppingCriteria
from videollamb.vid_utils import load_video
from videollamb.model.builder import load_pretrained_model
from videollamb.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollamb.train.train import smart_tokenizer_and_embedding_resize



from base import ViLLMBaseModel

class VideoLLaMB(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        self.model_path = model_args['model_path']
        self.device = model_args['device']
        cache_dir = 'cache_dir'

        load_4bit, load_8bit = False, False
        self.num_frames = 32
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, processor, _ = load_pretrained_model(self.model_path, None, model_name, self.num_frames)
        self.model = self.model.to(self.device)
        self.video_processor = processor["VIDEO"]
        if 'mistral' in self.model_path:
            self.conv_mode = "mistral"
        else:
            self.conv_mode = "llava_v1"
        
    def generate(self, instruction, video_path):
        
        
        # # clip
        # video = load_video(video_path, fps=1)
        # video_tensor = self.video_processor(video, return_tensors="pt")["pixel_values"].half().to(self.device)
        
        # language bind
        video_tensor = self.video_processor(video_path, fps=1, return_tensors="pt")["pixel_values"][0].half().to(self.device)
        
        # video_tensor = self.video_processor(video_path, fps=1, return_tensors="pt")["pixel_values"][0].half().to(self.device)
        # if type(video_tensor) is list:
        #     tensor = [video.to(self.model.device, dtype=torch.float16) for video in video_tensor]
        # else:
        #     tensor = video_tensor.to(self.model.device, dtype=torch.float16)

        if self.model.config.mm_use_x_start_end:
            instruction = DEFAULT_X_START_TOKEN['VIDEO'] + DEFAULT_X_TOKEN['VIDEO'] + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + instruction
        else:
            instruction = DEFAULT_X_TOKEN['VIDEO'] + '\n' + instruction
        
        # instruction = DEFAULT_X_TOKEN['IMAGE'] + '\n' + instruction
        
        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_x_token(prompt, self.tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                X=[video_tensor],
                X_modalities=["VIDEO"],
                X_sizes=[None],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                # streamer=streamer,
                use_cache=True,
                cache_position=None,
                stopping_criteria=[stopping_criteria])


        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        return outputs
