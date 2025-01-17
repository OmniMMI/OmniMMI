import argparse
import glob
import json
import math
import os
import os.path as osp
import shutil
import signal

import cv2
import numpy as np
import shortuuid
import torch
from filelock import FileLock
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Resize
from tqdm import tqdm

from vila.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vila.conversation import SeparatorStyle, conv_templates
from vila.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from vila.model.builder import load_pretrained_model
from vila.utils import disable_torch_init
from vila.mm_utils import opencv_extract_frames



from base import ViLLMBaseModel

class VILA(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        self.model_path = model_args['model_path']
        self.device = model_args['device']
        cache_dir = 'cache_dir'
        
        self.model_name = "VILA"

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.model_path, model_name, None)
        self.model = self.model.to(self.device)
        
        self.conv_mode = "llama_3"
        
    def generate(self, instruction, video_path):
        
        vid_cap = cv2.VideoCapture(video_path)
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        seconds = int(frame_count // fps)
        seconds = max(1, min(seconds, 128))
        imgs, num_frames = opencv_extract_frames(video_path, frames=seconds)
        num_frames_loaded = len(imgs)
        # print(num_frames_loaded)
        
        image_tensor = [
            # processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)
            self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            for image in imgs
        ]
        image_tensor = torch.stack(image_tensor)
        
        qs = "<image>\n" * num_frames_loaded + instruction
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        input_ids = torch.unsqueeze(input_ids, 0)
        input_ids = torch.as_tensor(input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        do_sample = True
        temperature = 0.2
        num_beams = 1
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=1024,
                num_beams=num_beams,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print(outputs)
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        return outputs
