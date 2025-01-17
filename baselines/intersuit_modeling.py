from intersuit.model.builder import load_pretrained_model
from intersuit.mm_utils import tokenizer_image_token, process_images
from intersuit.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)

from base import ViLLMBaseModel

class InterSuit(ViLLMBaseModel):
    def __init__(self, model_args) -> None:
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        # init model
        
        device = model_args['device']
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_args['model_path'],
            None,
            "llava_qwen",
            device_map=f"cuda:{device}"
        )
        self.model = self.model.to(f'cuda:{device}')
        self.model_args = model_args
        
        
    
    def generate(self, instruction, video_path, gen=True):

        gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
        
        
        frames = self.load_video(video_path, max_frames_num)
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.model.device)
        
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        
            
        if gen:
            # generate:
            # vr = VideoReader(video_path, ctx=cpu(0))
            # total_frame_num = len(vr)
            # uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            # frame_idx = uniform_sampled_frames.tolist()
            # frames = vr.get_batch(frame_idx).asnumpy()
            # video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        else:
            # thres
            outputs = self.model(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                return_dict=True
            )
            logits = outputs.logits
            temperature = 0.05
            nan_mask = torch.isnan(logits[:,:,0])
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            # print("nan indices: ", nan_indices.shape)
            assert nan_indices.shape[0] == 0
            posterior_prob = torch.softmax(logits[:,-1]/temperature, dim=-1)
            # print(posterior_prob<0)
            posterior_entropy = -torch.sum(posterior_prob*torch.log(posterior_prob+1e-5), dim=-1)
            # print("posterior entropy: ", posterior_entropy)
            # posterior_threshold = 0.09 # hard threshold
            posterior_alpha = 0.000001 # 0.3 # 0.2
            # threshold = torch.minimum(
            #     torch.ones_like(posterior_entropy) * posterior_threshold,
            #     torch.exp(-posterior_entropy) * posterior_alpha
            # )
            threshold  = torch.exp(-posterior_entropy) * posterior_alpha
            noise_prob = posterior_prob[:,151644] # qwen
            # print("noise probability: ", noise_prob)
            # print("threshold probability: ", threshold)
            if noise_prob > threshold:
                return ""
            else:
                return "GEN"
            
            
        return outputs
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        print(f"spare_frames: {spare_frames.shape}")
        return spare_frames  # (frames, height, width, channels)