from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)



from base import ViLLMBaseModel

class LongVA(ViLLMBaseModel):
    def __init__(self, model_args):
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
        
    def generate(self, instruction, video_path):

        gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
        
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs
