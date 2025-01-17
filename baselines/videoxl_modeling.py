from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)


from base import ViLLMBaseModel


class VideoXL(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        model_path = model_args["model_path"]
        device = "cuda:"+str(model_args["device"])

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map=device)
        self.model.config.beacon_ration = [8]
        
    def generate(self, instruction, video_path):
        
        max_frames_num = 128
        gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        
        instruction = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(instruction, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num-1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)

        beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
        num_tokens=TOKEN_PERFRAME *max_frames_num
        beacon_skip_last = beacon_skip_first  + num_tokens

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

        if IMAGE_TOKEN_INDEX in input_ids:
            transform_input_ids=transform_input_id(input_ids,num_tokens,self.model.config.vocab_size-1)

        output_ids=output_ids[:,transform_input_ids.shape[1]:]
        if output_ids.tolist() == []: return ""
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

