from intersuit_av.model.builder import load_pretrained_model
from intersuit_av.mm_utils import tokenizer_image_token, tokenizer_image_speech_token, process_images
from intersuit_av.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch, torchaudio
import numpy as np
# fix seed
torch.manual_seed(0)
import os

import ChatTTS
chat = ChatTTS.Chat()
chat.load(source='local', compile=True)
from num2words import num2words
import re

import whisper

def clean_text(text):
    def replace(match):
        num = match.group(0)
        return num2words(num)
    text = text.replace("-", "to").replace("\n", " ")
    return re.sub(r'\b\d+\b', replace, text)

from base import ViLLMBaseModel

class InterSuitAV(ViLLMBaseModel):
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
        
        
    
    def generate(self, instruction, video_path, gen=True, vocab=""):

        gen_kwargs = {"do_sample": True, "temperature": 1.0, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        max_frames_num = 16 # you can change this to several thousands so long you GPU memory can handle it :)
        # max_frames_num = 8
        
        frames = self.load_video(video_path, max_frames_num)
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.model.device)
        
        # # prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        # if vocab:
        #     prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n<speech>\n{vocab}<|im_end|>\n<|im_start|>assistant\n"
        # else:
        #     prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n<speech>\n<|im_end|>\n<|im_start|>assistant\n"

        # prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n<speech>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        # prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n<speech>\n<|im_end|>\n<|im_start|>assistant\n"
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image><|im_end|>\n<|im_start|>user\n<speech>\n<|im_end|>\n<|im_start|>assistant\n"
        
        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        input_ids = tokenizer_image_speech_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        
        # process speech
        audio_path = "./wav/" + video_path.split("/")[-1]+".wav"
        # if os.path.exists(audio_path): os.remove(audio_path) # init
        if not os.path.exists(audio_path):
            wav = chat.infer(clean_text(instruction))
            try:
                torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
        # print(instruction)
        # wav = chat.infer(clean_text(instruction))
        # try:
        #     torchaudio.save(audio_path, torch.from_numpy(wav[0]).unsqueeze(0), 24000)
        # except:
        #     torchaudio.save(audio_path, torch.from_numpy(wav[0]), 24000)
        speech = whisper.load_audio(audio_path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).to(device=self.model.device, dtype=torch.float16)
        speech_length = torch.LongTensor([speech.shape[0]]).to(self.model.device)
        # print(speech_length)
        # print(speech.shape)
        # print(audio_path)
        # os.remove(audio_path)
        
        if gen:
            # generate:
            # vr = VideoReader(video_path, ctx=cpu(0))
            # total_frame_num = len(vr)
            # uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            # frame_idx = uniform_sampled_frames.tolist()
            # frames = vr.get_batch(frame_idx).asnumpy()
            # video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=[video_tensor],  modalities=["video"], speeches=speech.unsqueeze(0), speech_lengths=speech_length, **gen_kwargs)
            
            
            
            # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs = self.tokenizer.batch_decode(output_ids)[0].strip()
            # print("*"*20)
            # print("this is audio: ", outputs)
        else:
            # thres
            for idx in range(gen_kwargs["max_new_tokens"]):
                outputs = self.model(
                    input_ids,
                    images=[video_tensor],
                    modalities=["video"],
                    return_dict=True,
                    speeches=speech.unsqueeze(0),
                    speech_lengths=speech_length
                )
                logits = outputs.logits
                # temperature = 0.05
                temperature = 1
                nan_mask = torch.isnan(logits[:,:,0])
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                # print("nan indices: ", nan_indices.shape)
                assert nan_indices.shape[0] == 0
                posterior_prob = torch.softmax(logits[:,-1]/temperature, dim=-1)
                noise_prob = posterior_prob[:,151644]
                # print("noise prob: ", noise_prob)
                # print(posterior_prob<0)
                posterior_entropy = -torch.sum(posterior_prob*torch.log(posterior_prob+1e-5), dim=-1)
                # print("posterior entropy: ", posterior_entropy)
                # posterior_threshold = 0.09 # hard threshold
                # posterior_alpha = 0.000001 # 0.3 # 0.2
                posterior_alpha = 1.0
                # threshold = torch.minimum(
                #     torch.ones_like(posterior_entropy) * posterior_threshold,
                #     torch.exp(-posterior_entropy) * posterior_alpha
                # )
                threshold  = torch.exp(-posterior_entropy) * posterior_alpha
                # threshold = 0.8
                noise_prob = posterior_prob[:,151644]
                print("threshold: ", threshold)
                print("noise prob: ", noise_prob)
                # print("noise probability: ", noise_prob)
                # print("threshold probability: ", threshold)
                if noise_prob > threshold:
                    return ""
                else:
                    # return "GEN"
                    logits[:, -1, 151644] = 0
                    next_token_ids = logits[:, -1:].argmax(-1)
                    # Append the predicted token to the sequence
                    input_ids = torch.cat([input_ids, next_token_ids], dim=-1)
                    if idx == 0:
                        output_ids = next_token_ids
                    else:
                        output_ids = torch.cat([output_ids, next_token_ids], dim=-1)
                    # Check for end-of-sequence token (EOS) to terminate generation
                    # print(next_token_ids)
                    if next_token_ids.squeeze(0).item() == self.tokenizer.eos_token_id:
                        break

        outputs = self.tokenizer.batch_decode(output_ids)[0].strip()
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