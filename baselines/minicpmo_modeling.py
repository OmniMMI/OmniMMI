import torch
import torchaudio
from PIL import Image
from transformers import AutoModel, AutoTokenizer

import os
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf
from decord import VideoReader, cpu

import ChatTTS
chat = ChatTTS.Chat()
chat.load(source='local', compile=True)
from num2words import num2words
import re

def clean_text(text):
    def replace(match):
        num = match.group(0)
        return num2words(num)
    text = text.replace("-", "to").replace("\n", " ")
    return re.sub(r'\b\d+\b', replace, text)

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

from base import ViLLMBaseModel

class MiniCPMO(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        # load omni model default, the default init_vision/init_audio/init_tts is True
        # if load vision-only model, please set init_audio=False and init_tts=False
        # if load audio-only model, please set init_vision=False
        model = AutoModel.from_pretrained(
            model_args["model_path"],
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )


        self.model = model.eval().cuda(model_args["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_args["model_path"], trust_remote_code=True)

        # In addition to vision-only mode, tts processor and vocos also needs to be initialized
        self.model.init_tts()

        
    def generate(self, instruction, video_path):
        
        # # process speech
        # audio_path = "./wav/" + video_path.split("/")[-1]+".wav"
        # if os.path.exists(audio_path): os.remove(audio_path) # init
        # if not os.path.exists(audio_path):
        #     wav = chat.infer(clean_text(instruction))[0]
        #     try:
        #         torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
        #     except:
        #         torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
        # audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # sys_msg = self.model.get_sys_prompt(mode='omni', language='en')
        # # or use default prompt
        # # sys_msg = model.get_sys_prompt(mode='omni', language='en')
        
        # contents = get_video_chunk_content(video_path)
        # contents.append(audio_input)
        # msg = {"role":"user", "content": contents}
        # msgs = [sys_msg, msg]
        
        # # please set generate_audio=True and output_audio_path to save the tts result
        # generate_audio = False
        # output_audio_path = 'output.wav'

        # res = self.model.chat(
        #     msgs=msgs,
        #     tokenizer=self.tokenizer,
        #     sampling=True,
        #     temperature=0.5,
        #     max_new_tokens=4096,
        #     omni_input=True, # please set omni_input=True when omni inference
        #     use_tts_template=True,
        #     generate_audio=generate_audio,
        #     output_audio_path=output_audio_path,
        #     max_slice_nums=1,
        #     use_image_id=False,
        #     return_dict=True,
        #     # max_inp_length=4096, #### prevent size mismatch #####
        # )
        # print(res.text)
        # answer = res.text
        
        
        # process speech
        audio_path = "./wav/" + video_path.split("/")[-1]+".wav"
        if os.path.exists(audio_path): os.remove(audio_path) # init
        if not os.path.exists(audio_path):
            wav = chat.infer(clean_text(instruction))[0]
            try:
                torchaudio.save(audio_path, torch.from_numpy(wav).unsqueeze(0), 24000)
            except:
                torchaudio.save(audio_path, torch.from_numpy(wav), 24000)
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        # encode frames
        frames = encode_video(video_path)
        # question = "Describe the video"
        msgs = [
            {'role': 'user', 'content': frames + [audio_input]}, 
        ]
        # Set decode params for video
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448
        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        
        return answer

# streaming generation mode


def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path).resize(0.5)
    print('video_duration:', video.duration)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    
    return contents