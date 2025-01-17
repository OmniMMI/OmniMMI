import argparse
import json
import sys
import os
import math
import numpy as np
import random
import uuid
from collections import defaultdict
from typing import Callable
from tqdm import tqdm

sys.path.append(os.getcwd())

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']


def load_model(TESTING_MODEL, device):
    if TESTING_MODEL == 'VideoChatGPT':
        from videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "VideoChat2":
        from videochat_modeling import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "VideoLLaVA":
        from videollava_modeling import VideoLLaVA
        ckpt_path = f"{CKPT_DIR}/VideoLLaVA/Video-LLaVA-7B"
        model = VideoLLaVA({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "LLaMA-VID":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
        model = LLaMAVID({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "LLaMA-VID-13B":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-13B"
        model = LLaMAVID({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "MiniGPT4-Video":
        from minigpt4video_modeling import MiniGPT4Video
        ckpt_path = f"{CKPT_DIR}/MiniGPT4-Video/checkpoints"
        model = MiniGPT4Video({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "PLLaVA":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
        model = PLLaVA({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "PLLaVA-13B":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-13b"
        model = PLLaVA({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "PLLaVA-34B":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-34b"
        model = PLLaVA({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "LLaVA-NeXT-Video":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-7B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "LLaVA-NeXT-Video-34B":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-34B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "ShareGPT4Video":
        from sharegpt4video_modeling import ShareGPT4Video
        ckpt_path = f"{CKPT_DIR}/ShareGPT4Video/sharegpt4video-8b"
        model = ShareGPT4Video({"model_path": ckpt_path, "device": device})
    elif TESTING_MODEL == "Gemini-1.5-pro":
        from gemini_modeling import Gemini
        model = Gemini({"model_path": None, "device": device})
    elif TESTING_MODEL == "GPT4O":
        from gpt4o_modeling import GPT4O
        model = GPT4O({"model_path": None, "device": device})
    elif TESTING_MODEL == "LongVA":
        from longva_modeling import LongVA
        ckpt_path = f"{CKPT_DIR}/LongVA-7B-Qwen2"
        model = LongVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LongVALlama":
        from longva_modeling import LongVA
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanext-llama31"
        model = LongVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LongVILA":
        from vila_modeling import VILA
        ckpt_path = f"{CKPT_DIR}/Llama-3-LongVILA-8B-1024Frames"
        model = VILA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LongLLaVA":
        from longllava_modeling import LongLLaVA
        ckpt_path = f"{CKPT_DIR}/LongLLaVA-9B"
        model = LongLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaMB":
        from videollamb_modeling import VideoLLaMB
        ckpt_path = f"{CKPT_DIR}/llava-7b-ft-rmtr1x-lvcn_16_4_pool12_new"
        model = VideoLLaMB({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoOnline":
        from videoonline_modeling import VideoOnline
        ckpt_path = f"{CKPT_DIR}/videollm-online-8b-v1plus"
        model = VideoOnline({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaMBOnline":
        from videollambonline_modeling import VideoLLaMBOnline
        ckpt_path = f"{CKPT_DIR}/llava-7b-ft-rmtr1x-lvcn_16_4_pool12_new"
        model = VideoLLaMBOnline({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuit":
        from intersuit_modeling import InterSuit
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-noise"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub20k-qwen2-RNS211"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-NRS211" # 26.5
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-ORNS1111" # 24
        model = InterSuit({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuitAV":
        from intersuit_av_modeling import InterSuitAV
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech" # 24
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns-0"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k-orns1111"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-NS-qwen2-speech-special"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-start"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-filter"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-filter-special-voiceassistant"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-start-filter"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-vst-vt-st-300k"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-voiceassistant-special"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech-cosyvoice"
        model = InterSuitAV({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuitOnline":
        from intersuitonline_modeling import InterSuitOnline
        # llama
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanext-llama31"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-llama31-rev"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-llama31-ONRS1111"
        
        # # qwen
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-rev"
        # ckpt_path = f"{CKPT_DIR}/LongVA-7B-Qwen2"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub30k-qwen2-RNS"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub40k-qwen2-ORNS"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub20k-qwen2-RNS211"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-ORNS1111" # 24
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-RNS211" # 20.5
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-NRS211" # 26.5
        model = InterSuitOnline({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuitOnlineAV":
        from intersuitonline_av_modeling import InterSuitOnlineAV
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns-0"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k-orns1111"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-NS-qwen2-speech-special"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-start"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-filter"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-filter-special-voiceassistant"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-start-filter"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-vst-vt-st-300k"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-voiceassistant-special"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech-cosyvoice"
        model = InterSuitOnlineAV({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuitOnline_Qwen":
        from intersuitonline_modeling import InterSuitOnline
        # llama
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanext-llama31"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-llama31-rev"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-llama31-ONRS1111"
        
        # # qwen
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-rev"
        ckpt_path = f"{CKPT_DIR}/LongVA-7B-Qwen2"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub30k-qwen2-RNS"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub40k-qwen2-ORNS"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub20k-qwen2-RNS211"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-ORNS1111" # 24
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-RNS211" # 20.5
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-NRS211" # 26.5
        model = InterSuitOnline({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoXL":
        from videoxl_modeling import VideoXL
        ckpt_path = f"{CKPT_DIR}/Video_XL/VideoXL_weight_8"
        model = VideoXL({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaMA2":
        from videollama2_modeling import VideoLLaMA2
        model = VideoLLaMA2({"model_path": f"{CKPT_DIR}/VideoLLaMA2.1-7B-AV", "device": 0})
    elif TESTING_MODEL == "InternLMXCO":
        from internlmxco_modeling import InternLMXCO
        ckpt_path = f"{CKPT_DIR}/internlm-xcomposer2d5-ol-7b"
        model = InternLMXCO({"model_path": ckpt_path, "device": 0})    

    return model



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                    default="", 
                    choices=["VideoChatGPT", "VideoChat2", "VideoLLaVA", "LLaMA-VID","MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video",
                                 "Gemini-1.5-pro", "GPT4O",
                                 "LongVA", "LongVALlama", "LongVILA", "LongLLaVA", "VideoLLaMB", "InterSuit", "VideoXL",
                                 "LLaMA-VID-13B", "PLLaVA-13B", 
                                 "PLLaVA-34B", "LLaVA-NeXT-Video-34B",
                                 "VideoOnline", "VideoLLaMBOnline", "InterSuitOnline", "InterSuitOnline_Qwen", "InternLMXCO",
                                 "VideoLLaMA2", "InterSuitAV", "InterSuitOnlineAV"], required=True)
    parser.add_argument("--benchmark_name", type=str, # general task
                        default="", 
                        choices=["ap", "md", "sg", "si", "pa", "pt"], required=True)
    
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.')
    parser.add_argument('--questions_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    
    model = load_model(args.model_name, args.device)
    
    # if args.model_name in ["VideoOnline", "VideoLLaMBOnline", "InterSuitOnline"]:
    if "online" in args.model_name.lower():
        from online_inference_utils import online_inference
        online_inference(
            model=model,
            model_name=args.model_name,
            benchmark_name=args.benchmark_name,
            questions_file=args.questions_file,
            num_chunks=args.num_chunks,
            chunk_id=args.chunk_idx,
            video_dir=args.video_dir,
            output_dir=args.output_dir
        )
    else:
        from inference_utils import inference
        inference(
            model=model,
            model_name=args.model_name,
            benchmark_name=args.benchmark_name,
            questions_file=args.questions_file,
            num_chunks=args.num_chunks,
            chunk_id=args.chunk_idx,
            video_dir=args.video_dir,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()