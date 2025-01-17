import os, sys
import json
import tqdm
import time
import argparse
sys.path.append(os.getcwd())



video_path = "../assets/case.mp4"
video_path = "../assets/test_video.mp4"
video_path = "assets/case_2fps_384.mp4"
video_path = "../assets/water.mp4"

instruction = "Is there a dog in the video? Response with 'yes' or 'no'."
instruction = "Describe the Video: "
# instruction = "Can you describe the video?"
# instruction = "How many peope in the video?"
# instruction = "In the video provide a detailed description of the visual and auditory elements, including the setting, the main subjects or characters, their actions, the background music or sounds, and any notable events or interactions that occur throughout the footage."
# instruction = "what's the next step to clean the scissors ?"
# instruction = "According to the video, how to make a wine "
instruction = "Is there a mixer in the video ?"
# instruction = "I am responsible for executing Clean the scissors. The video's content shows the task's development, Can you recreate the steps I'll need to take? What's next step?"
# debug
# instruction = "yes, I know."
# instruction = "Okay."
# instruction = "Sorry to interrupt."
# instruction = "Oh, absolutely."

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="", 
                    choices=["VideoChatGPT", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video",
                             "Gemini-1.5-pro", "GPT4O",
                             "LLaVA", "GPT4V", 
                             "Video-LLaMA-2-13B", "LLaMA-VID-13B", 
                             "PLLaVA-13B", "PLLaVA-34B", "LLaVA-NeXT-Video-34B",
                             "LongVA", "LongVILA", "LongLLaVA", "VideoLLaMB", "VideoXL", "InternLMXCO",
                             "VideoOnline", "VideoLLaMBOnline", "InterSuit", "InterSuitOnline",
                             "VideoLLaMA2", "InterSuitAV", "InterSuitOnlineAV"])
args = parser.parse_args()
TESTING_MODEL=args.model_name


def load_model(TESTING_MODEL):
    if TESTING_MODEL == 'VideoChatGPT':
        from videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Valley2":
        from valley_modeling import Valley
        ckpt_path = f"{CKPT_DIR}/Valley2-7b"
        model = Valley({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2-13B":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-13B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoChat2":
        from videochat_modeling import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaVA":
        from videollava_modeling import VideoLLaVA
        ckpt_path = f"{CKPT_DIR}/VideoLLaVA/Video-LLaVA-7B"
        model = VideoLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID-13B":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-13B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLaVIT":
        from videolavit_modeling import VideoLaVIT
        ckpt_path = f"{CKPT_DIR}/Video-LaVIT-v1"
        model = VideoLaVIT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "MiniGPT4-Video":
        from minigpt4video_modeling import MiniGPT4Video
        ckpt_path = f"{CKPT_DIR}/MiniGPT4-Video/checkpoints"
        model = MiniGPT4Video({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA-13B":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-13b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA-34B":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-34b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA-NeXT-Video":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-7B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA-NeXT-Video-34B":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-34B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "ShareGPT4Video":
        from sharegpt4video_modeling import ShareGPT4Video
        ckpt_path = f"{CKPT_DIR}/ShareGPT4Video/sharegpt4video-8b"
        model = ShareGPT4Video({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Gemini-1.5-pro":
        from gemini_modeling import Gemini
        model = Gemini({"model_path": None, "device": 0})
    elif TESTING_MODEL == "LLaVA":
        from llava_modeling import LLaVA
        ckpt_path = f"{CKPT_DIR}/LLaVA/llava-v1.5-7b"
        model = LLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "GPT4V":
        from gpt4v_modeling import GPT4V
        model = GPT4V({"model_path": None, "device": 0})
    elif TESTING_MODEL == "GPT4O":
        from gpt4o_modeling import GPT4O
        model = GPT4O({"model_path": None, "device": 0})
    elif TESTING_MODEL == "LongVA":
        from longva_modeling import LongVA
        ckpt_path = f"{CKPT_DIR}/LongVA-7B-Qwen2"
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
        ckpt_path = f"{CKPT_DIR}/qwen27b-llavanextsub10k-qwen2-ORNS1111"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech-cosyvoice"
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-start"
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanext-qwen2-speech-cosyvoice"
        model = InterSuit({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuitOnline":
        from intersuitonline_modeling import InterSuitOnline
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-rev"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-ORNS1111"
        model = InterSuitOnline({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaMA2":
        from videollama2_modeling import VideoLLaMA2
        model = VideoLLaMA2({"model_path": f"{CKPT_DIR}/VideoLLaMA2.1-7B-AV", "device": 0})
    elif TESTING_MODEL == "VideoXL":
        from videoxl_modeling import VideoXL
        ckpt_path = f"{CKPT_DIR}/Video_XL/VideoXL_weight_8"
        model = VideoXL({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InternLMXCO":
        from internlmxco_modeling import InternLMXCO
        ckpt_path = f"{CKPT_DIR}/internlm-xcomposer2d5-ol-7b"
        model = InternLMXCO({"model_path": ckpt_path, "device": 0})    
    
    elif TESTING_MODEL == "InterSuitAV":
        from intersuit_av_modeling import InterSuitAV
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech" # 24
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech-va"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant"
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k-orns1111"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech"
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-full-start"
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-vst-vt-st-300k"
        
        # ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-llavanext-speech-voiceassistant-special"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech-cosyvoice"
        # ckpt_path = f"{CKPT_DIR}/longva7b-llavanext-qwen2-speech-cosyvoice"
        model = InterSuitAV({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "InterSuitOnlineAV":
        from intersuitonline_av_modeling import InterSuitOnlineAV
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-qwen2-speech-va"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k"
        ckpt_path = f"{CKPT_DIR}/longva7b-qwen2-voiceassistant-100k-orns1111"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanextsub10k-ORNS-qwen2-speech-va-ns"
        ckpt_path = f"{CKPT_DIR}/longva7b-llavanext-qwen2-speech-cosyvoice"
        

        model = InterSuitOnlineAV({"model_path": ckpt_path, "device": 0})

    return model

model = load_model(TESTING_MODEL)

if TESTING_MODEL in ["VideoOnline", "VideoLLaMBOnline", "InterSuitOnline", "InterSuitOnlineAV"]:
    import transformers
    logger = transformers.logging.get_logger('liveinfer')
    from evaluations.online_inference_utils import ffmpeg_once
    src_video_path = video_path
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
    model.load_video(ffmpeg_video_path)
    question = instruction
    # liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
    if TESTING_MODEL == "InterSuitOnline" or TESTING_MODEL == "InterSuitOnlineAV":
        question = "Please notify me when there is a mixer."
        model.input_query_stream(question)
        print(f'(Current Time = 0s) User: {question}')
    pred = ""
    history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
    duration = model.video_duration
    num_frames = int(duration * model.frame_fps)
    # print("video duration: ", duration)
    timecosts = []
    pbar = tqdm.tqdm(total=num_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    for i in range(num_frames):
        if TESTING_MODEL in ["VideoOnline", "VideoLLaMBOnline"]:
            if i == num_frames - 5:
                model.input_query_stream(question)
        start_time = time.time()
        model.input_video_stream(i / model.frame_fps)
        query, response, video_time = model()
        end_time = time.time()
        timecosts.append(end_time-start_time)
        fps = (i+1) / sum(timecosts)
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
        # print("****TIME****: ", video_time)
        if query:
            history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
            # print(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
            pred = response
            # print(pred)
            print(f'(Current Time = {i/5}s) Assistant: The mixer appear at {video_time}s')
            # break
    # print(history)
else:
    pred = model.generate(
        instruction=instruction,
        video_path=video_path,
        gen=True
    )
    
print('-'*20)
print(f'Instruction:\t{instruction}')
print(f'Answer:\t{pred}')
print('-'*20)
