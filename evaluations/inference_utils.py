import os
import re
import json
import math
import time
import random        
from tqdm import tqdm

import numpy as np
import torch

import ast


def setup_seed(seed=428):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_random_chunk(lst, n, k, seed=42):
    lst = lst[:]
    random.seed(seed)
    random.shuffle(lst)
    chunks = split_list(lst, n)
    return chunks[k]

    
def inference(
    model,
    model_name,
    benchmark_name,
    questions_file,
    num_chunks,
    chunk_id,
    video_dir,
    output_dir,
    seed=42,
):
    setup_seed(seed)
    

    questions = json.load(open(questions_file))
    # questions = get_chunk(questions, num_chunks, chunk_id)
    questions = get_random_chunk(questions, num_chunks, chunk_id)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    preds_file = os.path.join(output_dir, f"{benchmark_name}_{model_name}_{chunk_id}.json")
    pred_file = open(preds_file, "w")

    
    for sample in tqdm(questions):

        # action prediction
        if benchmark_name == "ap":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            question = sample["question"]
            vocab = sample["vocab"]
            if "av" in model_name.lower():
                # vocab = f"Please select the next step from the provided vocabulary list: {vocab}."
                question = f"{question}, Please select the next step from the provided vocabulary list: {vocab}."
                with torch.no_grad():
                    output = model.generate(
                        instruction=question,
                        video_path=video_path,
                        # vocab=vocab
                    )
            else:
                question = f"{question}, Please select the next step from the provided vocabulary list: {vocab}."
                with torch.no_grad():
                    output = model.generate(
                        instruction=question,
                        video_path=video_path
                    )
            sample["pred"] = output
            pred_file.write(json.dumps(sample)+"\n")
        
        # speaker identification
        elif benchmark_name == "si":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            question = sample["question"]
            with torch.no_grad():
                output = model.generate(
                    instruction=question,
                    video_path=video_path
                )
            sample["pred"] = output
            pred_file.write(json.dumps(sample)+"\n")

        # multi-turn dependency
        elif benchmark_name == "md":
            vid = sample["video"].split(".mp4")[0]
            for idx, qa in enumerate(sample["qa"]):
                question = qa["question"]
                if idx != 0:
                    question = question.replace("##ANSWER##", output)
                s, e = map(float, qa["timestamp"].split("--"))
                e = str(int(e))
                video_path = os.path.join(video_dir, "clips", f"{vid}_{e}.mp4")
                with torch.no_grad():
                    output = model.generate(
                        instruction=question,
                        video_path=video_path
                    )
                qa["pred"] = output
            pred_file.write(json.dumps(sample)+"\n")
        
        # dynamic state grounding
        elif benchmark_name == "sg":
            vid = sample["video"].split(".mp4")[0]
            prompt = ""
            for idx, qa in enumerate(sample["qa"]):
                question = qa["question"]
                prompt += f"user\n{question}\nassistant\n"
                s, e = map(float, qa["timestamp"].split("--"))
                e = str(int(e))
                video_path = os.path.join(video_dir, "clips", f"{vid}_{e}.mp4")
                with torch.no_grad():
                    output = model.generate(
                        instruction=prompt,
                        video_path=video_path
                    )
                qa["pred"] = output
                prompt += f"{output}\n"
            pred_file.write(json.dumps(sample)+"\n")
        
        # proactive turn-taking
        elif benchmark_name == "pt":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            convs = sample["conversation"][-2:]
            query = ""
            for conv in convs:
                for k, v in conv.items():
                    query += f"{k}\n{v}\n"
            with torch.no_grad():
                output = model.generate(
                    instruction=query,
                    video_path=video_path,
                    gen=False
                )
            sample["pred"] = output
            pred_file.write(json.dumps(sample)+"\n")
        
        else:
            raise TypeError(f"INVALID benchmark_name: {benchmark_name}, please select from [ap, si, md, sg, pt]")

    pred_file.close()