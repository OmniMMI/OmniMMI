import openai
import re
import os
import argparse
import json
import ast
import time
import collections
from multiprocessing.pool import Pool
from tqdm import tqdm

import time, random
from openai import AzureOpenAI, OpenAI

from os import getenv
from dotenv import load_dotenv
load_dotenv()
API_BASE = getenv("API_BASE")
API_KEY = getenv("API_KEY")

REGIONS = {
        "gpt-35-turbo-0125": ["canadaeast", "northcentralus", "southcentralus"], # 0.5 1.5
        "gpt-4o-mini-2024-07-18": ["eastus"], # 0.15, 0.60
        "gpt-4o-2024-08-06": ["eastus", "eastus2", "northcentralus", "southcentralus", "swedencentral", "westus", "westus3"], # 2.5, 10.00
        "gpt-4o-2024-05-13": ["eastus", "eastus2", "northcentralus", "southcentralus", "westus", "westus3"], # 5.0, 15.00
        "gpt-4-turbo-2024-04-09": ["eastus2", "swedencentral"], # 10.0 30.00
        "gpt-4-vision-preview": ["australiaeast", "japaneast", "westus"] # 10.0 30.00
    }

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--model_name", type=str,
                        default="", 
                        choices=["VideoChatGPT", "VideoChat2", "VideoLLaVA", "LLaMA-VID","MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video",
                                 "Gemini-1.5-pro", "GPT4O",
                                 "LongVA", "LongVILA", "LongLLaVA", "VideoLLaMB", "InterSuit", "VideoXL",
                                 "LLaMA-VID-13B", "PLLaVA-13B", 
                                 "PLLaVA-34B", "LLaVA-NeXT-Video-34B",
                                 "VideoOnline", "VideoLLaMBOnline", "InterSuitOnline",
                                 "VideoLLaMA2", "vita", "miniomni2", "InternLMXCO",
                                 "InterSuitAV", "InterSuitOnlineAV"], required=True)
    parser.add_argument("--benchmark_name", type=str,
                        default="", 
                        choices=["ap", "md", "sg", "si", "pa", "pt"])
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args



def openai_api_1(model, messages):
    api_base = API_BASE
    api_key = API_KEY
    if api_base:
        client = OpenAI(
            api_key=api_key,
            api_base=api_base
        )
    else:
        client = OpenAI(
            api_key=api_key,
        )
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    response = response.choices[0].message.content
    return response

def openai_api_0(model, messages):
    api_base = API_BASE
    api_key = API_KEY
    # Compute the correctness score
    openai.api_key = api_key
    if api_key:
        openai.api_base = api_base
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    # Convert response to a Python dictionary.
    response_message = completion["choices"][0]["message"]["content"]
    return response_message

def azureopenai_api(model, messages):
    
    
    api_base = API_BASE
    api_key = API_KEY

    region = random.choice(REGIONS[model])
    endpoint = f"{api_base}/{region}"
    client = AzureOpenAI(
        api_key = api_key,
        api_version = "2024-02-01",
        azure_endpoint = endpoint,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    response = response.choices[0].message.content
    # print(response)
    
    return response

def openai_eval(question, answer, pred):
    messages=[
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string without codeblock. "
                    "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            }
        ]
    if API_BASE and 'bigai' in API_BASE:
        # response_message = azureopenai_api(model='gpt-35-turbo-0125', messages=messages)
        response_message = azureopenai_api(model='gpt-4o-2024-08-06', messages=messages)
    else:
        # Set the OpenAI API key.
        # response_message = openai_api_0(model="gpt-3.5-turbo-0125", messages=messages)
        response_message = openai_api_1(model="gpt-4-turbo", messages=messages)
    # Convert response to a Python list.
    response_dict = ast.literal_eval(response_message)
    return response_dict


def evaluate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-4o
    Returns a score for correctness.
    """

    for file in caption_files:
        key = file[:-5] # Strip file extension
        eval_set = prediction_set[key]
        
        question = eval_set["question"]
        answer = eval_set["answer"]
        pred = eval_set["pred"]
        
        try:
            # Compute the accuracy
            
            response_dict =  openai_eval(question, answer, pred)
            result_qa_pair = [response_dict, eval_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")

def multi_evaluate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-4o
    Returns a score for correctness.
    """

    for file in caption_files:
        key = file[:-5] # Strip file extension
        eval_set = prediction_set[key]
        score_set = {"qa":[]}
        
        for qa in eval_set["qa"]:
        
            question = qa["question"]
            answer = qa["answer"]
            pred = qa["pred"]
            
            try:
                # Compute the accuracy
                response_dict = openai_eval(question, answer, pred)
                score_set["qa"].append(response_dict)
                

            except Exception as e:
                print(f"Error processing file '{key}': {e}")
                
        result_qa_pair = [score_set, eval_set]

        # Save the question-answer pairs to a json file.
        with open(f"{output_dir}/{key}.json", "w") as f:
            json.dump(result_qa_pair, f)


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    
    
    
    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]
    # new_pred_contents = new_pred_contents[:8] # debug
    
    # proactive alerting
    if args.benchmark_name == "pa":
        if type(new_pred_contents[0]["pred"]) is list:
            acc = 0
            prec = 0
            iou = 0
            count = 0
            for pred_contents in new_pred_contents:
                count += 1
                gt_s, gt_e = pred_contents["answer"]
                preds = pred_contents["pred"]
                if preds == []: continue
                if preds[0] >= gt_s and preds[0] <= gt_e: acc += 1
                hit = 0
                for pred in preds:
                    if pred >= gt_s and pred <= gt_e: hit += 1
                prec += hit / len(preds)
                pred_s, pred_e = preds[0], preds[-1]
                iou += max(0, min(pred_e, gt_e) - max(pred_s, gt_s)) / (max(pred_e, gt_e) - min(pred_s, gt_s))
            print("All evaluation completed!")
            print("TASK: ", args.benchmark_name)
            print("MODEL: ", args.model_name)
            print("Total Data Point: ", count)
            print("Average Accuracy: ", acc / count)
            print("Average precision: ", prec / count)
            print("Average IoU: ", iou / count)
        else:
            hit_count = 0
            count = 0
            for pred_contents in new_pred_contents:
                gt_s, gt_e = pred_contents["answer"]
                pred = pred_contents["pred"]
                if pred >= gt_s and pred <= gt_e:
                    hit_count += 1
                count += 1
            print("All evaluation completed!")
            print("TASK: ", args.benchmark_name)
            print("MODEL: ", args.model_name)
            print("Total Data Point: ", count)
            print("Average Accuracy: ", hit_count / count)
        return
    # proactive turn-taking
    elif args.benchmark_name == "pt":
        hit_count = 0
        count = 0
        for pred_contents in new_pred_contents:
            pred = pred_contents["pred"]
            if pred == "" or pred.startswith("<2>"): hit_count += 1
            # if "yes" not in pred.lower(): hit_count += 1 # ixc2.5-ol
            count += 1
        print("All evaluation completed!")
        print("TASK: ", args.benchmark_name)
        print("MODEL: ", args.model_name)
        print("Total Data Point: ", count)
        print("Average Accuracy: ", hit_count / count)
        return
            

    # Generating list of id's and corresponding files
    id_list = [x['video'].split(".mp4")[0]+f"##{idx}" for idx, x in enumerate(new_pred_contents)]
    caption_files = [f"{idx}.json" for idx in id_list]

    output_dir = os.path.join(args.output_dir, args.benchmark_name, args.model_name, "gpt4o") 
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for idx, sample in enumerate(new_pred_contents):
        vid = sample['video'].split(".mp4")[0] + f"##{idx}"
        if args.benchmark_name == "plan" or args.benchmark_name == "speaker":
            answer = sample['answer']
            pred = sample['pred']
            ques = sample['question']
            eval_set = {"question": ques, "answer": answer, "pred": pred}
        elif args.benchmark_name == "dependency" or args.benchmark_name == "transition":
            eval_set = {"qa": []}
            for qa in sample["qa"]:
                answer = qa['answer']
                # pred = qa['pred']
                if "pred" in qa: pred = qa["pred"]
                else: pred = ""
                ques = qa['question']
                eval_set["qa"].append({"question": ques, "answer": answer, "pred": pred})
            
        prediction_set[vid] = eval_set

    num_tasks = args.num_tasks

    start = time.time()
    # While loop to ensure that all captions are processed.
    while True:
        # try:
            
        # Files that have not been processed yet.
        completed_files = os.listdir(output_dir)
        print(f"completed_files: {len(completed_files)}")

        # Files that have not been processed yet.
        incomplete_files = [f for f in caption_files if f not in completed_files]
        print(f"incomplete_files: {len(incomplete_files)}")

        # Break the loop when there are no incomplete files
        if len(incomplete_files) == 0:
            break
        if len(incomplete_files) <= num_tasks:
            num_tasks = 1

        # Split tasks into parts.
        part_len = len(incomplete_files) // num_tasks
        all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
        task_args = [(prediction_set, part, output_dir, args) for part in all_parts]

        # Use a pool of workers to process the files in parallel.
        with Pool() as pool:
            if args.benchmark_name == "ap" or args.benchmark_name == "si":
                pool.starmap(evaluate, task_args)
            elif args.benchmark_name == "md" or args.benchmark_name == "sg":
                pool.starmap(multi_evaluate, task_args)
            else:
                raise TypeError(f"INVALID benchmark_name: {args.benchmark_name}, please select from [ap, si, dm, sg]")
                            

        # except Exception as e:
        #     print(f"Error: {e}")

    end = time.time()
    eval_hs = (end-start) // 3600
    eval_mins = (end-start) % 3600 // 60
    print(f"Evaluation takes {eval_hs} hours {eval_mins} minutes")


    # Combine all the processed files into one
    combined_contents = {}
    json_path = os.path.join(args.output_dir, args.benchmark_name, args.model_name, "results.json") 

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file, indent=4)
    print("All evaluation completed!")
    print("TASK: ", args.benchmark_name)
    print("MODEL: ", args.model_name)

    if args.benchmark_name == "ap" or args.benchmark_name == "si":
        # Calculate precision and recall
        score_sum = 0
        count = 0
        yes_count = 0
        no_count = 0
        for key, result in tqdm(combined_contents.items()):
            # Computing score
            count += 1
            score_sum += int(result[0]['score'])
            pred = result[0]['pred']
            if "yes" in pred.lower(): yes_count += 1
            elif "no" in pred.lower(): no_count += 1
        score_avg = score_sum / count
        accuracy = yes_count / count
        print("Average accuracy:", accuracy)
        print("Average score:", score_avg)
    elif args.benchmark_name == "md" or args.benchmark_name == "sg":
        score_sum = 0
        count = 0
        yes_count = 0
        no_count = 0
        item_acc = collections.defaultdict(list)
        item_step_acc = collections.defaultdict(list)
        for key, result in tqdm(combined_contents.items()):
            item_score_sum = 0
            item_count = 0
            item_yes_count = 0
            for idx, res in enumerate(result[0]["qa"]):
                score = res["score"]
                pred = res["pred"]
                item_count += 1
                item_score_sum += int(score)
                if "yes" in pred.lower():
                    item_acc[idx].append(1)
                    item_step_acc[idx].append(1 and item_step_acc.get(idx-1, [1])[-1])
                    item_yes_count += 1
                else:
                    item_acc[idx].append(0)
                    item_step_acc[idx].append(0)
            item_score_avg = item_score_sum / item_count
            score_sum += item_score_avg
            if item_count == item_yes_count: yes_count += 1
            else: no_count += 1
            count += 1
        score_avg = score_sum / count
        accuracy = yes_count / count
        print("Average accuracy:", accuracy)
        print("Average score:", score_avg)
        print("Each accumulate step accuracy: ")
        for k, lst in item_step_acc.items():
            # print(lst)
            print(k, ": ", sum(lst)/len(lst))
        print("Each step accuracy: ")
        for k, lst in item_acc.items():
            print(k, ": ", sum(lst)/len(lst))
        
    
    else:
        raise TypeError(f"INVALID benchmark_name: {args.benchmark_name}, please select from [plan, speaker, dependency, transition]")


if __name__ == "__main__":
    main()

