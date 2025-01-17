#!/bin/bash
#SBATCH --job-name=pt
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv0a
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_logs/pt.out
#SBATCH --error=./slurm_logs/pt.error.out

benchmark_name="pt"
cache_dir="./cache_dir"
input_dir="../omnimmi"
video_dir="${input_dir}/videos"
questions_file="${input_dir}/proactive_turntaking.json"
output_dir="../results"
num_tasks=8

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_names=("MiniGPT4-Video" "VideoChatGPT" "VideoLLaVA" "VideoChat2" "LLaMA-VID" "PLLaVA" "LLaVA-NeXT-Video" "ShareGPT4Video" "LongVA" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B")
environments=("minigpt4_video" "video_chatgpt" "videollava" "videochat2" "llamavid" "pllava" "llavanext" "share4video" "llongva" "pllava" "pllava" "llavanext")


model_names=("LongVILA")
environments=("vila")

model_names=("LongLLaVA")
environments=("longllava")

model_names=("LLaMA-VID")
environments=("llamavid")

model_names=("VideoLLaMB")
environments=("llava")

model_names=("VideoOnline" "VideoLLaMBOnline")
environments=("videoonline" "llava")

model_names=("InterSuit")
environments=("longva")

model_names=("InterSuitAV")
environments=("longva")

model_names=("InternLMXCO")
environments=("internlmx")


for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    environment="${environments[$i]}"
    source ~/scratch/anaconda3/bin/activate
    # source ~/anaconda3/bin/activate
    conda activate "$environment"
    for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ../evaluations/inference.py \
        --model_name ${model_name} \
        --benchmark_name ${benchmark_name} \
        --cache_dir ${cache_dir} \
        --video_dir ${video_dir} \
        --questions_file ${questions_file} \
        --output_dir ${output_dir} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
    done
    wait
    
    output_file=${output_dir}/${benchmark_name}_${model_name}.jsonl
    
    # Clear out the output file if it exists.
    > "$output_file"
    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${output_dir}/${benchmark_name}_${model_name}_${IDX}.json >> "$output_file"
    done

    # python ../evaluations/evaluate.py \
    #     --model_name ${model_name} \
    #     --benchmark_name ${benchmark_name} \
    #     --pred_path ${output_file} \
    #     --output_dir ${output_dir} \
    #     --num_tasks ${num_tasks}
done
