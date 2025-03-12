#!/bin/bash
#SBATCH --job-name=speaker
#SBATCH --partition=HGX,DGX
##SBATCH --exclude=hgx-hyperplane[02]
#SBATCH --account=research
#SBATCH --qos=lv1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/speaker.out
#SBATCH --error=./slurm_logs/speaker.error.out

benchmark_name="si"
cache_dir="./cache_dir"
input_dir="../omnimmi"
video_dir="${input_dir}/videos"
questions_file="${input_dir}/general/speaker.json"
output_dir="../results"
num_tasks=8

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_names=("MiniGPT4-Video" "VideoChatGPT" "VideoLLaVA" "VideoChat2" "LLaMA-VID" "PLLaVA" "LLaVA-NeXT-Video" "ShareGPT4Video" "LongVA" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B")

for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    
    output_file=${output_dir}/${benchmark_name}_${model_name}.jsonl

    python ../evaluations/evaluate.py \
        --model_name ${model_name} \
        --benchmark_name ${benchmark_name} \
        --pred_path ${output_file} \
        --output_dir ${output_dir} \
        --num_tasks ${num_tasks}
done
