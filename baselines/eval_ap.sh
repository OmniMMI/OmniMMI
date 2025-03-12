benchmark_name="ap"
cache_dir="./cache_dir"
input_dir="../omnimmi"
video_dir="${input_dir}/videos"
questions_file="${input_dir}/action_prediction.json"
output_dir="../results"
num_tasks=8

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_names=("VideoChatGPT" "VideoChat2" "VideoLLaVA" "LLaMA-VID" "MiniGPT4-Video" "PLLaVA" "LLaVA-NeXT-Video" "ShareGPT4Video" "LLaMA-VID-13B" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B" "LongVA" "LongVILA" "LongLLaVA")


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
