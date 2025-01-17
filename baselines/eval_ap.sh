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
environments=("video_chatgpt" "videochat2" "videollava" "llamavid" "minigpt4_video" "pllava" "llavanext" "share4video" "llamavid" "pllava" "pllava" "llavanext" "llongva" "vila" "longllava")

model_names=("LongVA" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B")
environments=("llongva" "pllava" "pllava" "llavanext")

model_names=("LongVA")
environments=("llongva")

model_names=("LongLLaVA" "LongVILA")
environments=("longllava" "vila")

model_names=("LLaMA-VID")
environments=("llamavid")

model_names=("VideoOnline")
environments=("videoonline")

# model_names=("VideoChatGPT" "VideoChat2" "VideoLLaVA" "LLaMA-VID" "MiniGPT4-Video" "PLLaVA" "LLaVA-NeXT-Video" "ShareGPT4Video" "LLaMA-VID-13B" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B" "LongVA" "LongVILA" "LongLLaVA")

model_names=("Gemini-1.5-pro")

model_names=("GPT4O")

model_names=("InterSuit")

model_names=("LongVA")

model_names=("InterSuit")

model_names=("miniomni2" "vita")

model_names=("InterSuit")

model_names=("InterSuitAV")

# model_names=("InternLMXCO")

# model_names=("InterSuit" "LongVA" "LongVILA" "VideoLLaMA2" "VideoXL")

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
