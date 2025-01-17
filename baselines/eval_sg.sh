benchmark_name="sg"
cache_dir="./cache_dir"
input_dir="../omnimmi"
video_dir="${input_dir}/videos"
questions_file="${input_dir}/dynamic_state_grounding.json"
output_dir="../results"
num_tasks=8

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


model_names=("MiniGPT4-Video" "VideoChatGPT" "VideoLLaVA" "VideoChat2" "LLaMA-VID" "PLLaVA" "LLaVA-NeXT-Video" "ShareGPT4Video" "LongVA" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B")
environments=("minigpt4_video" "video_chatgpt" "videollava" "videochat2" "llamavid" "pllava" "llavanext" "share4video" "llongva" "pllava" "pllava" "llavanext")

# model_names=("LongVA" "LLaMA-VID-13B" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B")
# environments=("llongva" "llamavid" "pllava" "pllava" "llavanext")

# model_names=("MiniGPT4-Video")
# environments=("minigpt4_video")

model_names=("LLaMA-VID-13B" "LongLLaVA" "LongVILA" "VideoLLaMB")

model_names=("LongVILA")

model_names=("VideoOnline")

model_names=("VideoChatGPT" "VideoChat2" "VideoLLaVA" "LLaMA-VID" "MiniGPT4-Video" "PLLaVA" "LLaVA-NeXT-Video" "ShareGPT4Video" "LLaMA-VID-13B" "PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B" "LongVA" "LongVILA" "LongLLaVA" "VideoLLaMB")

model_names=("VideoLLaMB")

model_names=("VideoOnline")

model_names=("Gemini-1.5-pro")

model_names=("GPT4O")

model_names=("InterSuit")

# model_names=("VideoLLaMA2")

model_names=("GPT4O" "Gemini-1.5-pro")

model_names=("vita" "miniomni2")

model_names=("VideoLLaMA2")

model_names=("InterSuitAV")

# model_names=("InternLMXCO")

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
