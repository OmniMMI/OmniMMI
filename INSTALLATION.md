

To set up the environments, follow the instructions in the existing repositories and download the necessary checkpoints. Additionally, we offer guidance for this step, which addresses potential issues such as package version conflicts and system-related problems.


1. *(optional) checkpoints allow for manual downloading; otherwise, the model will download automatically if the Internet works fine.*
2. use `export DECORD_EOF_RETRY_MAX=20480` to prevent possible issues from decord 


*For models supporting speech input, we use ChatTTS to convert queries into speech for a fair comparison. Please install an additional TTS tool:*
```bash
pip install ChatTTS
pip install num2words
```


- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
    - Installation: [Instruction](https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file#installation-wrench) 
    - Checkpoints:
        - Source: [Video-ChatGPT-7B, LLaVA-Lightening-7B-v1-1](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/offline_demo.md#download-video-chatgpt-weights), [clip-vit (optional)](https://huggingface.co/openai/clip-vit-large-patch14)
        - Structure:
            ``` 
                ├── checkpoints/Video-ChatGPT-7B
                    ├── LLaVA-7B-Lightening-v1-1
                    ├── Video-ChatGPT-7B
                    └── clip-vit-large-patch14 (optional)
            ```


- [VideoChat2](https://github.com/OpenGVLab/Ask-Anything)
    - Installation: [Instruction](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2#usage)
        - Possible Issue 1: `ERROR: Could not find a version that satisfies the requirement torch==1.13.1+cu117` --> Solution: `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
        - Possible Issue 2: `flash-attention error` --> Solution: inference doesn't need flash-attention
    - Checkpoints: 
        - Source: [llama-7b](https://github.com/OpenGVLab/Ask-Anything/issues/150), [UMT-L-Qformer, VideoChat2_7B_stage2, VideoChat2_7B_stage3, Vicuna-7B-delta + script](https://github.com/OpenGVLab/Ask-Anything/issues/130)
        - Structure: 
            ``` 
                ├── checkpoints/VideoChat2
                    ├── umt_l16_qformer.pth
                    ├── videochat2_7b_stage2.pth
                    ├── videochat2_7b_stage3.pth
                    └── vicuna-7b-v0
            ```
    
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), 
    - Installation: [Instruction](https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation)
    - Checkpoints:
        - Source: [Video-LLaVA-7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B), [LanguageBind_Video (optional)](https://huggingface.co/LanguageBind/LanguageBind_Video_merge), [LanguageBind_Image (optional)](https://huggingface.co/LanguageBind/LanguageBind_Image)
        - Structure: 
            ``` 
                ├── checkpoints/VideoLLaVA
                    ├── Video-LLaVA-7B
                    ├── LanguageBind_Video_merge (optional)
                    └── LanguageBind_Image (optional)
            ```




- [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
    - Installation: [Instruction](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [llama-vid-7b-full-224-video-fps-1](https://huggingface.co/YanweiLi/llama-vid-7b-full-224-video-fps-1), [llama-vid-13b-full-224-video-fps-1](https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1) [eva_vit_g](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [bert (optional)](https://huggingface.co/openai/bert-base-uncased)
        - Structure: 
            ``` 
                ├── checkpoints/LLaMA-VID-7B
                    ├── llama-vid-7b-full-224-video-fps-1
                    ├── LAVIS/eva_vit_g.pth
                    └── bert-base-uncased (optional)
                ├── checkpoints/LLaMA-VID-13B
                    ├── llama-vid-13b-full-224-video-fps-1
                    ├── LAVIS/eva_vit_g.pth
                    └── bert-base-uncased (optional)
            ```

- [MiniGPT4-video](https://github.com/Vision-CAIR/MiniGPT4-video)
    - Installation: [Instruction](https://github.com/Vision-CAIR/MiniGPT4-video?tab=readme-ov-file#rocket-demo)
    - Checkpoints:
        - Source: [video_mistral_checkpoint_last](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/blob/main/checkpoints/video_mistral_checkpoint_last.pth), [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [vit (optional)](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
        - Structure: 
            ``` 
                ├── checkpoints/MiniGPT4-Video
                    ├── checkpoints/video_mistral_checkpoint_last.pth
                    ├── Mistral-7B-Instruct-v0.2
                    └── eva_vit_g.pth (optional)
            ```

- [PLLaVA](https://github.com/magic-research/PLLaVA)
    - Installation: [Instruction](https://github.com/magic-research/PLLaVA?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [pllava-7b](https://huggingface.co/ermu2001/pllava-7b), [pllava-13b](https://huggingface.co/ermu2001/pllava-13b), [pllava-34b](https://huggingface.co/ermu2001/pllava-34b)
        - Structure:
            ``` 
                ├── checkpoints/PLLaVA
                    └── pllava-7b
                ├── checkpoints/PLLaVA
                    └── pllava-13b
                ├── checkpoints/PLLaVA
                    └── pllava-34b
            ```

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
    - Installation: [Instruction](https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file#installation)
    - Checkpoints:
        - Source: [LLaVA-NeXT-Video-7B-DPO](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO), [LLaVA-NeXT-Video-34B-DPO](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-34B-DPO)
    - Structure:
        ``` 
            ├── checkpoints/PLLaVA
                └── LLaVA-NeXT-Video-7B-DPO
            ├── checkpoints/PLLaVA
                └── LLaVA-NeXT-Video-34B-DPO
        ```

- [ShareGPT4Video](https://github.com/ShareGPT4Omni/ShareGPT4Video)
    - Installation: [Instruction](https://github.com/ShareGPT4Omni/ShareGPT4Video?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [sharegpt4video-8b](https://huggingface.co/Lin-Chen/sharegpt4video-8b)
    - Structure:
        ``` 
            ├── checkpoints/ShareGPT4Video
                └── sharegpt4video-8b
        ```

- [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)
    - Installation: [Instruction](https://github.com/EvolvingLMMs-Lab/LongVA?tab=readme-ov-file#installation)
    - Checkpoints:
        - Source: [LongVA](https://huggingface.co/collections/lmms-lab/longva-667538e09329dbc7ea498057)
    - Structure:
        ``` 
            ├── checkpoints/LongVA-7B-Qwen2
        ```

- [LongVILA](https://github.com/NVlabs/VILA/tree/main/longvila)
    - Installation: [Instruction](https://github.com/NVlabs/VILA/tree/main/longvila#installation)
    - Checkpoints:
        - Source: Expired
    - Structure:
        ``` 
            ├── checkpoints/Llama-3-LongVILA-8B-1024Frames
        ```

- [LongLLaVA](https://github.com/FreedomIntelligence/LongLLaVA)
    - Installation: [Instruction](https://github.com/FreedomIntelligence/LongLLaVA?tab=readme-ov-file#1-environment-setup)
    - Checkpoints:
        - Source: Expired
    - Structure: [LongLLaVA-9B](https://huggingface.co/FreedomIntelligence/LongLLaVA-9B)
        ``` 
            ├── checkpoints/LongLLaVA-9B
        ```

- [VideoLLM-online-8B](https://github.com/showlab/videollm-online)
    - Installation: [Instruction](https://github.com/showlab/videollm-online?tab=readme-ov-file#installation)
    - Checkpoints:
        - Source: [videollm-online-8b-v1plus](https://huggingface.co/chenjoya/videollm-online-8b-v1plus)
    - Structure: 
        ``` 
            ├── checkpoints/videollm-online-8b-v1plus
        ```

- [VideoLLaMB](https://github.com/showlab/videollm-online)
    - Installation: [Instruction](https://github.com/bigai-nlco/VideoLLaMB?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [VideoLLaMB](https://huggingface.co/ColorfulAI/VideoLLaMB)
    - Structure: 
        ``` 
            ├── checkpoints/Videollamb-llava-1.5-7b
        ```

- [VideoLLaMA2-7B](https://github.com/showlab/videollm-online)
    - Installation: [Instruction](https://github.com/bigai-nlco/VideoLLaMB?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [VideoLLaMB](https://huggingface.co/ColorfulAI/VideoLLaMB)
    - Structure: 
        ``` 
            ├── checkpoints/Videollamb-llava-1.5-7b
        ```

- [InternLM-XComposer2.5-OmniLive](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-OmniLive)
    - Installation: [Instruction](https://github.com/InternLM/InternLM-XComposer/blob/main/docs/install.md)
    - Checkpoints:
        - Source: [IXC2.5-OL](https://huggingface.co/internlm/internlm-xcomposer2d5-ol-7b)
    - Structure:
        ```
            ├── checkpoints/internlm-xcomposer2d5-ol-7b
        ```

- [MiniCPM-o 2.6](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9)
    - Installation: [Instruction](https://github.com/OpenBMB/MiniCPM-o?tab=readme-ov-file#inference)
    - Checkpoints:
        - Source: [MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)
    - Structure: 
        ```
            ├── checkpoints/MiniCPM-o-2_6
        ```

- [Gemini API](https://github.com/google-gemini/cookbook)

- [GPT4V](https://openai.com/index/gpt-4v-system-card/)
    - Installation: make a .env under baselines/gpt4v; set `API_BASE` and `API_KEY`