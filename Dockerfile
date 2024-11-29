FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-contrib-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install torchsde einops diffusers transformers accelerate peft timm kornia scikit-image matplotlib blend-modes segment-anything spandrel wget pydantic aiohttp && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/sipherxyz/comfyui-art-venture /content/ComfyUI/custom_nodes/comfyui-art-venture && \
    git clone https://github.com/chflame163/ComfyUI_LayerStyle /content/ComfyUI/custom_nodes/ComfyUI_LayerStyle && \
    git clone https://github.com/cubiq/ComfyUI_essentials /content/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet /content/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet && \
    git clone -b dev https://github.com/camenduru/ComfyUI_SLK_joy_caption_two /content/ComfyUI/custom_nodes/ComfyUI_SLK_joy_caption_two && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus /content/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/leosamsHelloworldXL_helloworldXL70/resolve/main/leosamsHelloworldXL_helloworldXL70.safetensors -d /content/ComfyUI/models/checkpoints/sdxl -o leosamsHelloworldXL_helloworldXL70.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/alvdansen/midsommarcartoon/resolve/main/araminta_k_midsommar_cartoon.safetensors -d /content/ComfyUI/models/loras/sdxl -o araminta_k_midsommar_cartoon.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors -d /content/ComfyUI/models/controlnet/sdxl -o controlnet-union-sdxl-1.0-promax.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/yiwangsimple/BiRefNet-general-epoch_244/resolve/main/BiRefNet-general-epoch_244.pth -d /content/ComfyUI/models/BiRefNet/pth -o BiRefNet-general-epoch_244.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/raw/main/config.json -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/raw/main/generation_config.json -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/resolve/main/model-00001-of-00004.safetensors -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o model-00001-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/resolve/main/model-00002-of-00004.safetensors -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o model-00002-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/resolve/main/model-00003-of-00004.safetensors -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o model-00003-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/resolve/main/model-00004-of-00004.safetensors -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o model-00004-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/raw/main/model.safetensors.index.json -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/raw/main/special_tokens_map.json -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/raw/main/tokenizer.json -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2/raw/main/tokenizer_config.json -d /content/ComfyUI/models/LLM/Orenguteng--Llama-3.1-8B-Lexi-Uncensored-V2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/config.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/preprocessor_config.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/special_tokens_map.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/spiece.model -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/tokenizer.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/tokenizer_config.json -d /content/ComfyUI/models/clip/siglip-so400m-patch14-384 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/resolve/main/image_adapter.pt -d /content/ComfyUI/models/Joy_caption_two -o image_adapter.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/raw/main/config.yaml -d /content/ComfyUI/models/Joy_caption_two -o config.yaml && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/resolve/main/clip_model.pt -d /content/ComfyUI/models/Joy_caption_two -o clip_model.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/raw/main/text_model/tokenizer_config.json -d /content/ComfyUI/models/Joy_caption_two/text_model -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/raw/main/text_model/tokenizer.json -d /content/ComfyUI/models/Joy_caption_two/text_model -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/raw/main/text_model/special_tokens_map.json -d /content/ComfyUI/models/Joy_caption_two/text_model -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/resolve/main/text_model/adapter_model.safetensors -d /content/ComfyUI/models/Joy_caption_two/text_model -o adapter_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption-alpha-two/raw/main/text_model/adapter_config.json -d /content/ComfyUI/models/Joy_caption_two/text_model -o adapter_config.json

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py