# Qwen3-TTS voice-clone serverless worker for RunPod.
#
# Base: official PyTorch runtime image — Python 3.11 with torch 2.4.0 + CUDA 12.4
# already installed and matched (3.9 GB compressed). Avoids both the 7.4 GB devel
# base and any risk of pip resolving a different torch build.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# libsndfile1 -> soundfile · ffmpeg -> librosa decoding · sox -> qwen_tts post chain
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torchaudio must match the base image's torch exactly.
RUN pip install --no-cache-dir torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

COPY pyproject.toml MANIFEST.in LICENSE README.md README_UPSTREAM.md ./
COPY qwen_tts/ ./qwen_tts/

# Install the package without its dependency list (which pulls gradio, unused in a
# serverless worker), then add exactly what the request path needs.
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
        librosa \
        soundfile \
        sox \
        onnxruntime \
        einops \
        runpod \
        scipy \
        pyloudnorm \
        huggingface_hub

# Bake the model so cold start is load-from-disk only.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Fail the build loudly rather than crash-looping at runtime.
RUN python -c "import torch, runpod, soundfile, pyloudnorm, scipy; from qwen_tts import Qwen3TTSModel; print('imports OK, torch', torch.__version__)"

COPY reference/ ./reference/
COPY handler.py ./handler.py

CMD ["python", "-u", "/app/handler.py"]
