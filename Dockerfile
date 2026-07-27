# Qwen3-TTS voice-clone serverless worker for RunPod.
#
# Base: official PyTorch runtime image with CUDA 12.8.
#
# CUDA 12.8 / torch 2.8 is required, not a preference: RunPod serves this GPU tier
# with Blackwell RTX PRO 6000 MIG slices (compute capability sm_120). Torch builds
# on cu124 only ship kernels up to sm_90, so torch cannot initialise the device,
# RunPod's own fitness check fails, and the worker exits before polling the queue:
#
#   sm_120 is not compatible with the current PyTorch installation
#   Fitness check failed: _cuda_init_check | CUDA error: no kernel image is
#   available for execution on the device
#
# cu128 covers sm_120 (Blackwell) as well as sm_86/sm_89 (A5000, 3090, L4).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# libsndfile1 -> soundfile · ffmpeg -> librosa decoding · sox -> qwen_tts post chain
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torchaudio must match the base image's torch exactly.
RUN pip install --no-cache-dir torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

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

# The model is intentionally NOT baked in: it adds ~4 GB, which pushes the image
# past what a CI runner can build and makes every worker pull it. The handler
# falls back to downloading from the hub on first use (~1-2 min once per cold
# worker). Attach a network volume later if that cost matters.

# Fail the build loudly rather than crash-looping at runtime.
RUN python -c "import torch, runpod, soundfile, pyloudnorm, scipy; from qwen_tts import Qwen3TTSModel; print('imports OK, torch', torch.__version__, 'arch list', torch.cuda.get_arch_list())"

COPY reference/ ./reference/
COPY rp_handler.py ./rp_handler.py

CMD ["python", "-u", "/app/rp_handler.py"]
