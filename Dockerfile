# Slim variant — same worker, ~1/3 the image size.
#
# The runpod/pytorch:*-devel base is 7.4 GB compressed (~20 GB unpacked);
# with torch deps + the baked model that is ~30 GB every scale-to-zero worker
# must pull before serving a single request. This variant starts from the CUDA
# runtime image, installs a pinned cu124 torch, and skips gradio (the package
# lists it, but a serverless worker never launches a UI).
#
# Swap this in by renaming it to Dockerfile once verified.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# libsndfile1 -> soundfile · ffmpeg -> librosa/torchaudio decoding · sox -> qwen_tts post
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libsndfile1 ffmpeg sox git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pinned CUDA 12.4 torch stack — must be installed before the package so the
# resolver cannot pull a different torch build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.0 torchaudio==2.4.0

COPY pyproject.toml MANIFEST.in LICENSE README.md README_UPSTREAM.md ./
COPY qwen_tts/ ./qwen_tts/

# Install the package without its dependency list (which drags in gradio), then
# add exactly the runtime deps the handler path needs.
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

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Fail at build time, not at the first request.
RUN python -c "import torch, runpod, soundfile, pyloudnorm; from qwen_tts import Qwen3TTSModel; print('imports OK, torch', torch.__version__)"

COPY reference/ ./reference/
COPY handler.py ./handler.py

CMD ["python", "-u", "/app/handler.py"]
