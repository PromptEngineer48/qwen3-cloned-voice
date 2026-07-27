# Qwen3-TTS voice-clone serverless worker for RunPod.
# Installs the qwen_tts library from THIS repo (exact same code as the local
# pipeline) and bakes the model weights so requests never download anything.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# README.md is required: pyproject.toml declares `readme = "README.md"`,
# so setuptools fails the install if it is missing.
COPY pyproject.toml MANIFEST.in LICENSE README.md README_UPSTREAM.md ./
COPY qwen_tts/ ./qwen_tts/

# Pin torchaudio to the base image's torch (2.4.0) so installing the package
# does not drag in a different torch build and break CUDA.
RUN pip install --no-cache-dir torchaudio==2.4.0

RUN pip install --no-cache-dir . \
    runpod \
    soundfile \
    scipy \
    pyloudnorm \
    huggingface_hub

# Bake the model into the image (~4 GB) so cold start is load-from-disk only.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Fail the build here rather than at the first request if anything is wrong.
RUN python -c "import runpod, qwen_tts; from qwen_tts import Qwen3TTSModel; print('imports OK')"

COPY reference/ ./reference/
COPY handler.py ./handler.py

CMD ["python", "-u", "/app/handler.py"]
