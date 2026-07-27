# Qwen3-TTS voice-clone serverless worker for RunPod.
# Installs the qwen_tts library from THIS repo (exact same code as the local
# pipeline) and bakes the model weights so requests never download anything.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY pyproject.toml MANIFEST.in LICENSE README_UPSTREAM.md ./
COPY qwen_tts/ ./qwen_tts/

RUN pip install --no-cache-dir . \
    runpod \
    soundfile \
    scipy \
    pyloudnorm \
    huggingface_hub

# Bake the model into the image (~4 GB) so cold start is load-from-disk only.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# reference/ may contain only a README in the public repo — the handler then
# pulls the voice from the REF_AUDIO_URL endpoint secret at cold start.
COPY reference/ ./reference/
COPY handler.py ./handler.py

CMD ["python", "-u", "/app/handler.py"]
