# RunPod serverless worker: Qwen3-TTS voice clone.
#
# Reference voice resolution (first hit wins):
#   1. reference/reference.WAV + reference/reference.txt baked into the image
#      (only when the repo is private and the files are committed)
#   2. REF_AUDIO_URL + REF_TEXT env vars set as RunPod endpoint secrets —
#      downloaded once at cold start. REF_TEXT may be the transcript itself
#      or an https URL to a .txt file. Keeps the voice out of a public repo.
#
# Input  : {"input": {"text": "...", "language": "English"}}
# Output : {"audio_b64": "<base64 wav>", "sr": 24000, "duration": 8.42}
import base64
import io
import os
from pathlib import Path
from urllib import request as urlreq

import numpy as np
import runpod
import soundfile as sf
import torch
from scipy import signal

import pyloudnorm as pyln
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).parent
REF_DIR = ROOT / "reference"
REF_WAV = REF_DIR / "reference.WAV"
REF_TXT = REF_DIR / "reference.txt"

_model = None
_ref_ready = False
_ref_text = ""


def _download(url, dest):
    req = urlreq.Request(url, headers={"User-Agent": "qwen3-cloned-voice-worker"})
    with urlreq.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())


def ensure_reference():
    global _ref_ready, _ref_text
    if _ref_ready:
        return
    REF_DIR.mkdir(exist_ok=True)
    if not REF_WAV.exists():
        url = os.environ.get("REF_AUDIO_URL")
        if not url:
            raise RuntimeError(
                "No reference voice: commit reference/reference.WAV (private repo) "
                "or set the REF_AUDIO_URL endpoint secret."
            )
        _download(url, REF_WAV)
    if REF_TXT.exists():
        _ref_text = REF_TXT.read_text(encoding="utf-8-sig").strip()
    else:
        rt = os.environ.get("REF_TEXT", "").strip()
        if rt.startswith("http://") or rt.startswith("https://"):
            _download(rt, REF_TXT)
            _ref_text = REF_TXT.read_text(encoding="utf-8-sig").strip()
        elif rt:
            _ref_text = rt
        else:
            raise RuntimeError("No reference transcript: commit reference/reference.txt or set REF_TEXT.")
    _ref_ready = True


def get_model():
    global _model
    if _model is None:
        model_ref = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        try:
            from huggingface_hub import snapshot_download
            model_ref = snapshot_download(model_ref, local_files_only=True)
        except Exception:
            pass  # fall back to hub id; first call downloads
        _model = Qwen3TTSModel.from_pretrained(
            model_ref,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    return _model


# --- identical post chain to the local pipeline (local/voice-clone-test.py) ---

def _highpass(audio, sr, cutoff_hz=80.0):
    sos = signal.butter(4, cutoff_hz, btype="high", fs=sr, output="sos")
    return signal.sosfilt(sos, audio)


def _compress(audio, threshold_db=-8.0, ratio=2.5):
    threshold_lin = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold_lin
    gain = np.ones_like(audio)
    gain[mask] = (threshold_lin + (abs_audio[mask] - threshold_lin) / ratio) / abs_audio[mask]
    return audio * gain


def _treble_boost(audio, sr, gain_db=2.0, shelf_hz=4000.0):
    gain_lin = 10 ** (gain_db / 20)
    sos = signal.butter(2, shelf_hz, btype="high", fs=sr, output="sos")
    high = signal.sosfilt(sos, audio)
    return audio + high * (gain_lin - 1)


def _loudness_normalize(audio, sr, target_lufs=-27.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if np.isinf(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def process_audio(audio, sr):
    audio = audio.astype(np.float64)
    audio = _highpass(audio, sr)
    audio = _loudness_normalize(audio, sr)
    audio = _compress(audio)
    audio = _treble_boost(audio, sr)
    peak = np.max(np.abs(audio))
    if peak > 0.891:  # true-peak limit at -1 dBFS
        audio = audio * (0.891 / peak)
    return audio.astype(np.float32)


def handler(event):
    inp = event.get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "input.text is required"}
    language = inp.get("language", "English")

    ensure_reference()
    model = get_model()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=str(REF_WAV),
        ref_text=_ref_text,
    )
    processed = process_audio(wavs[0], sr)

    buf = io.BytesIO()
    sf.write(buf, processed, sr, format="WAV")
    return {
        "audio_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "sr": int(sr),
        "duration": round(len(processed) / sr, 3),
    }


runpod.serverless.start({"handler": handler})
