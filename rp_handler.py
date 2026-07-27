# RunPod serverless worker: Qwen3-TTS voice clone.
#
# Design note: only `runpod` is imported at module level. Everything heavy
# (torch, qwen_tts, audio libs) is imported lazily inside the request path and
# wrapped in try/except, so an import or CUDA failure is returned as job output
# instead of killing the process. A crashing worker is invisible — RunPod's
# public API exposes no worker logs — but a failed job body is readable.
#
# Input  : {"input": {"text": "...", "language": "English"}}
#          {"input": {"diag": true}}   -> environment report, no synthesis
# Output : {"audio_b64": "<base64 wav>", "sr": 24000, "duration": 8.42}
#          {"error": "...", "traceback": "..."} on failure
import base64
import io
import os
import sys
import traceback
from pathlib import Path
from urllib import request as urlreq

import runpod

ROOT = Path(__file__).parent
REF_DIR = ROOT / "reference"
REF_WAV = REF_DIR / "reference.WAV"
REF_TXT = REF_DIR / "reference.txt"

_model = None
_ref_text = None


def _download(url, dest):
    req = urlreq.Request(url, headers={"User-Agent": "qwen3-cloned-voice-worker"})
    with urlreq.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())


def ensure_reference():
    """Resolve the reference voice: baked files first, then env-var override."""
    global _ref_text
    if _ref_text is not None:
        return _ref_text
    REF_DIR.mkdir(exist_ok=True)
    if not REF_WAV.exists():
        url = os.environ.get("REF_AUDIO_URL")
        if not url:
            raise RuntimeError(
                "No reference voice: reference/reference.WAV is missing from the "
                "image and REF_AUDIO_URL is not set."
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
            raise RuntimeError("No reference transcript: reference/reference.txt missing and REF_TEXT unset.")
    return _ref_text


def get_model():
    global _model
    if _model is None:
        import torch
        from qwen_tts import Qwen3TTSModel
        model_ref = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        try:
            from huggingface_hub import snapshot_download
            model_ref = snapshot_download(model_ref, local_files_only=True)
        except Exception:
            pass  # not baked / not cached: fall back to the hub id
        _model = Qwen3TTSModel.from_pretrained(
            model_ref,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    return _model


# --- mastering chain, identical to the local pipeline (local/voice-clone-test.py) ---

def process_audio(audio, sr):
    import numpy as np
    import pyloudnorm as pyln
    from scipy import signal

    def highpass(a, cutoff_hz=80.0):
        sos = signal.butter(4, cutoff_hz, btype="high", fs=sr, output="sos")
        return signal.sosfilt(sos, a)

    def loudness_normalize(a, target_lufs=-27.0):
        loudness = pyln.Meter(sr).integrated_loudness(a)
        return a if np.isinf(loudness) else pyln.normalize.loudness(a, loudness, target_lufs)

    def compress(a, threshold_db=-8.0, ratio=2.5):
        threshold_lin = 10 ** (threshold_db / 20)
        abs_a = np.abs(a)
        mask = abs_a > threshold_lin
        gain = np.ones_like(a)
        gain[mask] = (threshold_lin + (abs_a[mask] - threshold_lin) / ratio) / abs_a[mask]
        return a * gain

    def treble_boost(a, gain_db=2.0, shelf_hz=4000.0):
        gain_lin = 10 ** (gain_db / 20)
        sos = signal.butter(2, shelf_hz, btype="high", fs=sr, output="sos")
        return a + signal.sosfilt(sos, a) * (gain_lin - 1)

    audio = audio.astype(np.float64)
    audio = highpass(audio)
    audio = loudness_normalize(audio)
    audio = compress(audio)
    audio = treble_boost(audio)
    peak = np.max(np.abs(audio))
    if peak > 0.891:  # true-peak limit at -1 dBFS
        audio = audio * (0.891 / peak)
    return audio.astype(np.float32)


def diagnostics():
    """Report what the container actually has, so failures are debuggable."""
    report = {
        "python": sys.version.split()[0],
        "cwd_files": sorted(p.name for p in ROOT.iterdir())[:25],
        "reference_wav": REF_WAV.exists(),
        "reference_txt": REF_TXT.exists(),
    }
    for mod in ("torch", "transformers", "soundfile", "scipy", "pyloudnorm", "librosa", "qwen_tts"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            report[mod] = f"IMPORT FAILED: {type(e).__name__}: {e}"
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        report["cuda_available"] = f"ERROR: {e}"
    return report


def handler(event):
    inp = event.get("input") or {}
    if inp.get("diag"):
        return diagnostics()

    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "input.text is required"}

    try:
        import soundfile as sf
        ref_text = ensure_reference()
        model = get_model()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=inp.get("language", "English"),
            ref_audio=str(REF_WAV),
            ref_text=ref_text,
        )
        processed = process_audio(wavs[0], sr)
        buf = io.BytesIO()
        sf.write(buf, processed, sr, format="WAV")
        return {
            "audio_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "sr": int(sr),
            "duration": round(len(processed) / sr, 3),
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-3000:],
            "diagnostics": diagnostics(),
        }


runpod.serverless.start({"handler": handler})
