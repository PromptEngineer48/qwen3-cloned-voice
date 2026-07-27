import json
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from scipy import signal
import pyloudnorm as pyln

from qwen_tts import Qwen3TTSModel


# --- "Sharp" post-processing chain (B262 Meetily build) ---
# Highpass 90 Hz | Loudness -16 LUFS | Compress 3:1 @ -10 dB
# Presence shelf +3.5 dB @ 3 kHz | Air shelf +2 dB @ 8 kHz | Peak limit -1.4 dBFS

def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 90.0) -> np.ndarray:
    sos = signal.butter(4, cutoff_hz, btype="high", fs=sr, output="sos")
    return signal.sosfilt(sos, audio)

def _compress(audio: np.ndarray, threshold_db: float = -10.0, ratio: float = 3.0) -> np.ndarray:
    threshold_lin = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold_lin
    gain = np.ones_like(audio)
    gain[mask] = (threshold_lin + (abs_audio[mask] - threshold_lin) / ratio) / abs_audio[mask]
    return audio * gain

def _high_shelf(audio: np.ndarray, sr: int, gain_db: float, shelf_hz: float) -> np.ndarray:
    gain_lin = 10 ** (gain_db / 20)
    sos = signal.butter(2, shelf_hz, btype="high", fs=sr, output="sos")
    high = signal.sosfilt(sos, audio)
    return audio + high * (gain_lin - 1)

def _loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if np.isinf(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs)

def process_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = audio.astype(np.float64)
    audio = _highpass(audio, sr)
    audio = _loudness_normalize(audio, sr, target_lufs=-16.0)
    audio = _compress(audio)
    audio = _high_shelf(audio, sr, gain_db=3.5, shelf_hz=3000.0)   # presence / clarity
    audio = _high_shelf(audio, sr, gain_db=2.0, shelf_hz=8000.0)   # air / sparkle
    peak = np.max(np.abs(audio))
    if peak > 0.851:  # -1.4 dBFS true-peak ceiling
        audio = audio * (0.851 / peak)
    return audio.astype(np.float32)


ROOT = Path(__file__).parent
REF_WAV = ROOT / "reference" / "reference.WAV"
REF_TXT = ROOT / "reference" / "reference.txt"
SCRIPTS = ROOT / "scripts.json"
OUT_DIR = ROOT / "out"


def main():
    if not REF_WAV.exists():
        raise SystemExit(f"Missing reference audio: {REF_WAV}")
    if not REF_TXT.exists():
        raise SystemExit(f"Missing reference text: {REF_TXT}")
    if not SCRIPTS.exists():
        raise SystemExit(f"Missing scripts file: {SCRIPTS}")

    OUT_DIR.mkdir(exist_ok=True)

    ref_text = REF_TXT.read_text(encoding="utf-8-sig").strip()
    scripts = json.loads(SCRIPTS.read_text(encoding="utf-8-sig"))["scripts"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    for item in scripts:
        sid = item["id"]
        text = item["text"]
        out_path = OUT_DIR / f"{sid}.wav"

        if out_path.exists():
            print(f"[{sid}] skip (exists)")
            continue

        print(f"[{sid}] {text[:60]}...")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(REF_WAV),
            ref_text=ref_text,
        )
        processed = process_audio(wavs[0], sr)
        sf.write(out_path, processed, sr)
        print(f"[{sid}] -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
