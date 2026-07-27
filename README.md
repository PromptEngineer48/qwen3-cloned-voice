# qwen3-cloned-voice

Qwen3-TTS voice cloning as a RunPod **serverless endpoint** — cloned-voice
narration for video pipelines from any machine (or phone-driven Claude Code
session), no local GPU needed.

Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (`qwen_tts` library
vendored in this repo, Apache-2.0 — see LICENSE and README_UPSTREAM.md).
Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, baked into the Docker image at build.

## Layout

| Path | What |
|---|---|
| `handler.py` | RunPod serverless handler (voice clone + mastering chain) |
| `Dockerfile` | Worker image: repo's own `qwen_tts` + baked model weights |
| `qwen_tts/` | The TTS library (vendored upstream source) |
| `reference/` | Reference voice slot — **not committed** in the public repo |
| `scripts/tts_client.py` | Batch client: `lines.json` → `<id>.mp3` + `manifest.json` |
| `scripts/runpod_api.py` | Ops helper: `health`, `say` smoke test, pod start/stop |
| `local/` | The original local-GPU scripts this endpoint replicates |

## Deploy (RunPod console, ~5 min)

1. Serverless → New Endpoint → **Import from GitHub** → this repo,
   Dockerfile at repo root.
2. GPU 24 GB tier (4090/A5000 class) · workers min 0, max 3 ·
   container disk ≥ 20 GB.
3. **Endpoint secrets** (because the voice is not in the public repo):
   - `REF_AUDIO_URL` — https URL to your `reference.WAV` (private gist raw
     URL, presigned S3, private HF resolve URL with token…)
   - `REF_TEXT` — the exact transcript of the reference audio (or an https
     URL to the .txt)
4. Deploy → copy endpoint id → `export RUNPOD_TTS_ENDPOINT_ID=...`

If your fork is **private**, skip the secrets: commit
`reference/reference.WAV` + `reference/reference.txt` instead — the image
bakes them.

## Use

```bash
export RUNPOD_API_KEY=rpa_...
export RUNPOD_TTS_ENDPOINT_ID=...

python scripts/runpod_api.py health
python scripts/runpod_api.py say "This is my cloned voice on RunPod."   # -> say.wav

# batch VO for a video: lines.json = [{"id": "vo_01_hook", "text": "..."}, ...]
python scripts/tts_client.py --lines vo/lines.json --out vo --workers 4
```

`tts_client.py` writes `<id>.mp3` per line plus `manifest.json`
(`[{id, file, duration}]`) — the exact contract of the local
`clone_voice_qwen.ps1`, so downstream video generators work unchanged.

## API

```jsonc
// POST /run  (or /runsync)
{"input": {"text": "Line to speak.", "language": "English"}}
// → {"audio_b64": "<base64 WAV>", "sr": 24000, "duration": 8.42}
```

Mastering chain (identical to the local pipeline): high-pass 80 Hz →
loudness −27 LUFS → compress 2.5:1 @ −8 dB → treble shelf +2 dB @ 4 kHz →
true-peak −1 dBFS.

## Cost

Scale-to-zero: $0 idle. Warm request ≈ 15–30 s of GPU per line; a 20-line
video VO ≈ a few cents. First request after idle pays a ~60–90 s cold start.
