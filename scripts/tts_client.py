# -*- coding: utf-8 -*-
"""Cloud VO generation: send lines.json to the RunPod Qwen3-TTS endpoint,
write <id>.mp3 per line + manifest.json — the exact output contract of
clone_voice_qwen.ps1, so every existing video generator works unchanged.

Usage:
  python tts_client.py --lines path/to/lines.json --out path/to/vo [--workers 4]

Env:
  RUNPOD_API_KEY          required
  RUNPOD_TTS_ENDPOINT_ID  required (serverless endpoint id)
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request as urlreq

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT = os.environ.get("RUNPOD_TTS_ENDPOINT_ID")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"


def _post(path, payload):
    req = urlreq.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urlreq.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(path):
    req = urlreq.Request(BASE + path, headers={"Authorization": f"Bearer {API_KEY}"})
    with urlreq.urlopen(req, timeout=60) as r:
        return json.load(r)


def synth_line(item, out_dir, max_wait=600):
    sid, text = item["id"], item["text"]
    job = _post("/run", {"input": {"text": text, "language": "English"}})
    job_id = job["id"]
    t0 = time.time()
    while True:
        st = _get(f"/status/{job_id}")
        s = st.get("status")
        if s == "COMPLETED":
            out = st["output"]
            if "error" in out:
                raise RuntimeError(f"{sid}: worker error {out['error']}")
            wav = out_dir / f"{sid}.wav"
            wav.write_bytes(base64.b64decode(out["audio_b64"]))
            return sid, out.get("duration")
        if s in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"{sid}: job {s}: {st.get('error')}")
        if time.time() - t0 > max_wait:
            raise RuntimeError(f"{sid}: timed out after {max_wait}s (status {s})")
        time.sleep(3 if s == "IN_PROGRESS" else 5)


def ffprobe_duration(p):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
        capture_output=True, text=True,
    )
    return round(float(r.stdout.strip()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("RUNPOD_API_KEY not set")
    if not ENDPOINT:
        sys.exit("RUNPOD_TTS_ENDPOINT_ID not set")

    items = json.loads(Path(args.lines).read_text(encoding="utf-8-sig"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Submitting {len(items)} lines to endpoint {ENDPOINT} ({args.workers} workers)")
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(synth_line, it, out_dir): it["id"] for it in items}
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                _, dur = fut.result()
                print(f"[{sid}] ok ({dur}s)")
            except Exception as e:
                print(f"[{sid}] FAIL {e}")
                failed.append(sid)
    if failed:
        sys.exit(f"{len(failed)} line(s) failed: {failed}")

    manifest = []
    for it in items:
        sid = it["id"]
        wav, mp3 = out_dir / f"{sid}.wav", out_dir / f"{sid}.mp3"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-q:a", "3", str(mp3)], check=True)
        wav.unlink()
        manifest.append({"id": sid, "file": mp3.name, "duration": ffprobe_duration(mp3)})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(m["duration"] for m in manifest)
    print(f"manifest.json written · {len(manifest)} lines · {total:.1f}s total VO")


if __name__ == "__main__":
    main()
