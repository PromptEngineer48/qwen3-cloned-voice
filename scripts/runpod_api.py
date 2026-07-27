# -*- coding: utf-8 -*-
"""Thin RunPod REST helper for pods and serverless endpoints.

Usage:
  python runpod_api.py pods                      # list pods
  python runpod_api.py pod-start <pod_id>
  python runpod_api.py pod-stop <pod_id>
  python runpod_api.py health                    # serverless endpoint health
  python runpod_api.py say "text to synthesize"  # one-off TTS smoke test -> say.wav

Env: RUNPOD_API_KEY (all) · RUNPOD_TTS_ENDPOINT_ID (health/say)
"""
import base64
import json
import os
import sys
from urllib import request as urlreq

API_KEY = os.environ.get("RUNPOD_API_KEY")
REST = "https://rest.runpod.io/v1"
SLS = "https://api.runpod.ai/v2"


def call(url, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urlreq.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urlreq.urlopen(req, timeout=90) as r:
        body = r.read()
        return json.loads(body) if body else {}


def main():
    if not API_KEY:
        sys.exit("RUNPOD_API_KEY not set")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pods"

    if cmd == "pods":
        out = call(f"{REST}/pods")
        pods = out if isinstance(out, list) else out.get("pods", out)
        print(json.dumps(pods, indent=1)[:4000])
    elif cmd in ("pod-start", "pod-stop", "pod-terminate"):
        pid = sys.argv[2]
        action = {"pod-start": "start", "pod-stop": "stop", "pod-terminate": ""}[cmd]
        if cmd == "pod-terminate":
            print(call(f"{REST}/pods/{pid}", method="DELETE"))
        else:
            print(call(f"{REST}/pods/{pid}/{action}", method="POST"))
    elif cmd == "health":
        ep = os.environ["RUNPOD_TTS_ENDPOINT_ID"]
        print(json.dumps(call(f"{SLS}/{ep}/health"), indent=1))
    elif cmd == "say":
        ep = os.environ["RUNPOD_TTS_ENDPOINT_ID"]
        text = sys.argv[2] if len(sys.argv) > 2 else "RunPod endpoint smoke test, one two three."
        out = call(f"{SLS}/{ep}/runsync", method="POST",
                   payload={"input": {"text": text, "language": "English"}})
        o = out.get("output") or {}
        if "audio_b64" in o:
            with open("say.wav", "wb") as f:
                f.write(base64.b64decode(o["audio_b64"]))
            print(f"say.wav written ({o.get('duration')}s)")
        else:
            print(json.dumps(out, indent=1)[:2000])
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
