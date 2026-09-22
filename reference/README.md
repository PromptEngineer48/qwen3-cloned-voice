# Reference voice

`reference.WAV` (26.1 s) + `reference.txt` (its exact transcript) are committed
and baked into the worker image — the endpoint always speaks in this voice with
zero per-request setup.

The handler also supports `REF_AUDIO_URL` / `REF_TEXT` endpoint secrets as an
override if you ever want to swap voices without rebuilding the repo.
