# Reference voice (not committed)

This repo is public, so the personal reference voice is **not** stored here —
anyone with `reference.WAV` can clone the voice.

The worker resolves the voice in this order:

1. `reference/reference.WAV` + `reference/reference.txt` in the image
   (commit them only if your fork is **private**)
2. `REF_AUDIO_URL` + `REF_TEXT` endpoint secrets set in the RunPod console —
   downloaded once at cold start, never stored in the repo or image registry.

Local dev: drop your `reference.WAV` (5–15 s clean speech) and
`reference.txt` (its exact transcript) in this folder — .gitignore keeps
them out of commits.
