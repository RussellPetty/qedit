# qedit

Drag-and-drop image editing UI for **Qwen-Image-Edit-2511** (20B) + an NSFW-capable LoRA
(`ScottzillaSystems/qwen-image-edit-plus-nsfw-lora`), deployed as a single RunPod GPU pod.

- `server.py` — FastAPI app: basic-auth, async edit jobs, model lifecycle
- `static/index.html` — the whole UI (no build step, no CDN)
- `start.sh` — pod boot script (clones this repo, caches venv + weights on container disk)

Deployed pod serves the UI on port 8000 via the RunPod proxy. Credentials live in the pod
env (`QEDIT_USER` / `QEDIT_PASS`); a local copy is in `.deploy.env` (gitignored).
