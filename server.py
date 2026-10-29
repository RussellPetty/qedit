"""qedit — Qwen-Image-Edit-2511 + NSFW LoRA behind a small authed web UI."""
import io
import os
import secrets
import threading
import time
import uuid

import torch
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from diffusers import QwenImageEditPlusPipeline

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
LORA_FILE = "qwen-image-edit-plus-nsfw-lora.safetensors"

AUTH_USER = os.environ.get("QEDIT_USER", "russell")
AUTH_PASS = os.environ.get("QEDIT_PASS", "changeme")

MAX_SIDE = 1328   # native working resolution for Qwen-Image-Edit
MAX_IMAGES = 4    # pipeline accepts multiple reference images
MAX_JOBS_KEPT = 24

app = FastAPI(title="qedit")
security = HTTPBasic()

_state = {"status": "loading", "detail": "starting up", "boot": time.time()}
_pipe = None
_gpu_lock = threading.Lock()
_jobs = {}
_jobs_lock = threading.Lock()


def _require_auth(creds: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(creds.username.encode(), AUTH_USER.encode())
    ok_pass = secrets.compare_digest(creds.password.encode(), AUTH_PASS.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(401, "bad credentials", headers={"WWW-Authenticate": "Basic"})


def _load_model():
    global _pipe
    try:
        t0 = time.time()
        _state.update(status="loading", detail="downloading weights — first boot takes 10-20 min")
        pipe = QwenImageEditPlusPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_FILE)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
        _state.update(status="ready", detail=f"model loaded in {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001
        _state.update(status="error", detail=repr(e)[:500])


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load_model, daemon=True).start()


def _prep(img: Image.Image) -> Image.Image:
    """RGB, capped at MAX_SIDE, dims rounded to /16 for the VAE."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        w, h = int(w * s), int(h * s)
    w = max(16, min(MAX_SIDE, w // 16 * 16))
    h = max(16, min(MAX_SIDE, h // 16 * 16))
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img


def _run_job(job, images, prompt, negative_prompt, steps, cfg, lora_scale, seed):
    with _gpu_lock:
        if _state["status"] != "ready":
            job.update(status="error", error="model not ready")
            return
        job["status"] = "running"
        job["started"] = time.time()
        try:
            _pipe.set_adapters(["default"], adapter_weights=[lora_scale])
            gen = torch.Generator("cuda").manual_seed(seed)
            t0 = time.time()
            with torch.inference_mode():
                out = _pipe(
                    image=images,
                    prompt=prompt,
                    negative_prompt=negative_prompt or " ",
                    true_cfg_scale=cfg,
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    generator=gen,
                    num_images_per_prompt=1,
                )
            img = out.images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            job.update(
                result=buf.getvalue(),
                width=img.width,
                height=img.height,
                took=round(time.time() - t0, 1),
                status="done",
            )
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=repr(e)[:500])


@app.get("/")
def index(_: None = Depends(_require_auth)):
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/healthz")
def healthz(_: None = Depends(_require_auth)):
    return {
        "status": _state["status"],
        "detail": _state["detail"],
        "gpu_busy": _gpu_lock.locked(),
        "uptime_s": round(time.time() - _state["boot"]),
    }


@app.get("/api/logs")
def logs(_: None = Depends(_require_auth)):
    try:
        with open("/qedit.log", "rb") as f:
            return Response(f.read()[-16000:].decode(errors="replace"), media_type="text/plain")
    except OSError:
        return Response("(no log file)", media_type="text/plain")


@app.post("/api/edit")
async def edit(
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    steps: int = Form(40),
    cfg: float = Form(4.0),
    lora_scale: float = Form(1.0),
    seed: int = Form(-1),
    images: list[UploadFile] = File(...),
    _: None = Depends(_require_auth),
):
    if _state["status"] != "ready":
        raise HTTPException(503, f"model not ready ({_state['detail']})")
    if not prompt.strip():
        raise HTTPException(400, "prompt is required")
    if not images:
        raise HTTPException(400, "at least one image is required")

    pil = []
    for f in images[:MAX_IMAGES]:
        data = await f.read()
        try:
            pil.append(_prep(Image.open(io.BytesIO(data))))
        except Exception:
            raise HTTPException(400, f"could not read image: {f.filename}")

    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id, "status": "queued", "created": time.time(),
        "prompt": prompt, "seed": seed,
        "result": None, "width": None, "height": None, "took": None,
        "started": None, "error": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
        done = [j for j in _jobs.values() if j["status"] in ("done", "error")]
        for old in sorted(done, key=lambda j: j["created"])[:-MAX_JOBS_KEPT]:
            _jobs.pop(old["id"], None)

    threading.Thread(
        target=_run_job,
        args=(job, pil, prompt, negative_prompt, int(steps), float(cfg), float(lora_scale), int(seed)),
        daemon=True,
    ).start()
    return {"id": job_id, "seed": seed}


@app.get("/api/job/{job_id}")
def job_status(job_id: str, _: None = Depends(_require_auth)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    out = {k: v for k, v in job.items() if k != "result"}
    if job["status"] == "queued" and _gpu_lock.locked():
        out["queued_behind"] = True
    return out


@app.get("/api/result/{job_id}")
def job_result(job_id: str, _: None = Depends(_require_auth)):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done" or not job["result"]:
        raise HTTPException(404, "result not ready")
    return Response(job["result"], media_type="image/png")
