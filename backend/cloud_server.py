"""
cloud_server.py — Voice Studio Cloud API for RunPod GPU deployment.

Endpoints:
  GET  /health         — check if model is loaded and ready (poll this!)
  POST /generate       — voice cloning (returns job_id immediately)
  GET  /status/{job_id} — check generation progress
  GET  /result/{job_id} — download finished audio
  POST /tts            — pure TTS without reference audio
"""
from __future__ import annotations

import os, re, sys, threading, tempfile, time, uuid, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
VOXCPM_DIR = ROOT / "VoxCPM"
SRC_DIR = VOXCPM_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voxcpm import VoxCPM
warnings.filterwarnings("ignore")

MODEL_PATH = os.environ.get("MODEL_PATH", str(VOXCPM_DIR / "pretrained_models" / "VoxCPM2"))
OUTPUT_DIR = Path("/tmp/voice-studio-outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_PRESETS = {
    "Fast": {"inference_timesteps": 6, "cfg_value": 1.8},
    "Balanced": {"inference_timesteps": 10, "cfg_value": 2.0},
    "High Similarity": {"inference_timesteps": 16, "cfg_value": 2.3},
}

STYLE_PRESETS = {
    "Natural": "natural spoken delivery, clear and grounded",
    "Deep Reflective": "deep reflective delivery, calm, philosophical, deliberate, intimate",
    "Warm Storyteller": "warm storyteller delivery, grounded, expressive, gentle pauses",
    "Soft Intimate": "soft intimate delivery, tender, close, quiet, slow",
    "Documentary": "documentary narration delivery, deep, composed, serious tone",
}

SPEED_INSTRUCTIONS: dict[str, str] = {
    "Normal":  "",
    "Slower":  "speak slowly, calm and clear pacing",
    "Slowest": "speak very slowly, calm and clear pacing",
}

_model = None
_status = "starting"
_status_detail = ""

MAX_REF_DURATION = 10.0
SILENCE_GAP = 0.5

# ─── In-memory job store ──────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Clean up old jobs after 10 minutes
_JOB_TTL = 600


def _cleanup_expired_jobs():
    """Remove jobs older than JOB_TTL."""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items() if now - j.get("created_at", 0) > _JOB_TTL]
        for jid in expired:
            # Delete result file if it exists
            path = _jobs[jid].get("result_path", "")
            if path:
                Path(path).unlink(missing_ok=True)
            del _jobs[jid]


def _trim_audio(path, max_duration=MAX_REF_DURATION):
    info = sf.info(path)
    if info.duration <= max_duration:
        return path
    audio, sr = sf.read(path)
    audio = audio[:int(max_duration * sr)]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    return tmp.name


def _split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Split text by sentence boundaries (Burmese-prioritized), keeping chunks under max_chars."""
    def _split_and_merge(parts: list[str], delimiter: str, max_len: int) -> list[str]:
        result = []
        buf = ""
        for part in parts:
            candidate = (buf + delimiter + part).strip() if buf else part
            if len(candidate) <= max_len:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                if len(part) > max_len:
                    result.append(part)
                else:
                    buf = part
        if buf:
            result.append(buf)
        return result

    step1 = re.split(r"(?<=[။.!?])\s*", text)
    step1 = [s.strip() for s in step1 if s.strip()]
    chunks = _split_and_merge(step1, " ", max_chars)

    final = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            sub_parts = re.split(r"(?<=[၊])\s*", chunk)
            sub_parts = [s.strip() for s in sub_parts if s.strip()]
            sub_chunks = _split_and_merge(sub_parts, " ", max_chars)
            final.extend(sub_chunks)

    final2 = []
    for chunk in final:
        if len(chunk) <= max_chars:
            final2.append(chunk)
        else:
            sub_parts = re.split(r"(?<=[,])\s*", chunk)
            sub_parts = [s.strip() for s in sub_parts if s.strip()]
            sub_chunks = _split_and_merge(sub_parts, " ", max_chars)
            final2.extend(sub_chunks)

    final3 = []
    for chunk in final2:
        if len(chunk) <= max_chars:
            final3.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                final3.append(chunk[i:i + max_chars].strip())

    return [c for c in final3 if c]


def _bootstrap_model():
    global _model, _status, _status_detail
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists() or not any(model_dir.iterdir()):
        _status = "downloading"
        _status_detail = f"Downloading VoxCPM2 to {model_dir} (~5-7GB, first time only)..."
        print(f"[Voice Studio] {_status_detail}")
        try:
            from huggingface_hub import snapshot_download
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download("openbmb/VoxCPM2", local_dir=str(model_dir))
            print(f"[Voice Studio] Download complete! Saved to {model_dir}")
        except Exception as e:
            _status = "error"
            _status_detail = f"Download failed: {e}"
            print(f"[Voice Studio] {_status_detail}")
            return
    else:
        print(f"[Voice Studio] Model found at {model_dir} — skipping download")
    _status = "loading"
    _status_detail = f"Loading model from {MODEL_PATH}..."
    print(f"[Voice Studio] {_status_detail}")
    try:
        _model = VoxCPM.from_pretrained(hf_model_id=MODEL_PATH, load_denoiser=False)
        _status = "ready"
        _status_detail = "Model loaded and ready!"
        print(f"[Voice Studio] {_status_detail}")
    except Exception as e:
        _status = "error"
        _status_detail = f"Model load failed: {e}"
        print(f"[Voice Studio] {_status_detail}")


def _run_generation(job_id: str, chunks: list[str], ref_path: str, style_desc: str,
                    quality_params: dict, temp_files: list[str]):
    """Background generation worker — runs in a daemon thread."""
    try:
        sample_rate = None
        all_wavs = []
        for i, chunk in enumerate(chunks):
            # Only prepend style instruction to FIRST chunk.
            # Subsequent chunks continue naturally without re-instructing,
            # which prevents garbled/nonsense audio from repeated instructions.
            if i == 0 and style_desc:
                formatted = f"({style_desc}){chunk}"
            else:
                formatted = chunk

            wav = _model.generate(
                text=formatted,
                reference_wav_path=ref_path,
                retry_badcase=False,
                **quality_params,
            )
            all_wavs.append(wav)
            if sample_rate is None:
                sample_rate = _model.tts_model.sample_rate

            import torch
            torch.cuda.empty_cache()

            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = {"done": i + 1, "total": len(chunks)}

        # Concatenate all chunks with silence gaps
        silence = np.zeros(int(SILENCE_GAP * sample_rate), dtype=np.float32)
        combined = all_wavs[0]
        for wav in all_wavs[1:]:
            combined = np.concatenate([combined, silence, wav])

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = OUTPUT_DIR / f"clone-{timestamp}.wav"
        sf.write(str(out_path), combined, sample_rate)

        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result_path"] = str(out_path)
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)
    finally:
        for f in temp_files:
            Path(f).unlink(missing_ok=True)


app = FastAPI(title="Voice Studio Cloud API", version="2.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_bootstrap_model, daemon=True).start()
    print("[Voice Studio] Server started. Poll /health for model status.")


@app.get("/health")
async def health_check():
    return {"status": _status, "detail": _status_detail, "ready": _status == "ready"}


@app.post("/generate")
async def generate_cloned_voice(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    quality: str = Form("Balanced"),
    style: str = Form("Natural"),
    custom_style: str = Form(""),
    speed: str = Form("Normal"),
):
    if _status != "ready":
        raise HTTPException(status_code=503, detail=f"Model not ready. Status: {_status}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    style_desc = custom_style.strip() or STYLE_PRESETS.get(style, STYLE_PRESETS["Natural"])
    quality_params = dict(QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Balanced"]))

    # Speed text instruction — will be prepended to first chunk only
    speed_instruction = SPEED_INSTRUCTIONS.get(speed, "")
    if speed_instruction:
        style_desc = f"{speed_instruction}, {style_desc}" if style_desc else speed_instruction

    suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
    ref_tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_files = [ref_tmp.name]
    try:
        ref_tmp.write(await reference_audio.read())
        ref_tmp.close()
        ref_path = _trim_audio(ref_tmp.name)

        chunks = _split_sentences(text.strip())
        if not chunks:
            chunks = [text.strip()]

        # Create job and start background generation
        _cleanup_expired_jobs()
        job_id = str(uuid.uuid4())[:8]
        job = {
            "id": job_id,
            "status": "processing",
            "progress": {"done": 0, "total": len(chunks)},
            "error": None,
            "result_path": None,
            "created_at": time.time(),
        }
        with _jobs_lock:
            _jobs[job_id] = job

        threading.Thread(
            target=_run_generation,
            args=(job_id, chunks, ref_path, style_desc, quality_params, temp_files),
            daemon=True,
        ).start()

        return {"job_id": job_id, "status": "processing", "progress": job["progress"]}
    except Exception as e:
        for f in temp_files:
            Path(f).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Generation setup failed: {str(e)}")


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    _cleanup_expired_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job.get("error"),
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    _cleanup_expired_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job["status"] == "processing":
        return {
            "job_id": job["id"],
            "status": "processing",
            "progress": job["progress"],
        }
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Unknown error")}

    result_path = job.get("result_path", "")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=500, detail="Result file not found")

    # Remove job after serving result
    with _jobs_lock:
        _jobs.pop(job_id, None)

    return FileResponse(
        path=result_path,
        media_type="audio/wav",
        filename=f"voice-clone-{job_id}.wav",
    )


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    quality: str = Form("Balanced"),
    style: str = Form("Natural"),
    custom_style: str = Form(""),
    speed: str = Form("Normal"),
):
    if _status != "ready":
        raise HTTPException(status_code=503, detail=f"Model not ready. Status: {_status}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    style_desc = custom_style.strip() or STYLE_PRESETS.get(style, STYLE_PRESETS["Natural"])
    quality_params = dict(QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Balanced"]))

    speed_instruction = SPEED_INSTRUCTIONS.get(speed, "")
    if speed_instruction:
        style_desc = f"{speed_instruction}, {style_desc}" if style_desc else speed_instruction

    formatted_text = f"({style_desc}){text.strip()}" if style_desc else text.strip()
    try:
        wav = _model.generate(text=formatted_text, retry_badcase=False, **quality_params)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = OUTPUT_DIR / f"tts-{timestamp}.wav"
        sf.write(str(out_path), wav, _model.tts_model.sample_rate)
        return FileResponse(path=str(out_path), media_type="audio/wav", filename=f"tts-{timestamp}.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
