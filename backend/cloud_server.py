"""
cloud_server.py — Voice Studio Cloud API for RunPod GPU deployment.

Endpoints:
  GET  /health      — check if model is loaded and ready (poll this!)
  POST /generate    — voice cloning (text + reference audio → WAV)
  POST /tts         — pure TTS without reference audio

On FIRST startup:
  - Automatically downloads VoxCPM2 (~5-7GB) to the Network Volume at MODEL_PATH
  - Server responds on port 8000 immediately; /health returns {"ready": false}
  - When model finishes downloading, /health returns {"ready": true}

On SUBSEQUENT startups (model already on volume):
  - Loads model directly from volume (~30-60 sec, no download needed)

Burmese language is fully supported by VoxCPM2.
"""
from __future__ import annotations

import os
import sys
import threading
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── VoxCPM path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VOXCPM_DIR = ROOT / "VoxCPM"
SRC_DIR = VOXCPM_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voxcpm import VoxCPM  # noqa: E402

warnings.filterwarnings("ignore")

# ── Model configuration ───────────────────────────────────────────────────────
# MODEL_PATH is set via environment variable in RunPod pod config.
# Default falls back to local VoxCPM directory.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(VOXCPM_DIR / "pretrained_models" / "VoxCPM2"),
)

OUTPUT_DIR = Path("/tmp/voice-studio-outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Presets ───────────────────────────────────────────────────────────────────
QUALITY_PRESETS = {
    "Fast":            {"inference_timesteps": 6,  "cfg_value": 1.8},
    "Balanced":        {"inference_timesteps": 10, "cfg_value": 2.0},
    "High Similarity": {"inference_timesteps": 16, "cfg_value": 2.3},
}

STYLE_PRESETS = {
    "Natural":         "natural spoken delivery, clear and grounded",
    "Deep Reflective": "deep reflective delivery, calm, philosophical, deliberate, intimate",
    "Warm Storyteller":"warm storyteller delivery, grounded, expressive, gentle pauses",
    "Soft Intimate":   "soft intimate delivery, tender, close, quiet, slow",
    "Documentary":     "documentary narration delivery, deep, composed, serious tone",
}

# ── Global state ──────────────────────────────────────────────────────────────
_model: VoxCPM | None = None
_status: str = "starting"   # "starting" | "downloading" | "loading" | "ready" | "error"
_status_detail: str = ""


def _bootstrap_model():
    """
    Background thread: downloads model to volume if needed, then loads it.
    Server responds on port 8000 immediately; poll /health for readiness.
    """
    global _model, _status, _status_detail

    model_dir = Path(MODEL_PATH)

    # ── Step 1: Download if not already on volume ─────────────────────────────
    if not model_dir.exists() or not any(model_dir.iterdir()):
        _status = "downloading"
        _status_detail = f"Downloading VoxCPM2 to {model_dir} (~5-7GB, first time only)..."
        print(f"[Voice Studio] {_status_detail}")

        try:
            from huggingface_hub import snapshot_download
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download("openbmb/VoxCPM2", local_dir=str(model_dir))
            print(f"[Voice Studio] ✅ Download complete! Saved to {model_dir}")
        except Exception as e:
            _status = "error"
            _status_detail = f"Download failed: {e}"
            print(f"[Voice Studio] ❌ {_status_detail}")
            return
    else:
        print(f"[Voice Studio] Model found at {model_dir} — skipping download")

    # ── Step 2: Load model into memory ───────────────────────────────────────
    _status = "loading"
    _status_detail = f"Loading model from {MODEL_PATH}..."
    print(f"[Voice Studio] {_status_detail}")

    try:
        _model = VoxCPM.from_pretrained(
            hf_model_id=MODEL_PATH,
            load_denoiser=False,
        )
        _status = "ready"
        _status_detail = "Model loaded and ready!"
        print(f"[Voice Studio] 🚀 {_status_detail}")
    except Exception as e:
        _status = "error"
        _status_detail = f"Model load failed: {e}"
        print(f"[Voice Studio] ❌ {_status_detail}")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Voice Studio Cloud API",
    description="VoxCPM2 voice cloning — supports Burmese, English, and 28 more languages.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Start model bootstrap in background. Server is immediately available."""
    thread = threading.Thread(target=_bootstrap_model, daemon=True)
    thread.start()
    print("[Voice Studio] Server started. Model loading in background — poll /health for status.")


@app.get("/health")
async def health_check():
    """
    Poll this endpoint to know when the model is ready.
    Returns {"ready": true} when generation is available.
    """
    return {
        "status": _status,
        "detail": _status_detail,
        "ready": _status == "ready",
    }


@app.post("/generate")
async def generate_cloned_voice(
    text: str = Form(..., description="Text to speak — Burmese (မြန်မာ) or English"),
    reference_audio: UploadFile = File(..., description="Reference voice clip WAV or MP3"),
    quality: str = Form("Balanced", description="Fast | Balanced | High Similarity"),
    style: str = Form("Natural", description="Style preset"),
    custom_style: str = Form("", description="Custom style description"),
):
    """Clone a voice. Returns a WAV file."""
    if _status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready yet. Status: {_status} — {_status_detail}. Poll /health."
        )
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    style_desc = custom_style.strip() or STYLE_PRESETS.get(style, STYLE_PRESETS["Natural"])
    formatted_text = f"({style_desc}){text.strip()}" if style_desc else text.strip()
    quality_params = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Balanced"])

    suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await reference_audio.read())
        tmp.close()

        wav = _model.generate(
            text=formatted_text,
            reference_wav_path=tmp.name,
            **quality_params,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = OUTPUT_DIR / f"clone-{timestamp}.wav"
        sf.write(str(out_path), wav, _model.tts_model.sample_rate)

        return FileResponse(
            path=str(out_path),
            media_type="audio/wav",
            filename=f"voice-clone-{timestamp}.wav",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    quality: str = Form("Balanced"),
    style: str = Form("Natural"),
    custom_style: str = Form(""),
):
    """Pure TTS — no reference audio needed."""
    if _status != "ready":
        raise HTTPException(status_code=503, detail=f"Model not ready. Status: {_status}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    style_desc = custom_style.strip() or STYLE_PRESETS.get(style, STYLE_PRESETS["Natural"])
    formatted_text = f"({style_desc}){text.strip()}" if style_desc else text.strip()
    quality_params = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Balanced"])

    try:
        wav = _model.generate(text=formatted_text, **quality_params)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = OUTPUT_DIR / f"tts-{timestamp}.wav"
        sf.write(str(out_path), wav, _model.tts_model.sample_rate)
        return FileResponse(path=str(out_path), media_type="audio/wav",
                            filename=f"tts-{timestamp}.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
