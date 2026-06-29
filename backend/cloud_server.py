"""
cloud_server.py — Voice Studio Cloud API for RunPod GPU deployment.

Endpoints:
  GET  /health      — check if model is loaded and ready (poll this!)
  POST /generate    — voice cloning (text + reference audio → WAV)
  POST /tts         — pure TTS without reference audio
"""
from __future__ import annotations

import os, re, sys, threading, tempfile, warnings
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

# Speed → min_len multiplier (text_tokens * factor).
# Higher min_len forces the model to generate more audio tokens → slower natural speech.
SPEED_PRESETS: dict[str, float | None] = {
    "Normal":  None,   # default min_len=2, model decides freely
    "Slower":  0.8,    # ~20% more audio tokens than text tokens
    "Slowest": 1.3,    # ~60% more audio tokens than text tokens
}

# Speed → additional text instruction prepended to style description.
# Works alongside min_len for dual-path slowdown (text instruction + model param).
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
    """Split text by sentence boundaries (Burmese-prioritized), keeping chunks under max_chars.

    Burmese uses ။ (section end) and ၊ (clause separator) as natural boundaries.
    English-like punctuation (., !, ?) and newlines are also respected.

    Strategy:
    1. Split on Burmese section end ။ — each gets its own chunk
    2. For chunks still over max_chars, split on Burmese clause separator ၊
    3. For chunks still over max_chars, split on English sentence boundaries
    4. For chunks still over max_chars, split on commas
    5. Final fallback: mid-sentence split at max_chars
    """
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
                # If a single element exceeds max_len, try next delimiter
                if len(part) > max_len:
                    result.append(part)  # pass through for deeper splitting
                else:
                    buf = part
        if buf:
            result.append(buf)
        return result

    # Step 1: Split on Burmese section end ။ + any sentence-ending punctuation
    step1 = re.split(r"(?<=[။.!?])\s*", text)
    step1 = [s.strip() for s in step1 if s.strip()]
    chunks = _split_and_merge(step1, " ", max_chars)

    # Step 2: For chunks still over limit, split on Burmese clause separator ၊
    final = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            sub_parts = re.split(r"(?<=[၊])\s*", chunk)
            sub_parts = [s.strip() for s in sub_parts if s.strip()]
            sub_chunks = _split_and_merge(sub_parts, " ", max_chars)
            final.extend(sub_chunks)

    # Step 3: For any remaining oversized chunks, split on commas
    final2 = []
    for chunk in final:
        if len(chunk) <= max_chars:
            final2.append(chunk)
        else:
            sub_parts = re.split(r"(?<=[,])\s*", chunk)
            sub_parts = [s.strip() for s in sub_parts if s.strip()]
            sub_chunks = _split_and_merge(sub_parts, " ", max_chars)
            final2.extend(sub_chunks)

    # Step 4: Final fallback — hard split for any stubbornly long chunks
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

app = FastAPI(title="Voice Studio Cloud API", version="2.1.0")
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

    # Speed → text instruction: VoxCPM2 interprets this naturally.
    # min_len was removed — it's a minimum-gate, not a speed control,
    # and it conflicts with the model's badcase retry logic.
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

        sample_rate = None
        all_wavs = []
        for chunk in chunks:
            formatted = f"({style_desc}){chunk}" if style_desc else chunk
            wav = _model.generate(text=formatted, reference_wav_path=ref_path, retry_badcase=False, **quality_params)
            all_wavs.append(wav)
            if sample_rate is None:
                sample_rate = _model.tts_model.sample_rate

        silence = np.zeros(int(SILENCE_GAP * sample_rate), dtype=np.float32)
        combined = all_wavs[0]
        for wav in all_wavs[1:]:
            combined = np.concatenate([combined, silence, wav])

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = OUTPUT_DIR / f"clone-{timestamp}.wav"
        sf.write(str(out_path), combined, sample_rate)
        return FileResponse(path=str(out_path), media_type="audio/wav", filename=f"voice-clone-{timestamp}.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        for f in temp_files:
            Path(f).unlink(missing_ok=True)

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

    # Speed → text instruction: VoxCPM2 interprets this naturally
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
