from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import warnings
import requests

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
VOXCPM_DIR = ROOT / "VoxCPM"
SRC_DIR = VOXCPM_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voxcpm import VoxCPM  # noqa: E402

warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*librosa.core.audio.__audioread_load.*")
warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

LOCAL_MODEL_PATH = str(VOXCPM_DIR / "pretrained_models" / "VoxCPM2")
DEFAULT_MODEL_ID = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "openbmb/VoxCPM2"
OUTPUT_DIR = VOXCPM_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_PRESETS = {
    "Fast": {"inference_timesteps": 6, "cfg_value": 1.8},
    "Balanced": {"inference_timesteps": 10, "cfg_value": 2.0},
    "High Similarity": {"inference_timesteps": 16, "cfg_value": 2.3},
}

STYLE_PRESETS = {
    "Natural": "natural spoken delivery, clear and grounded",
    "Deep Reflective": "deep reflective delivery, calm, philosophical, deliberate, intimate, unhurried, with thoughtful pauses",
    "Warm Storyteller": "warm storyteller delivery, grounded, expressive, gentle pauses, steady medium-slow pacing",
    "Soft Intimate": "soft intimate delivery, tender, close, quiet, slow and emotionally controlled",
    "Documentary": "documentary narration delivery, deep, composed, deliberate rhythm, serious tone",
}

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass
class ModelSession:
    source: str | None = None
    device: str = "auto"
    model: VoxCPM | None = None
    prompt_cache_key: tuple | None = None
    prompt_cache: dict | None = None


MODEL_SESSION = ModelSession()

app = FastAPI(title="Local Voice Cloner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")


def normalize_text(target_text: str) -> str:
    cleaned_target = " ".join((target_text or "").strip().split())
    if not cleaned_target:
        raise HTTPException(status_code=400, detail="Target text is required.")
    return cleaned_target


def resolve_model_source(model_source: str) -> str:
    cleaned = (model_source or "").strip()
    return cleaned or DEFAULT_MODEL_ID


def detect_language_hint(text: str) -> str:
    if re.search(r"[\u1000-\u109F]", text):
        return "Burmese"
    return "the same language as the input"


def get_or_load_model(model_source: str, device: str) -> tuple[VoxCPM, str]:
    resolved_source = resolve_model_source(model_source)
    resolved_device = (device or "auto").strip() or "auto"

    if (
        MODEL_SESSION.model is not None
        and MODEL_SESSION.source == resolved_source
        and MODEL_SESSION.device == resolved_device
    ):
        return MODEL_SESSION.model, f"Model ready in memory: {resolved_source} on {resolved_device}"

    MODEL_SESSION.model = VoxCPM.from_pretrained(
        hf_model_id=resolved_source,
        load_denoiser=False,
        device=None if resolved_device == "auto" else resolved_device,
        optimize=False,
    )
    MODEL_SESSION.source = resolved_source
    MODEL_SESSION.device = resolved_device
    MODEL_SESSION.prompt_cache_key = None
    MODEL_SESSION.prompt_cache = None
    return MODEL_SESSION.model, f"Model loaded: {resolved_source} on {resolved_device}"


def get_or_build_prompt_cache(
    model: VoxCPM,
    reference_audio_path: str,
    prompt_text: str | None,
) -> tuple[dict, str]:
    cache_key = (
        MODEL_SESSION.source,
        MODEL_SESSION.device,
        reference_audio_path,
        prompt_text or "",
    )

    if MODEL_SESSION.prompt_cache is not None and MODEL_SESSION.prompt_cache_key == cache_key:
        return MODEL_SESSION.prompt_cache, "Reference cache reused"

    prompt_cache = model.tts_model.build_prompt_cache(
        prompt_text=prompt_text,
        prompt_wav_path=reference_audio_path if prompt_text else None,
        reference_wav_path=reference_audio_path,
    )
    MODEL_SESSION.prompt_cache = prompt_cache
    MODEL_SESSION.prompt_cache_key = cache_key
    return prompt_cache, "Reference cache built"


def save_waveform(wav, sample_rate: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = OUTPUT_DIR / f"clone-{timestamp}.wav"
    sf.write(output_path, wav, sample_rate)
    return output_path


def sanitize_filename_part(text: str, fallback: str = "voice-clone") -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"[\\/:*?\"<>|]", "", cleaned)
    cleaned = cleaned.strip(" .-_")
    return cleaned[:40] or fallback


def guess_title_from_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return "voice-clone"

    primary = re.split(r"[.!?।…\n]", normalized, maxsplit=1)[0].strip()
    if not primary:
        primary = normalized

    if re.search(r"[\u1000-\u109F]", primary):
        words = primary.split()
        return sanitize_filename_part(" ".join(words[:4]), fallback="voice-clone")

    words = re.findall(r"[A-Za-z0-9']+", primary)
    return sanitize_filename_part("-".join(words[:5]), fallback="voice-clone")


def build_title_with_gemini(
    source_text: str,
    rewritten_text: str,
    gemini_api_key: str,
    gemini_model: str,
) -> str | None:
    api_key = (gemini_api_key or "").strip()
    if not api_key:
        return None

    model_name = (gemini_model or "").strip() or DEFAULT_GEMINI_MODEL
    language_hint = detect_language_hint(source_text or rewritten_text)
    prompt = (
        "Create a very short file title for an audio narration. "
        "Keep the same language as the input. Do not translate. "
        f"If the content is in {language_hint}, keep the title fully in {language_hint}. "
        "Return only a short usable title, ideally 2 to 5 words, without quotes, hashtags, emojis, or explanation. "
        "Do not include file extensions or dates.\n\n"
        f"Original:\n{source_text.strip()}\n\n"
        f"Narration:\n{rewritten_text.strip()}"
    )

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        },
        timeout=30,
    )

    if not response.ok:
        return None

    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        return None

    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    title_text = " ".join(part.get("text", "") for part in parts if part.get("text")).strip()
    if not title_text:
        return None
    return sanitize_filename_part(title_text, fallback="voice-clone")


def save_waveform_with_context(
    wav,
    sample_rate: int,
    source_text: str,
    rewritten_text: str,
    gemini_api_key: str,
    gemini_model: str,
) -> Path:
    title = build_title_with_gemini(
        source_text=source_text,
        rewritten_text=rewritten_text,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    ) or guess_title_from_text(rewritten_text or source_text)
    date_suffix = datetime.now().strftime("(%-d.%-m.%y)")
    output_path = OUTPUT_DIR / f"{title} {date_suffix}.wav"
    suffix = 2
    while output_path.exists():
        output_path = OUTPUT_DIR / f"{title} {date_suffix}-{suffix}.wav"
        suffix += 1
    sf.write(output_path, wav, sample_rate)
    return output_path


def rewrite_with_gemini(
    source_text: str,
    gemini_api_key: str,
    gemini_model: str,
    style_preset: str,
    style_text: str,
    target_wpm: int,
) -> str:
    api_key = (gemini_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required when rewrite is enabled.")

    model_name = (gemini_model or "").strip() or DEFAULT_GEMINI_MODEL
    language_hint = detect_language_hint(source_text)
    style_hint = " ".join(
        part.strip()
        for part in [
            STYLE_PRESETS.get(style_preset or "Natural", ""),
            (style_text or "").strip(),
        ]
        if part and part.strip()
    )
    prompt = (
        "Rewrite the following short post into a spoken narration script for voice synthesis. "
        "Keep the core idea intact, but expand it into a clear, reflective explanation with natural pauses. "
        "Preserve the same language as the input. Do not translate. "
        f"If the input is in {language_hint}, the output must stay fully in {language_hint}. "
        "Use punctuation, line breaks, commas, dashes, ellipses, and short sentences to create audible pauses and slower delivery. "
        "Write for spoken narration, not for reading silently. "
        "Avoid hype, avoid slang, avoid mentioning any public figure, and do not describe stage directions. "
        "If the input is short, expand it helpfully without drifting away from the original meaning. "
        "Add quiet excitement through wording and rhythm only. "
        f"Target pacing: about {target_wpm} words per minute, so the script should feel unhurried and slower than typical speech. "
        f"Delivery style: {style_hint or 'clear, grounded, and natural spoken delivery'}. "
        "Important: return only the rewritten narration text. Do not include labels, instructions, or style notes.\n\n"
        f"Post:\n{source_text.strip()}"
    )

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        },
        timeout=60,
    )

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini rewrite failed: {response.status_code} {response.text[:400]}",
        )

    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise HTTPException(status_code=502, detail="Gemini rewrite returned no candidates.")

    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    rewritten_text = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
    if not rewritten_text:
        raise HTTPException(status_code=502, detail="Gemini rewrite returned empty text.")

    return rewritten_text


@app.post("/rewrite-preview")
def rewrite_preview(
    target_text: str = Form(...),
    style_text: str = Form(""),
    style_preset: str = Form("Natural"),
    gemini_api_key: str = Form(""),
    gemini_model: str = Form(DEFAULT_GEMINI_MODEL),
    target_wpm: int = Form(105),
):
    source_text = normalize_text(target_text)
    rewritten_text = rewrite_with_gemini(
        source_text,
        gemini_api_key,
        gemini_model,
        style_preset,
        style_text,
        target_wpm,
    )
    return JSONResponse({"rewritten_text": rewritten_text})


def tensor_to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu()
        if hasattr(wav, "squeeze"):
            wav = wav.squeeze(0)
        wav = wav.numpy()
    return np.asarray(wav, dtype=np.float32)


def split_long_text(text: str, max_chars: int = 220) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentence_parts = re.split(r"(?<=[\.\!\?။…])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentence_parts:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            words = sentence.split(" ")
            word_chunk = ""
            for word in words:
                candidate = f"{word_chunk} {word}".strip()
                if word_chunk and len(candidate) > max_chars:
                    chunks.append(word_chunk)
                    word_chunk = word
                else:
                    word_chunk = candidate
            if word_chunk:
                if current and len(f"{current} {word_chunk}") <= max_chars:
                    current = f"{current} {word_chunk}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = word_chunk
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def synthesize_long_form(
    model: VoxCPM,
    base_prompt_cache: dict,
    text: str,
    inference_timesteps: int,
    cfg_value: float,
) -> np.ndarray:
    text_chunks = split_long_text(text, max_chars=170)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="Target text is empty after cleanup.")

    audio_chunks: list[np.ndarray] = []
    silence = np.zeros(int(model.tts_model.sample_rate * 0.12), dtype=np.float32)

    for index, text_chunk in enumerate(text_chunks):
        wav_tensor, _generated_text_tokens, _pred_audio_feat = model.tts_model.generate_with_prompt_cache(
            target_text=text_chunk,
            prompt_cache=base_prompt_cache,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            max_len=2048,
        )

        wav_np = tensor_to_numpy(wav_tensor)
        if wav_np.size == 0:
            raise RuntimeError("Generated audio was empty.")

        audio_chunks.append(wav_np)
        if index < len(text_chunks) - 1:
            audio_chunks.append(silence)

    return np.concatenate(audio_chunks)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.get("/history")
def history():
    items = []
    for wav_path in sorted(OUTPUT_DIR.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True):
        stat = wav_path.stat()
        items.append(
            {
                "name": wav_path.name,
                "title": wav_path.stem,
                "output_path": str(wav_path),
                "audio_url": f"http://127.0.0.1:{PORT}/files/{wav_path.name}",
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "size_bytes": stat.st_size,
            }
        )
    return JSONResponse({"items": items})


@app.post("/generate")
def generate(
    reference_audio_path: str = Form(...),
    target_text: str = Form(...),
    style_text: str = Form(""),
    style_preset: str = Form("Natural"),
    prepared_script: str = Form(""),
    prompt_text: str = Form(""),
    quality: str = Form("Balanced"),
    model_source: str = Form(DEFAULT_MODEL_ID),
    device: str = Form("auto"),
    rewrite_with_ai: str = Form("false"),
    gemini_api_key: str = Form(""),
    gemini_model: str = Form(DEFAULT_GEMINI_MODEL),
    target_wpm: int = Form(105),
):
    reference_path = Path(reference_audio_path)
    if not reference_path.exists():
        raise HTTPException(status_code=400, detail="Reference audio file does not exist.")

    if quality not in QUALITY_PRESETS:
        raise HTTPException(status_code=400, detail="Unsupported quality preset.")

    source_text = " ".join((target_text or "").strip().split())
    if not source_text:
        raise HTTPException(status_code=400, detail="Target text is required.")

    rewrite_enabled = (rewrite_with_ai or "").strip().lower() == "true"
    prepared_script_clean = normalize_text(prepared_script) if (prepared_script or "").strip() else ""
    if rewrite_enabled and prepared_script_clean:
        rewritten_text = prepared_script_clean
    elif rewrite_enabled:
        rewritten_text = rewrite_with_gemini(
            source_text,
            gemini_api_key,
            gemini_model,
            style_preset,
            style_text,
            target_wpm,
        )
    else:
        rewritten_text = source_text

    full_text = normalize_text(rewritten_text)
    full_text = re.sub(r"\s+", " ", full_text.replace("\n", " ")).strip()
    cleaned_prompt = " ".join((prompt_text or "").strip().split()) or None
    preset = QUALITY_PRESETS[quality]

    try:
        model, status = get_or_load_model(model_source, device)
        prompt_cache, prompt_cache_status = get_or_build_prompt_cache(
            model,
            str(reference_path),
            cleaned_prompt,
        )
        wav = synthesize_long_form(
            model,
            prompt_cache,
            full_text,
            inference_timesteps=preset["inference_timesteps"],
            cfg_value=preset["cfg_value"],
        )
        output_path = save_waveform_with_context(
            wav,
            model.tts_model.sample_rate,
            source_text=source_text,
            rewritten_text=rewritten_text,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    mode = "high similarity clone with transcript" if cleaned_prompt else "quick clone"
    return JSONResponse(
        {
            "status": "\n".join(
                [
                    status,
                    prompt_cache_status,
                    f"Quality preset: {quality}",
                    f"Style preset: {style_preset}",
                    f"Target WPM: {target_wpm}",
                    f"Rewrite with Gemini: {'on' if rewrite_enabled else 'off'}",
                    f"Mode: {mode}",
                    f"Saved file: {output_path}",
                ]
            ),
            "output_path": str(output_path),
            "audio_url": f"http://127.0.0.1:{PORT}/files/{output_path.name}",
            "rewritten_text": rewritten_text,
        }
    )


PORT = 7862


if __name__ == "__main__":
    import os

    PORT = int(os.environ.get("PORT", "7862"))
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
