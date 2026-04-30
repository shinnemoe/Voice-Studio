#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
CSS_TEXT = (ROOT / "local_ui.css").read_text(encoding="utf-8")

from voxcpm import VoxCPM  # noqa: E402


LOCAL_MODEL_PATH = str(ROOT / "pretrained_models" / "VoxCPM2")
DEFAULT_MODEL_ID = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else "openbmb/VoxCPM2"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_PRESETS = {
    "Fast": {"inference_timesteps": 6, "cfg_value": 1.8},
    "Balanced": {"inference_timesteps": 10, "cfg_value": 2.0},
    "High Similarity": {"inference_timesteps": 16, "cfg_value": 2.3},
}


@dataclass
class ModelSession:
    source: str | None = None
    device: str = "auto"
    model: VoxCPM | None = None


MODEL_SESSION = ModelSession()


def normalize_text(target_text: str, style_text: str) -> str:
    cleaned_target = " ".join((target_text or "").strip().split())
    cleaned_style = " ".join((style_text or "").strip().split())
    if not cleaned_target:
        raise gr.Error("Target text is required.")
    return f"({cleaned_style}){cleaned_target}" if cleaned_style else cleaned_target


def resolve_model_source(model_source: str) -> str:
    cleaned = (model_source or "").strip()
    return cleaned or DEFAULT_MODEL_ID


def get_or_load_model(model_source: str, device: str) -> tuple[VoxCPM, str]:
    resolved_source = resolve_model_source(model_source)
    resolved_device = device or "auto"

    if (
        MODEL_SESSION.model is not None
        and MODEL_SESSION.source == resolved_source
        and MODEL_SESSION.device == resolved_device
    ):
        return MODEL_SESSION.model, f"Model ready: {resolved_source} on {resolved_device}"

    MODEL_SESSION.model = VoxCPM.from_pretrained(
        hf_model_id=resolved_source,
        load_denoiser=False,
        device=None if resolved_device == "auto" else resolved_device,
    )
    MODEL_SESSION.source = resolved_source
    MODEL_SESSION.device = resolved_device
    return MODEL_SESSION.model, f"Model loaded: {resolved_source} on {resolved_device}"


def save_waveform(wav, sample_rate: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = OUTPUT_DIR / f"clone-{timestamp}.wav"
    sf.write(output_path, wav, sample_rate)
    return output_path


def generate_clone(
    reference_audio: str | None,
    target_text: str,
    style_text: str,
    prompt_text: str,
    quality: str,
    model_source: str,
    device: str,
):
    if not reference_audio:
        raise gr.Error("Reference audio is required.")

    full_text = normalize_text(target_text, style_text)
    prompt_text = " ".join((prompt_text or "").strip().split()) or None
    preset = QUALITY_PRESETS[quality]
    model, status = get_or_load_model(model_source, device)

    wav = model.generate(
        text=full_text,
        prompt_wav_path=reference_audio if prompt_text else None,
        prompt_text=prompt_text,
        reference_wav_path=reference_audio,
        inference_timesteps=preset["inference_timesteps"],
        cfg_value=preset["cfg_value"],
    )

    output_path = save_waveform(wav, model.tts_model.sample_rate)
    details = [
        status,
        f"Quality preset: {quality}",
        f"Saved file: {output_path}",
    ]
    if prompt_text:
        details.append("Mode: high similarity clone with transcript")
    else:
        details.append("Mode: quick clone")

    return str(output_path), "\n".join(details), str(output_path)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Local Voice Cloner") as demo:
        gr.HTML(f"<style>{CSS_TEXT}</style>")
        gr.Markdown(
            """
            # Local Voice Cloner
            Minimal workflow for VoxCPM2:
            1. Upload a short reference voice
            2. Type the target text
            3. Click generate
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                reference_audio = gr.Audio(
                    label="Reference audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                target_text = gr.Textbox(
                    label="Target text",
                    lines=5,
                    placeholder="Type what the cloned voice should say.",
                )
                style_text = gr.Textbox(
                    label="Optional style control",
                    lines=2,
                    placeholder="Example: warm, calm, slightly slower",
                )
                prompt_text = gr.Textbox(
                    label="Optional reference transcript",
                    lines=3,
                    placeholder="Paste the exact words from the reference audio for higher similarity.",
                )
            with gr.Column(scale=3):
                quality = gr.Radio(
                    choices=list(QUALITY_PRESETS.keys()),
                    value="Balanced",
                    label="Quality preset",
                )
                model_source = gr.Textbox(
                    label="Model source",
                    value=DEFAULT_MODEL_ID,
                    info="Use a Hugging Face repo id or local model path.",
                )
                device = gr.Dropdown(
                    choices=["auto", "cpu", "mps", "cuda"],
                    value="auto",
                    label="Device",
                )
                generate_button = gr.Button("Generate Clone", variant="primary")

        with gr.Row():
            output_audio = gr.Audio(label="Generated audio", type="filepath")
            status_box = gr.Textbox(label="Run status", lines=6)
            output_path = gr.Textbox(label="Saved file path")

        generate_button.click(
            fn=generate_clone,
            inputs=[
                reference_audio,
                target_text,
                style_text,
                prompt_text,
                quality,
                model_source,
                device,
            ],
            outputs=[output_audio, status_box, output_path],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    app = build_interface()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
    )
