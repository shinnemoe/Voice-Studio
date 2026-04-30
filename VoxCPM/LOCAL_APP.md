# Local Voice Cloner

This repo now includes a minimal local app entrypoint: `minimal_app.py`.

## What it does

- Quick clone:
  - upload reference audio
  - type target text
  - generate output
- High similarity clone:
  - paste the exact transcript of the reference audio
  - VoxCPM uses the reference clip as both prompt audio and reference audio
- Saves each result into `outputs/`

## Run locally

### 1. Install dependencies

If you use `uv`:

```bash
uv sync
```

If you use plain Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Start the app

```bash
uv run python minimal_app.py
```

Or:

```bash
python minimal_app.py
```

The UI runs on `http://127.0.0.1:7861`.

## Notes

- Default model source is `openbmb/VoxCPM2`
- First run may take time because model weights download and model warmup happen on demand
- `Device` options:
  - `auto`: automatic device selection
  - `cpu`: slower but simplest
  - `mps`: Apple Silicon GPU path on macOS
  - `cuda`: NVIDIA GPU path
