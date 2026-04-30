# Voice Studio

Voice Studio is a local desktop app for turning a short reference voice clip into narrated audio on your own machine.

It wraps the open-source `VoxCPM` voice cloning stack inside an Electron desktop interface so non-technical users can:

- pick a reference voice sample
- paste a post, note, or script
- optionally rewrite the text into a slower spoken narration with Gemini
- generate and review cloned audio locally
- browse previously generated outputs from a built-in history view

## Problem This Project Tries to Solve

`Voice cloning` = generating new speech that sounds like a reference speaker  
`Voice cloning` ဆိုတာ reference voice sample တစ်ခုကိုယူပြီး အဲဒီအသံနဲ့တူတဲ့ speech အသစ်ကို generate လုပ်တာပါ။

Most open-source voice cloning projects are still hard for normal users to run because they usually require:

- command line setup
- Python environment management
- model download steps
- manual script editing
- no desktop workflow for repeated use

This project tries to solve that gap by making local voice cloning feel more like a usable desktop tool instead of a research repo.

## What Voice Studio Includes

- `Electron desktop app` = JavaScript-based desktop shell  
  `Electron desktop app` က web technology ကိုသုံးပြီး desktop app အဖြစ် run လို့ရတဲ့ shell ပါ။
- `FastAPI backend` = lightweight Python API server  
  `FastAPI backend` က Python နဲ့ရေးထားတဲ့ local API server ပါ။
- `VoxCPM runtime` = speech generation engine  
  `VoxCPM runtime` က actual voice generation လုပ်ပေးတဲ့ model/runtime ပါ။
- `Gemini rewrite preview` = optional AI script rewriting before synthesis  
  `Gemini rewrite preview` က text ကို narration style ဖြစ်အောင် optional rewrite လုပ်ပေးတဲ့ feature ပါ။
- `History browser` = built-in list of previous generated audio files  
  `History browser` က generate လုပ်ပြီးသား audio files တွေကို app ထဲကနေပြန်ကြည့်လို့ရတဲ့ feature ပါ။

## Tech Stack

- `Electron`
  Desktop window, IPC bridge, file picker, and app packaging
- `HTML / CSS / Vanilla JavaScript`
  Renderer UI for the Studio, History, and Setup tabs
- `Python`
  Local backend runtime
- `FastAPI`
  Local HTTP API between Electron and the voice engine
- `VoxCPM`
  Open-source multilingual TTS and voice cloning model stack
- `NumPy` + `SoundFile`
  Audio array handling and `.wav` export
- `Google Gemini API` optional
  AI-assisted narration rewrite and title generation
- `uv`
  Python dependency management
- `npm` / `electron-builder`
  Desktop dependency management and macOS packaging

## Sources

This project is built on top of these upstream sources:

- `OpenBMB/VoxCPM`
  [https://github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- `VoxCPM2 model`
  [https://huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- `VoxCPM documentation`
  [https://voxcpm.readthedocs.io/en/latest/](https://voxcpm.readthedocs.io/en/latest/)
- `Google Gemini API`
  [https://ai.google.dev/](https://ai.google.dev/)

## Project Structure

```text
.
├── backend/             # Local FastAPI server
├── electron/            # Desktop shell and renderer UI
├── VoxCPM/              # Upstream voice generation codebase
├── package.json         # Electron app scripts
└── README.md
```

## Local Setup

### 1. Install Node.js dependencies

```bash
npm install
```

### 2. Install Python dependencies for VoxCPM

```bash
cd VoxCPM
uv sync
cd ..
```

### 3. Prepare model weights

`Model weights` = trained AI model files used during inference  
`Model weights` ဆိုတာ AI model run ဖို့လိုတဲ့ trained data files တွေပါ။

You have two options:

- use local weights under `VoxCPM/pretrained_models/VoxCPM2`
- or let the runtime load from `openbmb/VoxCPM2` if your environment is configured for that

Important:

- large model files should not be committed to GitHub
- this repo ignores local downloaded weights by default

## Run The App

```bash
npm start
```

The Electron app will automatically start the local Python backend.

## Build macOS App

`Packaging` = turning source code into a shareable app build  
`Packaging` ဆိုတာ source code ကို share လို့ရတဲ့ app build အဖြစ်ပြောင်းတာပါ။

```bash
npm run pack:mac
```

For a zipped output:

```bash
npm run dist:mac
```

## Usage Flow

1. Open `Setup`
2. Choose a reference audio file
3. Optionally add the exact transcript of that reference clip for better similarity
4. Paste your target text in `Studio`
5. Optionally enable Gemini rewrite to convert written text into a more spoken narration style
6. Click `Generate`
7. Review the output and open the output folder if needed

## Open Source Notes

- local model files are ignored from Git
- generated audio outputs are ignored from Git
- local virtual environments and caches are ignored from Git
- Gemini API usage is optional and requires your own API key

## Responsible Use

`Consent` = permission from the original speaker  
`Consent` ဆိုတာ original speaker ဆီက သဘောတူခွင့်ပြုချက် ရထားတာပါ။

Please use this project responsibly:

- only clone voices when you have permission
- do not impersonate people without consent
- clearly disclose synthetic audio when appropriate
- review your local laws and platform policies before publishing generated content

## License

This repository is released under the Apache 2.0 License.

Please also review the upstream licenses and notices from VoxCPM and any other third-party dependencies you redistribute.
