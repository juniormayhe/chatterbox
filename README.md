![Chatterbox Turbo Image](./Chatterbox-Turbo.jpg)


# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_turbo_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/chatterbox-turbo-demo)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

**Chatterbox** is a family of three state-of-the-art, open-source text-to-speech models by Resemble AI.

We are excited to introduce **Chatterbox-Turbo**, our most efficient model yet. Built on a streamlined 350M parameter architecture, **Turbo** delivers high-quality speech with less compute and VRAM than our previous models. We have also distilled the speech-token-to-mel decoder, previously a bottleneck, reducing generation from 10 steps to just **one**, while retaining high-fidelity audio output.

**Paralinguistic tags** are now native to the Turbo model, allowing you to use `[cough]`, `[laugh]`, `[chuckle]`, and more to add distinct realism. While Turbo was built primarily for low-latency voice agents, it excels at narration and creative workflows.

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

<img width="1200" height="600" alt="Podonos Turbo Eval" src="https://storage.googleapis.com/chatterbox-demo-samples/turbo/podonos_turbo.png" />

### ⚡ Model Zoo

Choose the right model for your application.

| Model                                                                                                           | Size | Languages | Key Features                                            | Best For                                     | 🤗                                                                  | Examples |
|:----------------------------------------------------------------------------------------------------------------| :--- | :--- |:--------------------------------------------------------|:---------------------------------------------|:--------------------------------------------------------------------------| :--- |
| **Chatterbox-Turbo**                                                                                            | **350M** | **English** | Paralinguistic Tags (`[laugh]`), Lower Compute and VRAM | Zero-shot voice agents,  Production          | [Demo](https://huggingface.co/spaces/ResembleAI/chatterbox-turbo-demo)        | [Listen](https://resemble-ai.github.io/chatterbox_turbo_demopage/) |
| Chatterbox-Multilingual [(Language list)](#supported-languages)                                                 | 500M | 23+ | Zero-shot cloning, Multiple Languages                   | Global applications, Localization            | [Demo](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS) | [Listen](https://resemble-ai.github.io/chatterbox_demopage/) |
| Chatterbox [(Tips and Tricks)](#original-chatterbox-tips)                                                       | 500M | English | CFG & Exaggeration tuning                               | General zero-shot TTS with creative controls | [Demo](https://huggingface.co/spaces/ResembleAI/Chatterbox)              | [Listen](https://resemble-ai.github.io/chatterbox_demopage/) |

## Installation
```shell
pip install chatterbox-tts
```

Alternatively, you can install from source:
```shell
# conda create -yn chatterbox python=3.11
# conda activate chatterbox

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```
We developed and tested Chatterbox on Python 3.11 on Debian 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.

## Usage

##### Chatterbox-Turbo

```python
import torchaudio as ta
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with Paralinguistic Tags
text = "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?"

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")

ta.save("test-turbo.wav", wav, model.sr)
```

##### Chatterbox and Chatterbox-Multilingual

```python

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(spanish_text, language_id="fr")
ta.save("test-french.wav", wav_french, model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```
See `example_tts.py` and `example_vc.py` for more examples.

---

## Scripts Overview

Purpose of every script in this repository.

### Minimal examples (repo root)

| Script | Purpose |
|---|---|
| `example_tts.py` | Smallest English example with `ChatterboxTTS` plus a multilingual (French) sample; shows optional voice cloning via `audio_prompt_path`. Auto-detects `cuda`/`mps`/`cpu`. |
| `example_tts_turbo.py` | Smallest example for the **Turbo** model (`ChatterboxTurboTTS`), demonstrating paralinguistic tags such as `[chuckle]`. |
| `example_for_mac.py` | `example_tts.py` patched for Apple Silicon (MPS): forces `map_location` on `torch.load` and tunes `exaggeration`/`cfg_weight`. |
| `example_vc.py` | Smallest voice-conversion example using `ChatterboxVC` (convert a source clip to a target voice). |

### Gradio web UIs (repo root)

| Script | Purpose |
|---|---|
| `gradio_tts_app.py` | Web UI for the original `ChatterboxTTS` with exaggeration, CFG, temperature and seed controls. |
| `gradio_tts_turbo_app.py` | Web UI for Chatterbox-Turbo with clickable paralinguistic-tag buttons and advanced sampling controls. The batch pipeline mirrors this app's defaults. |
| `gradio_vc_app.py` | Web UI for voice conversion (`ChatterboxVC`); serves on port `7861`. |
| `multilingual_app.py` | Web UI for `ChatterboxMultilingualTTS` across 23+ languages, with a built-in demo prompt per language. |

### Command-line tools (repo root)

| Script | Purpose |
|---|---|
| `revoice_cli.py` | Re-voice audio: Whisper ASR → text → `ChatterboxTTS` in a target voice, avoiding pitch-warp artifacts. Optional tempo-matching. |
| `revoice_timed_cli.py` | Timing-preserving re-voice: transcribes with segment-level timestamps, synthesizes each segment, time-stretches it to the original duration, and keeps the original silence gaps. |
| `vc_cli.py` | Voice conversion: warp a source clip to a target speaker via `ChatterboxVC`. |

### Batch automation (`automation/`)

| Script | Purpose |
|---|---|
| `batch_tts.py` | Main batch orchestrator. Splits a long `.txt`/`.md` into chunks, clones the reference voice, and writes sequential `audioNN.mp3` files plus `output.log`. Paragraph-aware leading silence (600 ms at a new line, 300 ms for a mid-line continuation) and a reproducible `--seed`. |
| `fast_tts.py` | Speed-first batch variant: writes WAV to a queue folder with no loudness normalization, no silence padding, and the `max_speed` preset. |
| `text_splitter.py` | Sentence/paragraph chunking. `split_text()` returns flat chunks; `split_text_with_breaks()` tags each chunk as line-start vs. continuation so the caller can vary silence. |
| `mp3_encoder.py` | Converts a WAV tensor to MP3 with LUFS loudness normalization and optional leading silence; falls back to WAV on encoder failure. |
| `add_silence.py` | Standalone utility that prepends silence to existing MP3 files in-place (single folder or recursive). |

Detailed flags for the two main CLI tools are below; see `automation/README.md` for the full batch reference.

---

## Command-Line Tools

Two ready-to-run CLI scripts are included at the repo root for common voice workflows.

---

### `revoice_cli.py` — Re-voice Audio (Recommended)

Transcribes the input audio with **Whisper**, then synthesizes the text fresh using **ChatterboxTTS** with a target voice as the speaker reference. Because the speech is generated from scratch, there are no pitch-shifting or signal-warping artifacts.

**Pipeline:**
```
input.mp3 → Whisper ASR → text → ChatterboxTTS (target voice) → output.mp3
```

**Basic usage:**
```bash
python revoice_cli.py --input speech.mp3 --target voice_ref.mp3 --output result.mp3
```

**Show the Whisper transcript before generating:**
```bash
python revoice_cli.py -i speech.mp3 -t voice_ref.mp3 -o result.mp3 --show-transcript
```

**Higher transcription accuracy (downloads a larger Whisper model):**
```bash
python revoice_cli.py -i speech.mp3 -t voice_ref.mp3 -o result.mp3 --whisper-model small
```

**Optional tempo-matching** — attempts to match the output duration to the input. Only applied when the stretch ratio falls within the safe range (0.80–1.25); skipped automatically beyond that to avoid reverb artifacts:
```bash
python revoice_cli.py -i speech.mp3 -t voice_ref.mp3 -o result.mp3 --stretch
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` / `-i` | *required* | Input audio to transcribe (MP3, WAV, etc.) |
| `--target` / `-t` | *required* | Target voice reference audio |
| `--output` / `-o` | *required* | Output file (`.mp3` or `.wav`) |
| `--whisper-model` | `base` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large` |
| `--exaggeration` | `0.5` | TTS emotional intensity (0.0–1.0) |
| `--stretch` | off | Opt-in tempo-matching to input duration |
| `--show-transcript` | off | Print Whisper transcript before generating |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

> **Note:** Requires `transformers` for Whisper (`pip install transformers`). The Whisper model is downloaded from Hugging Face on first use.

---

### `vc_cli.py` — Voice Conversion

Transforms the voice characteristics of a source audio file to match a target speaker using **ChatterboxVC**, preserving the original speech content and timing.

**Basic usage:**
```bash
python vc_cli.py --input source.mp3 --target voice_ref.mp3 --output result.mp3
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` / `-i` | *required* | Source audio (voice to convert) |
| `--target` / `-t` | *required* | Target voice reference audio |
| `--output` / `-o` | *required* | Output file (`.mp3` or `.wav`) |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

> **Tip:** For large pitch differences between source and target (e.g. female → male), `revoice_cli.py` will generally produce cleaner results than `vc_cli.py` because it synthesizes speech fresh rather than warping the source signal.

---

## Supported Languages 
Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

## Original Chatterbox Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.


## Local Modifications (this fork)

This fork diverges from upstream Resemble AI Chatterbox with the following changes, tuned for narration batches on Windows + CUDA:

- **float32 dtype fixes** (`src/chatterbox/models/s3tokenizer/s3tokenizer.py`, `src/chatterbox/models/voice_encoder/voice_encoder.py`) — cast waveforms and mel spectrograms to `float32` before tokenization/inference to avoid dtype-mismatch errors on CUDA 12.x.
- **Loudness tuning** (`automation/mp3_encoder.py`) — default target loudness changed from `-27` to `-14 LUFS`, with a `-1 dBFS` peak limiter to prevent clipping.
- **Gradio Turbo defaults** (`gradio_tts_turbo_app.py`) — Advanced Options sliders default to the pronunciation-accuracy values used by the batch pipeline (temperature `0.5`, top-p `0.9`, top-k `100`, repetition penalty `1.1`).
- **Dependency pin** (`pyproject.toml`) — `torch`/`torchaudio` pinned to `2.5.1` for local CUDA compatibility.
- **Batch automation** (`automation/`) and the `revoice_*` CLIs — see [Scripts Overview](#scripts-overview).

## Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


## Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

## Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

## Citation
If you find this model useful, please consider citing.
```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
## Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
