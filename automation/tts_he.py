#!/usr/bin/env python3
"""Generate Hebrew speech with the standard multilingual Chatterbox model."""

from datetime import datetime
from pathlib import Path
import os
import sys

# Must be set before importing torch: the MPS backend lacks a few operations
# used both for reference-audio resampling and final waveform decoding.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

VOICE_REFERENCE = Path(
    "/Users/junior/Library/CloudStorage/OneDrive-Personal/Negocios/"
    "High Magick Practices/voice.mp3"
)


def main() -> int:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print('Usage: tts_he "Hebrew text"', file=sys.stderr)
        return 2
    if not VOICE_REFERENCE.is_file():
        print(f"Voice reference audio not found: {VOICE_REFERENCE}", file=sys.stderr)
        return 1

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading standard Chatterbox multilingual model on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    audio = model.generate(
        text,
        language_id="he",
        audio_prompt_path=str(VOICE_REFERENCE),
    )

    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    output = downloads / f"tts-he-{datetime.now():%Y%m%d-%H%M%S}.wav"
    torchaudio.save(str(output), audio, model.sr)
    print(f"Saved Hebrew audio to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
