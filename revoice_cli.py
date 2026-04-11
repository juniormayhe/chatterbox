"""
Re-voice CLI — Whisper + ChatterboxTTS Pipeline

Transcribes the input audio to text with Whisper, then synthesizes that text
fresh using ChatterboxTTS with the target voice as the speaker reference.

By default the output plays at the TTS natural pace (cleanest audio).
Use --stretch to attempt tempo-matching the input duration; it is only applied
when the ratio falls within the safe range (0.80–1.25) to avoid reverb artifacts.

Usage:
    python revoice_cli.py --input female.mp3 --target male.mp3 --output result.mp3
    python revoice_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --show-transcript
    python revoice_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --whisper-model small --stretch
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import librosa
import torch
import torchaudio

from transformers import pipeline as hf_pipeline
from chatterbox.tts import ChatterboxTTS
from automation.mp3_encoder import save_as_mp3
from automation.text_splitter import split_text


# Only stretch within this range — phase vocoder artifacts become audible beyond it
STRETCH_RATE_MIN = 0.80
STRETCH_RATE_MAX = 1.25
TTS_CHUNK_MAX_CHARS = 300


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_audio_duration(path: str) -> float:
    """Return duration in seconds using librosa at 24 kHz."""
    y, _ = librosa.load(path, sr=24000)
    return len(y) / 24000


def transcribe(audio_path: str, whisper_model: str, device: str) -> str:
    """Transcribe audio to text using the Whisper model via transformers.

    chunk_length_s=30 enables long-form transcription for audio over 30 seconds
    by splitting into overlapping 30s chunks internally.
    stride_length_s=5 adds a 5s overlap at each boundary to avoid cut-off words.
    """
    model_id = f"openai/whisper-{whisper_model}"
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    result = pipe(str(audio_path))
    text = result["text"].strip()
    if not text:
        raise ValueError("Whisper returned an empty transcript — is the input audio silent?")
    return text


def generate_tts(
    text: str,
    model: ChatterboxTTS,
    target_path: str,
    exaggeration: float,
) -> torch.Tensor:
    """
    Synthesize text in the target voice.
    For short text: single generate() call.
    For long text: split into chunks, prepare voice once, generate per chunk, concatenate.
    """
    if len(text) <= TTS_CHUNK_MAX_CHARS:
        return model.generate(
            text,
            audio_prompt_path=target_path,
            exaggeration=exaggeration,
        )

    # Long text — split and generate in chunks
    chunks = split_text(text, max_chars=TTS_CHUNK_MAX_CHARS)
    if not chunks:
        raise ValueError("Text splitting produced no chunks.")

    print(f"  Text split into {len(chunks)} chunk(s)")

    # Embed target voice once before the loop (not on every chunk)
    model.prepare_conditionals(target_path, exaggeration=exaggeration)

    wavs = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
        wav = model.generate(chunk, exaggeration=exaggeration)
        wavs.append(wav)

    # Concatenate along time axis — all tensors are [1, N] at model.sr
    return torch.cat(wavs, dim=1)


def time_stretch_to_duration(
    wav: torch.Tensor,
    sr: int,
    input_duration: float,
) -> torch.Tensor:
    """
    Time-stretch wav to approximately match input_duration, but only when the
    required ratio stays within the safe range (STRETCH_RATE_MIN–STRETCH_RATE_MAX).

    rate = tts_duration / input_duration:
      > 1 → speed up  (TTS longer than input)
      < 1 → slow down (TTS shorter than input)

    Outside the safe range the phase vocoder produces audible reverb/smearing,
    so the audio is returned unchanged and a warning is printed instead.

    n_fft=512 (vs librosa default 2048) reduces temporal smearing for the
    ratios that do fall within the safe range.
    """
    tts_duration = wav.shape[-1] / sr

    if tts_duration <= 0 or input_duration <= 0:
        return wav

    rate = tts_duration / input_duration
    print(f"  Stretch ratio: {rate:.3f}  (safe range: {STRETCH_RATE_MIN}–{STRETCH_RATE_MAX})")

    if rate < STRETCH_RATE_MIN or rate > STRETCH_RATE_MAX:
        print(f"  Ratio outside safe range — skipping stretch to avoid reverb artifacts")
        print(f"  Output will play at TTS natural pace ({tts_duration:.2f}s)")
        return wav

    wav_np = wav.squeeze(0).numpy()
    stretched = librosa.effects.time_stretch(wav_np, rate=rate, n_fft=512)
    return torch.from_numpy(stretched).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-voice audio: transcribe with Whisper, synthesize with ChatterboxTTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python revoice_cli.py --input female.mp3 --target male.mp3 --output result.mp3
  python revoice_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --whisper-model small --show-transcript
  python revoice_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --stretch
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input audio to transcribe (MP3, WAV, etc.)"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to target voice reference audio"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output audio file (.mp3 or .wav)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda, mps, or cpu (default: auto-detected)"
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Attempt to match input duration via time-stretch (only applied within safe ratio 0.80–1.25)"
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotional intensity for TTS, 0.0–1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--show-transcript",
        action="store_true",
        help="Print the Whisper transcript before generating audio"
    )
    return parser.parse_args()


def main():
    # Fix Unicode on Windows
    if sys.platform == "win32":
        os.system("chcp 65001 > nul")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        args = parse_args()
        device = args.device if args.device is not None else detect_device()

        input_path  = Path(args.input)
        target_path = Path(args.target)
        output_path = Path(args.output)

        # --- Validate files BEFORE loading any model ---
        if not input_path.exists():
            print(f"✗ Input file not found: {input_path}")
            sys.exit(1)

        if not target_path.exists():
            print(f"✗ Target voice file not found: {target_path}")
            sys.exit(1)

        output_ext = output_path.suffix.lower()
        if output_ext not in (".mp3", ".wav"):
            print(f"✗ Unsupported output format '{output_ext}'. Use .mp3 or .wav")
            sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Measure input duration (fast, no model needed) ---
        input_duration = None
        if args.stretch:
            print("Measuring input duration...")
            input_duration = get_audio_duration(str(input_path))
            print(f"  Input duration: {input_duration:.2f}s")

        # --- Step 1: Transcribe ---
        print(f"Transcribing with Whisper ({args.whisper_model}) on {device}...")
        text = transcribe(str(input_path), args.whisper_model, device)
        print(f"  Transcript: {len(text)} chars")

        if args.show_transcript:
            print(f"\nTranscript:\n  {text}\n")

        # --- Step 2: Load TTS model ---
        print(f"Loading ChatterboxTTS on {device}...")
        model = ChatterboxTTS.from_pretrained(device)
        print("  Model loaded")

        # --- Step 3: Generate speech ---
        print("Generating speech...")
        print(f"  Target voice: {target_path}")
        wav = generate_tts(text, model, str(target_path), args.exaggeration)
        tts_duration = wav.shape[-1] / model.sr
        print(f"  TTS duration: {tts_duration:.2f}s")

        # --- Step 4: Time-stretch to match input pace (opt-in) ---
        if args.stretch:
            print(f"Time-stretching to match input ({input_duration:.2f}s)...")
            wav = time_stretch_to_duration(wav, model.sr, input_duration)
            print(f"  Output duration: {wav.shape[-1] / model.sr:.2f}s")

        # --- Step 5: Save ---
        print(f"Saving output to: {output_path}")
        if output_ext == ".mp3":
            save_as_mp3(wav, model.sr, str(output_path))
        else:
            torchaudio.save(str(output_path), wav, model.sr)
        print("  Done")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
