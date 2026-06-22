"""
Re-voice Timed CLI — Whisper + ChatterboxTTS with timing preservation

Transcribes the input audio using Whisper with segment-level timestamps, synthesizes
each segment independently using ChatterboxTTS with the target voice, time-stretches
each segment to match the original segment duration, then stitches them back together
with the original silence gaps preserved.

Usage:
    python revoice_timed_cli.py --input speech.mp3 --target ref.mp3 --output result.mp3
    python revoice_timed_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --show-transcript
    python revoice_timed_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --force-stretch
    python revoice_timed_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --min-pause-ms 200
"""

import sys
import os
import argparse
from pathlib import Path

import librosa
import torch
import torchaudio

from transformers import pipeline as hf_pipeline
from chatterbox.tts import ChatterboxTTS
from automation.mp3_encoder import save_as_mp3


# Only stretch within this range — phase vocoder artifacts become audible beyond it
STRETCH_RATE_MIN = 0.80
STRETCH_RATE_MAX = 1.25


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def transcribe_with_segments(audio_path: str, whisper_model: str, device: str) -> list[dict]:
    """Transcribe audio and return segment-level timestamps.

    Returns a list of dicts: [{"text": str, "start": float, "end": float | None}, ...]
    end may be None for the last segment if Whisper cannot determine it.

    chunk_length_s=30 + stride_length_s=5 mirrors the long-form setup in revoice_cli.py.
    return_timestamps=True requests sentence/phrase-level chunk timestamps.
    """
    model_id = f"openai/whisper-{whisper_model}"
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    result = pipe(str(audio_path), return_timestamps=True)

    chunks = result.get("chunks") or []
    if not chunks:
        raise ValueError("Whisper returned no segments — is the input audio silent?")

    segments = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        ts = chunk.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        if start is None:
            continue
        segments.append({
            "text": text,
            "start": float(start),
            "end": float(end) if end is not None else None,
        })

    if not segments:
        raise ValueError("Whisper returned no segments with valid timestamps.")

    return segments


def make_silence(duration_s: float, sr: int) -> torch.Tensor:
    """Return a zero tensor of shape [1, N] representing silence."""
    if duration_s <= 0:
        return torch.zeros(1, 0)
    return torch.zeros(1, int(sr * duration_s))


def stretch_segment(
    wav: torch.Tensor,
    sr: int,
    target_duration: float | None,
    force_stretch: bool,
) -> torch.Tensor:
    """Time-stretch wav to target_duration seconds.

    rate = tts_duration / target_duration:
      > 1 → speed up  (TTS longer than original)
      < 1 → slow down (TTS shorter than original)

    Skips stretch and returns wav unchanged when:
      - target_duration is None or <= 0
      - ratio is outside [STRETCH_RATE_MIN, STRETCH_RATE_MAX] and force_stretch is False
    """
    if target_duration is None or target_duration <= 0:
        return wav

    tts_duration = wav.shape[-1] / sr
    if tts_duration <= 0:
        return wav

    rate = tts_duration / target_duration
    in_range = STRETCH_RATE_MIN <= rate <= STRETCH_RATE_MAX

    if not force_stretch and not in_range:
        print(f"    Ratio {rate:.3f} outside safe range [{STRETCH_RATE_MIN}–{STRETCH_RATE_MAX}] — keeping natural pace")
        return wav

    wav_np = wav.squeeze(0).numpy()
    stretched = librosa.effects.time_stretch(wav_np, rate=rate, n_fft=512)
    return torch.from_numpy(stretched).unsqueeze(0)


def build_timed_output(
    segments: list[dict],
    stretched_wavs: list[torch.Tensor],
    sr: int,
    min_pause_s: float,
) -> torch.Tensor:
    """Assemble TTS segments with original silence gaps into a single tensor.

    Inserts leading silence if the first segment starts after time 0,
    and inter-segment silence for each gap that exceeds min_pause_s.
    Negative gaps (Whisper timestamp overlaps) are clamped to zero.
    """
    parts: list[torch.Tensor] = []

    # Leading silence before the first segment
    leading = segments[0]["start"]
    if leading > min_pause_s:
        parts.append(make_silence(leading, sr))

    for i, (seg, wav) in enumerate(zip(segments, stretched_wavs)):
        parts.append(wav)

        if i < len(segments) - 1:
            next_start = segments[i + 1]["start"]
            seg_end = seg["end"]

            if seg_end is not None:
                gap = max(next_start - seg_end, 0.0)
            else:
                # Unknown segment end — no gap inserted; segments abut
                gap = 0.0

            if gap > min_pause_s:
                parts.append(make_silence(gap, sr))

    # Drop zero-width tensors so torch.cat never sees empty trailing entries
    parts = [p for p in parts if p.shape[-1] > 0]
    if not parts:
        raise ValueError("No audio parts to assemble — all segments were empty.")

    return torch.cat(parts, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-voice audio preserving original timing and pauses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python revoice_timed_cli.py --input speech.mp3 --target ref.mp3 --output result.mp3
  python revoice_timed_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --show-transcript
  python revoice_timed_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --force-stretch
  python revoice_timed_cli.py -i speech.mp3 -t ref.mp3 -o out.mp3 --min-pause-ms 200
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input audio to transcribe (MP3, WAV, etc.)",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to target voice reference audio",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output audio file (.mp3 or .wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda, mps, or cpu (default: auto-detected)",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotional intensity for TTS, 0.0–1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--show-transcript",
        action="store_true",
        help="Print Whisper segments with timestamps before generating audio",
    )
    parser.add_argument(
        "--force-stretch",
        action="store_true",
        help="Stretch all segments regardless of ratio (bypasses 0.80–1.25 safe range)",
    )
    parser.add_argument(
        "--min-pause-ms",
        type=int,
        default=50,
        help="Minimum gap (ms) to insert as silence; filters sub-threshold Whisper imprecision (default: 50)",
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
        min_pause_s = args.min_pause_ms / 1000.0

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

        # --- Step 1: Transcribe with segment timestamps ---
        print(f"Transcribing with Whisper ({args.whisper_model}) on {device}...")
        segments = transcribe_with_segments(str(input_path), args.whisper_model, device)

        # Drop segments with empty text; preserve their gap via surrounding timestamps
        valid_segments = [s for s in segments if s["text"]]
        if not valid_segments:
            print("✗ Whisper returned no non-empty segments.")
            sys.exit(1)

        print(f"  {len(valid_segments)} segment(s) detected")
        if len(valid_segments) == 1:
            print("  Warning: only one segment — no inter-segment pauses to preserve")

        if args.show_transcript:
            print("\nSegments:")
            for s in valid_segments:
                end_label = f"{s['end']:.2f}s" if s["end"] is not None else "?"
                print(f"  [{s['start']:.2f}s – {end_label}] {s['text']}")
            print()

        # --- Step 2: Load TTS model ---
        print(f"Loading ChatterboxTTS on {device}...")
        model = ChatterboxTTS.from_pretrained(device)
        print("  Model loaded")

        # --- Step 3: Embed target voice once (not per segment) ---
        print(f"Preparing voice: {target_path}")
        model.prepare_conditionals(str(target_path), exaggeration=args.exaggeration)

        # --- Step 4: Generate + stretch per segment ---
        print("Generating and stretching segments...")
        stretched_wavs: list[torch.Tensor] = []
        for i, seg in enumerate(valid_segments, 1):
            target_dur = (seg["end"] - seg["start"]) if seg["end"] is not None else None
            dur_label = f"{target_dur:.2f}s" if target_dur is not None else "unknown duration"
            print(f"  Segment {i}/{len(valid_segments)} [{seg['start']:.2f}s – {dur_label}] ({len(seg['text'])} chars)")

            wav = model.generate(seg["text"], exaggeration=args.exaggeration)
            wav = stretch_segment(wav, model.sr, target_dur, args.force_stretch)
            stretched_wavs.append(wav)

        # --- Step 5: Assemble with original silence gaps ---
        print("Assembling output with original timing...")
        output_wav = build_timed_output(valid_segments, stretched_wavs, model.sr, min_pause_s)
        print(f"  Output duration: {output_wav.shape[-1] / model.sr:.2f}s")

        # --- Step 6: Save ---
        print(f"Saving output to: {output_path}")
        if output_ext == ".mp3":
            save_as_mp3(output_wav, model.sr, str(output_path))
        else:
            torchaudio.save(str(output_path), output_wav, model.sr)
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
