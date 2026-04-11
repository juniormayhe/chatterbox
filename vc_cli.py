"""
Voice Conversion CLI

Converts the voice in a source audio file to match a target reference voice,
preserving the original speech content and timing, using ChatterboxVC.

The model transfers voice timbre but preserves the source audio's pitch.
Use --auto-pitch to automatically shift the output pitch to match the target,
or --pitch-shift to apply a manual semitone offset.

Usage:
    python vc_cli.py --input input.mp3 --target reference.mp3 --output result.mp3
    python vc_cli.py -i female.mp3 -t male.mp3 -o result.mp3 --auto-pitch
    python vc_cli.py -i input.wav -t voice.wav -o converted.wav --pitch-shift -8
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import librosa
import torch
import torchaudio

from chatterbox.vc import ChatterboxVC
from automation.mp3_encoder import save_as_mp3


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def estimate_median_f0(audio_path: str, sr: int = 24000) -> float | None:
    """Return the median voiced F0 (Hz) of an audio file, or None if undetectable."""
    y, _ = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz  — covers bass male
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — covers soprano female
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return None
    return float(np.median(voiced_f0))


def compute_pitch_shift(source_f0: float, target_f0: float) -> float:
    """Return the semitone shift needed to move source F0 to target F0."""
    return 12.0 * np.log2(target_f0 / source_f0)


def apply_pitch_shift(wav: torch.Tensor, sr: int, n_steps: float) -> torch.Tensor:
    """Shift pitch of a [1, samples] tensor by n_steps semitones.

    n_fft=512 (vs librosa default 2048) reduces phase-vocoder time smearing,
    which otherwise causes a reverb/echo artifact on large shifts.
    """
    wav_np = wav.squeeze(0).numpy()
    shifted = librosa.effects.pitch_shift(wav_np, sr=sr, n_steps=n_steps, n_fft=512)
    return torch.from_numpy(shifted).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert voice in source audio to match a target reference voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vc_cli.py --input input.mp3 --target reference.mp3 --output result.mp3
  python vc_cli.py -i female.mp3 -t male.mp3 -o result.mp3 --auto-pitch
  python vc_cli.py -i input.wav -t voice.wav -o converted.wav --pitch-shift -8 --device cpu
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to source audio file (voice to convert)"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to reference audio file (target voice)"
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
    pitch_group = parser.add_mutually_exclusive_group()
    pitch_group.add_argument(
        "--auto-pitch",
        action="store_true",
        help="Auto-detect pitch of target voice and shift output to match it"
    )
    pitch_group.add_argument(
        "--pitch-shift",
        type=float,
        metavar="SEMITONES",
        help="Manually shift output pitch by N semitones (e.g. -8 for male target)"
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

        input_path = Path(args.input)
        target_path = Path(args.target)
        output_path = Path(args.output)

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

        # Pre-compute target F0 before loading the model (fast); output F0 measured after conversion
        target_f0 = None
        pitch_steps = None
        if args.auto_pitch:
            print("Analyzing target pitch...")
            target_f0 = estimate_median_f0(str(target_path))
            if target_f0 is None:
                print("  ✗ Could not detect pitch in target file — skipping pitch shift")
            else:
                print(f"  Target median F0: {target_f0:.1f} Hz")
        elif args.pitch_shift is not None:
            pitch_steps = args.pitch_shift
            print(f"Pitch shift: {pitch_steps:+.2f} semitones (manual)")

        print(f"Loading ChatterboxVC on {device}...")
        model = ChatterboxVC.from_pretrained(device)
        print("  Model loaded")

        print("Converting voice...")
        print(f"  Input:  {input_path}")
        print(f"  Target: {target_path}")
        wav = model.generate(audio=str(input_path), target_voice_path=str(target_path))
        print("  Conversion complete")

        if args.auto_pitch and target_f0 is not None:
            print("Analyzing output pitch...")
            import tempfile, uuid
            tmp_wav = Path(tempfile.gettempdir()) / f"vc_tmp_{uuid.uuid4().hex}.wav"
            torchaudio.save(str(tmp_wav), wav, model.sr)
            out_f0 = estimate_median_f0(str(tmp_wav))
            tmp_wav.unlink(missing_ok=True)
            if out_f0 is None:
                print("  ✗ Could not detect pitch in output — skipping pitch shift")
            else:
                pitch_steps = compute_pitch_shift(out_f0, target_f0)
                print(f"  Output median F0: {out_f0:.1f} Hz")
                print(f"  Target median F0: {target_f0:.1f} Hz")
                print(f"  Correction shift: {pitch_steps:+.2f} semitones")

        if pitch_steps is not None and abs(pitch_steps) > 0.1:
            print(f"Applying pitch shift ({pitch_steps:+.2f} semitones)...")
            wav = apply_pitch_shift(wav, model.sr, pitch_steps)
            print("  Done")

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
