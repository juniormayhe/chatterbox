"""
Add Silence to MP3 Files

Prepends a configurable amount of silence (default 600ms) to the start of
MP3 files in-place. Useful for batch-processing already-generated audio files.

Usage:
    python automation/add_silence.py --directory downloads/my-episode
    python automation/add_silence.py --directory downloads --recursive
    python automation/add_silence.py --directory downloads/ep --silence_ms 1000
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import torchaudio


def prepend_silence(path: Path, silence_ms: int) -> None:
    """Load an MP3, prepend silence, overwrite in-place atomically."""
    wav, sr = torchaudio.load(str(path))
    silence_samples = int(sr * silence_ms / 1000)
    silence = torch.zeros(wav.shape[0], silence_samples, dtype=wav.dtype)
    result = torch.cat([silence, wav], dim=1)
    tmp = path.with_suffix(".tmp.mp3")
    try:
        # Note: output is re-encoded at torchaudio's default bitrate (typically 128 kbps)
        torchaudio.save(str(tmp), result, sr, format="mp3")
        os.replace(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def find_mp3s(directory: Path, recursive: bool):
    """Return a list of all .mp3 files in directory."""
    pattern = "**/*.mp3" if recursive else "*.mp3"
    return list(directory.glob(pattern))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepend silence to MP3 files in-place",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automation/add_silence.py --directory downloads/my-episode
  python automation/add_silence.py --directory downloads --recursive --silence_ms 1000
        """
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing MP3 files to process"
    )
    parser.add_argument(
        "--silence_ms",
        type=int,
        default=600,
        help="Milliseconds of silence to prepend (default: 600)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process MP3 files in subdirectories too"
    )
    return parser.parse_args()


def main():
    # Fix Unicode on Windows
    if sys.platform == "win32":
        os.system("chcp 65001 > nul")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.silence_ms < 0:
        print("✗ --silence_ms must be >= 0")
        sys.exit(1)
    directory = Path(args.directory)

    if not directory.exists():
        print(f"✗ Directory not found: {directory}")
        sys.exit(1)

    mp3_files = find_mp3s(directory, args.recursive)

    if not mp3_files:
        print(f"No MP3 files found in: {directory}")
        sys.exit(0)

    print(f"Found {len(mp3_files)} MP3 file(s) — prepending {args.silence_ms}ms silence...")

    success = 0
    failed = 0
    for path in mp3_files:
        try:
            prepend_silence(path, args.silence_ms)
            print(f"  ✓ {path.name}")
            success += 1
        except Exception as e:
            print(f"  ✗ {path.name}: {e}")
            failed += 1

    print(f"\n✓ Processed {success} file(s)", end="")
    if failed:
        print(f", {failed} failed", end="")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
