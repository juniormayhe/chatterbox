"""
Fast Batch TTS — maximum speed, minimal processing.

Outputs WAV files to a queue folder with no loudness normalization,
no silence padding, and max_speed inference preset.

Usage:
    python automation/fast_tts.py \
        --reference_audio "C:\\path\\to\\voice.mp3" \
        --text_file "C:\\path\\to\\script.txt"
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import random
import traceback
from datetime import datetime

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from text_splitter import split_text, split_markdown
from chatterbox.tts_optimized import ChatterboxTurboOptimized

DEFAULT_OUTPUT_DIR = r"C:\temp\audio-queue"


def set_seed(seed: int):
    """Seed all RNGs for reproducible generation (bisection / A-B testing)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fast batch TTS — max speed, WAV output to queue folder'
    )
    parser.add_argument('--reference_audio', type=str, required=True,
                        help='Reference audio file for voice cloning')
    parser.add_argument('--text_file', type=str, required=True,
                        help='Input text file (.txt or .md)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--max_chunk_chars', type=int, default=500,
                        help='Soft packing target per chunk in characters (default: 500). Single sentences longer than this are emitted whole.')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference (default: cuda)')
    # Generation settings tuned for pronunciation accuracy and phoneme stability
    # (reduces V->B, TH->D, S->Z substitutions). Min P is omitted: Turbo ignores it.
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature (default: 0.5, tuned for pronunciation accuracy)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p nucleus sampling (default: 0.9)')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top-k sampling (default: 100)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                        help='Repetition penalty (default: 1.1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducible output (default: 0 = random). Set a fixed value for A/B bisection.')
    return parser.parse_args()


def main():
    args = parse_args()

    reference_audio_path = Path(args.reference_audio)
    text_file_path = Path(args.text_file)
    output_dir = Path(args.output_dir)

    if not reference_audio_path.exists():
        print(f"✗ Reference audio not found: {reference_audio_path}")
        sys.exit(1)
    if not text_file_path.exists():
        print(f"✗ Text file not found: {text_file_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed != 0:
        set_seed(args.seed)

    print(f"Loading model on {args.device} (max_speed preset)...")
    model = ChatterboxTurboOptimized.from_pretrained(
        device=args.device,
        use_compile=False,
        use_mixed_precision=False,
        speed_preset='max_speed'
    )
    print("✓ Model loaded")

    print(f"Loading reference audio...")
    model.prepare_conditionals(str(reference_audio_path))
    print("✓ Speaker embeddings cached")

    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if text_file_path.suffix.lower() == '.md':
        chunks = split_markdown(text, max_chars=args.max_chunk_chars)
    else:
        chunks = split_text(text, max_chars=args.max_chunk_chars)

    print(f"✓ {len(chunks)} chunks → {output_dir}")

    successful = 0
    failed = 0
    start_time = datetime.now()

    for i, chunk in enumerate(tqdm(chunks, desc="Generating", unit="chunk")):
        chunk_num = i + 1
        try:
            wav = model.generate(
                chunk,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            out_path = output_dir / f"audio{chunk_num:02d}.wav"
            torchaudio.save(str(out_path), wav.cpu(), model.sr)
            successful += 1

        except Exception as e:
            failed += 1
            print(f"\n✗ Chunk {chunk_num} failed: {e}")
            traceback.print_exc()

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"Done: {successful}/{len(chunks)} chunks in {elapsed:.1f}s "
          f"({elapsed/len(chunks):.2f}s avg)")
    if failed:
        print(f"⚠️  {failed} failed")
    print(f"Output: {output_dir}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
