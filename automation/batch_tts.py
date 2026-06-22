"""
Batch Text-to-Speech Processing Script

Processes long text files into multiple MP3 audio files with voice cloning.

Usage:
    python automation/batch_tts.py \
        --reference_audio "C:\\path\\to\\voice.mp3" \
        --text_file "C:\\path\\to\\my-episode-01.txt" \
        --output_dir "downloads"
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import logging
import random
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Import local modules
from text_splitter import split_text_with_breaks
from mp3_encoder import save_as_mp3

# Import Chatterbox
from chatterbox.tts_optimized import ChatterboxTurboOptimized


def set_seed(seed: int):
    """Seed all RNGs for reproducible generation (bisection / A-B testing)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(log_file: Path):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch TTS processing with voice cloning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automation/batch_tts.py \\
      --reference_audio "voice.mp3" \\
      --text_file "my-episode-01.txt"

  python automation/batch_tts.py \\
      --reference_audio "C:\\Audio\\speaker.wav" \\
      --text_file "C:\\Documents\\script.md" \\
      --output_dir "output" \\
      --max_chunk_chars 250
        """
    )

    parser.add_argument(
        '--reference_audio',
        type=str,
        required=True,
        help='Path to reference audio file for voice cloning (MP3, WAV, etc.)'
    )

    parser.add_argument(
        '--text_file',
        type=str,
        required=True,
        help='Path to text file to process (.txt or .md)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(Path.home() / "Downloads"),
        help='Output directory for generated audio files (default: C:\\Users\\<username>\\Downloads)'
    )

    parser.add_argument(
        '--max_chunk_chars',
        type=int,
        default=150,
        help='Soft packing target per chunk in characters (default: 150). Single sentences longer than this are emitted whole.'
    )

    parser.add_argument(
        '--speed_preset',
        type=str,
        choices=['balanced', 'fast', 'max_speed'],
        default='balanced',
        help='TTS speed preset (default: balanced - matches Gradio defaults)'
    )

    parser.add_argument(
        '--target_lufs',
        type=float,
        default=-14.0,
        help='Target loudness normalization in LUFS (default: -14.0)'
    )

    parser.add_argument(
        '--bitrate',
        type=int,
        default=128,
        help='MP3 bitrate in kbps (default: 128)'
    )

    parser.add_argument(
        '--newline_silence_ms',
        type=int,
        default=600,
        help='Silence (ms) prepended to a chunk that starts a new line/paragraph (default: 600)'
    )

    parser.add_argument(
        '--sentence_silence_ms',
        type=int,
        default=300,
        help='Silence (ms) prepended to a chunk that continues the same line, split only by max_chars (default: 300)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )

    # Generation settings tuned for pronunciation accuracy and phoneme stability
    # (reduces V->B, TH->D, S->Z substitutions). Min P is omitted: the Turbo
    # model ignores it.
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='Sampling temperature (default: 0.5, tuned for pronunciation accuracy)'
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p nucleus sampling (default: 0.9)'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Top-k sampling (default: 100)'
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.1,
        help='Repetition penalty (default: 1.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible output (default: 0 = random). Set a fixed value for A/B bisection.'
    )

    return parser.parse_args()


def load_model(device: str, speed_preset: str):
    """
    Load optimized TTS model.

    Args:
        device: Device to use ('cuda' or 'cpu')
        speed_preset: Speed preset name

    Returns:
        Loaded ChatterboxTurboOptimized model
    """
    print(f"Loading optimized TTS model on {device}...")

    model = ChatterboxTurboOptimized.from_pretrained(
        device=device,
        use_compile=False,  # Windows doesn't support Triton
        use_mixed_precision=False,  # Dtype mismatch issues on Windows
        speed_preset=speed_preset
    )

    print("✓ Model loaded successfully")
    model.print_optimization_info()

    return model


def read_text_file(file_path: Path) -> str:
    """
    Read text from file (.txt or .md).

    Args:
        file_path: Path to text file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension not supported
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in ['.txt', '.md']:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .md")

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    """Main batch processing workflow."""
    args = parse_args()

    # Convert paths to Path objects
    reference_audio_path = Path(args.reference_audio)
    text_file_path = Path(args.text_file)
    output_base_dir = Path(args.output_dir)

    # Validate input files
    if not reference_audio_path.exists():
        print(f"✗ Error: Reference audio file not found: {reference_audio_path}")
        sys.exit(1)

    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
    if reference_audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        print(f"✗ Error: reference_audio must be an audio file (got '{reference_audio_path.suffix}'). Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    if not text_file_path.exists():
        print(f"✗ Error: Text file not found: {text_file_path}")
        sys.exit(1)

    # Create output directory based on text filename + timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = f"{text_file_path.stem}-{timestamp}"
    output_dir = output_base_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "output.log"
    setup_logging(log_file)

    logging.info("=" * 80)
    logging.info("BATCH TTS PROCESSING")
    logging.info("=" * 80)
    logging.info(f"Reference audio: {reference_audio_path}")
    logging.info(f"Text file: {text_file_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Max chunk chars: {args.max_chunk_chars}")
    logging.info(f"Speed preset: {args.speed_preset}")
    logging.info(f"Target LUFS: {args.target_lufs}")
    logging.info(f"MP3 bitrate: {args.bitrate} kbps")
    logging.info(f"Newline silence: {args.newline_silence_ms} ms")
    logging.info(f"Sentence silence: {args.sentence_silence_ms} ms")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"Top P: {args.top_p}")
    logging.info(f"Top K: {args.top_k}")
    logging.info(f"Repetition penalty: {args.repetition_penalty}")
    logging.info(f"Seed: {args.seed if args.seed != 0 else 'random'}")
    logging.info("")

    if args.seed != 0:
        set_seed(args.seed)

    try:
        # Load TTS model
        model = load_model(args.device, args.speed_preset)

        # Prepare conditionals from reference audio (one-time operation)
        logging.info(f"\nLoading reference audio and extracting speaker embeddings...")
        model.prepare_conditionals(str(reference_audio_path))
        logging.info("✓ Speaker embeddings cached")

        # Read text file
        logging.info(f"\nReading text file...")
        text = read_text_file(text_file_path)
        logging.info(f"✓ Read {len(text)} characters")

        # Split text into chunks (paragraph-aware: each chunk is tagged with
        # whether it starts a new line, which drives its leading silence).
        logging.info(f"\nSplitting text into chunks (max {args.max_chunk_chars} chars)...")
        chunks = split_text_with_breaks(text, max_chars=args.max_chunk_chars)

        logging.info(f"✓ Created {len(chunks)} chunks")
        logging.info(f"  Average chunk size: {sum(len(c) for c, _ in chunks) / len(chunks):.0f} chars")
        logging.info(f"  Min/Max: {min(len(c) for c, _ in chunks)}/{max(len(c) for c, _ in chunks)} chars")

        # Process chunks
        logging.info(f"\n" + "=" * 80)
        logging.info("GENERATING AUDIO")
        logging.info("=" * 80)

        successful = 0
        failed = 0
        start_time = datetime.now()

        for i, (chunk, is_para_start) in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
            chunk_num = i + 1

            # Longer pause when the chunk opens a new line/paragraph, shorter
            # pause when it merely continues the same line.
            silence_ms = args.newline_silence_ms if is_para_start else args.sentence_silence_ms

            try:
                # Generate audio
                wav = model.generate(
                    chunk,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )

                # Save as MP3
                mp3_path = output_dir / f"audio{chunk_num:02d}.mp3"
                save_as_mp3(
                    wav,
                    model.sr,
                    str(mp3_path),
                    target_lufs=args.target_lufs,
                    bitrate=args.bitrate * 1000,  # Convert kbps to bps
                    prepend_silence_ms=silence_ms
                )

                successful += 1
                logging.debug(f"✓ Chunk {chunk_num}/{len(chunks)}: {mp3_path.name}")

            except Exception as e:
                failed += 1
                error_msg = f"Chunk {chunk_num} failed: {str(e)}\n{traceback.format_exc()}"
                logging.error(error_msg)

                # Write detailed error to log file
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"ERROR - Chunk {chunk_num}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Text: {chunk[:200]}...\n" if len(chunk) > 200 else f"Text: {chunk}\n")
                    f.write(f"\n{error_msg}\n")

        # Calculate statistics
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # Print summary
        logging.info("\n" + "=" * 80)
        logging.info("BATCH PROCESSING COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Total chunks: {len(chunks)}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Success rate: {successful/len(chunks)*100:.1f}%")
        logging.info(f"Total time: {elapsed:.1f} seconds")
        logging.info(f"Average time per chunk: {elapsed/len(chunks):.2f} seconds")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Log file: {log_file}")
        logging.info("=" * 80)

        if failed > 0:
            logging.warning(f"\n⚠️  {failed} chunk(s) failed. Check {log_file} for details.")
        else:
            logging.info(f"\n🎉 All chunks processed successfully!")

        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        logging.info("\n\n⚠️  Processing interrupted by user")
        sys.exit(130)

    except Exception as e:
        logging.error(f"\n✗ Fatal error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
