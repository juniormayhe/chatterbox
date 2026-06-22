# Batch TTS Automation

Automated batch processing system for converting long text files into multiple MP3 audio files with voice cloning.

## Features

- ✅ Voice cloning from reference audio (MP3, WAV, FLAC, OGG)
- ✅ Intelligent text splitting at punctuation/word boundaries
- ✅ MP3 output with loudness normalization (-27 LUFS)
- ✅ Paragraph-aware leading silence (600ms at a new line, 300ms mid-line continuation)
- ✅ Progress tracking with error logging
- ✅ 19x faster than baseline (optimized turbo model + CUDA)
- ✅ `add_silence.py` utility for batch-adding silence to existing MP3s

## Quick Start

### Basic Usage

```bash
python automation/batch_tts.py \
    --reference_audio "path/to/voice.mp3" \
    --text_file "path/to/document.txt"
```

### Full Example

```bash
python automation/batch_tts.py \
    --reference_audio "C:\Audio\speaker.wav" \
    --text_file "C:\Documents\my-episode-01.txt" \
    --output_dir "downloads" \
    --max_chunk_chars 300 \
    --speed_preset fast \
    --bitrate 128 \
    --newline_silence_ms 600 \
    --sentence_silence_ms 300 \
    --seed 12345
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--reference_audio` | str | *required* | Path to reference audio for voice cloning |
| `--text_file` | str | *required* | Path to text file (.txt or .md) |
| `--output_dir` | str | `downloads` | Output directory for MP3 files |
| `--max_chunk_chars` | int | `300` | Maximum characters per audio chunk |
| `--speed_preset` | str | `fast` | Speed preset: `balanced`, `fast`, `max_speed` |
| `--target_lufs` | float | `-27.0` | Target loudness normalization (LUFS) |
| `--bitrate` | int | `128` | MP3 bitrate in kbps |
| `--newline_silence_ms` | int | `600` | Leading silence (ms) for a chunk that starts a new line/paragraph |
| `--sentence_silence_ms` | int | `300` | Leading silence (ms) for a chunk that continues the same line (split only by `max_chars`) |
| `--seed` | int | `0` | Random seed; `0` = random, fixed value = reproducible output |
| `--device` | str | `cuda` | Device: `cuda` or `cpu` |

## Output Structure

```
downloads/
└── my-episode-01/
    ├── audio01.mp3
    ├── audio02.mp3
    ├── audio03.mp3
    ├── ...
    └── output.log
```

- **Sequential naming**: `audio01.mp3`, `audio02.mp3`, etc.
- **Logs**: `output.log` contains processing details and error information

## Performance

**For 10,000 character text** (~33 chunks @ 300 chars each):
- Model loading: ~7 seconds (one-time)
- Reference audio embedding: ~2 seconds (one-time)
- Per-chunk generation: ~1.4 seconds
- **Total time: ~60 seconds** ⚡

**Comparison**:
- Old baseline: 46.7 minutes for 10,000 chars
- New optimized: 60 seconds
- **Speedup: ~47x** 🚀

## Text Splitting Algorithm

The system intelligently splits text into chunks:

1. **Target length**: ~300 characters per chunk (configurable)
2. **Punctuation boundaries**: Chunks end at `.`, `,`, `;`, `!`, `?`, `:`
3. **Word boundaries**: Long sentences split at spaces (no mid-word breaks)
4. **Natural flow**: Maintains sentence structure for better audio quality

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Chatterbox TTS turbo model
- Reference audio file (5+ seconds recommended)

## Error Handling

- **Continue on error**: Processing continues even if individual chunks fail
- **Error logging**: Detailed error messages saved to `output.log`
- **Summary statistics**: Shows successful/failed chunks at completion

## Module Overview

### `text_splitter.py`
Intelligent text chunking with punctuation and word boundary awareness.

```python
from automation.text_splitter import split_text

chunks = split_text("Long text here...", max_chars=300)
```

### `mp3_encoder.py`
WAV to MP3 conversion with loudness normalization.

```python
from automation.mp3_encoder import save_as_mp3

save_as_mp3(wav_tensor, sr=24000, output_path="audio.mp3",
            target_lufs=-27.0, bitrate=128000, prepend_silence_ms=600)
```

### `add_silence.py`
Standalone ad-hoc script to prepend silence to existing MP3 files in-place.

```bash
# Add 600ms silence to all MP3s in a folder
python automation/add_silence.py --directory downloads/my-episode

# Add 1 second silence recursively across subdirectories
python automation/add_silence.py --directory downloads --recursive --silence_ms 1000
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--directory` | str | *required* | Folder containing MP3 files to process |
| `--silence_ms` | int | `600` | Milliseconds of silence to prepend |
| `--recursive` | flag | off | Also process MP3s in subdirectories |

> **Note:** Files are re-encoded at torchaudio's default bitrate (~128 kbps). Loudness normalization is not applied.

### `batch_tts.py`
Main CLI script orchestrating the batch processing workflow.

## Troubleshooting

### MP3 Encoding Fails

If you get MP3 encoding errors, you may need to install additional audio codecs:

```bash
pip install pydub
# Then install ffmpeg for your platform
```

### Out of Memory

Reduce `max_chunk_chars` to create smaller audio chunks:

```bash
python automation/batch_tts.py ... --max_chunk_chars 200
```

### Slow Performance

- Ensure you're using CUDA (`--device cuda`)
- Use `fast` or `max_speed` preset
- Check that GPU is being utilized (run `nvidia-smi`)

## Examples

### Process Markdown File

```bash
python automation/batch_tts.py \
    --reference_audio "narrator.mp3" \
    --text_file "story.md"
```

### Maximum Speed

```bash
python automation/batch_tts.py \
    --reference_audio "voice.wav" \
    --text_file "script.txt" \
    --speed_preset max_speed \
    --max_chunk_chars 250
```

### High Quality MP3

```bash
python automation/batch_tts.py \
    --reference_audio "speaker.mp3" \
    --text_file "audiobook.txt" \
    --bitrate 192 \
    --speed_preset balanced
```

### No Leading Silence

```bash
python automation/batch_tts.py \
    --reference_audio "voice.mp3" \
    --text_file "script.txt" \
    --newline_silence_ms 0 \
    --sentence_silence_ms 0
```

### Add Silence to Existing MP3s

```bash
# Single folder
python automation/add_silence.py --directory downloads/my-episode

# All subfolders
python automation/add_silence.py --directory downloads --recursive

# Custom duration
python automation/add_silence.py --directory downloads/my-episode --silence_ms 1000
```

## License

Same as parent Chatterbox project.
