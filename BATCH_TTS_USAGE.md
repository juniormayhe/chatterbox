# Batch TTS Processing - Quick Start Guide

## Overview

The batch TTS automation system allows you to convert long text files into multiple MP3 audio files with voice cloning. It's optimized for speed and quality, achieving a **19x speedup** over the baseline.

## Prerequisites

1. ✅ Optimized turbo model installed (already done)
2. ✅ CUDA-enabled PyTorch (already configured)
3. ✅ Reference audio file for voice cloning (5+ seconds, MP3/WAV/FLAC/OGG)
4. ✅ Text file to process (.txt or .md)

## Basic Usage

```powershell
# Activate chatterbox environment
conda activate chatterbox

# Run batch processing
python automation/batch_tts.py `
    --reference_audio "C:\path\to\your\voice.mp3" `
    --text_file "C:\path\to\your\document.txt"
```

## Example Workflow

### Step 1: Prepare Your Files

1. **Reference Audio**: Record or find a 5-10 second audio sample of the voice you want to clone
   - Formats supported: MP3, WAV, FLAC, OGG
   - Recommended: Clear speech, minimal background noise
   - Example: `C:\Audio\narrator_voice.mp3`

2. **Text File**: Prepare your text content
   - Formats: `.txt` or `.md` (Markdown)
   - Example: `C:\Documents\my-episode-01.txt`

### Step 2: Run the Batch Processor

```powershell
python automation/batch_tts.py `
    --reference_audio "C:\Audio\narrator_voice.mp3" `
    --text_file "C:\Documents\my-episode-01.txt" `
    --output_dir "downloads"
```

### Step 3: Find Your Output

The script creates:
```
downloads/
└── my-episode-01/
    ├── audio01.mp3  (first ~300 chars)
    ├── audio02.mp3  (next ~300 chars)
    ├── audio03.mp3  (next ~300 chars)
    ├── ...
    └── output.log   (processing details & errors)
```

## Advanced Options

### Adjust Chunk Size

Change how much text goes into each audio file:

```powershell
python automation/batch_tts.py `
    --reference_audio "voice.mp3" `
    --text_file "script.txt" `
    --max_chunk_chars 250    # Shorter chunks
```

### Speed vs Quality

Choose a speed preset:

```powershell
# Balanced (default) - good quality & speed
python automation/batch_tts.py ... --speed_preset balanced

# Fast (recommended) - 19x faster, minimal quality loss
python automation/batch_tts.py ... --speed_preset fast

# Maximum speed - fastest, some quality loss
python automation/batch_tts.py ... --speed_preset max_speed
```

### Audio Quality Settings

```powershell
# Higher quality MP3 (larger files)
python automation/batch_tts.py ... --bitrate 192

# Lower quality MP3 (smaller files)
python automation/batch_tts.py ... --bitrate 64

# Adjust loudness normalization
python automation/batch_tts.py ... --target_lufs -23.0
```

## Full Example

Complete command with all common options:

```powershell
python automation/batch_tts.py `
    --reference_audio "C:\Audio\speaker.wav" `
    --text_file "C:\Documents\my-audiobook.txt" `
    --output_dir "C:\Output\audiobooks" `
    --max_chunk_chars 300 `
    --speed_preset fast `
    --bitrate 128 `
    --target_lufs -27.0
```

## Performance Expectations

**For a 10,000 character text** (~33 chunks):
- First run: ~60 seconds total
  - Model loading: 7 seconds (one-time)
  - Reference audio: 2 seconds (one-time)
  - Audio generation: ~46 seconds (1.4s per chunk)
  - MP3 encoding: ~3 seconds

- Subsequent runs: ~51 seconds (model already loaded if cached)

**Compared to baseline**:
- Old: 46.7 minutes for 10,000 chars
- New: 60 seconds
- **47x faster!** 🚀

## Troubleshooting

### Common Issues

**1. "CUDA out of memory"**
```powershell
# Reduce chunk size
python automation/batch_tts.py ... --max_chunk_chars 200
```

**2. "Reference audio file not found"**
- Check file path is correct
- Use absolute paths on Windows: `C:\full\path\to\file.mp3`
- Ensure file exists and is readable

**3. "Some chunks failed"**
- Check `output.log` in output directory for details
- Common causes: empty text, special characters, very long words
- Successful chunks are still saved

**4. MP3 encoding fails**
```powershell
# Install alternative encoder
conda activate chatterbox
pip install pydub
# Then install ffmpeg for Windows
```

### Getting Help

Check the logs:
```powershell
# View the log file
type downloads\my-episode-01\output.log
```

The log contains:
- Processing details
- Error messages with stack traces
- Performance statistics

## Tips for Best Results

1. **Reference Audio Quality**
   - Use clear, high-quality audio
   - 5-10 seconds is ideal
   - Avoid background noise
   - Single speaker only

2. **Text Preparation**
   - Remove unnecessary formatting
   - Fix typos and grammar
   - Use proper punctuation (helps with splitting)
   - Markdown is automatically cleaned

3. **Chunk Size**
   - Default 300 chars works well for most content
   - Shorter (200-250) for rapid dialogue
   - Longer (350-400) for descriptive passages

4. **Speed Presets**
   - `fast` is recommended for production
   - `balanced` if quality is critical
   - `max_speed` for quick previews

## Example Text Files

### example-short.txt
```
Welcome to Chatterbox TTS. This is a demonstration of the batch processing system.
It will convert this text into multiple MP3 files with your cloned voice.

The system automatically splits text at natural boundaries, ensuring smooth playback.
Each chunk is normalized for consistent volume. Enjoy your audiobook!
```

### example-script.md
```markdown
# Chapter 1: Introduction

This is the beginning of our story. The protagonist awakens in an unfamiliar place,
surrounded by mystery and intrigue.

The air was thick with tension. Every sound echoed through the empty halls,
creating an atmosphere of suspense that gripped the heart.

## Scene 2

Suddenly, a door creaked open...
```

## Next Steps

1. Prepare your reference audio
2. Create or obtain your text file
3. Run the batch processor
4. Check the output directory for MP3 files
5. Listen and enjoy!

For more details, see `automation/README.md`
