#!/bin/bash
# tts — one-shot voice cloning via automation/batch_tts.py.
#
# Usage:
#   tts "some spoken text"        # literal text
#   tts /path/to/voiceover.txt    # a .txt or .md script file
#
# Extra batch_tts.py flags can be appended and are forwarded, e.g.:
#   tts "hello" --speed_preset max_speed
#   tts script.txt --output_dir ~/Desktop --reference_audio other_voice.mp3
#
# Override the default voice without editing this file:
#   CHATTERBOX_VOICE=/path/to/voice.mp3 tts "hello"
cd "$(dirname "$0")" || exit 1

VOICE="${CHATTERBOX_VOICE:-/Users/junior/Library/CloudStorage/OneDrive-Personal/Negocios/High Magick Practices/voice.mp3}"

if [ -z "$1" ]; then
    echo 'Usage: tts "some text"   OR   tts /path/to/script.txt'
    exit 1
fi

TMPDIR_TTS=""
cleanup() { [ -n "$TMPDIR_TTS" ] && rm -rf "$TMPDIR_TTS"; }
trap cleanup EXIT

if [ -f "$1" ]; then
    TEXT_FILE="$1"
else
    # Literal text → temp file named "tts.txt"; batch_tts.py appends its own
    # -<timestamp> to the output folder, giving ~/Downloads/tts-<timestamp>/.
    TMPDIR_TTS="$(mktemp -d -t tts)"
    TEXT_FILE="$TMPDIR_TTS/tts.txt"
    printf '%s\n' "$1" > "$TEXT_FILE"
fi

.venv/bin/python automation/batch_tts.py \
    --reference_audio "$VOICE" \
    --text_file "$TEXT_FILE" \
    --device cpu \
    "${@:2}"
