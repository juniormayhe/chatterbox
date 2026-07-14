#!/bin/bash
# ttsjr — same as tts, but uses the "jr 1min.mp3" reference voice.
#
# Usage:
#   ttsjr "some spoken text"      # literal text
#   ttsjr /path/to/voiceover.txt  # a .txt or .md script file
#
# Thin wrapper: sets the reference voice and delegates to tts.sh, so all
# text/file handling and flag forwarding live in one place.
CHATTERBOX_VOICE="/Users/junior/Library/CloudStorage/OneDrive-Personal/Negocios/Cursos/jr 1min.mp3" \
    exec "$(dirname "$0")/tts.sh" "$@"
