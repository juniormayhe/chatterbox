#!/bin/bash
# Generate one Hebrew sentence with the standard Chatterbox multilingual model.
cd "$(dirname "$0")" || exit 1

# Chatterbox's MPS decoder uses an ISTFT operation that PyTorch currently
# dispatches to CPU through its documented MPS fallback.
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ "$#" -eq 0 ]; then
    echo 'Usage: tts_he "Hebrew text"'
    exit 1
fi

.venv/bin/python automation/tts_he.py "$@"
