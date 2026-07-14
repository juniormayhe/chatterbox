#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python gradio_tts_turbo_app.py "$@"
