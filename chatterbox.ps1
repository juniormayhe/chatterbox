# Run Chatterbox Gradio Turbo App
Push-Location C:\Users\junio\repos\chatterbox
try {
    & .venv\Scripts\python.exe gradio_tts_turbo_app.py $args
} finally {
    Pop-Location
}
