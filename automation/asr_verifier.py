"""
ASR verification for batch TTS.

Transcribes generated audio with openai-whisper and compares the transcript to
the input text via word error rate (WER). The batch script uses this to detect
hallucinated / invented words and regenerate offending chunks.

Whisper is fed a numpy float32 array directly (resampled to 16 kHz), so no
ffmpeg / temp files are required.
"""

import logging
import re
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_SR = 16000

# Strip everything except word characters and spaces for a forgiving comparison.
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace for fair comparison."""
    if not text:
        return ""
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def _word_levenshtein(ref: List[str], hyp: List[str]) -> int:
    """Word-level edit distance (insert/delete/substitute = cost 1)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    """
    Word error rate between two already-normalized strings.

    Returns 0.0 for an empty reference matched by an empty hypothesis, and 1.0
    for an empty reference with any hypothesis content (any output is "wrong").
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _word_levenshtein(ref_words, hyp_words) / len(ref_words)


class AsrVerifier:
    """Lazy-loaded whisper transcriber + scorer."""

    def __init__(self, model_name: str = "small", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None  # loaded on first use

    def _ensure_model(self):
        if self._model is None:
            import whisper  # imported lazily so the dep is only needed when used
            logger.info(f"Loading whisper ASR model '{self.model_name}' on {self.device}...")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info("✓ Whisper model loaded")
        return self._model

    def transcribe(self, wav, sr: int, language: Optional[str] = "en") -> str:
        """
        Transcribe an audio tensor/array.

        Args:
            wav: torch.Tensor or np.ndarray, shape (1, N) or (N,).
            sr: sample rate of ``wav``.
            language: forced language for whisper (None = auto-detect).
        """
        model = self._ensure_model()

        # To 1-D float32 numpy.
        if hasattr(wav, "detach"):  # torch tensor
            audio = wav.detach().cpu().numpy()
        else:
            audio = np.asarray(wav)
        audio = audio.squeeze().astype(np.float32)

        # Whisper expects mono 16 kHz.
        if sr != WHISPER_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)

        fp16 = self.device != "cpu"
        result = model.transcribe(audio, language=language, fp16=fp16)
        return result.get("text", "").strip()

    def score(self, reference_text: str, wav, sr: int) -> tuple:
        """
        Transcribe ``wav`` and score it against ``reference_text``.

        Returns:
            (wer, normalized_hypothesis) tuple.
        """
        hyp = self.transcribe(wav, sr)
        return wer(normalize(reference_text), normalize(hyp)), normalize(hyp)
