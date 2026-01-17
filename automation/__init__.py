"""
Automation package for batch TTS processing.
"""

from .text_splitter import split_text, split_markdown
from .mp3_encoder import save_as_mp3, save_as_wav, normalize_loudness

__all__ = [
    'split_text',
    'split_markdown',
    'save_as_mp3',
    'save_as_wav',
    'normalize_loudness',
]
