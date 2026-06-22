"""
Unit tests for the hallucination-mitigation primitives.

Run: pytest automation/test_hallucination.py
"""

import sys
from pathlib import Path

# Make both the chatterbox package (src/) and the automation modules importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from chatterbox.models.t3.inference.repetition_guard import (
    detect_token_loop,
    find_loop_period,
)
from asr_verifier import normalize, wer


# ---- Layer 1: loop detection ----

def test_detects_multi_token_loop():
    seq = [1, 2, 3] * 6  # period 3, repeated 6x
    assert detect_token_loop(seq) is True
    assert find_loop_period(seq) == 3


def test_detects_single_token_loop():
    assert detect_token_loop([7] * 5) is True
    assert find_loop_period([7] * 5) == 1


def test_clean_sequence_is_not_a_loop():
    assert detect_token_loop(list(range(50))) is False
    assert find_loop_period(list(range(50))) is None


def test_short_repeat_below_threshold_is_not_a_loop():
    # 4 identical tail tokens, min_repeats default 5 -> not a loop
    seq = [1, 2, 3, 4, 5, 9, 9, 9, 9]
    assert detect_token_loop(seq) is False


def test_loop_only_at_tail_not_mid_sequence():
    # repetition early, but clean tail -> not flagged
    seq = [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]
    assert detect_token_loop(seq) is False


def test_empty_sequence():
    assert detect_token_loop([]) is False


# ---- Layer 2: ASR scoring ----

def test_normalize_strips_punctuation_and_case():
    assert normalize("Hello, World!") == "hello world"
    assert normalize("  multiple   spaces  ") == "multiple spaces"


def test_wer_identical_is_zero():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0


def test_wer_single_substitution():
    # 1 wrong word out of 4
    assert wer("the quick brown fox", "the quick green fox") == 0.25


def test_wer_inserted_invented_word():
    # hypothesis invents an extra word -> 1 insertion over 4 ref words
    assert wer("the quick brown fox", "the quick brown lazy fox") == 0.25


def test_wer_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "anything here") == 1.0
