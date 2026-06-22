"""
Runaway / repetition loop guard for the Turbo AR decode loop.

The Turbo speech-token decoder (`T3.inference_turbo`) has no alignment-based
hallucination mitigation (the `AlignmentStreamAnalyzer` is Llama-specific and
gated behind `is_multilingual`, so it never runs for the GPT2 Turbo backbone).

When the model fails to emit EOS it can get stuck emitting a short cyclic
pattern of speech tokens, which the vocoder renders as babbled / invented
words at the tail of a clip. This module provides a cheap, architecture-
agnostic detector for that failure mode so the decode loop can break early.

Kept deliberately high-precision (low false-positive): it only fires on a
clearly cyclic tail. Arbitrary invented words that are *not* loops are caught
by the downstream ASR verification layer in the batch script.
"""

from typing import List, Optional


def detect_token_loop(
    token_ids: List[int],
    min_period: int = 1,
    max_period: int = 10,
    min_repeats: int = 5,
) -> bool:
    """
    Return True if the tail of ``token_ids`` is a cyclic pattern.

    Detects a block of length ``p`` (``min_period`` <= p <= ``max_period``) that
    repeats consecutively at least ``min_repeats`` times at the very end of the
    sequence.

    Args:
        token_ids: Generated speech token ids, in order.
        min_period: Smallest repeating-block length to consider (1 = a single
            token repeated).
        max_period: Largest repeating-block length to consider.
        min_repeats: Number of consecutive identical blocks required to call it
            a loop.

    Returns:
        True if a repeating tail pattern is found, else False.
    """
    return find_loop_period(token_ids, min_period, max_period, min_repeats) is not None


def find_loop_period(
    token_ids: List[int],
    min_period: int = 1,
    max_period: int = 10,
    min_repeats: int = 5,
) -> Optional[int]:
    """
    Like :func:`detect_token_loop` but returns the period of the smallest
    matching repeating block (useful for logging), or None if no loop.
    """
    n = len(token_ids)
    if n == 0 or min_repeats < 2:
        return None

    for period in range(min_period, max_period + 1):
        window = period * min_repeats
        if window > n:
            break  # not enough history for this (or any larger) period
        tail = token_ids[-window:]
        block = tail[:period]
        if all(tail[i] == block[i % period] for i in range(window)):
            return period

    return None
