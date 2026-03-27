# Prepend Silence to Generated MP3s

## Context

The batch TTS pipeline generates MP3 chunks that begin immediately with speech. Playback systems (media players, podcast apps, audiobook readers) benefit from a short leading silence so the first word is not clipped. The user requires ~600ms of silence at the start of every generated MP3. Additionally, a standalone utility is needed to retroactively add silence to already-generated files in bulk.

## Goals

1. All MP3s produced by `automation/batch_tts.py` have 600ms of silence prepended (configurable).
2. A new ad-hoc script can batch-prepend silence to existing MP3 files in-place.

## Approach: Option A — Extend `save_as_mp3()` + new `add_silence.py`

### Part 1 — `automation/mp3_encoder.py`

- Add `prepend_silence_ms: int = 600` parameter to `save_as_mp3()`.
- Before loudness normalization, prepend a zero tensor:

```python
silence = torch.zeros(1, int(sr * prepend_silence_ms / 1000))
wav = torch.cat([silence, wav], dim=1)
```

- Silence is inserted before normalization so the full waveform is processed consistently.
- Default is 600 so existing callers gain the silence automatically without code changes.

### Part 2 — `automation/batch_tts.py`

- Add `--prepend_silence_ms` CLI argument (type: `int`, default: `600`).
- Pass the value to `save_as_mp3()` on each chunk save call.

### Part 3 — `automation/add_silence.py` (new script)

Standalone ad-hoc script.

**CLI arguments:**

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `--directory` | Yes | — | Folder containing MP3 files |
| `--silence_ms` | No | `600` | Milliseconds of silence to prepend |
| `--recursive` | No | `False` | Also process subdirectories |

**Per-file logic:**

1. `wav, sr = torchaudio.load(path)` — load existing MP3
2. `silence = torch.zeros(channels, int(sr * silence_ms / 1000))` — build silence tensor matching channel count
3. `result = torch.cat([silence, wav], dim=1)` — prepend
4. `torchaudio.save(str(path), result, sr, format="mp3")` — overwrite in-place

**Output:** Progress line per file + final summary `✓ Processed N files, skipped M`.

## Files Modified / Created

| File | Change |
|------|--------|
| `automation/mp3_encoder.py` | Add `prepend_silence_ms` param to `save_as_mp3()` |
| `automation/batch_tts.py` | Add `--prepend_silence_ms` CLI arg, pass to `save_as_mp3()` |
| `automation/add_silence.py` | New standalone ad-hoc script |

## Verification

1. Run `batch_tts.py` with a short text file — confirm first MP3 has ~600ms of leading silence via `torchaudio.load()` and checking `wav.shape[1] / sr`.
2. Run `add_silence.py --directory <output_dir>` on existing MP3s — confirm each file is ~600ms longer than before.
3. Run with `--prepend_silence_ms 0` — confirm no silence is added.
4. Run `add_silence.py --recursive` — confirm subdirectory MP3s are also processed.
