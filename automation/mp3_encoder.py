"""
MP3 Encoder with Loudness Normalization

Converts WAV tensors to MP3 files with consistent loudness normalization.
"""

import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from pathlib import Path
from typing import Union


def normalize_loudness(
    wav: np.ndarray,
    sr: int,
    target_lufs: float = -14.0
) -> np.ndarray:
    """
    Normalize audio loudness to target LUFS.

    Args:
        wav: Audio waveform as numpy array (1D or 2D)
        sr: Sample rate in Hz
        target_lufs: Target loudness in LUFS (default: -14.0)

    Returns:
        Loudness-normalized audio as numpy array
    """
    # Ensure wav is 1D
    if wav.ndim > 1:
        wav = wav.squeeze()

    # Create loudness meter
    meter = pyln.Meter(sr)

    # Measure current loudness
    try:
        loudness = meter.integrated_loudness(wav)
    except ValueError:
        # Audio too short or silence, return as-is
        return wav

    # Normalize to target loudness
    try:
        normalized = pyln.normalize.loudness(wav, loudness, target_lufs)
    except (ValueError, ZeroDivisionError):
        return wav

    # Peak limit at -1 dBFS to prevent clipping
    peak_limit = 10 ** (-1.0 / 20)
    normalized = np.clip(normalized, -peak_limit, peak_limit)
    return normalized


def save_as_mp3(
    wav: torch.Tensor,
    sr: int,
    output_path: Union[str, Path],
    target_lufs: float = -14.0,
    bitrate: int = 128000,  # 128 kbps in bits per second
    prepend_silence_ms: int = 0
) -> None:
    """
    Save audio tensor as MP3 with loudness normalization.

    Args:
        wav: Audio tensor (shape: [1, num_samples] or [num_samples])
        sr: Sample rate in Hz
        output_path: Path to output MP3 file
        target_lufs: Target loudness in LUFS (default: -14.0)
        bitrate: MP3 bitrate in bits per second (default: 128000 for 128 kbps)
        prepend_silence_ms: Milliseconds of silence to prepend (default: 0)

    Raises:
        RuntimeError: If MP3 encoding fails
    """
    # Ensure shape is [1, num_samples]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Convert to numpy
    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().numpy()
    else:
        wav_np = np.array(wav)

    # Ensure correct shape [num_samples] for normalization
    wav_np = wav_np.squeeze()

    import librosa
    # Use effects.split to identify non-silent intervals (top_db=38 is safe for TTS room tone)
    # frame_length=4096 ensures we don't accidentally split during short intra-word consonants
    intervals = librosa.effects.split(wav_np, top_db=38, frame_length=4096, hop_length=1024)
    
    if len(intervals) > 0:
        max_silence_samples = int(sr * 400 / 1000)  # Cap internal silences at 400ms
        
        # Start with the first non-silent interval (this naturally trims leading silence)
        processed_wav = [wav_np[intervals[0][0]:intervals[0][1]]]
        last_end = intervals[0][1]
        
        for start, end in intervals[1:]:
            silence_gap = start - last_end
            if silence_gap > max_silence_samples:
                silence_gap = max_silence_samples
                
            processed_wav.append(wav_np[last_end:last_end + silence_gap])
            processed_wav.append(wav_np[start:end])
            last_end = end
            
        wav_np = np.concatenate(processed_wav)
        # Trailing silence is naturally trimmed since we stop at the last interval's end.
    
    # Prepend explicit silence requested by batch_tts (so pacing between chunks is exactly right)
    if prepend_silence_ms > 0:
        silence_samples = int(sr * prepend_silence_ms / 1000)
        silence_np = np.zeros(silence_samples, dtype=wav_np.dtype)
        wav_np = np.concatenate([silence_np, wav_np])

    # Normalize loudness
    normalized = normalize_loudness(wav_np, sr, target_lufs)

    # Convert back to tensor with shape [1, num_samples] for torchaudio
    normalized_tensor = torch.from_numpy(normalized).unsqueeze(0).float()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as MP3
    try:
        # torchaudio.save supports MP3 with backend encoders
        # The encoder_option allows setting bitrate
        torchaudio.save(
            str(output_path),
            normalized_tensor,
            sr,
            format="mp3",
            encoding="mp3",
            encoder="mp3",
            compression=bitrate // 1000  # Convert to kbps
        )
    except Exception as e:
        # Fallback: try without encoder options (older torchaudio versions)
        try:
            torchaudio.save(
                str(output_path),
                normalized_tensor,
                sr,
                format="mp3"
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to save MP3: {e}\n"
                f"Fallback also failed: {fallback_error}\n"
                f"You may need to install pydub or ffmpeg for MP3 support."
            )


def save_as_wav(
    wav: torch.Tensor,
    sr: int,
    output_path: Union[str, Path],
    target_lufs: float = -14.0
) -> None:
    """
    Save audio tensor as WAV with loudness normalization.

    Fallback option if MP3 encoding fails.

    Args:
        wav: Audio tensor (shape: [1, num_samples] or [num_samples])
        sr: Sample rate in Hz
        output_path: Path to output WAV file
        target_lufs: Target loudness in LUFS (default: -14.0)
    """
    # Convert to numpy
    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().numpy()
    else:
        wav_np = np.array(wav)

    # Ensure correct shape
    wav_np = wav_np.squeeze()

    import librosa
    intervals = librosa.effects.split(wav_np, top_db=38, frame_length=4096, hop_length=1024)
    
    if len(intervals) > 0:
        max_silence_samples = int(sr * 400 / 1000)
        
        processed_wav = [wav_np[intervals[0][0]:intervals[0][1]]]
        last_end = intervals[0][1]
        
        for start, end in intervals[1:]:
            silence_gap = start - last_end
            if silence_gap > max_silence_samples:
                silence_gap = max_silence_samples
                
            processed_wav.append(wav_np[last_end:last_end + silence_gap])
            processed_wav.append(wav_np[start:end])
            last_end = end
            
        wav_np = np.concatenate(processed_wav)

    # Normalize loudness
    normalized = normalize_loudness(wav_np, sr, target_lufs)

    # Convert back to tensor with shape [1, num_samples]
    normalized_tensor = torch.from_numpy(normalized).unsqueeze(0).float()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as WAV
    torchaudio.save(
        str(output_path),
        normalized_tensor,
        sr
    )


if __name__ == "__main__":
    # Fix Unicode on Windows
    import sys
    if sys.platform == 'win32':
        import os
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # Test MP3 encoding
    print("Testing MP3 encoder...")
    print("=" * 60)

    # Generate test audio (1 second of sine wave at 440 Hz)
    sr = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    test_wav = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)

    print(f"Test audio shape: {test_wav.shape}")
    print(f"Sample rate: {sr} Hz")

    # Test MP3 encoding
    try:
        save_as_mp3(test_wav, sr, "test_output.mp3", bitrate=128000)
        print("✓ MP3 encoding successful!")
        print(f"  Output: test_output.mp3")

        # Check file size
        file_size = Path("test_output.mp3").stat().st_size
        print(f"  File size: {file_size} bytes ({file_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"✗ MP3 encoding failed: {e}")

    # Test WAV encoding (fallback)
    try:
        save_as_wav(test_wav, sr, "test_output.wav")
        print("✓ WAV encoding successful!")
        print(f"  Output: test_output.wav")

        file_size = Path("test_output.wav").stat().st_size
        print(f"  File size: {file_size} bytes ({file_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"✗ WAV encoding failed: {e}")

    print("=" * 60)
