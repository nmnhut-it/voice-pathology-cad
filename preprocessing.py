"""
Signal preprocessing module for voice pathology analysis.
Implements: sample rate standardization, amplitude normalization,
noise reduction, SNR estimation, steady-state segmentation.
Based on proposal section A requirements.
"""
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
from math import gcd

from config import (
    TRIM_START_SEC,
    TRIM_END_SEC,
    MIN_DURATION_SEC,
    SNR_THRESHOLD_DB,
    AMPLITUDE_NORM_PEAK,
)

# Target sample rate for standardization (16kHz is standard for speech)
TARGET_SAMPLE_RATE = 16000

# Target duration in seconds (pad or truncate to this length)
TARGET_DURATION_SEC = 3.0

# Bandpass filter range for voice (remove sub-voice and ultra-high noise)
BANDPASS_LOW_HZ = 50
BANDPASS_HIGH_HZ = 8000


def resample_signal(
    signal: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    Resample signal to target sample rate using polyphase filtering.
    Preserves signal quality better than simple interpolation.
    """
    if orig_sr == target_sr:
        return signal
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return resample_poly(signal, up, down)


def standardize_duration(
    signal: np.ndarray, sample_rate: int,
    target_sec: float = TARGET_DURATION_SEC,
) -> np.ndarray:
    """
    Standardize signal duration: truncate if longer, zero-pad if shorter.
    Centers the signal when padding.
    """
    target_len = int(target_sec * sample_rate)
    current_len = len(signal)

    if current_len >= target_len:
        # Truncate from center
        start = (current_len - target_len) // 2
        return signal[start : start + target_len]

    # Zero-pad symmetrically
    pad_total = target_len - current_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(signal, (pad_left, pad_right), mode="constant")


def denoise_bandpass(
    signal: np.ndarray, sample_rate: int,
    low_hz: int = BANDPASS_LOW_HZ,
    high_hz: int = BANDPASS_HIGH_HZ,
) -> np.ndarray:
    """
    Apply bandpass Butterworth filter to remove noise outside voice range.
    Keeps frequencies between low_hz and high_hz.
    """
    nyquist = sample_rate / 2
    high_hz = min(high_hz, int(nyquist * 0.95))
    if low_hz >= high_hz:
        return signal
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal)


def normalize_amplitude(signal: np.ndarray) -> np.ndarray:
    """Peak-normalize signal to target amplitude (avoids clipping artifacts)."""
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal * (AMPLITUDE_NORM_PEAK / peak)


def estimate_snr_db(signal: np.ndarray, sample_rate: int) -> float:
    """
    Estimate SNR by comparing RMS of signal vs silent edges.
    Uses first/last 50ms as noise estimate (before/after phonation trim).
    Returns SNR in dB.
    """
    noise_samples = int(0.05 * sample_rate)  # 50ms
    if len(signal) < noise_samples * 4:
        return 0.0

    noise_segment = np.concatenate([
        signal[:noise_samples],
        signal[-noise_samples:],
    ])
    noise_rms = np.sqrt(np.mean(noise_segment ** 2))

    # Signal RMS from the middle portion
    mid_start = len(signal) // 4
    mid_end = 3 * len(signal) // 4
    signal_rms = np.sqrt(np.mean(signal[mid_start:mid_end] ** 2))

    if noise_rms == 0:
        return 60.0  # effectively noiseless
    return 20 * np.log10(signal_rms / noise_rms)


def trim_stable_segment(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove onset/offset transients, keep steady-state portion."""
    start_sample = int(TRIM_START_SEC * sample_rate)
    end_sample = len(signal) - int(TRIM_END_SEC * sample_rate)
    min_samples = int(MIN_DURATION_SEC * sample_rate)

    if end_sample - start_sample < min_samples:
        return signal
    return signal[start_sample:end_sample]


def preprocess_signal(
    signal: np.ndarray, sample_rate: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, float, bool, int]:
    """
    Full preprocessing pipeline per proposal section A:
    1. Estimate SNR on raw signal
    2. Resample to target sample rate
    3. Bandpass filter (denoise)
    4. Amplitude normalization
    5. Trim onset/offset to steady-state

    Returns: (processed_signal, snr_db, passes_quality_check, output_sr).
    """
    snr_db = estimate_snr_db(signal, sample_rate)
    passes_quality = snr_db >= SNR_THRESHOLD_DB

    resampled = resample_signal(signal, sample_rate, target_sr)
    denoised = denoise_bandpass(resampled, target_sr)
    normalized = normalize_amplitude(denoised)
    trimmed = trim_stable_segment(normalized, target_sr)

    return trimmed, snr_db, passes_quality, target_sr
