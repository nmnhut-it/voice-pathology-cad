"""
Signal preprocessing module for voice pathology analysis.
Implements: amplitude normalization, SNR estimation, steady-state segmentation.
Based on proposal section A requirements.
"""
import numpy as np

from config import (
    TRIM_START_SEC,
    TRIM_END_SEC,
    MIN_DURATION_SEC,
    SNR_THRESHOLD_DB,
    AMPLITUDE_NORM_PEAK,
)


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
    signal: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, float, bool]:
    """
    Full preprocessing pipeline per proposal section A.
    Returns: (processed_signal, snr_db, passes_quality_check).
    """
    snr_db = estimate_snr_db(signal, sample_rate)
    passes_quality = snr_db >= SNR_THRESHOLD_DB

    normalized = normalize_amplitude(signal)
    trimmed = trim_stable_segment(normalized, sample_rate)

    return trimmed, snr_db, passes_quality
