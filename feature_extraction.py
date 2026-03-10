"""
Acoustic feature extraction using Parselmouth (Praat) and Librosa.
Extracts: F0, formants+bandwidth, jitter, shimmer, HNR, CPP, MFCCs, spectral.
Covers proposal sections B (multidimensional features) and 7.3 (tools).
"""
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call

from config import (
    PITCH_FLOOR_HZ,
    PITCH_CEILING_HZ,
    MAX_NUM_FORMANTS,
    MAX_FORMANT_FREQ_HZ,
    NUM_MFCC_COEFFICIENTS,
    N_FFT,
    HOP_LENGTH,
    ALL_FEATURE_NAMES,
    ALL_OUTPUT_COLUMNS,
)
from preprocessing import preprocess_signal

# ~25ms analysis window, rounded up to next power of 2
SPECTRAL_WINDOW_SEC = 0.025


def _compute_fft_params(sample_rate: int) -> tuple[int, int]:
    """Compute appropriate N_FFT and hop_length for given sample rate."""
    window_samples = int(SPECTRAL_WINDOW_SEC * sample_rate)
    # Round up to next power of 2
    n_fft = 1
    while n_fft < window_samples:
        n_fft *= 2
    hop_length = n_fft // 2
    return n_fft, hop_length


def _extract_pitch_features(sound: parselmouth.Sound) -> dict:
    """Extract F0 mean, std, median from pitch contour."""
    pitch = call(sound, "To Pitch", 0.0, PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    if len(voiced) == 0:
        return {"f0_mean": np.nan, "f0_std": np.nan, "f0_median": np.nan}

    return {
        "f0_mean": np.mean(voiced),
        "f0_std": np.std(voiced),
        "f0_median": np.median(voiced),
    }


def _extract_perturbation_features(sound: parselmouth.Sound) -> dict:
    """Extract jitter and shimmer variants from point process."""
    point_process = call(
        sound, "To PointProcess (periodic, cc)",
        PITCH_FLOOR_HZ, PITCH_CEILING_HZ,
    )
    duration = sound.get_total_duration()

    return {
        "jitter_local": call(
            point_process, "Get jitter (local)",
            0.0, duration, 0.0001, 0.02, 1.3,
        ),
        "jitter_rap": call(
            point_process, "Get jitter (rap)",
            0.0, duration, 0.0001, 0.02, 1.3,
        ),
        "shimmer_local": call(
            [sound, point_process], "Get shimmer (local)",
            0.0, duration, 0.0001, 0.02, 1.3, 1.6,
        ),
        "shimmer_apq3": call(
            [sound, point_process], "Get shimmer (apq3)",
            0.0, duration, 0.0001, 0.02, 1.3, 1.6,
        ),
    }


def _extract_noise_features(sound: parselmouth.Sound) -> dict:
    """Extract HNR and CPP — key noise/periodicity indicators."""
    harmonicity = call(
        sound, "To Harmonicity (cc)", 0.01, PITCH_FLOOR_HZ, 0.1, 1.0
    )
    hnr_values = [
        harmonicity.get_value(t)
        for t in harmonicity.xs()
        if harmonicity.get_value(t) != -200
    ]
    hnr_mean = np.mean(hnr_values) if hnr_values else np.nan

    power_cepstrogram = call(
        sound, "To PowerCepstrogram", PITCH_FLOOR_HZ, 0.002, 5000.0, 50
    )
    cpps = call(
        power_cepstrogram, "Get CPPS",
        False, 0.01, 0.001, 60.0, 330.0, 0.05,
        "Parabolic", 0.001, 0.0, "Exponential decay", "Robust slow",
    )

    return {"hnr_mean": hnr_mean, "cpp_mean": cpps}


def _get_max_formant_freq(sample_rate: int) -> int:
    """
    Return appropriate max formant frequency for Burg analysis.
    Must stay well below Nyquist (sr/2). Standard values:
    - 8kHz (VOICED): 4000 Hz
    - 16kHz: 5000 Hz
    - 50kHz (SVD): 5000 Hz (formants are below 5kHz regardless of sr)
    """
    nyquist = sample_rate // 2
    return min(5000, nyquist - 500)


def _extract_formant_features(sound: parselmouth.Sound) -> dict:
    """
    Extract F1-F3 mean values and bandwidths (averaged over stable segment).
    Bandwidth reflects tissue damping — wider BW suggests pathology.
    """
    sr = int(sound.sampling_frequency)
    max_formant = _get_max_formant_freq(sr)
    formant = call(
        sound, "To Formant (burg)",
        0.0, MAX_NUM_FORMANTS, max_formant, 0.025, 50.0,
    )
    duration = sound.get_total_duration()
    # Average over middle 60% of the segment for robustness
    t_start = duration * 0.2
    t_end = duration * 0.8
    num_samples = 10
    time_points = np.linspace(t_start, t_end, num_samples)

    features = {}
    for formant_num in range(1, 4):
        freq_key = f"f{formant_num}_mean"
        bw_key = f"f{formant_num}_bw"

        freqs = []
        bws = []
        for t in time_points:
            f = call(formant, "Get value at time", formant_num, t, "Hertz", "Linear")
            b = call(formant, "Get bandwidth at time", formant_num, t, "Hertz", "Linear")
            if not np.isnan(f):
                freqs.append(f)
            if not np.isnan(b):
                bws.append(b)

        features[freq_key] = np.mean(freqs) if freqs else np.nan
        features[bw_key] = np.mean(bws) if bws else np.nan

    return features


def _extract_mfcc_features(signal: np.ndarray, sample_rate: int) -> dict:
    """
    Extract MFCC coefficients 1-13 (mean over time).
    MFCCs capture the spectral envelope shape of the vocal tract.
    """
    n_fft, hop_length = _compute_fft_params(sample_rate)
    signal_float = signal.astype(np.float32)
    mfccs = librosa.feature.mfcc(
        y=signal_float,
        sr=sample_rate,
        n_mfcc=NUM_MFCC_COEFFICIENTS,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return {
        f"mfcc_{i+1}": np.mean(mfccs[i])
        for i in range(NUM_MFCC_COEFFICIENTS)
    }


def _extract_spectral_features(signal: np.ndarray, sample_rate: int) -> dict:
    """
    Extract spectral shape descriptors via Librosa.
    Centroid, bandwidth, rolloff, flatness characterize spectral distribution.
    """
    n_fft, hop_length = _compute_fft_params(sample_rate)
    signal_float = signal.astype(np.float32)
    centroid = librosa.feature.spectral_centroid(
        y=signal_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    flatness = librosa.feature.spectral_flatness(
        y=signal_float, n_fft=n_fft, hop_length=hop_length
    )

    return {
        "spectral_centroid": np.mean(centroid),
        "spectral_bandwidth": np.mean(bandwidth),
        "spectral_rolloff": np.mean(rolloff),
        "spectral_flatness": np.mean(flatness),
    }


def extract_features_from_signal(
    signal: np.ndarray, sample_rate: int
) -> dict | None:
    """
    Full feature extraction pipeline for one audio signal.
    Applies preprocessing, then extracts all feature groups.
    Returns dict with all features + snr_db, or None on failure.
    """
    try:
        processed, snr_db, _, output_sr = preprocess_signal(signal, sample_rate)
        sound = parselmouth.Sound(processed, sampling_frequency=output_sr)

        features = {}
        features.update(_extract_pitch_features(sound))
        features.update(_extract_perturbation_features(sound))
        features.update(_extract_noise_features(sound))
        features.update(_extract_formant_features(sound))
        features.update(_extract_mfcc_features(processed, output_sr))
        features.update(_extract_spectral_features(processed, output_sr))
        features["snr_db"] = snr_db
        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None


def extract_features_dataframe(
    df: pd.DataFrame,
    checkpoint_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features for all records in the dataset DataFrame.
    Supports incremental checkpointing to avoid losing progress.
    Returns (feature_matrix, labels) with progress logging.
    """
    from pathlib import Path

    feature_rows = []
    valid_indices = []
    start_from = 0
    total = len(df)

    # Resume from checkpoint if available
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = pd.read_csv(checkpoint_path)
        start_from = len(ckpt)
        feature_rows = ckpt[ALL_OUTPUT_COLUMNS].to_dict("records")
        valid_indices = list(range(start_from))
        print(f"  Resuming from checkpoint: {start_from}/{total} records")

    for count, (idx, row) in enumerate(df.iterrows(), 1):
        if count <= start_from:
            continue
        features = extract_features_from_signal(row["signal"], row["sample_rate"])
        if features is not None:
            feature_rows.append(features)
            valid_indices.append(idx)
        if count % 20 == 0 or count == total:
            print(f"  Processed {count}/{total} records...")
        # Save checkpoint every 100 records
        if checkpoint_path and count % 100 == 0:
            _save_checkpoint(feature_rows, df, valid_indices, checkpoint_path)

    # Final save
    if checkpoint_path:
        _save_checkpoint(feature_rows, df, valid_indices, checkpoint_path)

    all_df = pd.DataFrame(feature_rows, columns=ALL_OUTPUT_COLUMNS)
    feature_df = all_df[ALL_FEATURE_NAMES]
    labels = df.loc[valid_indices, "label"].reset_index(drop=True)

    print(f"Extracted features for {len(feature_df)}/{len(df)} records "
          f"({len(ALL_FEATURE_NAMES)} model features)")
    return feature_df, labels


def _save_checkpoint(feature_rows, df, valid_indices, path):
    """Save intermediate feature extraction progress to CSV."""
    ckpt_df = pd.DataFrame(feature_rows, columns=ALL_OUTPUT_COLUMNS)
    ckpt_df["label"] = df.loc[valid_indices, "label"].values
    ckpt_df.to_csv(path, index=False)
    print(f"    [checkpoint saved: {len(ckpt_df)} records]")
