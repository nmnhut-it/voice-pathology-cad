"""
Configuration constants for the Voice Pathology CAD system.
Centralizes all paths, feature names, model parameters, and label mappings.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VOICED_CACHE_DIR = DATA_DIR / "voiced_cache"
SVD_DIR = DATA_DIR / "svd"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# --- PhysioNet VOICED Database ---
VOICED_PN_DIR = "voiced/1.0.0"
VOICED_TOTAL_RECORDS = 208
VOICED_RECORD_PREFIX = "voice"
VOICED_SAMPLE_RATE = 8000

# --- Label Mapping ---
LABEL_HEALTHY = 0
LABEL_PATHOLOGICAL = 1
HEALTHY_DIAGNOSIS = "healthy"

# Multi-class pathology categories in VOICED
PATHOLOGY_CATEGORIES = [
    "hyperkinetic dysphonia",
    "hypokinetic dysphonia",
    "reflux laryngitis",
    "vocal fold nodules",
    "prolapse",
    "glottic insufficiency",
    "vocal fold paralysis",
]

# --- Audio Preprocessing ---
TRIM_START_SEC = 0.5
TRIM_END_SEC = 0.5
MIN_DURATION_SEC = 1.0
SNR_THRESHOLD_DB = 30  # minimum acceptable SNR per proposal section A

# Amplitude normalization target peak
AMPLITUDE_NORM_PEAK = 0.95

# --- Pitch Analysis ---
PITCH_FLOOR_HZ = 75
PITCH_CEILING_HZ = 500

# --- Formant Analysis ---
MAX_NUM_FORMANTS = 5
MAX_FORMANT_FREQ_HZ = 4000  # Nyquist-limited for 8kHz data

# --- MFCC / Spectral ---
NUM_MFCC_COEFFICIENTS = 13
N_FFT = 512
HOP_LENGTH = 256

# --- Feature Names ---
SOURCE_FEATURES = ["f0_mean", "f0_std", "f0_median"]

PERTURBATION_FEATURES = [
    "jitter_local",
    "jitter_rap",
    "shimmer_local",
    "shimmer_apq3",
]

NOISE_FEATURES = ["hnr_mean", "cpp_mean"]

FORMANT_FEATURES = [
    "f1_mean", "f1_bw",
    "f2_mean", "f2_bw",
    "f3_mean", "f3_bw",
]

MFCC_FEATURES = [f"mfcc_{i}" for i in range(1, NUM_MFCC_COEFFICIENTS + 1)]

SPECTRAL_FEATURES = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
]

# snr_db is a recording quality metric, NOT a classification feature.
# Used for quality filtering only, not included in model input.
QUALITY_METRIC_NAMES = ["snr_db"]

ALL_FEATURE_NAMES = (
    SOURCE_FEATURES
    + PERTURBATION_FEATURES
    + NOISE_FEATURES
    + FORMANT_FEATURES
    + MFCC_FEATURES
    + SPECTRAL_FEATURES
)

# Full output columns include quality metrics for reporting
ALL_OUTPUT_COLUMNS = ALL_FEATURE_NAMES + QUALITY_METRIC_NAMES

# --- Model Parameters ---
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Train/val/test split per proposal: 70/15/15
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
# TEST_SIZE_PROPOSAL = 0.15 (remainder)
