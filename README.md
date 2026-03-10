# Voice Pathology CAD System

Computer-Aided Diagnosis system for voice pathology classification (Healthy vs Pathological) using acoustic features extracted from sustained vowel recordings.

## Results (SVD Dataset — 1983 samples, 32 features)

| Model | CV Accuracy | CV AUC | Test Accuracy | Test AUC | Sensitivity | Specificity |
|-------|-------------|--------|---------------|----------|-------------|-------------|
| **Random Forest** | **76.9%** | **0.849** | **75.5%** | **0.827** | 0.826 | 0.621 |
| SVM (RBF) | 74.6% | 0.835 | 74.2% | 0.825 | 0.703 | 0.816 |
| MLP | 74.8% | 0.825 | 74.5% | 0.821 | 0.836 | 0.573 |
| Logistic Regression | 71.8% | 0.811 | 72.8% | 0.820 | 0.713 | 0.757 |

**Target**: Accuracy > 90%, AUC > 0.95

## Top Features (by RF Gini Importance)

1. shimmer_local (0.068)
2. f3_mean (0.065)
3. hnr_mean (0.058)
4. shimmer_apq3 (0.050)
5. jitter_local (0.047)

## Datasets

- **SVD** (Saarbrucken Voice Database): ~2000 recordings, vowels /a/,/i/,/u/ at 50kHz
- **VOICED** (PhysioNet): 208 recordings, vowel /a/ at 8kHz

## Features (32 dimensions)

| Group | Features |
|-------|----------|
| Source | F0 mean, F0 std, F0 median |
| Perturbation | Jitter (local, RAP), Shimmer (local, APQ3) |
| Noise | HNR mean, CPP mean |
| Formant | F1-F3 mean + bandwidth |
| MFCC | 13 coefficients (mean over time) |
| Spectral | Centroid, Bandwidth, Rolloff, Flatness |

## Pipeline

```
Audio Signal → Preprocessing (normalize, trim) → Feature Extraction (Praat + Librosa)
→ Imputation → Scaling → Classification → Evaluation (5-fold CV + 70/15/15 split)
```

## Project Structure

```
config.py                 # Constants, feature names, model params
preprocessing.py          # Amplitude normalization, SNR, trimming
feature_extraction.py     # Praat + Librosa feature extraction (32 features)
data_loader.py            # VOICED (PhysioNet) loader
svd_loader.py             # SVD loader (NSP format parser)
model.py                  # RF, SVM, MLP, LR pipelines with evaluation
vowel_space.py            # VSA/VAI computation (Convex Hull on F1-F2)
train_improved.py         # Feature selection + SMOTE + GridSearchCV
main.py                   # VOICED pipeline
main_svd.py               # SVD pipeline (all 4 models)
visualize.py              # Plots (box plots, confusion matrix, radar chart)
```

## Requirements

```
numpy pandas scikit-learn librosa praat-parselmouth
imbalanced-learn seaborn matplotlib scipy joblib wfdb
```

## Usage

```bash
# Run SVD pipeline (extract features + train all models)
python main_svd.py

# Run improved training with feature selection + SMOTE + tuning
python train_improved.py
```
