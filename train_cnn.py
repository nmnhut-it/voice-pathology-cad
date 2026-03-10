"""
Train CNN on Mel-spectrograms for voice pathology classification.
Per proposal: CNN exploits time-frequency representations (section E).
Downloads VOICED data, generates spectrograms, trains a simple CNN.
Output: results/cnn_results.txt
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
import librosa

from config import (
    RESULTS_DIR, MODELS_DIR, RANDOM_STATE,
    TRAIN_SIZE, VOICED_SAMPLE_RATE, N_FFT, HOP_LENGTH,
)
from data_loader import load_voiced_dataset
from preprocessing import preprocess_signal

# --- CNN Config ---
NUM_MEL_BANDS = 64
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 16
CNN_LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8
TARGET_SPEC_LENGTH = 128  # fixed width for spectrograms


class MelSpectrogramDataset(Dataset):
    """PyTorch dataset that converts raw audio to mel-spectrograms."""

    def __init__(self, signals, labels, sample_rate):
        self.signals = signals
        self.labels = labels
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        processed, _, _ = preprocess_signal(signal, self.sample_rate)

        mel_spec = librosa.feature.melspectrogram(
            y=processed.astype(np.float32),
            sr=self.sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=NUM_MEL_BANDS,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = _pad_or_truncate(mel_spec_db, TARGET_SPEC_LENGTH)

        # Normalize to [0, 1]
        spec_min = mel_spec_db.min()
        spec_max = mel_spec_db.max()
        if spec_max > spec_min:
            mel_spec_db = (mel_spec_db - spec_min) / (spec_max - spec_min)

        # Shape: (1, n_mels, time) — single channel
        tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        label = torch.FloatTensor([self.labels[idx]])
        return tensor, label


def _pad_or_truncate(spectrogram: np.ndarray, target_length: int) -> np.ndarray:
    """Ensure spectrogram has fixed time dimension."""
    _, current_length = spectrogram.shape
    if current_length >= target_length:
        return spectrogram[:, :target_length]
    padding = np.full(
        (spectrogram.shape[0], target_length - current_length),
        spectrogram.min(),
    )
    return np.concatenate([spectrogram, padding], axis=1)


class VoiceCNN(nn.Module):
    """Simple CNN for mel-spectrogram binary classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def _evaluate(model, loader, device):
    """Evaluate model, return predictions and probabilities."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.numpy().flatten().astype(int))
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading VOICED dataset...")
    start = time.time()
    df = load_voiced_dataset()
    print(f"  Loaded in {time.time() - start:.1f}s")

    signals = df["signal"].tolist()
    labels = df["label"].tolist()
    sr = VOICED_SAMPLE_RATE

    # 70/15/15 split
    idx = list(range(len(signals)))
    idx_train, idx_temp, _, y_temp = train_test_split(
        idx, labels, test_size=1.0 - TRAIN_SIZE,
        stratify=labels, random_state=RANDOM_STATE,
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE,
    )

    train_ds = MelSpectrogramDataset(
        [signals[i] for i in idx_train], [labels[i] for i in idx_train], sr
    )
    val_ds = MelSpectrogramDataset(
        [signals[i] for i in idx_val], [labels[i] for i in idx_val], sr
    )
    test_ds = MelSpectrogramDataset(
        [signals[i] for i in idx_test], [labels[i] for i in idx_test], sr
    )

    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=CNN_BATCH_SIZE)

    # Handle class imbalance with weighted loss
    n_pos = sum(labels[i] for i in idx_train)
    n_neg = len(idx_train) - n_pos
    pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)

    model = VoiceCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)

    # Training loop with early stopping
    print(f"\nTraining CNN for up to {CNN_EPOCHS} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(CNN_EPOCHS):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(val_ds)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "cnn_v2.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_v2.pt", weights_only=True))
    y_true, y_pred, y_proba = _evaluate(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.0
    report = classification_report(
        y_true, y_pred, target_names=["Healthy", "Pathological"]
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- CNN Test Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save results
    output_path = RESULTS_DIR / "cnn_results.txt"
    with open(output_path, "w") as f:
        f.write("CNN (Mel-Spectrogram) Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Architecture: 3-layer CNN, {NUM_MEL_BANDS} mel bands\n")
        f.write(f"Epochs trained: {epoch+1}\n\n")
        f.write(f"Test Accuracy:  {accuracy:.4f}\n")
        f.write(f"Test F1-score:  {f1:.4f}\n")
        f.write(f"Test ROC-AUC:   {auc:.4f}\n")
        f.write(f"\n{report}\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
