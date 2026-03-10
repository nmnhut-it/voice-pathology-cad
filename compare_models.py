"""
Compare all trained models (RF, SVM, MLP, CNN) side by side.
Reads result files from results/ directory, outputs comparison table.
"""
import re
from pathlib import Path

from config import RESULTS_DIR

RESULT_FILES = {
    "Random Forest": "rf_results.txt",
    "SVM (RBF)": "svm_results.txt",
    "MLP": "mlp_results.txt",
    "CNN": "cnn_results.txt",
}

METRIC_PATTERNS = {
    "Accuracy": r"Test Accuracy:\s+([\d.]+)",
    "F1-score": r"Test F1-score:\s+([\d.]+)",
    "ROC-AUC": r"Test ROC-AUC:\s+([\d.]+)",
}


def parse_results(filepath: Path) -> dict:
    """Extract metric values from a result file."""
    if not filepath.exists():
        return {}
    text = filepath.read_text()
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        metrics[name] = float(match.group(1)) if match else None
    return metrics


def main():
    print("=" * 65)
    print("MODEL COMPARISON - Voice Pathology Classification")
    print("=" * 65)
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1-score':>10} {'ROC-AUC':>10}")
    print("-" * 52)

    for model_name, filename in RESULT_FILES.items():
        metrics = parse_results(RESULTS_DIR / filename)
        if not metrics:
            print(f"{model_name:<20} {'(not run)':>10}")
            continue

        acc = metrics.get("Accuracy")
        f1 = metrics.get("F1-score")
        auc = metrics.get("ROC-AUC")
        print(f"{model_name:<20} "
              f"{acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    print("-" * 52)
    print("\nTarget: Accuracy > 90%, AUC > 0.95")


if __name__ == "__main__":
    main()
