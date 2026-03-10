"""
Train and evaluate MLP (Multi-Layer Perceptron) on extracted voice features.
MLP is reported as most effective for acoustic feature input (proposal ref [11]).
Reads features_v2.csv, outputs results to results/mlp_results.txt.
"""
import sys
import pandas as pd
from pathlib import Path

from config import DATA_DIR, RESULTS_DIR
from model import evaluate_cross_validation, train_and_evaluate, ModelType
from visualize import plot_confusion_matrix

FEATURES_PATH = DATA_DIR / "features_v2.csv"
OUTPUT_PATH = RESULTS_DIR / "mlp_results.txt"


def main():
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found. Run main.py first.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Loaded {len(df)} samples, {len(X.columns)} features")
    print(f"Class distribution: {y.value_counts().to_dict()}\n")

    # Cross-validation
    print("=" * 50)
    print("MLP (128-64) - 5-fold Stratified CV")
    print("=" * 50)
    cv_metrics = evaluate_cross_validation(X, y, ModelType.MLP)

    # Train/test evaluation
    print("\n" + "=" * 50)
    print("MLP (128-64) - 70/15/15 Split")
    print("=" * 50)
    results = train_and_evaluate(X, y, ModelType.MLP)

    # Save confusion matrix plot
    plot_confusion_matrix(results["confusion_matrix"])
    target = RESULTS_DIR / "confusion_matrix_mlp.png"
    target.unlink(missing_ok=True)
    Path(RESULTS_DIR / "confusion_matrix.png").rename(target)

    # Write results to file
    with open(OUTPUT_PATH, "w") as f:
        f.write("MLP (Multi-Layer Perceptron) Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("Architecture: 128-64, ReLU, early stopping\n\n")
        f.write("Cross-Validation:\n")
        for metric, vals in cv_metrics.items():
            f.write(f"  {metric}: {vals['mean']:.4f} (+/- {vals['std']:.4f})\n")
        f.write(f"\nTest Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Test F1-score:  {results['f1']:.4f}\n")
        f.write(f"Test ROC-AUC:   {results['roc_auc']:.4f}\n")
        f.write(f"\n{results['classification_report']}\n")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
