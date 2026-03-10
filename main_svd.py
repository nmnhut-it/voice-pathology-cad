"""
Main pipeline for SVD dataset: load -> extract features -> train all models.
SVD has ~2000+ records at 50kHz with vowels /a/, /i/, /u/.
This is expected to significantly outperform the VOICED pipeline (208 samples at 8kHz).
"""
import time
import pandas as pd

from config import RESULTS_DIR, DATA_DIR, ALL_FEATURE_NAMES
from svd_loader import load_svd_dataset
from feature_extraction import extract_features_dataframe
from model import (
    evaluate_cross_validation,
    train_and_evaluate,
    get_feature_importance,
    ModelType,
)
from visualize import (
    plot_feature_distributions,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_radar_chart,
)

SVD_RESULTS_DIR = RESULTS_DIR / "svd"
SVD_FEATURES_PATH = DATA_DIR / "svd_features.csv"


def run_pipeline():
    """Full pipeline on SVD dataset with all three model types."""
    SVD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load SVD data (vowel /a/ normal pitch)
    print("=" * 60)
    print("STEP 1: Loading SVD dataset (vowel /a/)...")
    print("=" * 60)
    start = time.time()
    df = load_svd_dataset("a")
    print(f"  Loaded in {time.time() - start:.1f}s\n")

    # Step 2: Feature extraction
    print("=" * 60)
    print("STEP 2: Extracting acoustic features...")
    print("=" * 60)
    start = time.time()
    checkpoint_path = str(DATA_DIR / "svd_features_checkpoint.csv")
    X, y = extract_features_dataframe(df, checkpoint_path=checkpoint_path)
    print(f"  Extracted in {time.time() - start:.1f}s\n")

    # Save features
    feature_output = X.copy()
    feature_output["label"] = y.values
    feature_output.to_csv(SVD_FEATURES_PATH, index=False)
    print(f"  Saved to {SVD_FEATURES_PATH}")
    print(f"  Shape: {X.shape[0]} samples x {X.shape[1]} features\n")

    # Step 3: Train and evaluate all models
    all_results = {}
    for model_type in [
        ModelType.RANDOM_FOREST, ModelType.SVM,
        ModelType.MLP, ModelType.LOGISTIC_REGRESSION,
    ]:
        print("=" * 60)
        print(f"MODEL: {model_type.upper()} - 5-fold CV")
        print("=" * 60)
        cv_metrics = evaluate_cross_validation(X, y, model_type)

        print(f"\n{model_type.upper()} - 70/15/15 Split")
        print("-" * 40)
        results = train_and_evaluate(X, y, model_type)
        all_results[model_type] = {
            "cv": cv_metrics,
            "test": results,
        }
        print()

    # Step 4: Feature importance (RF only)
    print("=" * 60)
    print("Feature Importance (Random Forest)")
    print("=" * 60)
    importance_df = get_feature_importance(X, y)

    # Step 5: Visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    plot_feature_distributions(X, y)
    rf_cm = all_results[ModelType.RANDOM_FOREST]["test"]["confusion_matrix"]
    plot_confusion_matrix(rf_cm)
    if importance_df is not None:
        plot_feature_importance(importance_df)
    plot_radar_chart(X, y)

    # Move plots to svd results dir
    for png in RESULTS_DIR.glob("*.png"):
        target = SVD_RESULTS_DIR / png.name
        target.unlink(missing_ok=True)
        png.rename(target)

    # Summary comparison
    print("\n" + "=" * 60)
    print("SVD RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'CV Acc':>8} {'CV AUC':>8} "
          f"{'Test Acc':>9} {'Test AUC':>9}")
    print("-" * 56)
    for model_type, res in all_results.items():
        cv_acc = res["cv"]["accuracy"]["mean"]
        cv_auc = res["cv"]["roc_auc"]["mean"]
        t_acc = res["test"]["accuracy"]
        t_auc = res["test"]["roc_auc"]
        print(f"{model_type:<20} {cv_acc:>8.4f} {cv_auc:>8.4f} "
              f"{t_acc:>9.4f} {t_auc:>9.4f}")
    print("-" * 56)
    print("Target: Accuracy > 90%, AUC > 0.95")

    # Save summary
    summary_path = SVD_RESULTS_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("SVD Dataset - Model Comparison\n")
        f.write(f"Samples: {len(X)}, Features: {len(X.columns)}\n\n")
        for model_type, res in all_results.items():
            f.write(f"{model_type}:\n")
            f.write(f"  CV Accuracy:  {res['cv']['accuracy']['mean']:.4f}\n")
            f.write(f"  CV AUC:       {res['cv']['roc_auc']['mean']:.4f}\n")
            f.write(f"  Test Accuracy:  {res['test']['accuracy']:.4f}\n")
            f.write(f"  Test F1-score:  {res['test']['f1']:.4f}\n")
            f.write(f"  Test ROC-AUC:   {res['test']['roc_auc']:.4f}\n\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_pipeline()
