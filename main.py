"""
Main pipeline: Voice Pathology CAD System v2
Improved preprocessing, expanded features (MFCCs, spectral, formant BW).
Input: VOICED database from PhysioNet (208 sustained vowel /a/ recordings)
Output: Features CSV, model files, evaluation metrics, visualization plots
"""
import time
import pandas as pd

from config import RESULTS_DIR, DATA_DIR, SNR_THRESHOLD_DB
from data_loader import load_voiced_dataset
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


def run_pipeline():
    """Execute the full CAD pipeline: load -> preprocess -> extract -> train."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading VOICED dataset from PhysioNet...")
    print("=" * 60)
    start = time.time()
    df = load_voiced_dataset()
    print(f"  Loaded in {time.time() - start:.1f}s\n")

    df[["record_id", "diagnosis", "label", "age", "sex"]].to_csv(
        RESULTS_DIR / "dataset_metadata.csv", index=False
    )

    # Step 2: Extract features (includes preprocessing)
    print("=" * 60)
    print("STEP 2: Preprocessing + feature extraction...")
    print("=" * 60)
    start = time.time()
    X, y = extract_features_dataframe(df)
    print(f"  Extracted in {time.time() - start:.1f}s\n")

    # Report SNR quality
    snr_col = X["snr_db"]
    low_snr_count = (snr_col < SNR_THRESHOLD_DB).sum()
    print(f"  SNR stats: mean={snr_col.mean():.1f} dB, "
          f"min={snr_col.min():.1f} dB, "
          f"below {SNR_THRESHOLD_DB} dB: {low_snr_count}/{len(X)}")

    # Save features
    feature_path = DATA_DIR / "features_v2.csv"
    feature_output = X.copy()
    feature_output["label"] = y.values
    feature_output.to_csv(feature_path, index=False)
    print(f"  Features saved to {feature_path}\n")

    # Step 3: Random Forest cross-validation
    print("=" * 60)
    print("STEP 3: Random Forest - 5-fold CV...")
    print("=" * 60)
    evaluate_cross_validation(X, y, ModelType.RANDOM_FOREST)

    # Step 4: Random Forest train/test evaluation
    print("\n" + "=" * 60)
    print("STEP 4: Random Forest - 70/15/15 split...")
    print("=" * 60)
    rf_results = train_and_evaluate(X, y, ModelType.RANDOM_FOREST)

    # Step 5: Feature importance
    print("\n" + "=" * 60)
    print("STEP 5: Feature importance analysis...")
    print("=" * 60)
    importance_df = get_feature_importance(X, y)

    # Step 6: Visualizations
    print("\n" + "=" * 60)
    print("STEP 6: Generating visualizations...")
    print("=" * 60)
    plot_feature_distributions(X, y)
    plot_confusion_matrix(rf_results["confusion_matrix"])
    if importance_df is not None:
        plot_feature_importance(importance_df)
    plot_radar_chart(X, y)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (v2 - improved features)")
    print("=" * 60)
    print(f"  Features: {len(X.columns)} dimensions")
    print(f"  Test Accuracy:  {rf_results['accuracy']:.4f}")
    print(f"  Test F1-score:  {rf_results['f1']:.4f}")
    print(f"  Test ROC-AUC:   {rf_results['roc_auc']:.4f}")
    print(f"  Results dir: {RESULTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
