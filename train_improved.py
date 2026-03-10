"""
Improved training pipeline with feature selection, hyperparameter tuning, and SMOTE.
Diagnoses and fixes data quality issues before training.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    RANDOM_STATE,
    CV_FOLDS,
    DATA_DIR,
    RESULTS_DIR,
)
from model import ModelType, split_data, evaluate_predictions

# Features that are 99.9% NaN in SVD data (pitch detection failure at 50kHz)
SVD_NAN_FEATURES = [
    "f0_mean", "f0_std", "f0_median",
    "jitter_local", "jitter_rap",
    "shimmer_local", "shimmer_apq3",
]

# Threshold: drop features with more than this fraction of NaN values
NAN_DROP_THRESHOLD = 0.5

SVD_RESULTS_DIR = RESULTS_DIR / "svd_improved"


def load_and_clean_features(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load features CSV, drop high-NaN columns, report data quality."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    # Report NaN columns
    nan_fractions = X.isnull().mean()
    high_nan = nan_fractions[nan_fractions > NAN_DROP_THRESHOLD]
    if len(high_nan) > 0:
        print(f"Dropping {len(high_nan)} features with >{NAN_DROP_THRESHOLD*100:.0f}% NaN:")
        for feat, frac in high_nan.items():
            print(f"  {feat}: {frac*100:.1f}% NaN")
        X = X.drop(columns=high_nan.index)

    # Report remaining NaN
    remaining_nan = X.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"Remaining NaN cells: {remaining_nan} (will be imputed)")
    else:
        print("No remaining NaN values.")

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    return X, y


def analyze_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank features by mutual information with the target label."""
    # Impute for MI calculation
    imputer = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns
    )

    mi_scores = mutual_info_classif(X_clean, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mutual_info": mi_scores,
    }).sort_values("mutual_info", ascending=False)

    print("\nMutual Information Ranking:")
    for _, row in mi_df.iterrows():
        bar = "#" * int(row["mutual_info"] * 200)
        print(f"  {row['feature']:22s} {row['mutual_info']:.4f} {bar}")

    return mi_df


def _get_param_grid(model_type: str) -> dict:
    """Return hyperparameter search grid for each model type."""
    if model_type == ModelType.RANDOM_FOREST:
        return {
            "classifier__n_estimators": [100, 200, 500],
            "classifier__max_depth": [5, 10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
        }
    if model_type == ModelType.SVM:
        return {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__gamma": ["scale", "auto", 0.01, 0.001],
            "classifier__kernel": ["rbf", "poly"],
        }
    if model_type == ModelType.MLP:
        return {
            "classifier__hidden_layer_sizes": [
                (128, 64), (256, 128), (128, 64, 32), (64, 32),
            ],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__learning_rate_init": [0.001, 0.01],
        }
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver": ["lbfgs", "liblinear"],
            "classifier__penalty": ["l2"],
        }
    raise ValueError(f"Unknown model type: {model_type}")


def _create_classifier(model_type: str):
    """Create base classifier for hyperparameter tuning."""
    if model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_type == ModelType.SVM:
        return SVC(
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
    if model_type == ModelType.MLP:
        return MLPClassifier(
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def build_smote_pipeline(model_type: str) -> ImbPipeline:
    """Build pipeline with SMOTE oversampling for class imbalance."""
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", _create_classifier(model_type)),
    ])


def build_base_pipeline(model_type: str) -> Pipeline:
    """Build standard pipeline without SMOTE."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", _create_classifier(model_type)),
    ])


def evaluate_cv(pipeline, X, y, label: str) -> dict:
    """Run stratified k-fold CV and return metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]

    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    metrics = {}
    print(f"\n  {label}:")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        metrics[metric] = {"mean": np.mean(scores), "std": np.std(scores)}
        print(f"    {metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    return metrics


def run_hyperparameter_search(
    X: pd.DataFrame, y: pd.Series, model_type: str
) -> dict:
    """GridSearchCV for optimal hyperparameters."""
    pipeline = build_base_pipeline(model_type)
    param_grid = _get_param_grid(model_type)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X, y)

    print(f"\n  Best params: {grid_search.best_params_}")
    print(f"  Best CV AUC: {grid_search.best_score_:.4f}")
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
    }


def train_test_evaluation(pipeline, X, y, label: str) -> dict:
    """Train/test split evaluation with results printing."""
    splits = split_data(X, y)
    pipeline.fit(splits["X_train"], splits["y_train"])

    y_pred = pipeline.predict(splits["X_test"])
    y_proba = pipeline.predict_proba(splits["X_test"])[:, 1]
    results = evaluate_predictions(splits["y_test"], y_pred, y_proba)

    print(f"\n  {label} - Test Results:")
    print(f"    Accuracy: {results['accuracy']:.4f}")
    print(f"    F1-score: {results['f1']:.4f}")
    print(f"    ROC-AUC:  {results['roc_auc']:.4f}")
    return results


def select_top_features(
    X: pd.DataFrame, mi_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    """Select top-k features by mutual information score."""
    top_features = mi_df.head(top_k)["feature"].tolist()
    return X[top_features]


def run_full_comparison(X: pd.DataFrame, y: pd.Series):
    """
    Compare all model configurations:
    - Baseline (25 features, default params)
    - Feature selection (top-k features)
    - SMOTE oversampling
    - Hyperparameter tuning
    """
    SVD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Feature importance analysis
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    mi_df = analyze_feature_importance(X, y)

    # Try top-k feature subsets
    feature_counts = [10, 15, 20, len(X.columns)]
    model_types = [
        ModelType.RANDOM_FOREST, ModelType.SVM,
        ModelType.MLP, ModelType.LOGISTIC_REGRESSION,
    ]
    all_results = []

    # Phase 1: Feature selection comparison
    print("\n" + "=" * 60)
    print("PHASE 1: FEATURE SELECTION (CV)")
    print("=" * 60)
    for model_type in model_types:
        for k in feature_counts:
            k = min(k, len(X.columns))
            label = f"{model_type} (top-{k})"
            X_subset = select_top_features(X, mi_df, k) if k < len(X.columns) else X
            pipeline = build_base_pipeline(model_type)
            metrics = evaluate_cv(pipeline, X_subset, y, label)
            all_results.append({
                "model": model_type,
                "features": k,
                "smote": False,
                "tuned": False,
                "cv_accuracy": metrics["accuracy"]["mean"],
                "cv_auc": metrics["roc_auc"]["mean"],
            })

    # Phase 2: SMOTE comparison (best feature count)
    print("\n" + "=" * 60)
    print("PHASE 2: SMOTE OVERSAMPLING (CV)")
    print("=" * 60)
    for model_type in model_types:
        # Use all features for SMOTE comparison
        label = f"{model_type} + SMOTE"
        pipeline = build_smote_pipeline(model_type)
        metrics = evaluate_cv(pipeline, X, y, label)
        all_results.append({
            "model": model_type,
            "features": len(X.columns),
            "smote": True,
            "tuned": False,
            "cv_accuracy": metrics["accuracy"]["mean"],
            "cv_auc": metrics["roc_auc"]["mean"],
        })

    # Phase 3: Hyperparameter tuning for best configs
    print("\n" + "=" * 60)
    print("PHASE 3: HYPERPARAMETER TUNING")
    print("=" * 60)
    tuned_results = {}
    for model_type in model_types:
        print(f"\n--- {model_type.upper()} ---")
        grid_result = run_hyperparameter_search(X, y, model_type)
        tuned_results[model_type] = grid_result
        all_results.append({
            "model": model_type,
            "features": len(X.columns),
            "smote": False,
            "tuned": True,
            "cv_accuracy": None,  # grid search uses AUC
            "cv_auc": grid_result["best_score"],
        })

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(all_results)
    print(f"\n{'Model':<22} {'Feats':>5} {'SMOTE':>5} {'Tuned':>5} "
          f"{'CV Acc':>8} {'CV AUC':>8}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        acc_str = f"{row['cv_accuracy']:.4f}" if row['cv_accuracy'] else "  N/A "
        print(f"{row['model']:<22} {row['features']:>5} "
              f"{'Yes':>5 if row['smote'] else 'No':>5} "
              f"{'Yes':>5 if row['tuned'] else 'No':>5} "
              f"{acc_str:>8} {row['cv_auc']:.4f}")

    # Save results
    results_df.to_csv(SVD_RESULTS_DIR / "comparison_results.csv", index=False)
    mi_df.to_csv(SVD_RESULTS_DIR / "feature_importance_mi.csv", index=False)

    # Save tuned params
    with open(SVD_RESULTS_DIR / "tuned_params.txt", "w") as f:
        for model_type, result in tuned_results.items():
            f.write(f"{model_type}:\n")
            f.write(f"  Best AUC: {result['best_score']:.4f}\n")
            f.write(f"  Params: {result['best_params']}\n\n")

    print(f"\nResults saved to {SVD_RESULTS_DIR}/")
    return results_df, mi_df, tuned_results


def main():
    """Run improved training on SVD features."""
    print("=" * 60)
    print("IMPROVED TRAINING PIPELINE (SVD)")
    print("=" * 60)

    csv_path = str(DATA_DIR / "svd_features.csv")
    X, y = load_and_clean_features(csv_path)

    run_full_comparison(X, y)


if __name__ == "__main__":
    main()
