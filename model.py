"""
Classification model builder and evaluator for voice pathology.
Supports Random Forest, SVM, MLP pipelines with unified evaluation.
Covers proposal section E (classification models).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib

from config import (
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_STATE,
    TRAIN_SIZE,
    CV_FOLDS,
    MODELS_DIR,
)

TARGET_NAMES = ["Healthy", "Pathological"]


class ModelType:
    """Enum-like class for supported model types."""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    MLP = "mlp"
    LOGISTIC_REGRESSION = "logistic_regression"


def _create_classifier(model_type: str):
    """Factory: return classifier instance by model type."""
    if model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_type == ModelType.SVM:
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
    if model_type == ModelType.MLP:
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
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


def build_pipeline(model_type: str = ModelType.RANDOM_FOREST) -> Pipeline:
    """Build sklearn pipeline: imputation -> scaling -> classifier."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", _create_classifier(model_type)),
    ])


def split_data(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    70/15/15 stratified split per proposal section E.
    Returns dict with train/val/test X and y.
    """
    test_val_size = 1.0 - TRAIN_SIZE
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, stratify=y, random_state=RANDOM_STATE
    )
    # Split remaining 30% into 15%/15% (i.e., 50/50 of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    """Compute all evaluation metrics from predictions."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, target_names=TARGET_NAMES
        ),
    }


def evaluate_cross_validation(
    X: pd.DataFrame, y: pd.Series,
    model_type: str = ModelType.RANDOM_FOREST,
) -> dict:
    """Run stratified k-fold cross-validation, return metric scores."""
    pipeline = build_pipeline(model_type)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    metrics = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        metrics[metric] = {"mean": np.mean(scores), "std": np.std(scores)}
        print(f"  {metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    return metrics


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series,
    model_type: str = ModelType.RANDOM_FOREST,
) -> dict:
    """Train on 70/15/15 split, evaluate on test set, save model."""
    splits = split_data(X, y)
    pipeline = build_pipeline(model_type)
    pipeline.fit(splits["X_train"], splits["y_train"])

    y_pred = pipeline.predict(splits["X_test"])
    y_proba = pipeline.predict_proba(splits["X_test"])[:, 1]
    results = evaluate_predictions(splits["y_test"], y_pred, y_proba)

    print(f"\n--- {model_type.upper()} Test Results ---")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Sensitivity: {results['sensitivity']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print(f"F1-score:    {results['f1']:.4f}")
    print(f"ROC-AUC:     {results['roc_auc']:.4f}")
    print(f"\n{results['classification_report']}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_type}_v2.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    results["pipeline"] = pipeline
    return results


def get_feature_importance(
    X: pd.DataFrame, y: pd.Series,
    model_type: str = ModelType.RANDOM_FOREST,
) -> pd.DataFrame | None:
    """Return feature importances (only for tree-based models)."""
    if model_type != ModelType.RANDOM_FOREST:
        print(f"Feature importance not available for {model_type}")
        return None

    pipeline = build_pipeline(model_type)
    pipeline.fit(X, y)

    importances = pipeline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nFeature Importances (top 15):")
    for _, row in importance_df.head(15).iterrows():
        bar = "#" * int(row["importance"] * 80)
        print(f"  {row['feature']:22s} {row['importance']:.4f} {bar}")

    return importance_df
