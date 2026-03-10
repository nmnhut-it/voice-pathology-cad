"""
Visualization module for voice pathology analysis results.
Generates feature distribution plots, confusion matrix, and radar chart.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from config import RESULTS_DIR, LABEL_HEALTHY, LABEL_PATHOLOGICAL, ALL_FEATURE_NAMES

# Readable labels for plots
LABEL_NAMES = {LABEL_HEALTHY: "Healthy", LABEL_PATHOLOGICAL: "Pathological"}


def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series):
    """Box plots comparing feature distributions between healthy and pathological."""
    _ensure_results_dir()
    n_features = len(X.columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    plot_df = X.copy()
    plot_df["label"] = y.map(LABEL_NAMES)

    for i, feature in enumerate(X.columns):
        sns.boxplot(data=plot_df, x="label", y=feature, ax=axes[i], palette="Set2")
        axes[i].set_title(feature)
        axes[i].set_xlabel("")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "feature_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray):
    """Save confusion matrix as heatmap."""
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Pathological"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - Random Forest v1")

    path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(importance_df: pd.DataFrame):
    """Horizontal bar chart of feature importances."""
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df_sorted = importance_df.sort_values("importance", ascending=True)
    ax.barh(
        importance_df_sorted["feature"],
        importance_df_sorted["importance"],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance - Random Forest v1")

    path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_radar_chart(X: pd.DataFrame, y: pd.Series):
    """
    Radar chart comparing mean feature values (normalized) between groups.
    Useful for visualizing multi-parameter differences.
    """
    _ensure_results_dir()
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled["label"] = y.values

    means = X_scaled.groupby("label").mean()
    features = means.columns.tolist()
    n_features = len(features)

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for label, name in LABEL_NAMES.items():
        if label in means.index:
            values = means.loc[label].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=8)
    ax.set_title("Radar Chart: Healthy vs Pathological", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = RESULTS_DIR / "radar_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
