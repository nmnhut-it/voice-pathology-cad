"""
Longitudinal voice change analysis (proposal section D).
Computes ΔV = V(T1) - V(T0) to quantify pre/post-surgery voice changes.
Works with feature vectors from any two time points.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    ALL_FEATURE_NAMES,
    SOURCE_FEATURES,
    PERTURBATION_FEATURES,
    NOISE_FEATURES,
    FORMANT_FEATURES,
    RESULTS_DIR,
)

# Feature groups for clinical reporting
CLINICAL_GROUPS = {
    "Source (F0)": SOURCE_FEATURES,
    "Perturbation": PERTURBATION_FEATURES,
    "Noise/Periodicity": NOISE_FEATURES,
    "Formant": FORMANT_FEATURES,
}

# Features where increase = improvement (higher is healthier)
IMPROVEMENT_IF_INCREASE = {"hnr_mean", "cpp_mean"}

# Features where decrease = improvement (lower is healthier)
IMPROVEMENT_IF_DECREASE = {
    "jitter_local", "jitter_rap",
    "shimmer_local", "shimmer_apq3",
    "f0_std",
}


def compute_delta_v(
    v_t0: dict[str, float], v_t1: dict[str, float],
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute ΔV = V(T1) - V(T0) for each feature.
    Positive delta = feature increased post-surgery.
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES

    delta = {}
    for feat in feature_names:
        val_t0 = v_t0.get(feat, np.nan)
        val_t1 = v_t1.get(feat, np.nan)
        if np.isnan(val_t0) or np.isnan(val_t1):
            delta[feat] = np.nan
        else:
            delta[feat] = val_t1 - val_t0
    return delta


def assess_change_direction(delta: dict[str, float]) -> dict[str, str]:
    """
    Assess whether each feature change indicates improvement or decline.
    Returns dict of feature -> 'improved' | 'declined' | 'unchanged' | 'unknown'.
    """
    assessment = {}
    for feat, change in delta.items():
        if np.isnan(change):
            assessment[feat] = "unknown"
            continue
        if abs(change) < 1e-6:
            assessment[feat] = "unchanged"
            continue

        if feat in IMPROVEMENT_IF_INCREASE:
            assessment[feat] = "improved" if change > 0 else "declined"
        elif feat in IMPROVEMENT_IF_DECREASE:
            assessment[feat] = "improved" if change < 0 else "declined"
        else:
            assessment[feat] = "changed"
    return assessment


def generate_longitudinal_report(
    v_t0: dict[str, float],
    v_t1: dict[str, float],
) -> str:
    """Generate clinical report of voice changes between T0 and T1."""
    delta = compute_delta_v(v_t0, v_t1)
    assessment = assess_change_direction(delta)

    report = "## Longitudinal Voice Change Report\n\n"
    report += "### ΔV = V(T1) - V(T0)\n\n"

    for group_name, features in CLINICAL_GROUPS.items():
        report += f"#### {group_name}\n"
        report += "| Parameter | T0 | T1 | ΔV | Direction |\n"
        report += "|-----------|----|----|-----|----------|\n"

        for feat in features:
            t0_val = v_t0.get(feat, np.nan)
            t1_val = v_t1.get(feat, np.nan)
            d = delta.get(feat, np.nan)
            status = assessment.get(feat, "unknown")

            if np.isnan(d):
                report += f"| {feat} | N/A | N/A | N/A | - |\n"
            else:
                sign = "+" if d > 0 else ""
                report += f"| {feat} | {t0_val:.4f} | {t1_val:.4f} | {sign}{d:.4f} | {status} |\n"
        report += "\n"

    # Overall summary
    improved = sum(1 for v in assessment.values() if v == "improved")
    declined = sum(1 for v in assessment.values() if v == "declined")
    total_assessed = improved + declined
    if total_assessed > 0:
        report += f"### Summary\n"
        report += f"- Improved: {improved}/{total_assessed} parameters\n"
        report += f"- Declined: {declined}/{total_assessed} parameters\n"
        pct = improved / total_assessed * 100
        report += f"- Overall improvement rate: {pct:.0f}%\n"

    return report


def plot_delta_v_chart(
    v_t0: dict[str, float],
    v_t1: dict[str, float],
    output_path=None,
) -> plt.Figure:
    """Bar chart of ΔV for clinical features, colored by improvement."""
    clinical_feats = []
    for feats in CLINICAL_GROUPS.values():
        clinical_feats.extend(feats)

    delta = compute_delta_v(v_t0, v_t1, clinical_feats)
    assessment = assess_change_direction(delta)

    # Filter valid features
    feats = [f for f in clinical_feats if not np.isnan(delta.get(f, np.nan))]
    if not feats:
        return None

    values = [delta[f] for f in feats]
    colors = []
    for f in feats:
        status = assessment[f]
        if status == "improved":
            colors.append("#2ecc71")
        elif status == "declined":
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")

    fig, ax = plt.subplots(figsize=(10, max(6, len(feats) * 0.4)))
    bars = ax.barh(feats, values, color=colors)
    ax.set_xlabel("ΔV (T1 - T0)")
    ax.set_title("Voice Parameter Changes: Pre vs Post Surgery")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Improved"),
        Patch(facecolor="#e74c3c", label="Declined"),
        Patch(facecolor="#95a5a6", label="Changed"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig
