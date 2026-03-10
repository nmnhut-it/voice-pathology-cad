"""
Vowel Space Analysis module (proposal section C).
Computes VSA (Vowel Space Area) and VAI (Vowel Articulation Index)
from multi-vowel formant measurements on the F1-F2 plane.
Uses Convex Hull for VSA and standard VAI formula.
"""
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.spatial import ConvexHull

from config import (
    MAX_NUM_FORMANTS,
    PITCH_FLOOR_HZ,
    RESULTS_DIR,
    LABEL_HEALTHY,
    LABEL_PATHOLOGICAL,
)
from preprocessing import preprocess_signal

# Formant ceiling depends on sample rate and speaker sex
FORMANT_CEILING_MALE = 5000
FORMANT_CEILING_FEMALE = 5500
FORMANT_CEILING_DEFAULT = 5000


def _get_formant_ceiling(sex: str) -> int:
    """Return appropriate max formant frequency based on speaker sex."""
    if sex in ("m", "M", "male"):
        return FORMANT_CEILING_MALE
    if sex in ("w", "W", "f", "F", "female"):
        return FORMANT_CEILING_FEMALE
    return FORMANT_CEILING_DEFAULT


def extract_formants_for_vowel(
    signal: np.ndarray, sample_rate: int, sex: str = ""
) -> tuple[float, float]:
    """
    Extract mean F1 and F2 from a single vowel signal.
    Returns (f1_mean, f2_mean) in Hz.
    """
    processed, _, _ = preprocess_signal(signal, sample_rate)
    sound = parselmouth.Sound(processed, sampling_frequency=sample_rate)
    max_freq = _get_formant_ceiling(sex)

    formant = call(
        sound, "To Formant (burg)",
        0.0, MAX_NUM_FORMANTS, max_freq, 0.025, 50.0,
    )

    duration = sound.get_total_duration()
    t_start = duration * 0.2
    t_end = duration * 0.8
    time_points = np.linspace(t_start, t_end, 10)

    f1_vals, f2_vals = [], []
    for t in time_points:
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        if not np.isnan(f1):
            f1_vals.append(f1)
        if not np.isnan(f2):
            f2_vals.append(f2)

    f1_mean = np.mean(f1_vals) if f1_vals else np.nan
    f2_mean = np.mean(f2_vals) if f2_vals else np.nan
    return f1_mean, f2_mean


def compute_vowel_space_features(
    multivowel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute VSA and VAI for each record in a multi-vowel DataFrame.
    Expects columns: signal_a, signal_i, signal_u, sample_rate, sex, label.

    VSA = area of triangle formed by /a/, /i/, /u/ on F1-F2 plane.
    VAI = (F2i + F1a) / (F1i + F1u + F2a + F2u)
      where Fni = formant n of vowel i.

    Returns DataFrame with: record_id, f1_a, f2_a, f1_i, f2_i, f1_u, f2_u,
    vsa, vai, label.
    """
    results = []
    total = len(multivowel_df)

    for count, (_, row) in enumerate(multivowel_df.iterrows(), 1):
        sr = row["sample_rate"]
        sex = row.get("sex", "")

        f1_a, f2_a = extract_formants_for_vowel(row["signal_a"], sr, sex)
        f1_i, f2_i = extract_formants_for_vowel(row["signal_i"], sr, sex)
        f1_u, f2_u = extract_formants_for_vowel(row["signal_u"], sr, sex)

        # VSA: triangle area using shoelace formula
        vsa = _triangle_area(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u)

        # VAI: Vowel Articulation Index (Roy et al., 2009)
        vai = _compute_vai(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u)

        results.append({
            "record_id": row.get("record_id", count),
            "f1_a": f1_a, "f2_a": f2_a,
            "f1_i": f1_i, "f2_i": f2_i,
            "f1_u": f1_u, "f2_u": f2_u,
            "vsa": vsa,
            "vai": vai,
            "label": row["label"],
        })

        if count % 50 == 0 or count == total:
            print(f"  VSA/VAI: {count}/{total} records processed")

    return pd.DataFrame(results)


def _triangle_area(
    f1_a: float, f2_a: float,
    f1_i: float, f2_i: float,
    f1_u: float, f2_u: float,
) -> float:
    """Compute triangle area on F1-F2 plane via shoelace formula."""
    if any(np.isnan(v) for v in [f1_a, f2_a, f1_i, f2_i, f1_u, f2_u]):
        return np.nan
    return 0.5 * abs(
        f1_a * (f2_i - f2_u)
        + f1_i * (f2_u - f2_a)
        + f1_u * (f2_a - f2_i)
    )


def _compute_vai(
    f1_a: float, f2_a: float,
    f1_i: float, f2_i: float,
    f1_u: float, f2_u: float,
) -> float:
    """
    Vowel Articulation Index (Roy et al., 2009).
    VAI = (F2i + F1a) / (F1i + F1u + F2a + F2u)
    Higher VAI indicates better articulatory precision.
    """
    if any(np.isnan(v) for v in [f1_a, f2_a, f1_i, f2_i, f1_u, f2_u]):
        return np.nan
    denominator = f1_i + f1_u + f2_a + f2_u
    if denominator == 0:
        return np.nan
    return (f2_i + f1_a) / denominator


def plot_vowel_space(
    vsa_df: pd.DataFrame,
    output_dir=None,
):
    """
    Plot VSA map on F1-F2 plane, comparing healthy vs pathological.
    Shows mean vowel positions and triangles for each group.
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    label_names = {LABEL_HEALTHY: "Healthy", LABEL_PATHOLOGICAL: "Pathological"}
    colors = {LABEL_HEALTHY: "#2ecc71", LABEL_PATHOLOGICAL: "#e74c3c"}

    for label, name in label_names.items():
        group = vsa_df[vsa_df["label"] == label]
        if group.empty:
            continue

        # Plot individual vowel positions (faded)
        for vowel, f1_col, f2_col, marker in [
            ("/a/", "f1_a", "f2_a", "o"),
            ("/i/", "f1_i", "f2_i", "s"),
            ("/u/", "f1_u", "f2_u", "^"),
        ]:
            ax.scatter(
                group[f2_col], group[f1_col],
                c=colors[label], marker=marker, alpha=0.15, s=20,
            )

        # Plot mean triangle
        mean_f1_a = group["f1_a"].mean()
        mean_f2_a = group["f2_a"].mean()
        mean_f1_i = group["f1_i"].mean()
        mean_f2_i = group["f2_i"].mean()
        mean_f1_u = group["f1_u"].mean()
        mean_f2_u = group["f2_u"].mean()

        triangle_f2 = [mean_f2_a, mean_f2_i, mean_f2_u, mean_f2_a]
        triangle_f1 = [mean_f1_a, mean_f1_i, mean_f1_u, mean_f1_a]
        ax.plot(triangle_f2, triangle_f1, "o-", color=colors[label],
                linewidth=2, markersize=10, label=name)
        ax.fill(triangle_f2, triangle_f1, color=colors[label], alpha=0.15)

        # Label vowel positions
        offset = 30
        ax.annotate("/a/", (mean_f2_a, mean_f1_a), fontsize=12,
                     fontweight="bold", color=colors[label],
                     xytext=(offset, offset), textcoords="offset points")
        ax.annotate("/i/", (mean_f2_i, mean_f1_i), fontsize=12,
                     fontweight="bold", color=colors[label],
                     xytext=(offset, -offset), textcoords="offset points")
        ax.annotate("/u/", (mean_f2_u, mean_f1_u), fontsize=12,
                     fontweight="bold", color=colors[label],
                     xytext=(-offset, offset), textcoords="offset points")

    ax.set_xlabel("F2 (Hz)", fontsize=12)
    ax.set_ylabel("F1 (Hz)", fontsize=12)
    ax.set_title("Vowel Space Map (F1-F2 Plane)", fontsize=14)
    ax.invert_yaxis()  # Convention: high F1 at bottom
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    path = output_dir / "vowel_space_map.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Print summary statistics
    for label, name in label_names.items():
        group = vsa_df[vsa_df["label"] == label]
        if group.empty:
            continue
        print(f"\n{name}:")
        print(f"  VSA mean: {group['vsa'].mean():.1f} (std: {group['vsa'].std():.1f})")
        print(f"  VAI mean: {group['vai'].mean():.4f} (std: {group['vai'].std():.4f})")
