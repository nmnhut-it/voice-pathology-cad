"""
CAD Web Application for Voice Pathology Analysis.
Implements proposal section F: VSA Map, Radar Chart, clinical report.
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io
import soundfile as sf
from pathlib import Path

from config import (
    ALL_FEATURE_NAMES,
    SOURCE_FEATURES,
    PERTURBATION_FEATURES,
    NOISE_FEATURES,
    FORMANT_FEATURES,
    MFCC_FEATURES,
    SPECTRAL_FEATURES,
    MODELS_DIR,
    RESULTS_DIR,
    DATA_DIR,
    LABEL_HEALTHY,
    LABEL_PATHOLOGICAL,
)
from preprocessing import preprocess_signal
from feature_extraction import extract_features_from_signal

# Feature groups for radar chart display
RADAR_GROUPS = {
    "Source (F0)": SOURCE_FEATURES,
    "Perturbation": PERTURBATION_FEATURES,
    "Noise": NOISE_FEATURES,
    "Formant": FORMANT_FEATURES,
    "Spectral": SPECTRAL_FEATURES,
}

LABEL_NAMES = {LABEL_HEALTHY: "Healthy", LABEL_PATHOLOGICAL: "Pathological"}
LABEL_COLORS = {LABEL_HEALTHY: "#2ecc71", LABEL_PATHOLOGICAL: "#e74c3c"}


def load_model(model_name: str):
    """Load a saved model from disk."""
    path = MODELS_DIR / f"{model_name}_v2.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def load_reference_stats() -> pd.DataFrame | None:
    """Load reference feature statistics from SVD dataset."""
    path = DATA_DIR / "svd_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def extract_features_from_audio(audio_data: np.ndarray, sr: int) -> dict | None:
    """Extract features from uploaded audio."""
    return extract_features_from_signal(audio_data, sr)


def plot_radar_chart(features: dict, ref_df: pd.DataFrame) -> plt.Figure:
    """
    Radar chart comparing patient features vs healthy/pathological means.
    Shows multi-parameter profile per proposal section F.
    """
    # Select key clinical features for radar
    radar_features = [
        "f0_mean", "f0_std", "jitter_local", "shimmer_local",
        "hnr_mean", "cpp_mean", "f1_mean", "f2_mean", "f3_mean",
        "spectral_centroid", "spectral_flatness",
    ]
    available = [f for f in radar_features if f in features and not np.isnan(features[f])]
    if len(available) < 3:
        return None

    # Normalize features to 0-1 range using reference data
    values = []
    healthy_vals = []
    pathological_vals = []
    labels = []

    for feat in available:
        col = ref_df[feat].dropna()
        feat_min, feat_max = col.min(), col.max()
        feat_range = feat_max - feat_min
        if feat_range == 0:
            continue

        val = (features[feat] - feat_min) / feat_range
        h_mean = (ref_df[ref_df["label"] == LABEL_HEALTHY][feat].mean() - feat_min) / feat_range
        p_mean = (ref_df[ref_df["label"] == LABEL_PATHOLOGICAL][feat].mean() - feat_min) / feat_range

        values.append(np.clip(val, 0, 1))
        healthy_vals.append(np.clip(h_mean, 0, 1))
        pathological_vals.append(np.clip(p_mean, 0, 1))
        labels.append(feat.replace("_", " ").title())

    n = len(labels)
    if n < 3:
        return None

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    healthy_vals += healthy_vals[:1]
    pathological_vals += pathological_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, healthy_vals, "o-", color="#2ecc71", linewidth=2, label="Healthy (mean)")
    ax.fill(angles, healthy_vals, alpha=0.1, color="#2ecc71")
    ax.plot(angles, pathological_vals, "o-", color="#e74c3c", linewidth=2, label="Pathological (mean)")
    ax.fill(angles, pathological_vals, alpha=0.1, color="#e74c3c")
    ax.plot(angles, values, "o-", color="#3498db", linewidth=3, label="Patient")
    ax.fill(angles, values, alpha=0.2, color="#3498db")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_title("Multi-Parameter Voice Profile", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig


def plot_vsa_map(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u, ref_df=None) -> plt.Figure:
    """Plot VSA map on F1-F2 plane per proposal section F."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot patient triangle
    f2_pts = [f2_a, f2_i, f2_u, f2_a]
    f1_pts = [f1_a, f1_i, f1_u, f1_a]
    ax.plot(f2_pts, f1_pts, "o-", color="#3498db", linewidth=3, markersize=12, label="Patient")
    ax.fill(f2_pts, f1_pts, alpha=0.2, color="#3498db")

    ax.annotate("/a/", (f2_a, f1_a), fontsize=14, fontweight="bold",
                xytext=(15, 15), textcoords="offset points")
    ax.annotate("/i/", (f2_i, f1_i), fontsize=14, fontweight="bold",
                xytext=(15, -15), textcoords="offset points")
    ax.annotate("/u/", (f2_u, f1_u), fontsize=14, fontweight="bold",
                xytext=(-25, 15), textcoords="offset points")

    ax.set_xlabel("F2 (Hz)", fontsize=12)
    ax.set_ylabel("F1 (Hz)", fontsize=12)
    ax.set_title("Vowel Space Area (VSA) Map", fontsize=14)
    ax.invert_yaxis()
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig


def generate_clinical_report(
    features: dict, prediction: int, probability: float,
    ref_df: pd.DataFrame,
) -> str:
    """Generate clinical summary report per proposal section F."""
    label = LABEL_NAMES[prediction]
    confidence = probability if prediction == 1 else (1 - probability)

    report = f"""
## Clinical Voice Analysis Report

### Classification Result
- **Prediction**: {label}
- **Confidence**: {confidence:.1%}

### Acoustic Parameters
| Parameter | Value | Healthy Mean | Pathological Mean | Status |
|-----------|-------|-------------|-------------------|--------|
"""
    clinical_features = [
        ("F0 (Hz)", "f0_mean"),
        ("F0 SD (Hz)", "f0_std"),
        ("Jitter Local", "jitter_local"),
        ("Jitter RAP", "jitter_rap"),
        ("Shimmer Local", "shimmer_local"),
        ("Shimmer APQ3", "shimmer_apq3"),
        ("HNR (dB)", "hnr_mean"),
        ("CPP (dB)", "cpp_mean"),
        ("F1 (Hz)", "f1_mean"),
        ("F2 (Hz)", "f2_mean"),
        ("F3 (Hz)", "f3_mean"),
    ]

    for display_name, feat_name in clinical_features:
        val = features.get(feat_name, np.nan)
        if np.isnan(val):
            continue
        h_mean = ref_df[ref_df["label"] == LABEL_HEALTHY][feat_name].mean()
        p_mean = ref_df[ref_df["label"] == LABEL_PATHOLOGICAL][feat_name].mean()

        # Determine if value is closer to healthy or pathological
        h_dist = abs(val - h_mean)
        p_dist = abs(val - p_mean)
        status = "Normal" if h_dist < p_dist else "Abnormal"
        icon = "+" if status == "Normal" else "!"

        report += f"| {display_name} | {val:.4f} | {h_mean:.4f} | {p_mean:.4f} | {icon} {status} |\n"

    report += """
### Notes
- Status is determined by proximity to healthy vs pathological population means.
- This is a screening tool and does not replace clinical diagnosis.
"""
    return report


def main():
    """Streamlit CAD application."""
    st.set_page_config(
        page_title="Voice Pathology CAD",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Voice Pathology CAD System")
    st.markdown(
        "Computer-Aided Diagnosis for voice pathology classification "
        "based on multidimensional acoustic feature analysis."
    )

    # Sidebar - model selection
    st.sidebar.header("Settings")
    model_names = {
        "Random Forest": "random_forest",
        "SVM": "svm",
        "MLP (Neural Network)": "mlp",
        "Logistic Regression": "logistic_regression",
    }
    selected_model = st.sidebar.selectbox("Classification Model", list(model_names.keys()))
    model_key = model_names[selected_model]

    # Load model and reference data
    model = load_model(model_key)
    ref_df = load_reference_stats()

    if model is None:
        st.error(f"Model '{model_key}' not found. Run main_svd.py first.")
        return
    if ref_df is None:
        st.error("Reference data not found. Run main_svd.py first.")
        return

    st.sidebar.success(f"Model loaded: {selected_model}")
    st.sidebar.info(f"Reference data: {len(ref_df)} samples")

    # Main tabs
    tab_analyze, tab_results, tab_vsa = st.tabs([
        "Analyze Voice", "Dataset Results", "VSA Analysis",
    ])

    with tab_analyze:
        st.header("Upload Voice Recording")
        st.markdown("Upload a sustained vowel /a/ recording (WAV format).")

        uploaded_file = st.file_uploader(
            "Choose audio file", type=["wav", "flac", "ogg"],
        )

        if uploaded_file is not None:
            # Read audio
            audio_data, sr = sf.read(io.BytesIO(uploaded_file.read()))
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]  # mono

            st.audio(uploaded_file, format="audio/wav")
            st.write(f"Duration: {len(audio_data)/sr:.2f}s, Sample rate: {sr} Hz")

            with st.spinner("Extracting acoustic features..."):
                features = extract_features_from_audio(audio_data, sr)

            if features is None:
                st.error("Feature extraction failed.")
                return

            # Classify
            feat_vector = pd.DataFrame([{
                k: features[k] for k in ALL_FEATURE_NAMES
            }])
            prediction = model.predict(feat_vector)[0]
            proba = model.predict_proba(feat_vector)[0]

            # Display result
            col1, col2 = st.columns(2)
            with col1:
                label = LABEL_NAMES[prediction]
                color = LABEL_COLORS[prediction]
                confidence = proba[1] if prediction == 1 else proba[0]
                st.markdown(
                    f"### Result: <span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{confidence:.1%}")

            with col2:
                st.markdown("### Feature Summary")
                summary = {
                    "F0": f"{features.get('f0_mean', 0):.1f} Hz",
                    "Jitter": f"{features.get('jitter_local', 0):.5f}",
                    "Shimmer": f"{features.get('shimmer_local', 0):.5f}",
                    "HNR": f"{features.get('hnr_mean', 0):.1f} dB",
                    "CPP": f"{features.get('cpp_mean', 0):.1f} dB",
                }
                for k, v in summary.items():
                    st.write(f"**{k}**: {v}")

            # Radar chart
            st.subheader("Multi-Parameter Voice Profile")
            radar_fig = plot_radar_chart(features, ref_df)
            if radar_fig:
                st.pyplot(radar_fig)
                plt.close(radar_fig)

            # Clinical report
            st.subheader("Clinical Report")
            report = generate_clinical_report(features, prediction, proba[1], ref_df)
            st.markdown(report)

    with tab_results:
        st.header("Dataset Results (SVD)")

        # Show summary
        summary_path = RESULTS_DIR / "svd" / "summary.txt"
        if summary_path.exists():
            st.code(summary_path.read_text(), language="text")

        # Show plots
        col1, col2 = st.columns(2)
        plot_dir = RESULTS_DIR / "svd"
        cm_path = plot_dir / "confusion_matrix.png"
        fi_path = plot_dir / "feature_importance.png"
        radar_path = plot_dir / "radar_chart.png"
        dist_path = plot_dir / "feature_distributions.png"

        if cm_path.exists():
            with col1:
                st.image(str(cm_path), caption="Confusion Matrix (RF)")
        if fi_path.exists():
            with col2:
                st.image(str(fi_path), caption="Feature Importance")
        if dist_path.exists():
            st.image(str(dist_path), caption="Feature Distributions")

    with tab_vsa:
        st.header("Vowel Space Analysis (VSA)")
        st.markdown(
            "Upload three vowel recordings (/a/, /i/, /u/) to compute "
            "Vowel Space Area and Vowel Articulation Index."
        )

        col_a, col_i, col_u = st.columns(3)
        with col_a:
            file_a = st.file_uploader("Vowel /a/", type=["wav"], key="vsa_a")
        with col_i:
            file_i = st.file_uploader("Vowel /i/", type=["wav"], key="vsa_i")
        with col_u:
            file_u = st.file_uploader("Vowel /u/", type=["wav"], key="vsa_u")

        if file_a and file_i and file_u:
            with st.spinner("Extracting formants from all vowels..."):
                from vowel_space import extract_formants_for_vowel, _triangle_area, _compute_vai

                audio_a, sr_a = sf.read(io.BytesIO(file_a.read()))
                audio_i, sr_i = sf.read(io.BytesIO(file_i.read()))
                audio_u, sr_u = sf.read(io.BytesIO(file_u.read()))

                if audio_a.ndim > 1:
                    audio_a = audio_a[:, 0]
                if audio_i.ndim > 1:
                    audio_i = audio_i[:, 0]
                if audio_u.ndim > 1:
                    audio_u = audio_u[:, 0]

                f1_a, f2_a = extract_formants_for_vowel(audio_a, sr_a)
                f1_i, f2_i = extract_formants_for_vowel(audio_i, sr_i)
                f1_u, f2_u = extract_formants_for_vowel(audio_u, sr_u)

            vsa = _triangle_area(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u)
            vai = _compute_vai(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("VSA (Vowel Space Area)", f"{vsa:.0f} Hz²")
                st.metric("VAI (Vowel Articulation Index)", f"{vai:.4f}")

                st.markdown("#### Formant Values")
                formant_data = pd.DataFrame({
                    "Vowel": ["/a/", "/i/", "/u/"],
                    "F1 (Hz)": [f1_a, f1_i, f1_u],
                    "F2 (Hz)": [f2_a, f2_i, f2_u],
                })
                st.dataframe(formant_data, hide_index=True)

            with col2:
                vsa_fig = plot_vsa_map(f1_a, f2_a, f1_i, f2_i, f1_u, f2_u)
                st.pyplot(vsa_fig)
                plt.close(vsa_fig)


if __name__ == "__main__":
    main()
