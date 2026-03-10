"""
Data loader for the Saarbrücken Voice Database (SVD).
Reads NSP audio files from zip archives, parses metadata.
SVD has ~2275 recordings with vowels /a/, /i/, /u/ at 50kHz.
"""
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SVD_DIR,
    LABEL_HEALTHY,
    LABEL_PATHOLOGICAL,
)

SVD_SAMPLE_RATE = 50000
SVD_ZIPS_DIR = Path("D:/svd-data/zips")
SVD_METADATA_PATH = SVD_DIR / "combined_metadata.csv"

# Vowel file suffixes: normal pitch only for baseline analysis
VOWEL_SUFFIXES = {"a": "-a_n.nsp", "i": "-i_n.nsp", "u": "-u_n.nsp"}

# AufnahmeTyp mapping
HEALTHY_TYPE = "n"


def _parse_nsp_audio(data: bytes) -> np.ndarray | None:
    """
    Parse FORM/DS16 NSP file format to numpy array.
    SVD uses IFF-like container with 16-bit little-endian PCM at 50kHz.
    Despite IFF's big-endian convention, SVD files store all numeric
    fields (chunk sizes, audio samples) in little-endian byte order.
    """
    sda_pos = data.find(b"SDA_")
    if sda_pos < 0:
        return None
    sda_size = struct.unpack("<I", data[sda_pos + 4 : sda_pos + 8])[0]
    audio_bytes = data[sda_pos + 8 : sda_pos + 8 + sda_size]
    samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float64)
    return samples / 32768.0


def _extract_vowels_from_zip(
    zip_path: Path, record_ids: list[int]
) -> dict[int, dict[str, np.ndarray]]:
    """
    Extract normal-pitch vowel signals for given record IDs from one zip.
    Returns {record_id: {"a": signal, "i": signal, "u": signal}}.
    """
    results = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            for rid in record_ids:
                vowels = {}
                for vowel, suffix in VOWEL_SUFFIXES.items():
                    filepath = f"{rid}/vowels/{rid}{suffix}"
                    if filepath in names:
                        with zf.open(filepath) as f:
                            signal = _parse_nsp_audio(f.read())
                            if signal is not None and len(signal) > 0:
                                vowels[vowel] = signal
                if vowels:
                    results[rid] = vowels
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  Warning: cannot read {zip_path.name}: {e}")
    return results


def load_svd_metadata() -> pd.DataFrame:
    """Load and enrich SVD metadata with binary labels."""
    df = pd.read_csv(SVD_METADATA_PATH)
    df["label"] = df["AufnahmeTyp"].apply(
        lambda x: LABEL_HEALTHY if x == HEALTHY_TYPE else LABEL_PATHOLOGICAL
    )
    return df


def load_svd_dataset(vowel: str = "a") -> pd.DataFrame:
    """
    Load SVD audio for a single vowel from all zip archives.
    Returns DataFrame with: record_id, signal, sample_rate, label, sex, pathology.
    """
    metadata = load_svd_metadata()
    print(f"SVD metadata: {len(metadata)} records "
          f"({(metadata['label'] == LABEL_HEALTHY).sum()} healthy, "
          f"{(metadata['label'] == LABEL_PATHOLOGICAL).sum()} pathological)")

    # Group records by pathology (= zip file name)
    # Healthy records are in healthy.zip
    zip_files = list(SVD_ZIPS_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip archives")

    all_records = []
    for zip_path in zip_files:
        pathology_name = zip_path.stem
        if pathology_name == "healthy":
            records_in_zip = metadata[metadata["AufnahmeTyp"] == HEALTHY_TYPE]
        else:
            records_in_zip = metadata[metadata["Pathologien"] == pathology_name]

        if records_in_zip.empty:
            continue

        record_ids = records_in_zip["AufnahmeID"].tolist()
        vowel_data = _extract_vowels_from_zip(zip_path, record_ids)

        for rid, vowels in vowel_data.items():
            if vowel not in vowels:
                continue
            row = metadata[metadata["AufnahmeID"] == rid].iloc[0]
            all_records.append({
                "record_id": rid,
                "signal": vowels[vowel],
                "sample_rate": SVD_SAMPLE_RATE,
                "label": row["label"],
                "sex": row["Geschlecht"],
                "pathology": row.get("Pathologien", ""),
            })

        loaded = len(all_records)
        if loaded % 200 < len(vowel_data):
            print(f"  Loaded {loaded} records so far "
                  f"(from {pathology_name})...")

    df = pd.DataFrame(all_records)
    n_healthy = (df["label"] == LABEL_HEALTHY).sum()
    n_pathological = (df["label"] == LABEL_PATHOLOGICAL).sum()
    print(f"\nLoaded {len(df)} records with vowel /{vowel}/: "
          f"{n_healthy} healthy, {n_pathological} pathological")
    return df


def load_svd_multivowel() -> pd.DataFrame:
    """
    Load SVD with all three vowels /a/, /i/, /u/ per record.
    Returns DataFrame with: record_id, signal_a, signal_i, signal_u,
    sample_rate, label, sex, pathology.
    Needed for VSA/VAI computation.
    """
    metadata = load_svd_metadata()
    zip_files = list(SVD_ZIPS_DIR.glob("*.zip"))
    print(f"Loading multi-vowel data from {len(zip_files)} zip archives...")

    all_records = []
    for zip_path in zip_files:
        pathology_name = zip_path.stem
        if pathology_name == "healthy":
            records_in_zip = metadata[metadata["AufnahmeTyp"] == HEALTHY_TYPE]
        else:
            records_in_zip = metadata[metadata["Pathologien"] == pathology_name]

        if records_in_zip.empty:
            continue

        record_ids = records_in_zip["AufnahmeID"].tolist()
        vowel_data = _extract_vowels_from_zip(zip_path, record_ids)

        for rid, vowels in vowel_data.items():
            # Require all three vowels
            if not all(v in vowels for v in ["a", "i", "u"]):
                continue
            row = metadata[metadata["AufnahmeID"] == rid].iloc[0]
            all_records.append({
                "record_id": rid,
                "signal_a": vowels["a"],
                "signal_i": vowels["i"],
                "signal_u": vowels["u"],
                "sample_rate": SVD_SAMPLE_RATE,
                "label": row["label"],
                "sex": row["Geschlecht"],
                "pathology": row.get("Pathologien", ""),
            })

        loaded = len(all_records)
        if loaded % 200 < len(vowel_data):
            print(f"  Loaded {loaded} records (from {pathology_name})...")

    df = pd.DataFrame(all_records)
    n_h = (df["label"] == LABEL_HEALTHY).sum()
    n_p = (df["label"] == LABEL_PATHOLOGICAL).sum()
    print(f"\nLoaded {len(df)} records with all 3 vowels: "
          f"{n_h} healthy, {n_p} pathological")
    return df


if __name__ == "__main__":
    df = load_svd_dataset("a")
    print(df[["record_id", "label", "sex", "pathology"]].head(10))
