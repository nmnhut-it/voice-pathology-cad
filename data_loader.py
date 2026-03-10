"""
Data loader for the VOICED (PhysioNet) database.
Downloads records via wfdb, parses metadata, and returns audio signals with labels.
"""
import re
import numpy as np
import pandas as pd
import wfdb

from config import (
    VOICED_PN_DIR,
    VOICED_TOTAL_RECORDS,
    VOICED_RECORD_PREFIX,
    LABEL_HEALTHY,
    LABEL_PATHOLOGICAL,
    HEALTHY_DIAGNOSIS,
)


def _parse_record_comments(comments: list[str]) -> dict:
    """Extract age, sex, diagnosis from WFDB record comments."""
    metadata = {}
    comment_text = " ".join(comments)
    for field in ["age", "sex", "diagnoses", "medications"]:
        match = re.search(rf"<{field}>:\s*([^<]+)", comment_text)
        metadata[field] = match.group(1).strip() if match else None
    return metadata


def _assign_binary_label(diagnosis: str | None) -> int:
    """Map diagnosis string to binary label (healthy=0, pathological=1)."""
    if diagnosis and diagnosis.lower().strip() == HEALTHY_DIAGNOSIS:
        return LABEL_HEALTHY
    return LABEL_PATHOLOGICAL


def load_voiced_dataset() -> pd.DataFrame:
    """
    Download and load all VOICED records from PhysioNet.
    Returns DataFrame with columns: record_id, signal (numpy array),
    sample_rate, age, sex, diagnosis, label.
    """
    records = []
    for i in range(1, VOICED_TOTAL_RECORDS + 1):
        record_name = f"{VOICED_RECORD_PREFIX}{i:03d}"
        try:
            rec = wfdb.rdrecord(record_name, pn_dir=VOICED_PN_DIR)
            metadata = _parse_record_comments(rec.comments)
            signal = rec.p_signal[:, 0]  # single channel

            records.append({
                "record_id": record_name,
                "signal": signal,
                "sample_rate": rec.fs,
                "age": int(metadata["age"]) if metadata.get("age") else None,
                "sex": metadata.get("sex"),
                "diagnosis": metadata.get("diagnoses"),
                "label": _assign_binary_label(metadata.get("diagnoses")),
            })
            if i % 20 == 0 or i == VOICED_TOTAL_RECORDS:
                print(f"  Downloaded {i}/{VOICED_TOTAL_RECORDS} records...")
        except Exception as e:
            print(f"Warning: skipping {record_name}: {e}")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records: "
          f"{(df['label'] == LABEL_HEALTHY).sum()} healthy, "
          f"{(df['label'] == LABEL_PATHOLOGICAL).sum()} pathological")
    return df


if __name__ == "__main__":
    df = load_voiced_dataset()
    print(df[["record_id", "diagnosis", "label", "age", "sex"]].head(10))
    print("\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts())
