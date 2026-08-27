"""
Load the raw Kepler KOI table and produce a clean, model-ready dataset.

Run: python 01_load_clean.py
Output: data/koi_clean.csv
"""
import pandas as pd

RAW_PATH = "data/cumulative.csv"
OUT_PATH = "data/koi_clean.csv"

# Columns we care about: label + physically meaningful predictors
FEATURE_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad",
]
LABEL_COL = "koi_disposition"


def load_raw(path: str) -> pd.DataFrame:
    # NASA archive CSVs have comment lines starting with '#'
    df = pd.read_csv(path, comment="#")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[[LABEL_COL] + FEATURE_COLS].copy()

    # Drop CANDIDATE rows for a clean binary problem (confirmed vs false positive).
    # (Optional: keep them and predict on them later as "unlabeled" cases.)
    keep = keep[keep[LABEL_COL].isin(["CONFIRMED", "FALSE POSITIVE"])]

    before = len(keep)
    keep = keep.dropna()
    print(f"Dropped {before - len(keep)} rows with missing values "
          f"({len(keep)} remaining)")

    keep["label"] = (keep[LABEL_COL] == "CONFIRMED").astype(int)

    print("\nClass balance:")
    print(keep["label"].value_counts(normalize=True).round(3))

    return keep


if __name__ == "__main__":
    raw = load_raw(RAW_PATH)
    clean_df = clean(raw)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned data to {OUT_PATH}")
