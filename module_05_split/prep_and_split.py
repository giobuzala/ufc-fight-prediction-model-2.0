"""
Data preparation and temporal train/val/test split for UFC fight prediction.
Module 05: Split only. Model training in module_06_model.

Reads: input/ufc_fights_fe.csv (local or from Azure when --storage azure/both).
- Drops fighter_1, fighter_2.
- Target: winner = 1 if fighter_1 won, 0 if fighter_2 won; drops draws/NC.
- Filters events from year 2000 onward.
- Imputes missing values (median numeric, most_frequent categorical).
- One-hot encodes: division, fighter1_stance, fighter2_stance.
- StandardScaler on all features.
- Temporal split: 70% train, 15% val, 15% test (by event_date ascending).

Writes: output/preprocessor.joblib, output/X_train.npz, etc., output/split_info.json
(local and, when --storage azure/both, same files to container; also copies to module_06 input).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_INPUT = _SCRIPT_DIR / "input" / "ufc_fights_fe.csv"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "output"
MODULE_06_INPUT = _PROJECT_ROOT / "module_06_model" / "input"
DEFAULT_BLOB_INPUT = "module_05_split/input/ufc_fights_fe.csv"
BLOB_OUTPUT_PREFIX = "module_05_split/output"
BLOB_MODULE_06_PREFIX = "module_06_model/input"

# Columns to drop before modeling
DROP_COLS = {"fighter_1", "fighter_2"}

# Outcome columns: use only winner for main model; keep finish_type/finish_technique for later
OUTCOME_COLS = ["finish_type", "finish_technique"]

# Categorical columns to one-hot encode
CAT_COLS = ["division", "fighter1_stance", "fighter2_stance"]

# Minimum year to include
MIN_YEAR = 2000

# Temporal split fractions: train, val, test
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

RANDOM_SEED = 42


def _parse_event_date(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _coerce_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Convert numeric columns; empty strings become NaN. Returns df."""
    out = df.copy()
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _randomize_fighter_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    UFC Stats lists winner as fighter_1, so we have 100% fighter_1 wins.
    For ~50% of rows, swap fighter_1<->fighter_2 and paired stats so target is balanced.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    swap = rng.random(len(df)) < 0.5

    out = df.copy()

    # Swap fighter_1 <-> fighter_2
    out.loc[swap, "fighter_1"], out.loc[swap, "fighter_2"] = (
        df.loc[swap, "fighter_2"].values,
        df.loc[swap, "fighter_1"].values,
    )

    # Swap fighter1_* <-> fighter2_* column pairs
    f1_cols = [c for c in df.columns if c.startswith("fighter1_")]
    for c1 in f1_cols:
        c2 = c1.replace("fighter1_", "fighter2_", 1)
        if c2 in df.columns:
            out.loc[swap, c1] = df.loc[swap, c2].values
            out.loc[swap, c2] = df.loc[swap, c1].values

    # Negate differential columns (diff = f1 - f2; after swap, diff = f2 - f1 = -diff)
    diff_cols = [c for c in df.columns if c.endswith("_differential")]
    for c in diff_cols:
        out.loc[swap, c] = pd.to_numeric(out.loc[swap, c], errors="coerce") * -1

    return out


def _load_input_df(local_path: Path, storage: str, blob_input_path: str) -> pd.DataFrame:
    """Load input from local or Azure (Parquet then CSV); fall back to local when storage is both."""
    storage = (storage or "local").strip().lower()
    blob_parquet = blob_input_path.replace(".csv", ".parquet") if blob_input_path.endswith(".csv") else blob_input_path + ".parquet"

    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import read_csv_from_azure, read_parquet_from_azure

        try:
            try:
                return read_parquet_from_azure(blob_parquet).astype(str).fillna("")
            except FileNotFoundError:
                return read_csv_from_azure(blob_input_path, dtype=str).fillna("")
        except FileNotFoundError:
            if storage == "azure":
                raise
            pass

    if storage in ("local", "both") and local_path.exists():
        return pd.read_csv(local_path)
    raise FileNotFoundError(f"Input not found: {local_path}")


def _filter_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter by year, randomize fighter order, create target, drop invalid rows."""
    df = df.copy()
    df["_event_dt"] = df["event_date"].apply(_parse_event_date)
    df = df[df["_event_dt"].notna()].copy()
    df["_year"] = df["_event_dt"].dt.year
    df = df[df["_year"] >= MIN_YEAR].copy()

    # Randomize fighter order so target is ~50/50 (UFC Stats lists winner first)
    df = _randomize_fighter_order(df)

    # Target: 1 if fighter_1 won, 0 if fighter_2 won
    if "fighter_1" not in df.columns or "fighter_2" not in df.columns:
        raise ValueError("fighter_1 and fighter_2 must be present to create target")
    won_f1 = (df["winner"].fillna("").str.strip() == df["fighter_1"].fillna("").str.strip())
    won_f2 = (df["winner"].fillna("").str.strip() == df["fighter_2"].fillna("").str.strip())
    valid = won_f1 | won_f2
    df = df[valid].copy()
    df["y"] = won_f1[valid].astype(int)

    return df


def get_feature_columns(df: pd.DataFrame, differential_only: bool = False) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, cat_cols) for modeling. Excludes ids, target, outcome."""
    exclude = {"event_date", "winner", "y", "_event_dt", "_year", "fighter_1", "fighter_2"}
    exclude.update(OUTCOME_COLS)
    exclude.update(DROP_COLS)

    all_cols = [c for c in df.columns if c not in exclude]
    cat = [c for c in CAT_COLS if c in all_cols]
    numeric = [c for c in all_cols if c not in cat]

    if differential_only:
        # For logistic regression: use only differentials (+ fight-level); drop fighter1/fighter2 absolutes
        numeric = [
            c for c in numeric
            if c.endswith("_differential") or c in ("weight_class", "number_of_rounds")
        ]

    return numeric, cat


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build ColumnTransformer: impute -> encode (cat) -> StandardScaler on combined output."""
    from sklearn.pipeline import Pipeline

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
                    ("scale", StandardScaler()),
                ]),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-d", "--differential-only", action="store_true", help="Use only differential features for numerics (better for logistic regression)")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to read input and write output. local = disk only, azure = blob only, both = disk + blob.",
    )
    parser.add_argument("--blob-input", default=DEFAULT_BLOB_INPUT, help="Blob path for input (when reading from Azure).")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = (args.storage or "local").strip().lower()

    df = _filter_and_prepare(_load_input_df(args.input, storage, args.blob_input))
    df = df.sort_values("_event_dt").reset_index(drop=True)
    n = len(df)

    # Temporal split
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))

    numeric_cols, cat_cols = get_feature_columns(df, differential_only=args.differential_only)
    df = _coerce_numeric(df, numeric_cols)
    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()

    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df["y"].values
    X_val = val_df[numeric_cols + cat_cols]
    y_val = val_df["y"].values
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df["y"].values

    pre = build_preprocessor(numeric_cols, cat_cols)
    X_train_arr = pre.fit_transform(X_train)
    X_val_arr = pre.transform(X_val)
    X_test_arr = pre.transform(X_test)

    # Save to module_05 output and module_06 input
    suf = "_diff" if args.differential_only else ""
    output_paths = [out_dir, MODULE_06_INPUT]
    for save_dir in output_paths:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"preprocessor": pre, "numeric_cols": numeric_cols, "cat_cols": cat_cols},
            save_dir / f"preprocessor{suf}.joblib",
        )
        np.savez_compressed(save_dir / f"X_train{suf}.npz", X=X_train_arr)
        np.savez_compressed(save_dir / f"y_train{suf}.npz", y=y_train)
        np.savez_compressed(save_dir / f"X_val{suf}.npz", X=X_val_arr)
        np.savez_compressed(save_dir / f"y_val{suf}.npz", y=y_val)
        np.savez_compressed(save_dir / f"X_test{suf}.npz", X=X_test_arr)
        np.savez_compressed(save_dir / f"y_test{suf}.npz", y=y_test)

    split_info = {
        "n_total": n,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_date_range": [
            train_df["_event_dt"].min().isoformat() if len(train_df) else None,
            train_df["_event_dt"].max().isoformat() if len(train_df) else None,
        ],
        "val_date_range": [
            val_df["_event_dt"].min().isoformat() if len(val_df) else None,
            val_df["_event_dt"].max().isoformat() if len(val_df) else None,
        ],
        "test_date_range": [
            test_df["_event_dt"].min().isoformat() if len(test_df) else None,
            test_df["_event_dt"].max().isoformat() if len(test_df) else None,
        ],
        "n_features": X_train_arr.shape[1],
    }
    for save_dir in output_paths:
        with open(save_dir / f"split_info{suf}.json", "w") as f:
            json.dump(split_info, f, indent=2)

    # Upload to Azure when storage is azure/both (same .joblib, .npz, .json to container)
    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import upload_file_to_azure

        file_names = [
            f"preprocessor{suf}.joblib",
            f"X_train{suf}.npz", f"y_train{suf}.npz",
            f"X_val{suf}.npz", f"y_val{suf}.npz",
            f"X_test{suf}.npz", f"y_test{suf}.npz",
            f"split_info{suf}.json",
        ]
        for fname in file_names:
            upload_file_to_azure(str(out_dir / fname), f"{BLOB_OUTPUT_PREFIX}/{fname}")
            upload_file_to_azure(str(MODULE_06_INPUT / fname), f"{BLOB_MODULE_06_PREFIX}/{fname}")
        print(f"  Uploaded to Azure: {BLOB_OUTPUT_PREFIX}/ and {BLOB_MODULE_06_PREFIX}/")

    suffix = " (differential-only)" if args.differential_only else ""
    print(f"Prepared {n} fights (from {MIN_YEAR}+){suffix}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Features: {X_train_arr.shape[1]}")
    print(f"  Saved to {out_dir} and {MODULE_06_INPUT}")


if __name__ == "__main__":
    main()
