"""
Clean raw UFC fighter data for training.

Reads: input/raw_ufc_fighters.csv (local or from Azure when --storage azure/both).
- Normalizes height to inches (e.g. 5' 10" -> 70), weight to numeric lbs, reach to inches.
Writes: output/clean_ufc_fighters.csv (local) and optionally CSV+Parquet to Azure (see --storage).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_INPUT_PATH = _SCRIPT_DIR / "input" / "raw_ufc_fighters.csv"
DEFAULT_OUTPUT_PATH = _SCRIPT_DIR / "output" / "clean_ufc_fighters.csv"
DEFAULT_BLOB_INPUT_PATH = "module_02_clean_fighters/input/raw_ufc_fighters.csv"
DEFAULT_BLOB_CSV_PATH = "module_02_clean_fighters/output/clean_ufc_fighters.csv"
DEFAULT_BLOB_PARQUET_PATH = "module_02_clean_fighters/output/clean_ufc_fighters.parquet"
# Downstream module input (so module_03 can read from local/container).
MODULE_03_INPUT_BLOB_CSV = "module_03_clean_fights/input/clean_ufc_fighters.csv"
MODULE_03_INPUT_BLOB_PARQUET = "module_03_clean_fights/input/clean_ufc_fighters.parquet"

FIELDNAMES = ["fighter_url", "full_name", "height", "weight", "reach", "stance", "date_of_birth"]


def _parse_height_inches(value: str) -> int | str:
    """Convert e.g. 5' 10" to total inches; return '' if unparseable."""
    value = (value or "").strip()
    m = re.match(r"^(\d+)\s*'\s*(\d+)\s*\"?\s*$", value)
    if not m:
        return ""
    feet, inches = int(m.group(1)), int(m.group(2))
    return feet * 12 + inches


def _parse_weight_numeric(value: str) -> int | str:
    """Parse '155 lbs' to int; return '' if unparseable."""
    value = (value or "").strip()
    m = re.search(r"^(\d+)\s*lbs\.?\s*$", value, re.IGNORECASE)
    if not m:
        return ""
    return int(m.group(1))


def _parse_reach_inches(value: str) -> int | str:
    """Parse '70"' to int inches; return '' if unparseable."""
    value = (value or "").strip()
    m = re.match(r"^(\d+)\s*\"?\s*$", value)
    if not m:
        return ""
    return int(m.group(1))


def clean_row(row: dict) -> dict:
    out = dict(row)
    out["height"] = _parse_height_inches(row.get("height") or "")
    out["weight"] = _parse_weight_numeric(row.get("weight") or "")
    out["reach"] = _parse_reach_inches(row.get("reach") or "")
    return out


# Output helpers


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in FIELDNAMES:
        if c not in df.columns:
            df[c] = ""
    df = df[FIELDNAMES].fillna("")
    # Parquet requires consistent types; height/weight/reach can be int or "" from clean_row.
    return df.astype(str)


def _load_input(
    local_path: Path,
    storage: str,
    blob_input_path: str,
) -> list[dict]:
    """Load input: when storage is both or azure, use cloud (Parquet then CSV); only then fall back to local."""
    storage = (storage or "local").strip().lower()
    blob_input_parquet = blob_input_path.replace(".csv", ".parquet") if blob_input_path.endswith(".csv") else blob_input_path + ".parquet"

    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import read_csv_from_azure, read_parquet_from_azure

        try:
            try:
                df = read_parquet_from_azure(blob_input_parquet).astype(str).fillna("")
            except FileNotFoundError:
                df = read_csv_from_azure(blob_input_path, dtype=str).fillna("")
            for c in FIELDNAMES:
                if c not in df.columns:
                    df[c] = ""
            return df[FIELDNAMES].fillna("").to_dict(orient="records")
        except FileNotFoundError:
            if storage == "local":
                raise
            # storage == "both": fall back to local
            pass

    if storage in ("local", "both") and local_path.exists():
        with open(local_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for r in rows:
                for c in FIELDNAMES:
                    if c not in r:
                        r[c] = ""
            return rows
    raise FileNotFoundError(f"Input not found: {local_path}")


def _write_outputs(
    *,
    records: list[dict],
    local_csv_path: Path,
    storage: str,
    blob_csv_path: str,
    blob_parquet_path: str,
) -> None:
    df = _records_to_df(records)

    if storage in ("local", "both"):
        local_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_csv_path, index=False)
        print(f"Wrote {len(df)} rows to {local_csv_path}")
        # Keep module_03 input in sync.
        module_03_input = _PROJECT_ROOT / "module_03_clean_fights" / "input" / "clean_ufc_fighters.csv"
        module_03_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(module_03_input, index=False)
        print(f"Updated module_03 input: {module_03_input}")

    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import write_csv_to_azure, write_parquet_to_azure

            write_csv_to_azure(df, blob_csv_path, index=False)
            write_parquet_to_azure(df, blob_parquet_path, index=False)
            print(f"Wrote {len(df)} rows to Azure blobs: {blob_csv_path} and {blob_parquet_path}")
            write_csv_to_azure(df, MODULE_03_INPUT_BLOB_CSV, index=False)
            write_parquet_to_azure(df, MODULE_03_INPUT_BLOB_PARQUET, index=False)
            print(f"Updated module_03 input in container: {MODULE_03_INPUT_BLOB_CSV} and {MODULE_03_INPUT_BLOB_PARQUET}")
        except Exception as e:
            print(f"Failed to write to Azure: {e}")
            raise


def clean_fighters(
    input_path: str | Path,
    output_path: str | Path,
    storage: str = "local",
    blob_input_path: str = DEFAULT_BLOB_INPUT_PATH,
    blob_csv_path: str = DEFAULT_BLOB_CSV_PATH,
    blob_parquet_path: str = DEFAULT_BLOB_PARQUET_PATH,
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    storage = (storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    rows = _load_input(input_path, storage, blob_input_path)
    cleaned = [clean_row(r) for r in rows]

    _write_outputs(
        records=cleaned,
        local_csv_path=output_path,
        storage=storage,
        blob_csv_path=blob_csv_path,
        blob_parquet_path=blob_parquet_path,
    )
    return len(cleaned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw UFC fighters: height/reach to inches, weight to numeric.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_PATH, help="Input CSV (default: <script_dir>/input/raw_ufc_fighters.csv)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV (default: <script_dir>/output/clean_ufc_fighters.csv)")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to read input and write output. local = disk only, azure = blob only, both = disk + blob.",
    )
    parser.add_argument("--blob-input", default=DEFAULT_BLOB_INPUT_PATH, help="Blob path for input CSV (when reading from Azure).")
    parser.add_argument("--blob-csv", default=DEFAULT_BLOB_CSV_PATH, help="Blob path for output CSV.")
    parser.add_argument("--blob-parquet", default=DEFAULT_BLOB_PARQUET_PATH, help="Blob path for output Parquet.")
    args = parser.parse_args()

    n = clean_fighters(
        args.input,
        args.output,
        storage=args.storage,
        blob_input_path=args.blob_input,
        blob_csv_path=args.blob_csv,
        blob_parquet_path=args.blob_parquet,
    )
    print(f"Cleaned {n} rows.")
