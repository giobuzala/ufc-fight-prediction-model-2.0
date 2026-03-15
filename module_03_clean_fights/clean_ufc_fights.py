"""
Clean raw UFC fight data for training.

Reads: input/raw_ufc_fights.csv, input/clean_ufc_fighters.csv (local or from Azure when --storage azure/both).
- Joins fighter attributes (DOB, age, height, reach, stance) by fighter_1_url/fighter_2_url, fallback by name.
- Adds division (Men/Women), normalizes weight_class, splits method into finish_type + finish_technique.
- Converts time/ctrl to seconds; expands "X of Y" strike columns to landed/attempted/pct.
Writes: output/clean_ufc_fights.csv (local) and optionally CSV+Parquet to Azure (see --storage).
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_INPUT_PATH = _SCRIPT_DIR / "input" / "raw_ufc_fights.csv"
DEFAULT_OUTPUT_PATH = _SCRIPT_DIR / "output" / "clean_ufc_fights.csv"
DEFAULT_FIGHTERS_PATH = _SCRIPT_DIR / "input" / "clean_ufc_fighters.csv"
DEFAULT_BLOB_INPUT_FIGHTS = "module_03_clean_fights/input/raw_ufc_fights.csv"
DEFAULT_BLOB_INPUT_FIGHTERS = "module_03_clean_fights/input/clean_ufc_fighters.csv"
DEFAULT_BLOB_OUTPUT_CSV = "module_03_clean_fights/output/clean_ufc_fights.csv"
DEFAULT_BLOB_OUTPUT_PARQUET = "module_03_clean_fights/output/clean_ufc_fights.parquet"
# Downstream module input (so module_04 can read from local/container).
MODULE_04_INPUT_BLOB_CSV = "module_04_feature_engineering/input/clean_ufc_fights.csv"
MODULE_04_INPUT_BLOB_PARQUET = "module_04_feature_engineering/input/clean_ufc_fights.parquet"

# Columns in "X of Y" format to expand into _landed, _attempted, _pct
OF_Y_COLUMNS = [
    "fighter1_sig_str", "fighter2_sig_str",
    "fighter1_total_str", "fighter2_total_str",
    "fighter1_td", "fighter2_td",
    "fighter1_head_sig_str", "fighter2_head_sig_str",
    "fighter1_body_sig_str", "fighter2_body_sig_str",
    "fighter1_leg_sig_str", "fighter2_leg_sig_str",
    "fighter1_distance_sig_str", "fighter2_distance_sig_str",
    "fighter1_clinch_sig_str", "fighter2_clinch_sig_str",
    "fighter1_ground_sig_str", "fighter2_ground_sig_str",
]

# Sig strike breakdown (raw) -> output base name for _sig_str_landed, _sig_str_attempted, _sig_str_pct
SIG_STR_BREAKDOWN_RAW_TO_BASE = {
    "fighter1_head_sig_str": "fighter1_head", "fighter2_head_sig_str": "fighter2_head",
    "fighter1_body_sig_str": "fighter1_body", "fighter2_body_sig_str": "fighter2_body",
    "fighter1_leg_sig_str": "fighter1_leg", "fighter2_leg_sig_str": "fighter2_leg",
    "fighter1_distance_sig_str": "fighter1_distance", "fighter2_distance_sig_str": "fighter2_distance",
    "fighter1_clinch_sig_str": "fighter1_clinch", "fighter2_clinch_sig_str": "fighter2_clinch",
    "fighter1_ground_sig_str": "fighter1_ground", "fighter2_ground_sig_str": "fighter2_ground",
}
# Fallback for legacy CSVs where raw column names may differ (same keys as above)
SIG_STR_BREAKDOWN_OLD_RAW = dict(SIG_STR_BREAKDOWN_RAW_TO_BASE)
STRIKE_OF_Y_COLUMNS = set(SIG_STR_BREAKDOWN_RAW_TO_BASE)

CLEAN_COLUMNS = [
    "event_url", "event_name", "event_date", "event_location",
    "division",
    "fighter_1", "fighter_2", "fighter_1_url", "fighter_2_url",
    "fighter1_dob", "fighter1_age", "fighter2_dob", "fighter2_age",
    "fighter1_height", "fighter2_height", "fighter1_reach", "fighter2_reach",
    "fighter1_stance", "fighter2_stance",
    "winner",
    "weight_class", "finish_type", "finish_technique", "round", "total_fight_time_seconds", "number_of_rounds",
    "fighter1_kd", "fighter2_kd",
    "fighter1_sig_str_landed", "fighter1_sig_str_attempted", "fighter1_sig_str_pct",
    "fighter2_sig_str_landed", "fighter2_sig_str_attempted", "fighter2_sig_str_pct",
    "fighter1_total_str_landed", "fighter1_total_str_attempted", "fighter1_total_str_pct",
    "fighter2_total_str_landed", "fighter2_total_str_attempted", "fighter2_total_str_pct",
    "fighter1_td_landed", "fighter1_td_attempted", "fighter1_td_pct",
    "fighter2_td_landed", "fighter2_td_attempted", "fighter2_td_pct",
    "fighter1_sub_att", "fighter2_sub_att",
    "fighter1_rev", "fighter2_rev",
    "fighter1_ctrl_seconds", "fighter2_ctrl_seconds",
    "fighter1_head_sig_str_landed", "fighter1_head_sig_str_attempted", "fighter1_head_sig_str_pct",
    "fighter2_head_sig_str_landed", "fighter2_head_sig_str_attempted", "fighter2_head_sig_str_pct",
    "fighter1_body_sig_str_landed", "fighter1_body_sig_str_attempted", "fighter1_body_sig_str_pct",
    "fighter2_body_sig_str_landed", "fighter2_body_sig_str_attempted", "fighter2_body_sig_str_pct",
    "fighter1_leg_sig_str_landed", "fighter1_leg_sig_str_attempted", "fighter1_leg_sig_str_pct",
    "fighter2_leg_sig_str_landed", "fighter2_leg_sig_str_attempted", "fighter2_leg_sig_str_pct",
    "fighter1_distance_sig_str_landed", "fighter1_distance_sig_str_attempted", "fighter1_distance_sig_str_pct",
    "fighter2_distance_sig_str_landed", "fighter2_distance_sig_str_attempted", "fighter2_distance_sig_str_pct",
    "fighter1_clinch_sig_str_landed", "fighter1_clinch_sig_str_attempted", "fighter1_clinch_sig_str_pct",
    "fighter2_clinch_sig_str_landed", "fighter2_clinch_sig_str_attempted", "fighter2_clinch_sig_str_pct",
    "fighter1_ground_sig_str_landed", "fighter1_ground_sig_str_attempted", "fighter1_ground_sig_str_pct",
    "fighter2_ground_sig_str_landed", "fighter2_ground_sig_str_attempted", "fighter2_ground_sig_str_pct",
]


def _load_input_generic(
    local_path: Path,
    storage: str,
    blob_path: str,
) -> list[dict]:
    """Load rows from local or Azure. When storage is both/azure, use cloud (Parquet then CSV) first."""
    storage = (storage or "local").strip().lower()
    blob_parquet = blob_path.replace(".csv", ".parquet") if blob_path.endswith(".csv") else blob_path + ".parquet"

    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import read_csv_from_azure, read_parquet_from_azure

        try:
            try:
                df = read_parquet_from_azure(blob_parquet).astype(str).fillna("")
            except FileNotFoundError:
                df = read_csv_from_azure(blob_path, dtype=str).fillna("")
            return df.to_dict(orient="records")
        except FileNotFoundError:
            if storage == "azure":
                raise
            pass

    if storage in ("local", "both") and local_path.exists():
        with open(local_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise FileNotFoundError(f"Input not found: {local_path}")


def _fighter_rows_to_lookups(rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Build (by_url, by_name) from fighter rows (same logic as load_fighters)."""
    by_url: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for row in rows:
        url = (row.get("fighter_url") or "").strip()
        name = (row.get("full_name") or "").strip()
        attrs = {
            "date_of_birth": (row.get("date_of_birth") or "").strip(),
            "height": (row.get("height") or "").strip(),
            "reach": (row.get("reach") or "").strip(),
            "stance": (row.get("stance") or "").strip(),
        }
        if url:
            by_url[url] = attrs
        if name and name not in by_name:
            by_name[name] = attrs
    return by_url, by_name


def _records_to_df(records: list[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[columns].fillna("")
    return df.astype(str)


def _write_outputs(
    *,
    records: list[dict],
    columns: list[str],
    local_csv_path: Path,
    storage: str,
    blob_csv_path: str,
    blob_parquet_path: str,
) -> None:
    df = _records_to_df(records, columns)

    if storage in ("local", "both"):
        local_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_csv_path, index=False)
        print(f"Wrote {len(df)} rows to {local_csv_path}")
        # Keep module_04 input in sync.
        module_04_input = _PROJECT_ROOT / "module_04_feature_engineering" / "input" / "clean_ufc_fights.csv"
        module_04_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(module_04_input, index=False)
        print(f"Updated module_04 input: {module_04_input}")

    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import write_csv_to_azure, write_parquet_to_azure

            write_csv_to_azure(df, blob_csv_path, index=False)
            write_parquet_to_azure(df, blob_parquet_path, index=False)
            print(f"Wrote {len(df)} rows to Azure blobs: {blob_csv_path} and {blob_parquet_path}")
            write_csv_to_azure(df, MODULE_04_INPUT_BLOB_CSV, index=False)
            write_parquet_to_azure(df, MODULE_04_INPUT_BLOB_PARQUET, index=False)
            print(f"Updated module_04 input in container: {MODULE_04_INPUT_BLOB_CSV} and {MODULE_04_INPUT_BLOB_PARQUET}")
        except Exception as e:
            print(f"Failed to write to Azure: {e}")
            raise


def _parse_of_y(value: str) -> tuple[int, int, float | str]:
    """Parse 'X of Y' format; return (landed, attempted, pct)."""
    value = (value or "").strip()
    m = re.search(r"(\d+)\s+of\s+(\d+)", value)
    if not m:
        return 0, 0, ""
    landed, attempted = int(m.group(1)), int(m.group(2))
    pct = round(landed / attempted, 4) if attempted else ""
    return landed, attempted, pct


def _parse_time_to_seconds(value: str) -> int | str:
    """Parse M:SS to seconds. Used for single duration (e.g. ctrl time)."""
    value = (value or "").strip()
    m = re.match(r"^(\d+):(\d{1,2})$", value)
    if not m:
        return ""
    return int(m.group(1)) * 60 + int(m.group(2))


def _fight_time_to_seconds(round_val: str, time_val: str, number_of_rounds_val: str) -> int | str:
    """Compute total fight duration in seconds.
    Raw 'time' is time WITHIN the ending round (e.g. '5:00' in round 3 = end of round 3).
    Total = (round - 1) * 300 + time_in_round_seconds.
    For DEC with round/time, we use this. If round/time missing, use number_of_rounds * 300."""
    time_sec = _parse_time_to_seconds(time_val)
    if time_sec == "":
        nr = (number_of_rounds_val or "").strip()
        if nr:
            try:
                return int(nr) * 300  # scheduled distance
            except (ValueError, TypeError):
                pass
        return ""
    round_num = (round_val or "").strip()
    if not round_num:
        return time_sec  # only round 1, or unknown
    try:
        r = int(round_num)
        return (r - 1) * 300 + int(time_sec)
    except (ValueError, TypeError):
        return time_sec


def _parse_event_date(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _parse_dob(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _age_at_event(event_date: datetime | None, dob: datetime | None) -> int | str:
    if event_date is None or dob is None or dob > event_date:
        return ""
    return (event_date - dob).days // 365


def clean_row(row: dict, fighters_by_url: dict[str, dict], fighters_by_name: dict[str, dict]) -> dict:
    out = dict(row)
    event_dt = _parse_event_date(row.get("event_date") or "")
    f1_name = (row.get("fighter_1") or "").strip()
    f2_name = (row.get("fighter_2") or "").strip()
    f1_url = (row.get("fighter_1_url") or "").strip()
    f2_url = (row.get("fighter_2_url") or "").strip()
    out["fighter_1_url"] = f1_url
    out["fighter_2_url"] = f2_url
    f1 = fighters_by_url.get(f1_url, {}) if f1_url else fighters_by_name.get(f1_name, {})
    f2 = fighters_by_url.get(f2_url, {}) if f2_url else fighters_by_name.get(f2_name, {})
    out["fighter1_dob"] = f1.get("date_of_birth", "")
    out["fighter2_dob"] = f2.get("date_of_birth", "")
    out["fighter1_age"] = _age_at_event(event_dt, _parse_dob(f1.get("date_of_birth", "")))
    out["fighter2_age"] = _age_at_event(event_dt, _parse_dob(f2.get("date_of_birth", "")))
    out["fighter1_height"] = f1.get("height", "")
    out["fighter2_height"] = f2.get("height", "")
    out["fighter1_reach"] = f1.get("reach", "")
    out["fighter2_reach"] = f2.get("reach", "")
    out["fighter1_stance"] = f1.get("stance", "")
    out["fighter2_stance"] = f2.get("stance", "")

    weight = (row.get("weight_class") or "").strip()
    if weight.startswith("Women's "):
        division = "Women"
        weight_class_clean = weight[len("Women's ") :].strip()
    else:
        division = "Men"
        weight_class_clean = weight

    method = (row.get("method") or "").strip()
    if " " in method:
        finish_type, finish_technique = method.split(" ", 1)
    else:
        finish_type, finish_technique = method, ""

    out["division"] = division
    out["weight_class"] = weight_class_clean
    out["finish_type"] = finish_type
    out["finish_technique"] = finish_technique
    out["total_fight_time_seconds"] = _fight_time_to_seconds(
        row.get("round") or "",
        row.get("time") or "",
        row.get("number_of_rounds") or "",
    )
    out["fighter1_ctrl_seconds"] = _parse_time_to_seconds(row.get("fighter1_ctrl") or "")
    out["fighter2_ctrl_seconds"] = _parse_time_to_seconds(row.get("fighter2_ctrl") or "")

    for col in OF_Y_COLUMNS:
        val = row.get(col) or row.get(SIG_STR_BREAKDOWN_OLD_RAW.get(col, "")) or ""
        landed, attempted, pct = _parse_of_y(val)
        if col in SIG_STR_BREAKDOWN_RAW_TO_BASE:
            base = SIG_STR_BREAKDOWN_RAW_TO_BASE[col]
            out[f"{base}_sig_str_landed"] = landed
            out[f"{base}_sig_str_attempted"] = attempted
            out[f"{base}_sig_str_pct"] = pct
        else:
            infix = "_str_" if col in STRIKE_OF_Y_COLUMNS else "_"
            out[f"{col}{infix}landed"] = landed
            out[f"{col}{infix}attempted"] = attempted
            out[f"{col}{infix}pct"] = pct

    return out


def clean_fights(
    input_path: str | Path,
    output_path: str | Path,
    fighters_path: str | Path = DEFAULT_FIGHTERS_PATH,
    storage: str = "local",
    blob_input_fights: str = DEFAULT_BLOB_INPUT_FIGHTS,
    blob_input_fighters: str = DEFAULT_BLOB_INPUT_FIGHTERS,
    blob_output_csv: str = DEFAULT_BLOB_OUTPUT_CSV,
    blob_output_parquet: str = DEFAULT_BLOB_OUTPUT_PARQUET,
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    fighters_path = Path(fighters_path)
    storage = (storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    raw_rows = _load_input_generic(input_path, storage, blob_input_fights)
    fighter_rows = _load_input_generic(fighters_path, storage, blob_input_fighters)
    fighters_by_url, fighters_by_name = _fighter_rows_to_lookups(fighter_rows)

    cleaned = [clean_row(row, fighters_by_url, fighters_by_name) for row in raw_rows]

    _write_outputs(
        records=cleaned,
        columns=CLEAN_COLUMNS,
        local_csv_path=output_path,
        storage=storage,
        blob_csv_path=blob_output_csv,
        blob_parquet_path=blob_output_parquet,
    )
    return len(cleaned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw UFC fights: join fighters, then clean.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_PATH, help="Raw fights CSV (default: input/raw_ufc_fights.csv)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV (default: output/clean_ufc_fights.csv)")
    parser.add_argument("--fighters", "-f", type=Path, default=DEFAULT_FIGHTERS_PATH, help="Clean fighters CSV (default: input/clean_ufc_fighters.csv)")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to read inputs and write output. local = disk only, azure = blob only, both = disk + blob.",
    )
    parser.add_argument("--blob-input-fights", default=DEFAULT_BLOB_INPUT_FIGHTS, help="Blob path for raw fights input.")
    parser.add_argument("--blob-input-fighters", default=DEFAULT_BLOB_INPUT_FIGHTERS, help="Blob path for clean fighters input.")
    parser.add_argument("--blob-output-csv", default=DEFAULT_BLOB_OUTPUT_CSV, help="Blob path for output CSV.")
    parser.add_argument("--blob-output-parquet", default=DEFAULT_BLOB_OUTPUT_PARQUET, help="Blob path for output Parquet.")
    args = parser.parse_args()

    n = clean_fights(
        args.input,
        args.output,
        args.fighters,
        storage=args.storage,
        blob_input_fights=args.blob_input_fights,
        blob_input_fighters=args.blob_input_fighters,
        blob_output_csv=args.blob_output_csv,
        blob_output_parquet=args.blob_output_parquet,
    )
    print(f"Cleaned {n} rows.")
