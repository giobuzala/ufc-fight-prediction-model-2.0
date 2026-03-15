"""
Feature engineering for UFC fights.

Reads: input/clean_ufc_fights.csv (local or from Azure when --storage azure/both).
- Drops event_url, event_name, event_location, DOB, fighter URLs.
- Normalizes finish_type: M-DEC/S-DEC/U-DEC -> DEC; filters out CNC, DQ, Other, Overturned.
- Adds fighter record (wins/losses), debut flags, rolling per-fight stats (avg pct, per-min, differentials).
- Weight class to lbs; catchweight inferred from fighters' past weights when possible.
- At end: drop blank weight_class/winner, then write. Year filtering done in module_05.
Writes: output/ufc_fights_fe.csv (local) and optionally CSV+Parquet to Azure; also writes to module_05 input.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_INPUT_PATH = _SCRIPT_DIR / "input" / "clean_ufc_fights.csv"
DEFAULT_OUTPUT_PATH = _SCRIPT_DIR / "output" / "ufc_fights_fe.csv"
MODULE_05_INPUT_PATH = _PROJECT_ROOT / "module_05_split" / "input" / "ufc_fights_fe.csv"
DEFAULT_BLOB_INPUT = "module_04_feature_engineering/input/clean_ufc_fights.csv"
DEFAULT_BLOB_CSV = "module_04_feature_engineering/output/ufc_fights_fe.csv"
DEFAULT_BLOB_PARQUET = "module_04_feature_engineering/output/ufc_fights_fe.parquet"
MODULE_05_INPUT_BLOB_CSV = "module_05_split/input/ufc_fights_fe.csv"
MODULE_05_INPUT_BLOB_PARQUET = "module_05_split/input/ufc_fights_fe.parquet"

# --- Output column config ---
DROP_COLUMNS = {"event_url", "event_name", "event_location", "fighter1_dob", "fighter2_dob", "fighter_1_url", "fighter_2_url", "round"}

# Debut flags: inserted after fighter2_stance
DEBUT_COLUMNS = ["fighter1_is_debut", "fighter2_is_debut"]
# Replace fighter1_kd, fighter2_kd in output
KD_REPLACEMENT_COLUMNS = ["fighter1_avg_kd_per_min", "fighter2_avg_kd_per_min"]
KD_DIFF_COLUMN = ["avg_kd_per_min_differential"]  # fighter1_avg_kd_per_min - fighter2_avg_kd_per_min
# Replace the 6 sig str columns in output; each differential immediately after its pair
SIG_STR_REPLACEMENT_COLUMNS = [
    "fighter1_avg_sig_str_pct", "fighter2_avg_sig_str_pct", "avg_sig_str_pct_differential",
    "fighter1_avg_sig_str_per_min", "fighter2_avg_sig_str_per_min", "avg_sig_str_per_min_differential",
]
# Replace the 6 total str columns in output
TOTAL_STR_REPLACEMENT_COLUMNS = [
    "fighter1_avg_total_str_pct", "fighter2_avg_total_str_pct", "avg_total_str_pct_differential",
    "fighter1_avg_total_str_per_min", "fighter2_avg_total_str_per_min", "avg_total_str_per_min_differential",
]
# Replace the 6 td columns in output
TD_REPLACEMENT_COLUMNS = [
    "fighter1_avg_td_pct", "fighter2_avg_td_pct", "avg_td_pct_differential",
    "fighter1_avg_td_per_min", "fighter2_avg_td_per_min", "avg_td_per_min_differential",
]
# Replace sub_att and rev with per-min (like knockdowns)
SUB_ATT_REPLACEMENT_COLUMNS = ["fighter1_avg_sub_att_per_min", "fighter2_avg_sub_att_per_min", "avg_sub_att_per_min_differential"]
REV_REPLACEMENT_COLUMNS = ["fighter1_avg_rev_per_min", "fighter2_avg_rev_per_min", "avg_rev_per_min_differential"]
# Replace ctrl_seconds with avg control seconds per minute (like knockdowns)
CTRL_REPLACEMENT_COLUMNS = ["fighter1_avg_ctrl_seconds_per_min", "fighter2_avg_ctrl_seconds_per_min", "avg_ctrl_seconds_per_min_differential"]
# Replace the 12 head/body/leg + distance/clinch/ground columns: avg block then share block for each group
# Head/body/leg block: avg cols + diffs, then head/body/leg share cols + diffs
# Distance/clinch/ground block: avg cols + diffs, then distance/clinch/ground share cols + diffs
HEAD_BODY_LEG_EXTRA_COLUMNS = [
    "fighter1_avg_head_str_pct", "fighter2_avg_head_str_pct", "avg_head_str_pct_differential",
    "fighter1_avg_head_str_per_min", "fighter2_avg_head_str_per_min", "avg_head_str_per_min_differential",
    "fighter1_avg_body_str_pct", "fighter2_avg_body_str_pct", "avg_body_str_pct_differential",
    "fighter1_avg_body_str_per_min", "fighter2_avg_body_str_per_min", "avg_body_str_per_min_differential",
    "fighter1_avg_leg_str_pct", "fighter2_avg_leg_str_pct", "avg_leg_str_pct_differential",
    "fighter1_avg_leg_str_per_min", "fighter2_avg_leg_str_per_min", "avg_leg_str_per_min_differential",
    "fighter1_head_sig_landed_pct", "fighter2_head_sig_landed_pct", "head_sig_landed_pct_differential",
    "fighter1_body_sig_landed_pct", "fighter2_body_sig_landed_pct", "body_sig_landed_pct_differential",
    "fighter1_leg_sig_landed_pct", "fighter2_leg_sig_landed_pct", "leg_sig_landed_pct_differential",
    "fighter1_avg_distance_str_pct", "fighter2_avg_distance_str_pct", "avg_distance_str_pct_differential",
    "fighter1_avg_distance_str_per_min", "fighter2_avg_distance_str_per_min", "avg_distance_str_per_min_differential",
    "fighter1_avg_clinch_str_pct", "fighter2_avg_clinch_str_pct", "avg_clinch_str_pct_differential",
    "fighter1_avg_clinch_str_per_min", "fighter2_avg_clinch_str_per_min", "avg_clinch_str_per_min_differential",
    "fighter1_avg_ground_str_pct", "fighter2_avg_ground_str_pct", "avg_ground_str_pct_differential",
    "fighter1_avg_ground_str_per_min", "fighter2_avg_ground_str_per_min", "avg_ground_str_per_min_differential",
    "fighter1_distance_sig_landed_pct", "fighter2_distance_sig_landed_pct", "distance_sig_landed_pct_differential",
    "fighter1_clinch_sig_landed_pct", "fighter2_clinch_sig_landed_pct", "clinch_sig_landed_pct_differential",
    "fighter1_ground_sig_landed_pct", "fighter2_ground_sig_landed_pct", "ground_sig_landed_pct_differential",
]
# (fighter1_col, fighter2_col, diff_col_name) for every pair that gets a differential (except winner)
# Used to compute row[diff_col] = parse(col1) - parse(col2). Int columns use round_to=None.
PAIRED_DIFFERENTIALS: list[tuple[str, str, str]] = [
    ("fighter1_age", "fighter2_age", "age_differential"),
    ("fighter1_avg_sig_str_pct", "fighter2_avg_sig_str_pct", "avg_sig_str_pct_differential"),
    ("fighter1_avg_sig_str_per_min", "fighter2_avg_sig_str_per_min", "avg_sig_str_per_min_differential"),
    ("fighter1_avg_total_str_pct", "fighter2_avg_total_str_pct", "avg_total_str_pct_differential"),
    ("fighter1_avg_total_str_per_min", "fighter2_avg_total_str_per_min", "avg_total_str_per_min_differential"),
    ("fighter1_avg_td_pct", "fighter2_avg_td_pct", "avg_td_pct_differential"),
    ("fighter1_avg_td_per_min", "fighter2_avg_td_per_min", "avg_td_per_min_differential"),
    ("fighter1_avg_sub_att_per_min", "fighter2_avg_sub_att_per_min", "avg_sub_att_per_min_differential"),
    ("fighter1_avg_rev_per_min", "fighter2_avg_rev_per_min", "avg_rev_per_min_differential"),
    ("fighter1_avg_ctrl_seconds_per_min", "fighter2_avg_ctrl_seconds_per_min", "avg_ctrl_seconds_per_min_differential"),
    ("fighter1_avg_head_str_pct", "fighter2_avg_head_str_pct", "avg_head_str_pct_differential"),
    ("fighter1_avg_head_str_per_min", "fighter2_avg_head_str_per_min", "avg_head_str_per_min_differential"),
    ("fighter1_avg_body_str_pct", "fighter2_avg_body_str_pct", "avg_body_str_pct_differential"),
    ("fighter1_avg_body_str_per_min", "fighter2_avg_body_str_per_min", "avg_body_str_per_min_differential"),
    ("fighter1_avg_leg_str_pct", "fighter2_avg_leg_str_pct", "avg_leg_str_pct_differential"),
    ("fighter1_avg_leg_str_per_min", "fighter2_avg_leg_str_per_min", "avg_leg_str_per_min_differential"),
    ("fighter1_head_sig_landed_pct", "fighter2_head_sig_landed_pct", "head_sig_landed_pct_differential"),
    ("fighter1_body_sig_landed_pct", "fighter2_body_sig_landed_pct", "body_sig_landed_pct_differential"),
    ("fighter1_leg_sig_landed_pct", "fighter2_leg_sig_landed_pct", "leg_sig_landed_pct_differential"),
    ("fighter1_avg_distance_str_pct", "fighter2_avg_distance_str_pct", "avg_distance_str_pct_differential"),
    ("fighter1_avg_distance_str_per_min", "fighter2_avg_distance_str_per_min", "avg_distance_str_per_min_differential"),
    ("fighter1_avg_clinch_str_pct", "fighter2_avg_clinch_str_pct", "avg_clinch_str_pct_differential"),
    ("fighter1_avg_clinch_str_per_min", "fighter2_avg_clinch_str_per_min", "avg_clinch_str_per_min_differential"),
    ("fighter1_avg_ground_str_pct", "fighter2_avg_ground_str_pct", "avg_ground_str_pct_differential"),
    ("fighter1_avg_ground_str_per_min", "fighter2_avg_ground_str_per_min", "avg_ground_str_per_min_differential"),
    ("fighter1_distance_sig_landed_pct", "fighter2_distance_sig_landed_pct", "distance_sig_landed_pct_differential"),
    ("fighter1_clinch_sig_landed_pct", "fighter2_clinch_sig_landed_pct", "clinch_sig_landed_pct_differential"),
    ("fighter1_ground_sig_landed_pct", "fighter2_ground_sig_landed_pct", "ground_sig_landed_pct_differential"),
    ("fighter1_avg_sig_str_absorbed_per_min", "fighter2_avg_sig_str_absorbed_per_min", "avg_sig_str_absorbed_per_min_differential"),
    ("fighter1_avg_sub_absorbed_per_min", "fighter2_avg_sub_absorbed_per_min", "avg_sub_absorbed_per_min_differential"),
    ("fighter1_dec_wins", "fighter2_dec_wins", "dec_wins_differential"),
    ("fighter1_ko_wins", "fighter2_ko_wins", "ko_wins_differential"),
    ("fighter1_sub_wins", "fighter2_sub_wins", "sub_wins_differential"),
    ("fighter1_total_fight_minutes", "fighter2_total_fight_minutes", "total_fight_minutes_differential"),
]
# Record/streak columns with differentials interleaved (diff after each pair)
RECORD_AND_DIFF_COLUMNS = [
    "fighter1_total_wins", "fighter2_total_wins", "win_differential",
    "fighter1_total_losses", "fighter2_total_losses", "loss_differential",
]
STREAK_AND_DIFF_COLUMNS = [
    "fighter1_win_streak", "fighter2_win_streak", "win_streak_differential",
    "fighter1_lose_streak", "fighter2_lose_streak", "lose_streak_differential",
]
# Replace total_fight_time_seconds with rolling total fight minutes (cumulative cage time before each fight)
TOTAL_FIGHT_MINUTES_COLUMNS = [
    "fighter1_total_fight_minutes", "fighter2_total_fight_minutes", "total_fight_minutes_differential",
]
# Rolling totals of wins by finish type (decisions, knockouts, submissions)
FINISH_WINS_COLUMNS = [
    "fighter1_dec_wins", "fighter2_dec_wins", "dec_wins_differential",
    "fighter1_ko_wins", "fighter2_ko_wins", "ko_wins_differential",
    "fighter1_sub_wins", "fighter2_sub_wins", "sub_wins_differential",
]
AGE_DIFF_COLUMN = ["age_differential"]
HEIGHT_DIFF_COLUMN = ["height_differential"]
REACH_DIFF_COLUMN = ["reach_differential"]
# Absorbed stats (sig strikes taken, times submitted): rolling per-min, placed before winner
ABSORBED_COLUMNS = [
    "fighter1_avg_sig_str_absorbed_per_min", "fighter2_avg_sig_str_absorbed_per_min", "avg_sig_str_absorbed_per_min_differential",
    "fighter1_avg_sub_absorbed_per_min", "fighter2_avg_sub_absorbed_per_min", "avg_sub_absorbed_per_min_differential",
]

# Original columns to drop from output (replaced by the above)
DROP_ORIGINAL_STATS = {
    "fighter1_kd", "fighter2_kd",
    "fighter1_sig_str_landed", "fighter1_sig_str_attempted", "fighter1_sig_str_pct",
    "fighter2_sig_str_landed", "fighter2_sig_str_attempted", "fighter2_sig_str_pct",
    "fighter1_total_str_landed", "fighter1_total_str_attempted", "fighter1_total_str_pct",
    "fighter2_total_str_landed", "fighter2_total_str_attempted", "fighter2_total_str_pct",
    "fighter1_td_landed", "fighter1_td_attempted", "fighter1_td_pct",
    "fighter2_td_landed", "fighter2_td_attempted", "fighter2_td_pct",
    "fighter1_sub_att", "fighter2_sub_att", "fighter1_rev", "fighter2_rev",
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
    "total_fight_time_seconds",  # replaced by fighter1/2_total_fight_minutes (rolling cumulative)
}

# Finish_type: normalize decisions; exclude no-contest / DQ / other / overturned
DEC_VARIANTS = {"M-DEC", "S-DEC", "U-DEC"}
FINISH_TYPES_EXCLUDE = {"CNC", "DQ", "OTHER", "OVERTURNED"}


# --- I/O ---
def _load_input(
    local_path: Path,
    storage: str,
    blob_input_path: str,
) -> tuple[list[dict], list[str]]:
    """Load input rows and fieldnames. When storage is azure/both, use cloud (Parquet then CSV); fall back to local for both."""
    storage = (storage or "local").strip().lower()
    blob_parquet = blob_input_path.replace(".csv", ".parquet") if blob_input_path.endswith(".csv") else blob_input_path + ".parquet"

    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import read_csv_from_azure, read_parquet_from_azure

        try:
            try:
                df = read_parquet_from_azure(blob_parquet).astype(str).fillna("")
            except FileNotFoundError:
                df = read_csv_from_azure(blob_input_path, dtype=str).fillna("")
            columns = list(df.columns)
            return df.to_dict(orient="records"), columns
        except FileNotFoundError:
            if storage == "azure":
                raise
            pass

    if storage in ("local", "both") and local_path.exists():
        with open(local_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
            if not fieldnames and rows:
                fieldnames = list(rows[0].keys())
        return rows, fieldnames
    raise FileNotFoundError(f"Input not found: {local_path}")


def _write_outputs(
    *,
    records: list[dict],
    columns: list[str],
    local_csv_path: Path,
    storage: str,
    blob_csv_path: str,
    blob_parquet_path: str,
) -> None:
    """Write output to local path and module_05 input; when storage is azure/both, also write CSV+Parquet to Azure."""
    df = pd.DataFrame.from_records(records)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[columns].fillna("").astype(str)

    if storage in ("local", "both"):
        local_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_csv_path, index=False)
        print(f"Wrote {len(df)} rows to {local_csv_path}")
        module_05_input = MODULE_05_INPUT_PATH
        module_05_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(module_05_input, index=False)
        print(f"Updated module_05 input: {module_05_input}")

    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import write_csv_to_azure, write_parquet_to_azure

            write_csv_to_azure(df, blob_csv_path, index=False)
            write_parquet_to_azure(df, blob_parquet_path, index=False)
            print(f"Wrote {len(df)} rows to Azure blobs: {blob_csv_path} and {blob_parquet_path}")
            write_csv_to_azure(df, MODULE_05_INPUT_BLOB_CSV, index=False)
            write_parquet_to_azure(df, MODULE_05_INPUT_BLOB_PARQUET, index=False)
            print(f"Updated module_05 input in container: {MODULE_05_INPUT_BLOB_CSV} and {MODULE_05_INPUT_BLOB_PARQUET}")
        except Exception as e:
            print(f"Failed to write to Azure: {e}")
            raise


# --- Helpers ---
def _parse_event_date(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _numeric_diff(row: dict, col1: str, col2: str, round_to: int | None = 6) -> str:
    """Compute row[col1] - row[col2] as string; empty if either missing/invalid. round_to=None for int."""
    try:
        v1, v2 = row.get(col1), row.get(col2)
        v1 = str(v1).strip() if v1 is not None and v1 != "" else ""
        v2 = str(v2).strip() if v2 is not None and v2 != "" else ""
        if not v1 or not v2:
            return ""
        x1, x2 = float(v1), float(v2)
        if round_to is None:
            return str(int(round(x1 - x2)))
        return str(round(x1 - x2, round_to))
    except (ValueError, TypeError):
        return ""


def _safe_int(s: str, default: int = 0) -> int:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def _fight_minutes(row: dict) -> float:
    """Fight length in minutes from total_fight_time_seconds or number_of_rounds * 5."""
    ts = (row.get("total_fight_time_seconds") or "").strip()
    if ts:
        try:
            return int(ts) / 60.0
        except (ValueError, TypeError):
            pass
    nr = (row.get("number_of_rounds") or "").strip()
    if nr:
        try:
            return int(nr) * 5.0
        except (ValueError, TypeError):
            pass
    return 0.0


# UFC weight class name -> limit in lbs (upper limit for non-heavyweight; heavyweight is 265 max)
WEIGHT_CLASS_LBS = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}
# Labels we treat as catchweight/open weight and try to infer from fighters' usual weight (normalized: lower, no extra spaces)
def _is_catchweight_or_open(wc: str) -> bool:
    n = (wc or "").strip().lower().replace("  ", " ")
    return "catchweight" in n or "catch weight" in n or "open weight" in n


def _weight_class_to_lbs(wc: str) -> str:
    """Convert weight_class string to lbs (numeric string). Unknown/Catchweight -> ''."""
    wc = (wc or "").strip()
    if wc in WEIGHT_CLASS_LBS:
        return str(WEIGHT_CLASS_LBS[wc])
    return ""


def _most_common_weight(weights: list[int]) -> int | None:
    """Return the most common (mode) weight in lbs, or None if empty."""
    if not weights:
        return None
    (w, _) = Counter(weights).most_common(1)[0]
    return w


def _classify_outcome(winner: str, f1: str, f2: str) -> tuple[str, str]:
    """Return (result_fighter1, result_fighter2): 'win', 'loss', or 'other'."""
    w = (winner or "").strip()
    f1, f2 = (f1 or "").strip(), (f2 or "").strip()
    if w == f1:
        return "win", "loss"
    if w == f2:
        return "loss", "win"
    return "other", "other"


def _event_year(row: dict) -> int:
    """Event date year; 0 if missing or unparseable."""
    dt = _parse_event_date(row.get("event_date") or "")
    return dt.year if dt else 0


def _rate(num: float, denom: float, decimals: int = 6) -> str:
    """Return num/denom rounded to decimals as string, or '' if denom <= 0."""
    return str(round(num / denom, decimals)) if denom > 0 else ""


def _rolling_stat(stats: dict[str, float]) -> tuple[str, ...]:
    """Compute rolling averages from accumulated fight stats (before current fight).
    Returns 30 values: pct/per_min for sig, total, td, sub_att, rev, ctrl; head/body/leg;
    distance/clinch/ground; sig/sub absorbed; head/body/leg/distance/clinch/ground shares."""
    if (
        stats["minutes"] <= 0
        and stats["sig_attempted"] <= 0
        and stats["total_attempted"] <= 0
        and stats["td_attempted"] <= 0
    ):
        return ("",) * 30
    mins = stats["minutes"]
    sig_pct = _rate(stats["sig_landed"], stats["sig_attempted"])
    sig_pm = _rate(stats["sig_landed"], mins)
    kd_pm = _rate(stats["kd"], mins)
    total_pct = _rate(stats["total_landed"], stats["total_attempted"])
    total_pm = _rate(stats["total_landed"], mins)
    td_pct = _rate(stats["td_landed"], stats["td_attempted"])
    td_pm = _rate(stats["td_landed"], mins)
    sub_att_pm = _rate(stats["sub_att"], mins)
    rev_pm = _rate(stats["rev"], mins)
    ctrl_pm = _rate(stats["ctrl_seconds"], mins)
    head_pct = _rate(stats["head_landed"], stats["head_attempted"])
    head_pm = _rate(stats["head_landed"], mins)
    body_pct = _rate(stats["body_landed"], stats["body_attempted"])
    body_pm = _rate(stats["body_landed"], mins)
    leg_pct = _rate(stats["leg_landed"], stats["leg_attempted"])
    leg_pm = _rate(stats["leg_landed"], mins)
    distance_pct = _rate(stats["distance_landed"], stats["distance_attempted"])
    distance_pm = _rate(stats["distance_landed"], mins)
    clinch_pct = _rate(stats["clinch_landed"], stats["clinch_attempted"])
    clinch_pm = _rate(stats["clinch_landed"], mins)
    ground_pct = _rate(stats["ground_landed"], stats["ground_attempted"])
    ground_pm = _rate(stats["ground_landed"], mins)
    sig_absorbed_pm = _rate(stats["sig_absorbed"], mins)
    sub_absorbed_pm = _rate(stats["sub_absorbed"], mins)
    # Share of head/body/leg within (head+body+leg) total; each group sums to 100
    tot_hbl = stats["head_landed"] + stats["body_landed"] + stats["leg_landed"]
    if tot_hbl > 0:
        head_share = str(round(100.0 * stats["head_landed"] / tot_hbl, 2))
        body_share = str(round(100.0 * stats["body_landed"] / tot_hbl, 2))
        leg_share = str(round(100.0 - 100.0 * (stats["head_landed"] + stats["body_landed"]) / tot_hbl, 2))
    else:
        head_share = body_share = leg_share = ""
    tot_dcg = stats["distance_landed"] + stats["clinch_landed"] + stats["ground_landed"]
    if tot_dcg > 0:
        distance_share = str(round(100.0 * stats["distance_landed"] / tot_dcg, 2))
        clinch_share = str(round(100.0 * stats["clinch_landed"] / tot_dcg, 2))
        ground_share = str(round(100.0 - 100.0 * (stats["distance_landed"] + stats["clinch_landed"]) / tot_dcg, 2))
    else:
        distance_share = clinch_share = ground_share = ""
    return (
        sig_pct, sig_pm, kd_pm, total_pct, total_pm, td_pct, td_pm, sub_att_pm, rev_pm, ctrl_pm,
        head_pct, head_pm, body_pct, body_pm, leg_pct, leg_pm,
        distance_pct, distance_pm, clinch_pct, clinch_pm, ground_pct, ground_pm,
        sig_absorbed_pm, sub_absorbed_pm,
        head_share, body_share, leg_share, distance_share, clinch_share, ground_share,
    )


def compute_fighter_state_snapshot(input_path: Path | None = None, rows: list[dict] | None = None) -> dict:
    """
    Replay FE state-update logic over fights and return fighter state after last fight.
    Either input_path or rows must be provided. Rows must be pre-filtered and sorted by event_date.
    Returns: {fighter_id: {"records": {...}, "streaks": {...}, "fight_stats": {...},
              "finish_wins": {...}, "fighter_weights": [...], "total_fight_minutes": float}}
    """
    if rows is None:
        if input_path is None:
            raise ValueError("Either input_path or rows must be provided")
        with open(input_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    for row in rows:
        ft = (row.get("finish_type") or "").strip().upper()
        if ft in DEC_VARIANTS:
            row["finish_type"] = "DEC"
    rows = [r for r in rows if (r.get("finish_type") or "").strip().upper() not in FINISH_TYPES_EXCLUDE]
    if not rows:
        return {}
    for i, row in enumerate(rows):
        row["_orig_idx"] = i
    rows.sort(key=lambda r: (_parse_event_date(r.get("event_date") or "") is None, _parse_event_date(r.get("event_date") or "") or datetime.min, r["_orig_idx"]))

    records: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "other": 0})
    streaks: dict[str, dict[str, int]] = defaultdict(lambda: {"win_streak": 0, "lose_streak": 0})
    finish_wins: dict[str, dict[str, int]] = defaultdict(lambda: {"dec_wins": 0, "ko_wins": 0, "sub_wins": 0})
    fighter_weights: dict[str, list[int]] = defaultdict(list)
    fight_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "kd": 0.0, "sig_landed": 0.0, "sig_attempted": 0.0, "sig_absorbed": 0.0,
            "total_landed": 0.0, "total_attempted": 0.0,
            "td_landed": 0.0, "td_attempted": 0.0,
            "sub_att": 0.0, "sub_absorbed": 0.0, "rev": 0.0, "ctrl_seconds": 0.0,
            "head_landed": 0.0, "head_attempted": 0.0, "body_landed": 0.0, "body_attempted": 0.0,
            "leg_landed": 0.0, "leg_attempted": 0.0,
            "distance_landed": 0.0, "distance_attempted": 0.0,
            "clinch_landed": 0.0, "clinch_attempted": 0.0,
            "ground_landed": 0.0, "ground_attempted": 0.0,
            "minutes": 0.0,
        }
    )

    for row in rows:
        f1 = (row.get("fighter_1") or "").strip()
        f2 = (row.get("fighter_2") or "").strip()
        id1 = (row.get("fighter_1_url") or "").strip() or f1
        id2 = (row.get("fighter_2_url") or "").strip() or f2
        winner = row.get("winner") or ""
        minutes = _fight_minutes(row)
        kd1 = _safe_int(row.get("fighter1_kd"))
        kd2 = _safe_int(row.get("fighter2_kd"))
        sig_l1 = _safe_int(row.get("fighter1_sig_str_landed"))
        sig_a1 = _safe_int(row.get("fighter1_sig_str_attempted"))
        sig_l2 = _safe_int(row.get("fighter2_sig_str_landed"))
        sig_a2 = _safe_int(row.get("fighter2_sig_str_attempted"))
        total_l1 = _safe_int(row.get("fighter1_total_str_landed"))
        total_a1 = _safe_int(row.get("fighter1_total_str_attempted"))
        total_l2 = _safe_int(row.get("fighter2_total_str_landed"))
        total_a2 = _safe_int(row.get("fighter2_total_str_attempted"))
        td_l1 = _safe_int(row.get("fighter1_td_landed"))
        td_a1 = _safe_int(row.get("fighter1_td_attempted"))
        td_l2 = _safe_int(row.get("fighter2_td_landed"))
        td_a2 = _safe_int(row.get("fighter2_td_attempted"))
        sub_att1 = _safe_int(row.get("fighter1_sub_att"))
        sub_att2 = _safe_int(row.get("fighter2_sub_att"))
        rev1 = _safe_int(row.get("fighter1_rev"))
        rev2 = _safe_int(row.get("fighter2_rev"))
        ctrl1 = _safe_int(row.get("fighter1_ctrl_seconds"))
        ctrl2 = _safe_int(row.get("fighter2_ctrl_seconds"))
        fight_stats[id1]["head_landed"] += _safe_int(row.get("fighter1_head_sig_str_landed"))
        fight_stats[id1]["head_attempted"] += _safe_int(row.get("fighter1_head_sig_str_attempted"))
        fight_stats[id1]["body_landed"] += _safe_int(row.get("fighter1_body_sig_str_landed"))
        fight_stats[id1]["body_attempted"] += _safe_int(row.get("fighter1_body_sig_str_attempted"))
        fight_stats[id1]["leg_landed"] += _safe_int(row.get("fighter1_leg_sig_str_landed"))
        fight_stats[id1]["leg_attempted"] += _safe_int(row.get("fighter1_leg_sig_str_attempted"))
        fight_stats[id1]["distance_landed"] += _safe_int(row.get("fighter1_distance_sig_str_landed"))
        fight_stats[id1]["distance_attempted"] += _safe_int(row.get("fighter1_distance_sig_str_attempted"))
        fight_stats[id1]["clinch_landed"] += _safe_int(row.get("fighter1_clinch_sig_str_landed"))
        fight_stats[id1]["clinch_attempted"] += _safe_int(row.get("fighter1_clinch_sig_str_attempted"))
        fight_stats[id1]["ground_landed"] += _safe_int(row.get("fighter1_ground_sig_str_landed"))
        fight_stats[id1]["ground_attempted"] += _safe_int(row.get("fighter1_ground_sig_str_attempted"))
        fight_stats[id2]["head_landed"] += _safe_int(row.get("fighter2_head_sig_str_landed"))
        fight_stats[id2]["head_attempted"] += _safe_int(row.get("fighter2_head_sig_str_attempted"))
        fight_stats[id2]["body_landed"] += _safe_int(row.get("fighter2_body_sig_str_landed"))
        fight_stats[id2]["body_attempted"] += _safe_int(row.get("fighter2_body_sig_str_attempted"))
        fight_stats[id2]["leg_landed"] += _safe_int(row.get("fighter2_leg_sig_str_landed"))
        fight_stats[id2]["leg_attempted"] += _safe_int(row.get("fighter2_leg_sig_str_attempted"))
        fight_stats[id2]["distance_landed"] += _safe_int(row.get("fighter2_distance_sig_str_landed"))
        fight_stats[id2]["distance_attempted"] += _safe_int(row.get("fighter2_distance_sig_str_attempted"))
        fight_stats[id2]["clinch_landed"] += _safe_int(row.get("fighter2_clinch_sig_str_landed"))
        fight_stats[id2]["clinch_attempted"] += _safe_int(row.get("fighter2_clinch_sig_str_attempted"))
        fight_stats[id2]["ground_landed"] += _safe_int(row.get("fighter2_ground_sig_str_landed"))
        fight_stats[id2]["ground_attempted"] += _safe_int(row.get("fighter2_ground_sig_str_attempted"))
        fight_stats[id1]["kd"] += kd1
        fight_stats[id1]["sig_landed"] += sig_l1
        fight_stats[id1]["sig_attempted"] += sig_a1
        fight_stats[id1]["total_landed"] += total_l1
        fight_stats[id1]["total_attempted"] += total_a1
        fight_stats[id1]["td_landed"] += td_l1
        fight_stats[id1]["td_attempted"] += td_a1
        fight_stats[id1]["minutes"] += minutes
        fight_stats[id2]["kd"] += kd2
        fight_stats[id2]["sig_landed"] += sig_l2
        fight_stats[id2]["sig_attempted"] += sig_a2
        fight_stats[id2]["total_landed"] += total_l2
        fight_stats[id2]["total_attempted"] += total_a2
        fight_stats[id2]["td_landed"] += td_l2
        fight_stats[id2]["td_attempted"] += td_a2
        fight_stats[id1]["sub_att"] += sub_att1
        fight_stats[id1]["rev"] += rev1
        fight_stats[id2]["sub_att"] += sub_att2
        fight_stats[id2]["rev"] += rev2
        fight_stats[id1]["ctrl_seconds"] += ctrl1
        fight_stats[id2]["ctrl_seconds"] += ctrl2
        fight_stats[id2]["minutes"] += minutes
        fight_stats[id1]["sig_absorbed"] += sig_l2
        fight_stats[id2]["sig_absorbed"] += sig_l1
        ft = (row.get("finish_type") or "").strip().upper()
        if ft == "SUB":
            if (winner or "").strip() == f2:
                fight_stats[id1]["sub_absorbed"] += 1
            elif (winner or "").strip() == f1:
                fight_stats[id2]["sub_absorbed"] += 1
        res1, res2 = _classify_outcome(winner, f1, f2)
        key1 = "wins" if res1 == "win" else ("losses" if res1 == "loss" else "other")
        key2 = "wins" if res2 == "win" else ("losses" if res2 == "loss" else "other")
        records[id1][key1] += 1
        records[id2][key2] += 1
        if res1 == "win":
            if ft == "DEC":
                finish_wins[id1]["dec_wins"] += 1
            elif ft == "SUB":
                finish_wins[id1]["sub_wins"] += 1
            elif ft in ("KO", "TKO", "KO/TKO"):
                finish_wins[id1]["ko_wins"] += 1
        if res2 == "win":
            if ft == "DEC":
                finish_wins[id2]["dec_wins"] += 1
            elif ft == "SUB":
                finish_wins[id2]["sub_wins"] += 1
            elif ft in ("KO", "TKO", "KO/TKO"):
                finish_wins[id2]["ko_wins"] += 1
        if res1 == "win":
            streaks[id1]["win_streak"] += 1
            streaks[id1]["lose_streak"] = 0
            streaks[id2]["lose_streak"] += 1
            streaks[id2]["win_streak"] = 0
        elif res1 == "loss":
            streaks[id1]["lose_streak"] += 1
            streaks[id1]["win_streak"] = 0
            streaks[id2]["win_streak"] += 1
            streaks[id2]["lose_streak"] = 0
        else:
            streaks[id1]["win_streak"] = 0
            streaks[id1]["lose_streak"] = 0
            streaks[id2]["win_streak"] = 0
            streaks[id2]["lose_streak"] = 0
        raw_wc = (row.get("weight_class") or "").strip()
        lbs = _weight_class_to_lbs(raw_wc)
        if not lbs and _is_catchweight_or_open(raw_wc):
            typical1 = _most_common_weight(fighter_weights[id1])
            typical2 = _most_common_weight(fighter_weights[id2])
            if typical1 is not None and typical2 is not None and typical1 == typical2:
                lbs = str(typical1)
        if lbs and lbs.isdigit():
            w = int(lbs)
            fighter_weights[id1].append(w)
            fighter_weights[id2].append(w)

    result: dict = {}
    for fid in set(records.keys()) | set(streaks.keys()) | set(fight_stats.keys()):
        result[fid] = {
            "records": dict(records[fid]),
            "streaks": dict(streaks[fid]),
            "fight_stats": {k: float(v) for k, v in fight_stats[fid].items()},
            "finish_wins": dict(finish_wins[fid]),
            "fighter_weights": list(fighter_weights[fid]),
            "total_fight_minutes": float(fight_stats[fid]["minutes"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering: drop cols, add fighter records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input clean_ufc_fights CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to read input and write output. local = disk only, azure = blob only, both = disk + blob.",
    )
    parser.add_argument("--blob-input", default=DEFAULT_BLOB_INPUT, help="Blob path for input (when reading from Azure).")
    parser.add_argument("--blob-csv", default=DEFAULT_BLOB_CSV, help="Blob path for output CSV.")
    parser.add_argument("--blob-parquet", default=DEFAULT_BLOB_PARQUET, help="Blob path for output Parquet.")
    args = parser.parse_args()

    storage = (args.storage or "local").strip().lower()
    rows, fieldnames = _load_input(args.input, storage, args.blob_input)

    # Normalize finish_type and drop excluded outcomes
    for row in rows:
        ft = (row.get("finish_type") or "").strip().upper()
        if ft in DEC_VARIANTS:
            row["finish_type"] = "DEC"
    rows = [r for r in rows if (r.get("finish_type") or "").strip().upper() not in FINISH_TYPES_EXCLUDE]

    if not rows:
        empty_columns = [c for c in fieldnames if c not in DROP_COLUMNS]
        _write_outputs(
            records=[],
            columns=empty_columns,
            local_csv_path=args.output,
            storage=storage,
            blob_csv_path=args.blob_csv,
            blob_parquet_path=args.blob_parquet,
        )
        return

    # Preserve original order (index) for output
    for i, row in enumerate(rows):
        row["_orig_idx"] = i

    # Sort by event_date ascending for correct record computation (missing dates last)
    def sort_key(r: dict) -> tuple[bool, datetime, int]:
        dt = _parse_event_date(r.get("event_date", "") or "")
        return (dt is None, dt or datetime.min, r["_orig_idx"])

    rows.sort(key=sort_key)

    # Running record per fighter: identity (URL if present, else name) -> {wins, losses, other}
    records: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "other": 0})
    # Consecutive win/loss streaks going into each fight (reset on draw/NC)
    streaks: dict[str, dict[str, int]] = defaultdict(lambda: {"win_streak": 0, "lose_streak": 0})
    # Rolling totals of wins by finish type (dec, ko, sub) going into each fight
    finish_wins: dict[str, dict[str, int]] = defaultdict(lambda: {"dec_wins": 0, "ko_wins": 0, "sub_wins": 0})
    # Per-fighter list of weight_class (lbs) from past fights (standard weights only) for catchweight inference
    fighter_weights: dict[str, list[int]] = defaultdict(list)
    # Rolling stats per fighter (before this fight): kd, sig, total str, td, sub_att, rev, ctrl_seconds, head/body/leg, minutes
    fight_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "kd": 0.0, "sig_landed": 0.0, "sig_attempted": 0.0,
            "sig_absorbed": 0.0,
            "total_landed": 0.0, "total_attempted": 0.0,
            "td_landed": 0.0, "td_attempted": 0.0,
            "sub_att": 0.0, "sub_absorbed": 0.0, "rev": 0.0, "ctrl_seconds": 0.0,
            "head_landed": 0.0, "head_attempted": 0.0,
            "body_landed": 0.0, "body_attempted": 0.0,
            "leg_landed": 0.0, "leg_attempted": 0.0,
            "distance_landed": 0.0, "distance_attempted": 0.0,
            "clinch_landed": 0.0, "clinch_attempted": 0.0,
            "ground_landed": 0.0, "ground_attempted": 0.0,
            "minutes": 0.0,
        }
    )

    for row in rows:
        f1 = (row.get("fighter_1") or "").strip()
        f2 = (row.get("fighter_2") or "").strip()
        # Use fighter URL as identity when present so duplicate names (e.g. two Jean Silvas) get correct separate records
        id1 = (row.get("fighter_1_url") or "").strip() or f1
        id2 = (row.get("fighter_2_url") or "").strip() or f2
        winner = row.get("winner") or ""

        # Record going INTO this fight
        r1 = records[id1]
        r2 = records[id2]
        row["fighter1_total_wins"] = r1["wins"]
        row["fighter1_total_losses"] = r1["losses"]
        row["fighter2_total_wins"] = r2["wins"]
        row["fighter2_total_losses"] = r2["losses"]
        row["win_differential"] = r1["wins"] - r2["wins"]
        row["loss_differential"] = r1["losses"] - r2["losses"]
        row["fighter1_win_streak"] = streaks[id1]["win_streak"]
        row["fighter2_win_streak"] = streaks[id2]["win_streak"]
        row["fighter1_lose_streak"] = streaks[id1]["lose_streak"]
        row["fighter2_lose_streak"] = streaks[id2]["lose_streak"]
        row["win_streak_differential"] = streaks[id1]["win_streak"] - streaks[id2]["win_streak"]
        row["lose_streak_differential"] = streaks[id1]["lose_streak"] - streaks[id2]["lose_streak"]
        row["fighter1_dec_wins"] = finish_wins[id1]["dec_wins"]
        row["fighter2_dec_wins"] = finish_wins[id2]["dec_wins"]
        row["fighter1_ko_wins"] = finish_wins[id1]["ko_wins"]
        row["fighter2_ko_wins"] = finish_wins[id2]["ko_wins"]
        row["fighter1_sub_wins"] = finish_wins[id1]["sub_wins"]
        row["fighter2_sub_wins"] = finish_wins[id2]["sub_wins"]
        row["fighter1_total_fight_minutes"] = round(fight_stats[id1]["minutes"], 6)
        row["fighter2_total_fight_minutes"] = round(fight_stats[id2]["minutes"], 6)
        h1, h2 = _safe_int(row.get("fighter1_height")), _safe_int(row.get("fighter2_height"))
        rch1, rch2 = _safe_int(row.get("fighter1_reach")), _safe_int(row.get("fighter2_reach"))
        row["height_differential"] = h1 - h2 if (h1 or h2) else ""
        row["reach_differential"] = rch1 - rch2 if (rch1 or rch2) else ""
        raw_wc = (row.get("weight_class") or "").strip()
        lbs = _weight_class_to_lbs(raw_wc)
        if lbs:
            row["weight_class"] = lbs
        elif _is_catchweight_or_open(raw_wc):
            typical1 = _most_common_weight(fighter_weights[id1])
            typical2 = _most_common_weight(fighter_weights[id2])
            row["weight_class"] = str(typical1) if typical1 is not None and typical2 is not None and typical1 == typical2 else ""
        else:
            row["weight_class"] = ""

        # Prior fight count (for debut and rolling stats)
        n1 = r1["wins"] + r1["losses"] + r1["other"]
        n2 = r2["wins"] + r2["losses"] + r2["other"]

        s1 = fight_stats[id1]
        s2 = fight_stats[id2]
        row["fighter1_is_debut"] = 1 if n1 == 0 else 0
        row["fighter2_is_debut"] = 1 if n2 == 0 else 0
        if n1 == 0:
            row["fighter1_avg_sig_str_pct"] = ""
            row["fighter1_avg_sig_str_per_min"] = ""
            row["fighter1_avg_kd_per_min"] = ""
            row["fighter1_avg_total_str_pct"] = ""
            row["fighter1_avg_total_str_per_min"] = ""
            row["fighter1_avg_td_pct"] = ""
            row["fighter1_avg_td_per_min"] = ""
            row["fighter1_avg_sub_att_per_min"] = ""
            row["fighter1_avg_rev_per_min"] = ""
            row["fighter1_avg_ctrl_seconds_per_min"] = ""
            row["fighter1_avg_head_str_pct"] = ""
            row["fighter1_avg_head_str_per_min"] = ""
            row["fighter1_avg_body_str_pct"] = ""
            row["fighter1_avg_body_str_per_min"] = ""
            row["fighter1_avg_leg_str_pct"] = ""
            row["fighter1_avg_leg_str_per_min"] = ""
            row["fighter1_avg_distance_str_pct"] = ""
            row["fighter1_avg_distance_str_per_min"] = ""
            row["fighter1_avg_clinch_str_pct"] = ""
            row["fighter1_avg_clinch_str_per_min"] = ""
            row["fighter1_avg_ground_str_pct"] = ""
            row["fighter1_avg_ground_str_per_min"] = ""
            row["fighter1_head_sig_landed_pct"] = ""
            row["fighter1_body_sig_landed_pct"] = ""
            row["fighter1_leg_sig_landed_pct"] = ""
            row["fighter1_distance_sig_landed_pct"] = ""
            row["fighter1_clinch_sig_landed_pct"] = ""
            row["fighter1_ground_sig_landed_pct"] = ""
            row["fighter1_avg_sig_str_absorbed_per_min"] = ""
            row["fighter1_avg_sub_absorbed_per_min"] = ""
        else:
            r1_vals = _rolling_stat(s1)
            (
                sig_pct1, sig_pm1, kd1, total_pct1, total_pm1,
                td_pct1, td_pm1, sub_att_pm1, rev_pm1, ctrl_pm1,
                head_pct1, head_pm1, body_pct1, body_pm1, leg_pct1, leg_pm1,
                distance_pct1, distance_pm1, clinch_pct1, clinch_pm1, ground_pct1, ground_pm1,
                sig_absorbed_pm1, sub_absorbed_pm1,
                head_share1, body_share1, leg_share1, distance_share1, clinch_share1, ground_share1,
            ) = r1_vals
            row["fighter1_avg_sig_str_pct"] = sig_pct1
            row["fighter1_avg_sig_str_per_min"] = sig_pm1
            row["fighter1_avg_kd_per_min"] = kd1
            row["fighter1_avg_total_str_pct"] = total_pct1
            row["fighter1_avg_total_str_per_min"] = total_pm1
            row["fighter1_avg_td_pct"] = td_pct1
            row["fighter1_avg_td_per_min"] = td_pm1
            row["fighter1_avg_sub_att_per_min"] = sub_att_pm1
            row["fighter1_avg_rev_per_min"] = rev_pm1
            row["fighter1_avg_ctrl_seconds_per_min"] = ctrl_pm1
            row["fighter1_avg_head_str_pct"] = head_pct1
            row["fighter1_avg_head_str_per_min"] = head_pm1
            row["fighter1_avg_body_str_pct"] = body_pct1
            row["fighter1_avg_body_str_per_min"] = body_pm1
            row["fighter1_avg_leg_str_pct"] = leg_pct1
            row["fighter1_avg_leg_str_per_min"] = leg_pm1
            row["fighter1_avg_distance_str_pct"] = distance_pct1
            row["fighter1_avg_distance_str_per_min"] = distance_pm1
            row["fighter1_avg_clinch_str_pct"] = clinch_pct1
            row["fighter1_avg_clinch_str_per_min"] = clinch_pm1
            row["fighter1_avg_ground_str_pct"] = ground_pct1
            row["fighter1_avg_ground_str_per_min"] = ground_pm1
            row["fighter1_head_sig_landed_pct"] = head_share1
            row["fighter1_body_sig_landed_pct"] = body_share1
            row["fighter1_leg_sig_landed_pct"] = leg_share1
            row["fighter1_distance_sig_landed_pct"] = distance_share1
            row["fighter1_clinch_sig_landed_pct"] = clinch_share1
            row["fighter1_ground_sig_landed_pct"] = ground_share1
            row["fighter1_avg_sig_str_absorbed_per_min"] = sig_absorbed_pm1
            row["fighter1_avg_sub_absorbed_per_min"] = sub_absorbed_pm1
        if n2 == 0:
            row["fighter2_avg_sig_str_pct"] = ""
            row["fighter2_avg_sig_str_per_min"] = ""
            row["fighter2_avg_kd_per_min"] = ""
            row["fighter2_avg_total_str_pct"] = ""
            row["fighter2_avg_total_str_per_min"] = ""
            row["fighter2_avg_td_pct"] = ""
            row["fighter2_avg_td_per_min"] = ""
            row["fighter2_avg_sub_att_per_min"] = ""
            row["fighter2_avg_rev_per_min"] = ""
            row["fighter2_avg_ctrl_seconds_per_min"] = ""
            row["fighter2_avg_head_str_pct"] = ""
            row["fighter2_avg_head_str_per_min"] = ""
            row["fighter2_avg_body_str_pct"] = ""
            row["fighter2_avg_body_str_per_min"] = ""
            row["fighter2_avg_leg_str_pct"] = ""
            row["fighter2_avg_leg_str_per_min"] = ""
            row["fighter2_avg_distance_str_pct"] = ""
            row["fighter2_avg_distance_str_per_min"] = ""
            row["fighter2_avg_clinch_str_pct"] = ""
            row["fighter2_avg_clinch_str_per_min"] = ""
            row["fighter2_avg_ground_str_pct"] = ""
            row["fighter2_avg_ground_str_per_min"] = ""
            row["fighter2_head_sig_landed_pct"] = ""
            row["fighter2_body_sig_landed_pct"] = ""
            row["fighter2_leg_sig_landed_pct"] = ""
            row["fighter2_distance_sig_landed_pct"] = ""
            row["fighter2_clinch_sig_landed_pct"] = ""
            row["fighter2_ground_sig_landed_pct"] = ""
            row["fighter2_avg_sig_str_absorbed_per_min"] = ""
            row["fighter2_avg_sub_absorbed_per_min"] = ""
        else:
            r2_vals = _rolling_stat(s2)
            (
                sig_pct2, sig_pm2, kd2, total_pct2, total_pm2,
                td_pct2, td_pm2, sub_att_pm2, rev_pm2, ctrl_pm2,
                head_pct2, head_pm2, body_pct2, body_pm2, leg_pct2, leg_pm2,
                distance_pct2, distance_pm2, clinch_pct2, clinch_pm2, ground_pct2, ground_pm2,
                sig_absorbed_pm2, sub_absorbed_pm2,
                head_share2, body_share2, leg_share2, distance_share2, clinch_share2, ground_share2,
            ) = r2_vals
            row["fighter2_avg_sig_str_pct"] = sig_pct2
            row["fighter2_avg_sig_str_per_min"] = sig_pm2
            row["fighter2_avg_kd_per_min"] = kd2
            row["fighter2_avg_total_str_pct"] = total_pct2
            row["fighter2_avg_total_str_per_min"] = total_pm2
            row["fighter2_avg_td_pct"] = td_pct2
            row["fighter2_avg_td_per_min"] = td_pm2
            row["fighter2_avg_sub_att_per_min"] = sub_att_pm2
            row["fighter2_avg_rev_per_min"] = rev_pm2
            row["fighter2_avg_ctrl_seconds_per_min"] = ctrl_pm2
            row["fighter2_avg_head_str_pct"] = head_pct2
            row["fighter2_avg_head_str_per_min"] = head_pm2
            row["fighter2_avg_body_str_pct"] = body_pct2
            row["fighter2_avg_body_str_per_min"] = body_pm2
            row["fighter2_avg_leg_str_pct"] = leg_pct2
            row["fighter2_avg_leg_str_per_min"] = leg_pm2
            row["fighter2_avg_distance_str_pct"] = distance_pct2
            row["fighter2_avg_distance_str_per_min"] = distance_pm2
            row["fighter2_avg_clinch_str_pct"] = clinch_pct2
            row["fighter2_avg_clinch_str_per_min"] = clinch_pm2
            row["fighter2_avg_ground_str_pct"] = ground_pct2
            row["fighter2_avg_ground_str_per_min"] = ground_pm2
            row["fighter2_head_sig_landed_pct"] = head_share2
            row["fighter2_body_sig_landed_pct"] = body_share2
            row["fighter2_leg_sig_landed_pct"] = leg_share2
            row["fighter2_distance_sig_landed_pct"] = distance_share2
            row["fighter2_clinch_sig_landed_pct"] = clinch_share2
            row["fighter2_ground_sig_landed_pct"] = ground_share2
            row["fighter2_avg_sig_str_absorbed_per_min"] = sig_absorbed_pm2
            row["fighter2_avg_sub_absorbed_per_min"] = sub_absorbed_pm2

        # Knockdown differential (fighter1 - fighter2 avg KD per min)
        try:
            kd1_s = (row.get("fighter1_avg_kd_per_min") or "").strip()
            kd2_s = (row.get("fighter2_avg_kd_per_min") or "").strip()
            if kd1_s and kd2_s:
                row["avg_kd_per_min_differential"] = str(round(float(kd1_s) - float(kd2_s), 6))
            else:
                row["avg_kd_per_min_differential"] = ""
        except (ValueError, TypeError):
            row["avg_kd_per_min_differential"] = ""

        # All other paired differentials (fighter1 - fighter2); share cols already set from rolling avg
        int_diffs = {"age_differential", "dec_wins_differential", "ko_wins_differential", "sub_wins_differential"}
        for col1, col2, diff_name in PAIRED_DIFFERENTIALS:
            round_to = None if diff_name in int_diffs else 6
            row[diff_name] = _numeric_diff(row, col1, col2, round_to=round_to)

        # Update rolling stats with this fight
        minutes = _fight_minutes(row)
        kd1 = _safe_int(row.get("fighter1_kd"))
        kd2 = _safe_int(row.get("fighter2_kd"))
        sig_l1 = _safe_int(row.get("fighter1_sig_str_landed"))
        sig_a1 = _safe_int(row.get("fighter1_sig_str_attempted"))
        sig_l2 = _safe_int(row.get("fighter2_sig_str_landed"))
        sig_a2 = _safe_int(row.get("fighter2_sig_str_attempted"))
        total_l1 = _safe_int(row.get("fighter1_total_str_landed"))
        total_a1 = _safe_int(row.get("fighter1_total_str_attempted"))
        total_l2 = _safe_int(row.get("fighter2_total_str_landed"))
        total_a2 = _safe_int(row.get("fighter2_total_str_attempted"))
        td_l1 = _safe_int(row.get("fighter1_td_landed"))
        td_a1 = _safe_int(row.get("fighter1_td_attempted"))
        td_l2 = _safe_int(row.get("fighter2_td_landed"))
        td_a2 = _safe_int(row.get("fighter2_td_attempted"))
        sub_att1 = _safe_int(row.get("fighter1_sub_att"))
        sub_att2 = _safe_int(row.get("fighter2_sub_att"))
        rev1 = _safe_int(row.get("fighter1_rev"))
        rev2 = _safe_int(row.get("fighter2_rev"))
        ctrl1 = _safe_int(row.get("fighter1_ctrl_seconds"))
        ctrl2 = _safe_int(row.get("fighter2_ctrl_seconds"))
        fight_stats[id1]["head_landed"] += _safe_int(row.get("fighter1_head_sig_str_landed"))
        fight_stats[id1]["head_attempted"] += _safe_int(row.get("fighter1_head_sig_str_attempted"))
        fight_stats[id1]["body_landed"] += _safe_int(row.get("fighter1_body_sig_str_landed"))
        fight_stats[id1]["body_attempted"] += _safe_int(row.get("fighter1_body_sig_str_attempted"))
        fight_stats[id1]["leg_landed"] += _safe_int(row.get("fighter1_leg_sig_str_landed"))
        fight_stats[id1]["leg_attempted"] += _safe_int(row.get("fighter1_leg_sig_str_attempted"))
        fight_stats[id1]["distance_landed"] += _safe_int(row.get("fighter1_distance_sig_str_landed"))
        fight_stats[id1]["distance_attempted"] += _safe_int(row.get("fighter1_distance_sig_str_attempted"))
        fight_stats[id1]["clinch_landed"] += _safe_int(row.get("fighter1_clinch_sig_str_landed"))
        fight_stats[id1]["clinch_attempted"] += _safe_int(row.get("fighter1_clinch_sig_str_attempted"))
        fight_stats[id1]["ground_landed"] += _safe_int(row.get("fighter1_ground_sig_str_landed"))
        fight_stats[id1]["ground_attempted"] += _safe_int(row.get("fighter1_ground_sig_str_attempted"))
        fight_stats[id2]["head_landed"] += _safe_int(row.get("fighter2_head_sig_str_landed"))
        fight_stats[id2]["head_attempted"] += _safe_int(row.get("fighter2_head_sig_str_attempted"))
        fight_stats[id2]["body_landed"] += _safe_int(row.get("fighter2_body_sig_str_landed"))
        fight_stats[id2]["body_attempted"] += _safe_int(row.get("fighter2_body_sig_str_attempted"))
        fight_stats[id2]["leg_landed"] += _safe_int(row.get("fighter2_leg_sig_str_landed"))
        fight_stats[id2]["leg_attempted"] += _safe_int(row.get("fighter2_leg_sig_str_attempted"))
        fight_stats[id2]["distance_landed"] += _safe_int(row.get("fighter2_distance_sig_str_landed"))
        fight_stats[id2]["distance_attempted"] += _safe_int(row.get("fighter2_distance_sig_str_attempted"))
        fight_stats[id2]["clinch_landed"] += _safe_int(row.get("fighter2_clinch_sig_str_landed"))
        fight_stats[id2]["clinch_attempted"] += _safe_int(row.get("fighter2_clinch_sig_str_attempted"))
        fight_stats[id2]["ground_landed"] += _safe_int(row.get("fighter2_ground_sig_str_landed"))
        fight_stats[id2]["ground_attempted"] += _safe_int(row.get("fighter2_ground_sig_str_attempted"))
        fight_stats[id1]["kd"] += kd1
        fight_stats[id1]["sig_landed"] += sig_l1
        fight_stats[id1]["sig_attempted"] += sig_a1
        fight_stats[id1]["total_landed"] += total_l1
        fight_stats[id1]["total_attempted"] += total_a1
        fight_stats[id1]["td_landed"] += td_l1
        fight_stats[id1]["td_attempted"] += td_a1
        fight_stats[id1]["minutes"] += minutes
        fight_stats[id2]["kd"] += kd2
        fight_stats[id2]["sig_landed"] += sig_l2
        fight_stats[id2]["sig_attempted"] += sig_a2
        fight_stats[id2]["total_landed"] += total_l2
        fight_stats[id2]["total_attempted"] += total_a2
        fight_stats[id2]["td_landed"] += td_l2
        fight_stats[id2]["td_attempted"] += td_a2
        fight_stats[id1]["sub_att"] += sub_att1
        fight_stats[id1]["rev"] += rev1
        fight_stats[id2]["sub_att"] += sub_att2
        fight_stats[id2]["rev"] += rev2
        fight_stats[id1]["ctrl_seconds"] += ctrl1
        fight_stats[id2]["ctrl_seconds"] += ctrl2
        fight_stats[id2]["minutes"] += minutes
        # Absorbed: sig strikes = opponent landed; sub absorbed = lost by submission
        fight_stats[id1]["sig_absorbed"] += sig_l2
        fight_stats[id2]["sig_absorbed"] += sig_l1
        ft = (row.get("finish_type") or "").strip().upper()
        if ft == "SUB":
            if (winner or "").strip() == f2:
                fight_stats[id1]["sub_absorbed"] += 1
            elif (winner or "").strip() == f1:
                fight_stats[id2]["sub_absorbed"] += 1

        # Update records and finish_wins
        res1, res2 = _classify_outcome(winner, f1, f2)
        key1 = "wins" if res1 == "win" else ("losses" if res1 == "loss" else "other")
        key2 = "wins" if res2 == "win" else ("losses" if res2 == "loss" else "other")
        records[id1][key1] += 1
        records[id2][key2] += 1
        if res1 == "win":
            if ft == "DEC":
                finish_wins[id1]["dec_wins"] += 1
            elif ft == "SUB":
                finish_wins[id1]["sub_wins"] += 1
            elif ft in ("KO", "TKO", "KO/TKO"):
                finish_wins[id1]["ko_wins"] += 1
        if res2 == "win":
            if ft == "DEC":
                finish_wins[id2]["dec_wins"] += 1
            elif ft == "SUB":
                finish_wins[id2]["sub_wins"] += 1
            elif ft in ("KO", "TKO", "KO/TKO"):
                finish_wins[id2]["ko_wins"] += 1
        # Update win/lose streaks (draw or NC resets both streaks for both fighters)
        if res1 == "win":
            streaks[id1]["win_streak"] += 1
            streaks[id1]["lose_streak"] = 0
            streaks[id2]["lose_streak"] += 1
            streaks[id2]["win_streak"] = 0
        elif res1 == "loss":
            streaks[id1]["lose_streak"] += 1
            streaks[id1]["win_streak"] = 0
            streaks[id2]["win_streak"] += 1
            streaks[id2]["lose_streak"] = 0
        else:
            streaks[id1]["win_streak"] = 0
            streaks[id1]["lose_streak"] = 0
            streaks[id2]["win_streak"] = 0
            streaks[id2]["lose_streak"] = 0

        # Update fighter weight history when this fight had a numeric weight (for future catchweight inference)
        wc_val = (row.get("weight_class") or "").strip()
        if wc_val and wc_val.isdigit():
            w = int(wc_val)
            fighter_weights[id1].append(w)
            fighter_weights[id2].append(w)

    # Restore original input order
    rows.sort(key=lambda r: r["_orig_idx"])

    rows = [r for r in rows if (r.get("weight_class") or "").strip() != ""]

    # Build output column list (replace raw stats with rolling avg/diff columns, reorder division/weight_class/number_of_rounds) drop requested and originals; insert replacements after their bases
    out_names = []
    for c in fieldnames:
        if c in DROP_COLUMNS:
            continue
        if c == "fighter1_kd":
            out_names.extend(KD_REPLACEMENT_COLUMNS)
            out_names.extend(KD_DIFF_COLUMN)
            continue
        if c == "fighter2_kd":
            continue
        if c == "fighter1_sig_str_landed":
            out_names.extend(SIG_STR_REPLACEMENT_COLUMNS)
            continue
        if c in (
            "fighter1_sig_str_attempted", "fighter1_sig_str_pct",
            "fighter2_sig_str_landed", "fighter2_sig_str_attempted", "fighter2_sig_str_pct",
        ):
            continue
        if c == "fighter1_total_str_landed":
            out_names.extend(TOTAL_STR_REPLACEMENT_COLUMNS)
            continue
        if c in (
            "fighter1_total_str_attempted", "fighter1_total_str_pct",
            "fighter2_total_str_landed", "fighter2_total_str_attempted", "fighter2_total_str_pct",
        ):
            continue
        if c == "fighter1_td_landed":
            out_names.extend(TD_REPLACEMENT_COLUMNS)
            continue
        if c in (
            "fighter1_td_attempted", "fighter1_td_pct",
            "fighter2_td_landed", "fighter2_td_attempted", "fighter2_td_pct",
        ):
            continue
        if c == "fighter1_sub_att":
            out_names.extend(SUB_ATT_REPLACEMENT_COLUMNS)
            continue
        if c == "fighter2_sub_att":
            continue
        if c == "fighter1_rev":
            out_names.extend(REV_REPLACEMENT_COLUMNS)
            continue
        if c == "fighter2_rev":
            continue
        if c == "fighter1_ctrl_seconds":
            out_names.extend(CTRL_REPLACEMENT_COLUMNS)
            continue
        if c == "fighter2_ctrl_seconds":
            continue
        if c == "fighter1_head_sig_str_landed":
            out_names.extend(HEAD_BODY_LEG_EXTRA_COLUMNS)
            continue
        if c in (
            "fighter1_head_sig_str_attempted", "fighter1_head_sig_str_pct",
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
        ):
            continue
        if c == "winner":
            continue
        if c in ("finish_type", "finish_technique"):
            continue  # emitted at very end as outcome columns
        if c == "total_fight_time_seconds":
            out_names.extend(TOTAL_FIGHT_MINUTES_COLUMNS)
            continue
        if c == "weight_class" or c == "number_of_rounds":
            continue  # emitted after division
        out_names.append(c)
        if c == "division":
            out_names.extend(["weight_class", "number_of_rounds"])
        if c == "fighter2_age":
            out_names.extend(AGE_DIFF_COLUMN)
        if c == "fighter2_height":
            out_names.extend(HEIGHT_DIFF_COLUMN)
        if c == "fighter2_reach":
            out_names.extend(REACH_DIFF_COLUMN)
        if c == "fighter2_stance":
            out_names.extend(DEBUT_COLUMNS)
        if c == "fighter_2":
            out_names.extend(RECORD_AND_DIFF_COLUMNS)
            out_names.extend(STREAK_AND_DIFF_COLUMNS)
            out_names.extend(FINISH_WINS_COLUMNS)
    out_names.extend(ABSORBED_COLUMNS)
    out_names.append("winner")
    for oc in ("finish_type", "finish_technique"):
        if oc in fieldnames:
            out_names.append(oc)

    if "fighter_2" not in fieldnames:
        base = [c for c in fieldnames if c not in DROP_COLUMNS and c not in DROP_ORIGINAL_STATS]
        # Move weight_class and number_of_rounds to immediately after division
        to_move = [x for x in ("weight_class", "number_of_rounds") if x in base]
        for col in to_move:
            base.remove(col)
        div_idx = next((i for i, x in enumerate(base) if x == "division"), -1)
        if div_idx >= 0 and to_move:
            base = base[: div_idx + 1] + to_move + base[div_idx + 1 :]
        idx = next((i for i, x in enumerate(base) if x == "fighter1_head_sig_str_landed"), len(base))
        replacements = []
        if "fighter1_kd" in fieldnames:
            replacements.extend(KD_REPLACEMENT_COLUMNS)
            replacements.extend(KD_DIFF_COLUMN)
        if "fighter1_sig_str_landed" in fieldnames:
            replacements.extend(SIG_STR_REPLACEMENT_COLUMNS)
        if "fighter1_total_str_landed" in fieldnames:
            replacements.extend(TOTAL_STR_REPLACEMENT_COLUMNS)
        if "fighter1_td_landed" in fieldnames:
            replacements.extend(TD_REPLACEMENT_COLUMNS)
        if "fighter1_sub_att" in fieldnames:
            replacements.extend(SUB_ATT_REPLACEMENT_COLUMNS)
        if "fighter1_rev" in fieldnames:
            replacements.extend(REV_REPLACEMENT_COLUMNS)
        if "fighter1_ctrl_seconds" in fieldnames:
            replacements.extend(CTRL_REPLACEMENT_COLUMNS)
        if replacements:
            base = base[:idx] + replacements + base[idx:]
        # Insert reach diff after fighter2_reach, debut after fighter2_stance
        reach_idx = next((i for i, x in enumerate(base) if x == "fighter2_reach"), -1)
        if reach_idx >= 0:
            base = base[: reach_idx + 1] + REACH_DIFF_COLUMN + base[reach_idx + 1 :]
        stance_idx = next((i for i, x in enumerate(base) if x == "fighter2_stance"), -1)
        if stance_idx >= 0:
            base = base[: stance_idx + 1] + DEBUT_COLUMNS + base[stance_idx + 1 :]
        # Insert total fight minutes (replaces total_fight_time_seconds) after debut columns
        debut_end = next((i for i, x in enumerate(base) if x == "fighter2_is_debut"), -1)
        if debut_end >= 0:
            base = base[: debut_end + 1] + TOTAL_FIGHT_MINUTES_COLUMNS + base[debut_end + 1 :]
        age_idx = next((i for i, x in enumerate(base) if x == "fighter2_age"), -1)
        if age_idx >= 0:
            base = base[: age_idx + 1] + AGE_DIFF_COLUMN + base[age_idx + 1 :]
        height_idx = next((i for i, x in enumerate(base) if x == "fighter2_height"), -1)
        if height_idx >= 0:
            base = base[: height_idx + 1] + HEIGHT_DIFF_COLUMN + base[height_idx + 1 :]
        # Insert head/body/leg replacement columns where the block was (before distance_sig_str)
        dist_idx = next((i for i, x in enumerate(base) if x == "fighter1_distance_sig_str_landed"), -1)
        if dist_idx >= 0:
            base = base[:dist_idx] + HEAD_BODY_LEG_EXTRA_COLUMNS + base[dist_idx:]
        out_names = base + RECORD_AND_DIFF_COLUMNS + STREAK_AND_DIFF_COLUMNS + FINISH_WINS_COLUMNS + ABSORBED_COLUMNS
        if "winner" in base:
            out_names = [x for x in out_names if x != "winner"] + ["winner"]
        outcome_cols = [x for x in ("finish_type", "finish_technique") if x in fieldnames]
        out_names = [x for x in out_names if x not in outcome_cols] + outcome_cols

    # Final filter: non-blank winner (year filtering done in module_05)
    rows = [r for r in rows if (r.get("winner") or "").strip() != ""]

    # Drop internal and replaced columns, then write output and module_05 input (local and/or Azure)
    for row in rows:
        for col in DROP_COLUMNS | DROP_ORIGINAL_STATS | {"_orig_idx"}:
            row.pop(col, None)

    _write_outputs(
        records=rows,
        columns=out_names,
        local_csv_path=args.output,
        storage=storage,
        blob_csv_path=args.blob_csv,
        blob_parquet_path=args.blob_parquet,
    )


if __name__ == "__main__":
    main()
