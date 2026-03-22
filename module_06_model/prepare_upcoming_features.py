"""
Prepare upcoming fights for prediction. Output of module 6, input of module 7.

Loads upcoming fights from local CSV (storage=local) or Azure Blob Parquet (storage=azure/both),
computes fighter state, builds feature rows.
Saves to module_07_predict/input/upcoming_for_prediction.joblib
"""

import csv
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULE_04 = _PROJECT_ROOT / "module_04_feature_engineering"
_MODULE_01 = _PROJECT_ROOT / "module_01_scrapers"
_MODULE_02 = _PROJECT_ROOT / "module_02_clean_fighters"
MODULE_07_INPUT = _PROJECT_ROOT / "module_07_predict" / "input"

sys.path.insert(0, str(_MODULE_04))
from feature_engineering import (
    PAIRED_DIFFERENTIALS,
    _numeric_diff,
    _parse_event_date,
    _rolling_stat,
    _weight_class_to_lbs,
    compute_fighter_state_snapshot,
)

DEFAULT_UPCOMING = _MODULE_01 / "output" / "raw_ufc_upcoming.csv"
# Same blob path as module_01_scrapers/ufc_upcoming_scraper.py (Parquet + CSV)
UPCOMING_BLOB_PARQUET = "module_01_scrapers/output/raw_ufc_upcoming.parquet"
DEFAULT_CLEAN_FIGHTS = _MODULE_04 / "input" / "clean_ufc_fights.csv"
DEFAULT_CLEAN_FIGHTERS = _MODULE_02 / "output" / "clean_ufc_fighters.csv"


def _load_upcoming_rows_from_local_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            r
            for r in csv.DictReader(f)
            if (r.get("fighter_1") or "").strip() and (r.get("fighter_2") or "").strip()
        ]


def _rows_from_parquet_dataframe(df) -> list[dict]:
    """Normalize Parquet rows to string dicts like CSV DictReader."""
    import pandas as pd

    df = df.fillna("")
    out = []
    for rec in df.to_dict("records"):
        row = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                row[k] = ""
            elif isinstance(v, str):
                row[k] = v.strip()
            else:
                row[k] = str(v).strip()
        out.append(row)
    return [
        r
        for r in out
        if (r.get("fighter_1") or "").strip() and (r.get("fighter_2") or "").strip()
    ]


def _load_upcoming_rows(storage: str, upcoming_path: Path) -> tuple[list[dict], str]:
    """
    Load upcoming fight rows. local = disk CSV only; azure = Azure Parquet only;
    both = Parquet first, then local CSV on failure.
    """
    storage = (storage or "local").strip().lower()
    if storage == "local":
        if not upcoming_path.exists():
            return [], "local CSV missing"
        rows = _load_upcoming_rows_from_local_csv(upcoming_path)
        return rows, str(upcoming_path)

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from module_00_utils.azure_storage import read_parquet_from_azure

    if storage == "azure":
        df = read_parquet_from_azure(UPCOMING_BLOB_PARQUET)
        rows = _rows_from_parquet_dataframe(df)
        return rows, f"azure:{UPCOMING_BLOB_PARQUET}"

    # both
    try:
        df = read_parquet_from_azure(UPCOMING_BLOB_PARQUET)
        rows = _rows_from_parquet_dataframe(df)
        return rows, f"azure:{UPCOMING_BLOB_PARQUET}"
    except Exception as e:
        print(f"Could not read upcoming Parquet from Azure ({e}); falling back to local CSV.")
        if not upcoming_path.exists():
            return [], "azure failed and local CSV missing"
        rows = _load_upcoming_rows_from_local_csv(upcoming_path)
        return rows, f"local fallback {upcoming_path}"


def _event_date_to_calendar_date(val) -> date | None:
    """
    Parse event_date from CSV (e.g. 'March 15, 2026') or Parquet (datetime / '2026-03-15' / ISO).

    Parquet often serializes dates as ISO strings; _parse_event_date only handles month-name
    formats, so past events were incorrectly kept when loading from Azure.
    """
    if val is None or val == "":
        return None
    try:
        import pandas as pd

        if isinstance(val, pd.Timestamp):
            return val.date()
    except ImportError:
        pass
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    s = str(val).strip()
    if not s:
        return None
    dt = _parse_event_date(s)
    if dt is not None:
        return dt.date()
    # ISO / Parquet string forms (e.g. 2026-03-15, 2026-03-15 00:00:00)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    try:
        s_iso = s.replace("Z", "+00:00", 1) if s.endswith("Z") else s
        return datetime.fromisoformat(s_iso).date()
    except ValueError:
        pass
    return None


def _filter_past_events(upcoming: list[dict]) -> tuple[list[dict], int]:
    """Drop rows whose event_date parses and is before today (calendar)."""
    today = date.today()
    skipped = 0
    kept = []
    for r in upcoming:
        ed = r.get("event_date")
        ev_day = _event_date_to_calendar_date(ed)
        if ev_day is not None and ev_day < today:
            skipped += 1
            continue
        kept.append(r)
    return kept, skipped


def _load_completed_event_urls(clean_fights_path: Path) -> set[str]:
    """Event URLs that already have results in clean_ufc_fights (card completed / in history)."""
    urls: set[str] = set()
    if not clean_fights_path.exists():
        return urls
    with open(clean_fights_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("event_url") or "").strip()
            if u:
                urls.add(u)
    return urls


def _filter_already_completed_events(
    upcoming: list[dict],
    completed_event_urls: set[str],
) -> tuple[list[dict], int]:
    """
    Drop upcoming rows whose event already appears in clean fight history.

    UFCStats sometimes keeps finished cards on /events/upcoming with a stale scheduled date;
    calendar filtering alone cannot remove those rows.
    """
    if not completed_event_urls:
        return upcoming, 0
    skipped = 0
    kept: list[dict] = []
    for r in upcoming:
        eu = (r.get("event_url") or "").strip()
        if eu and eu in completed_event_urls:
            skipped += 1
            continue
        kept.append(r)
    return kept, skipped


def filter_joblib_pairs_for_future_events(
    feature_rows: list,
    fight_metadata: list[dict],
    *,
    clean_fights_path: Path | None = None,
) -> tuple[list, list[dict], int, int]:
    """
    Drop past dates and events already in clean_ufc_fights (by event_url).
    Returns (feature_rows, fight_metadata, skipped_past_date, skipped_completed).
    """
    today = date.today()
    skipped_past = 0
    skipped_done = 0
    kept_f: list = []
    kept_m: list[dict] = []
    cfp = clean_fights_path if clean_fights_path is not None else DEFAULT_CLEAN_FIGHTS
    completed_urls = _load_completed_event_urls(cfp)

    for fr, fm in zip(feature_rows, fight_metadata):
        ev_day = _event_date_to_calendar_date(fm.get("event_date"))
        if ev_day is not None and ev_day < today:
            skipped_past += 1
            continue
        eu = (fm.get("event_url") or "").strip()
        if eu and eu in completed_urls:
            skipped_done += 1
            continue
        kept_f.append(fr)
        kept_m.append(fm)
    return kept_f, kept_m, skipped_past, skipped_done


def _weight_class_to_lbs_upcoming(wc: str) -> str:
    wc = (wc or "").strip().replace("Women's ", "").strip()
    return _weight_class_to_lbs(wc)


def _division_from_weight_class(wc: str) -> str:
    return "Women" if (wc or "").strip().startswith("Women's ") else "Men"


def _parse_dob(s: str):
    """Parse date of birth string to datetime."""
    from datetime import datetime
    s = (s or "").strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _age_at_date(dob, event_date) -> int | str:
    if dob is None or event_date is None:
        return ""
    delta = event_date - dob
    return int(delta.days / 365.25)


def load_fighters_by_url(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("fighter_url") or "").strip()
            if not url:
                continue
            url_norm = url.rstrip("/")
            out[url_norm] = {
                "height": (row.get("height") or "").strip(),
                "reach": (row.get("reach") or "").strip(),
                "stance": (row.get("stance") or "Unknown").strip() or "Unknown",
                "date_of_birth": (row.get("date_of_birth") or "").strip(),
                "full_name": (row.get("full_name") or "").strip(),
            }
            if url != url_norm:
                out[url] = out[url_norm]
    return out


def build_feature_row(fight, state1, state2, attrs1, attrs2, event_date) -> dict:
    """Build one feature row for an upcoming fight (differential-only schema)."""
    r1 = state1 or {}
    r2 = state2 or {}
    rec1, rec2 = r1.get("records", {}), r2.get("records", {})
    str1, str2 = r1.get("streaks", {}), r2.get("streaks", {})
    fw1, fw2 = r1.get("finish_wins", {}), r2.get("finish_wins", {})
    fs1, fs2 = r1.get("fight_stats", {}), r2.get("fight_stats", {})

    def empty_stats():
        return {"kd": 0.0, "sig_landed": 0.0, "sig_attempted": 0.0, "sig_absorbed": 0.0,
                "total_landed": 0.0, "total_attempted": 0.0, "td_landed": 0.0, "td_attempted": 0.0,
                "sub_att": 0.0, "sub_absorbed": 0.0, "rev": 0.0, "ctrl_seconds": 0.0,
                "head_landed": 0.0, "head_attempted": 0.0, "body_landed": 0.0, "body_attempted": 0.0,
                "leg_landed": 0.0, "leg_attempted": 0.0, "distance_landed": 0.0, "distance_attempted": 0.0,
                "clinch_landed": 0.0, "clinch_attempted": 0.0, "ground_landed": 0.0, "ground_attempted": 0.0,
                "minutes": 0.0}
    s1, s2 = fs1 or empty_stats(), fs2 or empty_stats()
    n1 = rec1.get("wins", 0) + rec1.get("losses", 0) + rec1.get("other", 0)
    n2 = rec2.get("wins", 0) + rec2.get("losses", 0) + rec2.get("other", 0)

    row = {}
    row["fighter1_total_wins"] = rec1.get("wins", 0)
    row["fighter2_total_wins"] = rec2.get("wins", 0)
    row["win_differential"] = row["fighter1_total_wins"] - row["fighter2_total_wins"]
    row["fighter1_total_losses"] = rec1.get("losses", 0)
    row["fighter2_total_losses"] = rec2.get("losses", 0)
    row["loss_differential"] = row["fighter1_total_losses"] - row["fighter2_total_losses"]
    row["fighter1_win_streak"] = str1.get("win_streak", 0)
    row["fighter2_win_streak"] = str2.get("win_streak", 0)
    row["win_streak_differential"] = row["fighter1_win_streak"] - row["fighter2_win_streak"]
    row["fighter1_lose_streak"] = str1.get("lose_streak", 0)
    row["fighter2_lose_streak"] = str2.get("lose_streak", 0)
    row["lose_streak_differential"] = row["fighter1_lose_streak"] - row["fighter2_lose_streak"]
    row["fighter1_dec_wins"] = fw1.get("dec_wins", 0)
    row["fighter2_dec_wins"] = fw2.get("dec_wins", 0)
    row["dec_wins_differential"] = row["fighter1_dec_wins"] - row["fighter2_dec_wins"]
    row["fighter1_ko_wins"] = fw1.get("ko_wins", 0)
    row["fighter2_ko_wins"] = fw2.get("ko_wins", 0)
    row["ko_wins_differential"] = row["fighter1_ko_wins"] - row["fighter2_ko_wins"]
    row["fighter1_sub_wins"] = fw1.get("sub_wins", 0)
    row["fighter2_sub_wins"] = fw2.get("sub_wins", 0)
    row["sub_wins_differential"] = row["fighter1_sub_wins"] - row["fighter2_sub_wins"]

    age1 = _age_at_date(_parse_dob(attrs1.get("date_of_birth")), event_date)
    age2 = _age_at_date(_parse_dob(attrs2.get("date_of_birth")), event_date)
    row["fighter1_age"] = age1 if age1 != "" else ""
    row["fighter2_age"] = age2 if age2 != "" else ""
    row["age_differential"] = (age1 - age2) if isinstance(age1, int) and isinstance(age2, int) else ""

    h1 = int(attrs1.get("height")) if attrs1.get("height") and str(attrs1.get("height")).isdigit() else ""
    h2 = int(attrs2.get("height")) if attrs2.get("height") and str(attrs2.get("height")).isdigit() else ""
    rch1 = int(attrs1.get("reach")) if attrs1.get("reach") and str(attrs1.get("reach")).isdigit() else ""
    rch2 = int(attrs2.get("reach")) if attrs2.get("reach") and str(attrs2.get("reach")).isdigit() else ""
    row["fighter1_height"] = h1
    row["fighter2_height"] = h2
    row["height_differential"] = (h1 - h2) if h1 != "" and h2 != "" else ""
    row["fighter1_reach"] = rch1
    row["fighter2_reach"] = rch2
    row["reach_differential"] = (rch1 - rch2) if rch1 != "" and rch2 != "" else ""
    row["fighter1_stance"] = attrs1.get("stance", "Unknown")
    row["fighter2_stance"] = attrs2.get("stance", "Unknown")
    row["fighter1_is_debut"] = 1 if n1 == 0 else 0
    row["fighter2_is_debut"] = 1 if n2 == 0 else 0
    min1, min2 = r1.get("total_fight_minutes", 0), r2.get("total_fight_minutes", 0)
    row["fighter1_total_fight_minutes"] = round(min1, 6)
    row["fighter2_total_fight_minutes"] = round(min2, 6)
    row["total_fight_minutes_differential"] = round(min1 - min2, 6) if min1 or min2 else ""

    ROLLING_COLS = [
        ("fighter1_avg_sig_str_pct", "fighter2_avg_sig_str_pct"),
        ("fighter1_avg_sig_str_per_min", "fighter2_avg_sig_str_per_min"),
        ("fighter1_avg_kd_per_min", "fighter2_avg_kd_per_min"),
        ("fighter1_avg_total_str_pct", "fighter2_avg_total_str_pct"),
        ("fighter1_avg_total_str_per_min", "fighter2_avg_total_str_per_min"),
        ("fighter1_avg_td_pct", "fighter2_avg_td_pct"),
        ("fighter1_avg_td_per_min", "fighter2_avg_td_per_min"),
        ("fighter1_avg_sub_att_per_min", "fighter2_avg_sub_att_per_min"),
        ("fighter1_avg_rev_per_min", "fighter2_avg_rev_per_min"),
        ("fighter1_avg_ctrl_seconds_per_min", "fighter2_avg_ctrl_seconds_per_min"),
        ("fighter1_avg_head_str_pct", "fighter2_avg_head_str_pct"),
        ("fighter1_avg_head_str_per_min", "fighter2_avg_head_str_per_min"),
        ("fighter1_avg_body_str_pct", "fighter2_avg_body_str_pct"),
        ("fighter1_avg_body_str_per_min", "fighter2_avg_body_str_per_min"),
        ("fighter1_avg_leg_str_pct", "fighter2_avg_leg_str_pct"),
        ("fighter1_avg_leg_str_per_min", "fighter2_avg_leg_str_per_min"),
        ("fighter1_avg_distance_str_pct", "fighter2_avg_distance_str_pct"),
        ("fighter1_avg_distance_str_per_min", "fighter2_avg_distance_str_per_min"),
        ("fighter1_avg_clinch_str_pct", "fighter2_avg_clinch_str_pct"),
        ("fighter1_avg_clinch_str_per_min", "fighter2_avg_clinch_str_per_min"),
        ("fighter1_avg_ground_str_pct", "fighter2_avg_ground_str_pct"),
        ("fighter1_avg_ground_str_per_min", "fighter2_avg_ground_str_per_min"),
        ("fighter1_avg_sig_str_absorbed_per_min", "fighter2_avg_sig_str_absorbed_per_min"),
        ("fighter1_avg_sub_absorbed_per_min", "fighter2_avg_sub_absorbed_per_min"),
        ("fighter1_head_sig_landed_pct", "fighter2_head_sig_landed_pct"),
        ("fighter1_body_sig_landed_pct", "fighter2_body_sig_landed_pct"),
        ("fighter1_leg_sig_landed_pct", "fighter2_leg_sig_landed_pct"),
        ("fighter1_distance_sig_landed_pct", "fighter2_distance_sig_landed_pct"),
        ("fighter1_clinch_sig_landed_pct", "fighter2_clinch_sig_landed_pct"),
        ("fighter1_ground_sig_landed_pct", "fighter2_ground_sig_landed_pct"),
    ]
    r1_vals, r2_vals = _rolling_stat(s1), _rolling_stat(s2)
    for idx, (col1, col2) in enumerate(ROLLING_COLS):
        row[col1] = r1_vals[idx] if idx < len(r1_vals) else ""
        row[col2] = r2_vals[idx] if idx < len(r2_vals) else ""
    try:
        kd1_s, kd2_s = row.get("fighter1_avg_kd_per_min", ""), row.get("fighter2_avg_kd_per_min", "")
        row["avg_kd_per_min_differential"] = str(round(float(kd1_s) - float(kd2_s), 6)) if kd1_s and kd2_s else ""
    except (ValueError, TypeError):
        row["avg_kd_per_min_differential"] = ""
    int_diffs = {"age_differential", "dec_wins_differential", "ko_wins_differential", "sub_wins_differential"}
    for col1, col2, diff_name in PAIRED_DIFFERENTIALS:
        if diff_name in row:
            continue
        row[diff_name] = _numeric_diff(row, col1, col2, round_to=None if diff_name in int_diffs else 6)
    wc_raw = (fight.get("weight_class") or "").strip()
    row["weight_class"] = _weight_class_to_lbs_upcoming(wc_raw) or ""
    row["number_of_rounds"] = (fight.get("number_of_rounds") or "3").strip()
    row["division"] = _division_from_weight_class(wc_raw)
    return row


def main(upcoming_path=None, clean_fights_path=None, clean_fighters_path=None, output_dir=None, storage: str = "local"):
    import joblib

    upcoming_path = upcoming_path or DEFAULT_UPCOMING
    clean_fights_path = clean_fights_path or DEFAULT_CLEAN_FIGHTS
    clean_fighters_path = clean_fighters_path or DEFAULT_CLEAN_FIGHTERS
    output_dir = output_dir or MODULE_07_INPUT
    storage = (storage or "local").strip().lower()

    if storage == "local" and not upcoming_path.exists():
        print(f"Upcoming fights not found: {upcoming_path}. Skipping prepare.")
        return
    if not clean_fights_path.exists() or not clean_fighters_path.exists():
        print("Clean fights/fighters not found. Skipping prepare.")
        return

    # Load preprocessor to get feature column order
    preproc_path = Path(__file__).resolve().parent / "input" / "preprocessor_diff.joblib"
    if not preproc_path.exists():
        preproc_path = Path(__file__).resolve().parent / "output" / "preprocessor_diff.joblib"
    if not preproc_path.exists():
        print("Preprocessor not found. Run prep_and_split -d first. Skipping prepare.")
        return

    bundle = joblib.load(preproc_path)
    feature_cols = bundle["numeric_cols"] + bundle["cat_cols"]

    state_snapshot = compute_fighter_state_snapshot(input_path=clean_fights_path)
    fighters_by_url = load_fighters_by_url(clean_fighters_path)

    def get_attrs(url, name):
        url_norm = (url or "").strip().rstrip("/")
        attrs = fighters_by_url.get(url_norm) or fighters_by_url.get(url_norm + "/") or {}
        if not attrs:
            attrs = next((v for v in fighters_by_url.values() if v.get("full_name") == name), {})
        return attrs or {"height": "", "reach": "", "stance": "Unknown", "date_of_birth": "", "full_name": name}

    def get_state(url, name):
        url_norm = (url or "").strip().rstrip("/")
        return state_snapshot.get(url_norm) or state_snapshot.get(url_norm + "/") or state_snapshot.get(name)

    upcoming, upcoming_src = _load_upcoming_rows(storage, upcoming_path)
    if not upcoming:
        print(f"No upcoming fights loaded (source={upcoming_src}). Skipping prepare.")
        return
    print(f"Loaded {len(upcoming)} upcoming row(s) from {upcoming_src}.")

    upcoming, skipped_past = _filter_past_events(upcoming)
    if skipped_past:
        print(f"Excluded {skipped_past} row(s) with event date before {date.today().isoformat()}.")

    completed_urls = _load_completed_event_urls(clean_fights_path)
    upcoming, skipped_done = _filter_already_completed_events(upcoming, completed_urls)
    if skipped_done:
        print(
            f"Excluded {skipped_done} row(s) whose event_url already appears in clean fight history "
            f"(card completed; UFCStats may still list it as upcoming)."
        )

    if not upcoming:
        print("No upcoming fights after date / completed-event filters. Skipping prepare.")
        return

    event_date_cache = {}
    feature_rows = []
    fight_metadata = []
    for fight in upcoming:
        url1 = (fight.get("fighter_1_url") or "").strip()
        url2 = (fight.get("fighter_2_url") or "").strip()
        name1 = (fight.get("fighter_1") or "").strip()
        name2 = (fight.get("fighter_2") or "").strip()
        ed_str = (fight.get("event_date") or "").strip()
        event_dt = event_date_cache.get(ed_str) or _parse_event_date(ed_str)
        if ed_str:
            event_date_cache[ed_str] = event_dt
        state1, state2 = get_state(url1, name1), get_state(url2, name2)
        attrs1, attrs2 = get_attrs(url1, name1), get_attrs(url2, name2)
        row = build_feature_row(fight, state1, state2, attrs1, attrs2, event_dt)
        feature_rows.append(row)
        fight_metadata.append({
            "event_url": (fight.get("event_url") or "").strip(),
            "event_name": fight.get("event_name", ""),
            "event_date": fight.get("event_date", ""),
            "location": fight.get("event_location", ""),
            "fighter_1": name1,
            "fighter_2": name2,
            "weight_class": fight.get("weight_class", ""),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "upcoming_for_prediction.joblib"
    joblib.dump({
        "feature_rows": feature_rows,
        "fight_metadata": fight_metadata,
        "feature_cols": feature_cols,
    }, out_path)
    print(f"Prepared {len(upcoming)} fights -> {out_path}")


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(description="Prepare upcoming fights for module 7")
    _p.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="local = raw_ufc_upcoming.csv only; azure = Azure Parquet only; both = Parquet then CSV fallback",
    )
    _args = _p.parse_args()
    main(storage=_args.storage)
