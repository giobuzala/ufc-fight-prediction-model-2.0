"""
UFC Fight Data Scraper
Scrapes fight data from UFCStats.com and saves to CSV.
UFCStats has rich fight-level stats (strikes, takedowns, etc.) ideal for prediction models.

Modes:
- Full: scrape_all_fights() — all events (use for initial load or rare full refresh).
- Incremental: scrape_new_fights_only() — only events after the latest in existing CSV, then append (newest first).

Both full and incremental scrapes always include fighter_1_url and fighter_2_url (from event page links)
so downstream joins can correctly disambiguate duplicate names (e.g. two "Jean Silva"s).
"""

import argparse
import csv
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_CSV_PATH = _SCRIPT_DIR / "output" / "raw_ufc_fights.csv"
DEFAULT_BLOB_CSV_PATH = "module_01_scrapers/output/raw_ufc_fights.csv"
DEFAULT_BLOB_PARQUET_PATH = "module_01_scrapers/output/raw_ufc_fights.parquet"
# Blob paths for module_03 input so the container has them when running module 3 from cloud.
MODULE_03_INPUT_BLOB_CSV = "module_03_clean_fights/input/raw_ufc_fights.csv"
MODULE_03_INPUT_BLOB_PARQUET = "module_03_clean_fights/input/raw_ufc_fights.parquet"

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# CSV column order for output (fight info + per-fight stats)
CSV_FIELDNAMES = [
    "event_url", "event_name", "event_date", "event_location", "referee",
    "fighter_1", "fighter_2", "fighter_1_url", "fighter_2_url", "winner",
    "weight_class", "method", "round", "time", "number_of_rounds",
    "fighter1_kd", "fighter2_kd",
    "fighter1_sig_str", "fighter2_sig_str",
    "fighter1_total_str", "fighter2_total_str",
    "fighter1_td", "fighter2_td",
    "fighter1_sub_att", "fighter2_sub_att",
    "fighter1_rev", "fighter2_rev",
    "fighter1_ctrl", "fighter2_ctrl",
    "fighter1_head_sig_str", "fighter2_head_sig_str",
    "fighter1_body_sig_str", "fighter2_body_sig_str",
    "fighter1_leg_sig_str", "fighter2_leg_sig_str",
    "fighter1_distance_sig_str", "fighter2_distance_sig_str",
    "fighter1_clinch_sig_str", "fighter2_clinch_sig_str",
    "fighter1_ground_sig_str", "fighter2_ground_sig_str",
]


def _parse_of_y(s: str) -> tuple[int, int]:
    """Parse 'X of Y' format. Returns (landed, attempted) or (0, 0)."""
    m = re.search(r"(\d+)\s+of\s+(\d+)", str(s))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _format_of_y(landed: int, attempted: int) -> str:
    return f"{landed} of {attempted}"


def _parse_ctrl(s: str) -> int:
    """Parse 'M:SS' to total seconds."""
    m = re.search(r"(\d+):(\d+)", str(s))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def _format_ctrl(seconds: int) -> str:
    if seconds < 0:
        return "0:00"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _safe_int(s: str) -> int:
    try:
        return int(re.search(r"\d+", str(s) or "").group(0)) if re.search(r"\d+", str(s) or "") else 0
    except (ValueError, AttributeError):
        return 0


def _parse_event_date(s: str) -> datetime | None:
    """Parse event_date string like 'February 07, 2026' or 'January 31, 2026' for comparison."""
    if not s or not s.strip():
        return None
    s = s.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):  # Full month and abbreviated
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def get_latest_event_date_from_csv(path: str | Path) -> str | None:
    """
    Read existing CSV (ordered descending by date) and return the event_date of the
    first data row (most recent), or None if file missing/empty/no data rows.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        row = next(reader, None)
        return row.get("event_date", "").strip() or None if row else None


# Output helpers (local CSV; Azure CSV + Parquet when storage is azure/both)


def _records_to_df(records: list[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[columns]
    return df.fillna("")


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
        # Always keep module_03 input in sync so it has raw_ufc_fights.csv.
        module_03_input = _PROJECT_ROOT / "module_03_clean_fights" / "input" / "raw_ufc_fights.csv"
        module_03_input.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_csv_path, module_03_input)
        print(f"Updated module_03 input: {module_03_input}")

    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import write_csv_to_azure, write_parquet_to_azure

            write_csv_to_azure(df, blob_csv_path, index=False)
            write_parquet_to_azure(df, blob_parquet_path, index=False)
            print(f"Wrote {len(df)} rows to Azure blobs: {blob_csv_path} and {blob_parquet_path}")
            # Always keep module_03 input in container so it has raw_ufc_fights.
            write_csv_to_azure(df, MODULE_03_INPUT_BLOB_CSV, index=False)
            write_parquet_to_azure(df, MODULE_03_INPUT_BLOB_PARQUET, index=False)
            print(f"Updated module_03 input in container: {MODULE_03_INPUT_BLOB_CSV} and {MODULE_03_INPUT_BLOB_PARQUET}")
        except Exception as e:
            print(f"Failed to write to Azure: {e}")
            raise


# Scraping


def get_events_list(session: requests.Session) -> list[dict]:
    """Fetch all past events from UFCStats."""
    resp = session.get(EVENTS_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    for row in soup.find_all("tr"):
        link = row.find("a", href=re.compile(r"event-details"))
        if not link:
            continue
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name:
            continue
        cells = row.find_all("td")
        first_cell_text = cells[0].get_text(separator=" ", strip=True) if cells else ""
        date_match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}",
            first_cell_text,
            re.IGNORECASE,
        )
        date = date_match.group(0) if date_match else ""
        location = cells[1].get_text(strip=True) if len(cells) >= 2 else ""

        events.append(
            {
                "event_url": urljoin(BASE_URL, href),
                "event_name": name,
                "event_date": date,
                "event_location": location,
            }
        )

    return events


def get_fight_details(session: requests.Session, fight_url: str) -> dict:
    """Fetch per-fight stats from fight-details page (TOTALS, SIGNIFICANT STRIKES).
    Returns only: KD, Sig.Str, Total Str, TD, Sub Att, Rev, CTRL, Head, Body, Leg, Distance, Clinch, Ground.
    Formats: Sig/Total/TD/Head/Body/Leg/Distance/Clinch/Ground as "X of Y", KD/Sub Att/Rev as int, CTRL as "M:SS".
    """
    empty = {
        "referee": "",
        "number_of_rounds": "",
        "fighter1_kd": "", "fighter2_kd": "",
        "fighter1_sig_str": "", "fighter2_sig_str": "",
        "fighter1_total_str": "", "fighter2_total_str": "",
        "fighter1_td": "", "fighter2_td": "",
        "fighter1_sub_att": "", "fighter2_sub_att": "",
        "fighter1_rev": "", "fighter2_rev": "",
        "fighter1_ctrl": "", "fighter2_ctrl": "",
        "fighter1_head_sig_str": "", "fighter2_head_sig_str": "",
        "fighter1_body_sig_str": "", "fighter2_body_sig_str": "",
        "fighter1_leg_sig_str": "", "fighter2_leg_sig_str": "",
        "fighter1_distance_sig_str": "", "fighter2_distance_sig_str": "",
        "fighter1_clinch_sig_str": "", "fighter2_clinch_sig_str": "",
        "fighter1_ground_sig_str": "", "fighter2_ground_sig_str": "",
    }
    try:
        resp = session.get(fight_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        tables = soup.find_all("table", class_="b-fight-details__table")
        if len(tables) < 2:
            return empty

        # Referee: "Referee: Jason Herzog"
        for i_tag in soup.find_all("i"):
            t = i_tag.get_text(strip=True)
            if t.startswith("Referee:"):
                empty["referee"] = t.replace("Referee:", "", 1).strip()
                break
        # TIME FORMAT: "Time format:5 Rnd (5-5-5-5-5)" or "Time format:3 Rnd (5-5-5)" -> number of rounds
        for i_tag in soup.find_all("i"):
            t = i_tag.get_text(strip=True)
            if "Time format" in t and "Rnd" in t:
                m = re.search(r"(\d+)\s*Rnd", t)
                if m:
                    empty["number_of_rounds"] = m.group(1)
                break

        # TOTALS table: sum all per-round rows to get fight totals
        totals_rows = tables[0].find_all("tr")
        data_rows = [r for r in totals_rows if r.find("p", class_="b-fight-details__table-text")]
        kd1 = kd2 = sub1 = sub2 = rev1 = rev2 = 0
        sig1_l, sig1_a = 0, 0
        sig2_l, sig2_a = 0, 0
        tot1_l, tot1_a = 0, 0
        tot2_l, tot2_a = 0, 0
        td1_l, td1_a = 0, 0
        td2_l, td2_a = 0, 0
        ctrl1 = ctrl2 = 0
        for totals_row in data_rows:
            pts = totals_row.find_all("p", class_="b-fight-details__table-text")
            if len(pts) >= 20:
                kd1 += _safe_int(pts[2].get_text(strip=True))
                kd2 += _safe_int(pts[3].get_text(strip=True))
                l1, a1 = _parse_of_y(pts[4].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[5].get_text(strip=True))
                sig1_l += l1
                sig1_a += a1
                sig2_l += l2
                sig2_a += a2
                l1, a1 = _parse_of_y(pts[8].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[9].get_text(strip=True))
                tot1_l += l1
                tot1_a += a1
                tot2_l += l2
                tot2_a += a2
                l1, a1 = _parse_of_y(pts[10].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[11].get_text(strip=True))
                td1_l += l1
                td1_a += a1
                td2_l += l2
                td2_a += a2
                sub1 += _safe_int(pts[14].get_text(strip=True))
                sub2 += _safe_int(pts[15].get_text(strip=True))
                rev1 += _safe_int(pts[16].get_text(strip=True))
                rev2 += _safe_int(pts[17].get_text(strip=True))
                ctrl1 += _parse_ctrl(pts[18].get_text(strip=True))
                ctrl2 += _parse_ctrl(pts[19].get_text(strip=True))
        empty["fighter1_kd"] = str(kd1)
        empty["fighter2_kd"] = str(kd2)
        empty["fighter1_sig_str"] = _format_of_y(sig1_l, sig1_a)
        empty["fighter2_sig_str"] = _format_of_y(sig2_l, sig2_a)
        empty["fighter1_total_str"] = _format_of_y(tot1_l, tot1_a)
        empty["fighter2_total_str"] = _format_of_y(tot2_l, tot2_a)
        empty["fighter1_td"] = _format_of_y(td1_l, td1_a)
        empty["fighter2_td"] = _format_of_y(td2_l, td2_a)
        empty["fighter1_sub_att"] = str(sub1)
        empty["fighter2_sub_att"] = str(sub2)
        empty["fighter1_rev"] = str(rev1)
        empty["fighter2_rev"] = str(rev2)
        empty["fighter1_ctrl"] = _format_ctrl(ctrl1)
        empty["fighter2_ctrl"] = _format_ctrl(ctrl2)

        # SIGNIFICANT STRIKES table: sum all per-round rows for Head, Body, Leg, Distance, Clinch, Ground
        sig_rows = tables[1].find_all("tr")
        sig_data_rows = [r for r in sig_rows if r.find("p", class_="b-fight-details__table-text")]
        h1, ha1, h2, ha2 = 0, 0, 0, 0
        b1, ba1, b2, ba2 = 0, 0, 0, 0
        leg1, lega1, leg2, lega2 = 0, 0, 0, 0
        d1, da1, d2, da2 = 0, 0, 0, 0
        c1, ca1, c2, ca2 = 0, 0, 0, 0
        g1, ga1, g2, ga2 = 0, 0, 0, 0
        for sig_row in sig_data_rows:
            pts = sig_row.find_all("p", class_="b-fight-details__table-text")
            if len(pts) >= 18:
                l1, a1 = _parse_of_y(pts[6].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[7].get_text(strip=True))
                h1 += l1
                ha1 += a1
                h2 += l2
                ha2 += a2
                l1, a1 = _parse_of_y(pts[8].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[9].get_text(strip=True))
                b1 += l1
                ba1 += a1
                b2 += l2
                ba2 += a2
                l1, a1 = _parse_of_y(pts[10].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[11].get_text(strip=True))
                leg1 += l1
                lega1 += a1
                leg2 += l2
                lega2 += a2
                l1, a1 = _parse_of_y(pts[12].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[13].get_text(strip=True))
                d1 += l1
                da1 += a1
                d2 += l2
                da2 += a2
                l1, a1 = _parse_of_y(pts[14].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[15].get_text(strip=True))
                c1 += l1
                ca1 += a1
                c2 += l2
                ca2 += a2
                l1, a1 = _parse_of_y(pts[16].get_text(strip=True))
                l2, a2 = _parse_of_y(pts[17].get_text(strip=True))
                g1 += l1
                ga1 += a1
                g2 += l2
                ga2 += a2
        empty["fighter1_head_sig_str"] = _format_of_y(h1, ha1)
        empty["fighter2_head_sig_str"] = _format_of_y(h2, ha2)
        empty["fighter1_body_sig_str"] = _format_of_y(b1, ba1)
        empty["fighter2_body_sig_str"] = _format_of_y(b2, ba2)
        empty["fighter1_leg_sig_str"] = _format_of_y(leg1, lega1)
        empty["fighter2_leg_sig_str"] = _format_of_y(leg2, lega2)
        empty["fighter1_distance_sig_str"] = _format_of_y(d1, da1)
        empty["fighter2_distance_sig_str"] = _format_of_y(d2, da2)
        empty["fighter1_clinch_sig_str"] = _format_of_y(c1, ca1)
        empty["fighter2_clinch_sig_str"] = _format_of_y(c2, ca2)
        empty["fighter1_ground_sig_str"] = _format_of_y(g1, ga1)
        empty["fighter2_ground_sig_str"] = _format_of_y(g2, ga2)

        return empty
    except Exception:
        return empty


def get_fights_for_event(
    session: requests.Session, event_url: str, event_info: dict, fetch_details: bool = True
) -> list[dict]:
    """Fetch all fights for a single event. fetch_details=True fetches per-fight stats (slower)."""
    resp = session.get(event_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    fights = []
    for row in soup.find_all("tr"):
        if row.find("th"):
            continue
        fight_link = row.find("a", href=re.compile(r"fight-details"))
        if not fight_link:
            continue

        fight_details_url = urljoin(BASE_URL, fight_link.get("href", ""))

        link_text = fight_link.get_text(strip=True).lower()
        if "draw" in link_text:
            outcome = "draw"
        elif "nc" in link_text or "no contest" in link_text:
            outcome = "nc"
        elif "dq" in link_text or "disqualif" in link_text:
            outcome = "dq"
        elif "win" in link_text:
            outcome = "win"
        elif "loss" in link_text:
            outcome = "loss"
        else:
            # Upcoming bout / no recorded result yet
            outcome = ""

        fighter_links = row.find_all("a", href=re.compile(r"fighter-details"))
        fighter1 = fighter_links[0].get_text(strip=True) if len(fighter_links) > 0 else ""
        fighter2 = fighter_links[1].get_text(strip=True) if len(fighter_links) > 1 else ""
        fighter1_url = urljoin(BASE_URL, fighter_links[0].get("href", "")) if len(fighter_links) > 0 else ""
        fighter2_url = urljoin(BASE_URL, fighter_links[1].get("href", "")) if len(fighter_links) > 1 else ""

        if not fighter1 or not fighter2:
            continue

        cells = row.find_all("td")
        weight_class = method = round_num = time_val = ""
        if len(cells) >= 10:
            weight_class = cells[6].get_text(strip=True)
            method = cells[7].get_text(separator=" ", strip=True)
            round_num = cells[8].get_text(strip=True)
            time_val = cells[9].get_text(strip=True)

        if outcome == "":
            continue

        winner = "" if outcome in ("draw", "nc", "dq") else (fighter1 if outcome == "win" else fighter2)

        base = {
            **event_info,
            "fighter_1": fighter1,
            "fighter_2": fighter2,
            "fighter_1_url": fighter1_url,
            "fighter_2_url": fighter2_url,
            "winner": winner,
            "weight_class": weight_class,
            "method": method,
            "round": round_num,
            "time": time_val,
        }
        assert "fighter_1_url" in base and "fighter_2_url" in base, "Always scrape fighter URLs (full and incremental)."

        if fetch_details:
            details_data = get_fight_details(session, fight_details_url)
            base.update(details_data)
            time.sleep(0.3)
        else:
            base.update({k: "" for k in (
                "referee", "number_of_rounds",
                "fighter1_kd", "fighter2_kd", "fighter1_sig_str", "fighter2_sig_str",
                "fighter1_total_str", "fighter2_total_str", "fighter1_td", "fighter2_td",
                "fighter1_sub_att", "fighter2_sub_att", "fighter1_rev", "fighter2_rev",
                "fighter1_ctrl", "fighter2_ctrl",
                "fighter1_head_sig_str", "fighter2_head_sig_str",
                "fighter1_body_sig_str", "fighter2_body_sig_str",
                "fighter1_leg_sig_str", "fighter2_leg_sig_str",
                "fighter1_distance_sig_str", "fighter2_distance_sig_str",
                "fighter1_clinch_sig_str", "fighter2_clinch_sig_str",
                "fighter1_ground_sig_str", "fighter2_ground_sig_str",
            )})

        fights.append(base)

    return fights


def scrape_all_fights(max_events: int | None = None) -> list[dict]:
    """Scrape all events and fights. Set max_events to limit for testing."""
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    print("Fetching events list...")
    events = get_events_list(session)
    print(f"Found {len(events)} events.")

    # Safety: ignore future-dated events (upcoming cards sometimes appear unexpectedly)
    today = datetime.now().date()
    events = [
        e for e in events
        if (dt := _parse_event_date(e.get("event_date") or "")) and dt.date() <= today
    ]

    if max_events:
        events = events[:max_events]
        print(f"Limiting to {max_events} events.")

    all_fights = []
    for i, evt in enumerate(events):
        print(f"[{i+1}/{len(events)}] {evt['event_name']}...")
        try:
            fights = get_fights_for_event(
                session, evt["event_url"], evt
            )
            all_fights.extend(fights)
            time.sleep(0.5)  # Be respectful to the server
        except Exception as e:
            print(f"  Error: {e}")

    return all_fights


def scrape_new_fights_only(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    max_events: int | None = None,
    storage: str = "local",
    blob_csv_path: str = DEFAULT_BLOB_CSV_PATH,
    blob_parquet_path: str = DEFAULT_BLOB_PARQUET_PATH,
) -> list[dict]:
    """
    Scrape only events that are after the latest event date in the existing CSV,
    then prepend those fights to the file (keeping descending-by-date order).
    If the CSV is missing or empty, falls back to a full scrape and writes the file.
    """
    csv_path = Path(csv_path)
    storage = (storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    latest_date_str = None
    existing_rows: list[dict] = []

    # When storage is both or azure, use cloud (Parquet then CSV) first; fall back to local only for "both" if cloud missing.
    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import read_csv_from_azure, read_parquet_from_azure

            try:
                existing_df = read_parquet_from_azure(blob_parquet_path).astype(str).fillna("")
            except FileNotFoundError:
                existing_df = read_csv_from_azure(blob_csv_path, dtype=str).fillna("")
            if not existing_df.empty and "event_date" in existing_df.columns:
                latest_date_str = str(existing_df.iloc[0].get("event_date") or "").strip() or None
            for c in CSV_FIELDNAMES:
                if c not in existing_df.columns:
                    existing_df[c] = ""
            existing_rows = existing_df[CSV_FIELDNAMES].fillna("").to_dict(orient="records")
        except FileNotFoundError:
            if storage == "azure":
                latest_date_str = None
                existing_rows = []
            # storage == "both": fall back to local below

    if not existing_rows and storage in ("local", "both") and csv_path.exists():
        latest_date_str = get_latest_event_date_from_csv(csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    for key in CSV_FIELDNAMES:
                        if key not in row:
                            row[key] = ""
                    existing_rows.append(row)

    latest_dt = _parse_event_date(latest_date_str) if latest_date_str else None

    print("Fetching events list...")
    events = get_events_list(session)
    print(f"Found {len(events)} total events.")

    # Safety: ignore future-dated events (upcoming cards sometimes appear unexpectedly)
    today = datetime.now().date()

    if latest_dt is not None:
        # Keep only events strictly after the latest we have (newer = later date), but not in the future
        events = [
            e for e in events
            if (dt := _parse_event_date(e.get("event_date") or ""))
            and dt.date() <= today
            and dt > latest_dt
        ]
        print(f"Events after {latest_date_str} (excluding future): {len(events)}.")
    else:
        # No cutoff: still exclude future events
        events = [
            e for e in events
            if (dt := _parse_event_date(e.get("event_date") or "")) and dt.date() <= today
        ]
        print("No existing CSV or empty — will perform full scrape (excluding future events).")

    if max_events is not None and events:
        events = events[:max_events]
        print(f"Limiting to {max_events} events.")

    if not events:
        print("No new events to scrape.")
        if existing_rows:
            _write_outputs(
                records=existing_rows,
                columns=CSV_FIELDNAMES,
                local_csv_path=csv_path,
                storage=storage,
                blob_csv_path=blob_csv_path,
                blob_parquet_path=blob_parquet_path,
            )
            print(f"Wrote existing {len(existing_rows)} rows to local and/or Azure.")
        return []

    new_fights = []
    for i, evt in enumerate(events):
        print(f"[{i+1}/{len(events)}] {evt['event_name']}...")
        try:
            fights = get_fights_for_event(session, evt["event_url"], evt)
            new_fights.extend(fights)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error: {e}")

    if not new_fights:
        print("No new fights to append.")
        if existing_rows:
            _write_outputs(
                records=existing_rows,
                columns=CSV_FIELDNAMES,
                local_csv_path=csv_path,
                storage=storage,
                blob_csv_path=blob_csv_path,
                blob_parquet_path=blob_parquet_path,
            )
            print(f"Wrote existing {len(existing_rows)} rows to local and/or Azure.")
        return []

    combined = new_fights + existing_rows
    _write_outputs(
        records=combined,
        columns=CSV_FIELDNAMES,
        local_csv_path=csv_path,
        storage=storage,
        blob_csv_path=blob_csv_path,
        blob_parquet_path=blob_parquet_path,
    )
    print(f"Appended {len(new_fights)} new fights. Total rows: {len(combined)}.")
    return new_fights


def save_outputs(
    fights: list[dict],
    csv_path: str | Path = DEFAULT_CSV_PATH,
    storage: str = "local",
    blob_csv_path: str = DEFAULT_BLOB_CSV_PATH,
    blob_parquet_path: str = DEFAULT_BLOB_PARQUET_PATH,
) -> None:
    """Write fights to CSV (local) and optionally CSV+Parquet to Azure."""
    if not fights:
        print("No fights to save.")
        return
    csv_path = Path(csv_path)
    storage = (storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    _write_outputs(
        records=fights,
        columns=CSV_FIELDNAMES,
        local_csv_path=csv_path,
        storage=storage,
        blob_csv_path=blob_csv_path,
        blob_parquet_path=blob_parquet_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UFC fight scraper: full (all events) or incremental (only new events after latest in CSV)."
    )
    parser.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        help="Only scrape events after the latest event in raw_ufc_fights.csv and append (keeps file descending by date).",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help="Output CSV path (default: <script_dir>/output/raw_ufc_fights.csv).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        metavar="N",
        help="Cap number of events to scrape (for testing).",
    )
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to write outputs. local writes to disk, azure writes to Blob, both writes to both.",
    )
    parser.add_argument(
        "--blob-csv",
        default=DEFAULT_BLOB_CSV_PATH,
        help="Blob path for CSV output (inside the container).",
    )
    parser.add_argument(
        "--blob-parquet",
        default=DEFAULT_BLOB_PARQUET_PATH,
        help="Blob path for Parquet output (inside the container).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if args.incremental:
        scrape_new_fights_only(
            csv_path=csv_path,
            max_events=args.max_events,
            storage=args.storage,
            blob_csv_path=args.blob_csv,
            blob_parquet_path=args.blob_parquet,
        )
    else:
        fights = scrape_all_fights(max_events=args.max_events)
        save_outputs(
            fights,
            csv_path,
            storage=args.storage,
            blob_csv_path=args.blob_csv,
            blob_parquet_path=args.blob_parquet,
        )
