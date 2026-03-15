"""
Scrape upcoming UFC fights from UFCStats.com.

Fetches event list from /statistics/events/upcoming, then for each event
extracts fight matchups (fighter names, URLs, weight class). No per-fight stats.
Output: local CSV and/or Azure CSV+Parquet (see --storage).
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString

BASE_URL = "http://ufcstats.com"
UPCOMING_URL = f"{BASE_URL}/statistics/events/upcoming?page=all"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_OUTPUT_PATH = _SCRIPT_DIR / "output" / "raw_ufc_upcoming.csv"
DEFAULT_BLOB_CSV_PATH = "module_01_scrapers/output/raw_ufc_upcoming.csv"
DEFAULT_BLOB_PARQUET_PATH = "module_01_scrapers/output/raw_ufc_upcoming.parquet"

CSV_FIELDNAMES = [
    "event_url", "event_name", "event_date", "event_location",
    "fighter_1", "fighter_2", "fighter_1_url", "fighter_2_url",
    "weight_class", "number_of_rounds",
]


# Output helpers


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in CSV_FIELDNAMES:
        if c not in df.columns:
            df[c] = ""
    df = df[CSV_FIELDNAMES]
    return df.fillna("")


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

    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import write_csv_to_azure, write_parquet_to_azure

            write_csv_to_azure(df, blob_csv_path, index=False)
            write_parquet_to_azure(df, blob_parquet_path, index=False)
            print(f"Wrote {len(df)} rows to Azure blobs: {blob_csv_path} and {blob_parquet_path}")
        except Exception as e:
            print(f"Failed to write to Azure: {e}")
            raise


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s


# Scraping


def get_upcoming_events(session: requests.Session) -> list[dict]:
    """Fetch upcoming events from UFCStats."""
    resp = session.get(UPCOMING_URL)
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

        events.append({
            "event_url": urljoin(BASE_URL, href),
            "event_name": name,
            "event_date": date,
            "event_location": location,
        })

    return events


# Weight class names that appear on the page (for matching)
WEIGHT_CLASSES = [
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
    "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
    "Heavyweight", "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight", "Catch Weight",
]


def get_fights_for_upcoming_event(session: requests.Session, event_url: str, event_info: dict) -> list[dict]:
    """Parse upcoming event page for fight matchups. No fight stats (bout not happened).
    Page structure: fighter1 link, fighter2 link, "View Matchup", weight_class, repeat.
    """
    resp = session.get(event_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Iterate document order: pairs of fighter-details links, then weight class text.
    fights_data = []
    pending = []
    for elem in soup.descendants:
        if elem.name == "a" and "fighter-details" in (elem.get("href") or ""):
            pending.append(elem)
            if len(pending) == 2:
                f1 = pending[0].get_text(strip=True)
                f2 = pending[1].get_text(strip=True)
                if f1 and f2:
                    fights_data.append({"a1": pending[0], "a2": pending[1], "wc": ""})
                pending = []
        elif isinstance(elem, NavigableString):
            text = (elem or "").strip()
            if text in WEIGHT_CLASSES and fights_data and not fights_data[-1]["wc"]:
                fights_data[-1]["wc"] = text

    fights = []
    for i, fd in enumerate(fights_data):
        a1, a2 = fd["a1"], fd["a2"]
        f1 = a1.get_text(strip=True)
        f2 = a2.get_text(strip=True)
        url1 = urljoin(BASE_URL, a1.get("href", ""))
        url2 = urljoin(BASE_URL, a2.get("href", ""))
        number_of_rounds = "5" if i == 0 else "3"
        fights.append({
            **event_info,
            "fighter_1": f1,
            "fighter_2": f2,
            "fighter_1_url": url1,
            "fighter_2_url": url2,
            "weight_class": fd["wc"],
            "number_of_rounds": number_of_rounds,
        })
    return fights


def main():
    parser = argparse.ArgumentParser(description="Scrape upcoming UFC fights from UFCStats")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-events", type=int, default=None, help="Limit events to scrape")
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

    session = _session()
    print("Fetching upcoming events...")
    events = get_upcoming_events(session)
    print(f"Found {len(events)} upcoming events.")

    if args.max_events and events:
        events = events[: args.max_events]

    all_fights = []
    for i, ev in enumerate(events):
        print(f"[{i+1}/{len(events)}] {ev['event_name']}...")
        try:
            fights = get_fights_for_upcoming_event(session, ev["event_url"], ev)
            all_fights.extend(fights)
            print(f"  {len(fights)} fights")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.5)

    storage = (args.storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    _write_outputs(
        records=all_fights,
        local_csv_path=args.output,
        storage=storage,
        blob_csv_path=args.blob_csv,
        blob_parquet_path=args.blob_parquet,
    )


if __name__ == "__main__":
    main()
