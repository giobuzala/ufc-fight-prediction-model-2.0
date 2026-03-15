"""
UFC Fighter Stats Scraper
Scrapes fighter profile data from UFCStats.com (statistics/fighters).
Output: fighter_url, full_name, height, weight, reach, stance, date_of_birth

Modes:
- Full: scrape_all_fighters() — all fighters (use for initial load or periodic refresh).
- Incremental: scrape_new_fighters_only() — only fighters whose fighter_url is not already in the CSV.
"""

import argparse
import csv
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://ufcstats.com"
FIGHTERS_URL = f"{BASE_URL}/statistics/fighters"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_CSV_PATH = _SCRIPT_DIR / "output" / "raw_ufc_fighters.csv"
DEFAULT_BLOB_CSV_PATH = "module_01_scrapers/output/raw_ufc_fighters.csv"
DEFAULT_BLOB_PARQUET_PATH = "module_01_scrapers/output/raw_ufc_fighters.parquet"
# Blob paths for module_02 input so the container has them when running module 2 from cloud.
MODULE_02_INPUT_BLOB_CSV = "module_02_clean_fighters/input/raw_ufc_fighters.csv"
MODULE_02_INPUT_BLOB_PARQUET = "module_02_clean_fighters/input/raw_ufc_fighters.parquet"

FIELDNAMES = ["fighter_url", "full_name", "height", "weight", "reach", "stance", "date_of_birth"]

# Reused for all requests to avoid repeated dict literals.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# Output helpers


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in FIELDNAMES:
        if c not in df.columns:
            df[c] = ""
    df = df[FIELDNAMES]
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
        # Keep module_02 input in sync when writing the default fighter output.
        if local_csv_path.resolve() == Path(DEFAULT_CSV_PATH).resolve():
            module_02_input = _PROJECT_ROOT / "module_02_clean_fighters" / "input" / "raw_ufc_fighters.csv"
            module_02_input.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_csv_path, module_02_input)
            print(f"Updated module_02 input: {module_02_input}")

    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import write_csv_to_azure, write_parquet_to_azure

            write_csv_to_azure(df, blob_csv_path, index=False)
            write_parquet_to_azure(df, blob_parquet_path, index=False)
            print(f"Wrote {len(df)} rows to Azure blobs: {blob_csv_path} and {blob_parquet_path}")
            # So module_2 can read input from the container when running in cloud.
            if blob_csv_path == DEFAULT_BLOB_CSV_PATH:
                write_csv_to_azure(df, MODULE_02_INPUT_BLOB_CSV, index=False)
                write_parquet_to_azure(df, MODULE_02_INPUT_BLOB_PARQUET, index=False)
                print(f"Updated module_02 input in container: {MODULE_02_INPUT_BLOB_CSV} and {MODULE_02_INPUT_BLOB_PARQUET}")
        except Exception as e:
            print(f"Failed to write to Azure: {e}")
            raise


# Scraping


def get_fighter_detail_urls(session: requests.Session, char: str = "a") -> list[str]:
    """Fetch all fighter detail URLs for a given letter (a-z)."""
    url = f"{FIGHTERS_URL}?char={char}&page=all"
    try:
        resp = session.get(url)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    seen = set()
    for a in soup.find_all("a", href=re.compile(r"fighter-details/[a-z0-9]+")):
        href = a.get("href", "")
        if not href or href in seen:
            continue
        seen.add(href)
        full_url = urljoin(BASE_URL, href)
        urls.append(full_url)

    return urls


def get_fighter_details(session: requests.Session, detail_url: str) -> dict | None:
    """Scrape fighter_url, full_name, height, weight, reach, stance, date_of_birth from fighter details page."""
    try:
        resp = session.get(detail_url)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Failed to fetch {detail_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    data = {
        "fighter_url": detail_url,
        "full_name": "",
        "height": "",
        "weight": "",
        "reach": "",
        "stance": "",
        "date_of_birth": "",
    }

    span = soup.find("span", class_=re.compile(r"b-content__title"))
    if span:
        data["full_name"] = span.get_text(strip=True)
    if not data["full_name"]:
        h2 = soup.find("h2", class_=re.compile(r"b-content__title"))
        if h2:
            txt = h2.get_text(strip=True)
            m = re.match(r"^(.+?)(?:Record:.*)?$", txt)
            data["full_name"] = m.group(1).strip() if m else txt

    for li in soup.find_all("li"):
        txt = li.get_text(strip=True)
        for label, key in [
            ("Height:", "height"),
            ("Weight:", "weight"),
            ("Reach:", "reach"),
            ("STANCE:", "stance"),
            ("DOB:", "date_of_birth"),
        ]:
            if txt.startswith(label):
                val = txt[len(label) :].strip()
                data[key] = "" if val in ("--", "-") else val
                break

    return data


def scrape_all_fighters(
    max_per_letter: int | None = None,
    letters: str | None = None,
) -> list[dict]:
    """Scrape fighters from UFCStats. Set max_per_letter to limit; letters to restrict (e.g. 'a' or 'abc')."""
    session = requests.Session()
    session.headers.update(HEADERS)

    all_fighters = []
    seen_urls = set()
    chars = letters if letters else "abcdefghijklmnopqrstuvwxyz"

    for char in chars:
        print(f"Fetching fighters ({char})...")
        urls = get_fighter_detail_urls(session, char)
        if max_per_letter:
            urls = urls[:max_per_letter]

        for i, url in enumerate(urls):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            details = get_fighter_details(session, url)
            if details and details.get("full_name"):
                all_fighters.append(details)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(urls)}...")
            time.sleep(0.15)

        time.sleep(0.25)

    return all_fighters


def _load_existing_fighter_urls(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "fighter_url" not in reader.fieldnames:
            return set()
        return {row.get("fighter_url", "").strip() for row in reader if row.get("fighter_url", "").strip()}


def scrape_new_fighters_only(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    max_per_letter: int | None = None,
    letters: str | None = None,
    storage: str = "local",
    blob_csv_path: str = DEFAULT_BLOB_CSV_PATH,
    blob_parquet_path: str = DEFAULT_BLOB_PARQUET_PATH,
) -> list[dict]:
    csv_path = Path(csv_path)
    storage = (storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    existing_urls: set[str] = set()
    existing_rows: list[dict] = []

    # When storage is both or azure, use cloud (Parquet then CSV) first; fall back to local only for "both" if cloud missing.
    if storage in ("azure", "both"):
        try:
            from module_00_utils.azure_storage import read_csv_from_azure, read_parquet_from_azure

            try:
                existing_df = read_parquet_from_azure(blob_parquet_path).astype(str).fillna("")
            except FileNotFoundError:
                existing_df = read_csv_from_azure(blob_csv_path, dtype=str).fillna("")
            if "fighter_url" not in existing_df.columns:
                print("Existing Azure CSV has no fighter_url column; run a full refresh once to add fighter_url, then incremental will work.")
                for c in FIELDNAMES:
                    if c not in existing_df.columns:
                        existing_df[c] = ""
                existing_rows = existing_df[FIELDNAMES].fillna("").to_dict(orient="records")
                if existing_rows:
                    _write_outputs(
                        records=existing_rows,
                        local_csv_path=csv_path,
                        storage=storage,
                        blob_csv_path=blob_csv_path,
                        blob_parquet_path=blob_parquet_path,
                    )
                    print(f"Wrote existing {len(existing_rows)} rows to local and/or Azure.")
                return []
            existing_urls = {
                str(u).strip()
                for u in existing_df["fighter_url"].tolist()
                if str(u).strip()
            }
            for c in FIELDNAMES:
                if c not in existing_df.columns:
                    existing_df[c] = ""
            existing_rows = existing_df[FIELDNAMES].fillna("").to_dict(orient="records")
        except FileNotFoundError:
            if storage == "azure":
                existing_urls = set()
                existing_rows = []
            # storage == "both": fall through to try local

    if not existing_rows and storage in ("local", "both") and csv_path.exists():
        existing_urls = _load_existing_fighter_urls(csv_path)
        if not existing_urls:
            print("Existing CSV has no fighter_url column; run a full refresh once to add fighter_url, then incremental will work.")
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                for r in existing_rows:
                    for c in FIELDNAMES:
                        if c not in r:
                            r[c] = ""
            if existing_rows:
                _write_outputs(
                    records=existing_rows,
                    local_csv_path=csv_path,
                    storage=storage,
                    blob_csv_path=blob_csv_path,
                    blob_parquet_path=blob_parquet_path,
                )
                print(f"Wrote existing {len(existing_rows)} rows to local and/or Azure.")
            return []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "fighter_url" in reader.fieldnames:
                existing_rows = list(reader)

    session = requests.Session()
    session.headers.update(HEADERS)

    chars = letters if letters else "abcdefghijklmnopqrstuvwxyz"
    new_urls: list[str] = []
    for char in chars:
        print(f"Fetching fighters index ({char})...")
        urls = get_fighter_detail_urls(session, char)
        if max_per_letter:
            urls = urls[:max_per_letter]
        for u in urls:
            if u not in existing_urls:
                new_urls.append(u)
        time.sleep(0.15)

    # Deduplicate URLs (same fighter can appear under multiple letters)
    new_urls = list(dict.fromkeys(new_urls))

    if not new_urls:
        print("No new fighters found.")
        if existing_rows:
            _write_outputs(
                records=existing_rows,
                local_csv_path=csv_path,
                storage=storage,
                blob_csv_path=blob_csv_path,
                blob_parquet_path=blob_parquet_path,
            )
            print(f"Wrote existing {len(existing_rows)} rows to local and/or Azure.")
        return []

    print(f"New fighters to scrape: {len(new_urls)}")
    new_fighters: list[dict] = []
    for i, url in enumerate(new_urls):
        details = get_fighter_details(session, url)
        if details and details.get("full_name"):
            new_fighters.append(details)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(new_urls)}...")
        time.sleep(0.15)

    if not new_fighters:
        print("No new fighter details scraped.")
        if existing_rows:
            _write_outputs(
                records=existing_rows,
                local_csv_path=csv_path,
                storage=storage,
                blob_csv_path=blob_csv_path,
                blob_parquet_path=blob_parquet_path,
            )
            print(f"Wrote existing {len(existing_rows)} rows to local and/or Azure.")
        return []

    combined = existing_rows + new_fighters
    combined.sort(key=lambda r: (r.get("full_name", "") or "").lower())

    _write_outputs(
        records=combined,
        local_csv_path=csv_path,
        storage=storage,
        blob_csv_path=blob_csv_path,
        blob_parquet_path=blob_parquet_path,
    )
    print(f"Appended {len(new_fighters)} new fighters. Total rows: {len(combined)}.")
    return new_fighters


def save_outputs(
    fighters: list[dict],
    csv_path: str | Path = DEFAULT_CSV_PATH,
    storage: str = "local",
    blob_csv_path: str = DEFAULT_BLOB_CSV_PATH,
    blob_parquet_path: str = DEFAULT_BLOB_PARQUET_PATH,
) -> None:
    """Write fighters to CSV (local) and optionally CSV+Parquet to Azure."""
    if not fighters:
        print("No fighters to save.")
        return
    csv_path = Path(csv_path)
    storage = (storage or "local").strip().lower()
    if storage not in ("local", "azure", "both"):
        raise ValueError("storage must be one of: local, azure, both")

    _write_outputs(
        records=fighters,
        local_csv_path=csv_path,
        storage=storage,
        blob_csv_path=blob_csv_path,
        blob_parquet_path=blob_parquet_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UFC fighter scraper: full (all fighters) or incremental (only new fighter URLs)."
    )
    parser.add_argument("--incremental", "-i", action="store_true", help="Only scrape new fighter URLs.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Output CSV path (default: <script_dir>/output/raw_ufc_fighters.csv).")
    parser.add_argument("--max-per-letter", type=int, default=None, metavar="N", help="Cap fighters per letter (for testing).")
    parser.add_argument("--letters", default=None, help="Restrict letters (e.g. 'a' or 'abc').")
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
        scrape_new_fighters_only(
            csv_path=csv_path,
            max_per_letter=args.max_per_letter,
            letters=args.letters,
            storage=args.storage,
            blob_csv_path=args.blob_csv,
            blob_parquet_path=args.blob_parquet,
        )
    else:
        fighters = scrape_all_fighters(max_per_letter=args.max_per_letter, letters=args.letters)
        save_outputs(
            fighters,
            csv_path,
            storage=args.storage,
            blob_csv_path=args.blob_csv,
            blob_parquet_path=args.blob_parquet,
        )
