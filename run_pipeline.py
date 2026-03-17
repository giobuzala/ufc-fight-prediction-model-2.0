"""
Run the full UFC fight prediction pipeline.

1. Scrape: past fights (incremental), fighters (incremental), upcoming (all)
2. Clean: fighters, then fights
3. Feature engineering
4. Split data
5. Train model
6. Predict upcoming fights

Use --quick for reduced training (~1 min).
Use --storage to control data location (local / azure / both).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd: list[str], desc: str):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")

    r = subprocess.run(
        [sys.executable] + cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if r.stdout:
        print(r.stdout)

    if r.stderr:
        print(r.stderr, file=sys.stderr)

    if r.returncode != 0:
        print(f"Failed: {' '.join(cmd)}")
        sys.exit(r.returncode)


def scrape(storage_args):
    run(
        ["module_01_scrapers/ufc_fight_scraper.py", "--incremental"] + storage_args,
        "1. Scrape past fights (incremental)",
    )
    run(
        ["module_01_scrapers/ufc_fighter_scraper.py", "--incremental"] + storage_args,
        "2. Scrape fighters (incremental)",
    )
    run(
        ["module_01_scrapers/ufc_upcoming_scraper.py"] + storage_args,
        "3. Scrape upcoming fights (all)",
    )


def clean(storage_args):
    run(
        ["module_02_clean_fighters/clean_ufc_fighters.py"] + storage_args,
        "4. Clean fighters",
    )
    run(
        ["module_03_clean_fights/clean_ufc_fights.py"] + storage_args,
        "5. Clean fights",
    )


def features(storage_args):
    run(
        ["module_04_feature_engineering/feature_engineering.py"] + storage_args,
        "6. Feature engineering",
    )


def split(storage_args):
    run(
        ["module_05_split/prep_and_split.py", "-d"] + storage_args,
        "7. Split data",
    )


def train(storage_args, quick):
    train_cmd = ["module_06_model/train_tuned.py"] + storage_args
    if quick:
        train_cmd.append("--quick")
    run(train_cmd, "8. Train best model" + (" (quick)" if quick else ""))


def predict(storage_args):
    run(
        ["module_07_predict/predict_upcoming.py"] + storage_args,
        "9. Predict upcoming fights",
    )


def main():
    parser = argparse.ArgumentParser(description="Run UFC fight prediction pipeline")
    parser.add_argument("--quick", action="store_true", help="Use quick training (~1 min)")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="both",
        help="Data location (default: both)",
    )
    args = parser.parse_args()

    storage_args = ["--storage", args.storage]

    start = time.perf_counter()
    scrape(storage_args)
    clean(storage_args)
    features(storage_args)
    split(storage_args)
    train(storage_args, args.quick)
    predict(storage_args)

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print("Pipeline complete. Predictions: module_07_predict/output/upcoming_predictions_yyyymmdd.csv")
    print(f"Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
