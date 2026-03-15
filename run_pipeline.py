"""
Run the full UFC fight prediction pipeline.

1. Scrape: past fights (incremental), fighters (incremental), upcoming (all)
   (Scrapers write to next module inputs.)
2. Clean: fighters, then fights (each module writes to the next module's input.)
3. Feature engineering
4. Split data
5. Train: full grid search, pick best model (also prepares upcoming for module 7)
6. Predict: upcoming fights -> module_07_predict/output/upcoming_predictions_yyyymmdd.csv

Use --quick for reduced training (~1 min) during development.
Use --storage both to also read/write Azure Blob (connection string from Key Vault, see module_00_utils.azure_storage).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], desc: str):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"Failed: {' '.join(cmd)}")
        sys.exit(r.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run UFC fight prediction pipeline")
    parser.add_argument("--quick", action="store_true", help="Use quick training (~1 min) for testing")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="both",
        help="Where to read/write data. both = disk + Azure Blob (default), local = disk only.",
    )
    args = parser.parse_args()

    storage_args = ["--storage", args.storage]

    start = time.perf_counter()
    # 1. Scrape
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

    # 2. Clean
    run(
        ["module_02_clean_fighters/clean_ufc_fighters.py"] + storage_args,
        "4. Clean fighters",
    )
    run(
        ["module_03_clean_fights/clean_ufc_fights.py"] + storage_args,
        "5. Clean fights",
    )

    # 3. Feature engineering
    run(
        ["module_04_feature_engineering/feature_engineering.py"] + storage_args,
        "6. Feature engineering",
    )

    # 4. Split
    run(
        ["module_05_split/prep_and_split.py", "-d"] + storage_args,
        "7. Split data",
    )

    # 5. Train
    train_cmd = ["module_06_model/train_tuned.py"] + storage_args
    if args.quick:
        train_cmd.append("--quick")
    run(train_cmd, "8. Train best model" + (" (quick)" if args.quick else ""))

    # 6. Predict
    run(
        ["module_07_predict/predict_upcoming.py"] + storage_args,
        "9. Predict upcoming fights",
    )

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print("Pipeline complete. Predictions: module_07_predict/output/upcoming_predictions_yyyymmdd.csv")
    print(f"Total runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
