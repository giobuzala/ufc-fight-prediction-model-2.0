"""
Train baseline ML models (no hyperparameter tuning) and report train/test accuracy.

Reads: input/X_train.npz, etc.
Use -d to load differential-only features (X_train_diff.npz, etc.)
Writes: output/model_selection_summary_yyyymmdd.csv; also prepares module_07 input (upcoming_for_prediction.joblib).
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
INPUT_DIR = _SCRIPT_DIR / "input"
OUTPUT_DIR = _SCRIPT_DIR / "output"
MODULE_07_INPUT = _PROJECT_ROOT / "module_07_predict" / "input"
BLOB_OUTPUT_PREFIX = "module_06_model/output"
BLOB_MODULE_07_PREFIX = "module_07_predict/input"


def load_data(suffix: str = ""):
    """Load train and test arrays from input/. suffix='_diff' for differential-only."""
    X_train = np.load(INPUT_DIR / f"X_train{suffix}.npz")["X"]
    y_train = np.load(INPUT_DIR / f"y_train{suffix}.npz")["y"]
    X_test = np.load(INPUT_DIR / f"X_test{suffix}.npz")["X"]
    y_test = np.load(INPUT_DIR / f"y_test{suffix}.npz")["y"]
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--differential", action="store_true", help="Use differential-only features")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to write summary CSV. local = disk only, azure/both = also upload to Azure.",
    )
    args = parser.parse_args()
    storage = (args.storage or "local").strip().lower()

    suf = "_diff" if args.differential else ""
    X_train, y_train, X_test, y_test = load_data(suf)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ]

    feat_label = " (differential-only)" if args.differential else ""
    n_train, n_test = len(X_train), len(X_test)
    n_feat = X_train.shape[1]
    print(f"Train: {n_train} samples | Test: {n_test} samples | Features: {n_feat}{feat_label}")
    print("-" * 60)
    print(f"{'Model':<25} {'Train Acc':>12} {'Test Acc':>12}")
    print("-" * 60)

    rows = []
    for name, model in models:
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"{name:<25} {train_acc:>12.4f} {test_acc:>12.4f}")
        rows.append({
            "model": name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "n_train": n_train,
            "n_test": n_test,
            "n_features_used": n_feat,
        })

    print("-" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_date = date.today().strftime("%Y%m%d")
    summary_fname = f"model_selection_summary_{summary_date}.csv"
    summary_path = OUTPUT_DIR / summary_fname
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "train_accuracy", "test_accuracy", "n_train", "n_test", "n_features_used"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {summary_path}")
    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import upload_file_to_azure
        upload_file_to_azure(str(summary_path), f"{BLOB_OUTPUT_PREFIX}/{summary_fname}")
        print(f"Uploaded to Azure: {BLOB_OUTPUT_PREFIX}/{summary_fname}")

    # Prepare upcoming fights for module 7 (same as train_tuned)
    print("\nPreparing upcoming fights for module 7...")
    from prepare_upcoming_features import main as prepare_main
    prepare_main(storage=storage)
    if storage in ("azure", "both"):
        m07_joblib = MODULE_07_INPUT / "upcoming_for_prediction.joblib"
        if m07_joblib.exists():
            from module_00_utils.azure_storage import upload_file_to_azure
            upload_file_to_azure(str(m07_joblib), f"{BLOB_MODULE_07_PREFIX}/upcoming_for_prediction.joblib")
            print(f"Uploaded to Azure: {BLOB_MODULE_07_PREFIX}/upcoming_for_prediction.joblib")


if __name__ == "__main__":
    main()
