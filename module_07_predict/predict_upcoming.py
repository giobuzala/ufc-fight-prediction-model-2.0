"""
Predict upcoming UFC fights. Module 7.

Reads from module_06 output: best_model.joblib, preprocessor_diff.joblib
Reads from module_07 input: upcoming_for_prediction.joblib (prepared by module_06)

Writes: output/upcoming_predictions_yyyymmdd.csv (local and optionally Azure, CSV only).
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODULE_06_OUTPUT = _PROJECT_ROOT / "module_06_model" / "output"
MODULE_07_INPUT = Path(__file__).resolve().parent / "input"
MODULE_07_OUTPUT = Path(__file__).resolve().parent / "output"
BLOB_OUTPUT_PREFIX = "module_07_predict/output"

DEFAULT_MODEL = MODULE_06_OUTPUT / "best_model.joblib"
DEFAULT_PREPROCESSOR = MODULE_06_OUTPUT / "preprocessor_diff.joblib"
DEFAULT_UPCOMING_FEATURES = MODULE_07_INPUT / "upcoming_for_prediction.joblib"


def main():
    parser = argparse.ArgumentParser(description="Predict upcoming UFC fights")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR)
    parser.add_argument("--upcoming-features", type=Path, default=DEFAULT_UPCOMING_FEATURES)
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV path (default: output/upcoming_predictions_yyyymmdd.csv)")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to write output. local = disk only, azure/both = also upload CSV to Azure.",
    )
    args = parser.parse_args()

    output_path = args.output or (MODULE_07_OUTPUT / f"upcoming_predictions_{date.today():%Y%m%d}.csv")
    storage = (args.storage or "local").strip().lower()

    if not args.upcoming_features.exists():
        print(f"Error: Upcoming features not found: {args.upcoming_features}")
        print("Run module_06_model/train_tuned.py first (it prepares upcoming data for module 7)")
        sys.exit(1)
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not args.preprocessor.exists():
        print(f"Error: Preprocessor not found: {args.preprocessor}")
        sys.exit(1)

    data = joblib.load(args.upcoming_features)
    feature_rows = data["feature_rows"]
    fight_metadata = data["fight_metadata"]
    feature_cols = data["feature_cols"]

    from module_06_model.prepare_upcoming_features import filter_joblib_pairs_for_future_events

    feature_rows, fight_metadata, skipped_past = filter_joblib_pairs_for_future_events(
        feature_rows, fight_metadata
    )
    if skipped_past:
        print(f"Excluded {skipped_past} fight(s) with event date before today (past cards).")

    if not feature_rows:
        print("No upcoming fights to predict.")
        sys.exit(0)

    preproc_bundle = joblib.load(args.preprocessor)
    preprocessor = preproc_bundle["preprocessor"]

    model_bundle = joblib.load(args.model)
    model = model_bundle["model"]
    selector = model_bundle.get("selector")

    # Build feature matrix matching preprocessor column order
    X = pd.DataFrame(feature_rows)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = pd.NA
    X = X[feature_cols]
    numeric_cols = preproc_bundle["numeric_cols"]
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X_transformed = preprocessor.transform(X)
    if selector is not None:
        X_transformed = selector.transform(X_transformed)

    probas = model.predict_proba(X_transformed)
    prob_f1_wins = probas[:, 1]
    confidence = probas.max(axis=1)
    predicted = (prob_f1_wins >= 0.5).astype(int)

    results = []
    for i, meta in enumerate(fight_metadata):
        f1 = meta["fighter_1"]
        f2 = meta["fighter_2"]
        winner = f1 if predicted[i] else f2
        results.append({
            "event_name": meta["event_name"],
            "event_date": meta["event_date"],
            "location": meta["location"],
            "fighter_1": f1,
            "fighter_2": f2,
            "weight_class": meta["weight_class"],
            "predicted_winner": winner,
            "confidence": round(float(confidence[i]), 4),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nPredictions ({len(results)} fights):")
    for r in results:
        print(f"  {r['fighter_1']} vs {r['fighter_2']} -> {r['predicted_winner']} ({r['confidence']:.1%})")
    print(f"\nSaved to {output_path}")

    if storage in ("azure", "both"):
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from module_00_utils.azure_storage import upload_file_to_azure
        blob_name = f"{BLOB_OUTPUT_PREFIX}/{output_path.name}"
        upload_file_to_azure(str(output_path), blob_name)
        print(f"Uploaded to Azure: {blob_name}")


if __name__ == "__main__":
    main()
