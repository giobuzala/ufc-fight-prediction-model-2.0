"""
Full grid search across many models. Always pick the absolute best by validation score.

Reads: input/ (split data from module_05), or from Azure when --storage azure/both.
Writes: output/best_model.joblib, output/preprocessor_diff.joblib, output/model_selection_summary_yyyymmdd.csv
        Copies upcoming features to module_07/input/ (local and optionally Azure).
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

import warnings
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

INPUT_DIR = _SCRIPT_DIR / "input"
OUTPUT_DIR = _SCRIPT_DIR / "output"
MODULE_07_INPUT = _PROJECT_ROOT / "module_07_predict" / "input"
BLOB_INPUT_PREFIX = "module_06_model/input"
BLOB_OUTPUT_PREFIX = "module_06_model/output"
BLOB_MODULE_07_PREFIX = "module_07_predict/input"
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 3


def load_data(suffix: str = "_diff", input_dir: Path | None = None):
    """Load train/val/test arrays. input_dir defaults to INPUT_DIR."""
    base = input_dir or INPUT_DIR
    X_train = np.load(base / f"X_train{suffix}.npz")["X"]
    y_train = np.load(base / f"y_train{suffix}.npz")["y"]
    X_val = np.load(base / f"X_val{suffix}.npz")["X"]
    y_val = np.load(base / f"y_val{suffix}.npz")["y"]
    X_test = np.load(base / f"X_test{suffix}.npz")["X"]
    y_test = np.load(base / f"y_test{suffix}.npz")["y"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_data_from_storage(storage: str, suffix: str = "_diff"):
    """Load arrays from Azure (when azure/both) or local. Fall back to local when storage is both and Azure fails."""
    storage = (storage or "local").strip().lower()
    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import download_file_from_azure

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
                    blob_path = f"{BLOB_INPUT_PREFIX}/{name}{suffix}.npz"
                    local_path = tmp / f"{name}{suffix}.npz"
                    download_file_from_azure(blob_path, str(local_path))
                return load_data(suffix, input_dir=tmp)
        except Exception as e:
            if storage == "azure":
                raise
            print(f"  Azure load failed ({e}), falling back to local input/")
    return load_data(suffix)


def get_ts_cv():
    return TimeSeriesSplit(n_splits=CV_FOLDS, test_size=int(5670 * 0.15), gap=0)


def select_features(X_train, y_train, X_val, X_test, k: int | None = None):
    if k is None or k >= X_train.shape[1]:
        return X_train, X_val, X_test, None
    selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
    X_train_s = selector.fit_transform(X_train, y_train)
    X_val_s = selector.transform(X_val)
    X_test_s = selector.transform(X_test)
    return X_train_s, X_val_s, X_test_s, selector


def grid_search(estimator, param_grid, X_train, y_train, name: str = ""):
    """Full grid search. Returns (best_estimator, best_val_score)."""
    search = GridSearchCV(
        estimator, param_grid, cv=get_ts_cv(), scoring="accuracy",
        n_jobs=N_JOBS, verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_


def _run_search(X_train, y_train, X_val, y_val, X_test, y_test, n_feat, quick: bool, summary_date_str: str | None = None):
    """Run model search. quick=True: LR+RF only, k=40, minimal grids, 2-fold (~1 min)."""
    if summary_date_str is None:
        summary_date_str = date.today().strftime("%Y%m%d")
    best_overall = None
    best_overall_val = 0.0
    best_overall_name = ""
    cv_n = 2 if quick else CV_FOLDS

    def do_grid(est, params, Xt, y_tr, verbose=1):
        cv = TimeSeriesSplit(n_splits=cv_n, test_size=int(5670 * 0.15), gap=0)
        gs = GridSearchCV(est, params, cv=cv, scoring="accuracy", n_jobs=N_JOBS, verbose=verbose)
        gs.fit(Xt, y_tr)
        return gs.best_estimator_, gs.best_score_

    k_options = [40] if quick else [None, 50, 40, 35, 30]

    for k in k_options:
        label = f"all ({n_feat})" if k is None else str(k)
        print(f"\n{'='*60}")
        print(f"Feature selection: {label}")
        print("=" * 60)
        Xt, Xv, Xte, sel = select_features(X_train, y_train, X_val, X_test, k)
        candidates = []

        # Logistic Regression
        lr_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ])
        lr_params = {"clf__C": [0.1, 1.0], "clf__solver": ["lbfgs"]} if quick else {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "saga"],
            "clf__class_weight": [None, "balanced"],
        }
        try:
            model, _ = do_grid(lr_pipe, lr_params, Xt, y_train, verbose=0 if quick else 1)
            va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
            candidates.append(("LR", model, va, ta))
            print(f"  LR: Val={va:.4f} Test={ta:.4f}")
        except Exception as e:
            print(f"  LR failed: {e}")

        # Random Forest
        rf_params = {"n_estimators": [100], "max_depth": [6], "min_samples_leaf": [4]} if quick else {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, 10],
            "min_samples_leaf": [2, 4, 8],
        }
        try:
            model, _ = do_grid(
                RandomForestClassifier(random_state=RANDOM_STATE),
                rf_params, Xt, y_train, verbose=0 if quick else 1
            )
            va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
            candidates.append(("RF", model, va, ta))
            print(f"  RF: Val={va:.4f} Test={ta:.4f}")
        except Exception as e:
            print(f"  RF failed: {e}")

        # ExtraTrees (skipped in quick mode)
        if not quick:
            et_params = {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "min_samples_leaf": [2, 4],
            }
            try:
                model, _ = do_grid(
                    ExtraTreesClassifier(random_state=RANDOM_STATE),
                    et_params, Xt, y_train
                )
                va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
                candidates.append(("ET", model, va, ta))
                print(f"  ET: Val={va:.4f} Test={ta:.4f}")
            except Exception as e:
                print(f"  ET failed: {e}")

        # Gradient Boosting, XGBoost, AdaBoost, SVC, MLP (skipped in quick mode)
        if not quick:
            gb_params = {
                "n_estimators": [200, 300],
                "max_depth": [4, 5, 6],
                "learning_rate": [0.05, 0.1],
            }
            try:
                model, _ = do_grid(
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                    gb_params, Xt, y_train
                )
                va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
                candidates.append(("GB", model, va, ta))
                print(f"  GB: Val={va:.4f} Test={ta:.4f}")
            except Exception as e:
                print(f"  GB failed: {e}")

        if not quick and HAS_XGB:
            xgb_params = {
                "n_estimators": [200, 300],
                "max_depth": [4, 5, 6],
                "learning_rate": [0.05, 0.1],
            }
            try:
                model, _ = do_grid(
                    XGBClassifier(random_state=RANDOM_STATE),
                    xgb_params, Xt, y_train
                )
                va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
                candidates.append(("XGB", model, va, ta))
                print(f"  XGB: Val={va:.4f} Test={ta:.4f}")
            except Exception as e:
                print(f"  XGB failed: {e}")

        if not quick:
            ada_params = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.5, 1.0],
            }
            try:
                model, _ = do_grid(
                    AdaBoostClassifier(random_state=RANDOM_STATE),
                    ada_params, Xt, y_train
                )
                va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
                candidates.append(("AdaBoost", model, va, ta))
                print(f"  AdaBoost: Val={va:.4f} Test={ta:.4f}")
            except Exception as e:
                print(f"  AdaBoost failed: {e}")

            svc_pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
            ])
            svc_params = {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", "auto"],
                "clf__class_weight": [None, "balanced"],
            }
            try:
                model, _ = do_grid(svc_pipe, svc_params, Xt, y_train)
                va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
                candidates.append(("SVC", model, va, ta))
                print(f"  SVC: Val={va:.4f} Test={ta:.4f}")
            except Exception as e:
                print(f"  SVC failed: {e}")

            mlp_pipe = Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(max_iter=500, random_state=RANDOM_STATE, early_stopping=True)),
            ])
            mlp_params = {
                "clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "clf__alpha": [0.001, 0.01],
                "clf__batch_size": [64, 128],
            }
            try:
                model, _ = do_grid(mlp_pipe, mlp_params, Xt, y_train)
                va, ta = model.score(Xv, y_val), model.score(Xte, y_test)
                candidates.append(("MLP", model, va, ta))
                print(f"  MLP: Val={va:.4f} Test={ta:.4f}")
            except Exception as e:
                print(f"  MLP failed: {e}")

        for name, model, va, ta in candidates:
            if va > best_overall_val:
                best_overall_val = va
                best_overall = (model, sel, k)
                best_overall_name = f"{name} (k={label})"

    print("\n" + "=" * 60)
    print("BEST:", best_overall_name)
    if best_overall:
        model, sel, k = best_overall
        Xt, Xv, Xte, _ = select_features(X_train, y_train, X_val, X_test, k)
        X_full = np.vstack([Xt, Xv])
        y_full = np.concatenate([y_train, y_val])
        model_refit = clone(model)
        model_refit.fit(X_full, y_full)
        val_acc = model_refit.score(Xv, y_val)
        test_acc = model_refit.score(Xte, y_test)
        train_acc = model_refit.score(Xt, y_train)
        print(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model_refit, "selector": sel, "k_features": k},
            OUTPUT_DIR / "best_model.joblib",
        )
        print(f"  Saved to {OUTPUT_DIR / 'best_model.joblib'}")

        # Save model selection summary to CSV
        import csv
        short_name = best_overall_name.split(" (k=")[0] if " (k=" in best_overall_name else best_overall_name
        MODEL_FULL_NAMES = {
            "LR": "Logistic Regression",
            "RF": "Random Forest",
            "ET": "Extra Trees",
            "GB": "Gradient Boosting",
            "XGB": "XGBoost",
            "AdaBoost": "AdaBoost",
            "SVC": "Support Vector Classifier",
            "MLP": "Multilayer Perceptron",
        }
        model_name = MODEL_FULL_NAMES.get(short_name, short_name)
        k_str = str(k) if k is not None else f"all ({n_feat})"
        params_str = ""
        if hasattr(model_refit, "get_params"):
            p = model_refit.get_params()
            # Keep tuned params (C, max_depth, n_estimators, etc.), drop random_state
            key_params = {k: v for k, v in p.items() if any(x in k for x in ["C", "max_depth", "n_estimators", "learning_rate", "alpha", "hidden_layer", "gamma", "solver", "class_weight"]) and "random_state" not in k}
            params_str = str(key_params)[:400]
        # Summary only (one row): local CSV and Azure CSV; no Parquet
        summary_path = OUTPUT_DIR / f"model_selection_summary_{summary_date_str}.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model", "k_features", "train_accuracy", "val_accuracy", "test_accuracy",
                "n_train", "n_val", "n_test", "n_features_used", "hyperparameters",
            ])
            w.writeheader()
            w.writerow({
                "model": model_name,
                "k_features": k_str,
                "train_accuracy": round(train_acc, 4),
                "val_accuracy": round(val_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_test": len(y_test),
                "n_features_used": Xt.shape[1],
                "hyperparameters": params_str,
            })
        print(f"  Saved {summary_path}")
    print("=" * 60)

    # Copy preprocessor to output (module_05 already saves to module_06/input)
    preproc_path = INPUT_DIR / "preprocessor_diff.joblib"
    if preproc_path.exists():
        import shutil
        shutil.copy(preproc_path, OUTPUT_DIR / "preprocessor_diff.joblib")
        print(f"  Copied preprocessor to {OUTPUT_DIR / 'preprocessor_diff.joblib'}")

    # Prepare upcoming fights for module 7
    print("\nPreparing upcoming fights for module 7...")
    from prepare_upcoming_features import main as prepare_main
    prepare_main()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train best model via grid search")
    parser.add_argument("--quick", action="store_true", help="Reduced search for testing (~1 min)")
    parser.add_argument(
        "--storage",
        choices=["local", "azure", "both"],
        default="local",
        help="Where to read input and write output. local = disk only, azure = blob only, both = disk + blob.",
    )
    args = parser.parse_args()
    storage = (args.storage or "local").strip().lower()

    print("Loading from input/..." + (" (Azure)" if storage in ("azure", "both") else ""))
    X_train, y_train, X_val, y_val, X_test, y_test = _load_data_from_storage(storage)
    # When loading from Azure, ensure preprocessor is in INPUT_DIR for the copy step later
    if storage in ("azure", "both") and not (INPUT_DIR / "preprocessor_diff.joblib").exists():
        from module_00_utils.azure_storage import download_file_from_azure
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            download_file_from_azure(f"{BLOB_INPUT_PREFIX}/preprocessor_diff.joblib", str(INPUT_DIR / "preprocessor_diff.joblib"))
        except Exception as e:
            print(f"  Could not download preprocessor from Azure: {e}")
    n_feat = X_train.shape[1]

    if args.quick:
        print("QUICK MODE: LR + RF, k=40, minimal grids, 2-fold CV (~1 min)\n")

    summary_date = date.today().strftime("%Y%m%d")
    _run_search(X_train, y_train, X_val, y_val, X_test, y_test, n_feat, quick=args.quick, summary_date_str=summary_date)

    if storage in ("azure", "both"):
        from module_00_utils.azure_storage import upload_file_to_azure

        # Upload outputs as-is (summary is CSV only, not Parquet)
        summary_fname = f"model_selection_summary_{summary_date}.csv"
        for fname in ("best_model.joblib", "preprocessor_diff.joblib", summary_fname):
            p = OUTPUT_DIR / fname
            if p.exists():
                upload_file_to_azure(str(p), f"{BLOB_OUTPUT_PREFIX}/{fname}")
        m07_joblib = MODULE_07_INPUT / "upcoming_for_prediction.joblib"
        if m07_joblib.exists():
            upload_file_to_azure(str(m07_joblib), f"{BLOB_MODULE_07_PREFIX}/upcoming_for_prediction.joblib")
        print(f"  Uploaded to Azure: {BLOB_OUTPUT_PREFIX}/ and {BLOB_MODULE_07_PREFIX}/")


if __name__ == "__main__":
    main()
