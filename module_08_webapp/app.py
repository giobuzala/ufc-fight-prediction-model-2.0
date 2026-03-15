"""
UFC Fight Prediction Web App.

Two tabs: Predict Winner (head-to-head) | Upcoming Fights (list with predictions).

Uses data from pipeline outputs. Fighters = those who fought in past 5 years (from last pipeline run).
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request, jsonify

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "module_04_feature_engineering"))
sys.path.insert(0, str(_PROJECT_ROOT / "module_06_model"))

# Paths
CLEAN_FIGHTS = _PROJECT_ROOT / "module_04_feature_engineering" / "input" / "clean_ufc_fights.csv"
if not CLEAN_FIGHTS.exists():
    CLEAN_FIGHTS = _PROJECT_ROOT / "module_03_clean_fights" / "output" / "clean_ufc_fights.csv"
CLEAN_FIGHTERS = _PROJECT_ROOT / "module_02_clean_fighters" / "output" / "clean_ufc_fighters.csv"
UPCOMING_PREDICTIONS = _PROJECT_ROOT / "module_07_predict" / "output" / "upcoming_predictions.csv"
MODEL_PATH = _PROJECT_ROOT / "module_06_model" / "output" / "best_model.joblib"
PREPROC_PATH = _PROJECT_ROOT / "module_06_model" / "output" / "preprocessor_diff.joblib"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True


def _parse_event_date(s: str) -> datetime | None:
    s = (s or "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load_fighters_past_n_years(n_years: int = 5) -> tuple[list[dict], datetime]:
    """
    Load fighters who fought in the past n years.
    Returns (list of {name, url, division}, cutoff_date).
    Uses most recent event_date in data as reference (pipeline run date).
    """
    if not CLEAN_FIGHTS.exists():
        return [], datetime.now()
    fighters_by_name: dict[str, dict] = {}
    max_date = None
    with open(CLEAN_FIGHTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ed_str = (row.get("event_date") or "").strip()
            dt = _parse_event_date(ed_str)
            if dt:
                max_date = max(max_date, dt) if max_date else dt
            div = (row.get("division") or "Men").strip()
            for lbl, url_key in [("fighter_1", "fighter_1_url"), ("fighter_2", "fighter_2_url")]:
                name = (row.get(lbl) or "").strip()
                url = (row.get(url_key) or "").strip().rstrip("/")
                if name:
                    fighters_by_name[name] = {"name": name, "url": url or name, "division": div}
    if not max_date:
        max_date = datetime.now()
    cutoff = max_date - timedelta(days=n_years * 365)
    # Filter: only include if they have a fight on or after cutoff
    with open(CLEAN_FIGHTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        valid_names = set()
        for row in reader:
            dt = _parse_event_date((row.get("event_date") or "").strip())
            if dt and dt >= cutoff:
                valid_names.add((row.get("fighter_1") or "").strip())
                valid_names.add((row.get("fighter_2") or "").strip())
    filtered = [v for k, v in fighters_by_name.items() if k in valid_names]
    men = sorted([f for f in filtered if f["division"] == "Men"], key=lambda x: x["name"])
    women = sorted([f for f in filtered if f["division"] == "Women"], key=lambda x: x["name"])
    return {"men": men, "women": women}, max_date


def load_fighter_attrs() -> dict:
    """fighter_url -> {height, reach, stance, date_of_birth, full_name}"""
    out = {}
    if not CLEAN_FIGHTERS.exists():
        return out
    with open(CLEAN_FIGHTERS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("fighter_url") or "").strip().rstrip("/")
            if not url:
                continue
            out[url] = {
                "height": (row.get("height") or "").strip(),
                "reach": (row.get("reach") or "").strip(),
                "stance": (row.get("stance") or "Unknown").strip() or "Unknown",
                "date_of_birth": (row.get("date_of_birth") or "").strip(),
                "full_name": (row.get("full_name") or "").strip(),
            }
            if url + "/" not in out:
                out[url + "/"] = out[url]
    return out


def get_fighter_record_from_state(state: dict) -> str:
    r = state.get("records", {})
    w = r.get("wins", 0)
    l = r.get("losses", 0)
    o = r.get("other", 0)
    extra = f", {o}NC" if o else ""
    return f"{w}-{l}-0{extra}" if (w or l or o) else "0-0-0"


def _inches_to_ft_in(inches) -> str:
    """Convert inches (int or str) to feet'inches\" e.g. 71 -> 5'11\"."""
    try:
        n = int(inches)
        return f"{n // 12}'{n % 12}\""
    except (TypeError, ValueError):
        return "" if not inches else str(inches)


def predict_matchup(fighter1_name: str, fighter1_url: str, fighter2_name: str, fighter2_url: str,
                    weight_class: str, number_of_rounds: str = "3") -> dict | None:
    """Run model prediction for a custom matchup. Returns {winner, confidence, prob_f1} or None."""
    if not MODEL_PATH.exists() or not PREPROC_PATH.exists():
        return None
    import joblib
    import pandas as pd
    from feature_engineering import compute_fighter_state_snapshot
    from prepare_upcoming_features import build_feature_row, load_fighters_by_url, _parse_event_date, _parse_dob, _age_at_date
    from prepare_upcoming_features import _weight_class_to_lbs_upcoming, _division_from_weight_class

    clean_fights = CLEAN_FIGHTS
    clean_fighters = CLEAN_FIGHTERS
    if not clean_fights.exists() or not clean_fighters.exists():
        return None

    state_snapshot = compute_fighter_state_snapshot(input_path=clean_fights)
    fighters_by_url = load_fighters_by_url(clean_fighters)

    def get_attrs(url, name):
        url_norm = (url or "").strip().rstrip("/")
        attrs = fighters_by_url.get(url_norm) or fighters_by_url.get(url_norm + "/") or {}
        if not attrs:
            attrs = next((v for v in fighters_by_url.values() if v.get("full_name") == name), {})
        return attrs or {"height": "", "reach": "", "stance": "Unknown", "date_of_birth": "", "full_name": name}

    def get_state(url, name):
        url_norm = (url or "").strip().rstrip("/")
        return state_snapshot.get(url_norm) or state_snapshot.get(url_norm + "/") or state_snapshot.get(name)

    fight = {
        "weight_class": weight_class or "Middleweight",
        "number_of_rounds": (number_of_rounds or "3").strip(),
    }
    event_dt = datetime.now()
    state1 = get_state(fighter1_url, fighter1_name)
    state2 = get_state(fighter2_url, fighter2_name)
    attrs1 = get_attrs(fighter1_url, fighter1_name)
    attrs2 = get_attrs(fighter2_url, fighter2_name)
    row = build_feature_row(fight, state1, state2, attrs1, attrs2, event_dt)

    preproc_bundle = joblib.load(PREPROC_PATH)
    model_bundle = joblib.load(MODEL_PATH)
    preprocessor = preproc_bundle["preprocessor"]
    model = model_bundle["model"]
    selector = model_bundle.get("selector")
    feature_cols = preproc_bundle["numeric_cols"] + preproc_bundle["cat_cols"]

    X = pd.DataFrame([row])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = pd.NA
    X = X[feature_cols]
    for c in preproc_bundle["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X_transformed = preprocessor.transform(X)
    if selector is not None:
        X_transformed = selector.transform(X_transformed)

    probas = model.predict_proba(X_transformed)[0]
    prob_f1 = float(probas[1])
    winner = fighter1_name if prob_f1 >= 0.5 else fighter2_name
    confidence = float(max(probas))
    return {"winner": winner, "confidence": round(confidence, 4), "prob_fighter1": round(prob_f1, 4)}


@app.route("/")
def index():
    fighters_data, ref_date = load_fighters_past_n_years(5)
    upcoming = []
    if UPCOMING_PREDICTIONS.exists():
        with open(UPCOMING_PREDICTIONS, newline="", encoding="utf-8") as f:
            upcoming = list(csv.DictReader(f))
    return render_template(
        "index.html",
        fighters=fighters_data,
        upcoming=upcoming,
        ref_date=ref_date.strftime("%B %d, %Y") if ref_date else "",
    )


def _load_upcoming_lookup() -> dict:
    """Build {frozenset({f1,f2}): {predicted_winner, confidence}} for matchups in upcoming."""
    lookup = {}
    if not UPCOMING_PREDICTIONS.exists():
        return lookup
    with open(UPCOMING_PREDICTIONS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = (row.get("fighter_1") or "").strip(), (row.get("fighter_2") or "").strip()
            if a and b:
                key = frozenset({a, b})
                conf = float(row.get("confidence", 0) or 0)
                lookup[key] = {"winner": row.get("predicted_winner", ""), "confidence": conf}
    return lookup


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    f1 = (data.get("fighter1") or "").strip()
    f2 = (data.get("fighter2") or "").strip()
    wc = (data.get("weight_class") or "Middleweight").strip()
    rounds = str(data.get("number_of_rounds") or "3").strip()
    if not f1 or not f2 or f1 == f2:
        return jsonify({"error": "Select two different fighters"}), 400
    fighters_data, _ = load_fighters_past_n_years(5)
    all_f = {x["name"]: x for lst in [fighters_data["men"], fighters_data["women"]] for x in lst}
    if f1 not in all_f or f2 not in all_f:
        return jsonify({"error": "Fighter(s) not found in database"}), 400

    # Use upcoming prediction if this exact matchup exists (same result as Upcoming Fights tab)
    upcoming_lookup = _load_upcoming_lookup()
    key = frozenset({f1, f2})
    if key in upcoming_lookup:
        return jsonify(upcoming_lookup[key])

    info1 = all_f[f1]
    info2 = all_f[f2]
    result = predict_matchup(f1, info1["url"], f2, info2["url"], wc, rounds)
    if result is None:
        return jsonify({"error": "Prediction failed (model or data missing)"}), 500
    return jsonify(result)


@app.route("/api/fighter-stats")
def api_fighter_stats():
    """Return stats for display: record, height, reach, etc. for both fighters."""
    f1_name = (request.args.get("fighter1") or "").strip()
    f2_name = (request.args.get("fighter2") or "").strip()
    if not f1_name or not f2_name:
        return jsonify({"error": "fighter1 and fighter2 required"}), 400
    fighters_data, _ = load_fighters_past_n_years(5)
    all_f = {x["name"]: x for lst in [fighters_data["men"], fighters_data["women"]] for x in lst}
    attrs_load = load_fighter_attrs()
    try:
        from feature_engineering import compute_fighter_state_snapshot
        state_snap = compute_fighter_state_snapshot(input_path=CLEAN_FIGHTS) if CLEAN_FIGHTS.exists() else {}
    except Exception:
        state_snap = {}
    out = {"fighter1": {}, "fighter2": {}}
    for i, (name, key) in enumerate([(f1_name, "fighter1"), (f2_name, "fighter2")]):
        if name not in all_f:
            continue
        info = all_f[name]
        url = info["url"]
        attrs = attrs_load.get(url.rstrip("/")) or attrs_load.get(url) or {}
        state = state_snap.get(url.rstrip("/")) or state_snap.get(url) or state_snap.get(name) or {}
        rec = get_fighter_record_from_state(state)
        h = attrs.get("height") or ""
        rch = attrs.get("reach") or ""
        stance = attrs.get("stance") or "Unknown"
        height_str = _inches_to_ft_in(h) if h else ""
        reach_str = f"{rch} in" if rch and str(rch).isdigit() else ""
        out[key] = {
            "name": name,
            "record": rec,
            "height": height_str,
            "reach": reach_str,
            "stance": stance,
        }
    return jsonify(out)


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
