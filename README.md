# UFC Fight Prediction Model 2.0

Modular pipeline (run in order): **module_01** → **module_02** → **module_03** → **module_04** → **module_05** → **module_06** → **module_07**.

## Folder layout

```
module_01_scrapers/     # 1. Fetch raw data from UFCStats.com
  output/               # raw_ufc_fights.csv, raw_ufc_fighters.csv, raw_ufc_upcoming.csv

module_02_clean_fighters/   # 2. Normalize fighter CSV
  input/  output/          # clean_ufc_fighters.csv

module_03_clean_fights/     # 3. Join fighters + clean fight CSV
  input/  output/          # clean_ufc_fights.csv

module_04_feature_engineering/   # 4. Feature engineering
  input/  output/          # ufc_fights_fe.csv

module_05_split/           # 5. Prep + temporal split
  input/  output/          # preprocessor_diff.joblib, X_train_diff.npz, etc.

module_06_model/           # 6. Full grid search, pick best model
  input/  output/          # best_model.joblib, preprocessor_diff.joblib
                           # → prepares upcoming_for_prediction.joblib for module_07

module_07_predict/         # 7. Predict upcoming fights
  input/                   # upcoming_for_prediction.joblib (from module_06)
  output/                  # upcoming_predictions.csv
```

## Pipeline (run in order)

**Option: run everything at once**
```bash
python run_pipeline.py
```

**Or run step by step:**

1. **Scrape** (past incremental, fighters incremental, upcoming all):
   ```bash
   python module_01_scrapers/ufc_fight_scraper.py --incremental
   python module_01_scrapers/ufc_fighter_scraper.py --incremental
   python module_01_scrapers/ufc_upcoming_scraper.py
   ```

2. **Copy** scraped files into module_02 and module_03 `input/`, then **clean**:
   ```bash
   python module_02_clean_fighters/clean_ufc_fighters.py
   python module_03_clean_fights/clean_ufc_fights.py
   ```

3. **Feature engineering**
   ```bash
   python module_04_feature_engineering/feature_engineering.py
   ```

4. **Split**
   ```bash
   python module_05_split/prep_and_split.py -d
   ```

5. **Train** (full grid search: LR, RF, ET, GB, XGB, AdaBoost, SVC, MLP; picks best by validation)
   ```bash
   python module_06_model/train_tuned.py
   ```
   Also prepares upcoming fights → `module_07_predict/input/upcoming_for_prediction.joblib`

6. **Predict**
   ```bash
   python module_07_predict/predict_upcoming.py
   ```
   Output: `module_07_predict/output/upcoming_predictions.csv`

## Scraper options

- **Fight scraper:** `--incremental` to only add events after the latest in the CSV; `--max-events N` for testing.
- **Fighter scraper:** `--incremental` to only add new fighter URLs; `--max-per-letter N`, `--letters abc` for testing.
- **Upcoming scraper:** scrapes all upcoming events by default; `--max-events N` to limit.
