# Data Directory

This folder contains the raw, processed, and modeling datasets for the UFC Matchup Dynamics & Style Discovery project.

## 1. Data Sources

* **Primary (fights / fighters / events):** UFC Fights, Fighters, and Events Dataset (Kaggle) — *Author:* Amine Alibi — scraped UFC Stats (match results, attributes, event metadata).
* **Optional (betting):** UFC betting odds CSV (Kaggle or equivalent) placed under `raw/kaggle_odds/` — used by notebook **14** to build `processed/ufc_fight_odds_clean.csv`, then by notebook **15** for Vegas baselines. See root `README.md` for the full notebook list.

## 2. File Structure

### Raw Data (`/data/raw/`)

* `ufc_fighters.csv` — Fighter profiles (height, reach, stance, DOB).
* `ufc_fight_stats.csv` — Round-by-round statistics (strikes, takedowns, control time).
* `ufc_events.csv` — Event metadata (date, location, attendance).
* `raw_fights_detailed.csv` — Detailed fight export used by notebook **01** (paths may vary; see that notebook’s `INPUT_PATH`).
* **`kaggle_odds/UFC_betting_odds.csv`** *(optional)* — Multi-row moneyline history. Notebook **14** filters to `region == 'us'`, valid decimal odds, fights that appear in `ufc_fight_stats_cleaned.csv`, and **one row per `Fight_Id`** (latest `adding_date` in the file; see notebook markdown for why event-vs-scrape dates are handled that way).

### Processed Data (`/data/processed/`)

* **`ufc_fight_stats_cleaned.csv`** — Fight-level table: merged stats, outcomes, identifiers (`Fight_Id`, `Event_Id_x`). Produced by **01**; consumed by nearly all downstream notebooks.
* **`ufc_modeling_data_final.csv`** — One row per qualified fighter: career aggregates, weight-class Z-scores, positional ratios. Produced by **05**.
* **`ufc_fight_odds_clean.csv`** *(optional)* — One row per `Fight_Id`: `fighter_1`/`fighter_2`, decimal `odds_1`/`odds_2`, de-vigged `p_fighter_1`/`p_fighter_2`, `event_date`, `adding_date`, `source`, `region`. Produced by **14** when raw odds exist.

### Model Outputs (`/data/processed/`)

* `ufc_clusters_baseline.csv` — K-Means cluster assignments (e.g. k=5), from **08**.
* `ufc_gmm_comparison.csv` — GMM soft assignments, Hybrid Scores, and style columns used in **10** and later notebooks.

## 3. Key Feature Definitions

* **`Sig_Str_PM_Z`** — Significant strikes per minute, Z-scored within weight class.
* **`Control_Ratio`** — Share of fight time in dominant control positions.
* **`Hybrid_Score`** — Entropy of the GMM probability vector: high values suggest a generalist blend of styles; low values suggest a specialist.

## 4. Notebook ↔ Artifact Map (quick reference)

| Artifact | Typical producer notebook |
|----------|---------------------------|
| `ufc_fight_stats_cleaned.csv` | 01 |
| Fighter-level aggregates / modeling table | 02 → 05 |
| `ufc_gmm_comparison.csv` | 10 |
| `ufc_fight_odds_clean.csv` | 14 |
| Figures / tables only (no new CSV) | 03, 04, 06, 07, 09, 11–13, 15–17 |

All notebook **code cells** include a standard header and workflow summary comments; see root `README.md` under **Notebooks**.
