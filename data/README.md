# Data Directory

This folder contains the raw, processed, and modeling datasets for the UFC Matchup Dynamics & Style Discovery project.

## 1. Data Sources

* **Primary (fights / fighters / events):** UFC Fights, Fighters, and Events Dataset (Kaggle) — *Author:* Amine Alibi — scraped UFC Stats (match results, attributes, event metadata).
* **Optional (betting):** UFC betting odds CSV (Kaggle or equivalent) placed under `raw/kaggle_odds/` — used by notebook **14** to build `processed/ufc_fight_odds_clean.csv`, then by notebooks **15**, **20**, and **22** for Vegas baselines and as a feature in the upset-prediction pipeline. See root `README.md` for the full notebook list.

## 2. File Structure

### Raw Data (`/data/raw/`)

Tracked paths include only `.gitkeep`; after cloning, place the Kaggle UFC export files here (see list below). Same for `processed/` until notebooks populate outputs.

* `raw_fighters.csv` — Fighter profiles (height, reach, stance, DOB).
* `raw_fights.csv` — Fight-level rows before detailed stats are attached.
* `raw_fights_detailed.csv` — Detailed fight export used by notebook **01** (paths may vary; see that notebook's `INPUT_PATH`).
* `raw_details.csv` — Round-by-round / detail table used in the cleaning step.
* `raw_events.csv` — Event metadata (date, location). Joined throughout the modeling stack to attach `Event_Date` and produce chronological splits.
* **`kaggle_odds/UFC_betting_odds.csv`** *(optional)* — Multi-row moneyline history. Notebook **14** filters to `region == 'us'`, valid decimal odds, fights that appear in `ufc_fight_stats_cleaned.csv`, and **one row per `Fight_Id`** (latest `adding_date` in the file; see notebook markdown for why event-vs-scrape dates are handled that way).

### Processed Data (`/data/processed/`)

In a fresh clone only `.gitkeep` is tracked here; CSV/JSON outputs appear after you run the notebooks.

Tables produced by the exploratory stack (NB 01–17):

* **`ufc_fight_stats_cleaned.csv`** — Fight-level table: merged stats, outcomes, identifiers (`Fight_Id`, `Event_Id_x`). Produced by **01**; consumed by nearly all downstream notebooks.
* **`ufc_modeling_data_final.csv`** — One row per qualified fighter: career aggregates, weight-class Z-scores, positional ratios. Produced by **05**.
* **`ufc_fighter_style_features.csv`** — Intermediate fighter-level style feature table from **02**.
* **`ufc_fight_odds_clean.csv`** *(optional)* — One row per `Fight_Id`: `fighter_1`/`fighter_2`, decimal `odds_1`/`odds_2`, de-vigged `p_fighter_1`/`p_fighter_2`, `event_date`, `adding_date`, `source`, `region`. Produced by **14** when raw odds exist.

### Style Discovery Outputs (`/data/processed/`)

* `ufc_clusters_baseline.csv` — K-Means cluster assignments (e.g. k=5), from **08**.
* `ufc_gmm_comparison.csv` — GMM hard labels, Hybrid Scores, and **soft posteriors `pk3_0..pk3_2` and `pk5_0..pk5_4`** for each qualified fighter. Produced by **10**.
* `ufc_styles_probabilistic.csv` — Complementary probabilistic style table used by the EDA notebooks.
* `ufc_ae_embeddings.csv` — Three-dimensional autoencoder embeddings (`Emb_1, Emb_2, Emb_3`) keyed by `Fighter`. Produced by **12**.

### Unified Modeling Stack Artifacts (`/data/processed/`, NB 18–23)

* **`ufc_fighter_profiles_rolling.csv`** — One row per fight, from the A-corner perspective, containing **leakage-free pre-fight rolling aggregates** (career fights, win rate, KO/SUB/DEC rates, sig-strike / takedown / submission / control / knockdown per-minute rates, accuracy, positional share, recent-5 form) and `days_since_last_A/B`. Produced by **18** via `scripts/matchup_utils.build_rolling_profiles`.
* **`ufc_matchup_features.csv`** — **Canonical matchup table** consumed by NB 19–23. One row per unique `Fight_Id` with:
  * identifiers: `Fight_Id`, `Event_Id_x`, `Event_Date`, `Fighter_A`, `Fighter_B`, `Weight_Class`, `Win_A`;
  * pre-fight form: the rolling columns above for both corners;
  * style: `A_pk3_*`, `B_pk3_*`, `A_pk5_*`, `B_pk5_*` GMM posteriors, Hybrid Scores, and Z-scored style features for each corner;
  * matchup transforms: per-feature `delta_<x>` (A − B) and `mean_<x>` columns;
  * embeddings: `A_Emb_1..3`, `B_Emb_1..3`, their deltas and means;
  * physical attributes: height / reach / age / stance one-hots per corner plus deltas;
  * context: `Weight_Class` one-hots, `title_fight`, `is_5rd`;
  * ratings: chronological Elo (`A_elo_pre`, `B_elo_pre`, `delta_elo`) and Glicko-2 (`A_glicko_pre`, `B_glicko_pre`, `A_rd_pre`, `B_rd_pre`, `delta_glicko`);
  * empirical style-pair heatmap probability `p_heatmap_A`;
  * Vegas: `p_vegas_A` (de-vigged, **only present when odds are available**);
  * `split` ∈ `{train, val, test}` from an event-ordered **60/20/20** cut.
* **`ufc_feature_groups.json`** — Named feature groups (e.g. `style_gmm_probs`, `hybrid`, `ae_embedding`, `rolling_rates`, `form`, `physical`, `weight_class`, `ratings`, `heatmap`, `vegas`, `z_style`) mapping to the corresponding columns in `ufc_matchup_features.csv`. Used by NB 19's ablation ladder and NB 22's grouped permutation importance.

## 3. Key Feature Definitions

* **`Sig_Str_PM_Z`** — Significant strikes per minute, Z-scored within weight class.
* **`Control_Ratio`** — Share of fight time in dominant control positions.
* **`Hybrid_Score`** — Entropy of the GMM probability vector: high values suggest a generalist blend of styles; low values suggest a specialist.
* **`pk{K}_{i}`** — Soft posterior probability that a fighter belongs to latent style *i* under a GMM with *K* components (K ∈ {3, 5}). Rows sum to 1 for a given *K*.
* **`A_pre_*` / `B_pre_*`** — Strictly pre-fight rolling career means for fighter A / B (no leakage: the current fight is **not** included in these aggregates).
* **`A_recent5_*` / `B_recent5_*`** — Same, but computed over each fighter's last five pre-fight bouts.
* **`delta_*`** / **`mean_*`** — Matchup transforms (A − B and (A+B)/2) of any per-corner feature.
* **`elo_pre`**, **`glicko_pre`**, **`rd_pre`** — Ratings computed by walking fights in chronological order; the "pre" suffix indicates the rating **before** the current fight is resolved.
* **`p_heatmap_A`** — Empirical win rate of A's `Cluster_k5` vs B's `Cluster_k5`, computed across all fights with both fighters' static GMM hard assignments available (notebook 18 builds the pivot on the full styled-fight table). It is therefore a static, retrospective cluster-pair descriptor, not a strict pre-fight feature; see Section 3.3 ("Leakage scope and feature blocks") of the thesis.
* **`p_vegas_A`** — Implied win probability for fighter A after removing the bookmaker vig from US moneylines.

## 4. Notebook ↔ Artifact Map (quick reference)

| Artifact | Producer notebook(s) | Main consumers |
|----------|----------------------|----------------|
| `ufc_fight_stats_cleaned.csv` | **01** | 02, 05–13, 18 |
| `ufc_modeling_data_final.csv` | **05** | 06–13, 18 |
| `ufc_clusters_baseline.csv` | **08** | 11, 13 |
| `ufc_gmm_comparison.csv` | **10** | 11, 13, 18 |
| `ufc_ae_embeddings.csv` | **12** | 18 |
| `ufc_fight_odds_clean.csv` | **14** | 15, 18, 20, 22 |
| `ufc_fighter_profiles_rolling.csv` | **18** | 19–23 |
| `ufc_matchup_features.csv` | **18** | **19, 20, 21, 22, 23** |
| `ufc_feature_groups.json` | **18** | 19 (ablation ladder), 22 (grouped permutation importance) |
| Figures / tables only (no new CSV) | 03, 04, 06, 07, 09, 11, 13, 15–17, 19–23 | — |

Supplementary thesis notebooks **24** and **25** read existing processed tables and do not introduce new canonical CSVs beyond figures/tables.
