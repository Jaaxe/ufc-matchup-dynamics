# Uncovering Fighting Styles and Matchup Asymmetries in the UFC via Unsupervised and Self-Supervised Learning

**Author:** Jarvis Xie  
**Advisor:** Robert Wooster, Department of Statistics and Data Science  
**Date:** 2025-03-22

---

## Introduction and Motivation

In professional mixed martial arts (MMA), fighters are routinely described using informal labels—"striker," "grappler," "pressure fighter," "wrestle-boxer"—that fans and analysts use to explain matchup outcomes. However, these labels are subjective, loosely defined, and rarely grounded in data. They are applied inconsistently across fighters and fail to capture the continuous spectrum of fighting behavior.

Most existing UFC analytics focus on predicting win-loss outcomes using hand-engineered features—strike rates, takedown percentages, control time—without explicitly modeling *style* as a latent factor. As a result, these models often miss the underlying dynamics that determine *why* certain matchups favor one fighter over another. A dominant striker may lose to a grappler not because of raw skill, but because of stylistic mismatch: the grappler's style exploits the striker's weakness.

This project is motivated by the idea that fighting style is a real but unobserved factor that shapes both fight outcomes and fight dynamics. Rather than imposing predefined categories, we aim to *discover* fighting styles directly from UFC fight data using unsupervised and self-supervised learning. This enables a data-driven understanding of matchup asymmetries, stylistic mismatches, and cases where a favored fighter loses to an underdog whose style creates an unfavorable dynamic.

---

## Project Overview and Core Goals

The project pursues four interlocking goals:

1. **Novel fighter profile classification** — Discover latent fighting styles from fight statistics without predefined labels. Unlike conventional "striker vs. grappler" taxonomies, these styles emerge from the data and capture nuanced behavioral patterns (e.g., high-volume strikers, control-oriented wrestlers, submission threats, hybrid generalists).
2. **Outcome prediction** — Predict who wins a fight using style-aware representations. The key distinction from prior work is that our predictors use learned embeddings or cluster assignments that encode *how* fighters fight, not just aggregate statistics.
3. **Fight dynamics prediction** — Predict *what a fight will look like* before it happens: strike vs. grapple mix, control-time distribution, likelihood of a finish vs. decision, and method of victory. This goes beyond outcome prediction to characterize the expected flow and structure of the contest.
4. **Upset prediction** — Identify when and why favored fighters lose to stylistically mismatched opponents. We treat upsets not as noise but as interpretable events: a grappler exploits a striker's weak takedown defense, or a pressure fighter nullifies a counter-striker's range. This has practical implications for betting, fan understanding, and fight promotion.

---

## Problem Statement

The specific problems addressed in this project are:

- **Can latent fighting styles be learned directly from UFC fight statistics?** Existing work either uses hand-crafted style labels or ignores style entirely. We test whether unsupervised and self-supervised methods can recover meaningful stylistic structure from per-fight metrics.
- **Do these learned styles explain matchup dynamics that simple win-loss models miss?** We ask whether style-based representations improve prediction accuracy and, more importantly, whether they capture *asymmetric* interactions—e.g., Style A tends to beat Style B even when other factors (age, experience, raw stats) are controlled.
- **Can we predict stylistic upsets?** When a fighter who is otherwise expected to win instead loses, is the loss explainable by stylistic mismatch? Can we identify high-risk favorites whose style leaves them vulnerable to particular opponents?

---

## Methodology

The project emphasizes modern representation learning techniques designed to capture structure in sequential and relational data, rather than relying solely on static feature engineering.

### Representation

Fighters and fights are represented as vectors derived from per-fight statistics:

- **Striking:** Significant strikes landed/attempted per minute, positional breakdown (distance, clinch, ground)
- **Grappling:** Takedowns attempted per minute, submission attempts, control time
- **Context:** Fight duration, weight class, opponent strength

All metrics are normalized to account for fight length and weight-class context. Fight-level data are aggregated into fighter-level profiles (career averages) and, where applicable, used to construct sequential representations of a fighter's fight history.

### Style Discovery

Three complementary approaches are used to learn stylistic representations:


| Method                            | Description                                                                                                                                            | Output                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| **K-Means**                       | Hard clustering of fighters in feature space. Each fighter is assigned to a single style cluster.                                                      | Discrete cluster labels (e.g., k=5)                                         |
| **Gaussian Mixture Models (GMM)** | Soft clustering that assigns each fighter a probability distribution over styles. Captures hybrids (e.g., 60% striker, 40% grappler).                  | Cluster assignments plus *Hybrid Score* (entropy of the probability vector) |
| **Autoencoders**                  | Neural networks that compress fighter stats into a low-dimensional bottleneck and reconstruct them. The bottleneck layer yields continuous embeddings. | Dense embeddings (e.g., 3-D latent vectors)                                 |


These models are chosen for their ability to capture nonlinear structure and variability in fighting behavior. GMM and autoencoders, in particular, allow for continuous or probabilistic style assignments, reflecting the reality that fighters rarely fit neatly into a single category.

### Matchup Modeling

Once styles or embeddings are learned, they are used to analyze and predict matchup behavior:

- **Style interaction matrices** — For each pair of style clusters (e.g., Striker vs. Grappler), compute win rates, sample sizes, and significance tests (e.g., chi-square). This reveals which styles tend to beat which.
- **Hybrid analysis** — Test whether "hybrid" fighters (high entropy, generalists) perform better than specialists, using statistical tests on win rates vs. hybrid score.
- **Unified matchup table** — Notebook **18** produces a single canonical feature matrix (`ufc_matchup_features.csv`) with **leakage-free** pre-fight rolling aggregates, Elo / Glicko-2 ratings walked in chronological order, GMM posteriors, autoencoder embeddings, physical attributes, weight-class one-hots, empirical heatmap probabilities, de-vigged Vegas probabilities, and an event-ordered 60/20/20 `split` column.
- **Supervised prediction** — Notebook **19** trains a bakeoff of six model families (logistic regression, random forest, histogram gradient boosting, XGBoost, LightGBM, MLP) plus an **isotonic stacking** meta-learner over seven nested feature-set ablations (`z_only → +gmm → +ae → +rolling → +ratings → +physical_wc → full → full+vegas`). All models share the same split and corner symmetry augmentation so metrics are directly comparable.
- **Vegas comparison** — Notebook **20** benchmarks the project's stack against de-vigged Vegas probabilities on the objective target `Win_A`, and breaks the calibration down by weight class, k=5 style-pair cell, and hybrid-score quintile.

### Downstream Tasks and Interpretability

Embeddings, cluster assignments, and rolling pre-fight features feed four linked downstream tasks:

- **Outcome prediction** — notebook **19** (bakeoff + stack) and **20** (Vegas comparison on `Win_A`).
- **Fight dynamics prediction** — notebook **21**: binary finish, 3-way method (KO/SUB/DEC), a hierarchical 6-way decomposition `P(finish) × P(method | finish) × P(winner | method)`, and regressions for duration / sig strikes / takedowns / control / knockdowns / submission attempts / ground-strike share. Includes the **style-divergence → finish-rate** thesis plot.
- **Upset prediction** — notebook **22**: Vegas-as-feature residual model with ROC-AUC, top-*k* lift, a de-vigged ROI simulation, and grouped permutation importance over feature groups from `ufc_feature_groups.json`.
- **Sequence modeling (stretch)** — notebook **23**: a Siamese GRU over each fighter's last 8 pre-fight histories plus tabular context, compared against the NB 19 HGB baseline on identical test events.

The project emphasizes interpretability: we aim for models that capture *meaningful* stylistic structure rather than noise. Visualization of clusters, embedding spaces, interaction matrices, reliability curves, and grouped feature-importance bars is central to this goal.

---

## Data

### Source

The project uses publicly available UFC fight datasets compiled from official UFC statistics. Specifically:

- **Origin:** UFC Fights, Fighters, and Events Dataset (Kaggle)  
- **Author:** Amine Alibi  
- **Description:** Scraped UFC Stats including match results, fighter physical attributes, and event details.

### Included Metrics

**Fight-level information:**

- Significant strikes landed/attempted (overall and by position: distance, clinch, ground)
- Takedowns attempted/landed
- Submission attempts
- Control time (seconds)
- Fight duration
- Method of victory (KO/TKO, submission, decision, etc.)

**Fighter-level metadata:**

- Age, height, reach
- Stance (orthodox, southpaw, switch)
- Weight class

### Data Preparation Tasks

- **Cleaning:** Handling missing or inconsistent values, normalizing dates and weight classes, merging fight-level and fighter-level data.
- **Aggregation:** Aggregating fight-level data into fighter-level summaries (career averages for each metric).
- **Normalization:** Z-scoring key statistics (e.g., strikes per minute, takedown attempts) relative to weight-class averages to account for division-specific baselines.
- **Bias mitigation:** Filtering fighters with limited fight history (e.g., fewer than 5 fights) to avoid introducing noise or spurious clusters from small samples.
- **Sequential representation (if applicable):** Constructing ordered sequences of per-fight stats for each fighter to support temporal or sequence-based models.

### Supplementary Data

- **Betting odds:** Kaggle UFC betting odds CSV under `data/raw/kaggle_odds/`; cleaned output is documented in `data/README.md` and produced in notebook **14**, then consumed by the Vegas baseline in notebook **15**, as a joined column in the unified feature table in **18**, for the objective Vegas comparison in **20**, and as a feature (and as the reference distribution for the ROI simulation) in the upset-prediction pipeline in **22**.
- Round-level statistics for finer-grained dynamics — *optional extension*

### File Structure

See `data/README.md` for full details. Key outputs:

- `ufc_fight_stats_cleaned.csv` — Cleaned fight-level data with outcomes and merged stats
- `ufc_modeling_data_final.csv` — Fighter-level style profiles with Z-scored features
- `ufc_gmm_comparison.csv` — GMM cluster assignments and Hybrid Scores
- `ufc_clusters_baseline.csv` — K-Means cluster assignments
- `ufc_fight_odds_clean.csv` — De-vigged US moneyline implied probabilities per fight (notebook **14**, requires raw Kaggle odds)

---

## Repository Structure

### Notebooks (`notebooks/`)

The analysis pipeline is organized as a sequential workflow in five parts. Run notebooks in numerical order (**01–23**) from the `notebooks/` directory (paths use `../data/`). **Interpretability:** Each notebook includes goal statements, figure interpretations, and takeaway sections. **Code cells:** Every code cell begins with a short **header** (notebook name, cell index, linked markdown section) and a **workflow summary** of how to read that cell. Shared helpers for the unified modeling stack live in `scripts/matchup_utils.py` (rolling pre-fight aggregates, Elo/Glicko-2, event splits, weight-class one-hots, corner symmetrization).

- **Part I — Data and EDA (01–07).** Cleaning, fighter-level aggregation, bias checks, comprehensive EDA, and matchup-level deltas.
- **Part II — Style Discovery (08–12).** K-Means, GMM (selection → final fit with soft posteriors and Hybrid Scores), style-pair interaction matrix, and autoencoder embeddings.
- **Part III — First-pass prediction (13–17, legacy).** Initial upset analysis, odds cleaning, and first-cut Vegas / boosting / calibration comparisons. Retained for continuity; **superseded** by Part V on the objective `Win_A` target and a single unified feature matrix.
- **Part IV — Unified feature construction (18).** Builds the canonical `ufc_matchup_features.csv` table with leakage-free pre-fight rolling rates, Elo, Glicko-2, GMM posteriors, AE embeddings, physicals, Weight_Class one-hots, heatmap, and Vegas, plus an event-ordered 60/20/20 `split` column. All of Part V reads this one file.
- **Part V — Unified modeling stack (19–23).** Win/loss bakeoff + isotonic stacking, Vegas comparison on `Win_A` with groupwise calibration, fight dynamics suite, upset prediction with Vegas as a feature, and an optional Siamese GRU sequence model.


| #   | Notebook                            | Purpose                                                                                                                                       |
| --- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| *Part I — Data and EDA*                   |||
| 01  | `01_data_cleaning.ipynb`            | Load raw fight data, parse per-fighter stats, output cleaned fight-level dataset                                                              |
| 02  | `02_feature_engineering.ipynb`      | Aggregate fights into fighter-level style profiles (per-minute rates, positional ratios, control ratio)                                       |
| 03  | `03_bias_checks.ipynb`              | Validate weight-class bias and sample-size stability; informs min-fights filter and normalization                                            |
| 04  | `04_eda_external_data.ipynb`        | Explore physical attributes (ape index, stance) and external style labels                                                                     |
| 05  | `05_finalize_dataset.ipynb`         | Apply experience filter (min 5 fights), weight-class Z-score normalization, produce final modeling dataset; includes normalization validation |
| 06  | `06_eda_comprehensive.ipynb`        | Master EDA: temporal trends, weight-class bias, style space, damage anatomy, referee effects, glass-cannon analysis                         |
| 07  | `07_matchup_analysis.ipynb`         | Matchup-level analysis: advantage deltas, correlation with outcomes, style divergence vs. finish rate                                       |
| *Part II — Style Discovery*                |||
| 08  | `08_kmeans_style_discovery.ipynb`   | K-Means clustering (k=4,5,6); elbow/silhouette model selection; cluster interpretation                                                         |
| 09  | `09_gmm_model_selection.ipynb`      | GMM model selection via AIC/BIC; introduces soft clustering and Hybrid Score concept                                                          |
| 10  | `10_gmm_style_discovery.ipynb`      | Fit GMM at k=3 and k=5; save cluster assignments, Hybrid Scores, **and soft posteriors `pk3_*`/`pk5_*`** to `ufc_gmm_comparison.csv`          |
| 11  | `11_style_interaction_matrix.ipynb` | Core matchup asymmetry: win-rate matrix by style pair, hybrid vs. specialist analysis, chi-square tests                                       |
| 12  | `12_autoencoder_supervised.ipynb`   | Autoencoder embeddings; supervised outcome prediction using embedding differences (Random Forest); **persists 3-D embeddings to `ufc_ae_embeddings.csv`** |
| *Part III — First-pass prediction (legacy)* |||
| 13  | `13_upset_analysis.ipynb`           | Elo-based "favorite," style heatmap vs upsets, Z-delta RF with **event-based** train/test split                                             |
| 14  | `14_odds_cleaning.ipynb`            | Clean Kaggle UFC odds → one row per `Fight_Id`, US market, de-vig implied probs → `ufc_fight_odds_clean.csv`                                  |
| 15  | `15_vegas_vs_predictions.ipynb`    | *(Legacy)* Compare Vegas implied **P(Vegas favorite wins)** to heatmap, Z-RF, and AE+RF on the **same** test events as notebook 13. Superseded by NB 20. |
| 16  | `16_fight_dynamics_vegas.ipynb`    | *(Legacy)* 4-feature 6-way method props vs `HistGradientBoosting`; **finish vs decision** vs Vegas. Superseded by NB 21.                    |
| 17  | `17_boosting_calibration_winloss.ipynb` | *(Legacy)* **Win/loss baselines** on 4 Z-deltas. Superseded by NB 19's ablation ladder.                                                 |
| *Part IV — Unified feature construction*  |||
| **18**  | `18_build_matchup_features.ipynb`  | **Unified feature table.** Produces `ufc_matchup_features.csv` + `ufc_fighter_profiles_rolling.csv` + `ufc_feature_groups.json`. Combines rolling pre-fight rates (leakage-free), Elo, Glicko-2, GMM posteriors, AE embeddings, physical attributes, Weight_Class one-hots, Vegas de-vigged probs, and `split` column (train/val/test by event date, 60/20/20). |
| *Part V — Unified modeling stack*         |||
| **19**  | `19_winloss_bakeoff.ipynb`         | **Win/loss bakeoff.** LR/RF/HGB/XGB/LightGBM/MLP + **isotonic stack**. Seven feature ablations (`z_only → +gmm → +ae → +rolling → +ratings → +physical_wc → full → full+vegas`). Walk-forward CV on the train+val portion. Reports AUC, Brier, log-loss, accuracy, ECE on val and test. |
| **20**  | `20_vegas_winloss_comparison.ipynb`| **Vegas vs the project** on the objective label `Win_A` (fixes NB 15's circular target). **Groupwise calibration** by Weight_Class, k=5 cluster pair, and Hybrid-score quintile. |
| **21**  | `21_dynamics_suite.ipynb`          | **Fight dynamics.** Binary finish, 3-way method (KO/SUB/DEC), 6-way method with **hierarchical decomposition** `P(finish) × P(method\|finish) × P(winner\|method)`, and regressions for duration / sig strikes / takedowns / control / knockdowns / submission attempts / ground-strike share. Includes the **style-divergence → finish-rate** thesis plot. |
| **22**  | `22_upset_prediction_vegas.ipynb`  | **Upset prediction.** Treats Vegas as a **feature** (idea #2) and uses the signal `r = p_model(Win_A) − p_vegas_A` as an upset detector. Reports ROC-AUC, top-k lift, a de-vigged ROI simulation, and grouped permutation importance over feature groups. |
| **23**  | `23_sequence_model.ipynb`          | **Sequence model** (optional stretch). Siamese GRU over last-8 pre-fight histories per corner + tabular context. Honest baseline comparison against HGB on identical test events. |


### Running the Pipeline

From the project root:

```bash
python run_pipeline.py
```

This executes all notebooks (**01–23**) in numerical order. Notebook **12** uses **PyTorch** for the autoencoder; **14–16**, **20**, **22** consume `data/raw/kaggle_odds/UFC_betting_odds.csv`. Notebook **17** optionally uses **XGBoost** (`pip install xgboost`); **19** optionally uses **LightGBM** (`pip install lightgbm`). Notebook **23** requires **PyTorch**. Notebooks **15–17** are retained for continuity but are superseded by **19–22**, which load a single unified feature matrix produced by **18** and use a consistent 60/20/20 event-ordered split.

### Data (`data/`)

- `raw/` — Original UFC datasets (plus optional `kaggle_odds/`)
- `processed/` — Cleaned data, fighter profiles, style-discovery outputs, the unified matchup table (`ufc_matchup_features.csv`), rolling profiles, autoencoder embeddings, and feature-group metadata. See `data/README.md` for the full artifact map.

### LaTeX Thesis Draft (`latex/`)

The thesis report compiles with `XeLaTeX` (or any LaTeX engine with `fontspec`) from `latex/main.tex`, e.g. `latexmk -xelatex main.tex` inside `latex/`. See the existing `latexmkrc` for the template's default build configuration.

---

## Deliverables


| Deliverable                       | Status      | Description                                                                                                         |
| --------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| **Cleaned UFC dataset**           | ✓ Done      | Well-documented, reproducible pipeline from raw data to modeling-ready tables                                       |
| **Fighter style representations** | ✓ Done      | K-Means clusters, GMM probabilistic assignments, autoencoder embeddings                                             |
| **Matchup analysis**              | ✓ Done      | Style interaction matrices, win rates by style pair, hybrid vs. specialist analysis, statistical significance tests |
| **Fight dynamics models**         | ✓ Done      | Notebook **21**: binary finish, 3-way and hierarchical 6-way method, duration / strike / takedown regressions, style-divergence → finish-rate plot |
| **Upset prediction**              | ✓ Done      | Notebook **22**: Vegas-as-feature residual model with ROC / lift / ROI vs Elo-residual and heatmap-residual baselines |
| **Win-loss bakeoff**              | ✓ Done      | Notebook **19**: 7-way feature ablation × 6 model families + isotonic stack, walk-forward CV on train+val          |
| **Vegas comparison**              | ✓ Done      | Notebook **20**: `Win_A` objective target, groupwise calibration by weight class / cluster pair / hybrid quintile  |
| **Visualizations**                | ✓ Done      | Cluster plots, embedding visualizations, matchup heatmaps, reliability curves, grouped permutation-importance bars, style-divergence → finish-rate plot, lift / ROI curves (NB 11, 12, 19, 20, 21, 22) |
| **Final report and poster**       | In progress | Thesis LaTeX draft lives in `latex/` (see `latex/main.tex`); poster outstanding                                     |


---

## Novelty and Contributions

This project contributes:

1. **Data-driven style discovery** — Fighting styles are learned from data rather than imposed. No predefined labels (e.g., "striker") are required; the model discovers structure that may or may not align with conventional categories.
2. **Embedding-based representation** — We use neural embeddings and probabilistic cluster assignments instead of relying solely on hand-engineered aggregates. This allows for nonlinear relationships and continuous style spectra.
3. **Unified framework** — A single pipeline supports outcome prediction, fight dynamics prediction, and upset identification. Style representations serve all three tasks, enabling interpretable connections between how fighters fight and how matchups unfold.
4. **Matchup asymmetry focus** — We explicitly model *asymmetric* style interactions—Style A may beat Style B more often than B beats A—rather than treating matchups as symmetric. This aligns with how analysts and fans reason about fights.

---

## Timeline


| Phase                  | Weeks        | Activities                                                                                                     |
| ---------------------- | ------------ | -------------------------------------------------------------------------------------------------------------- |
| **Status & proposal**  | 1–2          | Status update, project pre-approval, proposal submission                                                       |
| **Data & EDA**         | 3–5          | Data acquisition, cleaning, exploratory data analysis; begin writing alongside analysis                        |
| **Style discovery**    | 6–8          | Unsupervised and self-supervised style discovery (K-Means, GMM, autoencoder); continued writing and refinement |
| **Matchup & dynamics** | 9–10         | Matchup analysis, fight-structure modeling, upset-prediction exploration; finalize main results                |
| **First draft**        | 11           | Finalize first draft of report and poster; submit to advisor for feedback                                      |
| **Revision**           | 12–13        | Revise report and poster based on advisor feedback                                                             |
| **Submission**         | Reading week | Finalize report and submit                                                                                     |


