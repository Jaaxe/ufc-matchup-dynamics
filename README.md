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
- **Supervised prediction** — Construct matchup-level features (e.g., embedding differences between Fighter A and Fighter B) and train classifiers (**Random Forest**, **gradient boosting**, optional **XGBoost**, **logistic regression**, **calibration**; see notebook **17**) to predict fight outcomes. Compare against baselines that use only raw statistics (and **Vegas** where odds exist; notebooks **15–16**).

### Downstream Tasks and Interpretability

Embeddings and cluster assignments are used for:

- Outcome prediction (who wins)
- Fight dynamics prediction — notebook **16** (method props vs boosting; finish vs decision); finer strike/grapple mix targets remain *optional*
- Upset / favorite analysis — notebooks **13** (Elo favorite) and **15** (Vegas favorite vs models where odds exist)

The project emphasizes interpretability: we aim for models that capture *meaningful* stylistic structure rather than noise. Visualization of clusters, embedding spaces, and interaction matrices is central to this goal.

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

- **Betting odds:** Kaggle UFC betting odds CSV under `data/raw/kaggle_odds/`; cleaned output is documented in `data/README.md` and produced in notebook **14**, then used for Vegas baselines in **15**.
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

The analysis pipeline is organized as a sequential workflow. Run notebooks in numerical order (**01–17**) from the `notebooks/` directory (paths use `../data/`). All notebooks are read-only except those that write pipeline outputs; running the full sequence reproduces processed artifacts when inputs are present. **Interpretability:** Each notebook includes goal statements, figure interpretations, and takeaway sections. **Code cells:** Every code cell begins with a short **header** (notebook name, cell index, linked markdown section) and a **workflow summary** of how to read that cell; additional `# ---` section banners appear in denser notebooks (e.g. 13–16). Optional maintenance scripts live in `scripts/` (`annotate_notebook_cells.py`, `enrich_workflow_blurbs.py`) if you need to re-apply the same commenting pattern after large edits.


| #   | Notebook                            | Purpose                                                                                                                                       |
| --- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | `01_data_cleaning.ipynb`            | Load raw fight data, parse per-fighter stats, output cleaned fight-level dataset                                                              |
| 02  | `02_feature_engineering.ipynb`      | Aggregate fights into fighter-level style profiles (per-minute rates, positional ratios, control ratio)                                       |
| 03  | `03_bias_checks.ipynb`              | Validate weight-class bias and sample-size stability; informs min-fights filter and normalization                                            |
| 04  | `04_eda_external_data.ipynb`        | Explore physical attributes (ape index, stance) and external style labels                                                                     |
| 05  | `05_finalize_dataset.ipynb`         | Apply experience filter (min 5 fights), weight-class Z-score normalization, produce final modeling dataset; includes normalization validation |
| 06  | `06_eda_comprehensive.ipynb`        | Master EDA: temporal trends, weight-class bias, style space, damage anatomy, referee effects, glass-cannon analysis                         |
| 07  | `07_matchup_analysis.ipynb`         | Matchup-level analysis: advantage deltas, correlation with outcomes, style divergence vs. finish rate                                       |
| 08  | `08_kmeans_style_discovery.ipynb`   | K-Means clustering (k=4,5,6); elbow/silhouette model selection; cluster interpretation                                                         |
| 09  | `09_gmm_model_selection.ipynb`      | GMM model selection via AIC/BIC; introduces soft clustering and Hybrid Score concept                                                          |
| 10  | `10_gmm_style_discovery.ipynb`      | Fit GMM at k=3 and k=5; save cluster assignments and Hybrid Scores to `ufc_gmm_comparison.csv`                                                |
| 11  | `11_style_interaction_matrix.ipynb` | Core matchup asymmetry: win-rate matrix by style pair, hybrid vs. specialist analysis, chi-square tests                                       |
| 12  | `12_autoencoder_supervised.ipynb`   | Autoencoder embeddings; supervised outcome prediction using embedding differences (Random Forest)                                             |
| 13  | `13_upset_analysis.ipynb`           | Elo-based “favorite,” style heatmap vs upsets, Z-delta RF with **event-based** train/test split                                             |
| 14  | `14_odds_cleaning.ipynb`            | Clean Kaggle UFC odds → one row per `Fight_Id`, US market, de-vig implied probs → `ufc_fight_odds_clean.csv`                                  |
| 15  | `15_vegas_vs_predictions.ipynb`    | Compare Vegas implied **P(Vegas favorite wins)** to heatmap, Z-RF, and AE+RF on the **same** test events as notebook 13                      |
| 16  | `16_fight_dynamics_vegas.ipynb`    | **Fight dynamics:** six-way method props (KO/SUB/DEC per fighter) vs `HistGradientBoosting`; **finish vs decision** vs Vegas; event split   |
| 17  | `17_boosting_calibration_winloss.ipynb` | **Win/loss baselines:** HistGradientBoosting, calibrated HGB, logistic vs RF; optional **XGBoost** if installed                          |


### Running the Pipeline

From the project root:

```bash
python run_pipeline.py
```

This executes all notebooks (**01–17**) in numerical order. Notebook **15** uses **PyTorch** for the autoencoder path; **14–16** need `data/raw/kaggle_odds/UFC_betting_odds.csv`. Notebook **16** uses only sklearn; it requires rows with all six **method prop** prices (coverage is a subset of fights). Notebook **17** optionally uses **XGBoost** (`pip install xgboost`). Alternatively, run notebooks manually from the `notebooks/` directory in Jupyter.

### Data (`data/`)

- `raw/` — Original UFC datasets
- `processed/` — Cleaned data, fighter profiles, and model outputs (cluster assignments, GMM results). See `data/README.md`.

---

## Deliverables


| Deliverable                       | Status      | Description                                                                                                         |
| --------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| **Cleaned UFC dataset**           | ✓ Done      | Well-documented, reproducible pipeline from raw data to modeling-ready tables                                       |
| **Fighter style representations** | ✓ Done      | K-Means clusters, GMM probabilistic assignments, autoencoder embeddings                                             |
| **Matchup analysis**              | ✓ Done      | Style interaction matrices, win rates by style pair, hybrid vs. specialist analysis, statistical significance tests |
| **Fight dynamics models**         | In progress | Notebook **16**: six-way method distribution + finish vs decision vs de-vigged Vegas props (subset with full prop prices) |
| **Upset prediction exploration**  | In progress | Notebook **13** (Elo favorite vs heatmap / Z-RF); notebook **15** compares models to **Vegas** where odds exist     |
| **Visualizations**                | In progress | Cluster plots, embedding visualizations, matchup heatmaps, style profiles                                           |
| **Final report and poster**       | Planned     | Written report and project poster for thesis submission                                                             |


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


