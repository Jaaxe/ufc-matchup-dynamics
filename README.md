# Uncovering Fighting Styles and Matchup Asymmetries in the UFC via Unsupervised and Self-Supervised Learning

**Author:** Jarvis Xie 

**Advisor:** Robert Wooster, Department of Statistics and Data Science 

**Date:** 2025-01-25 

## Project Overview

In professional mixed martial arts (MMA), fighters are often described using informal notions of "fighting style," such as striker, grappler, or pressure fighter. These labels are commonly used to explain matchup outcomes, yet they are subjective, loosely defined, and rarely grounded in data.

This project is motivated by the idea that fighting style is a real but unobserved factor that shapes both fight outcomes and fight dynamics. Rather than imposing predefined style categories, this project aims to discover fighting styles directly from UFC fight data using unsupervised and self-supervised learning. This allows for a data-driven understanding of matchup dynamics, stylistic mismatches, and asymmetric outcomes.

## Problem Statement

Most existing UFC analytics focus on predicting win-loss outcomes using hand-engineered features without explicitly modeling style. The specific problem addressed in this project is identifying latent fighting styles from UFC fight statistics and understanding how interactions between these styles influence fight outcomes and dynamics.

In particular, the project focuses on explaining favorable/unfavorable matchups, stylistic asymmetries, and cases where a fighter who is otherwise expected to win instead loses due to stylistic mismatch.

## Methodology

The project focuses on modern representation learning techniques designed to capture structure in sequential and relational data, rather than relying solely on static feature engineering.

*  **Representation:** Fighters or fights will be represented using vectors derived from per-fight statistics (e.g., striking, grappling, and control metrics), with normalization to account for fight length and context.

*  **Style Discovery:** To learn stylistic representations, the project will emphasize embedding-based approaches, specifically neural embedding models and autoencoder-style architectures. These models are chosen for their ability to model nonlinear structure and variability in fighting behavior.

*  **Matchup Modeling:** The project may also explore representation learning methods that leverage opponent relationships to learn embeddings that are predictive of relative performance when two fighters are paired.

## Data

The project will use publicly available UFC fight datasets compiled from official UFC statistics. These datasets include:

*  **Fight-level information:** Significant strikes, takedowns, control time, fight duration, and method of victory.

*  **Fighter-level metadata:** Age, reach, stance, and weight class.

Primary data preparation tasks will involve cleaning raw statistics, handling missing values, and aggregating fight-level data into fighter-level summaries.

## Deliverables

* A cleaned and well-documented UFC fight dataset.
* Learned latent representations or clusters corresponding to fighting styles.
* Analysis of favorable and unfavorable stylistic matchups and matchup asymmetries.
* Models describing expected fight dynamics based on stylistic matchups.
* Visualizations illustrating styles and matchup behavior.
* A final written report and a project poster.

## Timeline

* **Weeks 3-5:** Data acquisition, cleaning, and exploratory data analysis; begin writing alongside analysis.

* **Weeks 6-8:** Unsupervised and self-supervised style discovery; continued writing and refinement.

* **Weeks 9-10:** Matchup analysis and fight-structure modeling; finalize main results.

* **Week 11:** Finalize first draft of report and poster; submit draft to advisor for feedback.

* **Weeks 12-13:** Revise report and poster based on advisor feedback.

* **Reading week:** Finalize report and submit.
