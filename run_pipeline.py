#!/usr/bin/env python3
"""
Run the full analysis pipeline: all notebooks in numerical order (01–17).
Execute from project root: python run_pipeline.py
"""
# Non-interactive backend for headless/scripted execution
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"
PIPELINE = [
    "01_data_cleaning.ipynb",
    "02_feature_engineering.ipynb",
    "03_bias_checks.ipynb",
    "04_eda_external_data.ipynb",
    "05_finalize_dataset.ipynb",
    "06_eda_comprehensive.ipynb",
    "07_matchup_analysis.ipynb",
    "08_kmeans_style_discovery.ipynb",
    "09_gmm_model_selection.ipynb",
    "10_gmm_style_discovery.ipynb",
    "11_style_interaction_matrix.ipynb",
    "12_autoencoder_supervised.ipynb",
    "13_upset_analysis.ipynb",
    "14_odds_cleaning.ipynb",
    "15_vegas_vs_predictions.ipynb",
    "16_fight_dynamics_vegas.ipynb",
    "17_boosting_calibration_winloss.ipynb",
    "18_build_matchup_features.ipynb",
    "19_winloss_bakeoff.ipynb",
    "20_vegas_winloss_comparison.ipynb",
    "21_dynamics_suite.ipynb",
    "22_upset_prediction_vegas.ipynb",
    "23_sequence_model.ipynb",
]


def run_notebook(path: Path) -> None:
    """Execute every code cell in order (shared global namespace, like a single script)."""
    with open(path) as f:
        nb = json.load(f)
    g = {}
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        try:
            exec(src, g)
        except Exception as e:
            print(f"FAIL {path.name} cell {i}: {e}", file=sys.stderr)
            raise
    print(f"OK {path.name}")


def main():
    # Notebooks use relative paths such as ../data/processed/... — cwd must be notebooks/.
    import os
    os.chdir(NOTEBOOKS_DIR)

    for name in PIPELINE:
        path = NOTEBOOKS_DIR / name
        if not path.exists():
            print(f"Skip {name} (not found)", file=sys.stderr)
            continue
        run_notebook(path)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
