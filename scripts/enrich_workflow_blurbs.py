#!/usr/bin/env python3
"""Insert a short # Workflow: bullet list after the standard notebook header in each code cell."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
WORKFLOW_TAG = "# INLINE_WORKFLOW_SUMMARY v1"


def end_of_header_line_idx(lines: list[str]) -> int:
    """Return line index after the *closing* banner of NOTEBOOK_CODE_HEADER, or 0."""
    if not lines:
        return 0
    head = "\n".join(lines[:40])
    if "NOTEBOOK_CODE_HEADER v1" not in head:
        return 0
    # Header format: opening banner ... content ... closing banner (same width)
    banners = [
        i
        for i, ln in enumerate(lines[:35])
        if ln.startswith("# " + "=" * 10)
    ]
    if len(banners) >= 2:
        return banners[-1] + 1
    return 0


def build_workflow_bullets(src: str) -> list[str]:
    bullets: list[str] = []
    s = src
    if WORKFLOW_TAG in s:
        return []
    if "INPUT_PATH" in s or "OUTPUT_PATH" in s or "RAW" in s or "OUT" in s:
        bullets.append(
            "Input/output paths are configured at the top; adjust if your data live elsewhere."
        )
    if "read_csv" in s:
        bullets.append("CSV reads use paths relative to the notebooks/ working directory.")
    if "to_csv" in s:
        bullets.append("Processed outputs are written so downstream notebooks can load them.")
    if ".merge(" in s:
        bullets.append("Merges link fighters, fights, styles, or odds on shared keys.")
    if "groupby" in s:
        bullets.append("Groupby operations aggregate fight or fighter statistics.")
    if "pivot_table" in s:
        bullets.append("Pivot tables build matrices (e.g., style vs style win rates).")
    if "RandomForest" in s:
        bullets.append("RandomForest* fits on train rows; predict_proba yields calibrated-ish scores.")
    if "train_test_split" in s:
        bullets.append("Train/test split: check random_state and stratify options for reproducibility.")
    if "Event_Id" in s and "train_mask" in s:
        bullets.append("Event-based holdout keeps entire cards in train or test (reduces leakage).")
    if "torch" in s or "nn.Module" in s:
        bullets.append("PyTorch block: tensor shapes follow (batch, features); device may be CPU or CUDA.")
    if "StandardScaler" in s:
        bullets.append("Scaler is fit on training-like data only when a split is explicit.")
    if "log_loss" in s or "brier_score_loss" in s:
        bullets.append("Probabilities are clipped before log loss to avoid log(0).")
    if not bullets:
        bullets.append("Follow the code top-to-bottom; prints document shapes and key counts.")
    return bullets


def insert_workflow(src: str) -> str:
    if WORKFLOW_TAG in src:
        return src
    lines = src.split("\n")
    j = end_of_header_line_idx(lines)
    bullets = build_workflow_bullets(src)
    block = [WORKFLOW_TAG, "# Workflow summary (how to read this cell):"]
    for b in bullets:
        block.append(f"#   • {b}")
    block.append("")
    new_lines = lines[:j] + block + lines[j:]
    return "\n".join(new_lines)


def process(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        new_src = insert_workflow(src)
        if new_src != src:
            cell["source"] = new_src
            n += 1
    if n:
        path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return n


def main():
    t = 0
    for path in sorted(NB_DIR.glob("*.ipynb")):
        k = process(path)
        if k:
            print(f"{path.name}: inserted workflow in {k} cells")
        t += k
    print(f"Total cells updated: {t}")


if __name__ == "__main__":
    main()
