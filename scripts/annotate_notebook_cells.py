#!/usr/bin/env python3
"""
One-off / reusable: prepend structured comments to every code cell in notebooks/.
Skips cells that already start with the NOTEBOOK_CODE_HEADER marker.
Run from repo root: python scripts/annotate_notebook_cells.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MARKER_LINE = "# NOTEBOOK_CODE_HEADER v1"

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def prev_markdown_heading(cells: list, idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        c = cells[j]
        if c.get("cell_type") != "markdown":
            continue
        text = "".join(c.get("source", []))
        for line in text.splitlines():
            t = line.strip()
            if not t or t.startswith("!["):
                continue
            return re.sub(r"^#+\s*", "", t).strip()[:200]
    return "(no markdown section above)"


def code_hints(src: str) -> list[str]:
    hints: list[str] = []
    if re.search(r"^\s*import |^from \w+", src, re.M):
        hints.append("# Dependencies: see import statements below.")
    if "read_csv" in src:
        hints.append("# Loads one or more CSV files (paths usually relative to notebooks/).")
    if "to_csv" in src:
        hints.append("# Persists a DataFrame to CSV under data/processed or similar.")
    if re.search(r"\.merge\(", src):
        hints.append("# Joins tables on fighter, fight, or event keys.")
    if re.search(r"\.groupby\(", src):
        hints.append("# Groups rows for aggregation (means, counts, etc.).")
    if "RandomForest" in src or "XGB" in src or "LogisticRegression" in src:
        hints.append("# Fits or scores a supervised sklearn model.")
    if "torch" in src or "nn.Module" in src:
        hints.append("# PyTorch: neural network definition and/or training loop.")
    if "plt." in src or "sns." in src:
        hints.append("# Builds matplotlib/seaborn figures.")
    if "KMeans" in src or "GaussianMixture" in src:
        hints.append("# Unsupervised clustering (K-Means or GMM).")
    if "StandardScaler" in src:
        hints.append("# Standardizes features (mean 0, unit variance) where used.")
    return hints


def build_header(nb_name: str, cell_idx: int, md_title: str, src: str) -> str:
    lines = [
        "# " + "=" * 72,
        MARKER_LINE,
        f"# File: {nb_name} | code cell index: {cell_idx}",
        f"# Section (from markdown above): {md_title}",
        "# " + "-" * 72,
    ]
    for h in code_hints(src):
        if h not in lines:
            lines.append(h)
    lines.append("# " + "=" * 72)
    lines.append("")
    return "\n".join(lines)


def process_notebook(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    changed = 0
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if MARKER_LINE in src[:2000]:
            continue
        md = prev_markdown_heading(cells, i)
        header = build_header(path.name, i, md, src)
        cell["source"] = header + src
        changed += 1
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return changed


def main():
    total = 0
    for path in sorted(NB_DIR.glob("*.ipynb")):
        n = process_notebook(path)
        if n:
            print(f"{path.name}: updated {n} code cells")
        total += n
    print(f"Done. Total code cells updated: {total}")


if __name__ == "__main__":
    main()
