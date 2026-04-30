"""Bootstrap CIs for notebook 20 (stack vs Vegas) and notebook 22 (upset ROI by threshold).

Run from repo root: python scripts/run_bootstrap_cis.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, "data", "processed")
sys.path.insert(0, HERE)

import matchup_utils as mu  # noqa: E402

SEED = 42
N_BOOT = 2000


def _paired_boot(y, p_a, p_b, n_boot=N_BOOT, seed=SEED):
    """Return dict of (mean, lo, hi, two-sided p) for AUC/Brier/LogLoss differences (A - B)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_auc = np.empty(n_boot)
    d_brier = np.empty(n_boot)
    d_logl = np.empty(n_boot)
    eps = 1e-6
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
        p_a_b = np.clip(p_a[idx], eps, 1 - eps)
        p_b_b = np.clip(p_b[idx], eps, 1 - eps)
        d_auc[b] = roc_auc_score(yb, p_a_b) - roc_auc_score(yb, p_b_b)
        d_brier[b] = brier_score_loss(yb, p_a_b) - brier_score_loss(yb, p_b_b)
        d_logl[b] = log_loss(yb, p_a_b) - log_loss(yb, p_b_b)

    def _ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        p = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
        return float(arr.mean()), float(lo), float(hi), float(max(p, 1.0 / (n_boot + 1)))

    return {"d_auc": _ci(d_auc), "d_brier": _ci(d_brier), "d_logl": _ci(d_logl)}


def _fmt(label, stats):
    for k, (mean, lo, hi, p) in stats.items():
        print(f"  {label:<22s} {k:>8s}: {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")


def nb20_bootstrap():
    print("=" * 72)
    print("NB 20 -- paired bootstrap CIs for Stack+Vegas and Stack vs Vegas")
    print("=" * 72)

    m = pd.read_csv(os.path.join(DATA, "ufc_matchup_features.csv"), low_memory=False)
    with open(os.path.join(DATA, "ufc_feature_groups.json")) as f:
        groups = json.load(f)

    def valid(cols):
        return [c for c in cols if c in m.columns]

    z_only = valid([f"delta_{c}" for c in [
        "Sig_Str_PM_Z", "Takedown_Att_PM_Z", "Sub_Att_PM_Z", "Control_Ratio_Z"
    ]])
    style_feats = valid([c for c in groups["style_gmm_probs"] + groups["hybrid"]
                         if c.startswith(("A_", "B_", "delta_"))]) + valid(groups["heatmap"])
    ae_feats = valid([c for c in groups["ae"] if c.startswith(("A_", "B_", "delta_"))])
    roll_feats = valid([c for c in groups["rolling_rates"] if c.startswith(("delta_", "mean_"))]) \
                 + valid(groups["context_days"])
    rating_feats = valid(groups["elo_glicko"])
    phys_feats = valid([c for c in groups["physical"] if c.startswith(("A_", "B_", "delta_"))]) \
                 + valid(groups["stance_wc"])
    FULL = list(dict.fromkeys(z_only + style_feats + ae_feats + roll_feats
                               + rating_feats + phys_feats))

    mv = m.dropna(subset=["p_vegas_A"]).copy()
    train = mv[mv["split"] == "train"].reset_index(drop=True)
    val = mv[mv["split"] == "val"].reset_index(drop=True)
    test = mv[mv["split"] == "test"].reset_index(drop=True)
    train_sym = mu.symmetrize_matchup(train, FULL)

    models = {
        "LR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000)),
        ]),
        "HGB": HistGradientBoostingClassifier(
            max_iter=400, max_depth=8, learning_rate=0.05,
            l2_regularization=0.1, random_state=SEED),
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=400, max_depth=10,
                                          min_samples_leaf=5,
                                          random_state=SEED, n_jobs=-1)),
        ]),
        "MLP": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                  early_stopping=True, random_state=SEED)),
        ]),
    }

    val_probs, test_probs = {}, {}
    for name, clf in models.items():
        clf.fit(train_sym[FULL].values, train_sym["Win_A"].values)
        val_probs[name] = clf.predict_proba(val[FULL].values)[:, 1]
        test_probs[name] = clf.predict_proba(test[FULL].values)[:, 1]
    test_probs["Vegas"] = test["p_vegas_A"].values

    base_cols = ["LR", "HGB", "RF"]
    Bv = np.column_stack([val_probs[c] for c in base_cols])
    Bt = np.column_stack([test_probs[c] for c in base_cols])

    meta = LogisticRegression(max_iter=1000).fit(Bv, val["Win_A"].values)
    stack_val = meta.predict_proba(Bv)[:, 1]
    stack_test = meta.predict_proba(Bt)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(stack_val, val["Win_A"].values)
    test_probs["Stack"] = iso.transform(stack_test)

    Bv_v = np.column_stack([Bv, val["p_vegas_A"].values])
    Bt_v = np.column_stack([Bt, test["p_vegas_A"].values])
    meta_v = LogisticRegression(max_iter=1000).fit(Bv_v, val["Win_A"].values)
    sv = meta_v.predict_proba(Bv_v)[:, 1]
    st = meta_v.predict_proba(Bt_v)[:, 1]
    iso_v = IsotonicRegression(out_of_bounds="clip").fit(sv, val["Win_A"].values)
    test_probs["Stack+Vegas"] = iso_v.transform(st)

    y_te = test["Win_A"].values.astype(int)
    n = len(y_te)
    print(f"\nTest slice: n={n} Vegas-aligned fights")
    for name in ["Vegas", "Stack", "Stack+Vegas"]:
        p = np.clip(test_probs[name], 1e-6, 1 - 1e-6)
        print(f"  {name:<12s} AUC={roc_auc_score(y_te, p):.4f}  "
              f"Brier={brier_score_loss(y_te, p):.4f}  "
              f"LogLoss={log_loss(y_te, p):.4f}")

    print(f"\nPaired bootstrap (n_boot={N_BOOT}, seed={SEED}):\n")
    print("Stack+Vegas - Vegas")
    _fmt("Stack+Vegas - Vegas", _paired_boot(y_te, test_probs["Stack+Vegas"], test_probs["Vegas"]))
    print("\nStack - Vegas (no Vegas as a feature)")
    _fmt("Stack - Vegas", _paired_boot(y_te, test_probs["Stack"], test_probs["Vegas"]))
    return {
        "n": n,
        "stack_v_vs_vegas": _paired_boot(y_te, test_probs["Stack+Vegas"], test_probs["Vegas"]),
        "stack_vs_vegas": _paired_boot(y_te, test_probs["Stack"], test_probs["Vegas"]),
    }


def nb22_bootstrap():
    """Reproduce NB 22's HGB(full+vegas) upset residual pipeline, then bootstrap ROI CIs."""
    print("\n" + "=" * 72)
    print("NB 22 -- bootstrap CIs for upset ROI (reproducing NB 22 pipeline)")
    print("=" * 72)

    m = pd.read_csv(os.path.join(DATA, "ufc_matchup_features.csv"), low_memory=False)
    with open(os.path.join(DATA, "ufc_feature_groups.json")) as f:
        groups = json.load(f)

    def valid(cols):
        return [c for c in cols if c in m.columns]

    FULL = (
        valid([f"delta_{c}" for c in [
            "Sig_Str_PM_Z", "Takedown_Att_PM_Z", "Sub_Att_PM_Z", "Control_Ratio_Z"
        ]])
        + valid([c for c in groups["style_gmm_probs"] + groups["hybrid"]
                 if c.startswith(("A_", "B_", "delta_"))])
        + valid(groups["heatmap"])
        + valid([c for c in groups["ae"] if c.startswith(("A_", "B_", "delta_"))])
        + valid([c for c in groups["rolling_rates"] if c.startswith(("delta_", "mean_"))])
        + valid(groups["context_days"]) + valid(groups["elo_glicko"])
        + valid([c for c in groups["physical"] if c.startswith(("A_", "B_", "delta_"))])
        + valid(groups["stance_wc"])
    )
    FULL = list(dict.fromkeys(FULL))
    FULL_V = FULL + valid(groups["vegas"])

    vegas_m = m[m["has_vegas"] == 1].reset_index(drop=True).copy()
    vegas_m["y_upset_any"] = (((vegas_m["Win_A"] == 1) & (vegas_m["p_vegas_A"] < 0.5))
                              | ((vegas_m["Win_A"] == 0) & (vegas_m["p_vegas_A"] > 0.5))).astype(int)

    train = vegas_m[vegas_m["split"] == "train"].reset_index(drop=True)
    val = vegas_m[vegas_m["split"] == "val"].reset_index(drop=True)
    test = vegas_m[vegas_m["split"] == "test"].reset_index(drop=True)
    train_sym = mu.symmetrize_matchup(train, feat_cols=[], label_col="Win_A")

    def fit_hgb(cols):
        clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=8, learning_rate=0.05,
            l2_regularization=0.1, random_state=SEED)
        clf.fit(train_sym[cols].values, train_sym["Win_A"].values)
        return clf

    m_wv = fit_hgb(FULL_V)
    p_wv_va = m_wv.predict_proba(val[FULL_V].values)[:, 1]
    p_wv = m_wv.predict_proba(test[FULL_V].values)[:, 1]
    iso_wv = IsotonicRegression(out_of_bounds="clip").fit(p_wv_va, val["Win_A"].values)
    p_wv = iso_wv.transform(p_wv)

    p_vegas = test["p_vegas_A"].values
    r_wv = p_wv - p_vegas
    y = test["Win_A"].values.astype(int)
    y_upset_any = test["y_upset_any"].values.astype(int)

    # Exactly NB 22's upset score: signed by which corner is the market dog.
    # Positive score == model bets on the underdog winning.
    s_wv = np.where(p_vegas < 0.5, r_wv, -r_wv)

    # ROI on dog-side bets when score >= threshold (replicating NB 22 cell 4)
    dog_won = np.where(p_vegas < 0.5, y == 1, y == 0).astype(int)
    dog_prob = np.where(p_vegas < 0.5, p_vegas, 1 - p_vegas)
    payoff_on_win = (1.0 / np.maximum(dog_prob, 1e-6)) - 1.0  # decimal profit on $1 stake
    # profit array per fight conditional on placing a bet (vs not)
    profit_if_bet = np.where(dog_won == 1, payoff_on_win, -1.0)

    print(f"\nTest slice: n={len(y)} Vegas-aligned fights")
    print(f"  Upset detector AUC (r_wv -> y_upset_any): {roc_auc_score(y_upset_any, s_wv):.4f}")

    def bootstrap_roi(thr, n_boot=N_BOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        n = len(y)
        rois, counts, hits = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            mask = s_wv[idx] >= thr
            if mask.sum() == 0:
                continue
            rois.append(float(profit_if_bet[idx][mask].mean()))
            counts.append(int(mask.sum()))
            hits.append(float(dog_won[idx][mask].mean()))
        arr = np.array(rois)
        if arr.size < 20:
            return None
        lo, hi = np.percentile(arr, [2.5, 97.5])
        # observed / point estimate
        obs_mask = s_wv >= thr
        obs_roi = float(profit_if_bet[obs_mask].mean()) if obs_mask.sum() else np.nan
        obs_n = int(obs_mask.sum())
        obs_hit = float(dog_won[obs_mask].mean()) if obs_mask.sum() else np.nan
        p_leq_0 = float((arr <= 0).mean())
        return {
            "thr": thr,
            "obs_n_bets": obs_n,
            "obs_hit_rate": obs_hit,
            "obs_roi": obs_roi,
            "boot_mean_roi": float(arr.mean()),
            "boot_ci_low": float(lo),
            "boot_ci_high": float(hi),
            "p_roi_leq_0": p_leq_0,
            "median_n_bets": int(np.median(counts)),
        }

    thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    print(f"\nBootstrap ROI (n_boot={N_BOOT}, seed={SEED}):")
    print(f"{'thr':>5} {'n_bets':>7} {'hit':>6} {'obs_ROI':>9} {'boot_mean':>10} "
          f"{'95% CI':>22} {'P(ROI<=0)':>11}")
    results = []
    for thr in thresholds:
        row = bootstrap_roi(thr)
        if row is None:
            print(f"{thr:>5.2f}   (too few resamples)")
            continue
        ci = f"[{row['boot_ci_low']:+.3f}, {row['boot_ci_high']:+.3f}]"
        print(f"{thr:>5.2f} {row['obs_n_bets']:>7d} {row['obs_hit_rate']:>6.3f} "
              f"{row['obs_roi']:+9.3f} {row['boot_mean_roi']:+10.3f} "
              f"{ci:>22s} {row['p_roi_leq_0']:>11.3f}")
        results.append(row)
    return {"roi_by_threshold": results}


if __name__ == "__main__":
    out20 = nb20_bootstrap()
    out22 = nb22_bootstrap()
    print("\nDone.")
