"""Shared helpers for UFC matchup / dynamics modeling (notebooks 18-23).

Handles the concerns flagged in the project plan:
    * event-date-ordered train / val / test splits (no Event_Id_x ordering)
    * walk-forward CV folds on the train portion
    * leakage-free rolling (pre-fight) career aggregates
    * Elo and Glicko-2 ratings walked in chronological order
    * parsing of raw physical attributes (height / reach / stance)
    * last-k rolling form features + days-since-last-fight
    * method-bucket mapping (6-way / 3-way / finish) for dynamics
    * weight-class one-hot + matchup symmetrization helpers
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Event dates / splits
# ---------------------------------------------------------------------------

def load_event_dates(events_path: str = "../data/raw/raw_events.csv") -> pd.DataFrame:
    """Return DataFrame keyed by `Event_Id` with parsed `Event_Date`."""
    ev = pd.read_csv(events_path)
    ev["Event_Date"] = pd.to_datetime(ev["Date"], errors="coerce", utc=False)
    ev = ev[["Event_Id", "Event_Date"]].dropna(subset=["Event_Date"])
    return ev


def attach_event_dates(
    fights: pd.DataFrame,
    events_path: str = "../data/raw/raw_events.csv",
    fight_event_col: str = "Event_Id_x",
) -> pd.DataFrame:
    ev = load_event_dates(events_path)
    out = fights.merge(
        ev.rename(columns={"Event_Id": fight_event_col}),
        on=fight_event_col,
        how="left",
    )
    if out["Event_Date"].isna().any():
        miss = out["Event_Date"].isna().sum()
        print(f"[matchup_utils] WARNING: {miss} rows missing Event_Date after merge.")
    return out


def event_ordered_split(
    fight_events: pd.Series,
    event_dates: pd.Series,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
) -> dict[str, np.ndarray]:
    """Return boolean masks (train/val/test) ordered by event date.

    `fight_events` and `event_dates` are parallel Series (row-aligned).
    """
    df = pd.DataFrame({"ev": fight_events.values, "dt": pd.to_datetime(event_dates).values})
    ev_order = (
        df.dropna(subset=["dt"])  # guard
        .groupby("ev", as_index=False)["dt"].min()
        .sort_values("dt")
        .reset_index(drop=True)
    )
    n = len(ev_order)
    i_train = int(round(frac_train * n))
    i_val = int(round((frac_train + frac_val) * n))
    train_ev = set(ev_order["ev"].iloc[:i_train])
    val_ev = set(ev_order["ev"].iloc[i_train:i_val])
    test_ev = set(ev_order["ev"].iloc[i_val:])
    ev_arr = df["ev"].values
    return {
        "train": np.array([e in train_ev for e in ev_arr]),
        "val": np.array([e in val_ev for e in ev_arr]),
        "test": np.array([e in test_ev for e in ev_arr]),
    }


def walk_forward_folds(
    fight_events: pd.Series,
    event_dates: pd.Series,
    n_folds: int = 5,
    min_train_frac: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds over events (train_mask, val_mask)."""
    df = pd.DataFrame({"ev": fight_events.values, "dt": pd.to_datetime(event_dates).values})
    ev_order = (
        df.dropna(subset=["dt"])  # guard
        .groupby("ev", as_index=False)["dt"].min()
        .sort_values("dt")
        .reset_index(drop=True)
    )
    n = len(ev_order)
    base = int(round(min_train_frac * n))
    remaining = n - base
    fold_size = max(1, remaining // n_folds)
    folds = []
    ev_arr = df["ev"].values
    for k in range(n_folds):
        start = base + k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        if start >= n:
            break
        train_ev = set(ev_order["ev"].iloc[:start])
        val_ev = set(ev_order["ev"].iloc[start:end])
        tr = np.array([e in train_ev for e in ev_arr])
        va = np.array([e in val_ev for e in ev_arr])
        folds.append((tr, va))
    return folds


# ---------------------------------------------------------------------------
# Physical attribute parsing (raw_fighters.csv)
# ---------------------------------------------------------------------------

_HT_RE = re.compile(r"(\d+)'\s*(\d+)")


def parse_height_inches(s: object) -> float:
    if not isinstance(s, str) or s.strip() in ("--", ""):
        return np.nan
    m = _HT_RE.search(s)
    if not m:
        return np.nan
    return float(m.group(1)) * 12 + float(m.group(2))


def parse_reach_inches(s: object) -> float:
    if not isinstance(s, str) or s.strip() in ("--", ""):
        return np.nan
    s2 = s.replace('"', '').strip()
    try:
        return float(s2)
    except ValueError:
        return np.nan


def load_physical(path: str = "../data/raw/raw_fighters.csv") -> pd.DataFrame:
    rf = pd.read_csv(path)
    rf["Fighter"] = (
        rf["First"].fillna("").astype(str).str.strip() + " " +
        rf["Last"].fillna("").astype(str).str.strip()
    ).str.strip()
    rf["Height_In"] = rf["Ht."].map(parse_height_inches)
    rf["Reach_In"] = rf["Reach"].map(parse_reach_inches)
    rf["Stance"] = rf["Stance"].fillna("Unknown").replace({"--": "Unknown", "": "Unknown"})
    return rf[["Fighter", "Height_In", "Reach_In", "Stance"]].drop_duplicates("Fighter")


# ---------------------------------------------------------------------------
# Rolling pre-fight profiles + ratings
# ---------------------------------------------------------------------------

@dataclass
class FighterAccum:
    fights: int = 0
    wins: int = 0
    ko_for: int = 0
    sub_for: int = 0
    dec_for: int = 0
    ko_against: int = 0
    sub_against: int = 0
    seconds: float = 0.0
    sig_str_landed: float = 0.0
    sig_str_att: float = 0.0
    sig_str_absorbed: float = 0.0  # opponent's landed against this fighter
    td_landed: float = 0.0
    td_att: float = 0.0
    td_absorbed: float = 0.0
    sub_att: float = 0.0
    control_sec: float = 0.0
    ground_str: float = 0.0
    clinch_str: float = 0.0
    distance_str: float = 0.0
    knockdowns_for: float = 0.0
    knockdowns_against: float = 0.0
    last_date: pd.Timestamp | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=5))


def _per_min(num: float, seconds: float) -> float:
    return num / (seconds / 60.0) if seconds > 0 else 0.0


def _fraction(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def build_rolling_profiles(
    fights: pd.DataFrame,
    event_dates_path: str = "../data/raw/raw_events.csv",
) -> pd.DataFrame:
    """One row per Fight_Id (Fighter_A = the row-fighter from ``drop_duplicates``)
    with pre-fight aggregates for both corners.

    The cleaned CSV carries each fight twice (once per corner). We deduplicate
    to one row per fight but look up the opponent's per-fight stats from the
    *other* perspective so each fighter accumulates their own landed strikes
    rather than double-counting the fight-level totals.
    """
    ev = load_event_dates(event_dates_path)
    ev_map = dict(zip(ev["Event_Id"], ev["Event_Date"]))

    df = fights.copy()
    df["Event_Date"] = df["Event_Id_x"].map(ev_map)
    df["Win_A"] = df["Won"].astype(int)

    # Build fast lookup for opponent-side stats: (Fight_Id, Fighter) -> row
    stat_cols = [
        "Sig_Str_Landed", "Sig_Str_Att", "Takedowns_Landed", "Takedowns_Att",
        "Sub_Attempts", "Control_Seconds", "Ground_Strikes_Landed",
        "Clinch_Strikes_Landed", "Distance_Strikes_Landed", "Knockdowns",
    ]
    side_lookup = (
        df[["Fight_Id", "Fighter"] + stat_cols]
        .set_index(["Fight_Id", "Fighter"])
        .to_dict("index")
    )

    # One row per fight, ordered by date (deterministic tie-breaker).
    df = (
        df.drop_duplicates("Fight_Id", keep="first")
        .sort_values(["Event_Date", "Event_Id_x", "Fight_Id"]).reset_index(drop=True)
    )

    def opp_stat(fight_id, fighter, key):
        r = side_lookup.get((fight_id, fighter))
        return float(r[key]) if r is not None else 0.0

    acc: dict[str, FighterAccum] = defaultdict(FighterAccum)
    rows = []
    for r in df.itertuples(index=False):
        a = acc[r.Fighter]
        b = acc[r.Opponent]
        days_since_a = (
            (r.Event_Date - a.last_date).days if a.last_date is not None and pd.notna(r.Event_Date) else np.nan
        )
        days_since_b = (
            (r.Event_Date - b.last_date).days if b.last_date is not None and pd.notna(r.Event_Date) else np.nan
        )

        def snap(x: FighterAccum) -> dict:
            seconds = x.seconds
            # per-fight averages over recent window
            rc = list(x.recent)
            recent_wins = np.mean([w for (w, *_rest) in rc]) if rc else np.nan
            recent_finish = np.mean([1.0 if ko or sub else 0.0 for (_, ko, sub, *_rest) in rc]) if rc else np.nan
            recent_ctrl = np.mean([c for (*_rest, c) in rc]) if rc else np.nan
            return {
                "pre_fights": x.fights,
                "pre_win_rate": _fraction(x.wins, x.fights),
                "pre_ko_rate_for": _fraction(x.ko_for, x.fights),
                "pre_sub_rate_for": _fraction(x.sub_for, x.fights),
                "pre_dec_rate": _fraction(x.dec_for, x.fights),
                "pre_ko_rate_against": _fraction(x.ko_against, x.fights),
                "pre_sub_rate_against": _fraction(x.sub_against, x.fights),
                "pre_sig_str_pm": _per_min(x.sig_str_landed, seconds),
                "pre_sig_str_att_pm": _per_min(x.sig_str_att, seconds),
                "pre_sig_str_absorbed_pm": _per_min(x.sig_str_absorbed, seconds),
                "pre_sig_str_acc": _fraction(x.sig_str_landed, x.sig_str_att),
                "pre_td_landed_pm": _per_min(x.td_landed, seconds),
                "pre_td_att_pm": _per_min(x.td_att, seconds),
                "pre_td_acc": _fraction(x.td_landed, x.td_att),
                "pre_td_absorbed_pm": _per_min(x.td_absorbed, seconds),
                "pre_sub_att_pm": _per_min(x.sub_att, seconds),
                "pre_control_ratio": _fraction(x.control_sec, seconds),
                "pre_ground_share": _fraction(x.ground_str, x.sig_str_landed),
                "pre_clinch_share": _fraction(x.clinch_str, x.sig_str_landed),
                "pre_distance_share": _fraction(x.distance_str, x.sig_str_landed),
                "pre_kd_for_pm": _per_min(x.knockdowns_for, seconds),
                "pre_kd_against_pm": _per_min(x.knockdowns_against, seconds),
                "recent5_win_rate": recent_wins,
                "recent5_finish_rate": recent_finish,
                "recent5_ctrl_ratio": recent_ctrl,
            }

        rec = {
            "Fight_Id": r.Fight_Id,
            "Event_Id_x": r.Event_Id_x,
            "Event_Date": r.Event_Date,
            "Fighter": r.Fighter,
            "Opponent": r.Opponent,
            "Weight_Class": r.Weight_Class,
            "Win_A": r.Win_A,
            "days_since_last_A": days_since_a,
            "days_since_last_B": days_since_b,
        }
        for k, v in snap(a).items():
            rec[f"A_{k}"] = v
        for k, v in snap(b).items():
            rec[f"B_{k}"] = v
        rows.append(rec)

        # Update accumulators *after* recording the pre-fight snapshot.
        method_u = str(r.Method).upper()
        ko = "KO" in method_u or "TKO" in method_u
        sub = method_u.startswith("SUB")
        dec = ("DEC" in method_u) and not ko and not sub

        # A-side stats come from this row (the A-perspective row).
        a_sig = float(getattr(r, "Sig_Str_Landed", 0) or 0)
        a_sig_att = float(getattr(r, "Sig_Str_Att", 0) or 0)
        a_td = float(getattr(r, "Takedowns_Landed", 0) or 0)
        a_td_att = float(getattr(r, "Takedowns_Att", 0) or 0)
        a_sub_att = float(getattr(r, "Sub_Attempts", 0) or 0)
        a_ctrl = float(getattr(r, "Control_Seconds", 0) or 0)
        a_ground = float(getattr(r, "Ground_Strikes_Landed", 0) or 0)
        a_clinch = float(getattr(r, "Clinch_Strikes_Landed", 0) or 0)
        a_dist = float(getattr(r, "Distance_Strikes_Landed", 0) or 0)
        a_kd = float(getattr(r, "Knockdowns", 0) or 0)

        # B-side stats come from the opposite-corner row.
        b_sig = opp_stat(r.Fight_Id, r.Opponent, "Sig_Str_Landed")
        b_sig_att = opp_stat(r.Fight_Id, r.Opponent, "Sig_Str_Att")
        b_td = opp_stat(r.Fight_Id, r.Opponent, "Takedowns_Landed")
        b_td_att = opp_stat(r.Fight_Id, r.Opponent, "Takedowns_Att")
        b_sub_att = opp_stat(r.Fight_Id, r.Opponent, "Sub_Attempts")
        b_ctrl = opp_stat(r.Fight_Id, r.Opponent, "Control_Seconds")
        b_ground = opp_stat(r.Fight_Id, r.Opponent, "Ground_Strikes_Landed")
        b_clinch = opp_stat(r.Fight_Id, r.Opponent, "Clinch_Strikes_Landed")
        b_dist = opp_stat(r.Fight_Id, r.Opponent, "Distance_Strikes_Landed")
        b_kd = opp_stat(r.Fight_Id, r.Opponent, "Knockdowns")

        seconds = float(r.total_fight_seconds)
        a.fights += 1
        b.fights += 1
        a.seconds += seconds
        b.seconds += seconds
        a.sig_str_landed += a_sig;  a.sig_str_att += a_sig_att
        b.sig_str_landed += b_sig;  b.sig_str_att += b_sig_att
        a.sig_str_absorbed += b_sig; b.sig_str_absorbed += a_sig
        a.td_landed += a_td;  a.td_att += a_td_att
        b.td_landed += b_td;  b.td_att += b_td_att
        a.td_absorbed += b_td; b.td_absorbed += a_td
        a.sub_att += a_sub_att; b.sub_att += b_sub_att
        a.control_sec += a_ctrl; b.control_sec += b_ctrl
        a.ground_str += a_ground; b.ground_str += b_ground
        a.clinch_str += a_clinch; b.clinch_str += b_clinch
        a.distance_str += a_dist; b.distance_str += b_dist
        a.knockdowns_for += a_kd; b.knockdowns_for += b_kd
        a.knockdowns_against += b_kd; b.knockdowns_against += a_kd

        if r.Win_A == 1:
            a.wins += 1
            if ko:
                a.ko_for += 1
                b.ko_against += 1
            elif sub:
                a.sub_for += 1
                b.sub_against += 1
            elif dec:
                a.dec_for += 1
                b.dec_for += 1
        else:
            b.wins += 1
            if ko:
                b.ko_for += 1
                a.ko_against += 1
            elif sub:
                b.sub_for += 1
                a.sub_against += 1
            elif dec:
                a.dec_for += 1
                b.dec_for += 1

        a.last_date = r.Event_Date
        b.last_date = r.Event_Date
        a.recent.append((1.0 if r.Win_A == 1 else 0.0, 1.0 if ko else 0.0, 1.0 if sub else 0.0,
                         _fraction(a_ctrl, seconds)))
        b.recent.append((1.0 if r.Win_A == 0 else 0.0, 1.0 if ko else 0.0, 1.0 if sub else 0.0,
                         _fraction(b_ctrl, seconds)))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Elo + Glicko-2
# ---------------------------------------------------------------------------

def walk_elo(fights_sorted: pd.DataFrame, k: float = 32.0, init: float = 1500.0) -> pd.DataFrame:
    """Return per-fight pre-fight Elo for fighter A and B, plus favorite flag."""
    rating: dict[str, float] = defaultdict(lambda: init)
    out = []
    for r in fights_sorted.itertuples(index=False):
        ra = rating[r.Fighter]
        rb = rating[r.Opponent]
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        score_a = 1.0 if r.Win_A == 1 else 0.0
        rating[r.Fighter] = ra + k * (score_a - exp_a)
        rating[r.Opponent] = rb + k * ((1 - score_a) - (1 - exp_a))
        out.append({
            "Fight_Id": r.Fight_Id,
            "Elo_A_pre": ra,
            "Elo_B_pre": rb,
            "elo_diff": ra - rb,
            "elo_p_A": exp_a,
            "elo_favorite_is_A": ra > rb,
        })
    return pd.DataFrame(out)


# Glicko-2 (compact implementation; follows Glickman 2013).
_G2_TAU = 0.5
_G2_Q = math.log(10.0) / 400.0


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _glicko2_update(mu: float, phi: float, sigma: float,
                    mu_o: float, phi_o: float, s: float) -> tuple[float, float, float]:
    g_j = _g(phi_o)
    E_j = _E(mu, mu_o, phi_o)
    v = 1.0 / (g_j * g_j * E_j * (1 - E_j) + 1e-12)
    delta = v * g_j * (s - E_j)
    a = math.log(sigma * sigma)
    A = a
    eps = 1e-6
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while True:
            trial = a - k * _G2_TAU
            f_trial = (math.exp(trial) * (delta * delta - phi * phi - v - math.exp(trial))
                       / (2 * (phi * phi + v + math.exp(trial)) ** 2)
                       - (trial - a) / (_G2_TAU * _G2_TAU))
            if f_trial < 0:
                k += 1
                if k > 100:
                    B = trial
                    break
                continue
            B = trial
            break

    def f(x):
        ex = math.exp(x)
        return (ex * (delta * delta - phi * phi - v - ex)
                / (2 * (phi * phi + v + ex) ** 2)
                - (x - a) / (_G2_TAU * _G2_TAU))

    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA) if (fB - fA) != 0 else (A + B) / 2
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC
    sigma_new = math.exp(A / 2.0)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * g_j * (s - E_j)
    return mu_new, phi_new, sigma_new


def walk_glicko(fights_sorted: pd.DataFrame) -> pd.DataFrame:
    """Per-fight pre-fight Glicko-2 (mu, phi, sigma) on Glicko-2 scale plus rating 400*mu+1500."""
    state: dict[str, tuple[float, float, float]] = defaultdict(lambda: (0.0, 350.0 * _G2_Q, 0.06))
    out = []
    for r in fights_sorted.itertuples(index=False):
        mu_a, phi_a, sig_a = state[r.Fighter]
        mu_b, phi_b, sig_b = state[r.Opponent]
        # Pre-fight rating + uncertainty
        out.append({
            "Fight_Id": r.Fight_Id,
            "Glicko_A_pre": 1500 + mu_a / _G2_Q,
            "Glicko_B_pre": 1500 + mu_b / _G2_Q,
            "Glicko_A_phi": phi_a / _G2_Q,
            "Glicko_B_phi": phi_b / _G2_Q,
            "glicko_diff": (mu_a - mu_b) / _G2_Q,
            "glicko_p_A": _E(mu_a, mu_b, math.sqrt(phi_a * phi_a + phi_b * phi_b)),
        })
        s_a = 1.0 if r.Win_A == 1 else 0.0
        mu_a_new, phi_a_new, sig_a_new = _glicko2_update(mu_a, phi_a, sig_a, mu_b, phi_b, s_a)
        mu_b_new, phi_b_new, sig_b_new = _glicko2_update(mu_b, phi_b, sig_b, mu_a, phi_a, 1 - s_a)
        state[r.Fighter] = (mu_a_new, phi_a_new, sig_a_new)
        state[r.Opponent] = (mu_b_new, phi_b_new, sig_b_new)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Method / dynamics targets
# ---------------------------------------------------------------------------

def method_bucket(method: str, win_a: int) -> str | None:
    mu = str(method).strip().upper()
    bad = ("OVERTURNED", "CNC", " DQ", "NO CONTEST", "DECLARED", "COULD NOT CONTINUE")
    if any(b in mu for b in bad) or mu.startswith("DQ"):
        return None
    is_dec = ("DEC" in mu) or mu.startswith(("U-", "S-", "M-"))
    is_sub = mu.startswith("SUB")
    is_ko = ("KO" in mu) or ("TKO" in mu)
    if is_dec and not is_ko and not is_sub:
        return "f1_dec" if win_a else "f2_dec"
    if is_sub:
        return "f1_sub" if win_a else "f2_sub"
    if is_ko:
        return "f1_ko" if win_a else "f2_ko"
    return None


METHOD_6 = ["f1_ko", "f2_ko", "f1_sub", "f2_sub", "f1_dec", "f2_dec"]
METHOD_3 = ["ko", "sub", "dec"]


def method_3way(method: str) -> str | None:
    mu = str(method).strip().upper()
    bad = ("OVERTURNED", "CNC", "NO CONTEST", "DECLARED", "COULD NOT CONTINUE")
    if any(b in mu for b in bad) or mu.startswith("DQ"):
        return None
    if mu.startswith("SUB"):
        return "sub"
    if ("KO" in mu) or ("TKO" in mu):
        return "ko"
    if ("DEC" in mu) or mu.startswith(("U-", "S-", "M-")):
        return "dec"
    return None


# ---------------------------------------------------------------------------
# Weight class + symmetry helpers
# ---------------------------------------------------------------------------

def add_weight_class_dummies(df: pd.DataFrame, col: str = "Weight_Class") -> tuple[pd.DataFrame, list[str]]:
    d = pd.get_dummies(df[col].fillna("Unknown"), prefix="wc").astype(float)
    cols = list(d.columns)
    return pd.concat([df.reset_index(drop=True), d.reset_index(drop=True)], axis=1), cols


def symmetrize_matchup(df: pd.DataFrame, feat_cols: Iterable[str],
                       label_col: str = "Win_A",
                       a_prefix: str = "A_", b_prefix: str = "B_",
                       diff_prefix: str = "delta_") -> pd.DataFrame:
    """Double the training frame by swapping A/B and flipping the label.

    Assumes each A_/B_ column has a twin, and delta_* columns can be negated in-place.
    """
    swap = df.copy()
    swap[label_col] = 1 - swap[label_col]
    rename = {}
    a_cols = [c for c in df.columns if c.startswith(a_prefix)]
    b_cols = [c for c in df.columns if c.startswith(b_prefix)]
    for c in a_cols:
        rename[c] = b_prefix + c[len(a_prefix):]
    for c in b_cols:
        rename[c] = a_prefix + c[len(b_prefix):]
    swap = swap.rename(columns=rename)
    delta_cols = [c for c in df.columns if c.startswith(diff_prefix)]
    for c in delta_cols:
        swap[c] = -swap[c]
    return pd.concat([df, swap], ignore_index=True)
