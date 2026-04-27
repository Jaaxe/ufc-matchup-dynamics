"""Recompute NB 24's per-upset table exactly, plus the summary cell that
crashed because of a column-name bug (A_Emb_1 vs A_ae1)."""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = 'data/processed'
m = pd.read_csv(f'{DATA}/ufc_matchup_features.csv')
m['Event_Date'] = pd.to_datetime(m['Event_Date'])
with open(f'{DATA}/ufc_feature_groups.json') as f:
    groups = json.load(f)

feat_cols = []
for g in ['style_gmm_probs', 'hybrid', 'ae_embedding', 'rolling_rates',
          'form', 'physical', 'weight_class', 'ratings', 'heatmap', 'z_style']:
    if g in groups:
        feat_cols += [c for c in groups[g] if c in m.columns]
feat_cols = sorted(set(feat_cols))


def prospective_score(row, min_train=500):
    cutoff = row['Event_Date']
    prior = m[m['Event_Date'] < cutoff].copy()
    if len(prior) < min_train:
        return None, len(prior)
    flip = prior.copy()
    for c in prior.columns:
        if c.startswith('delta_'):
            flip[c] = -flip[c]
        elif c.startswith('A_') and 'B_' + c[2:] in prior.columns:
            a, b = prior[c], prior['B_' + c[2:]]
            flip[c], flip['B_' + c[2:]] = b, a
    flip['Win_A'] = 1 - flip['Win_A']
    train = pd.concat([prior, flip], ignore_index=True)
    X_tr = train[feat_cols].fillna(train[feat_cols].median()).to_numpy()
    y_tr = train['Win_A'].to_numpy()
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    X_te = row[feat_cols].fillna(train[feat_cols].median()).to_frame().T.to_numpy()
    return float(clf.predict_proba(X_te)[0, 1]), len(train)


upsets = [
    ('Matt Serra',  'Georges St-Pierre', '2007-04'),
    ('Ronda Rousey','Holly Holm',        '2015-11'),
    ('Renan Barao', 'TJ Dillashaw',      '2014-05'),
    ('Kamaru Usman','Leon Edwards',      '2022-08'),
    ('Amanda Nunes','Julianna Pena',     '2021-12'),
]

rows = []
for a, b, dh in upsets:
    mask = (((m['Fighter_A'] == a) & (m['Fighter_B'] == b)) |
            ((m['Fighter_A'] == b) & (m['Fighter_B'] == a)))
    mask &= m['Event_Date'].dt.strftime('%Y-%m') == dh
    r = m[mask].iloc[0]
    p_m, n = prospective_score(r)
    vA = np.array([r.get(f'A_ae{i+1}', np.nan) for i in range(3)], dtype=float)
    vB = np.array([r.get(f'B_ae{i+1}', np.nan) for i in range(3)], dtype=float)
    div = float(np.linalg.norm(vA - vB)) if not (np.isnan(vA).any() or np.isnan(vB).any()) else np.nan
    winner = r['Fighter_A'] if r['Win_A'] == 1 else r['Fighter_B']
    p_vegas = r.get('p_vegas_A', np.nan)
    rows.append({
        'date':      r['Event_Date'].strftime('%Y-%m-%d'),
        'A':         r['Fighter_A'],
        'B':         r['Fighter_B'],
        'winner':    winner,
        'method':    r.get('method_6', '?'),
        'vegas_A':   p_vegas,
        'model_A':   p_m,
        'residual_A':(p_m - p_vegas) if (p_m is not None and not pd.isna(p_vegas)) else np.nan,
        'style_div': div,
        'cluster_A': r.get('A_Cluster_k5', np.nan),
        'cluster_B': r.get('B_Cluster_k5', np.nan),
    })

out = pd.DataFrame(rows)
print('Per-fight residuals (symmetrized training, NB 24 hyperparameters):')
print(out.round(3).to_string(index=False))

print('\nResidual on the EVENTUAL WINNER (positive = model flagged correctly):')
for r in rows:
    if r['model_A'] is None or pd.isna(r.get('vegas_A', np.nan)):
        continue
    winner_is_A = r['winner'] == r['A']
    p_win_model = r['model_A'] if winner_is_A else (1 - r['model_A'])
    p_win_vegas = r['vegas_A'] if winner_is_A else (1 - r['vegas_A'])
    loser = r['B'] if winner_is_A else r['A']
    thr_tag = '  <-- would bet at theta=0.15' if (p_win_model - p_win_vegas) >= 0.15 else ''
    print(f"  {r['winner']:>18s} beat {loser:<22s}"
          f"  resid_on_winner = {p_win_model - p_win_vegas:+.3f}"
          f"  (model={p_win_model:.2f}, vegas={p_win_vegas:.2f}, AE_div={r['style_div']:.2f}){thr_tag}")
