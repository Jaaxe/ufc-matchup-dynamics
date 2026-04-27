"""One-off helper: compute the canonical K=5 cluster label table
for the thesis, mirroring cell 8 of notebook 25.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, "data", "processed")

gmm = pd.read_csv(os.path.join(DATA, "ufc_gmm_comparison.csv"))
ae = pd.read_csv(os.path.join(DATA, "ufc_ae_embeddings.csv"))
df = gmm.merge(ae, on="Fighter", how="left")

# Identify Z-score feature columns
z_cols = ["Sig_Str_PM_Z", "Takedown_Att_PM_Z", "Sub_Att_PM_Z", "Control_Ratio_Z"]
missing = [c for c in z_cols if c not in df.columns]
if missing:
    # Load from the modeling dataset
    fp = pd.read_csv(os.path.join(DATA, "ufc_modeling_data_final.csv"))
    df = df.merge(fp[["Fighter"] + [c for c in z_cols if c in fp.columns]],
                  on="Fighter", how="left")

prof = df.groupby("Cluster_k5")[z_cols].mean().round(2)
prof["n_pool"] = df.groupby("Cluster_k5").size()

def _label_row(r):
    if r["Sub_Att_PM_Z"] > 0.8 and r["Sig_Str_PM_Z"] < -0.4:
        return "submission-leaning grappler"
    if r["Takedown_Att_PM_Z"] > 0.6 and r["Control_Ratio_Z"] > 0.6:
        return "takedown-heavy wrestler"
    if r["Sig_Str_PM_Z"] > 0.3 and r["Takedown_Att_PM_Z"] < -0.2:
        return "active distance striker"
    if abs(r["Sig_Str_PM_Z"]) < 0.25 and abs(r["Takedown_Att_PM_Z"]) < 0.25:
        return "baseline generalist"
    return "low-activity"

prof["label"] = prof.apply(_label_row, axis=1)
print("K=5 cluster-ID -> narrative-label map:")
print(prof[["label", "n_pool"] + z_cols].to_string())

champions = [
    "Max Holloway", "Israel Adesanya", "Francis Ngannou",
    "Khabib Nurmagomedov", "Islam Makhachev", "Jon Jones",
    "Amanda Nunes", "Jose Aldo", "Alexander Volkanovski",
    "Georges St-Pierre", "Kamaru Usman", "Leon Edwards",
    "Conor McGregor", "Daniel Cormier", "Demetrious Johnson",
    "Henry Cejudo", "Dustin Poirier", "Zhang Weili",
    "Matt Hughes", "Charles Oliveira", "Alex Pereira",
    "Randy Couture", "TJ Dillashaw", "Renan Barao",
    "Dominick Cruz",
]
print("\nChampion placements by cluster:")
for k, row in prof.iterrows():
    lab = row["label"]
    sub = df[(df["Cluster_k5"] == k) & (df["Fighter"].isin(champions))]
    names = sub.sort_values("Hybrid_Score_k5")["Fighter"].tolist() \
        if "Hybrid_Score_k5" in sub.columns else sub["Fighter"].tolist()
    print(f"  Cluster {int(k)} ({lab:<30s}, n={int(row['n_pool'])}): "
          f"{len(names)} of panel -> {', '.join(names) if names else '(none)'}")
