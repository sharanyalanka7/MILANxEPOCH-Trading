"""
Stage-2 diagnostic script.
Usage:
    python stage2_diagnostics.py path/to/predictions.csv
It diagnoses the prediction file without changing the original experiment.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rank_ic(g, pred, actual):
    return g[pred].corr(g[actual], method="spearman")

def main(path):
    df = pd.read_csv(path)
    required = ["fold","actual","ExtraTrees","RandomForest","GradientBoosting","Ensemble","timestamp","asset_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("\\nDATA")
    print(df.shape)
    print("Folds:", sorted(df["fold"].unique()))
    print("Dates:", df["timestamp"].nunique())
    print("Assets:", df["asset_id"].nunique())

    models = ["ExtraTrees","RandomForest","GradientBoosting","Ensemble"]
    rows=[]
    for m in models:
        ic = df[m].corr(df["actual"])
        ric = df[m].corr(df["actual"], method="spearman")
        rows.append([m,ic,ric])
    print("\\nOVERALL CORRELATIONS")
    print(pd.DataFrame(rows, columns=["model","IC","RankIC"]).to_string(index=False))

    fold_rows=[]
    for fold, g in df.groupby("fold"):
        for m in models:
            fold_rows.append([fold,m,g[m].corr(g["actual"]),g[m].corr(g["actual"],method="spearman")])
    fold_df=pd.DataFrame(fold_rows,columns=["fold","model","IC","RankIC"])
    print("\\nFOLD CORRELATIONS")
    print(fold_df.to_string(index=False))

    plt.figure(figsize=(9,5))
    for m in models:
        z=fold_df[fold_df.model==m]
        plt.plot(z["fold"],z["IC"],marker="o",label=m)
    plt.axhline(0,linewidth=1)
    plt.xlabel("Fold"); plt.ylabel("IC"); plt.title("IC by Fold"); plt.legend()
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(9,5))
    for m in models:
        z=fold_df[fold_df.model==m]
        plt.plot(z["fold"],z["RankIC"],marker="o",label=m)
    plt.axhline(0,linewidth=1)
    plt.xlabel("Fold"); plt.ylabel("RankIC"); plt.title("RankIC by Fold"); plt.legend()
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python stage2_diagnostics.py predictions.csv")
    main(sys.argv[1])
