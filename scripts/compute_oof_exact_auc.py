# ...existing code...
#!/usr/bin/env python3
"""
Compute exact OOF AUC by joining oof.csv with train.csv when possible.
Writes experiments/cv_gap_analysis/oof_detailed.csv and prints gaps.
"""
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
EXP_ROOT = ROOT / "experiments"
OUT_DIR = EXP_ROOT / "cv_gap_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CAND = [ROOT / "train.csv", ROOT / "sample_train.csv", Path(os.environ.get('KAGGLE_DATASET_TRAIN', '/kaggle/input/<dataset-slug>/train.csv'))]

def find_train():
    for p in TRAIN_CAND:
        if p.exists():
            return p
    return None

def read_df_safe(p):
    try:
        return pd.read_csv(p)
    except Exception:
        encs = ['utf-8','utf-8-sig','latin1','iso-8859-1','cp1252']
        for e in encs:
            try:
                return pd.read_csv(p, encoding=e)
            except Exception:
                continue
    return None

def find_oof_paths(root=EXP_ROOT):
    res = []
    for p in root.rglob("*/artifacts/oof.csv"):
        exp_dir = p.parents[1]
        res.append((exp_dir.name, p))
    return sorted(res)

def compute_auc(oof_df, train_df=None):
    cols = [c.lower() for c in oof_df.columns.astype(str)]
    oof_df.columns = cols
    if "oof_pred" not in cols:
        # try common rename patterns
        if len(oof_df.columns) >= 2:
            oof_df = oof_df.rename(columns={oof_df.columns[1]: "oof_pred"})
        else:
            return np.nan
    # case1: oof already contains target
    if "rule_violation" in oof_df.columns:
        try:
            return float(roc_auc_score(oof_df["rule_violation"].astype(int), oof_df["oof_pred"]))
        except Exception:
            return np.nan
    # case2: join by row_id
    if "row_id" in oof_df.columns and train_df is not None and "row_id" in train_df.columns:
        merged = oof_df.merge(train_df[["row_id","rule_violation"]], on="row_id", how="left")
        if merged["rule_violation"].notna().any():
            try:
                return float(roc_auc_score(merged["rule_violation"].astype(int), merged["oof_pred"]))
            except Exception:
                return np.nan
    # case3: align by index if lengths match
    if train_df is not None and len(oof_df) == len(train_df):
        try:
            return float(roc_auc_score(train_df["rule_violation"].astype(int), oof_df["oof_pred"]))
        except Exception:
            return np.nan
    return np.nan

def main():
    train_path = find_train()
    train_df = read_df_safe(train_path) if train_path else None
    oofs = find_oof_paths()
    rows = []
    for name, op in oofs:
        run_json = op.parent.parent / "run.json"
        metrics_json = op.parent / "metrics.json"
        run = {}
        if run_json.exists():
            try:
                run = json.loads(run_json.read_text())
            except Exception:
                run = {}
        elif metrics_json.exists():
            try:
                run = json.loads(metrics_json.read_text())
            except Exception:
                run = {}
        oof_df = read_df_safe(op)
        if oof_df is None or oof_df.empty:
            comp = np.nan
        else:
            comp = compute_auc(oof_df.copy(), train_df)
        recorded_oof = run.get("oof_auc") or run.get("cv_auc")
        recorded_lb = None
        for k in ("public_lb","lb","public_score","recorded_public_lb"):
            if k in run:
                recorded_lb = run[k]; break
        rows.append({
            "experiment": name,
            "oof_path": str(op),
            "computed_oof_auc": None if np.isnan(comp) else float(comp),
            "recorded_oof_auc": recorded_oof,
            "recorded_public_lb": recorded_lb,
            "run_json": str(run_json) if run_json.exists() else ""
        })
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "oof_detailed.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote -> {out_path}")
    print(out.to_string(index=False))
    # print gaps where possible
    for _, r in out.iterrows():
        if pd.notna(r["computed_oof_auc"]) and r["recorded_public_lb"] is not None:
            gap = float(r["recorded_public_lb"]) - float(r["computed_oof_auc"])
            print(f"{r['experiment']}: LB {r['recorded_public_lb']} - computed OOF {r['computed_oof_auc']:.6f} = gap {gap:.6f}")

if __name__ == "__main__":
    main()
# ...existing code...