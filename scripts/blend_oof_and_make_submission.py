# ...existing code...
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def safe_read_csv(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists(): return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def load_oof(exp_dir: Path) -> pd.DataFrame | None:
    oof = safe_read_csv(exp_dir / "artifacts" / "oof.csv")
    if oof is None: return None
    cols = list(oof.columns)
    # 標準: row_id,oof_pred,rule_violation,subreddit
    if "oof_pred" not in cols:
        # 互換: pred/probなどをoof_predに合わせる（存在すれば）
        for c in ["pred","prob","prediction","y_pred"]:
            if c in cols:
                oof = oof.rename(columns={c:"oof_pred"})
                break
    if "rule_violation" not in oof.columns and "target" in oof.columns:
        oof = oof.rename(columns={"target":"rule_violation"})
    return oof if {"row_id","oof_pred"}.issubset(oof.columns) else None

def fit_blender(oof_list: list[pd.DataFrame]) -> LogisticRegression | None:
    # row_idで内積し、yはどれか1つから拾う（同一データ前提）
    base = oof_list[0][["row_id","rule_violation"]].copy() if "rule_violation" in oof_list[0].columns else None
    Xs = []
    for i,df in enumerate(oof_list):
        df2 = df[["row_id","oof_pred"]].rename(columns={"oof_pred":f"p{i}"})
        if i == 0:
            mat = df2.copy()
        else:
            mat = mat.merge(df2, on="row_id", how="inner")
        if base is not None and "rule_violation" not in mat.columns and "rule_violation" in df.columns:
            base = base.merge(df[["row_id","rule_violation"]], on="row_id", how="inner")
    if base is None or "rule_violation" not in base.columns:
        return None
    merged = mat.merge(base[["row_id","rule_violation"]], on="row_id", how="inner")
    y = merged["rule_violation"].astype(int).values
    feats = [c for c in merged.columns if c.startswith("p")]
    X = merged[feats].values
    if X.shape[1] == 0: return None
    # ロジスティック回帰（Platt相当の線形ブレンド）
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:,1])
    print(f"Fitted blender on {X.shape} AUC={auc:.6f}")
    return clf

def blend_submissions(sub_paths: list[Path], out_path: Path, blender: LogisticRegression | None):
    subs = []
    for i,p in enumerate(sub_paths):
        df = safe_read_csv(p)
        if df is None or not {"row_id","rule_violation"}.issubset(df.columns):
            continue
        subs.append(df.rename(columns={"rule_violation":f"s{i}"}))
    if not subs:
        raise SystemExit("No valid submissions to blend.")
    sub = subs[0]
    for i in range(1,len(subs)):
        sub = sub.merge(subs[i], on="row_id", how="inner")
    score_cols = [c for c in sub.columns if c.startswith("s")]
    if blender is None or len(score_cols) == 1:
        # 等重み
        sub["rule_violation"] = np.mean(sub[score_cols].values, axis=1)
    else:
        X = sub[score_cols].values
        sub["rule_violation"] = blender.predict_proba(X)[:,1]
    out = sub[["row_id","rule_violation"]].copy()
    out["rule_violation"] = np.clip(out["rule_violation"].astype(np.float32), 0.0, 1.0)
    out.to_csv(out_path, index=False)
    print(f"Saved blended submission -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, help="experiment dirs like experiments/EXP004D experiments/EXP004G")
    ap.add_argument("--subs", nargs="+", required=True, help="submission paths aligned with --exp order")
    ap.add_argument("--out", required=True, help="output submission path")
    args = ap.parse_args()

    exp_dirs = [Path(e) for e in args.exp]
    oofs = [load_oof(Path(e)) for e in exp_dirs]
    oofs = [o for o in oofs if o is not None]
    blender = fit_blender(oofs) if len(oofs) >= 1 else None

    sub_paths = [Path(s) for s in args.subs]
    blend_submissions(sub_paths, Path(args.out), blender)

if __name__ == "__main__":
    main()
# ...existing code...