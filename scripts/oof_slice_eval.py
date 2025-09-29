import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_safe(y_true, y_pred):
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="oof.csv")
    ap.add_argument("--train", required=True, help="sample_train.csv")
    ap.add_argument("--out", default="oof_slice_metrics.csv", help="出力CSV")
    ap.add_argument("--len-bins", default="0,50,100,200,400,1000", help="本文長ビン")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    train = pd.read_csv(args.train)

    # row_idで突合
    if "row_id" in oof.columns and "row_id" in train.columns:
        df = pd.merge(train, oof[["row_id", "oof_pred"]], on="row_id", how="inner")
    else:
        n = min(len(train), len(oof))
        df = train.iloc[:n].copy()
        df["oof_pred"] = oof.iloc[:n]["oof_pred"].values

    # 本文長ビン
    if "body" in df.columns:
        bins = [int(x) for x in args.len_bins.split(",")]
        df["body_len"] = df["body"].astype(str).str.len()
        df["len_bin"] = pd.cut(df["body_len"], bins=bins, right=False, include_lowest=True)
    else:
        df["len_bin"] = "unknown"

    # 全体AUC
    overall = auc_safe(df["rule_violation"], df["oof_pred"])

    rows = []
    def add_group(col):
        for k, g in df.groupby(col):
            rows.append({
                "group_by": col,
                "key": str(k),
                "count": int(len(g)),
                "auc": float(auc_safe(g["rule_violation"], g["oof_pred"]))
            })

    for col in ["rule", "subreddit", "len_bin"]:
        if col in df.columns:
            add_group(col)

    res = pd.DataFrame(rows).sort_values(["group_by", "auc"])
    res["overall_auc"] = overall
    res["delta_vs_overall"] = res["auc"] - overall

    res.to_csv(args.out, index=False)
    print(f"全体AUC: {overall:.4f}")
    print("スライスごとのAUC（下位5件）:")
    print(res.groupby("group_by").apply(lambda x: x.nsmallest(5, "auc")))

if __name__ == "__main__":
    main()