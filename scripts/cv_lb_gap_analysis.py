#!/usr/bin/env python3
"""
CV vs LB gap analysis with manual LB score input
- 探索: experiments/*/notebooks/**/artifacts/oof.csv, run.json
- 手動LB指定: --manual-lb <exp_name> <score>
- 出力: experiments/cv_gap_analysis/cv_lb_gap_summary.csv
"""
import json, sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(".")
EXP_ROOT = ROOT / "experiments"
OUT_DIR = ROOT / "experiments" / "cv_gap_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_train():
    candidates = [
        ROOT / "train.csv", 
        ROOT / "sample_train.csv",
        Path("/kaggle/input/jigsaw-agile-community-rules/train.csv")  # Kaggle 環境用
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(p, encoding='latin1')
                except Exception as e:
                    print(f"Failed to read {p} with latin1: {e}")
                    continue
            except Exception as e:
                print(f"Failed to read {p}: {e}")
                continue
    return None

def read_oof(p):
    try:
        df = pd.read_csv(p, encoding='utf-8')
        if len(df.columns) >= 4:
            df.columns = ['row_id', 'oof_pred', 'actual', 'group_col']
        elif len(df.columns) == 3:
            df.columns = ['row_id', 'oof_pred', 'actual']
        elif len(df.columns) == 2:
            df.columns = ['row_id', 'oof_pred']
        # カラム名が rule_violation の場合、actual にリネーム
        if 'rule_violation' in df.columns:
            df.rename(columns={'rule_violation': 'actual'}, inplace=True)
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(p, encoding='latin1')
            if len(df.columns) >= 4:
                df.columns = ['row_id', 'oof_pred', 'actual', 'group_col']
            elif len(df.columns) == 3:
                df.columns = ['row_id', 'oof_pred', 'actual']
            elif len(df.columns) == 2:
                df.columns = ['row_id', 'oof_pred']
            # カラム名が rule_violation の場合、actual にリネーム
            if 'rule_violation' in df.columns:
                df.rename(columns={'rule_violation': 'actual'}, inplace=True)
            return df
        except Exception as e:
            print(f"Failed to read {p} with latin1: {e}")
            return None
    except Exception as e:
        print(f"Failed to read {p}: {e}")
        return None

def compute_auc_from_oof(df, train_df=None):
    if df is None:
        return None
    if 'actual' in df.columns:
        try:
            return roc_auc_score(df['actual'], df['oof_pred'])
        except Exception as e:
            print(f"AUC computation failed: {e}")
            return None
    elif train_df is not None and 'row_id' in df.columns:
        # train_df と join してラベルを取得
        try:
            merged = df.merge(train_df[['row_id', 'rule_violation']], on='row_id', how='left')
            if 'rule_violation' in merged.columns:
                # NaN を除去して AUC 計算
                merged = merged.dropna(subset=['rule_violation', 'oof_pred'])
                if len(merged) > 0:
                    return roc_auc_score(merged['rule_violation'], merged['oof_pred'])
                else:
                    print("No valid rows after dropna for AUC computation.")
                    return None
        except Exception as e:
            print(f"Join-based AUC failed: {e}")
    return None

def load_json_if_exists(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def discover_experiments(root=EXP_ROOT):
    exps = []
    # main experiments
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'): 
            continue
        oof_path = exp_dir / "artifacts" / "oof.csv"
        if oof_path.exists():
            exps.append((exp_dir.name, oof_path))
    
    # CV strategy variants
    cv_base = root / "EXP003D" / "notebooks" / "CV-strategy"
    if cv_base.exists():
        for variant in cv_base.iterdir():
            if not variant.is_dir(): 
                continue
            oof_path = variant / "artifacts" / "oof.csv"
            if oof_path.exists():
                exps.append((variant.name, oof_path))
    return exps

def main():
    parser = argparse.ArgumentParser(description="CV-LB gap analysis with manual LB scores")
    parser.add_argument("--manual-lb", nargs=2, action="append", metavar=("EXP", "SCORE"), 
                        help="Manual LB score: --manual-lb 'v3_GroupKFold (md5(body))' 0.514")
    args = parser.parse_args()

    # build manual LB mapping
    manual_lb = {}
    if args.manual_lb:
        for exp, score in args.manual_lb:
            try:
                manual_lb[exp] = float(score)
                print(f"Manual LB: {exp} -> {score}")
            except ValueError:
                print(f"Invalid score for {exp}: {score}")

    train_df = find_train()
    print(f"Found train data: {train_df is not None}")
    
    experiments = discover_experiments()
    print(f"Found {len(experiments)} experiments")
    
    results = []
    for name, oof_path in experiments:
        print(f"Processing: {name}")
        oof_df = read_oof(oof_path)
        computed_oof_auc = compute_auc_from_oof(oof_df, train_df)
        
        # find run.json
        run_json_candidates = [
            oof_path.parent / "run.json",
            oof_path.parent.parent / "run.json",
            EXP_ROOT / name / "run.json"
        ]
        run_json = None
        run = {}
        for candidate in run_json_candidates:
            if candidate.exists():
                run_json = candidate
                run = load_json_if_exists(candidate)
                break
        
        # get recorded values
        recorded_oof = run.get("oof_auc")
        
        # prioritize manual LB, then run.json (leaderboard.public_lb もチェック)
        public_lb = manual_lb.get(name)
        if public_lb is None:
            # run.json の構造を柔軟にチェック
            if "leaderboard" in run and isinstance(run["leaderboard"], dict):
                public_lb = run["leaderboard"].get("public_lb")
            if public_lb is None:
                for k in ("public_lb", "lb", "public_score"):
                    if k in run:
                        public_lb = run[k]
                        break
        
        results.append({
            "experiment": name,
            "oof_path": str(oof_path),
            "computed_oof_auc": computed_oof_auc,
            "recorded_oof_auc": recorded_oof,
            "recorded_public_lb": public_lb,
            "run_json": str(run_json) if run_json else ""
        })
    
    # save results
    df_results = pd.DataFrame(results)
    df_results["gap"] = pd.to_numeric(df_results["recorded_public_lb"], errors='coerce') - pd.to_numeric(df_results["computed_oof_auc"], errors='coerce')
    
    out_path = OUT_DIR / "cv_lb_gap_summary.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Wrote summary -> {out_path}")
    
    # display gap analysis
    print("\n=== CV-LB Gap Analysis ===")
    valid_gaps = df_results.dropna(subset=["gap"])
    if len(valid_gaps) > 0:
        print(valid_gaps[["experiment", "computed_oof_auc", "recorded_public_lb", "gap"]].to_string(index=False))
        print(f"\nMean gap: {valid_gaps['gap'].mean():.4f}")
        print(f"Std gap: {valid_gaps['gap'].std():.4f}")
    else:
        print("No valid gaps found. Check manual LB scores.")
        # 追加: 各実験の public_lb が None の場合、警告
        missing_lb = df_results[df_results["recorded_public_lb"].isna()]
        if len(missing_lb) > 0:
            print("Missing LB scores for:")
            for _, row in missing_lb.iterrows():
                print(f"  - {row['experiment']}: run.json at {row['run_json']}")

if __name__ == "__main__":
    main()