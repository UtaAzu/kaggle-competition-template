"""Utility to save standardized OOF and submission files for experiments.

Usage example:
    from scripts.save_oof import save_oof_and_submission
    save_oof_and_submission(ARTIFACTS_DIR, OUT_DIR, train_df, test_df, oof, test_preds, groups=groups)

Outputs:
  artifacts/<EXP_ID>/oof.csv (columns: row_id, oof_pred, rule_violation (if available), group (if provided))
  <OUT_DIR>/submission.csv
  artifacts/<EXP_ID>/metrics.json (basic)
  <OUT_DIR>/run.json (augmented)
"""
from pathlib import Path
from typing import Optional, Sequence, Dict, Any
import pandas as pd
import numpy as np
import json


def save_oof_and_submission(
    artifacts_dir: Path,
    out_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oof: np.ndarray,
    test_preds: np.ndarray,
    groups: Optional[Sequence] = None,
    target_col: str = "rule_violation",
    oof_name: str = "oof.csv",
    submission_name: str = "submission.csv",
    metrics_name: str = "metrics.json",
    run_name: str = "run.json",
) -> Dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    out_dir = Path(out_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build OOF dataframe
    if "row_id" in train.columns:
        oof_df = train[["row_id"]].copy()
    else:
        oof_df = pd.DataFrame({"row_id": train.index})
    oof_df["oof_pred"] = list(oof)

    if target_col in train.columns:
        oof_df[target_col] = train[target_col].values

    if groups is not None:
        try:
            oof_df["group"] = pd.Series(groups).values
        except Exception:
            oof_df["group"] = groups

    oof_path = artifacts_dir / oof_name
    oof_df.to_csv(oof_path, index=False)

    # Save submission
    if "row_id" in test.columns:
        sub_ids = test["row_id"]
    else:
        sub_ids = np.arange(len(test))
    sub = pd.DataFrame({"row_id": sub_ids, target_col: list(test_preds)})
    submission_path = out_dir / submission_name
    sub.to_csv(submission_path, index=False)

    # Basic metrics file
    metrics = {
        "n_samples": int(len(oof_df)),
    }
    try:
        import sklearn.metrics as _m
        if target_col in oof_df.columns:
            try:
                metrics["oof_auc"] = float(_m.roc_auc_score(oof_df[target_col].astype(int), oof_df["oof_pred"]))
            except Exception:
                metrics["oof_auc"] = None
    except Exception:
        metrics["oof_auc"] = None

    metrics_path = artifacts_dir / metrics_name
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Minimal run.json augmentation
    run = {
        "artifacts_dir": str(artifacts_dir),
        "oof_path": str(oof_path),
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
    }
    run_path = out_dir / run_name
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)

    return {"oof_path": str(oof_path), "submission_path": str(submission_path), "metrics_path": str(metrics_path), "run_path": str(run_path)}
# ...existing code...
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Sequence

def save_oof_and_submission(
    artifacts_dir: Path,
    out_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oof: np.ndarray,
    test_preds: np.ndarray,
    groups: Optional[Sequence] = None,
    target_col: str = "rule_violation",
    oof_name: str = "oof.csv",
    submission_name: str = "submission.csv",
):
    """
    標準形式で OOF と submission を保存する共通関数。
    - oof: 長さ == len(train) の予測確率
    - test_preds: 長さ == len(test) の予測確率（submission用）
    出力:
      artifacts_dir / oof_name
      out_dir / submission_name
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # oof_df 組立
    if "row_id" in train.columns:
        oof_df = train[["row_id"]].copy()
    else:
        oof_df = pd.DataFrame({"row_id": train.index})
    oof_df["oof_pred"] = oof

    if target_col in train.columns:
        oof_df[target_col] = train[target_col]

    if groups is not None:
        oof_df["group"] = groups

    oof_path = artifacts_dir / oof_name
    oof_df.to_csv(oof_path, index=False)

    # submission
    if "row_id" in test.columns:
        sub_ids = test["row_id"]
    else:
        sub_ids = np.arange(len(test))
    sub = pd.DataFrame({"row_id": sub_ids, "rule_violation": test_preds})
    submission_path = out_dir / submission_name
    sub.to_csv(submission_path, index=False)

    return {"oof_path": str(oof_path), "submission_path": str(submission_path)}
# ...existing code...