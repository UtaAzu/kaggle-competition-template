#!/usr/bin/env python3
"""
Main training entry point for Kaggle competition.
Usage: python train.py --config config/train_config.yaml --mode tfidf
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, spmatrix

sys.path.append(str(Path(__file__).parent / "src"))

# === 1. 📝 設定ブロック (Configuration Block) ===
@dataclass
class TrainConfig:
    exp_id: str = "EXP_UNSET"
    seed: int = 42
    n_splits: int = 5
    stratify: bool = True
    group_col: Optional[str] = None
    text_col: str = "text"
    target_col: str = "rule_violation"
    input_train: str = "sample_train.csv"
    input_test: Optional[str] = None
    out_root: str = "experiments"
    is_kaggle: bool = False

    def __post_init__(self):
        self.is_kaggle = os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls(**data)
        obj.__post_init__()
        return obj

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        obj = cls(**data)
        obj.__post_init__()
        return obj

# === 2. ⏱️ ロギングブロック (Logging Block) ===
def setup_logger(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("train")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    logger.setLevel(level)
    return logger

def log_environment_versions(logger: logging.Logger):
    import platform, importlib
    logger.info(f"Python: {sys.version.replace(chr(10),' ')}")
    logger.info(f"Platform: {platform.platform()}")
    for lib in ("numpy","pandas","sklearn","torch","transformers","datasets"):
        try:
            m = importlib.import_module(lib)
            logger.info(f"{lib}: {getattr(m,'__version__','unknown')}")
        except Exception:
            logger.info(f"{lib}: not available")

# === 3. 📊 データロードブロック (Data Loading Block) ===
def read_csv_flexible(path: str, **kwargs) -> pd.DataFrame:
    encodings = [None, "utf-8-sig", "utf-8", "latin-1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            if enc is None:
                df = pd.read_csv(path, **kwargs)
            else:
                df = pd.read_csv(path, encoding=enc, **kwargs)
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin-1", **kwargs)

# === 4. 🔧 前処理ブロック (Preprocessing Block) ===
# TF-IDF ベクトル化（ここに配置）

# === 5. 🎓 学習・評価関数ブロック (Training & Evaluation Functions Block) ===
def run_tfidf_cv(cfg: TrainConfig, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Loading train data: %s", cfg.input_train)
    train = read_csv_flexible(cfg.input_train)
    text_col = cfg.text_col
    target = cfg.target_col

    if target not in train.columns:
        logger.error(f"Target column '{target}' not found in train data.")
        return {}

    X_text = train[text_col].fillna("")
    y: pd.Series = train[target].astype(int)

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X: csr_matrix = vec.fit_transform(X_text)  # type: ignore  # Pylanceがspmatrixとして認識するが、実際はcsr_matrix

    oof_preds = np.zeros(len(train))
    outs = ensure_out_dirs(cfg)
    if cfg.group_col and cfg.group_col in train.columns:
        splitter = GroupKFold(n_splits=cfg.n_splits)
        groups = train[cfg.group_col]
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        splits = splitter.split(X, y)

    fold = 0
    fold_aucs = []
    for tr_idx, val_idx in splits:
        fold += 1
        X_tr = X[tr_idx, :]  # csr_matrix supports slicing
        X_val = X[val_idx, :]
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        pred = clf.predict_proba(X_val)[:,1]
        oof_preds[val_idx] = pred
        auc = roc_auc_score(y_val, pred)
        fold_aucs.append(float(auc))
        logger.info("Fold %d auc: %.5f", fold, auc)
        if outs["models"] is not None:
            joblib.dump(clf, outs["models"] / f"model_fold_{fold}.joblib")

    overall_auc = roc_auc_score(y, oof_preds)
    logger.info("OOF AUC: %.5f", overall_auc)

    oof_df = pd.DataFrame({"row_id": train.index, "oof_pred": oof_preds})
    oof_df[target] = train[target]
    if cfg.group_col and cfg.group_col in train.columns:
        oof_df[cfg.group_col] = train[cfg.group_col]
    if outs["artifacts"] is not None:
        oof_path = outs["artifacts"] / "oof.csv"
        oof_df.to_csv(oof_path, index=False)

        metrics = {
            "oof_auc": float(overall_auc),
            "fold_aucs": fold_aucs,
            "date": datetime.utcnow().isoformat() + "Z"
        }
        with open(outs["artifacts"] / "metrics.json", "w") as f:
            json.dump(metrics, f)

        run_json = {
            "exp_id": cfg.exp_id,
            "config": asdict(cfg),
            "metrics": metrics,
            "date": datetime.utcnow().isoformat() + "Z",
            "git_commit": get_git_commit()
        }
        if outs["base"] is not None:
            with open(outs["base"] / "run.json", "w") as f:
                json.dump(run_json, f)

        try:
            import shutil
            kaggle_out = outs.get("kaggle_out")
            if kaggle_out is not None and outs["artifacts"] is not None:
                shutil.copy(oof_path, kaggle_out / "oof.csv")
                shutil.copy(outs["artifacts"] / "metrics.json", kaggle_out / "metrics.json")
                if outs["base"] is not None:
                    shutil.copy(outs["base"] / "run.json", Path("/kaggle/output") / "run.json")
            kaggle_models = outs.get("kaggle_models")
            if kaggle_models is not None and outs["models"] is not None:
                for f in outs["models"].glob("*"):
                    if f.is_file():
                        shutil.copy(f, kaggle_models / f.name)
                    elif f.is_dir():
                        shutil.copytree(f, kaggle_models / f.name, dirs_exist_ok=True)
                prune_excess_artifacts(kaggle_models)
        except Exception as e:
            logger.error(f"Could not copy artifacts to /kaggle/output: {e}")
            sys.exit(1)

        models_dir = outs.get("models")
        if models_dir is not None:
            prune_excess_artifacts(models_dir)
        return {"oof_path": str(oof_path), "metrics": metrics}
    return {}

def get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

# === 6. 💾 成果物保存ブロック (Artifact Saving Block) ===
def prune_excess_artifacts(model_dir: Optional[Path]):
    # 002prompt 推奨: 除外リストを明記
    # 除外: checkpoint-*, optimizer.pt, scheduler.pt, rng_state.pth, training_args.bin
    import shutil
    if model_dir is None:
        return
    for fold_dir in model_dir.glob("fold_*"):
        for ckpt in fold_dir.glob("checkpoint-*"):
            if ckpt.is_dir():
                shutil.rmtree(ckpt)
        for fname in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]:
            f = fold_dir / fname
            if f.exists():
                f.unlink()

def ensure_out_dirs(cfg: TrainConfig) -> Dict[str, Optional[Path]]:
    exp_id = cfg.exp_id
    out_root = cfg.out_root
    base = Path(out_root) / exp_id
    artifacts = base / "artifacts"
    models = base / "models"
    artifacts.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    kaggle_out = Path("/kaggle/output") / exp_id / "artifacts" if cfg.is_kaggle else None
    kaggle_models = Path("/kaggle/output") / exp_id / "models" if cfg.is_kaggle else None
    try:
        if kaggle_out is not None:
            kaggle_out.mkdir(parents=True, exist_ok=True)
        if kaggle_models is not None:
            kaggle_models.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger = logging.getLogger("train")
        logger.error(f"Could not create Kaggle output dirs: {e}")
        sys.exit(1)
    return {
        "base": base,
        "artifacts": artifacts,
        "models": models,
        "kaggle_out": kaggle_out,
        "kaggle_models": kaggle_models,
    }

def run_llm_placeholder(cfg: TrainConfig, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("LLM training placeholder called. Implement Trainer/PEFT here.")
    # 例:
    # model.save_pretrained(str(models_dir))
    # tokenizer.save_pretrained(str(models_dir/'tokenizer'))
    # prune_excess_artifacts(models_dir)
    # 必要最低限の成果物のみ保存（model.safetensors, tokenizer, config.json等。checkpoint, optimizer.pt等は除外）
    return {}

# === 7. 🚀 実行トリガーブロック (Execution Trigger Block) ===
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", help="Path to config file (YAML or JSON)", required=False)
    parser.add_argument("--exp_id", help="Experiment id", required=False)
    parser.add_argument("--mode", choices=["tfidf", "llm"], default="tfidf")
    args = parser.parse_args(argv)

    if args.config:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            cfg = TrainConfig.from_yaml(args.config)
        else:
            cfg = TrainConfig.from_json(args.config)
    else:
        cfg = TrainConfig()
    if args.exp_id:
        cfg.exp_id = args.exp_id

    outs = ensure_out_dirs(cfg)
    if outs["base"] is not None:
        log_path = outs["base"] / "train.log"
        logger = setup_logger(log_path)
        log_environment_versions(logger)

        logger.info("Run config: %s", cfg)

        if args.mode == "tfidf":
            run_tfidf_cv(cfg, logger)
        else:
            run_llm_placeholder(cfg, logger)

        logger.info("Training completed successfully!")
        return 0
    return 1

# === 8. 🔍 検証ブロック (Validation Block) ===
# 002prompt 推奨: 追加の検証（例: 出力チェック）
def validate_outputs(outs: Dict[str, Optional[Path]], logger: logging.Logger):
    if outs["artifacts"] is not None and not outs["artifacts"].exists():
        logger.error("Artifacts dir not created")
        sys.exit(1)
    # 他の検証を追加可能

# main 内で呼び出し
# validate_outputs(outs, logger)

if __name__ == "__main__":
    raise SystemExit(main())
