"""
Data processing utilities for Kaggle competition.
"""
import hashlib
import logging
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Optional import for StratifiedGroupKFold (requires scikit-learn >= 1.1)
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore


def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """Load training or test data."""
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Loaded data from {file_path}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def create_text_features(df: pd.DataFrame, text_fields: List[str]) -> pd.DataFrame:
    """Create combined text features for TF-IDF."""
    df = df.copy()
    
    # Handle missing values by filling with empty string
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('')
    
    # Combine all text fields with special separators
    text_parts = []
    for field in text_fields:
        if field in df.columns:
            text_parts.append(df[field].astype(str))
    
    if text_parts:
        # Concatenate series properly
        df['combined_text'] = text_parts[0]
        for i in range(1, len(text_parts)):
            df['combined_text'] = df['combined_text'] + ' [SEP] ' + text_parts[i]
    else:
        df['combined_text'] = ''
    
    logger.info(f"Created combined text features using fields: {text_fields}")
    return df


# --- Helpers for group id creation (used when cv.type == 'stratified_group') ---

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"https?://\S+", "[url]", s)
    s = re.sub(r"([!?.]){2,}", r"\1", s)
    return s


def _hash_text_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def create_cv_splits(
    df: pd.DataFrame,
    target_col: str,
    cv_config: dict,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create cross-validation splits.

    Supports:
      - cv.type == 'stratified'
      - cv.type == 'stratified_group' (requires StratifiedGroupKFold)
    """
    cv_type = cv_config.get("type", "stratified")
    n_splits = cv_config.get("n_splits", 5)
    random_state = cv_config.get("random_state", cv_config.get("seed", 42))
    shuffle = cv_config.get("shuffle", True)

    if cv_type == "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(df, df[target_col]))

    elif cv_type == "stratified_group":
        if StratifiedGroupKFold is None:
            raise RuntimeError(
                "StratifiedGroupKFold is unavailable. Please install scikit-learn >= 1.1."
            )
        group_col = cv_config.get("group_column", "group_id")
        # If group column is missing, optionally create from text column
        if group_col not in df.columns:
            text_col = cv_config.get("group_text_column")
            if not text_col or text_col not in df.columns:
                raise ValueError(
                    f"Group column '{group_col}' not found and no valid 'group_text_column' specified."
                )
            # Create group_id on the fly
            df = df.copy()
            df[group_col] = df[text_col].astype(str).map(_normalize_text).map(_hash_text_md5)
            logger.info(
                f"Created missing group column '{group_col}' from text column '{text_col}'."
            )
        groups = df[group_col]
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(X=np.zeros(len(df)), y=df[target_col], groups=groups))

    else:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    logger.info(f"Created {len(splits)} CV splits using {cv_type}")
    return splits


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning."""
    df = df.copy()
    
    # Remove unnamed columns that appear to be artifacts
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info(f"Removed unnamed columns: {unnamed_cols}")
    
    # Remove rows with missing target values
    if 'rule_violation' in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=['rule_violation'])
        final_len = len(df)
        if initial_len != final_len:
            logger.info(f"Removed {initial_len - final_len} rows with missing target values")
    
    return df