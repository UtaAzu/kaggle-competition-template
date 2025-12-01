"""Example metric utilities for template repo

This file provides simple F1 & RLE utilities and an example evaluation function.
Use or adapt these for your competition evaluation. The functions are intentionally
small and dependency-free for easy reading and unit-testing.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for binary labels.

    Args:
        y_true: binary ground truth array
        y_pred: binary predicted array
    Returns:
        precision, recall, f1
    """
    assert y_true.shape == y_pred.shape
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask to RLE (row-major)."""
    pixels = mask.flatten(order="F")  # For some competitions | Fortran order is used
    # Convert to run-length encoding
    runs = []
    prev = 0
    for i, p in enumerate(pixels, start=1):
        if p and not prev:
            runs.extend([i, 1])
            prev = 1
        elif p and prev:
            runs[-1] += 1
        else:
            prev = 0
    return "".join([str(x) + " " for x in runs]).strip()


def rle_decode(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE to a binary mask.

    Args:
        rle: run-length string "start length start length ..."
        shape: (height, width)

    Returns:
        mask array of shape
    """
    s = rle.strip().split()
    starts = [int(x) - 1 for x in s[0::2]]
    lengths = [int(x) for x in s[1::2]]
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for st, ln in zip(starts, lengths):
        mask[st:st + ln] = 1
    return mask.reshape((shape[1], shape[0])).T


def evaluate_oof(oof_df: pd.DataFrame, target_col: str = "label", pred_col: str = "pred") -> dict:
    """Evaluate OOF dataframe with simple binary F1.

    Expects oof_df to contain per-image label and pred columns.
    This is *example* code - customize thresholds and mask handling as needed.

    Returns a dict with precision, recall, f1 and counts.
    """
    # For classification threshold of 0.5 for example purposes
    y_true = (oof_df[target_col].values > 0).astype(int)
    y_pred = (oof_df[pred_col].values >= 0.5).astype(int)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": len(y_true)
    }
