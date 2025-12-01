#!/usr/bin/env python3
"""
Shared helper for finding training data CSV files.
Provides Kaggle-aware autodetection logic used by both CLI and notebook workflows.
"""

import os
import glob
from pathlib import Path


def find_train_csv():
    """
    訓練データCSVファイルを自動検出
    
    検出順序:
    1. EDA_TRAIN_PATH環境変数（設定されており、ファイルが存在する場合）
    2. 共通のローカルパス
    3. Kaggleの入力パス（再帰的検索）
    
    Returns:
        str or None: 見つかったファイルパス、または None
    """
    # 1. 環境変数 EDA_TRAIN_PATH のチェック
    env_path = os.getenv('EDA_TRAIN_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    
    # 2. 共通のローカルパス
    local_candidate_paths = [
        'data/train.csv',
        'data/raw/train.csv', 
        'input/train.csv',
        'input/*/train*.csv',
        'dataset/train.csv',
        'sample_train.csv'
    ]
    
    for path in local_candidate_paths:
        if os.path.exists(path):
            return path
    
    # 3. Kaggleの入力パス（優先度順）
    # 3a. 再帰的に train*.csv を検索
    kaggle_patterns = [
        '/kaggle/input/**/train*.csv',
        '/kaggle/input/*/train*.csv', 
        '/kaggle/input/**/*.csv'
    ]
    
    for pattern in kaggle_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # 最初にマッチしたファイルを返す
            return matches[0]
    
    return None