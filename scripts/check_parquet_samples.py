import pandas as pd
from pathlib import Path

# チェック対象ファイル
files = [
    "train_tracking/AdaptableSnail/438887472.parquet",
    "train_annotation/AdaptableSnail/438887472.parquet",
    "test_tracking/AdaptableSnail/438887472.parquet",
]

for f in files:
    path = Path(f)
    print(f"\n=== {path} ===")
    if path.exists():
        print("✅ ファイルが存在します")
        try:
            df = pd.read_parquet(path)
            print(f"  shape: {df.shape}")
            print("  columns:", list(df.columns))
            print(df.head())
        except Exception as e:
            print(f"  ⚠️ 読み込みエラー: {e}")
    else:
        print("❌ ファイルが存在しません")