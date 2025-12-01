#!/usr/bin/env python3
"""
EDA Runner - 探索的データ分析のためのCLIスクリプト

使用法:
    python tools/eda_runner.py
    python tools/eda_runner.py --train-path data/train.csv --text-col comment_text --label-col target
    EDA_TRAIN_PATH=data/train.csv python tools/eda_runner.py --out-dir output/eda
"""

import argparse
import os
import json
import warnings
import glob
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import shared helper
from utils.find_train_csv import find_train_csv

warnings.filterwarnings('ignore')

# matplotlib設定（非GUIバックエンド）
plt.switch_backend('Agg')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def detect_text_column(df):
    """テキスト列を自動検出"""
    candidates = ['text', 'comment_text', 'content', 'document', 'text_clean', 'body']
    for col in candidates:
        if col in df.columns:
            return col
    return None


def detect_label_column(df):
    """ラベル列を自動検出"""
    candidates = ['target', 'label', 'toxic', 'y', 'rule_violation']
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_data_with_encoding(file_path):
    """エンコーディングを自動検出してデータを読み込み"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✓ エンコーディング {encoding} で読み込み成功")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError("データの読み込みに失敗しました。エンコーディングを確認してください。")


def analyze_text_length(df, text_col, output_dir):
    """テキスト長分析"""
    print("📊 テキスト長分析を実行中...")
    
    # テキスト長特徴量計算
    text_data = df[text_col].fillna("").astype(str)
    df['char_len'] = text_data.str.len()
    df['token_len'] = text_data.str.split().str.len()
    
    # 統計情報
    char_stats = df['char_len'].describe().to_dict()
    token_stats = df['token_len'].describe().to_dict()
    
    print(f"  文字長統計: 平均 {char_stats['mean']:.1f}, 最大 {char_stats['max']}")
    print(f"  トークン長統計: 平均 {token_stats['mean']:.1f}, 最大 {token_stats['max']}")
    
    # 文字長ヒストグラム
    plt.figure(figsize=(12, 6))
    plt.hist(df['char_len'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('テキスト文字長分布')
    plt.xlabel('文字数')
    plt.ylabel('頻度')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_chars.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # トークン長ヒストグラム
    plt.figure(figsize=(12, 6))
    plt.hist(df['token_len'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('テキストトークン長分布')
    plt.xlabel('トークン数')
    plt.ylabel('頻度')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_tokens.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    return {
        "char_len": char_stats,
        "token_len": token_stats
    }


def analyze_label_distribution(df, label_col, output_dir):
    """ラベル分布分析"""
    print("🎯 ラベル分布分析を実行中...")
    
    label_data = df[label_col].dropna()
    
    # 数値型かカテゴリ型かを判定
    if pd.api.types.is_numeric_dtype(label_data):
        print("  ラベルタイプ: 数値型")
        stats = label_data.describe().to_dict()
        
        # ヒストグラム
        plt.figure(figsize=(12, 6))
        plt.hist(label_data, bins=30, alpha=0.7, edgecolor='black')
        plt.title('ラベル分布（ヒストグラム）')
        plt.xlabel('ラベル値')
        plt.ylabel('頻度')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/label_hist.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        return {
            "type": "numeric",
            "stats": stats
        }
        
    else:
        print("  ラベルタイプ: カテゴリ型")
        value_counts = label_data.value_counts()
        print(f"  クラス数: {len(value_counts)}")
        
        # 棒グラフ
        plt.figure(figsize=(12, 6))
        value_counts.plot(kind='bar', alpha=0.7)
        plt.title('ラベル分布（棒グラフ）')
        plt.xlabel('ラベル')
        plt.ylabel('頻度')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/label_bar.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        return {
            "type": "categorical",
            "value_counts": value_counts.to_dict()
        }


def analyze_duplicates(df, text_col, output_dir):
    """重複検出分析"""
    print("🔍 重複検出分析を実行中...")
    
    # テキストの正規化
    normalized_text = df[text_col].fillna("").astype(str).str.lower().str.strip()
    normalized_text = normalized_text.str.replace(r'\s+', ' ', regex=True)
    
    # 重複カウント
    duplicate_counts = normalized_text.value_counts()
    duplicates = duplicate_counts[duplicate_counts > 1]
    
    duplicate_total = duplicates.sum() - len(duplicates)
    print(f"  重複サンプル数: {duplicate_total}")
    print(f"  重複テキスト種類数: {len(duplicates)}")
    
    if len(duplicates) > 0:
        # CSV保存
        top_duplicates = duplicates.head(10)
        duplicates_df = pd.DataFrame({
            'text': top_duplicates.index,
            'count': top_duplicates.values
        })
        duplicates_df.to_csv(f"{output_dir}/duplicates_top.csv", index=False, encoding='utf-8')
        
        return {
            "duplicate_count": int(duplicate_total),
            "has_duplicates": True,
            "duplicate_types": len(duplicates)
        }
    else:
        return {
            "duplicate_count": 0,
            "has_duplicates": False,
            "duplicate_types": 0
        }


def analyze_groups(df, output_dir):
    """グループ/リーク検査"""
    print("👥 グループ/リーク検査を実行中...")
    
    group_candidates = ['user_id', 'author', 'identity', 'comment_id', 'thread_id', 'post_id']
    found_groups = [col for col in group_candidates if col in df.columns]
    
    groups_info = {
        "found_columns": found_groups,
        "recommendations": []
    }
    
    if found_groups:
        print(f"  検出されたグループ列: {found_groups}")
        
        for col in found_groups:
            nunique = df[col].nunique()
            n_samples = len(df)
            top_freq = df[col].value_counts().head(5)
            
            print(f"    {col}: ユニーク数 {nunique}")
            
            # GroupKFoldの推奨判定
            if nunique < n_samples * 0.5:
                groups_info["recommendations"].append(f"{col}: GroupKFold推奨 (ユニーク数: {nunique})")
                print(f"      → GroupKFold推奨")
            
            groups_info[col] = {
                "nunique": nunique,
                "top_frequencies": top_freq.to_dict()
            }
    else:
        print("  グループ列が見つかりませんでした")
        groups_info["recommendations"].append("グループ列が見つからないため、通常のStratifiedKFoldを使用")

    # groups.json保存
    with open(f"{output_dir}/groups.json", "w", encoding="utf-8") as f:
        json.dump(groups_info, f, ensure_ascii=False, indent=2)

    return groups_info


def analyze_time_leakage(df, output_dir):
    """時系列リーク検査"""
    print("⏰ 時系列リーク検査を実行中...")
    
    time_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'posted'])]
    
    if time_candidates:
        print(f"  検出された時間列: {time_candidates}")
        
        time_info = {}
        
        for col in time_candidates:
            try:
                # 日時変換を試行
                datetime_series = pd.to_datetime(df[col], errors='coerce')
                non_null_dates = datetime_series.dropna()
                
                if len(non_null_dates) > 0:
                    print(f"    {col}: 有効な日時データ {len(non_null_dates)}件")
                    
                    # 月別カウント
                    monthly_counts = non_null_dates.dt.to_period('M').value_counts().sort_index()
                    
                    # 時系列プロット
                    plt.figure(figsize=(12, 6))
                    monthly_counts.plot(kind='bar')
                    plt.title(f'時系列分布 - {col}')
                    plt.xlabel('期間')
                    plt.ylabel('データ数')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/time_counts.png", dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    time_info[col] = {
                        "valid_count": len(non_null_dates),
                        "min_date": str(non_null_dates.min()),
                        "max_date": str(non_null_dates.max()),
                        "monthly_counts": {str(k): v for k, v in monthly_counts.to_dict().items()}
                    }
                    
            except Exception as e:
                print(f"    {col}: 日時変換エラー - {e}")
        
        return time_info
    else:
        print("  時間関連列が見つかりませんでした")
        return {}


def analyze_correlations(df, label_col, output_dir):
    """数値相関分析"""
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        print("⚠️  ラベルが数値型ではないため、数値相関分析をスキップします")
        return {"message": "ラベルが数値型ではない"}
    
    print("📈 数値相関分析を実行中...")
    
    # 数値列の選択（IDっぽい列は除外）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ID列の除外
    id_keywords = ['id', 'index', 'row']
    filtered_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in id_keywords)]
    
    if len(filtered_cols) > 1:  # ラベル列以外に数値列がある
        correlation_data = df[filtered_cols].corr()
        label_corrs = correlation_data[label_col].drop(label_col).sort_values(key=abs, ascending=False)
        
        print(f"  数値特徴量数: {len(filtered_cols)-1}")
        print(f"  最高相関: {label_corrs.iloc[0]:.4f} ({label_corrs.index[0]})")
        
        # 相関CSV保存
        label_corrs.to_csv(f"{output_dir}/correlations.csv", header=['correlation'])
        
        # 相関ヒートマップ（上位20個）
        top_corr_cols = [label_col] + label_corrs.head(19).index.tolist()
        correlation_subset = correlation_data.loc[top_corr_cols, top_corr_cols]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('数値特徴量相関ヒートマップ（上位20）')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        return {
            "top_correlations": label_corrs.head(10).to_dict(),
            "numeric_features_count": len(filtered_cols)
        }
    else:
        print("  十分な数値特徴量が見つかりませんでした")
        return {"message": "十分な数値特徴量なし"}


def run_eda(train_path, text_col, label_col, output_dir):
    """EDA実行のメイン関数"""
    print("🚀 EDA分析を開始します...")
    print(f"📁 訓練データパス: {train_path}")
    print(f"📄 テキスト列: {text_col}")
    print(f"🏷️  ラベル列: {label_col}")
    print(f"💾 出力ディレクトリ: {output_dir}")
    print()
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データ読み込み
    print("📖 データを読み込み中...")
    df = load_data_with_encoding(train_path)
    
    # 列の自動検出（引数で指定されていない場合）
    if not text_col:
        text_col = detect_text_column(df)
        print(f"📝 テキスト列を自動検出: {text_col}")
    
    if not label_col:
        label_col = detect_label_column(df)
        print(f"🏷️  ラベル列を自動検出: {label_col}")
    
    # 基本情報
    print(f"📊 データ形状: {df.shape}")
    
    # 欠損値の確認
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    if len(missing_cols) > 0:
        print(f"⚠️  欠損値あり: {len(missing_cols)}列")
    
    # サマリデータ初期化
    summary_data = {
        "dataset_shape": list(df.shape),
        "columns": list(df.columns),
        "null_counts": missing_info.to_dict(),
        "detected_text_col": text_col,
        "detected_label_col": label_col
    }
    
    # 各分析を実行
    try:
        if text_col and text_col in df.columns:
            summary_data["text_length_stats"] = analyze_text_length(df, text_col, output_dir)
            summary_data["duplicates"] = analyze_duplicates(df, text_col, output_dir)
        else:
            print("⚠️  テキスト列が見つかりません。テキスト分析をスキップします。")
    except Exception as e:
        print(f"❌ テキスト分析エラー: {e}")
    
    try:
        if label_col and label_col in df.columns:
            summary_data["label_stats"] = analyze_label_distribution(df, label_col, output_dir)
            summary_data["correlations"] = analyze_correlations(df, label_col, output_dir)
        else:
            print("⚠️  ラベル列が見つかりません。ラベル分析をスキップします。")
    except Exception as e:
        print(f"❌ ラベル分析エラー: {e}")
    
    try:
        summary_data["groups"] = analyze_groups(df, output_dir)
    except Exception as e:
        print(f"❌ グループ分析エラー: {e}")
    
    try:
        summary_data["time_analysis"] = analyze_time_leakage(df, output_dir)
    except Exception as e:
        print(f"❌ 時系列分析エラー: {e}")
    
    # 最終サマリファイル保存
    print("💾 サマリファイルを保存中...")
    with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 生成されたアーティファクトの一覧
    artifacts = glob.glob(f"{output_dir}/*")
    artifacts = [os.path.basename(path) for path in artifacts]
    
    print()
    print("✅ EDA分析が完了しました！")
    print(f"📂 生成されたアーティファクト ({len(artifacts)}個):")
    for artifact in sorted(artifacts):
        print(f"   - {artifact}")
    print(f"💾 保存先: {os.path.abspath(output_dir)}")


def main():
    """CLIメイン関数"""
    parser = argparse.ArgumentParser(
        description="探索的データ分析（EDA）CLIツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python tools/eda_runner.py
  python tools/eda_runner.py --train-path data/train.csv --text-col comment_text --label-col target
  EDA_TRAIN_PATH=data/train.csv EDA_TEXT_COL=comment_text EDA_LABEL_COL=target python tools/eda_runner.py --out-dir output/eda
        """
    )
    
    parser.add_argument(
        '--train-path',
        type=str,
        help='訓練データCSVファイルのパス（未指定時は自動検出）'
    )
    parser.add_argument(
        '--text-col',
        type=str,
        help='テキスト列名（未指定時は自動検出）'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        help='ラベル列名（未指定時は自動検出）'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        help='出力ディレクトリ（未指定時は環境に応じて自動設定）'
    )
    
    args = parser.parse_args()
    
    # 環境変数からの設定取得
    train_path = args.train_path or find_train_csv()
    text_col = args.text_col or os.getenv('EDA_TEXT_COL')
    label_col = args.label_col or os.getenv('EDA_LABEL_COL')
    
    # 出力ディレクトリのデフォルト設定
    if args.out_dir:
        out_dir = args.out_dir
    elif os.getenv('EDA_OUT_DIR'):
        out_dir = os.getenv('EDA_OUT_DIR')
    elif Path('/kaggle').exists():
        out_dir = os.getenv('WORKING_DIR', '/kaggle/working') + '/eda'
    else:
        out_dir = 'experiments/EXP001G/artifacts/eda'
    
    # 出力ディレクトリを作成
    os.makedirs(out_dir, exist_ok=True)
    
    if not train_path:
        print("❌ 訓練データファイルが見つかりません。")
        print("   --train-path で指定するか、EDA_TRAIN_PATH環境変数を設定してください。")
        print("   または、以下のパスのいずれかにファイルを配置してください:")
        print("   - data/train.csv")
        print("   - data/raw/train.csv")
        print("   - input/train.csv")
        print("   - input/*/train*.csv")
        print("   - dataset/train.csv")
        print("   - sample_train.csv")
        print("   - /kaggle/input/**/train*.csv (Kaggle環境)")
        return 1
    
    if not os.path.exists(train_path):
        print(f"❌ ファイルが存在しません: {train_path}")
        return 1
    
    try:
        run_eda(train_path, text_col, label_col, out_dir)
        return 0
    except Exception as e:
        print(f"❌ EDA実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())