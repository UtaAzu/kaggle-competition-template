import pandas as pd
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

"""
validate.py — Generic artifacts validator (Kaggle / Local compatible)

Usage:
    python validate.py --artifacts-dir <PATH>  # optional; defaults to scanning local experiments or /kaggle/input

This script inspects experiment artifact directories and summarizes metrics/OOF results. It is
general-purpose; the output schema is expected to contain 'oof.csv' or 'oof_all.csv' and a metrics JSON.
This script does not include competition-specific default paths; use `--artifacts-dir` or env vars, and
refer to `examples/archive/` for archived, competition-specific examples.
"""

# ===== 1. データ読み込み =====
KAGGLE_INPUT = Path('/kaggle/input')

def _resolve_artifact_dirs(artifacts_dir_arg: str = None):
    """Resolve artifact directories in a flexible way:
    - If user passed `--artifacts-dir`, use it (supports glob/dir)
    - If running on Kaggle and /kaggle/input exists, scan for any `*-artifacts` directories under /kaggle/input/*
    - Otherwise scan `experiments/**/*-artifacts` in repository
    """
    if artifacts_dir_arg:
        p = Path(artifacts_dir_arg)
        if p.is_dir():
            return sorted([p])
        else:
            # Support glob pattern
            return sorted([d for d in Path('.').glob(artifacts_dir_arg) if d.is_dir()])

    # Kaggle auto-detect
    if KAGGLE_INPUT.exists():
        artifact_candidates = []
        for d in KAGGLE_INPUT.glob('**/*-artifacts'):
            if d.is_dir():
                artifact_candidates.append(d)
        return sorted(artifact_candidates)
    # ローカル: リポ内を探索（深さ2まで）
    root = Path('.')
    artifact_dirs = []
    for pattern in ['experiments/*/*-artifacts', 'experiments/*/*/*-artifacts', 'experiments/*-artifacts']:
        artifact_dirs.extend([d for d in root.glob(pattern) if d.is_dir()])
    return sorted(set(artifact_dirs))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate experiment artifacts and summarize metrics (generic template)')
    parser.add_argument('--artifacts-dir', type=str, default=None, help='Path to artifact directory or glob; if not provided, auto-discover')
    args = parser.parse_args()

    artifact_dirs = _resolve_artifact_dirs(args.artifacts_dir)
    results = []

for artifact_dir in artifact_dirs:
    exp_id = artifact_dir.name.replace('-artifacts', '')
    
    # 2T系/3T系どちらにも対応
    oof_path = artifact_dir / 'oof.csv'
    metrics_path = artifact_dir / 'metrics.json'
    if not oof_path.exists():
        oof_path = artifact_dir / 'oof_all.csv'
    if not metrics_path.exists():
        metrics_path = artifact_dir / 'overall_metrics.json'
    run_path = artifact_dir / 'run.json'

    if not oof_path.exists() or not metrics_path.exists():
        print(f"⚠️  Skipping {exp_id}: missing oof.csv(oof_all.csv) or metrics.json(overall_metrics.json)")
        continue

    oof_df = pd.read_csv(oof_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    run = None
    if run_path.exists():
        with open(run_path) as f:
            run = json.load(f)

    # forged_nonempty_ratio, authentic_fp_ratioを取得（metricsから）
    forged_nonempty_ratio = (
        metrics.get('forged_nonempty_ratio') or
        metrics.get('forged_nonempty_ratio_mean')
    )
    authentic_fp_ratio = (
        metrics.get('authentic_fp_ratio') or
        metrics.get('authentic_nonempty_ratio') or
        metrics.get('authentic_fp_ratio_mean')
    )

    # ====== v0/v1 スキーマ互換処理 ======
    metric_version = metrics.get('metric_version', 'v0')
    # v1: 明示フィールドが揃っている前提
    mean_f1_forged = metrics.get('mean_f1_forged')
    f1_authentic = metrics.get('f1_authentic')
    macro_f1 = metrics.get('macro_f1') or metrics.get('macro_f1_mean')
    # v0: macro_f1 は forged-only の意味だった可能性 → 名寄せ
    if metric_version == 'v0':
        if mean_f1_forged is None:
            mean_f1_forged = metrics.get('macro_f1') or metrics.get('overall_f1')
        if f1_authentic is None and authentic_fp_ratio is not None:
            f1_authentic = 1 - authentic_fp_ratio
        if macro_f1 is None and (mean_f1_forged is not None and f1_authentic is not None):
            macro_f1 = (mean_f1_forged + f1_authentic) / 2

    results.append({
        'exp_id': exp_id,
        'metric_version': metric_version,
        'mean_f1': macro_f1 if macro_f1 is not None else (mean_f1_forged or 0.0),
        'mean_f1_forged': mean_f1_forged,
        'f1_authentic': f1_authentic,
        'macro_f1': macro_f1,
        'best_threshold': metrics.get('best_threshold'),
        'n_samples': metrics.get('n_samples', len(oof_df)),
        'oof_df': oof_df,
        'metrics': metrics,
        'run': run,
        'forged_nonempty_ratio': forged_nonempty_ratio,
        'authentic_fp_ratio': authentic_fp_ratio
    })

# ===== 2. 詳細サマリー（authentic/forgedの分離評価＋追加分析） =====
detailed_summary = []
histogram_bins = np.linspace(0, 1, 21)  # 20 bins
for r in results:
    oof_df = r['oof_df']
    # oof.csvが無い場合はmetrics値のみで埋める（ここでは常に存在するが念のため）
    if oof_df is None or oof_df.empty:
        detailed_summary.append({
            'exp_id': r['exp_id'],
            'mean_f1': r['mean_f1'],
            'best_threshold': r['best_threshold'],
            'n_samples': r['n_samples'],
            'n_authentic': None,
            'n_forged': None,
            'f1_mean': r['mean_f1'],
            'f1_median': None,
            'f1_std': None,
            'f1_q25': None,
            'f1_q75': None,
            'authentic_f1_mean': r['metrics'].get('authentic_f1_mean'),
            'forged_f1_mean': r['metrics'].get('forged_f1_mean'),
            'zero_f1_ratio': None,
            'perfect_f1_ratio': None,
            'forged_nonempty_ratio': r.get('forged_nonempty_ratio'),
            'authentic_fp_ratio': r.get('authentic_fp_ratio'),
            'forged_zero_f1_ratio': None,
            'authentic_perfect_f1_ratio': None,
            'f1_histogram': None
        })
        continue
    authentic = oof_df[oof_df['is_forged'] == 0] if 'is_forged' in oof_df.columns else pd.DataFrame()
    forged = oof_df[oof_df['is_forged'] == 1] if 'is_forged' in oof_df.columns else pd.DataFrame()
    
    # F1分布ヒストグラム（数値化・非ゼロのみ）
    # best_f1カラムがない場合は pred_nonempty から推定
    if 'best_f1' in oof_df.columns:
        f1_values = oof_df['best_f1']
    elif 'pred_nonempty' in oof_df.columns:
        print(f"⚠️  {r['exp_id']}: best_f1カラムなし → pred_nonemptyから推定")
        f1_values = oof_df['pred_nonempty'].astype(float)
    else:
        print(f"⚠️  {r['exp_id']}: best_f1, pred_nonempty両方なし → スキップ")
        f1_values = pd.Series([0.0] * len(oof_df))
    
    hist, bin_edges = np.histogram(f1_values, bins=histogram_bins)
    hist_str = ", ".join([f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}:{hist[i]}" 
                          for i in range(len(hist)) if hist[i] > 0])
    
    # ✅ forged画像の非空RLE提出率（偽造検出率）
    if len(forged) > 0:
        if 'pred_nonempty' in forged.columns:
            forged_nonempty_ratio = forged['pred_nonempty'].mean()
        else:
            if 'best_f1' in forged.columns:
                forged_nonempty_ratio = (forged['best_f1'] > 0).mean()
            else:
                forged_nonempty_ratio = None
                print(f"⚠️  {r['exp_id']}: pred_nonemptyカラムなし → F1>0で代替計算")
    else:
        forged_nonempty_ratio = None
    
    # ✅ authentic画像の非空RLE提出率（誤検出率）
    if len(authentic) > 0:
        if 'pred_nonempty' in authentic.columns:
            authentic_fp_ratio = authentic['pred_nonempty'].mean()
        else:
            if 'best_f1' in authentic.columns:
                authentic_fp_ratio = (authentic['best_f1'] > 0).mean()
            else:
                authentic_fp_ratio = None
    else:
        authentic_fp_ratio = None
    
    # F1分位点（best_f1またはpred_nonemptyカラムがある場合）
    f1_col_for_quantile = None
    if 'best_f1' in oof_df.columns:
        f1_col_for_quantile = 'best_f1'
    elif 'pred_nonempty' in oof_df.columns:
        f1_col_for_quantile = 'pred_nonempty'
    
    if f1_col_for_quantile is not None:
        f1_q25 = oof_df[f1_col_for_quantile].quantile(0.25)
        f1_q75 = oof_df[f1_col_for_quantile].quantile(0.75)
        # forged画像のF1=0率
        forged_zero_f1_ratio = (forged[f1_col_for_quantile] == 0).mean() if len(forged) > 0 else None
        # authentic画像のF1=1率
        authentic_perfect_f1_ratio = (authentic[f1_col_for_quantile] == 1).mean() if len(authentic) > 0 else None
    else:
        f1_q25 = None
        f1_q75 = None
        forged_zero_f1_ratio = None
        authentic_perfect_f1_ratio = None

    # ⚠️ best_f1カラムがない場合の代替処理
    if 'best_f1' in oof_df.columns:
        f1_col = 'best_f1'
    elif 'pred_nonempty' in oof_df.columns:
        f1_col = 'pred_nonempty'
    else:
        f1_col = None
        print(f"⚠️  {r['exp_id']}: best_f1, pred_nonempty両方なし → スキップ")
    
    if f1_col is not None:
        detailed_summary.append({
            'exp_id': r['exp_id'],
            'mean_f1': r['macro_f1'] if r.get('macro_f1') is not None else r['mean_f1'],
            'best_threshold': r['best_threshold'],
            'n_samples': r['n_samples'],
            'n_authentic': len(authentic),
            'n_forged': len(forged),
            'f1_mean': oof_df[f1_col].mean(),
            'f1_median': oof_df[f1_col].median(),
            'f1_std': oof_df[f1_col].std(),
            'f1_q25': f1_q25,
            'f1_q75': f1_q75,
            'authentic_f1_mean': r.get('f1_authentic') if r.get('f1_authentic') is not None else (authentic[f1_col].mean() if len(authentic) > 0 else None),
            'forged_f1_mean': r.get('mean_f1_forged') if r.get('mean_f1_forged') is not None else (forged[f1_col].mean() if len(forged) > 0 else None),
            'zero_f1_ratio': (oof_df[f1_col] == 0).sum() / len(oof_df),
            'perfect_f1_ratio': (oof_df[f1_col] == 1).sum() / len(oof_df),
            'forged_nonempty_ratio': forged_nonempty_ratio,
            'authentic_fp_ratio': authentic_fp_ratio,
            'forged_zero_f1_ratio': forged_zero_f1_ratio,
            'authentic_perfect_f1_ratio': authentic_perfect_f1_ratio,
            'f1_histogram': hist_str
        })

summary_df = pd.DataFrame(detailed_summary)
print("===== 📊 実験比較サマリー（詳細分析付き） =====")
print(summary_df.to_string(index=False))

# ===== 3. 異常検知: 完全一致実験を特定 =====
print("\n===== 🚨 異常検知: F1分布が完全一致する実験 =====")
grouped = summary_df.groupby(['f1_mean', 'f1_std', 'zero_f1_ratio']).agg(list)
duplicates = grouped[grouped['exp_id'].apply(len) > 1]
if len(duplicates) > 0:
    print("以下の実験グループは検証ロジックが実質的に同じ可能性:")
    for idx, row in duplicates.iterrows():
        print(f"  - {', '.join(row['exp_id'])}: mean={idx[0]:.4f}, std={idx[1]:.4f}")
else:
    print("✅ 全実験が異なる検証ロジックを使用")

# ===== 4. スコア改善トレンド =====
print("\n===== 📈 スコア改善トレンド =====")
summary_df_sorted = summary_df.sort_values('exp_id')
print(summary_df_sorted[['exp_id', 'mean_f1', 'f1_median', 'f1_q25', 'f1_q75', 'zero_f1_ratio', 'perfect_f1_ratio', 'forged_nonempty_ratio', 'authentic_fp_ratio']].to_string(index=False))

# ===== 7. 健全性判定 =====
print("\n===== 🏥 健全性判定（Sanity Check） =====")
for _, row in summary_df.iterrows():
    exp_id = row['exp_id']
    forged_nonempty = row['forged_nonempty_ratio']
    authentic_fp = row['authentic_fp_ratio']
    
    if pd.isna(forged_nonempty) or pd.isna(authentic_fp):
        print(f"⚠️  {exp_id}: 分離指標が計算不可")
        continue
    
    status = "✅"
    messages = []
    if forged_nonempty < 0.2:
        status = "❌"
        messages.append(f"forged_nonempty_ratio={forged_nonempty:.3f} < 0.2 → 退化解")
    if authentic_fp > 0.3:
        status = "❌"
        messages.append(f"authentic_fp_ratio={authentic_fp:.3f} > 0.3 → 誤検出過多")
    
    if messages:
        print(f"{status} {exp_id}: {', '.join(messages)}")
    else:
        print(f"{status} {exp_id}: 分離指標健全（forged={forged_nonempty:.3f}, auth_fp={authentic_fp:.3f}）")

# ===== 8. スコアランキング（分離指標考慮） =====
print("\n===== 🏆 スコアランキング（分離指標考慮） =====")
summary_df['健全性スコア'] = (
    summary_df['forged_nonempty_ratio'].fillna(0) * 0.5 +
    (1 - summary_df['authentic_fp_ratio'].fillna(1)) * 0.3 +
    summary_df['mean_f1'].fillna(0) * 0.2
)
top5 = summary_df.nlargest(5, '健全性スコア')[['exp_id', 'mean_f1', 'forged_nonempty_ratio', 'authentic_fp_ratio', '健全性スコア']]
print(top5.to_string(index=False))

# ===== 9. 分析結果のエクスポート =====
out_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('experiments/validate_outputs')
out_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(out_dir / 'validate_summary.csv', index=False)
print(f"\n✅ 分析結果を {out_dir / 'validate_summary.csv'} に保存")

export_data = {
    'summary': summary_df.to_dict(orient='records'),
    'sanity_check': {
        'degenerate_exps': summary_df[summary_df['forged_nonempty_ratio'] < 0.2]['exp_id'].tolist(),
        'high_fp_exps': summary_df[summary_df['authentic_fp_ratio'] > 0.3]['exp_id'].tolist()
    }
}
with open(out_dir / 'validate_report.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# ===== 10. F1分布の可視化 =====
N = min(6, len(results))
top_results = sorted(results, key=lambda r: (r.get('macro_f1') if r.get('macro_f1') is not None else r.get('mean_f1') or 0), reverse=True)[:N]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, r in enumerate(top_results):
    oof_df = r['oof_df']
    # best_f1カラムがない場合はスキップ
    if 'best_f1' not in oof_df.columns:
        continue
    axes[i].hist(oof_df['best_f1'], bins=histogram_bins, alpha=0.7, edgecolor='black', color='skyblue')
    title_score = r.get('macro_f1') if r.get('macro_f1') is not None else r.get('mean_f1', 0)
    axes[i].set_title(f"{r['exp_id']}\nmacro_f1={title_score:.3f}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel('F1 Score', fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].axvline(oof_df['best_f1'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[i].legend()

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
dist_path = out_dir / 'oof_f1_distributions.png'
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可視化を {dist_path} に保存")

# ===== 6. 推奨アクション =====
print("\n===== 💡 推奨アクション =====")
if (summary_df['zero_f1_ratio'] > 0.4).any():
    print("⚠️  F1=0が40%以上の実験あり → モデル学習・後処理ロジックを再確認")
if len(duplicates) > 0:
    print("⚠️  完全一致する実験あり → 検証ロジックの差分を確認")
print("✅ forged_nonempty_ratio, authentic_fp_ratio, F1分布ヒストグラムを確認し、extreme値の原因を調査")