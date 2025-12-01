# パラメータチューニング戦略ナレッジ

**Date**: 2025-11-29  
**Status**: Living Document (随時更新)  
**Purpose**: DINOv2/SAM2 パイプラインのパラメータ最適化に関する知見を蓄積  
**Latest Update**: モデル嗜好分析（v3/v4/v19 比較）を統合

---

### 📚 このドキュメントについて

**対象読者**: パラメータチューニングを実践する人  
**主な内容**:
- パラメータの重要度ランキング
- グリッドサーチの方法論
- チューニング実践戦略

**関連ドキュメント**:
| ドキュメント | 用途 | 対象者 |
|:---|:---|:---|
| **04_knowledge.md** | 理論・背景知識 | 研究者・分析者 |
| **05_hints.md** | 実装チェックリスト | エンジニア・実装者 |
| **11_parameter_tuning_knowledge.md** (ここ) | 実戦チューニング戦略 | チューニング担当者 |

---

## 📋 Executive Summary

EXP005T (DINOv2+SAM2) のパラメータチューニング経験から得られた知見をまとめました。  
**限られた GPU リソース下で、最大効率のチューニング戦略**を提供します。

### Key Principles
1. **Sequential Tuning**: パラメータを順次チューニング (並列ではなく)
2. **Importance Weighting**: 重要度の高いパラメータから優先
3. **Lazy Tuning**: アンサンブル・新モデル統合後に細部調整
4. **Grid Search**: 0.01 単位での細かい探索
5. **⭐ NEW: Model Preference Aware Tuning**: モデル構成に応じたパラメータ嗜好の違いを認識

---

## 🎯 モデル嗜好分析：DINOv2 単体 vs DINOv2 + SAM2（最新知見）

**関連ドキュメント**:
- 理論的背景 → [`docs/04_knowledge.md#dinov2-単体-vs-dinov2--sam2-のパラメータ嗜好分析`](04_knowledge.md#dinov2-単体-vs-dinov2--sam2-のパラメータ嗜好分析)
- 実装チェックリスト → [`docs/05_hints.md#モデル嗜好に基づく実装ガイドライン`](05_hints.md#モデル嗜好に基づく実装ガイドライン)

### 発見：パラメータの「嗜好」がモデル構成に依存する

```
【重要な発見】EXP005T v3/v4 実験より

v19 (DINOv2 単体推定):
  ✅ ALPHA_GRAD = 0.35 (aggressive)
  ✅ Forged F1 = 0.1565
  ✅ 検出力重視

v3 (DINOv2 + SAM2):
  ✅ ALPHA_GRAD = 0.30 (conservative)
  ✅ Forged F1 = 0.1169
  ✅ 精度重視

v4 (DINOv2 + SAM2, aggressive試行):
  ❌ ALPHA_GRAD = 0.35 (v19 と同じ)
  ❌ Forged F1 = 0.1035 (悪化: -11.5%)
  ❌ パラメータが「嗜好と不一致」
```

### 根本メカニズム

#### **パイプライン設計の最適化原理**

```
単段階モデル（v19 推定）:
  DINOv2 → 最終出力
  
  特性: 「広さ」を優先
  → aggressive（多くの候補検出）が最適
  → ALPHA_GRAD = 0.35

多段階パイプライン（v3）:
  DINOv2 (検出) → SAM2 (精密化) → 出力
  
  Stage 1 (DINOv2): 「狭く、正確に」
  → conservative（ノイズ少ない）が最適
  → ALPHA_GRAD = 0.30
  
  理由: SAM2 の入力は false positive が少ない必要
       ノイズが多い → SAM2 がノイズも refine → 精度低下
```

#### **v3 vs v4 の具体的な違い**

```python
v3（最適）:
  DINOv2(conservative=0.30)
    ↓ 「確実な改ざん」のみ出力（FP 少ない）
  SAM2 refinement
    ↓ 信頼できる入力のみ処理
  結果: Forged F1 = 0.1169, FP = 2.52% ✅

v4（非最適）:
  DINOv2(aggressive=0.35)
    ↓ 「可能性がある改ざん」を積極出力（FP 多い ← ノイズ含む）
  SAM2 refinement
    ↓ ノイズも refine される
    → ノイズが「改ざんに見える」領域が生成
  結果: Forged F1 = 0.1035, FP = 2.52% ❌
        ※ FP が同じなのは、ノイズが forged 判定されているから
```

### チューニング戦略への応用

#### **パラメータ推奨値（モデル構成別）**

| モデル構成 | ALPHA_GRAD | Threshold Coef | Min Area | 根拠 |
|:---|:---:|:---:|:---:|:---|
| **DINOv2 単体** | 0.35 | 0.30 | 400 | 検出力最大化 |
| **DINOv2 + SAM2** | 0.30 | 0.25 | 400 | SAM2 入力品質最優先 |
| **DINOv2 + 他 Refinement** | 0.28～0.32 | 0.20～0.25 | 300～400 | refinement 方式に応じ微調整 |
| **Ensemble** | 複数試行 | 複数試行 | 複数試行 | 各モデル個別最適化 |

#### **パラメータ調整の順序**

```
Step 1: モデル構成を確認
        ├─ 単体か多段階か
        ├─ refinement stage があるか
        └─ 決定: 基準 ALPHA_GRAD を選択

Step 2: Validate でベースラインを記録
        ├─ forged_nonempty_ratio
        ├─ authentic_fp_ratio
        └─ Forged F1（分布も記録）

Step 3: ALPHA_GRAD を ±0.02～0.05 の範囲で試行
        ├─ 3～5 パターンで validate
        ├─ 分離指標が改善する方向を確認
        └─ 最適値を決定

Step 4: Threshold Coef, Min Area を細調整
        ├─ ALPHA_GRAD を固定
        ├─ 他パラメータでグリッド探索
        └─ FP 削減とRecall のバランス最適化

Step 5: Ensemble 設計
        ├─ 嗜好が異なるモデルを組み合わせ
        └─ 相補的な強みを活用（→ 次セクション）
```

---

## パラメータ嗜好に基づく Ensemble 戦略

### 「相補的な嗜好」を持つモデルの組み合わせ

```python
# ✅ 推奨: 嗜好が異なるモデルを融合

aggressive_model = v19                    # Forged F1 = 0.1565 (高)
conservative_model = v3                   # Forged F1 = 0.1169 (低いが FP 少ない)

pred_ensemble = 0.6 * aggressive + 0.4 * conservative
# 期待値: Forged F1 向上 + Authentic 精度保持 → LB 0.32+

# ❌ 非推奨: 同じ嗜好を持つモデル

model_1 = v19(alpha_grad=0.35)
model_2 = v19_alt(alpha_grad=0.36)
# 両方 aggressive → 相補性がない → 効果薄い
```

### Weight 決定の考え方

```python
# モデルの「強み」に応じた weight 配分

if aggressive.forged_f1 > conservative.forged_f1:
    # aggressive が forged 検出に優れている
    weight_aggressive = 0.6
    weight_conservative = 0.4  # FP 削減用
    
# 期待効果:
#   - aggressive から「検出力」を借りる
#   - conservative から「精度」を借りる
#   - FP を conservative で削減
```

---



---

## 🎯 対象パラメータ一覧

### Post-processing パラメータ群

```python
# FILE: train.py / Config class
# これらのパラメータが最終的な検出性能を左右

ALPHA_GRAD = 0.30              # [1] 勾配強度 (最重要!)
THRESHOLD_COEF = 0.30          # [2] 閾値係数 (重要)
AREA_THRESHOLD = 400           # [3] 最小領域面積
MEAN_THRESHOLD = 0.30          # [4] 平均値閾値
KERNEL_CLOSE = 5               # [5] 形態処理: 閉じる
KERNEL_OPEN = 3                # [6] 形態処理: 開く
```

### 各パラメータの物理的意味

| パラメータ | 役割 | 影響度 | 感度 |
|----------|------|--------|------|
| **ALPHA_GRAD** | 勾配情報の重み | **最大** | **高** |
| **THRESHOLD_COEF** | 検出/非検出の分岐点 | **大** | **高** |
| **AREA_THRESHOLD** | 小領域フィルタリング | 中 | 中 |
| **MEAN_THRESHOLD** | 領域内平均値フィルタ | 小-中 | 低 |
| **KERNEL_CLOSE** | ノイズ削除 (閉じる) | 小 | 低 |
| **KERNEL_OPEN** | 穴埋め (開く) | 小 | 低 |

---

## 📊 実験履歴と知見

### 2025-11-29 実験結果 (最新)

#### v19 vs v3 vs v4 比較

```
┌─────────────────────┬────────┬────────┬────────┐
│ Metric              │  v19   │  v3    │  v4    │
├─────────────────────┼────────┼────────┼────────┤
│ Macro F1            │ 0.5614 │ 0.5681 │ 0.5633 │
│ Mean F1 (Forged)    │ 0.1565 │ 0.1614 │ 0.1519 │
│ F1 (Authentic)      │ 0.9664 │ 0.9748 │ 0.9748 │
│ Forged Detect %     │ 28.26% │ 28.99% │ 27.54% │
│ FP Ratio %          │  3.36% │  2.52% │  2.52% │
│ Alpha Grad          │ 0.30   │ 0.30   │ 0.35   │
│ SAM2 Integration    │   ❌   │   ✅   │   ✅   │
├─────────────────────┼────────┼────────┼────────┤
│ 推奨度              │  参考  │ ⭐⭐⭐ │ 学習版 │
└─────────────────────┴────────┴────────┴────────┘
```

---

### 🔑 発見① - SAM2 統合はパラメータ最適値を変える

**v19 (DINOv2 only) vs v3 (DINOv2+SAM2)**

```
DINOv2 only (v19):
  最適 alpha_grad = 0.35 (Prob-Avg TTA 最適値)
  Macro F1 = 0.5614

DINOv2+SAM2 (v3):
  最適 alpha_grad = 0.30 ← 変更!
  Macro F1 = 0.5681 (↑ +0.67%)

理由:
  SAM2 の精密化処理が、勾配情報を補完
  → 追加の勾配強調 (alpha_grad UP) は過度
  → むしろ検出漏れを増加させる
```

**重要な示唆**:
> **新しいモジュール統合時は、既存パラメータを re-tune する必要がある**

---

### 🔑 発見② - Alpha_grad は高感度パラメータ

**0.05 単位の変化が ~0.5% のスコア変動を起こす**

```
v3 (alpha=0.30): Macro F1 = 0.5681
v4 (alpha=0.35): Macro F1 = 0.5633
             Δ = -0.48% (わずか +0.05 で)

推定:
v5 (alpha=0.28): ~0.565-0.567 (微減)
v6 (alpha=0.29): ~0.567-0.569 (微増?)
v7 (alpha=0.31): ~0.567-0.569 (微増?)
v8 (alpha=0.32): ~0.563-0.565 (微減)
```

**結論**:
> **0.05 単位は粗すぎる。0.01 単位での Grid Search が必須**

---

### 🔑 発見③ - Authentic 認識は alpha_grad 非依存

**FP 率が不変**

```
v3 (alpha=0.30): FP Ratio = 2.52%, F1(Authentic) = 97.48%
v4 (alpha=0.35): FP Ratio = 2.52%, F1(Authentic) = 97.48%
                         ↑ 完全不変

理由:
  Authentic 画像への誤検出削減は、SAM2 の精密化が主要因
  alpha_grad の変化は、Authentic 判定ロジックに影響しない
```

**推奨**:
> **Authentic 認識精度を保ちながら、alpha_grad で Forged Recall を最適化できる**

---

## 🧠 チューニング戦略

### 戦略① - Sequential Tuning (推奨)

**1つずつ最適化する手法**

```
Phase 1: ALPHA_GRAD
├─ 固定: threshold_coef=0.30, area=400, mean=0.30, kc=5, ko=3
├─ チューニング対象: alpha_grad ∈ [0.28, 0.35]
├─ グリッド: [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
├─ 推定時間: 8 hours (× 8 パターン)
└─ 出力: Best_alpha_grad (例: 0.31)

Phase 2: THRESHOLD_COEF
├─ 固定: alpha_grad=Best (例: 0.31), area=400, mean=0.30, kc=5, ko=3
├─ チューニング対象: threshold_coef ∈ [0.20, 0.40]
├─ グリッド: [0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38, 0.40]
├─ 推定時間: 10 hours
└─ 出力: Best_threshold_coef

Phase 3: AREA_THRESHOLD
├─ 固定: alpha_grad=Best, threshold_coef=Best, mean=0.30, kc=5, ko=3
├─ チューニング対象: area ∈ [300, 500]
├─ グリッド: [300, 350, 400, 450, 500]
├─ 推定時間: 10 hours
└─ 出力: Best_area_threshold

Phase 4-6: 残り3パラメータ
├─ Mean_threshold, kernel_close, kernel_open
├─ 各 Phase で 4-5 パターン
└─ 総時間: 24 hours
```

**総リソース**: ~60 hours (exhaustive の ~20%)

---

### 戦略② - Lazy Tuning (最も推奨)

**強い基盤ができてから細部調整**

```
NOW (2025-11-29):
  ├─ ✅ train_cnn.py (EfficientNet) 実行開始
  ├─ ✅ v3 (alpha=0.30) ベースライン確定
  └─ → CNN 学習中 (~1-2 hours)

LATER (2-3日後):
  ├─ ✅ v3 + EfficientNet アンサンブル (baseline)
  ├─ ✅ LB スコア確認 (期待: 0.330+)
  └─ → CNN 完了, baseline 評価中

EVEN LATER (1週間後):
  ├─ Baseline スコアが 0.330 なら:
  │   └─ 満足, チューニング不要
  │
  └─ Baseline スコアが 0.325-0.329 なら:
      ├─ Phase 1 (alpha_grad チューニング) 開始
      ├─ v5-v8 を実験
      └─ 最適版を特定

FINAL (2週間後):
  ├─ 3 種の候補版を用意:
  │  ├─ v3 (baseline)
  │  ├─ v3_optimal (チューニング後)
  │  └─ v3_optimal + EfficientNet (final ensemble)
  └─ → LB に全て投入, 最高スコア選択
```

---

## 📈 パラメータグリッド設計

### Phase 1: ALPHA_GRAD (優先度1)

**最も重要で、最も敏感**

```python
# グリッド候補
alpha_grad_values = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]

# v3 がベースライン (0.30)
# 周辺をサンプリング: 0.29, 0.31, 0.32
# 確認: 0.28 (下限), 0.33-0.35 (上限)

実験ID:
  v5: alpha=0.28
  v6: alpha=0.29 ← 期待値: v3 に近い
  v3: alpha=0.30 ← ベースライン
  v7: alpha=0.31 ← 期待値: v3 を上回る可能性
  v8: alpha=0.32
  v9: alpha=0.33
  v10: alpha=0.34
  v4: alpha=0.35 ← 参考 (既実験, 低)
```

**想定スコア分布**:
```
Macro F1
0.570 |     v8 ○
0.568 |   v6 ○   v7 ○   v9 ○
0.566 |   
0.564 |   v3 ○   
0.562 | v5 ○         v10 ○
0.560 |                   v4 ○
      +─────────────────────────
        0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35
                         alpha_grad
```

**選択基準**:
- max(v5-v10) を選択
- v3 と同程度なら、v3 を継続 (保守的)

---

### Phase 2: THRESHOLD_COEF (優先度2)

**alpha_grad を固定後に実施**

```python
# グリッド候補 (現在: 0.30)
threshold_coef_values = [0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38, 0.40]

# 物理的意味:
#   低 (0.20): 敏感に反応, Recall ↑ (FP ↑)
#   高 (0.40): 保守的, Precision ↑ (検出漏れ ↑)

実験ID:
  v11: coef=0.20
  v12: coef=0.23
  v13: coef=0.26
  v14: coef=0.29
  v3*: coef=0.30 ← ベースライン (必ず含める)
  v15: coef=0.32
  v16: coef=0.35
  v17: coef=0.38
  v18: coef=0.40
```

---

### Phase 3: AREA_THRESHOLD (優先度3)

**小さい改ざんの扱いを最適化**

```python
# グリッド候補 (現在: 400)
area_threshold_values = [300, 350, 400, 450, 500]

# 物理的意味:
#   小 (300): 小さい改ざんも検出 (FP の可能性 ↑)
#   大 (500): 大きい改ざんのみ (漏れ増加)

実験ID:
  v19: area=300
  v20: area=350
  v3*: area=400 ← ベースライン
  v21: area=450
  v22: area=500
```

---

## 🛠️ 実装チェックリスト

### チューニング実施時のステップ

```python
# 1. パラメータを train.py に追加
class Config:
    ALPHA_GRAD = 0.30  # CLI arg: --alpha-grad で上書き可能
    THRESHOLD_COEF = 0.30
    # ...

# 2. CLI で上書き
if args.alpha_grad:
    CONFIG.ALPHA_GRAD = args.alpha_grad

# 3. 実験実行
# python train.py --skip-train --alpha-grad 0.31

# 4. 結果を保存
# experiments/EXP005T/v7_score0.XXX/
```

### メトリクス記録

```python
# experiments/EXP005T/v{N}_score0.{XXX}/overall_metrics.json
{
    "experiment_id": "EXP005T_v7",
    "version": "v7_alpha_0.31",
    "alpha_grad": 0.31,
    "threshold_coef": 0.30,
    "area_threshold": 400,
    "macro_f1": 0.5685,
    "mean_f1_forged": 0.1630,
    "f1_authentic": 0.9748,
    "created_at": "2025-11-30T12:34:56"
}
```

---

## 📋 チューニング進捗トラッキング

### 進捗テーブルテンプレート

```markdown
## Phase 1: ALPHA_GRAD チューニング進捗

| ID | Alpha | Macro F1 | Mean F1 | Status | Notes |
|----|-------|----------|---------|--------|-------|
| v3 | 0.30 | 0.5681 | 0.1614 | ✅ Done | Baseline |
| v5 | 0.28 | TBD | TBD | ⏳ Pending | 下限探索 |
| v6 | 0.29 | TBD | TBD | ⏳ Pending | |
| v7 | 0.31 | TBD | TBD | ⏳ Pending | 期待値高 |
| v8 | 0.32 | TBD | TBD | ⏳ Pending | |

**Progress**: 1/5 (20%)
**ETA**: 8 hours
```

---

## 💡 チューニング時の注意点

### 1. Authentic 認識精度は監視対象

```python
# FP Ratio が 3% 以上に悪化したら即座に中止
if fp_ratio > 0.03:
    logger.warning("⚠️ FP Ratio が上昇! チューニング中止検討")
    # 原因: threshold_coef が低すぎる可能性
```

### 2. Macro F1 vs Mean F1 のトレードオフ

```
通常: Macro F1 を指標とする
      (Forged/Authentic の両方をバランスよく)

ただし:
  - Forged 検出率を優先したい → Mean F1 重視
  - Authentic 精度を優先したい → F1(Authentic) 重視
```

### 3. パラメータ相互作用に注意

```
Phase 分離の理由:
  各パラメータが独立的に影響すると仮定
  → 順次チューニングで十分

ただし実際には相互作用の可能性:
  例: alpha_grad と threshold_coef の相互作用
  → 最終版では combined grid も検討
```

---

## 🚀 推奨チューニングスケジュール

### タイムライン (ポンポさんのリソース制約下)

```
2025-11-29 (NOW):
  ├─ train_cnn.py 実行開始
  ├─ v3 ベースラインで LB 投入準備
  └─ 予想: 0.314-0.320 (validation score)

2025-11-30~12-01:
  ├─ CNN 学習完了 (~1-2 hours)
  ├─ v3 + EfficientNet アンサンブル
  ├─ LB スコア確認 (期待: 0.330+)
  └─ 余力があれば Phase 1 開始

2025-12-02~12-05:
  ├─ Phase 1: alpha_grad チューニング (8 hours)
  ├─ 勝者: Best_alpha (例: v7 with 0.31)
  └─ LB 投入予定

2025-12-06~12-08:
  ├─ Phase 2: threshold_coef (10 hours)
  ├─ 勝者: Best_threshold
  └─ 余裕があれば Phase 3 へ

2025-12-09~:
  ├─ 最終調整 + 複合グリッド検索
  └─ 0.335+ を目指す
```

---

## 📚 参考資料

### 関連ドキュメント
- `VISUAL_ANALYSIS_REPORT_v19.md` - v19 詳細分析
- `VISUAL_ANALYSIS_REPORT_v3.md` - v3 詳細分析  
- `VISUAL_ANALYSIS_REPORT_v4.md` - v4 詳細分析

### 実験ログ
- `experiments/EXP005T/v{N}_score{X.XXX}/overall_metrics.json`

### コード参照
- `train.py` - Config class (パラメータ定義)
- `train.py` - enhanced_adaptive_mask() - パラメータ使用箇所

---

## 📝 更新履歴

| Date | Content | Author |
|------|---------|--------|
| 2025-11-29 | 初版作成; v3/v4 比較分析を基に知見記録 | つむぎ |

---

## 🎯 次のアクション

- [ ] train_cnn.py 実行開始
- [ ] v3 + EfficientNet アンサンブル版を作成
- [ ] LB スコア確認
- [ ] スコアに応じて Phase 1 チューニング開始
- [ ] チューニング完了後, このドキュメントを更新
