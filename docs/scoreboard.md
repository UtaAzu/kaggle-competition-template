# Experiment Scoreboard

## EXP001T シリーズ（v1～v9）

### EXP001T v1～v4
- Date: 2025-01-XX
- Git commit: TBD
- LB Score: **Scoring Error**
- 原因: 提出形式不備（RLE/case_id順序/空マスク未対応）
- 詳細: [`experiments/EXP001T/v5_score0.303/report_scoring-error.md`](experiments/EXP001T/v5_score0.303/report_scoring-error.md)

### EXP001T v5
- Date: 2025-01-XX
- Git commit: TBD
- CV Score: N/A（validate未実装）
- LB Score: **0.303**
- 主要変更: 提出形式修正（RLE・authentic・case_id順序）
- Notebook: [`experiments/EXP001T/v5_score0.303/recod-ai-luc-train.ipynb`](experiments/EXP001T/v5_score0.303/recod-ai-luc-train.ipynb)

### EXP001T v6
- Date: 2025-01-XX
- Git commit: TBD
- CV Score: N/A
- LB Score: **0.303**
- 主要変更: ResNet50 (pretrained=False), GroupKFold, Aug強化
- Notebook: [`experiments/EXP001T/v6_score0.303/recod-ai-luc-train.ipynb`](experiments/EXP001T/v6_score0.303/recod-ai-luc-train.ipynb)

### EXP001T v7
- Date: 2025-01-XX
- Git commit: TBD
- CV Score: 0.303（OOF mean F1, best_threshold=0.5）
- LB Score: **0.303**
- 主要変更: validate機能追加（OOF/閾値最適化/artifacts出力）
- Artifacts: [`experiments/EXP001T/v7_score0.303/exp001t-v7-artifacts/`](experiments/EXP001T/v7_score0.303/exp001t-v7-artifacts/)
- Notebook: [`experiments/EXP001T/v7_score0.303/recod-ai-luc-train.ipynb`](experiments/EXP001T/v7_score0.303/recod-ai-luc-train.ipynb)

### EXP001T v8
- Date: 2025-01-XX
- Git commit: TBD
- CV Score: 0.303
- LB Score: **0.303**
- 主要変更: ARTIFACTS_DIR命名修正（ハイフン区切り）
- Artifacts: [`experiments/EXP001T/v8_score0.303/exp001t-v8-artifacts/`](experiments/EXP001T/v8_score0.303/exp001t-v8-artifacts/)
- Notebook: [`experiments/EXP001T/v8_score0.303/recod-ai-luc-train.ipynb`](experiments/EXP001T/v8_score0.303/recod-ai-luc-train.ipynb)

### EXP001T v9
- Date: 2025-01-XX
- Git commit: TBD
- CV Score: 0.303（best config: mask_bin=0.5, min_area=25）
- LB Score: **0.303**
- 主要変更: mask二値化/min_areaのグリッド探索（9パターン）
- Artifacts: [`experiments/EXP001T/v9_score0.303/exp001t-v9-artifacts/`](experiments/EXP001T/v9_score0.303/exp001t-v9-artifacts/)
- Notebook: [`experiments/EXP001T/v9_score0.303/recod-ai-luc-train.ipynb`](experiments/EXP001T/v9_score0.303/recod-ai-luc-train.ipynb)
- run.json: [`experiments/EXP001T/v9_score0.303/run.json`](experiments/EXP001T/v9_score0.303/run.json)

---

## EXP002T シリーズ

### EXP002T v6
- Date: 2025-11-04
- Git commit: TBD
- CV Score: 0.244
- LB Score: **0.244**
- 主要変更: Ultra-Fast U-Net⚡, 5-fold, TTA準備
- Notebook: [`experiments/EXP002T/v6_score0.244/recod-ai-luc-train.ipynb`](experiments/EXP002T/v6_score0.244/recod-ai-luc-train.ipynb)

### EXP002T v3e
- Date: 2025-11-01
- Git commit: TBD
- CV Score: 0.301
- LB Score: **0.301**
- 主要変更: 後処理最適化・退化解ガード
- Notebook: [`experiments/EXP002T/v3e_score0.301/recod-ai-luc-train.ipynb`](experiments/EXP002T/v3e_score0.301/recod-ai-luc-train.ipynb)

### EXP002T v3c
- Date: 2025-11-01
- Git commit: TBD
- CV Score: 0.125
- LB Score: **0.125**
- 主要変更: 提出部分安全性強化
- Notebook: [`experiments/EXP002T/v3c_score0.125/recod-ai-luc-train.ipynb`](experiments/EXP002T/v3c_score0.125/recod-ai-luc-train.ipynb)

---

## EXP003T シリーズ

### EXP003T v3
- Date: 2025-11-09
- Git commit: TBD
- CV Score: 0.303
- LB Score: **0.303**
- 主要変更: DINOv2特徴量+ML, 分離指標健全run
- Notebook: [`experiments/EXP003T/v3_score0.303/recod-ai-luc-train.ipynb`](experiments/EXP003T/v3_score0.303/recod-ai-luc-train.ipynb)

### EXP003T v9
- Date: 2025-11-XX
- Git commit: TBD
- CV Score: 0.304
- LB Score: **0.304**
- 主要変更: DINOv2 + CNN Head (Custom), 10 epochs
- Notebook: [`experiments/EXP003T/v9_score0.304/recod-ai-luc-train.ipynb`](experiments/EXP003T/v9_score0.304/recod-ai-luc-train.ipynb)

### EXP003T v14
- Date: 2025-11-XX
- Git commit: TBD
- CV Score: 0.318
- LB Score: **0.318**
- 主要変更: DINOv2 + CNN Head, Borrowed Weights (Pretrained on similar task)
- Notebook: [`experiments/EXP003T/v14_score0.318/recod-ai-luc-train.ipynb`](experiments/EXP003T/v14_score0.318/recod-ai-luc-train.ipynb)

### EXP003T v15
- Date: 2025-11-XX
- Git commit: TBD
- CV Score: 0.318
- LB Score: **0.318**
- 主要変更: v14の再現性確認 (Reproducibility Check)
- Notebook: [`experiments/EXP003T/v15_score0.318/recod-ai-luc-train.ipynb`](experiments/EXP003T/v15_score0.318/recod-ai-luc-train.ipynb)

---

## EXP004E シリーズ (Ensemble実験)

### EXP004E v1
- Date: 2025-11-29
- LB Score: **0.304**
- 主要変更: DINOv2+DeepLabV3+ 初期実装
- Note: Ensemble基盤構築

### EXP004E v2
- Date: 2025-11-30
- CV Score: 0.2334
- LB Score: **0.277**
- 主要変更: ResNet50 Multi-Head Ensemble
- Note: Ensemble効果が薄い。確率分布が0.20-0.25に集約。
- 詳細: [`experiments/EXP004E/v2_score0.277/VISUAL_ANALYSIS_REPORT.md`](experiments/EXP004E/v2_score0.277/VISUAL_ANALYSIS_REPORT.md)

### EXP004E v3
- Date: 2025-11-30
- CV Score: 0.3135
- LB Score: **0.306**
- 主要変更: DINOv2 (0.6) + DeepLabV3+ (0.4) Weighted Ensemble
- Note: v2比 +34.3% F1改善。F1>0.6が4倍に増加。ただしEnsemble効果は限定的。
- 詳細: [`experiments/EXP004E/v3_score0.306/VISUAL_ANALYSIS_REPORT.md`](experiments/EXP004E/v3_score0.306/VISUAL_ANALYSIS_REPORT.md)

---

## EXP005T シリーズ (DINOv2+SAM2)

### EXP005T v0
- Date: 2025-11-28
- LB Score: **0.318**
- 主要変更: DINOv2-only baseline（SAM2なし）
- Note: **シリーズ最高スコア**。SAM2追加で改善せず、むしろ微減。

### EXP005T v1
- Date: 2025-11-XX
- LB Score: **0.314**
- 主要変更: SAM2 Zero-shot (AutomaticMaskGenerator)
- Note: v0より微減(-0.004)。SAM2の推論効果は限定的。

### EXP005T v2
- Date: 2025-11-XX
- CV Score: 0.104
- LB Score: **0.104**
- 主要変更: SAM2 Fine-tuning (Hiera+MaskDecoder)
- Note: **Failure**. 過剰検出(FP)多発。SAM2はDetectionに向かないことが判明。
- 詳細: [`experiments/EXP005T/v2_score0.104/report.md`](experiments/EXP005T/v2_score0.104/report.md)

### EXP005T v3
- Date: 2025-11-29
- CV Score: 0.5458
- LB Score: **0.314**
- 主要変更: DINOv2+SAM2 refiner (conservative: alpha_grad=0.30)
- Note: SAM2統合成功。FP=2.52%。Authentic判定の信頼性高い。
- 詳細: [`experiments/EXP005T/v3_score0.314/VISUAL_ANALYSIS_REPORT.md`](experiments/EXP005T/v3_score0.314/VISUAL_ANALYSIS_REPORT.md)

### EXP005T v4
- Date: 2025-11-30
- CV Score: 0.5633
- LB Score: **0.313**
- 主要変更: DINOv2+SAM2 refiner (aggressive: alpha_grad=0.35)
- Note: パラメータ嗜好不一致でv3より微減。SAM2にはconservativeが最適。

---

## 総括

- **シリーズ**: EXP001T/EXP002T/EXP003T（Mask R-CNN, U-Net, DINOv2+ML）
- **達成**: 0.303ベースライン確立、DINOv2系で分離指標健全runも達成
- **課題**: さらなる後処理・解像度UP・アンサンブル・特徴量拡張で0.305突破を目指す
- **詳細**: [`experiments/EXP001T/report_1T_summary.md`](experiments/EXP001T/report_1T_summary.md), [`experiments/EXP002T/interim_report.md`](experiments/EXP002T/interim_report.md), [`experiments/COMPREHENSIVE_REPORT.md`](experiments/COMPREHENSIVE_REPORT.md)