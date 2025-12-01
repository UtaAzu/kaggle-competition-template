# Strategy Template (Generic)

This document is a minimal, generic strategy template for describing your approach to a data science competition or ML project. Replace sections with your project-specific details.

Core sections:
- Problem statement and outputs
- Public/private dataset assumptions and schema
- Evaluation metric & reproducible implementation (place metric code under `src/metrics/`)
- Baselines and quick wins
- Training/validation/inference pipeline
- Postprocessing and submission format
- Experiment tracking & artifacts to save

Keep competition- or dataset-specific files and notebooks in `examples/archive/<competition>/` to keep the template generic.
# Strategy Template (Generic)

This repository contains a set of templates and examples for Kaggle-style competitions. Use this page to outline your strategy for the current competition or project using the following template.

Sections to fill in:

- Problem summary
- Evaluation metric: implement metric function(s) in `src/metrics/`
- Data layout and pre-processing
- Baseline model(s) and expected performance
- Training and inference pipeline
- Post-processing & submission format
- Experiment tracking / reproducibility checklist

For archived competition examples and past experiments, see `examples/archive/`.
# Strategy Template (Generic)

This document is a generic strategy template for describing the approach to a Kaggle-style competition or ML project. Replace example sections with your competition-specific details.
# Strategy Template (Generic)

This is a generic strategy template for describing your approach. Replace it with your competition-specific plan and references.
# 画像コンペ初期戦略（セグメンテーション）テンプレート — 分類＋インスタンス検出対応

準拠プロンプト: [.prompts/00_initial_strategy_analysis.md](.prompts/00_initial_strategy_analysis.md)  
評価ノート: `src/metrics/<your_metric_notebook>.ipynb`（例: F1/X metricsの整合）  
実験運用: [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md), [docs/scoreboard.md](docs/scoreboard.md)  
補助スクリプト: [train.py](train.py), [predict.py](predict.py)

---

## 0) クリティカルレビューの要点（反映事項）
- 評価要件に直結する2点を明示対応
  - 画像レベル分類（authentic/forged）
  - インスタンス（偽造箇所の個数）検出
- コピー＆ムーブの本質（自己相関・領域間類似）を入力設計・モデルへ注入
- 独自性: 合成コピー＆ムーブ生成と周波数/自己相関チャネル強化を中核に

---

## 1) 評価・提出の整合（例: F1ベース評価）
  - 画像ラベル: authentic / forged（確率としきい値で判定）
  - インスタンス数: しきい値・後処理後の領域数で推定
  - セグメンテーション: 各インスタンスのバイナリマスク（RLE提出想定）
  - 3軸同時最適化（CVでgrid/BO）
    - 画像ラベル閾値 τ_cls
    - マスク二値化閾値 τ_seg、最小面積 min_area、連結性 connectivity
    - インスタンス信頼度 τ_inst（インスタンス系モデル時）
## 1) 評価・提出の整合（Metric ベース）
 画像ごとに出力
  - 画像ラベル: authentic / forged（確率としきい値で判定）
  - インスタンス数: しきい値・後処理後の領域数で推定
  - セグメンテーション: 各インスタンスのバイナリマスク（RLE提出想定）
 - ローカル最適化
   - 3軸同時最適化（CVでgrid/BO）
     - 画像ラベル閾値 τ_cls
     - マスク二値化閾値 τ_seg、最小面積 min_area、連結性 connectivity
     - インスタンス信頼度 τ_inst（インスタンス系モデル時）
   - スコアは $F1=\frac{2PR}{P+R}$（提出仕様のcount/label整合を含む評価実装で評価）
 - 参照: `src/metrics/<your_metric_notebook>.ipynb`（汎用的な評価実装の例）

---

## 2) モデル方針（優先度順）

- P0 最優先: インスタンスセグメンテーション直行
  - Mask R-CNN（Detectron2/torchvision）または YOLOv8-seg 系
  - 長所: 個数推定とマスク生成を一貫して学習・推論でき、評価要件に合致
  - 実装要点:
    - 空予測時は authentic と判定（τ_instで制御）
    - small-object対策: FPN高解像度層の強化、anchor/stride調整、min_area後処理
    - TTA + NMS調整（instance重複抑制）

- P1 次点: カスケード（分類 → セマンティック → 後処理）
  - Stage-1: 画像分類（EfficientNet/ConvNeXt）で forged 事前フィルタ
  - Stage-2: U-Net++/DeepLabV3+ でセマンティックマスク
  - Stage-3: 後処理でインスタンス化
    - 二値化→開閉→距離変換→watershed→connected components→min_area除去
  - 長所: 計算効率/実装容易、分離困難ケースはwatershedで分割

- P2 挑戦的: マルチタスク・コピー&ムーブ特化
  - セグメント＋画像ラベル＋カウント回帰（または離散分類）を同時学習
  - 損失合成: $L = \lambda_{seg} L_{seg} + \lambda_{cls} L_{cls} + \lambda_{cnt} L_{cnt}$
  - 相関ブロック（自己相関/パッチ相関）をエンコーダ中に組み込み

---

## 3) コピー＆ムーブの「関係性」をモデルへ注入（特効薬）
- 自己相関/パッチ類似チャネル（入力追加）
  - multi-scale PatchNCC/SSIM マップ、Deep feature cosine 相関（FPN各段で計算し縮約）
  - correlation volume（FlowNet様）を生成し、Segヘッドへconcat
- Siamese/Triplet補助学習
  - 画像内パッチ対で「同一元か否か」を判定する補助タスク（蒸留でメインへ転写）
- 周波数領域特徴
  - DCT高周波エネルギー、ラプラシアン/High-pass、JPEG品質変化の局所指標
- アブレーションで寄与を定量化し、効果が高いチャネルのみ採用

---

## 4) 後処理と個数推定（セマンティック系の要）
- マスク二値化 τ_seg をCVで最適化
- 小物体ノイズ除去: min_area（連結成分面積）、細線化/開閉
- 接触分離: 距離変換 + watershed、境界スナップ（エッジ重み付き）
- 最終個数 = 有効インスタンス数（領域信頼度/面積でフィルタ）
- 画像ラベルは「個数>0 かつ信頼度>τ_inst」で forged、そうでなければ authentic

---

## 5) データ生成とAug（独創性・汎化の源泉）
- 合成コピー＆ムーブ・ジェネレータ（学習時 on-the-fly）
  - ランダムパッチ選択→変形（scale/rot/affine）→ぼかし/ノイズ→Poissonブレンディング→貼付
  - 貼付回数/サイズ/重なりをランダム化（個数多様化）
  - 画像レベルラベルとGTマスク/個数を自動生成（教師強化）
- ハードネガティブ採掘
  - 偽陽性の背景パターンを収集し再学習で抑制
- 周波数/圧縮Aug
  - JPEG品質揺らぎ、リサイズ/再サンプリング、色空間撹乱でロバスト化

---

## 6) 初期アクションプラン（短期3ステップ）

- Step 1: 評価/CVハーネスの固定化（最優先）
  - 目的: 画像ラベル/個数/マスクを含む提出仕様で $F1$ を厳密再現
  - アクション:
    - `src/metrics/example_metrics.py` を関数化（oof評価・閾値探索・RLE往復テストの例）
    - GroupKFold（画像/撮影単位group）とfold別最適閾値ログ
    - テスト: 空/単一/多インスタンス、極小物体、結合ケース
  - 成果物: 評価関数・CV分割・閾値メタ（metrics.json/oof.csv）。[docs/scoreboard.md](docs/scoreboard.md) へ記録

- Step 2: カスケード・ベースライン（高速で土台）
  - 目的: authentic除外と個数推定を両立する最小実装でCV確立
  - アクション:
    - Stage-1: 画像分類（軽量バックボーン）で forged prior
    - Stage-2: U-Net++（BCE+Dice）でセマンティック
    - Stage-3: CC+watershedでインスタンス化、(τ_cls, τ_seg, min_area) をCV最適化
    - 推論: TTA(flip)平均、RLE化、提出前検証
  - 成果物: ベースラインCV/LB、提出一式。再現コマンドは [train.py](train.py)/[predict.py](predict.py) に準拠

- Step 3: インスタンス直行（性能版）＋コピー＆ムーブ特化
  - 目的: 評価要件に最短距離で到達しつつ独自性で上積み
  - アクション:
    - Mask R-CNN/YOLOv8-segの学習（small-object/anchor調整）
    - 自己相関/周波数チャネルの追加アブレーション
    - 合成コピー＆ムーブでデータ拡張（比率/難易度スケジュール）
  - 成果物: P0モデルのCV改善、特化チャネルの有効性レポート。スコアは [docs/scoreboard.md](docs/scoreboard.md) に追記

---

## 7) 実装メモ（最小ユーティリティ）
- RLE・連結成分・watershed・閾値探索はユニットテスト必須
- ラベル整合
  - authentic: 有効インスタンス0
  - forged: 有効インスタンス>=1（信頼度/面積基準を満たす）
- 記録
  - fold別 τ_cls/τ_seg/min_area/τ_inst
  - OOF per-image: label_pred, count_pred, masks_meta（面積/スコア）

---

## 8) 次の独創的な一手（候補）
- 相関ボリューム学習（学習可能相関層）をU-Netボトルネックへ挿入
- パッチ検索（近傍場）に基づく「コピー元↔貼付先」の対応推定を補助出力化
- マルチタスクの count-head を Poisson/Ordinal 回帰で安定化

---
```// filepath: docs/00_initial_strategy.md
# 画像コンペ初期戦略（セグメンテーション）テンプレート — 分類＋インスタンス検出対応

準拠プロンプト: [.prompts/00_initial_strategy_analysis.md](.prompts/00_initial_strategy_analysis.md)  
評価ノート: `src/metrics/example_metrics.py`（汎用的な評価実装の例）  
実験運用: [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md), [docs/scoreboard.md](docs/scoreboard.md)  
補助スクリプト: [train.py](train.py), [predict.py](predict.py)

---

## 0) クリティカルレビューの要点（反映事項）
- 評価要件に直結する2点を明示対応
  - 画像レベル分類（authentic/forged）
  - インスタンス（偽造箇所の個数）検出
- コピー＆ムーブの本質（自己相関・領域間類似）を入力設計・モデルへ注入
- 独自性: 合成コピー＆ムーブ生成と周波数/自己相関チャネル強化を中核に

---

## 1) 評価・提出の整合（例: F1ベース評価）
- 画像ごとに出力
  - 画像ラベル: authentic / forged（確率としきい値で判定）
  - インスタンス数: しきい値・後処理後の領域数で推定
  - セグメンテーション: 各インスタンスのバイナリマスク（RLE提出想定）
- ローカル最適化
  - 3軸同時最適化（CVでgrid/BO）
    - 画像ラベル閾値 τ_cls
    - マスク二値化閾値 τ_seg、最小面積 min_area、連結性 connectivity
    - インスタンス信頼度 τ_inst（インスタンス系モデル時）
  - スコアは $F1=\frac{2PR}{P+R}$（提出仕様のcount/label整合を含む実装で評価）
  - 参照: `src/metrics/example_metrics.py`（汎用的な評価関数の例）

---

## 2) モデル方針（優先度順）

- P0 最優先: インスタンスセグメンテーション直行
  - Mask R-CNN（Detectron2/torchvision）または YOLOv8-seg 系
  - 長所: 個数推定とマスク生成を一貫して学習・推論でき、評価要件に合致
  - 実装要点:
    - 空予測時は authentic と判定（τ_instで制御）
    - small-object対策: FPN高解像度層の強化、anchor/stride調整、min_area後処理
    - TTA + NMS調整（instance重複抑制）

- P1 次点: カスケード（分類 → セマンティック → 後処理）
  - Stage-1: 画像分類（EfficientNet/ConvNeXt）で forged 事前フィルタ
  - Stage-2: U-Net++/DeepLabV3+ でセマンティックマスク
  - Stage-3: 後処理でインスタンス化
    - 二値化→開閉→距離変換→watershed→connected components→min_area除去
  - 長所: 計算効率/実装容易、分離困難ケースはwatershedで分割

- P2 挑戦的: マルチタスク・コピー&ムーブ特化
  - セグメント＋画像ラベル＋カウント回帰（または離散分類）を同時学習
  - 損失合成: $L = \lambda_{seg} L_{seg} + \lambda_{cls} L_{cls} + \lambda_{cnt} L_{cnt}$
  - 相関ブロック（自己相関/パッチ相関）をエンコーダ中に組み込み

---

## 3) コピー＆ムーブの「関係性」をモデルへ注入（特効薬）
- 自己相関/パッチ類似チャネル（入力追加）
  - multi-scale PatchNCC/SSIM マップ、Deep feature cosine 相関（FPN各段で計算し縮約）
  - correlation volume（FlowNet様）を生成し、Segヘッドへconcat
- Siamese/Triplet補助学習
  - 画像内パッチ対で「同一元か否か」を判定する補助タスク（蒸留でメインへ転写）
- 周波数領域特徴
  - DCT高周波エネルギー、ラプラシアン/High-pass、JPEG品質変化の局所指標
- アブレーションで寄与を定量化し、効果が高いチャネルのみ採用

---

## 4) 後処理と個数推定（セマンティック系の要）
- マスク二値化 τ_seg をCVで最適化
- 小物体ノイズ除去: min_area（連結成分面積）、細線化/開閉
- 接触分離: 距離変換 + watershed、境界スナップ（エッジ重み付き）
- 最終個数 = 有効インスタンス数（領域信頼度/面積でフィルタ）
- 画像ラベルは「個数>0 かつ信頼度>τ_inst」で forged、そうでなければ authentic

---

## 5) データ生成とAug（独創性・汎化の源泉）
- 合成コピー＆ムーブ・ジェネレータ（学習時 on-the-fly）
  - ランダムパッチ選択→変形（scale/rot/affine）→ぼかし/ノイズ→Poissonブレンディング→貼付
  - 貼付回数/サイズ/重なりをランダム化（個数多様化）
  - 画像レベルラベルとGTマスク/個数を自動生成（教師強化）
- ハードネガティブ採掘
  - 偽陽性の背景パターンを収集し再学習で抑制
- 周波数/圧縮Aug
  - JPEG品質揺らぎ、リサイズ/再サンプリング、色空間撹乱でロバスト化

---

## 6) 初期アクションプラン（短期3ステップ）

- Step 1: 評価/CVハーネスの固定化（最優先）
  - 目的: 画像ラベル/個数/マスクを含む提出仕様で $F1$ を厳密再現
  - アクション:
    - `src/metrics/example_metrics.py` を関数化（oof評価・閾値探索・RLE往復テストの例）
    - GroupKFold（画像/撮影単位group）とfold別最適閾値ログ
    - テスト: 空/単一/多インスタンス、極小物体、結合ケース
  - 成果物: 評価関数・CV分割・閾値メタ（metrics.json/oof.csv）。[docs/scoreboard.md](docs/scoreboard.md) へ記録

- Step 2: カスケード・ベースライン（高速で土台）
  - 目的: authentic除外と個数推定を両立する最小実装でCV確立
  - アクション:
    - Stage-1: 画像分類（軽量バックボーン）で forged prior
    - Stage-2: U-Net++（BCE+Dice）でセマンティック
    - Stage-3: CC+watershedでインスタンス化、(τ_cls, τ_seg, min_area) をCV最適化
    - 推論: TTA(flip)平均、RLE化、提出前検証
  - 成果物: ベースラインCV/LB、提出一式。再現コマンドは [train.py](train.py)/[predict.py](predict.py) に準拠

- Step 3: インスタンス直行（性能版）＋コピー＆ムーブ特化
  - 目的: 評価要件に最短距離で到達しつつ独自性で上積み
  - アクション:
    - Mask R-CNN/YOLOv8-segの学習（small-object/anchor調整）
    - 自己相関/周波数チャネルの追加アブレーション
    - 合成コピー＆ムーブでデータ拡張（比率/難易度スケジュール）
  - 成果物: P0モデルのCV改善、特化チャネルの有効性レポート。スコアは [docs/scoreboard.md](docs/scoreboard.md) に追記

---

## 7) 実装メモ（最小ユーティリティ）
- RLE・連結成分・watershed・閾値探索はユニットテスト必須
- ラベル整合
  - authentic: 有効インスタンス0
  - forged: 有効インスタンス>=1（信頼度/面積基準を満たす）
- 記録
  - fold別 τ_cls/τ_seg/min_area/τ_inst
  - OOF per-image: label_pred, count_pred, masks_meta（面積/スコア）

---

## 8) 次の独創的な一手（候補）
- 相関ボリューム学習（学習可能相関層）をU-Netボトルネックへ挿入
- パッチ検索（近傍場）に基づく「コピー元↔貼付先」の対応推定を補助出力化
- マルチタスクの count-head を Poisson/Ordinal 回帰で安定化

---