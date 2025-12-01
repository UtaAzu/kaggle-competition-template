# 実験ワークフローガイド（日本語）

本ガイドは「ツール実行 + 永続記録」を軸とした、再現性ある実験運用フローです。  
特に「何を」「どこで実行し」「何を」「どこに保存するか」を明確に定義します。
初心者向けの「簡易SOP」は [docs/SOP_SIMPLE.md](docs/SOP_SIMPLE.md) を参照してください。まずはこちらを習慣化し、慣れてきたら本ドキュメントの高品質手順へ移行してください。

- 原則
  1) すべての実験は「予約（Reserve）」してから開始する  
  2) すべての成果物は experiment 固有のディレクトリに自動保存する  
  3) すべてのメタデータは永続的に追跡・更新する
  4) **複数fold実験の場合は「foldごと成果物」と「全体集計成果物（overall_metrics.json等）」の両方を必ず生成・保存する**

参考ファイル
- 実行スクリプト: [scripts/reserve_experiment.py](scripts/reserve_experiment.py), [scripts/update_experiment_index.py](scripts/update_experiment_index.py)
- 学習/推論: [train.py](train.py), [predict.py](predict.py)
- ドキュメント/記録: [README.md](README.md), [docs/scoreboard.md](docs/scoreboard.md), [experiments/INDEX.md](experiments/INDEX.md)
- Kaggle向けセル生成プロンプト: [.prompts/02_build_training_framework.md](.prompts/02_build_training_framework.md), [.prompts/03_build_inference_pipeline.md](.prompts/03_build_inference_pipeline.md)

---

## 1. 役割と実行環境

- 実行環境
  - Codespaces/ローカル devcontainer（推奨）: 実験の予約・設定・記録・インデックス再生成・コミット/プッシュを行う
  - Kaggle Notebook: 学習/推論の本番実行（必要に応じて）。提出用 submission.csv を生成
- 典型的な分担
  - Codespacesで「準備と記録」
  - Kaggleで「学習/推論/提出」

---

## 2. クイックスタート（SOP）

開始点（What/Where）
- What: 次にやる実験の目的・受入基準を決める
- Where: GitHub Issue（推奨）または [README.md](README.md) の「Next Action」
- Save: Issue に目的/期限/受入基準、参照リンク（過去実験/Prompts）を記載

Step 1) 実験を予約する
- What: 実験番号（EXP_ID）の確保とディレクトリ・設定ファイル作成
- Where（実行）: Codespaces（Kaggleでは実施しない）
- How:
  ```bash
  # 基本予約
  python scripts/reserve_experiment.py --system L --title "my_experiment"
  # テンプレ継承
  python scripts/reserve_experiment.py --system L --title "tfidf_optimized" --template "tfidf_baseline"
  ```
- Save（自動生成されるもの）:
  - 実験メタ: experiments/EXPxxxL/run.json
  - 設定: config/experiments/EXPxxxL.yaml
  - 成果物ルート: artifacts/models/EXPxxxL/（空の器）
  - インデックス更新: [experiments/INDEX.md](experiments/INDEX.md)

Step 2) 設定を編集する
- What: データ/特徴/モデル/分割/CVなどの条件を設定
- Where（実行）: Codespaces
- How:
  ```bash
  vim config/experiments/EXPxxx?.yaml
  ```
- 注意: artifacts.experiment_name は変更しない（予約IDと連動）
- Save: 修正済みの YAML をコミット

Step 3) 学習を実行する（2パターン）
- パターンA: リポジトリの学習スクリプトで実行（推奨）
  - What: [train.py](train.py) を設定ファイルで実行
  - Where（実行）: Codespaces or サーバ（データが手元にある場合）
  - How:
    ```bash
    python train.py --config config/experiments/EXPxxx?.yaml
    ```
  - Save（自動）:
    - モデル/前処理/OOF/指標/設定スナップショット → artifacts/models/EXPxxx?/
    - メタ更新（status, cv, git_commit） → experiments/EXPxxx?/run.json
    - 可視化（自動）: 各foldの `*-artifacts/fold*/val_samples.png` に FP/FN/OK の代表例を保存
    - **複数foldの場合は「foldごと成果物（oof.csv, metrics.json, run.json等）」と「全体集計成果物（overall_metrics.json, oof_all.csv, validate_summary.csv等）」を必ず生成・保存すること**
- パターンB: Kaggle Notebookで自己完結スクリプトを実行
  - What: 単一ファイル訓練スクリプト（.promptsに準拠）をセルに貼り付け実行
  - Where（実行）: Kaggle Notebook
  - Save（Kaggle 側の出力）:
    - <WORKING_DIR>/artifacts/EXPxxx?/ に成果物（model_fold_*.pkl, vectorizer_*.pkl, oof.*, metrics.json, logs など）
    - 実行後の取り扱い:
      1) Kaggle の「Create Dataset」で「モデル成果物Dataset」を作成し、slugを記録
      2) もしくは成果物をダウンロードし、ローカルで artifacts/models/EXPxxx?/ に配置してコミット管理外（大容量はGit追跡外を推奨）

Step 4) 推論を実行し提出ファイルを作る
- 推奨: Kaggle Notebookで推論（コンペデータと同一環境）
- What: モデル成果物Datasetを「Add Data」し、推論スクリプトで submission.csv を出力
- Where（実行）: Kaggle Notebook
- How: 単一ファイル推論（.prompts準拠）を実行。出力は submission.csv のみ
- Save（Kaggle 側の出力）:
  - <WORKING_DIR>/submission.csv（形式: row_id,rule_violation）
  - 提出前チェック: 学習スクリプト内の `validate_submission_format` を必ず実行（case_id順・RLE偶数要素・authentic表記）。失敗時は提出を中止。
- 代替（ローカル）: [predict.py](predict.py) で検証用推論
  ```bash
  python predict.py --config config/experiments/EXPxxx?.yaml --test test.csv --output submission.csv
  ```

Step 5) 提出とスコア記録
- What: Kaggle に提出し LB スコア/Submission ID を記録
- Where（実行）: Kaggle（提出）、Codespaces（記録）
- Save:
  - experiments/EXPxxx?/run.json に leaderboard.public_lb / submission_id を追記
  - サマリ: [docs/scoreboard.md](docs/scoreboard.md) に1行追記
  - 概要: [README.md](README.md) のスコアボード/実験日誌に要約を追記

Step 6) 成果物の回収と保管（Kaggle→ローカル）
- What: 学習ログ/OOF/指標/設定/可視化などを実験ディレクトリへ収集
- Where（実行）: Codespaces
- Save（推奨配置）:
  - 正本（モデル類）: Kaggle Datasetにアップロードし、slugをrun.jsonに記録
  - 実験記録: experiments/EXPxxx?/ にレポート/ログ/リンク
    - 例: experiments/EXPxxx?/report.md, experiments/EXPxxx?/artifacts/（軽量物のみ）, experiments/EXPxxx?/README.md
    - **複数foldの場合は「foldごと成果物」と「全体集計成果物」の両方を必ず保存**

Step 7) 実験インデックスの再生成（重要）
- What: すべての run.json を走査し [experiments/INDEX.md](experiments/INDEX.md) を再生成
- Where（実行）: Codespaces（Kaggleでは実施しない）
- How:
  ```bash
  python scripts/update_experiment_index.py
  ```
- Save: experiments/INDEX.md を更新（コミット対象）

Step 8) コミットとIssueクローズ
- What: 変更をまとめてコミット/プッシュし、Issue をクローズ
- Where（実行）: Codespaces
- How:
  ```bash
  git add experiments/EXPxxx? docs/scoreboard.md experiments/INDEX.md README.md config/experiments/EXPxxx?.yaml
  git commit -m "EXPxxx?: record results, update scoreboard and index"
  git push
  ```
- Save: リポジトリに反映（PR運用も可）

---

## 3. ディレクトリ構造（標準）

```
experiments/EXP004G/
├── README.md            # 実験のRun Card（要約・所見・リンク）
├── report.md            # 詳細レポート（任意）
├── run.json             # 機械可読なメタデータ（必須）
├── exp004g-v1-artifacts/
│   ├── fold0/
│   ├── fold1/
│   ├── ...
│   ├── overall_metrics.json
│   ├── validate_summary.csv
│   └── oof_all.csv
├── artifacts/           # 実験固有の軽量成果物（画像/表など任意）
└── notebooks/           # 実験固有ノートブック（任意）

config/experiments/
└── EXP004G.yaml         # 実験設定（予約時に自動生成）

artifacts/models/EXP004G/
├── model.pkl / model_fold_*.pkl
├── tfidf_vectorizer.pkl / vectorizer_fold_*.pkl
├── oof_predictions.npy / oof.csv
├── cv_results.json / metrics.json / cv_audit_summary.json
└── config.json（スナップショット）
```

---

## 4. メタデータ（run.json）の例

```json
{
  "experiment_id": "EXP004L",
  "title": "EXP004L_enhanced_tfidf_char_ngrams",
  "system": "L",
  "kaggle_notebook": "mabe-linear-models",
  "kaggle_version": 2,
  "created_at": "2025-09-01T08:15:00.000Z",
  "status": "completed",
  "git_commit": "abc123f",
  "cv": {
    "oof_auc": 0.7234,
    "mean": 0.7234
  },
  "leaderboard": {
    "public_lb": 0.540,
    "submission_id": "12345678"
  },
  "artifacts": {
    "models_dir": "artifacts/models/EXP004L",
    "experiment_dir": "experiments/EXP004L",
    "dataset_uri": "kaggle://username/exp004l-dataset",
    "models_location": "kaggle_dataset"
  },
  "model_dataset_slug": "username/exp004l-dataset",
  "notes": "CV設計監査、char n-gram追加"
}
```

---

## 5. コマンド一覧（抜粋）

- 予約
  ```bash
  python scripts/reserve_experiment.py --system G --title "my_experiment"
  python scripts/reserve_experiment.py --system D --title "roberta_base" --template "tfidf_baseline"
  python scripts/reserve_experiment.py --system E --title "ensemble_test" --dry-run
  ```
- 学習
  ```bash
  python train.py --config config/experiments/EXP004G.yaml
  ```
- 推論
  ```bash
  python predict.py --config config/experiments/EXP004G.yaml --test test.csv --output submission.csv
  ```
- インデックス再生成（Codespacesで実行）
  ```bash
  python scripts/update_experiment_index.py
  ```

---

## 6. バリデーション/安全策

- 実験予約チェック
  - 実験ディレクトリが存在しない / run.json が無い / 設定に experiment_name が無い → 学習は失敗させる
- 成果物の保護
  - すべて experiment 固有ディレクトリに保存（上記構造）
- Git連携
  - コミットハッシュの自動取得、ステータス変更の追跡、任意コミットからの再現

- 可視化SOP（Two-Stage）
  - Stage A（学習直後・fold別）: `val_samples.png` に FP（赤輪郭）/FN（緑輪郭のみ）/OK を各3枚保存
  - Stage B（横断比較・validate）: `validate.py` を実行し、F1分布ヒスト（oof_f1_distributions.png）とサマリーCSV/JSONを `experiments/validate_outputs/` に保存
  - 目視で退化解（全authentic提出、極端なF1=0/1集中）を検知し、次アクションに反映
  - **複数foldの場合は「foldごと成果物」と「全体集計成果物」の両方を必ず確認・保存**

---

## 7. エラーと対処

- "Experiment not found": 予約忘れ → 予約を実行
  ```bash
  python scripts/reserve_experiment.py --system G --title "my_title"
  ```
- "Directory already exists": 既に予約済み → experiments/ を確認
- "Missing experiment_name": 設定不整合 → config の artifacts.experiment_name を確認
- ステータス確認
  ```bash
  cat experiments/EXP004G/run.json | jq '.status'
  ```
- 成果物確認
  ```bash
  ls -la artifacts/models/EXP004G/
  ```
- 設定デバッグ（読み込み/検証）
  ```bash
  python -c "
from src.utils.config import load_config, validate_experiment_reservation
config = load_config('config/experiments/EXP004G.yaml')
validate_experiment_reservation(config)
print('✅ Configuration valid')
"
  ```
- 実験一覧（インデックス）
  ```bash
  cat experiments/INDEX.md
  ```

---

## 8. ベストプラクティス

1) 必ず予約してから着手（構造とIDを先に固める）  
2) 説明的なタイトルを付与し、README/scoreboardの更新を習慣化  
3) 成功テンプレを焼き増して生産性を上げる  
4) インデックスは Codespaces で定期更新  
5) 大容量アーティファクトは Kaggle Datasetにアップロードし、slugをrun.jsonに記録  
6) Kaggle 推論は submission.csv のみを出力（ノイズを入れない）
7) **複数foldの場合はfoldごと＋全体集計成果物の両方を必ず保存・管理する**

---

## 9. Kaggle Notebook での実務ルール（要点）

- 学習ノート
  - 入力: 競技データ + （必要に応じ）コード/モデルDataset
  - 出力: <WORKING_DIR>/artifacts/EXPxxx?/（モデル、ベクトライザ、OOF、指標、ログ）
  - 次アクション: Dataset
  - 次アクション: Dataset化してslugを記録 or ダウンロードしてローカルへ収集
- 推論ノート
  - 入力: モデル成果物Dataset
  - 出力: <WORKING_DIR>/submission.csv（row_id,rule_violation のみ）
  - 次アクション: Kaggle へ提出、LB/Submission ID を run.json / scoreboard に記録

---

## 10. 参考プロンプト

- 自己完結・学習スクリプトの要件: [.prompts/02_build_training_framework.md](.prompts/02_build_training_framework.md)
- 自己完結・推論スクリプトの要件: [.prompts/03_build_inference_pipeline.md](.prompts/03_build_inference_pipeline.md)

---

## 11. 提出前チェックリスト（必須）

- [ ] 行数が `sample_submission.csv` と一致し、case_id順序が完全一致
- [ ] `annotation` は authentic か RLE（JSONの偶数配列、intのみ）
- [ ] `validate_submission_format` の実行が PASS（train.py 内で自動実行）
- [ ] 分離指標: forged_nonempty_ratio ≥ 0.2 かつ authentic_fp_ratio ≤ 0.3（退化解ガード）
- [ ] **複数foldの場合はfoldごと成果物＋全体集計成果物（overall_metrics.json等）が揃っていることを確認**