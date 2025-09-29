# Experiment Scoreboard

このファイルは詳細スコア表です。READMEには要約のみを記載し、本ファイルに実験の完全情報（データ/アーティファクト/環境/再現手順/差分）を残します。

## 記入ガイド（テンプレート）

### <EXPID> <short_title>
- Date:
- Git commit:
- Code Dataset: <slug> (vX)
- Model Dataset: <slug> (vX)
- Competition data: <slug> (vX)
- External data/models: <list with source URL, license> | None
- Features
  - Text fields:
  - Preprocess:
  - Vectorizer/Embedding params:
- Model
  - Type:
  - Key params:
- Training
  - Target:
  - CV: type=, n_splits=, stratify=, seed=
  - Class imbalance handling:
- Metrics (CV)
  - Primary: <metric> = mean ± std
  - Per-fold: [f1, f2, f3, f4, f5]
  - Other metrics (optional):
- Inference
  - experiment_name:
  - models_dir:
  - preprocessors_dir:
  - Threshold/postprocess:
- Repro commands
  - Train:
    ```bash
    python train.py --config <path>
    ```
  - Predict:
    ```bash
    python predict.py --config <path> --test <path> --output submission.csv
    ```
- Leaderboard
  - Public LB: <score> (submissionId=<id>, file vX) | pending
  - Private LB: <score> (公開後)
- Environment
  - Python / key libs: sklearn=, numpy=, pandas= …
  - Runtime: CPU/GPU, Kaggle Notebook
- Notes
  - Change vs previous:
  - What worked / didn’t:
  - Next:

---

## EXP001G_tfidf_baseline
- Date: 2025-08-29
- Git commit: 6845cde
- Code Dataset: kaggle-jigsaw2025 (v1)
- Model Dataset: kaggle-jigsaw2025-models (v1)
- Competition data: jigsaw-agile-community-rules (current)
- External data/models: None

- Features
  - Text fields: body, rule, positive_example_1, positive_example_2, negative_example_1, negative_example_2
  - Preprocess: lowercase, english stop_words, sublinear_tf
  - Vectorizer: TF-IDF(max_df=0.95, min_df=2, max_features=10000, ngram_range=(1,2))

- Model
  - Type: LogisticRegression
  - Params: C=1.0, max_iter=1000, random_state=42

- Training
  - Target: rule_violation
  - CV: stratified, n_splits=5, seed=42
  - Imbalance: (none)

- Metrics (CV)
  - AUC = 0.6153 ± 0.0329
  - Per-fold: [0.6755, 0.5958, 0.5852, 0.6255, 0.5957]

- Inference
  - experiment_name: EXP001G_tfidf_baseline
  - models_dir: /kaggle/input/kaggle-jigsaw2025-models/artifacts/models
  - preprocessors_dir: /kaggle/input/kaggle-jigsaw2025-models/artifacts/preprocessors

- Repro commands
  - Train (Kaggle):
    ```bash
    python /kaggle/input/kaggle-jigsaw2025/Kaggle-Jigsaw2025-main/train.py \
      --config /kaggle/working/config.yaml
    ```
  - Predict (Kaggle):
    ```bash
    python /kaggle/input/kaggle-jigsaw2025/Kaggle-Jigsaw2025-main/predict.py \
      --config /kaggle/working/config.yaml \
      --test /kaggle/input/jigsaw-agile-community-rules/test.csv \
      --output /kaggle/working/submission.csv
    ```

- Leaderboard
  - Public LB: 0.509 (submissionId=<id>)
  - Private LB: -

提出 ID の記録（任意だが推奨）:
- Kaggle UI: My Submissions から提出詳細を開くと URL 内に /submissions/<id> で確認
- CLI: ```bash
  kaggle competitions submissions -c jigsaw-agile-community-rules
  ```
  で submissionId とスコアの一覧を取得できます。

- Environment
  - Python: <fill>
  - sklearn: <fill>, numpy: <fill>, pandas: <fill>
  - Runtime: Kaggle CPU/GPU Notebook

- Notes
  - Change vs previous: initial baseline
  - Next: try C sweep, char-ngrams, per-field weighting, and linear SVM
  - Review: see experiments/EXP001G/devils_advocate.md

### EXP005G_cv_audit
- Date: 2025-09-02
- Git commit: TBD
- Code Dataset: TBD
- Model Dataset: TBD
- Competition data: jigsaw-agile-community-rules
- External data/models: None
- Features
  - Text fields: body, rule, positive_example_1/2, negative_example_1/2
  - Preprocess: lower, URL→<URL>, digits→<NUM>
  - Vectorizer/Embedding params: TF-IDF (max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95)
- Model
  - Type: LogisticRegression
  - Key params: C=1.0, max_iter=200
- Training
  - Target: rule_violation
  - CV: StratifiedGroupKFold, n_splits=5, group=md5_body_rule, seed=42
  - Class imbalance handling: None
- Metrics (CV)
  - Primary: AUC = 0.6137 ± 0.0188
  - Per-fold: [0.592235, 0.615373, 0.445141, 0.562363, 0.589882]
- Inference
  - experiment_name: EXP005G_cv_audit
  - models_dir: artifacts/models/EXP005G_cv_audit
  - preprocessors_dir: artifacts/models/EXP005G_cv_audit
  - Threshold/postprocess: None
- Leaderboard
  - Public LB: 0.522
- Environment
  - Python: 3.11.13
  - sklearn: 1.2.2, numpy: 1.26.4, pandas: 2.2.3
  - Runtime: Kaggle CPU Notebook
- Notes
  - Change vs previous: CV 監査（group_variation: md5_body_rule vs subreddit）
  - Next: 前処理強化 → EXP006G

### EXP006G_preproc_enhanced
- Date: 2025-09-02
- Git commit: TBD
- Code Dataset: TBD
- Model Dataset: TBD
- Competition data: jigsaw-agile-community-rules
- External data/models: None
- Features
  - Text fields: body, rule, positive_example_1/2, negative_example_1/2
  - Preprocess: lower, URL→<URL>, digits→<NUM>, emoji/symbol removal
  - Vectorizer/Embedding params: TF-IDF (max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95)
- Model
  - Type: LogisticRegression
  - Key params: C=1.0, max_iter=200
- Training
  - Target: rule_violation
  - CV: StratifiedGroupKFold, n_splits=5, group=md5_body_rule, seed=42
  - Class imbalance handling: None
- Metrics (CV)
  - Primary: AUC = 0.6040 ± 0.0279
  - Per-fold: [0.5913, 0.6187, 0.5654, 0.6483, 0.5963]
- Inference
  - experiment_name: EXP006G_preproc_enhanced
  - models_dir: artifacts/models/EXP006G_preproc_enhanced
  - preprocessors_dir: artifacts/models/EXP006G_preproc_enhanced
  - Threshold/postprocess: None
- Leaderboard
  - Public LB: 0.500
- Environment
  - Python: 3.11.13
  - sklearn: 1.2.2, numpy: 1.26.4, pandas: 2.2.3
  - Runtime: Kaggle CPU Notebook
- Notes
  - Change vs previous: 前処理強化（絵文字/記号除去追加）
  - Next: 校正/アンサンブル → EXP007G

### EXP006G\n- OOF AUC: 0.604000\n- Public LB: 0.500\n- Date: 2025-09-04\n

### EXP007G_cv_reaudit_and_ensemble
- Date: 2025-09-09
- System: G (Linear Models)
- CV OOF AUC: 0.512 (folds: [0.498, 0.525, 0.489, 0.532, 0.518])
- Public LB: 0.504
- Submission ID: TBD (ユーザーが手動で更新してください)
- Artifacts: artifacts/models/EXP007G, experiments/EXP007G/artifacts
- Environment: Python 3.x, sklearn, pandas, numpy
- Notes
  - Change vs previous: CV粒度変更＋アンサンブル導入。ギャップ微減。
  - What worked: アンサンブルで表現頑健化。
  - Next: LLM移行（Qwen2.5-LoRA）。
### EXP001D\n- OOF AUC: 0.500200\n- Public LB: TBD\n- Date: 2025-09-10\n

### EXP002D\n- OOF AUC: 0.586183\n- Public LB: 0.525\n- Date: 2025-09-11\n

### EXP003D\n- OOF AUC: 0.633442\n- Public LB: 0.514\n- Date: 2025-09-12\n
### EXP004D\n- OOF AUC: 0.576900\n- Public LB: 0.617\n- Date: 2025-09-26\n