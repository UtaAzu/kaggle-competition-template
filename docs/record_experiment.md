---

## 1. 実験終了後に走らせるべきスクリプト・プロンプト

### スクリプト

1. **record_experiment.sh**
   - 対話型で run.json・report.md・README.md・checklist.md などを自動生成
   - 例:  
     ```sh
     bash scripts/record_experiment.sh EXP004D --apply
     ```
   - 必要に応じて `--push` `--pr` `--reserve-if-missing` オプションも利用

2. **auto_finalize_experiment.py**
   - 非対話で run.json・report.md・checklist.md を生成（record_experiment.shから内部的に呼ばれる）
   - 例:  
     ```sh
     python3 scripts/auto_finalize_experiment.py EXP004D <git_commit> <oof_auc> <public_lb> <submission_id>
     ```

3. **finalize_and_publish.py**
   - 厳密なファイルチェックとPR作成補助（record_experiment.shからオプションで呼ばれる）
   - 例:  
     ```sh
     python3 scripts/finalize_and_publish.py EXP004D <git_commit> <oof_auc> <public_lb> <submission_id>
     ```

4. **update_experiment_index.py**
   - 実験インデックス（INDEX.md）とスコアボード（scoreboard.md）を更新
   - 例:  
     ```sh
     python3 scripts/update_experiment_index.py
     ```

---

### プロンプト

- **08_post_experiment_report.md**
  - 実験レポート・run.json・README・report.md・checklist.md の自動生成テンプレート
  - スクリプト実行後に内容を確認・編集する際のガイド

---

## 2. 標準ワークフローまとめ

1. record_experiment.sh を実行（対話型で必要情報を入力）
2. 必要に応じて finalize_and_publish.py で厳密チェック・PR作成
3. `update_experiment_index.py` でインデックス・スコアボード更新
4. 08_post_experiment_report.md を参考に内容をレビュー・編集

---

### 複数fold実験時の注意

- **foldごと成果物（oof.csv, metrics.json, run.json等）は `experiments/<EXP_ID>/<exp_id>-artifacts/fold*/` に必ず保存**
- **全体集計成果物（overall_metrics.json, oof_all.csv, validate_summary.csv等）も `experiments/<EXP_ID>/<exp_id>-artifacts/` 直下に必ず保存**
- **validate.pyや検証Notebookで全体集計成果物を自動生成・確認すること**
- **run.jsonやreport.mdには全体集計指標（mean, std, 分離指標など）を記載すること**

---

### 参考ファイル
- record_experiment.sh
- auto_finalize_experiment.py
- finalize_and_publish.py
- update_experiment_index.py
- 08_post_experiment_report.md

---

この流れで成果物・レポート・メタデータを整理してください.