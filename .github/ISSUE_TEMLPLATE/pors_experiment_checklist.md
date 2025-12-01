# Post Experiment: 記録・アップロード・PR（チェックリスト）

---
name: "Post Experiment: 記録・アップロード・PR"
about: 実験終了後に実施する必須チェックリスト（DL → 配置 → AI生成 → PR）
title: "Post Experiment | <EXP_ID>"
labels: documentation, maintenance
assignees: ""
---

## 入力（必ず埋める）
- EXP_ID:
- Date:
- Commit SHA:
- CV: oof_auc / fold_aucs
- Public LB / submission_id:
- Artifacts dir (例: artifacts/models/<EXP_ID>)
- Notes (短文)

## 収集（Kaggle から DL）
- [ ] /kaggle/working/submission.csv
- [ ] /kaggle/working/oof.csv or oof.* (あれば)
- [ ] /kaggle/working/metrics.json (あれば)
- [ ] train_notebook.ipynb / inference_notebook.ipynb（最終版）
- [ ] requirements.txt / pip-freeze.txt（推奨）

## 配置（ローカル/Repo）
- 軽量成果物 → `experiments/<EXP_ID>/<exp_id>-artifacts/`
  - submission.csv → `experiments/<EXP_ID>/<exp_id>-artifacts/submission.csv`
  - oof.csv → `experiments/<EXP_ID>/<exp_id>-artifacts/oof.csv`
  - metrics.json → `experiments/<EXP_ID>/<exp_id>-artifacts/metrics.json`
- ノートブック → `experiments/<EXP_ID>/notebooks/`
- 小図/ログ → `experiments/<EXP_ID>/<exp_id>-artifacts/plots/`
- 大きなモデル/アセット → `artifacts/models/<EXP_ID>/`（.gitignore 対象）

## AI に自動生成してもらう（このIssueに追記）
- [ ] 上の入力を埋めて `.prompts/99_post_experiment_master.md` を呼び出す
  - 生成物: `run.json`, `experiments/<EXP_ID>/report.md`, `experiments/<EXP_ID>/README.md` (Run Card),
    README.md の短評行パッチ、`docs/scoreboard.md` の追記パッチ、PR 文案
- [ ] `python scripts/update_experiment_index.py` を実行して `experiments/INDEX.md` 更新

## 反映
- [ ] git add / commit / push（新規ブランチ）
- [ ] PR を作成してレビュー依頼

参考: [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md), [docs/SOP_SIMPLE.md](docs/SOP_SIMPLE.md)