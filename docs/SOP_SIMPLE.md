# 簡易SOP（初心者向け・最小運用）

目的: 最低限の手順で実験を完遂し、成果をリポジトリに残す習慣をつける。

1) 開始（Where / What）
- Where: GitHub Issue または [README.md](README.md) の Next Action
- What: 実験名（例: EXP006G_tfidf_cleanup）と目的を1文で記載

2) 訓練（Where / What / Save）
- Where: Kaggle Notebook（推奨）またはローカル
- What: 学習を実行（`train.py` または ノートブック）
- Save:
  - Kaggle実行 → `<WORKING_DIR>/artifacts/<EXP_ID>/` に出力
  - ローカル実行 → `artifacts/models/<EXP_ID>/`
  - **複数foldの場合は「foldごと成果物」と「全体集計成果物（overall_metrics.json等）」の両方を必ず保存**

3) 推論・提出（Where / What / Save）
- Where: Kaggle Notebook（Add Data で学習成果物を追加）
- What: 推論を実行し `<WORKING_DIR>/submission.csv`（row_id,rule_violation）を作成
- Save: 提出後、Submission ID とスコアをメモ

4) 回収と記録（Where / What）
- Where: Codespaces / ローカル
- 必須で repo に入れるもの（`experiments/<EXP_ID>/` 内）
  - `run.json`（メタ）
  - `report.md`（短い所見）
  - `train_code.py`（実行可能な最終コード／ノートブックコピー）
  - `experiments/<EXP_ID>/<exp_id>-artifacts/`（軽量物: plots, metrics, oof, **foldごと成果物＋全体集計成果物**）
- モデル本体は `artifacts/models/<EXP_ID>/`（Git 管理外を推奨）

5) インデックス更新とコミット（Where / Cmd）
- Where: Codespaces / ローカル
- Cmd:
  ```bash
  python scripts/update_experiment_index.py
  git add experiments/<EXP_ID> docs/scoreboard.md experiments/INDEX.md
  git commit -m "EXPxxx: record results"
  git push
  ```

習慣化ルール（毎回）
- 実験完了時に上の必須ファイルを `experiments/<EXP_ID>/` に置く
- 週1回 `python scripts/update_experiment_index.py` を実行すれば十分

{
  "experiment_id": "EXPXXXG",
  "title": "短いタイトル（例: tfidf_cleanup）",
  "system": "G",
  "created_at": "2025-09-02T00:00:00Z",
  "status": "completed",
  "git_commit": "TBD",
  "cv": {"mean": null, "std": null, "folds": []},
  "leaderboard": {"public_lb": null, "submission_id": null},
  "artifacts": {"models_dir": "artifacts/models/EXPXXXG", "experiment_dir": "experiments/EXPXXXG"},
  "notes": "短い所見・次アクション"
}

# 実験完遂チェックリスト（最小）

- [ ] Issue or README の Next Action に目的を記載
- [ ] （任意）`python scripts/reserve_experiment.py` で EXP_ID を予約
- [ ] 学習を実行し、モデルと metrics を出力（`artifacts/models/<EXP_ID>/`）
- [ ] 推論を実行し、`<WORKING_DIR>/submission.csv` を生成して提出
- [ ] `experiments/<EXP_ID>/run.json` を更新（CV/Leaderboard/commit）
- [ ] `experiments/<EXP_ID>/report.md` に短い所見を書く（CV, LB, Gap）
- [ ] `experiments/<EXP_ID>/train_code.py`（または notebook コピー）を保存
- [ ] 軽量成果物（plots, oof, metrics.json、**foldごと成果物＋全体集計成果物**）を `experiments/<EXP_ID>/<exp_id>-artifacts/` に保存
- [ ] Codespacesで `python scripts/update_experiment_index.py` を実行（週次でも可）
- [ ] git commit / push して Issue を Close