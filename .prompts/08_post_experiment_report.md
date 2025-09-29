# ポスト実験・自動記録マスタープロンプト

あなたは私のリポジトリ運用アシスタントです。以下の「入力」を受け取り、差分パッチと生成ファイルを作成してください。出力はすべて markdown コードブロックで示し、各ファイルには `// filepath: <path>` を先頭に付けてください。既存ファイルの変更は最小差分で行い、参照リンクは相対パスで示してください。

【入力（必須）】
- EXP_ID:
- Date: YYYY-MM-DD
- Git commit SHA:
- CV:
  - oof_auc:
  - fold_aucs: [...]
- Leaderboard:
  - public_lb:
  - submission_id:
- Artifacts:
  - models_dir: artifacts/models/<EXP_ID>  # 廃止予定: Dataset slugで代用
  - experiment_dir: experiments/<EXP_ID>
- Optional: code_dataset_slug, model_dataset_slug, env_versions, notes (短文)

【出力要求】
1) run.json を `experiments/<EXP_ID>/run.json` に作成（fields: experiment_id, title, date, status, git_commit, cv, leaderboard, artifacts, model_dataset_slug, dataset_uri, models_location, notes）
2) Run Card / short report を `experiments/<EXP_ID>/report.md` と `experiments/<EXP_ID>/README.md`（Run Card 見出し）として作成
3) README.md に該当行（短評1行）を追記するための差分パッチ（既存フォーマットに合わせる）
4) docs/scoreboard.md に詳細セクションを追記する差分パッチ（テンプレ準拠）
5) PR タイトルと本文の草案（英日両方、短文）
6) `python scripts/update_experiment_index.py` を実行する旨の指示と、実行後に想定される `experiments/INDEX.md` の差分例を提示
7) experiments/<EXP_ID>/checklist.md（完了チェックリスト）を作成

【出力フォーマット（厳守）】
- すべてのファイルは4バッククォートで囲んだ `markdown` ブロックにし、先頭に `// filepath: <path>` 行を付けること。
- 参照は相対リンクで記述（例: [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md)）。
- 自動で埋められない値（例: submission_id, git_commit が "TBD" の場合）は "TBD" と記載し、ユーザに入力を促す短い注記を添える。

【注意】
- Public LB とコミット SHA はユーザが提供する必要がある（AIは提出履歴を自動で読むことができないため）。
- 生成物は必ず `experiments/<EXP_ID>/` に保存すること（軽量ファイルのみ git 管理、モデルはKaggle Datasetに配置）。
- run.jsonはfinalize_and_publish.py実行時に対話式で`model_dataset_slug`を追加し、`dataset_uri`と`models_location`を自動設定する。

完了したら、生成されたパッチをユーザに提示してください。