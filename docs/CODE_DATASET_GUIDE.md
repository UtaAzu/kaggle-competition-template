# Kaggleでの「コード専用Dataset」運用ガイド

このガイドは、学習/推論コード（train.py, predict.py, src/, configs/）をKaggleの複数ノートブックから再利用するための手順です。モデル成果物（学習済みベクトライザやモデル）は別の「成果物Dataset」として扱います。

## 方法A: Notebookの「Code」タブに一度アップロード（最も簡単）
1. Kaggle Notebook右側の「Code」タブ → Upload で以下をアップロード
   - train.py, predict.py, src/ フォルダ, configs/ フォルダ
2. 以降、そのノートブックでは次のセルだけで実行可能
   ```
   !pip -q install pyyaml
   !python train.py --config configs/tfidf_baseline.yaml
   ```

## 方法B: コード専用Datasetを1つ作成して使い回す（複数ノートブックで便利）
1. 本リポから以下のファイル/フォルダを1つのフォルダにまとめる（例: code-dataset/）
   - train.py, predict.py, src/, configs/
2. Kaggleの「Create Dataset」からそのフォルダをアップロード（名前例: jigsaw-code-scaffold）
3. 任意のノートブックで「Add Data」→ そのDatasetを追加し、先頭セルでブートストラップ
   ```
   !cp -r /kaggle/input/jigsaw-code-scaffold/src ./src
   !cp -r /kaggle/input/jigsaw-code-scaffold/configs ./configs
   !cp /kaggle/input/jigsaw-code-scaffold/train.py .
   !cp /kaggle/input/jigsaw-code-scaffold/predict.py .
   ```
4. 実行
   ```
   !pip -q install pyyaml
   !python train.py --config configs/tfidf_baseline.yaml
   ```

### Kaggle CLIでの自動化（任意）
- Kaggle CLI が使える環境なら、本リポの `tools/code-dataset/metadata.json` を編集して、次で公開できます。
  ```
  kaggle datasets create -p <code-dataset-dir>
  kaggle datasets version -p <code-dataset-dir> -m "update code"
  ```
  CLIが使えない場合はUIで十分です。

## 成果物Datasetの作り方（モデル/ベクトライザ等）
- 学習完了後に `/kaggle/working/outputs/<exp_name>` を「Create Dataset」で公開
- 推論ノートブックでは、そのDatasetを Add Data して `configs/...yaml` の `artifacts.dir` を差し替えます。

## トラブル回避のTips
- ノートブックからは `%run` ではなく `!python` で実行
- `exp_name` を必ず設定し、出力やDataset名に含める
- `cv_summary.json` と `metadata.json` に `exp_name` が入るため、追跡が容易です