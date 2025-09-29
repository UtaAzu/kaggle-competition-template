# 汎用Kaggle用・自己完結型・訓練パイプライン作成ガイドライン（成果物一元化改訂）

---

## 【あなたの役割】

あなたは、私のための専属**フレームワーク・アーキテクト**です。あなたの任務は、私のリポジトリにある現在の`train.py`（あるいは`.ipynb`）を、私が定義する「9つの論理ブロック」を完全に満たす、**最終的にKaggle Notebook上で単一ファイルとして実行可能**な、構造化された訓練パイプラインへと、全面的にリファクタリングしてください。

---

## 【最重要ルール：不明点の確認義務】

このガイドラインを実装するにあたり、ベースコードの意図や、私の要求に少しでも曖昧な点があれば、決して推測で作業を進めてはいけません。必ず、作業を中断し、私に具体的な確認質問を行ってください。(例：「`Config`クラスに`DEBUG`モードを追加しますが、`DEBUG=True`の時は、訓練データを最初の1000行にサンプリングする仕様でよろしいでしょうか？」)

---

## 【分析対象】

- **ベースコード:** リポジトリのルートにある`train.py`（あるいは`.ipynb`）の最新版。
- **設計思想:** このプロンプトで定義される、9つの論理ブロック。

---

## 【パイプラインに搭載すべき9つの論理ブロック】

最終的に生成する単一のPythonスクリプト (`train_kaggle.py`) は、以下の**9つの明確に分離された「論理ブロック」**で構成されていなければなりません。各ブロックの先頭には、`# === 1. Configuration ===` のような、明確なコメントヘッダーを付けてください。

---

### **0. 🏷️ メタ情報ブロック (Meta Information Block)**
- **要件:**
    - 実験番号（EXP_ID）、タイトル、目的、日付、著者、概要などのメタ情報を、スクリプトやノートブックの最上部に明記してください。
    - 例：
        ```python
        # === EXP004D: Qwen2.5-0.5B-Instruct LLM finetune (subreddit GroupKFold) ===
        # Date: 2025-09-17
        # Author: UtaAzu
        # Purpose: Fine-tune Qwen2.5-0.5B on subreddit-grouped CV for rule violation detection
        # Dataset: jigsaw-agile-community-rules (train.csv, test.csv)
        # Expected Outcome: CV OOF AUC > 0.6, Public LB improvement
        ```

---

### **1. 📝 設定ブロック (Configuration Block)**

- **要件:**
    - 実験に関する全てのパラメータを、単一の`Config`クラスで一元管理してください。
    - **必須パラメータカテゴリ** Configクラスには、少なくとも以下のカテゴリの情報を含めてください:
    - 実験識別子: EXP_ID, DESCRIPTION
    - パス設定: INPUT_DIR, OUTPUT_DIR
    - データ設定: TRAIN_CSV, TEST_CSV, ENCODING
    - 特徴量設定: TEXT_COLS, CONCAT_TEMPLATE
    - モデル設定: MODEL_NAME, LEARNING_RATE, N_SPLITS, RANDOM_STATE
    - 実行制御フラグ: DEBUG, TRAIN_FOLDS
    - **【環境自動適応】** この`Config`クラスには、**実行環境（Kaggle or Codespaces）を自動で検知**し、`INPUT_DIR`や`DEBUG`フラグを動的に切り替えるロジックを必ず含めてください。
        - **Kaggle環境の場合:** フルデータパス (`/kaggle/input/...`) を使用し、`DEBUG=False`とします。
        - **Codespaces/ローカル環境の場合:** ローカルパス (`./`) を使用し、`DEBUG=True`とします。

---

### **2. ⏱️ ロギングブロック (Logging Block)**

- **要件:** 実験の進捗を記録するための、シンプルな`Logger`クラス、または関数群を定義してください。ログはコンソールとテキストファイルの両方に出力されるようにしてください。
- **推奨:** 実行環境のバージョン情報（Python, numpy, pandas, sklearn, torch, transformers, datasets等）を`train.log`に記録すること。

---

### **3. 🔧 特徴量エンジニアリングブロック (Feature Engineering Block)**

- **要件:** 特徴量作成ロジックを、独立した`create_features(dataframe, config)`関数にまとめてください。

---

### **4. 🤖 モデル定義ブロック (Model Definition Block)**

- **要件:** モデルのアーキテクチャを定義する`build_model(config)`関数を定義してください。

---

### **5. 🎓 学習・評価関数ブロック (Training & Evaluation Functions Block)**

- **要件:**
    - CVの1 Fold分を学習・評価する`train_fold`関数を定義してください。
    - **【戻り値の定義 (デフォルト)】** この関数は、(学習済みモデル, 学習済み前処理器, このFoldでのOOF予測, このFoldの評価スコア) のタプルを返すことを基本仕様とします。  
      - 理由: 多数の既存実装（例: [`train.py`](train.py)、`experiments/*/notebooks`）がこの戻り値を想定しています。
    - **【LLM 特化注記（重要）】** Hugging Face 等の LLM を用いる場合はメモリ管理上の制約があるため、`train_fold` は次のいずれかを採用してください:
        - (A) デフォルトに従いモデルオブジェクトを返す（小モデル・メモリ余裕あり）、
        - (B) モデルを返さず辞書 `{fold:int, val_idx: array, val_probs: array, val_auc: float, model_dir: str}` を返す（推奨、LLM 系）。この場合は保存後に `del model; torch.cuda.empty_cache(); gc.collect()` を必須としてください。  
      - 理由: 多数の既存実装（例: [`train.py`](train.py)、`experiments/*/notebooks`）がこの戻り値を想定しています。
      - 実装は上記「デフォルト仕様」を維持しつつ、LLM 例外を明確に文章化すること。これにより既存ワークフロー（`scripts/save_oof.py`, `train.py` 等）との互換性を保てます。
    - **report_results** 関数を定義し、全Fold集約後の指標（overall OOF AUC, fold_aucs, mean/std など）を計算・ログ出力・返却してください。`report_results` の実装場所は「成果物保存ブロック」または「メイン実行ブロック」で良い旨を明示してください（保存処理と密接）。
      - 既存の LLM 実装例: [`src/pipelines/train_llm.py`](src/pipelines/train_llm.py), [`experiments/EXP004D/notebooks/jigsaw-exp004d.ipynb`](experiments/EXP004D/notebooks/jigsaw-exp004d.ipynb) を参照。

---

### **6. 💾 成果物保存ブロック (Artifact Saving Block)**

**要件:**
- **【v7方針：成果物一元化】**  
    - **推論・分析・再利用・Kaggle UI可視化すべての観点から、Config.OUTPUT_DIR（例：/kaggle/output/experiments/EXP_ID）直下に全成果物を集約保存することを必須**とします。
    - サブディレクトリ（models/fold_{i}/, tokenizer/ 等）は必ず OUTPUT_DIR配下で階層化すること。
    - **成果物の二重保存や冗長なコピー・ミラーリングは禁止**（artifacts/や/kaggle/working/へのミラー保存は不要）。
    - Kaggle Output UIで「Output」タブを開いたとき、OUTPUT_DIR配下の成果物が全て表示されることを最重視してください。

- 必須保存物:
  - モデル本体: model_fold_{fold_number}.safetensors または model_fold_{fold_number}.bin（models/fold_{i}/配下）
  - モデル設定: config.json
  - トークナイザー: tokenizer.json, vocab.json, merges.txt, tokenizer_config.json
  - OOF予測: oof.csv
  - 評価指標: metrics.json
  - 実行ログ: train.log
  - 設定ファイル: config.json
  - 実験メタ: run.json
    - **run.jsonはfinalize_and_publish.py実行時に対話式で`model_dataset_slug`を追加し、`dataset_uri`と`models_location`を自動設定する。**
  - checkpoint-*, optimizer.pt, scheduler.pt, rng_state.pth, training_args.bin などは保存しないこと（prune_excess_artifactsで削除）

- **save_oof必須要件:**  
    - oof.csv（row_id, oof_pred, rule_violation, group等）
    - metrics.json（oof_auc, fold_aucs, n_splits, seed等）
    - run.json（experiment_id, date, git_commit, cv.oof_auc, leaderboard.public_lb, model_dataset_slug等。提出後追記も可）
    - 出力パスは `OUTPUT_DIR/oof.csv` と `OUTPUT_DIR/submission.csv`（/kaggle/workingは非推奨、必ずOUTPUT_DIR）

---

### **7. 🔄 メイン実行ブロック (Main Pipeline Block)**

- **要件:**
    - これまで定義した全ての関数やクラスを呼び出し、データ読み込みから、CV実行、結果報告、成果物保存まで、**パイプライン全体の流れをオーケストレーション**する、メインの実行ロジックを記述してください。
    - このブロックが、スクリプトの「司令塔」となります。
    - 【処理フローの指定】
    - データを読み込み、create_featuresを呼び出す。
    - CVのループを開始する。
    - ループ内でtrain_foldを呼び出し、戻り値（モデル、OOF等）をリストに格納する。
    - ループ終了後、全Foldの結果をまとめる。
    - 収集した全アセット（モデルのリスト、結合したOOFデータフレームなど）を引数としてsave_artifactsを呼び出す。
    - 最終スコアをreport_resultsで報告する。

---

### **8. 🚀 実行トリガーブロック (Execution Trigger Block)**

- **要件:** スクリプトの最後に、`if __name__ == "__main__":`ブロックを設け、そこからメインの実行ロジックを呼び出すようにしてください。

---

### **9. 📊 環境バージョン記録ブロック (Environment Version Logging Block)**

- **要件:** 実験の再現性向上のため、実行環境のバージョン情報を記録してください。
    - 主要ライブラリのバージョン記録
    ```python
    libraries = ['scikit-learn', 'numpy', 'pandas', 'torch', 'transformers', 'datasets']
    for lib in libraries:
        try:
            version = __import__(lib).__version__
            logger.info(f"{lib}: {version}")
        except (ImportError, AttributeError):
            try:
                if lib == 'scikit-learn':
                    import sklearn
                    logger.info(f"sklearn: {sklearn.__version__}")
            except ImportError:
                logger.info(f"{lib}: not available")
    ```

---

## 【最終的な成果物】

- 上記の**9つの論理ブロック**で美しく構造化された、**単一の、自己完結したPythonスクリプト (`train_kaggle.py`)**。
- このスクリプトは、Kaggle Notebook上で、**`!python train_kaggle.py`**と実行するだけで、エラーなく、最後まで完走するものでなければならない。
- **成果物は必ず OUTPUT_DIR（例：/kaggle/output/experiments/EXP_ID）直下・配下に集約し、Kaggle Output UIで一目で全て確認・ダウンロードできるようにすること。**

---

## 【環境バージョン記録の推奨実装】

実験の再現性向上のため、実行環境のバージョン情報を記録することを強く推奨します。以下のコードを**ロギングブロック**内に含めることを検討してください:

```python
def log_environment_versions():
    """環境バージョン情報をログに記録"""
    import sys
    import platform
    
    logger.info("=== Environment Information ===")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    #...
```

---

## 【確認質問（実装前に回答必須）】

1. N_SPLITS=5 固定で良いか（はい=実装を進めます）  
2. BATCH_SIZE のデフォルト（例: 1024）で良いか？  
3. CUSTOM_LIBS_TO_INSTALL に含めるパッケージ一覧を確定できますか？（指定なければ空リスト）  
4. model artifacts の Kaggle Dataset 内の相対パスは `artifacts/models/{EXP_ID}/` で良いか？

== 出力形式（期待） ==
- 単一ファイル `train_kaggle.py`（上記全機能を実装）  
- 簡易 smoke-test セル（Kaggle Notebook 用）: `!python train_kaggle.py --test sample_test.csv --out /kaggle/working/submission.csv`  
- 完了したら、上記確認質問に回答してください。回答を受け次第、実装可能な `train_kaggle.py` を生成します。