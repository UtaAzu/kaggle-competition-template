# 汎用Kaggle用・自己完結型・推論パイプライン構築ガイドライン（強化版）

あなたは私の専属デプロイメント・エンジニアです。以下の仕様に従い、Kaggle Notebook 上で単一ファイルとして実行可能な自己完結型推論スクリプト `inference_kaggle.py` を作成してください。目的は「訓練コードと完全整合し、オフライン環境（インターネットOFF）でも動作し、メモリ制約に配慮した安全な推論」を保証することです。

重要: 不明点があれば必ず停止して明確な確認質問を投げてください。推測で実装を進めてはなりません。

== 高レベル要件 ==
- 必須：`Config`クラスで主要設定を一元管理すること（下記必須項目を含む）。
- 訓練側の `create_features` 関数は必ず [`train.py`](train.py) から「一字一句コピー」して推論スクリプト内に含めること（整合性保証）。
- モデル命名規約: 0-based fold、`model_fold_0.joblib` ... `model_fold_4.joblib`（N_SPLITS=5）。前処理器は `preprocessor_fold_{fold}.joblib`。
- 提出フォーマット: `submission.csv` を出力、列は `row_id,rule_violation`（`row_id` がなければ 0-based 連番で作成）。
- アンサンブル統合: デフォルトは「確率の単純平均」。
- メモリ配慮: 逐次ロード＋バッチ予測をデフォルト実装とする（バッチサイズ設定必須）。
- 障害時: アセット欠損、pip 失敗、予測形状不整合はログ出力し非0終了（exit code != 0）。
- 出力: `submission.csv` に加え `metadata.json` / `run.json` を出力し、使用アセット・ライブラリバージョン・読み込んだファイルを記録すること。
- Smoke test: `--test sample_test.csv` で動く簡易検証モードを実装すること（参考: [`sample_test.csv`](sample_test.csv)）。

== 必須項目（Config クラスに含める） ==
- EXP_ID (例: "EXP006G_preproc_enhanced")
- MODEL_DATASET_SLUG (Kaggle dataset slug, 例: "jigsaw-exp006g-dataset")
- CUSTOM_LIB_DATASET_SLUG (or None)
- CUSTOM_LIBS_TO_INSTALL (list[str], may be [])
- N_SPLITS (default 5), BATCH_SIZE (default 1024), RANDOM_STATE
- MODEL_BASE_PATH 動的構築ロジック（Kaggle / Local 切り替え）
- LOG_LEVEL, DEBUG フラグ

== 推奨フロー（実装フェーズ） ==
1. 環境判定（Kaggle / Local）→ パス組立て。
2. Offline libs の検査・安全インストール（`CUSTOM_LIB_DATASET_SLUG` が指定されれば、whl 存在確認→ pip offline install）。失敗時は明示的に abort。
3. 訓練コードから `create_features` をコピーし、そのまま使用（必須）。コピー済であることを hash（簡易チェックサム）で検証するオプション実装を推奨。
4. `load_artifacts` で `model_fold_{i}.joblib` と `preprocessor_fold_{i}.joblib` を読み込み可能にする。モデルの存在チェックと読み込み失敗時のエラー処理を厳格化。
5. 逐次ロード + バッチ推論を実装（foldごとにモデルをロード→全テストをバッチ処理で predict_proba → 加算 → アンロード）。
6. 予測統合は単純平均で出力。出力前に NaN/inf チェックを行う。
7. `submission.csv` と `metadata.json` / `run.json` を出力。`metadata.json` には `MODEL_DATASET_SLUG`, `CUSTOM_LIBS_TO_INSTALL`, loaded model filenames, Python/major libs versions を含める。
8. smoke-test と `--test` オプションを用意し、CI/手元検証を容易にする。

== 安全・品質チェック（必須） ==
- Dataset 内に求める whl が存在するかを検査（存在しない場合は明示エラー）。
- モデルファイルは 0..N_SPLITS-1 が揃っているか確認。
- 各 fold の予測は shape チェック、NaN/inf チェックを行う。
- 実行ログはコンソールと `inference.log` に出力。
- 失敗時は sys.exit(1/2/3) で明確終了コードを返す。

== 付録: 推奨コード断片（実装時にそのまま組み込むこと） ==
- 安全オフラインインストール（Environment Setup）:

```python
def install_custom_libs(cfg, logger):
    if not cfg.CUSTOM_LIB_DATASET_SLUG:
        logger.info("No CUSTOM_LIB_DATASET_SLUG specified, skipping offline install.")
        return
    base = Path("/kaggle/input") / cfg.CUSTOM_LIB_DATASET_SLUG
    if not base.exists():
        logger.error(f"Custom libs dataset not found: {base}")
        raise SystemExit(2)
    whls = list(base.glob("*.whl"))
    logger.info(f"Found {len(whls)} wheel(s) in {base}")
    for pkg in cfg.CUSTOM_LIBS_TO_INSTALL:
        matched = [p for p in whls if pkg in p.name]
        if not matched:
            logger.error(f"Required package wheel for {pkg} not found in {base}")
            raise SystemExit(3)
    cmd = ["pip", "install", "--no-index", "--find-links", str(base)] + cfg.CUSTOM_LIBS_TO_INSTALL
    import subprocess
    subprocess.check_call(cmd)
```

- 逐次ロード＋バッチ推論（Prediction Loop の例）:

```python
def batch_predict(test_texts, cfg, logger, batch_size=1024):
    import gc
    from joblib import load
    n = len(test_texts)
    preds = np.zeros(n, dtype=float)
    for fold in range(cfg.N_SPLITS):
        model_path = cfg.model_base / cfg.model_fname_tmpl.format(fold=fold)
        prep_path = cfg.model_base / cfg.prep_fname_tmpl.format(fold=fold)
        logger.info(f"Loading fold {fold} model: {model_path}")
        model = load(model_path)
        prep = load(prep_path)
        for i in range(0, n, batch_size):
            batch_idx = slice(i, min(i+batch_size, n))
            X_batch = [prep(t) for t in test_texts[batch_idx]]
            proba = model.predict_proba(X_batch)[:,1]
            preds[batch_idx] += proba
        del model, prep
        gc.collect()
    preds /= cfg.N_SPLITS
    # post-checks
    if not np.isfinite(preds).all():
        logger.error("Non-finite values in predictions")
        raise SystemExit(4)
    return preds
```

== ドキュメント出力（必須） ==
- `metadata.json` に以下を記録:
  - EXP_ID, MODEL_DATASET_SLUG, CUSTOM_LIBS_TO_INSTALL, loaded model filenames, timestamp, Python / sklearn / numpy / pandas バージョン。
- `run.json`（軽量）を生成し、`experiments/<EXP_ID>/run.json` へのコピー手順を README に明記。

== 確認質問（実装前に回答必須） ==
1. N_SPLITS=5 固定で良いか（はい=実装を進めます）  
2. BATCH_SIZE のデフォルト（例: 1024）で良いか？  
3. CUSTOM_LIBS_TO_INSTALL に含めるパッケージ一覧を確定できますか？（指定なければ空リスト）  
4. model artifacts の Kaggle Dataset 内の相対パスは `artifacts/models/{EXP_ID}/` で良いか？

== 出力形式（期待） ==
- 単一ファイル `inference_kaggle.py`（上記全機能を実装）  
- 簡易 smoke-test セル（Kaggle Notebook 用）: `!python inference_kaggle.py --test sample_test.csv --out /kaggle/working/submission.csv`  
- `metadata.json` / `run.json` / `submission.csv` を出力し、`experiments/<EXP_ID>/` に軽量成果物を収める指示を README に含めること（大きなモデルは `artifacts/models/` に置く）。

== 参照（実装時に必ず参照） ==
- 訓練コードの `create_features`: [`create_features`](train.py) in [train.py](train.py)  
- 既存推論 helper: [predict.py](predict.py)  
- 実験最終化スクリプト: [scripts/finalize_and_publish.py](scripts/finalize_and_publish.py)  
- 実験チェックリストテンプレ: [experiments/_template/CHECKLIST.md](experiments/_template/CHECKLIST.md)  
- 実験ワークフロー: [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md)  
- サンプルテスト: [sample_test.csv](sample_test.csv)  
- EXP006G のチェックリスト例: [experiments/EXP006G/checklist.md](experiments/EXP006G/checklist.md)

---

完了したら、上記確認質問に回答してください。回答を受け次第、実装可能な `inference_kaggle.py` と Kaggle 用 smoke-test セルを生成します。