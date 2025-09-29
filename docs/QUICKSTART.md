# QUICKSTART（セットアップ/学習/予測）

本ドキュメントは使用方法に特化しています。実験のスコア・命名規則・戦略・日誌は README を参照してください。

## セットアップ
```bash
pip install -r requirements.txt
```

## 学習（TF-IDF ベースライン）
```bash
python train.py --config config/tfidf_baseline.yaml
```

## 予測
```bash
python predict.py --config config/tfidf_baseline.yaml --test test.csv --output submission.csv
```

## Kaggle での利用
- コード一式を Notebook にアップロード、またはコード専用 Dataset を作成して Add Data してください。
- 詳細手順は次を参照: [docs/CODE_DATASET_GUIDE.md](CODE_DATASET_GUIDE.md)