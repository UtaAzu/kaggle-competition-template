# Kaggle Universal Starter Architect: "The Shapeshifter"

ポンポさん直伝の**完全自律型・データタイプ自動分岐プロンプト**です。AI自身がデータを判定し、テーブル/画像/テキストに応じた専門家に変身して、そのコンペ専用のスターターノートブックを1発で出力します。

---

## 🎬 使い方

コンペが始まったら、以下のプロンプトと一緒に `docs/01_overview.md`、`docs/02_data_description.md`（もしあれば `docs/03_original_data.md`）を投げるだけ。AIが自動判定して最適なコードを生成します。

---

## 📜 Universal Starter Architect Prompt (Final Draft)

```markdown
# Role: Kaggle Grandmaster Pipeline Architect

## Goal
コンペティション資料（`overview.md`, `data_description.md`）に基づき、**「Kaggle Notebook環境でそのまま動作する、信頼性の高いスターターコード」** を単一のPythonブロックで作成してください。

## Constraints (絶対遵守)
1. **Output:** Pythonコードブロック **1つのみ**。解説や分析はすべてコード冒頭の `Docstring` に記述すること。
2. **Environment:** `/kaggle/input/` パスを前提とする。`argparse` は使用せず、`class CFG` で管理する。
3. **Quality:** エラーなく完走し、必ず `submission.csv` を出力すること。

## Process: Route Selection Logic
データタイプを分析し、以下のいずれかのルートを選択して実装してください。

### 🛤️ Route A: Tabular (CSV/Parquet)
- **Target:** 構造化データ。
- **Model:** LightGBM or XGBoost (Single Model).
- **CV:** StratifiedKFold (分類) / KFold (回帰).
- **Feature Engineering:** データ特性に合わせて **「1つだけ」** 実装する。
  - (例) 匿名機能・Synthetic → `Count Encoding`
  - (例) 数値のみ → `Binning` (q=10)
  - (例) 時系列 → `Date Features`
  - **選択理由をコードコメントで明記すること**
- **Drift Check:** `CFG.perform_drift_check = True` の場合、簡易Adversarial Validationを実行するセルを含める。
  - Train vs Test で OOF AUC を計算し、Drift有無を判定
  - Original Data があれば Train vs Original も実施
- **Model Parameters:** `max_depth=4〜6` の浅い木で汎化性能を重視

### 🛤️ Route B: Image (Files + Labels)
- **Target:** 画像データ。
- **Lib:** `timm`, `albumentations`, `torch`.
- **Model:** `tf_efficientnet_b0_ns` などの軽量モデル。
- **Training:** `torch.cuda.amp.GradScaler` (AMP) を使用した高速学習ループ。
- **CV:** StratifiedKFold.
- **Augmentation:** Flip, Rotation, ColorJitter程度のBasic Augmentation（過度な複雑化回避）

### 🛤️ Route C: Text (NLP)
- **Target:** テキストデータ。
- **Lib:** `transformers`, `tokenizers`, `torch`.
- **Model:** `microsoft/deberta-v3-xsmall` などの軽量モデル。
- **Tokenization:** `AutoTokenizer` を使用、max_length=512。
- **Training:** `torch.cuda.amp.GradScaler` (AMP) を使用。
- **CV:** StratifiedKFold.

## Code Structure (Template)
コードは以下の順序で記述してください：

1. **Docstring Header:**
   - 選択したルート (A/B/C) とその理由（3-5行）
   - ターゲットの分析（分類/回帰/評価指標）
   - 採用したCV戦略と特徴量エンジニアリングの意図
   
2. **Imports & Setup:** 
   - 必要なライブラリのみimport（選択したルート専用）
   - `seed_everything()` 関数を含む
   
3. **Config:** 
   - `class CFG` で一元管理（seed, n_folds, params, paths）
   - `CFG.perform_drift_check` などのスイッチを含む
   
4. **Data Loading:** 
   - `/kaggle/input/` からの読み込み
   
5. **Minimal EDA:** 
   - 2-3つのセルでデータ理解を示す
   - (例) Target分布、欠損値確認、Train/Test比較
   
6. **Drift Check (Route A のみ、オプション):**
   - `CFG.perform_drift_check = True` の場合に実行
   - Adversarial Validation（Train vs Test、Train vs Original）
   
7. **Preprocessing / Dataset:** 
   - ルートに応じた処理
   - Route Aなら特徴量エンジニアリング（1つだけ）
   - Route B/Cなら Dataset/DataLoader 実装
   
8. **Model & Training Loop:** 
   - OOFスコアを計算・表示
   - Route B/Cは AMP (GradScaler) を使用
   
9. **Inference & Submission:** 
   - `submission.csv` の保存

## Philosophy
- **"Correctness over Complexity":** 複雑なモデルより、CVとLBが一致する堅牢なパイプラインを優先せよ。
- **"Show, Don't Just Tell":** 数値だけでなく、可視化で説得力を持たせよ（2-3セル）。
- **"Simple is Strong":** 特徴量もモデルも、シンプルに1つに絞れ。

---

**さあ、資料を読み込み、最適なスペシャリストとしてコードを生成してください。Action!!**
```

---

## 🎬 解説：この「脚本」の意図（ポンポさん承認版）

1. **究極のUX：Copy 1回で完結** → 判定理由もDocstringに。ユーザーはコードブロックをコピペするだけ。
2. **スイッチ式Drift Check** → `CFG.perform_drift_check = True` で重い診断をオンオフ可能。S5E12の教訓を標準装備しつつ軽量化。
3. **AMP（Mixed Precision）標準化** → 画像・テキストは `GradScaler` 必須。現代のKaggleでは常識。
4. **Kaggle環境前提を明記** → `/kaggle/input/` パス、argparse禁止、CFG管理。事故を未然に防ぐ。
5. **Code Structureで迷わせない** → 1→9の順序を明示。AIが構造を守りやすく、出力コードが美しくなる。
6. **自動分岐で出力をクリーンに** → AIがデータタイプを判定し、該当ルートのコードだけを吐く。画像コンペでGBDTのimportが混ざることはない。

この1枚をコンペ開始時に `docs/` 内のmdファイルと一緒に投げるだけで、AIが勝手に「今回はテーブルだな→LightGBM+Count Encoding」と判断し、Copyボタン1回で動くスターターを出力します💪✨

---

## 🎞️ 次のアクション

コンペが始まったら：
1. `docs/01_overview.md`, `docs/02_data_description.md` を用意
2. 上記プロンプトを投げる
3. 出力されたコードをKaggle Notebookにペースト
4. Run All → `submission.csv` が出力される
5. 結果を共有して改善サイクルへ

**お疲れ様でしたっ、せんぱい！次のコンペで最高のスタートダッシュを決めましょう🎬✨**
