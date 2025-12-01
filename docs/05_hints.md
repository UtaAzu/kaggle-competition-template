# 運用Tips・クイックリファレンス

---

### 📚 このドキュメントについて

**対象読者**: 実装・検証を進めるエンジニア  
**主な内容**:
- チェックリスト（提出前、実験終了時）
- 実装ガイドライン
- よくある失敗パターン

**関連ドキュメント**:
| ドキュメント | 用途 | 対象者 |
|:---|:---|:---|
| **04_knowledge.md** | 理論・背景知識 | 研究者・分析者 |
| **05_hints.md** (ここ) | 実装チェックリスト | エンジニア・実装者 |
| **11_parameter_tuning_knowledge.md** | 実戦チューニング戦略 | チューニング担当者 |

---

## 実験終了時の必須チェック（1T総括からの教訓）

### 提出前の健全性チェック
- [ ] 行数 = テスト画像数（sample_submission.csvと一致）
- [ ] case_id に重複なし、順序一致
- [ ] authentic 表記は厳密に `"authentic"`（大文字小文字・前後空白なし）
- [ ] RLE は `[start length ...]` 形式、数値は int 型（numpy.int64 は str変換）
- [ ] 空マスク時のフォールバック実装済み（`if combined.sum() == 0: return "authentic"`）
- [ ] validate_submission_format 関数で形式検証を必ず実施

### CV/LBギャップの健全性確認
- [ ] CV/LBギャップは ±0.01～0.03 が健全（0.000は過学習または検証不足の兆候）
- [ ] OOF を全画像分保存し、fold別スコアを記録（1 foldのみはデータ効率20%）
- [ ] 後処理パラメータ（閾値/min_area）はCVで最適化し、固定値禁止

### 成果物の標準配置
- [ ] `experiments/<EXP_ID>/<exp_id>-artifacts/` に以下を保存:
  - `oof.csv`: per-image予測（case_id, gt, pred, f1, best_threshold）
  - `metrics.json`: fold別スコア + mean/std + 最適閾値 + 分離指標
  - `run.json`: EXP_ID, CV, LB, git_commit, submission_id, 分離指標
  - `metadata.json`: 環境バージョン（Python/torch/torchvision等）
- [ ] モデルは Kaggle Dataset化し、slugを run.json に記録（Git管理外）

### 実験記録の自動化
- [ ] Codespaces で `python scripts/update_experiment_index.py` 実行（週次可）
- [ ] [`docs/scoreboard.md`](docs/scoreboard.md) に詳細セクション追記（テンプレ準拠）
- [ ] [`README.md`](README.md) のスコアボードに1行追記（要約のみ）

---

## モデル嗜好に基づく実装ガイドライン

**関連ドキュメント**:
- 理論的背景 → [`docs/04_knowledge.md#dinov2-単体-vs-dinov2--sam2-のパラメータ嗜好分析`](04_knowledge.md#dinov2-単体-vs-dinov2--sam2-のパラメータ嗜好分析)
- SAM2 に関する決定 → [`docs/04_knowledge.md#sam2-推論採用の中止決定`](04_knowledge.md#sam2-推論採用の中止決定) ⚠️ **重要**
- チューニング実践 → [`docs/11_parameter_tuning_knowledge.md`](11_parameter_tuning_knowledge.md)

### パラメータ調整前のモデル分析チェックリスト

- [ ] **このモデルは単体か、多段階パイプラインか？**
  - 単体（v19推定）→ aggressive パラメータ（ALPHA_GRAD=0.35）
  - 多段階（v3 + SAM2）→ conservative パラメータ（ALPHA_GRAD=0.30）
  - ⚠️ **SAM2 を推論で使うのは効果なし** (Top Kaggler 検証済み)
  
- [ ] **「新しい技術を使いたい」欲に注意**
  - SAM2 は推論の refiner ではなく、オフラインデータ生成用
  - 技術の本質を理解してから導入
  - 検証なしに導入しない（失敗例多数）
  
- [ ] **Refinement stage がある場合、入力の「信頼度」を確認**
  - false positive が多い → ノイズも refine される → 精度低下
  - false positive が少ない → 信頼できる候補のみ → 精密化効果的
  
- [ ] **パラメータ変更時は「モデル構成全体を固定」してから調整**
  - 複数要素（TTA, 解像度, backbone）の同時変更は避ける
  - 1要素ずつ validate で検証（分離指標 forged_nonempty_ratio, authentic_fp_ratio 監視）

- [ ] **他モデルの値をそのまま移植しない**
  - v19 の ALPHA_GRAD=0.35 が v3+SAM2 でも最適とは限らない（v4で実証）
  - 本モデルで validate 検証が必須

### パラメータ嗜好別の Ensemble 設計

- [ ] **異なる嗜好を持つモデルを組み合わせる**
  - aggressive（検出力重視）+ conservative（精度重視）
  - 両者の相補的な強みで LB スコア向上
  - 例: v19 (0.35) + v3 (0.30) で LB 0.32+ 期待

- [ ] **パラメータグリッドサーチは「嗜好が異なるモデル」を複数準備**
  - 単一パラメータだけでなく、複数の「好み」を網羅
  - 後で ensemble 候補を柔軟に選べる

---

## validate分析・分離指標重視の追加ヒント

- [ ] mean_f1だけでなく「forged_nonempty_ratio」「authentic_fp_ratio」を必ず記録・提出判断に使う
- [ ] forged_nonempty_ratio < 0.2、authentic_fp_ratio > 0.3 の場合は提出不可（退化解ガード）
- [ ] metrics.json/run.jsonに分離指標（forged_nonempty_ratio, authentic_fp_ratio）を必ず記録
- [ ] 退化解（全authentic提出・F1=0/1集中）はサマリー表・ヒストグラムで即座に検知
- [ ] validateノートでヒストグラム・分布分析（F1分布・極端値の原因調査）を毎回実施し、モデル・後処理の弱点を早期発見する
- [ ] アンサンブルは分離指標の補完関係を重視し、単純平均だけでなく論理和・stacking等も検討
- [ ] migration_planの成功基準・撤退条件に分離指標の下限・上限を明記

---



## よくある失敗パターンと対処法（1T v1～v9からの教訓）

### 失敗1: Scoring Error（v1～v4全滅）
**原因**:
**対処**:

**原因**:


## スコア改善の優先順位（P0～P2）

### P0（即座に実施）
- [ ] **入力解像度↑**: 256→384 or 512（Mask R-CNNの min/max_size整合）
- [ ] **後処理最適化**: score閾値τ, mask二値化σ, min_areaをCVでグリッド探索
- [ ] **全fold学習**: USE_FIRST_FOLD_ONLY=False → 5-fold平均でアンサンブル
- [ ] **分離指標健全化**: forged_nonempty_ratio ≥ 0.2, authentic_fp_ratio ≤ 0.3

### P1（中期）
- [ ] **事前学習有効化**: ResNet50 pretrained=True + 段階的unfreeze
- [ ] **TTA**: H/V flip のマスク論理和

### P2（長期）
- [ ] **モデル多様化・アンサンブル**: Mask R-CNN/U-Net/クラシカル手法の組み合わせ
- [ ] **特徴量拡張・meta-learner**: ELA, DCT, SHAP, stacking
---
## スコア改善のヒント（CMFD・クラシカル手法・後処理耐性）

### P0（即座に検討すべき戦略）

- **損失関数の見直し**  
  - BCE Loss単体では偽造領域が無視されやすい。Dice LossやBCE+DICE、Focal Lossなど「偽造領域を重視する損失関数」への切り替えが有効。
  - 例: `loss = 0.5 * BCE + 0.5 * DiceLoss`

- **DCT特徴＋クラシカル手法の活用**  
  - 画像をブロック分割→DCT特徴抽出→符号ベクトル化→kd-tree最近傍探索で重複領域検出
  - NN予測maskとアンサンブル、または後処理でDCTベースの重複領域を追加
  - 参考: [Copy-move forgery detection technique with DCT](https://www.sciencedirect.com/science/article/abs/pii/S2214212619307343)

- **幾何変換・ノイズ・明度変化への耐性強化**  
  - Augmentationで回転・スケール・ノイズ・明度変化を強化
  - mask生成・判定時に「繰り返し構造」や「ラベル・グリッド」を誤検出しない工夫（context-aware判定）

- **クラシカル手法とのアンサンブル**  
  - UNet等NN予測mask＋SIFT/DCT/テンプレートマッチ等のクラシカル手法による重複領域検出を後処理で組み合わせる

### P1（中期以降の戦略）

- **context-aware判定**  
  - 画像全体の構造やラベル情報も考慮し、自然な繰り返し（バーグラフ・細胞・ラベル）と偽造の区別を学習・判定

---

## 実験終了スクリプトの使い方

### 手動記録（リモート環境）
```bash
# 1. run.json を手動作成（テンプレ: experiments/_template/run.json）
# 2. report_1T_summary.md を作成
# 3. Codespacesで以下を実行
python scripts/update_experiment_index.py
git add experiments/EXP001T docs/scoreboard.md experiments/INDEX.md
git commit -m "EXP001T: v9 completion and series summary"
git push
```

### 自動記録（ローカル環境）
```bash
# Git commit SHA取得
git rev-parse --short HEAD

# 自動finalize（対話式）
python scripts/finalize_and_publish.py EXP001T_v9

# インデックス更新
python scripts/update_experiment_index.py
```

---

## 参考リンク

- 実験ワークフロー: [`docs/EXPERIMENT_WORKFLOW.md`](docs/EXPERIMENT_WORKFLOW.md)
- 提出形式要件: [`docs/LUC_submission_format.md`](docs/LUC_submission_format.md)
- スコアボード: [`docs/scoreboard.md`](docs/scoreboard.md)
- ナレッジ集: [`docs/04_knowledge.md`](docs/04_knowledge.md)
- 1T総括レポート: [`experiments/EXP001T/report_1T_summary.md`](experiments/EXP001T/report_1T_summary.md)
- 悪魔の代弁者レビュー: [`experiments/EXP001T/devils_advocate_review_1T.md`](experiments/EXP001T/devils_advocate_review_1T.md)

---

### スコア改善・健全性チェックの追加ヒント

- [ ] validate分析で「forged_nonempty_ratio（偽造検出率）」「authentic_fp_ratio（誤検出率）」を必ず記録し、両指標のバランスが取れているか確認する
- [ ] forged画像で空マスク提出が多い場合は、FALLBACK閾値・面積パラメータを調整し、非空RLE提出率を最大化する
- [ ] ヒストグラム・分布分析（F1分布・極端値の原因調査）をvalidateノートで毎回実施し、モデル・後処理の弱点を早期発見する
- [ ] 提出前にvalidate_submission_formatで形式検証を行い、case_id順・RLE偶数要素・authentic埋めの健全性を担保する

---

## EDAから得られる後処理設計のヒント

- EDAより、偽造領域（インスタンス）の面積分布は10〜50pxの小領域が多い。
- min_area=10〜25で偽造検出率UP、ただしFP増加リスクあり。グリッドサーチで最適化推奨。
- 極端なoutlier（1px, 50000px超）は個別対応。

#### 【クラシカルCMFD論文からのヒント】

- DCT, DWT, LBP, SIFT, SURFなどのクラシカル特徴量は、CNN単体では検出困難な偽造領域（小領域・平坦領域・幾何変換あり）に有効な場合がある。
- ブロックベース（周波数変換・テクスチャ）とキーポイントベース（SIFT/SURF等）のハイブリッドアプローチが有効。
- コピー領域のタイプ（背景/オブジェクト/生物/文字）ごとに特徴や検出難易度が異なるため、領域タイプ別の処理や閾値最適化も検討。
- 閾値やmin_areaなどのパラメータは画像ごとに最適化する工夫が有効。
- 特徴量の次元削減（PCA/SVD）やバッチ処理、GPU活用で計算コストを抑える。
- CNN/Transformerとクラシカル特徴量の併用も今後の有望な方向性。

---

## Copy-Move Forgery Detection（CMFD）分野の実践的ヒント・知見（詳細版）

---

### 1. 特徴抽出の多様性とハイブリッド化

- **周波数変換特徴（DCT, DWT, FFTなど）**  
  ノイズやJPEG圧縮に強く、テクスチャ領域の偽造検出に有効。ただし、回転や拡大縮小などの幾何学変換には弱い。
- **キーポイント特徴（SIFT, SURF, ORB, Harris等）**  
  スケール・回転変化に強く、物体や文字など明瞭な構造の偽造検出に有効。平坦領域（空・草地など）ではキーポイントが得られず検出力が低下。
- **テクスチャ・輝度特徴（LBP, GLCM, 色ヒストグラム等）**  
  背景や自然物の偽造検出に有効。
- **実践ポイント**  
  → 1種類の特徴量だけでなく、**複数の特徴量を組み合わせる（ハイブリッド化）**ことで、領域や偽造パターンごとの弱点を補完できる。

---

### 2. マッチング手法の工夫と効率化

- **特徴ベクトルのソート・KD-Tree・LSH（局所性鋭敏型ハッシュ）**  
  大規模画像や高解像度画像でも高速に類似ペアを探索可能。
- **最終的な類似度評価**  
  ユークリッド距離や相関係数で厳密に判定し、誤検出を減らす。
- **実践ポイント**  
  → **高速化と精度のバランス**を意識し、前段で候補を絞り、後段で厳密評価する2段階アプローチが有効。

---

### 3. 閾値・パラメータの自動最適化

- **画像ごとに最適な閾値（min_area, mask_threshold等）が異なる**  
  固定値では検出力が大きく変動するため、**自動チューニングやアンサンブル的な閾値設定**が有効。
- **実践ポイント**  
  → OOFや検証時に複数パラメータでグリッドサーチし、画像ごと・foldごとに最適値を選択する仕組みを導入。

---

### 4. コピー領域タイプごとの多様性対応

- **背景（草地・空など）・オブジェクト・生物・文字**  
  コピーされる領域のタイプごとに特徴や検出難易度が異なる。
- **実践ポイント**  
  → **領域タイプごとに異なる特徴量や後処理を適用**することで、検出精度を底上げできる可能性。

---

### 5. スケーラビリティ・計算コスト対策

- **特徴量次元削減（PCA, SVD等）やバッチ処理**  
  高次元特徴量や大規模データでも計算コストを抑制。
- **GPUや分散処理の活用**  
  Kaggle環境でもバッチ推論や特徴量圧縮を意識。
- **実践ポイント**  
  → **計算コストを意識した設計・実装**が大規模データ対応には必須。

---

### 6. ハイブリッドアプローチの重要性

- **ブロックベース（周波数・テクスチャ）とキーポイントベース（SIFT等）の組み合わせ**  
  それぞれの弱点（幾何変換・平坦領域）を補完。
- **CNN/Transformerとクラシカル特徴量の併用**  
  画像全体の文脈や局所特徴を同時に活用できる。
- **実践ポイント**  
  → **クラシカル手法＋深層学習のハイブリッド**は今後の有望な方向性。

---

### 7. 評価指標・データセットの多様性

- **ピクセル単位・画像単位の両方で評価**  
  → 偽陽性・偽陰性のバランスを意識。
- **公開データセット（CASIA, MICC-F220, CoMoFoD等）で事前検証**  
  → 汎用性・再現性のある手法開発に役立つ。

---

### Recod.ai LUCへの応用ポイント

- **U-NetやCNN単体では限界があるため、クラシカル特徴量（DCT, SIFT, LBP等）とのハイブリッドや前処理の工夫を検討**
- **小領域・平坦領域の偽造検出には、複数特徴量や領域タイプごとの後処理が有効**
- **閾値やmin_areaなどのパラメータは画像ごとに最適化する工夫が必要**
- **計算コスト・スケーラビリティも意識し、特徴量圧縮やバッチ処理を活用**
- **mask定義（コピー元・コピー先両方）に忠実な評価・学習を徹底**

---

### まとめ

- **「0.303の壁」を超えるには、クラシカル手法の併用、ハイブリッド化、領域タイプ別処理、パラメータ自動化などの工夫が不可欠**
- **CNN単体の限界を感じたら、CMFD分野の知見を積極的に導入・組み合わせること**

