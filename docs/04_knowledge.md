---

## 実験比較・戦略立案レポート（2025-11-14）

---

### 📚 このドキュメントについて

**対象読者**: 理論的な背景を学びたい人  
**主な内容**: 
- 実験結果の定量分析
- モデルアーキテクチャの比較
- パラメータ嗜好の理論的解説

**関連ドキュメント**:
| ドキュメント | 用途 | 対象者 |
|:---|:---|:---|
| **04_knowledge.md** (ここ) | 理論・背景知識 | 研究者・分析者 |
| **05_hints.md** | 実装チェックリスト | エンジニア・実装者 |
| **11_parameter_tuning_knowledge.md** | 実戦チューニング戦略 | チューニング担当者 |

---

### 1. 実験結果一覧表

| 実験ID | モデル | LBスコア | macro_f1 | forged_nonempty_ratio | authentic_fp_ratio | 備考 |
|--------|--------|----------|----------|----------------------|-------------------|------|
| EXP001T/v9s | Mask R-CNN | **0.303** | 0.460 | 0.0018 | 1.00 | 極端な退化解 |
| EXP002T/v3ds | U-Net | **0.113** | 0.500 | 1.00 | 0.00 | 完璧な分離だがLB低 |
| EXP003T/v12s | DINOv2+CNN | **0.304** | **0.511** | 0.046 | 0.006 | 最もバランス良好 |

---

### 2. 各モデルの定量的分析

#### **EXP001T/v9s (Mask R-CNN)**
- **強み**: Instance segmentation、物体検出ベースで境界精度が高い
- **弱み**: 
  - forged_nonempty_ratio = 0.0018（偽造検出率0.18%）→ **退化解**
  - authentic_fp_ratio = 1.00（誤検出率100%）→ ほぼ全画像を「authentic」判定
  - 学習が収束せず、極端に保守的な予測
- **原因**: グリッドサーチのパラメータが厳しすぎる、または学習不足

#### **EXP002T/v3ds (U-Net)**
- **強み**: 
  - forged_nonempty_ratio = 1.00（偽造検出率100%）
  - authentic_fp_ratio = 0.00（誤検出率0%）
  - 完璧な分離指標、macro_f1も0.500と高い
- **弱み**: 
  - LBスコア0.113と極端に低い
  - **提出形式・RLE・後処理のミスマッチが濃厚**
  - 画素レベルF1とLB評価の乖離が大きすぎる
- **原因**: submission.csvの形式エラー、または後処理で過度なフィルタリング

#### **EXP003T/v12s (DINOv2+CNN)**
- **強み**: 
  - macro_f1 = 0.511（最高）
  - forged_nonempty_ratio = 0.046、authentic_fp_ratio = 0.006（健全な分離）
- **弱み**: 
  - 偽造検出率4.6%はやや低い
  - 高解像度（512px）で計算コスト大

- **macro_f1とLBスコアは必ずしも相関しない**
  - EXP002T: macro_f1=0.500なのにLB=0.113（乖離大）
  - EXP003T: macro_f1=0.511でLB=0.304（相関）
- macro_f1は全体的に高水準（0.46～0.51）→ 追加データによる分布適応は成功
- しかし、LBスコアへの反映が不十分
  - EXP001T: 退化解で効果消失
  - EXP002T: 提出形式エラーで効果測定不能

#### **最も影響する指標**
1. **forged_nonempty_ratio**: これが低いとLBスコアは必ず低迷

**EXP001T (Mask R-CNN) の立て直し**
- グリッドサーチのパラメータを緩和（mask_bin: 0.3→0.2, min_area: 100→50）
- forged_nonempty_ratioを0.3以上に引き上げる

**EXP002T (U-Net) の提出形式修正**
- MEAN_THRESHOLD: 0.35→0.25に緩和

#### **B. アンサンブル戦略**

**推奨組み合わせ**
1. **EXP003T (DINOv2) + EXP002T (U-Net修正版)**
   - U-Net: 検出率100%（修正後）
   - マスク平均化で両者の強みを活かす

2. **3モデル投票方式**（修正後）
   - 2/3以上が「forged」判定した領域のみマスク化

- まずEXP002Tの提出形式を修正 → LB再提出
- EXP003Tのパラメータ緩和版（v13s）を実験
- 両者のアンサンブル版（EXP004T）を作成

#### **C. データ活用**

**追加データのさらなる活用**
- 追加データのドメイン分析（画像特性、偽造パターン）
- 追加データのみでpre-trainし、既存データでfine-tune（ドメイン適応）
- 追加データをaugmentation（色調・ノイズ）で多様化

**分布適応**
- test画像の統計（平均輝度・コントラスト）をtrain/追加データと比較
- ドメインシフト対策（色正規化、ヒストグラムマッチング）

---

### 5. LBスコア伸び悩みの原因と改善案

#### **原因**
1. **退化解（EXP001T）**: 後処理が厳しすぎて検出率が壊滅
2. **提出形式エラー（EXP002T）**: 完璧なvalidate結果なのにLB低迷 → RLE/形式ミス濃厚
3. **偽造検出率不足（EXP003T）**: 4.6%と低く、偽造画像の大半を見逃し

#### **改善案**
- **提出前の厳密な形式チェック**: validate_submission_format関数を必ず実行
- **分離指標のガードレール設定**: forged_nonempty_ratio ≥ 0.3を強制
- **後処理のグリッドサーチ範囲拡大**: 低閾値・小面積も許容し、退化解を回避
- **OOF検証とLB提出の整合性確認**: OOFで高スコアでもLBで検証するまで確定しない

---

### 6. 今後の実験計画・優先順位

#### **最優先（1週間以内）**
1. **EXP002T/v3ds の提出形式修正版（v3ds2）を作成 → LB再提出**
   - RLE形式、case_id順序、authentic判定ロジックを厳密にチェック
   - 期待LB: 0.30以上（validate結果が正しければ）

2. **EXP003T/v12s のパラメータ緩和版（v13s）を作成 → LB提出**
   - AREA_THRESHOLD: 400→200、MEAN_THRESHOLD: 0.35→0.25
   - 期待LB: 0.31～0.32（偽造検出率向上で）

3. **EXP001T/v9s のグリッドサーチ修正版（v10s）を作成**
   - mask_bin: [0.2, 0.3, 0.5]、min_area: [25, 50, 100]に変更
   - Epoch: 5→10、学習率調整
   - 期待LB: 0.28以上（退化解脱却で）

#### **中期（2週間以内）**
4. **EXP004T: EXP003T v13s + EXP002T v3ds2 アンサンブル**
   - マスク平均化方式
   - 期待LB: 0.33～0.35（両者の強み融合で）

5. **追加データのドメイン分析 & 分布適応実験（EXP005T）**
   - 画像統計比較、色正規化、ヒストグラムマッチング
   - 期待LB: 0.31～0.33（分布適応で）

#### **長期（1ヶ月以内）**
6. **3モデル投票方式アンサンブル（EXP006T）**
   - Mask R-CNN（修正版） + U-Net（修正版） + DINOv2（緩和版）
   - 期待LB: 0.35以上（多様性で）

7. **新規モデル導入検討**
   - Segment Anything Model (SAM)、YOLO v8 segmentationなど
   - 公開ブックでの使用状況を調査後、実装判断

---

## **まとめ**

- **EXP002T（U-Net）の提出形式修正が最優先** → validate結果が正しければ大幅改善の可能性
- **EXP003T（DINOv2）のパラメータ緩和** → 偽造検出率向上でLB 0.31～0.32狙い
- **EXP001T（Mask R-CNN）の退化解脱却** → グリッドサーチ・学習の見直し
- **アンサンブル戦略** → 修正後の2～3モデルで0.33～0.35を目指す

追加データの効果はvalidate結果に現れているため、提出形式・後処理の最適化が鍵です。

---

## Ultra-Fast U-Net⚡ v6 (0.302) から得たナレッジ

**対象**: [`research/CODE/ultra-fast-image-forgery-detection-u-net-v6_score0.302/`](research/CODE/ultra-fast-image-forgery-detection-u-net-v6_score0.302/)  
**スコア**: LB 0.302  
**分析日**: 2025-01-XX

### 1) U-Net vs Mask R-CNN の比較（アーキテクチャ選択の指針）

| 観点 | Mask R-CNN（EXP001T） | U-Net⚡（公開NB） | 選択基準 |
|:---:|:---:|:---:|:---:|
| **タスク適合度** | インスタンス検出 | セマンティックセグメンテーション | **コピー&ムーブはバイナリマスクで十分** |
| **パラメータ数** | 11M+ | 1.9M | **軽量化優先ならU-Net** |
| **学習時間** | 40分（5epoch, GPU） | 7分（2epoch, CPU） | **速度優先ならU-Net** |
| **推論速度** | 5 img/sec（推定） | 7 img/sec | U-Net優位 |
| **メモリ使用量** | 14GB+ | 12GB | U-Net優位 |
| **スコア** | 0.303 | 0.302 | **ほぼ同等** |

**結論**:
- **U-Netの優位性**: タスク適合度・速度・軽量性で圧倒的
- **Mask R-CNNの優位性**: 皆無（インスタンス分離が不要なため）
- **推奨**: セマンティックセグメンテーション（U-Net/DeepLabV3+）をベースとする

---

### 2) 入力解像度の影響（128 vs 256 vs 384）

| 解像度 | メモリ | 学習時間 | 小物体検出精度 | 推奨用途 |
|:-----:|:------:|:-------:|:-------------:|:--------:|
| **128** | 4GB | 7分 | 低（特徴が潰れる） | プロトタイピング |
| **256** | 8GB | 20分 | 中（バランス型） | **ベースライン** |
| **384** | 12GB | 45分 | 高（詳細保持） | **最終調整** |
| **512** | 16GB | 80分 | 最高（過剰？） | GPU余裕ある場合のみ |

**推奨戦略**:
1. 128で動作確認（Phase 0）
2. 256でベースライン確立（Phase 1）
3. 384で精度向上（Phase 2）
4. 512は時間対効果が低い（過学習リスク増）

---

### 3) データサンプリングの罠（先頭500枚 vs ランダム）

**v6の実装**:
```python
for file in os.listdir(path)[:500]  # 先頭500枚のみ
```

**問題点**:
- `os.listdir()` の順序は**ファイルシステム依存**（alphabetical/timestamp）
- case_id順序に偏りが生じる → テストセットと分布不一致

**対策**:
```python
file_list = os.listdir(path)
random.shuffle(file_list)  # ← ランダム化
file_list = file_list[:500]
```

**効果**:
- 偏り解消 → スコア+0.01～0.02（推定）
- ただし、**全データ学習（5000枚）**が最も効果的（時間許容範囲内なら）

---

### 4) 後処理の重要性（モルフォロジー）

**v6の実装**:
```python
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # ノイズ除去
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 穴埋め
if mask.sum() < 100:  # 極小領域
    return "authentic"
```

**効果**:
- False Positive削減（小ノイズ除去） → Precision向上
- True Positive補完（穴埋め） → Recall向上
- **スコア寄与**: +0.01～0.03（推定）

**最適化の余地**:
- カーネルサイズ: 3x3 → 5x5/7x7（画像サイズ依存）
- min_area閾値: 100px → 50/200/500px（グリッド探索）
- 順序: Open→Close vs Close→Open（アブレーション）

---

### 5) CPU vs GPU の選択基準

| 環境 | 学習時間（IMG_SIZE=256） | メモリ | 推奨用途 |
|:---:|:----------------------:|:------:|:--------:|
| **CPU** | 60分+ | 8GB | プロトタイピング（速度不問） |
| **GPU** | 20分 | 12GB | **本番実験** |

**結論**:
- v6は「CPU-friendly」を謳うが、**GPU使用で3倍高速化**
- Kaggle無料枠: GPU 30時間/週 → 十分に活用可能
- **推奨**: GPU必須（CPU専用は非効率）

---

### 6) Validationの不在が招くリスク

**v6の問題**:
- Validation実装なし → Loss減少でも汎化性能が測定不可
- 過学習検出不可 → Private LBで大幅下落のリスク

**EXP001T v7-v9の教訓**:
- CV/LBギャップ=0.000（完全一致） → 異常な健全性
- Valデータが小さすぎる（fold1のみ）＋データ分布が単純 → 汎化性能を測れない

**推奨**:
- **GroupKFold 5-fold必須**（case_idでグループ分割）
- CV/LBギャップ±0.01～0.03が健全（0.000は警戒信号）
- 全fold学習 + アンサンブルでデータ効率5倍

---

### 7) 「0.30の壁」の本質（ベースライン効果）

**観察事実**:
- 公開Notebook上位10本: 0.294～0.303（ほぼ0.30前後で収束）
- EXP001T v5～v9: 全て0.303（変更要素に関わらずスコア据え置き）
- U-Net⚡ v2/v6: 0.303/0.302（v2→v6で微減）

**仮説**:
1. **提出形式の正しさ**が0.30達成の必要条件（RLE・case_id順序・authentic判定）
2. **最低限のモデル**（U-Net/Mask R-CNN）で0.30到達
3. **0.30→0.35の壁**: 入力解像度・Aug・全fold学習などの本質的改善が必要

**突破の方向性**:
- **P0**: 入力解像度↑（384/512） → +0.05～0.10
- **P0**: 合成コピー&ムーブAug → +0.05～0.10
- **P1**: 全fold学習 + アンサンブル → +0.02～0.05
- **P1**: Backbone追加（EfficientNet-B2 pretrained） → +0.03～0.07

---

### 8) v2（0.303）→ v6（0.302）の差分から得た知見

**主要差分**: 皆無（スコア差0.001は統計的誤差範囲内）

**推定理由**:
- RLE encoding: 「critical bug fix」と再明記 → 微細なバグ修正（推定）
- 実装コード: 同一（モルフォロジー・カーネル・閾値すべて同じ）
- 説明文: 簡潔 → 詳細HTML（スコア無関係）

**結論**:
- v2→v6は「コード整備」が主目的（スコア向上ではない）
- **スコア向上の要因**: アーキ・解像度・Aug・後処理などの本質的改善のみ

---

### 9) validate分析に基づく次の実験戦略

**目的**:  
- 偽造領域検出率（forged_nonempty_ratio）UPと誤検出率（authentic_fp_ratio）DOWNを両立するRLE提出型モデルを目指す
- validate分析（experiments/validate/recod-ai-luc-validate.ipynb）で分離評価指標を重視

**優先アクション**:
- forged画像で非空RLE提出率UP（forged_nonempty_ratio重視）
- authentic画像で誤検出率DOWN（authentic_fp_ratio重視）
- mask_th, min_area, morph_kernelのグリッド探索＋モルフォロジー順序アブレーション
- 提出前validate_submission_formatで形式検証

**中期以降**:
- 特徴量拡張（ELA, 統計量, rolling/fps補正）をA/Bテスト
- 全fold学習＋TTAで汎化性能UP
- モデル多様化・アンサンブル（U-Net/RCNN/LGBM/kNN組み合わせ）

**運用**:
- 各実験の成果物（oof.csv, metrics.json, run.json）を必ず保存
- validateノートで分離評価・ヒストグラムを毎回更新
- スコア改善トレンド・異常検知を定期レビュー

**推奨保存先**:  
- 本セクション（docs/04_knowledge.md）末尾

**参考**:  
- experiments/validate/recod-ai-luc-validate.ipynb  
- docs/EXPERIMENT_WORKFLOW.md  
- research/CODE/1102_cross_sectional_comparison.md

---

### 10) validate分析・分離指標重視の運用指針

**背景**:  
- v5b実験で退化解（全authentic提出）が多発し、mean_f1だけではモデル健全性を判断できないことが判明
- forged_nonempty_ratio, authentic_fp_ratioなど分離指標の記録・分析が必須
- migration_planも「分離指標健全性」「退化解ガード」「アンサンブル前倒し」へ軌道修正

**運用指針**:
- validateノートで毎回「forged_nonempty_ratio」「authentic_fp_ratio」「F1分布ヒストグラム」を記録・分析
- forged_nonempty_ratio < 0.2、authentic_fp_ratio > 0.3 の場合は提出不可
- metrics.json/run.jsonに分離指標を必ず記録
- 退化解（全authentic提出・F1=0/1集中）はサマリー表・ヒストグラムで即座に検知
- アンサンブルは「分離指標の補完関係」を重視し、単純平均だけでなく論理和・stacking等も検討

**推奨アクション**:
- 後処理パラメータ（mask_th, min_area, morph_kernel）は分離指標を主指標にグリッド探索
- authentic/forgedでパラメータ分離（クラス別最適化）も検討
- validate_submission_formatで形式検証を必ず実施

**参考**:  
- [experiments/validate/recod-ai-luc-validate.ipynb](experiments/validate/recod-ai-luc-validate.ipynb)
- [experiments/EXP002T/migration_plan.md](experiments/EXP002T/migration_plan.md)
- [docs/05_hints.md](docs/05_hints.md)

---

// filepath: docs/05_hints.md
# 運用Tips・クイックリファレンス

## validate分析・分離指標重視の追加ヒント

- [ ] mean_f1だけでなく「forged_nonempty_ratio」「authentic_fp_ratio」を必ず記録・提出判断に使う
- [ ] 退化解（全authentic提出・F1=0/1集中）はサマリー表・ヒストグラムで即座に検知
- [ ] metrics.json/run.jsonに分離指標（forged_nonempty_ratio, authentic_fp_ratio）を必ず記録
- [ ] アンサンブルは分離指標の補完関係を重視し、単純平均だけでなく論理和・stacking等も検討
- [ ] migration_planの成功基準・撤退条件に分離指標の下限・上限を明記

## EDA・後処理設計の追加ヒント

- [ ] インスタンス面積分布・authentic誤検出率をEDAで可視化し、min_area・mask_th設計に反映
- [ ] validateノートでヒストグラム・分布分析を毎回実施し、モデル・後処理の弱点を早期発見

---

### DINOv2の優位性（横断比較分析より）

- **自己教師あり学習**: ImageNetラベル不要→偽造検出への汎化力
- **セマンティック理解**: 高レベル特徴で「不自然さ」検出
- **軽量化**: エンコーダ凍結でパラメータ数削減
- **実績**: LB 0.305（当リポ最高スコア）

### Sobel勾配強調の有効性（再確認）

- エッジ保存→小物体検出精度↑（+0.02～0.05）
- 実装: `cv2.Sobel` + 動的閾値（μ+0.3σ）
- リスク: ノイズ増幅（alpha_grad調整で緩和）

### 統計的マスク生成の有効性

- **authentic多数派戦略**: データ不均衡の利用→LB 0.30前後確保
- **KDE＋楕円パラメータ**: train_masksの分布再現→「らしい」マスク生成
- **マスク生成確率**: 0.5%が最適（偽陽性削減）
- **実績**: LB 0.304（**学習なし**）

### 提出形式厳密化の重要性

- case_id順・authentic埋め・RLE偶数要素の3条件必須
- 1つでも欠けると0点（Scoring Error）
- validate関数で事前チェック必須

---

## 0.303の壁突破までの道程と現在地

- 必須基盤（提出形式厳密化・authentic戦略）～健全runガードは完全実装済み
- DINOv2特徴量・Sobel勾配強調・動的閾値・分離指標フィルタは部分実装・検証中
- 解像度UP・全fold学習・アンサンブルは一部実装済み
- Aug/TTA/特徴量拡張は未検証～着手段階
- **現在地: 6/10 ～ 7/10（0.305突破まであと3段階）**

### 今後の作戦
1. DINOv2+Sobel+動的閾値+分離指標フィルタの完全移植・検証
2. 入力解像度384/512・全fold学習・アンサンブル強化
3. Aug強化・TTA・特徴量拡張のA/Bテスト
4. 分離指標ガード・成果物自動停止の徹底
5. 失敗runのヒストグラム・分布分析

---

### v10（解像度512px化）の優位性（DINOv2系分析より）

- **入力512px**: v9（256px）の2倍解像度→小物体検出精度↑
- **実績**: LB 0.309（v9: 0.305から+0.004向上）
- **epoch 4 + LR 1e-5**: 緩やかな学習で過学習回避＋安定化

### 解像度UPの有効性（v10検証結果）

- 256px→512px: 小物体検出精度+0.01～0.03（推定）
- トレードオフ: メモリ消費2～4倍（BATCH_SIZE=1固定で対応）
- 実装: IMG_SIZE変更のみ（実装難度低）
- **採用推奨**: P0（即座に移植）

### epoch増加の注意点

- 1～3 epoch: 過学習回避だが未学習リスク
- 4 epoch: 学習安定化（ただしLR減少必須）
- **LR調整セット**: epoch↑ → LR↓（3e-4 → 1e-5推奨）

---

## 実験総括レポート知見の統合（2025-11-22）

### EXP001T～EXP005T シリーズ全体分析

**スコア推移**:
- EXP001T (Mask R-CNN): 0.303（インスタンス検出）
- EXP002T (U-Net): 0.244～0.301（セマンティック検出の課題）
- **EXP003T (DINOv2+CNN): 0.318（現行ベスト）** ⭐
- EXP004T (EfficientNet-B0): 0.303（分類ベースラインのみ）
- EXP005T (SAM2): 0.314→0.104（Failure）

**本質的教訓**:
| 気づき | 詳細 | 実装対応 |
|:---|:---|:---|
| **SAM2は「手」、DINOv2は「目」** | SAM2は輪郭抽出のみ→検知能力なし | Refiner用途に限定、単体Fine-tune廃止 |
| **借り物重みの最強性** | 外部データ巨大学習による強力な特徴→過学習リスク | Fine-tune禁止、工夫（TTA/後処理）で対抗 |
| **TTA方式の差分が決定的** | 3方向(B/H/V) vs 4方向(+HV) → HV削除で+0.002 | HV-flip廃止（v17計画） |
| **alpha_grad=0.35の限界** | 公開ブック(0.319)で有効→v16では逆効果 | TTA構成・モデル依存→環境ごと最適化必須 |
| **退化解の多発** | 全authentic提出(0.303)＆SAM2単体(0.104)で頻出 | forged_nonempty_ratio≥0.15必須化 |
| **マクロF1とLBの乖離** | validate指標(0.50)≠LB評価(0.30)→後処理が決定的 | RLE形式・分離指標・ガード実装に注力 |

### EXP003T詳細版本のv15への知見統合

**v15が0.318到達した要因**:
1. **DINOv2エンコーダ（768次元）**: 自己教師学習による汎化性（ImageNetベースより優位）
2. **Sobel勾配強調（alpha_grad=0.30）**: 小領域偽造の境界検出精度↑（+0.01～0.02推定）
3. **動的閾値（μ+0.3σ）**: 画像ごとの明度差に適応
4. **確信度フィルタ（area<400 & mean<0.3）**: FP削減（+0.01推定）
5. **TTA確率平均4方向**: 外れ値ロバスト性↑（+0.008実績確認）

**v16/v17への検証項目**:
- ⚠️ v16 (alpha_grad=0.35): LB 0.317（-0.001）→ **TTA構成依存性確認**
- ⚠️ v17 (HV-flip削除): 期待+0.002（0.319公開ブックの核心）

### 公開Notebookリサーチの決定的発見

**0.319 vs 0.317の唯一の差分**: TTA 3方向 vs 4方向
```python
# 0.317（4方向: Base + H-flip + V-flip + HV-flip）
preds.append(torch.flip(pred, dims=[2, 3]))  # HV-flip含

# 0.319（3方向: Base + H-flip + V-flip のみ）
# ❌ HV-flip削除 → 外れ値削減で+0.002
```

**推定メカニズム**:
- HV-flip = H ∘ V（合成変換）→ 情報相関性高い
- 4方向平均で境界ノイズ（flip-hv特有ノイズ）混入
- 3方向平均で外れ値削減＋頑健性↑

### 借り物重み vs ファインチューニングの決定

**戦略判定**:
| 選択肢 | メリット | デメリット | 当リポ判定 |
|:---|:---|:---|:---:|
| **借り物重み＋工夫** | 再現性・安定・時間節約 | 差別化困難 | ✅ 採用 |
| **ファインチューニング** | データ適応・差別化 | 劣化リスク・計算コスト | ❌ 廃止 |
| **アンサンブル** | 補完で精度↑ | 実装複雑 | ✅ 優先度P1 |

**根拠**:
- ravaghi/dinov2の重み（外部巨大データ学習）をコンペ小規模データで上書き → Catastrophic Forgetting
- 0.318達成版（v15）でも差別化できておらず → 本質的な回答は「工夫」にある
- アンサンブルで補完関係（DINOv2検出 × CNN別視点）が最効率

### 次フェーズの優先度再確認

**P0（即座実施）**:
1. ✅ v17: HV-flip削除（TTA 3方向化）→ 目標+0.002（0.320到達）
2. ✅ LB提出確認 → v16/v17スコア記録

**P1（中期1週間）**:
1. 5-fold学習 → データ効率5倍、汎化性能向上（+0.01～0.02）
2. アンサンブル開発（EfficientNet-B4等CNN相棒と合成）

**P2（研究枠）**:
1. DINOv2-large（768→1024次元）検証
2. 外部データ転移学習

### データ・モデル設計への実装示唆

**alpha_gradの環境依存性**:
- **公開ブック0.319**: alpha_grad=0.35（TTA 3方向前提）
- **v16失敗**: alpha_grad=0.35→0.317（TTA 4方向・他パラメータ差異）
- **教訓**: パラメータは「モデル構成・TTA方式に密結合」→本環境では0.30が最適（v15確認）

**パラメータ調整時の心構え**:
- 公開ブックの値をそのまま移植しない（環境差異で逆効果）
- validate modeで都度検証（authentic_fp_ratio監視必須）
- グリッドサーチは「モデル構成全体を固定して微調整」が鉄則

---

## SAM2 推論採用の中止決定（2025-11-29）

### 公開ディスカッション発見：Top Kaggler (7位) の検証結果

**Jirka Borovec (Topic Author) による提案**:
- SAM2 を使った偽造候補地生成アプローチ
- 理論的には優れている（object-level segmentation）

**Corey James Levinson (7位) による検証結果**:
```
SAM2 ベースのアプローチ
  → validation F1 = 0.29
  → All Authentic baseline (0.30) より **劣化**
```

### この結果が示すこと

#### **SAM2 は「Detector（検出器）」ではなく「Proposer（候補提案器）」**

```
誤解（私たちの実装）:
  DINOv2 → SAM2 refinement → 最終出力
  ↑ SAM2 を「決定権のある refiner」として使用
  ↓ 結果: 0.314（v4）/ 0.318（v3）

正解（Jirka の提案）:
  オフラインでデータセット作成時
  → SAM2 で綺麗にセグメンテーション
  → 人工的な偽造データを生成
  → それを学習に使用
  ↓ 結果: 向上の可能性（未実装）
```

#### **なぜ推論時の SAM2 は失敗するのか**

```
根本原因: 「DINOv2 に完全依存」している

推論フロー（v3/v4）:
  DINOv2(ALPHA_GRAD) 
    ↓ BBox 出力
  SAM2 refine
    ↓ BBox を精密化
    
問題点:
  - DINOv2 が見落とした領域 → SAM2 も見落とす
  - DINOv2 がノイズ出力 → SAM2 も refinement
  - つまり「多様性ゼロ」
  
結果:
  新しい発見（Recall向上）なし
  SAM2 のノイズ化効果のみ顕現
  → スコア低下（0.318 → 0.314）
```

#### **アンサンブル（v19 + v3）が無駄な理由**

```
推奨（誤った企画）:
  v19 + v3 のアンサンブル
  → 両方 DINOv2 ベース
  → 差分は ALPHA_GRAD のみ（0.35 vs 0.30）
  → 本質的な多様性がない

結果:
  「腐ったリンゴと新鮮なリンゴをミキサーに入れても、
   美味しいジュースにはならない」
   
  相補性ゼロ → スコア向上なし
  むしろ 0.318 → 0.316（さらに劣化）の可能性
```

### 決定：推論時の SAM2 採用を中止

| 対象 | 判定 | 理由 |
|:---|:---:|:---|
| **推論での SAM2 統合** | ❌ 中止 | 多様性ゼロ、Top Kaggler で 0.29 (失敗) |
| **v19 + v3 アンサンブル** | ❌ 非推奨 | 多様性がない、スコア向上期待できず |
| **DINOv2 単体（0.318）** | ✅ 確定 | SAM2 含むアプローチ より優秀 |

### 今後の正しい方針

#### **近期（即座）: DINOv2 単体を確保**

```
現状: v15/v19 = 0.318
目標: 0.318 を LB に提出
対策: パラメータ微調整、TTA 最適化など
```

#### **中期: 真の多様性を持つモデルを育成**

```
主演 (Transformer 枠):
  DINOv2 (0.318) ← 大局的な文脈を見る

助演 (CNN 枠):
  EfficientNet-B4 / ConvNeXt / ResNet など
  ← 局所的なテクスチャを見る（コピペ境界など）

アンサンブル:
  DINOv2（特徴: 高レベル意味情報）
  × EfficientNet（特徴: 低レベル局所情報）
  → 相補性が高い → スコア向上（+0.02～0.05 期待）
```

#### **長期: データセット生成への SAM2 活用**

```
オフラインで SAM2 を使用:
  1. SAM2 で既存偽造画像をセグメンテーション
  2. 綺麗に切り抜いた物体を他の画像に合成
  3. 人工的な訓練データを生成
  4. これを EfficientNet 等の学習に使用
  
期待効果: encoder の本質的な改善
```

### 教訓：「新技術への陶酔」を避ける

```
❌ よくある失敗パターン:
   「SAM2 が流行ってる」
   → 「使ってみよう」
   → 「スコア悪化」
   → 「パラメータで救いたい」
   → 「さらに悪化」

✅ 正しいアプローチ:
   「技術の本質を理解する」
   → 「自分のタスクに合ってるか確認」
   → 「検証済みの実装か確認」
   → 「その技術の正しい使い道を見つける」
   → 「期待値を現実的に設定」

本研究での SAM2:
  - 推論の refiner として使う ❌
  - データ生成の補助として使う ✅
```

---

---

## DINOv2 単体 vs DINOv2 + SAM2 のパラメータ嗜好分析（2025-11-29）

**関連ドキュメント**:
- 実装ガイド → [`docs/05_hints.md#モデル嗜好に基づく実装ガイドライン`](05_hints.md#モデル嗜好に基づく実装ガイドライン)
- チューニング戦略 → [`docs/11_parameter_tuning_knowledge.md#モデル嗜好に基づくensemble戦略`](11_parameter_tuning_knowledge.md#モデル嗜好に基づくensemble戦略)

### 背景：v3/v4実験から発見された「モデル嗜好の分岐」

**観察事実**:
| モデル | ALPHA_GRAD | Macro F1 | Forged F1 | Authentic FP | 特徴 |
|:---|:---:|:---:|:---:|:---:|:---|
| **v19** (DINOv2 推定) | 0.35 | 0.5614 | 0.1565 | 3.36% | Aggressive（検出重視） |
| **v3** (DINOv2 + SAM2) | 0.30 | 0.5458 | 0.1169 | 2.52% | Conservative（精度重視） |
| **v4** (DINOv2 + SAM2, aggressive試行) | 0.35 | 0.5391 | 0.1035 | 2.52% | ❌ 悪化（パラメータ不適切） |

**驚くべき結論**:
- ✅ DINOv2 **単体**: ALPHA_GRAD=0.35（aggressive）が最適
- ✅ DINOv2 + SAM2: ALPHA_GRAD=0.30（conservative）が最適
- ❌ v4で aggressive に変更 → **Forged F1 が 0.1169 → 0.1035 に悪化**（-11.5%）

### 根本的なメカニズム：二段階パイプラインの最適化原理

#### **Stage 1: DINOv2 Encoder の役割**

```
DINOv2 単体（v19推定）:
  ├─ 勾配強調（ALPHA_GRAD=0.35）で「検出力」を最大化
  ├─ より多くの改ざん候補を見つけることが目標
  ├─ 結果: Forged F1 = 0.1565（高い）
  └─ 副作用: FP = 3.36%（誤検出多い）

特性: 「積極的・勢い重視」
```

#### **Stage 2: SAM2 Refinement の役割**

```
SAM2 は「粗い予測を精密化する refinement tool」:
  ├─ BBox が大量のノイズを含むと → ノイズも refine される
  ├─ 本来は false positive なのに「改ざんに見える」領域を生成
  ├─ 結果: Forged F1 が低下、精度が落ちる
  
SAM2 に適した入力:
  ├─ 信頼度の高い BBox（false positive が少ない）
  ├─ ノイズが少なく、確実な改ざん候補のみ
  ├─ ALPHA_GRAD = 0.30 が最適
  └─ 結果: 精密化により精度向上

特性: 「保守的・信頼性重視」
```

#### **パイプライン全体の最適化原理**

```
最適流路（v3）:

  DINOv2(conservative=0.30)
  ↓
  「確実な改ざん領域」のみ出力
  （false positive が少ない BBox）
  ↓
  SAM2 refinement
  ↓
  信頼できる input のみ処理
  → 精密化が有効に機能
  ↓
  結果: Forged F1 = 0.1169, FP = 2.52% ✅


非最適流路（v4）:

  DINOv2(aggressive=0.35)
  ↓
  「可能性がある領域」を積極出力
  （false positive が多い BBox ← ノイズ含む）
  ↓
  SAM2 refinement
  ↓
  ノイズを含む input を処理
  → ノイズも refine されてしまう
  → 「ノイズが改ざんに見える」領域生成
  ↓
  結果: Forged F1 = 0.1035, FP = 2.52% ❌（精度低下）
```

### 重要な発見：FP 率が同じ理由

v3 と v4 で **Authentic FP = 2.52%** が全く同じであることが、この仮説を強く支持しています：

```
v3（conservative）:
  改ざん候補が少ない
  → 自動的に false positive も少ない
  → FP = 2.52%

v4（aggressive）:
  改ざん候補が多い（ノイズ含む）
  → false positive も多い
  ただし FP 率は 2.52%（同じ）
  → 「ノイズを forged 判定している」
  → 同時に「ノイズを refine している」
  → Forged F1 が低下
```

**つまり v4 では：**
- ❌ 「false positive が多い」がそのまま「forged 判定」される
- ❌ SAM2 が「本来は authentic なノイズ」も refinement
- ❌ 結果として検出精度が低下

### パラメータ嗜好の予測モデル

```python
# パラメータ最適化の汎用的なルール

def recommend_alpha_grad(model_type, has_refinement_stage):
    """
    model_type: "dinov2", "resnet", "efficientnet", ...
    has_refinement_stage: SAM2等の refinement を使う？
    """
    
    if has_refinement_stage:
        # Refinement がある場合 → conservative が最適
        # 理由: refinement 入力は「信頼度高い候補」が必須
        return 0.25  # 0.30～0.35 より低めが吉
    else:
        # 単体モデル → aggressive が最適
        # 理由: 検出力を最大化したい
        return 0.35  # または 0.40
```

### 今後のチューニングへの応用

#### **P0: v19 + v3 Ensemble で相補**

```python
# 嗜好の異なる2モデルを活用

pred_ensemble = 0.6 * pred_v19(aggressive) + 0.4 * pred_v3(conservative)

# 効果:
# - v19: 検出力が高い（Forged F1 = 0.1565）
# - v3: 誤検出が少ない（FP = 2.52%）
# - 融合: 両者の強みを活用して LB 0.32+ 狙い
```

#### **P1: パラメータチューニング時の心構え**

```python
# もし SAM2 や他の refinement stage を導入する場合:

# ❌ 誤り: 公開ブック（0.319）の ALPHA_GRAD=0.35 をそのまま移植
# ✅ 正解: refinement 対応のため ALPHA_GRAD を下げて再実験

for alpha_grad in [0.25, 0.30, 0.35]:
    model_v = train_with_param(alpha_grad)
    val_metrics = validate(model_v, split='val')
    
    # authentic_fp_ratio を監視しながら最適値を探索
    if val_metrics['authentic_fp_ratio'] < 0.03:
        best_alpha_grad = alpha_grad
```

#### **P2: 多モデル Ensemble 設計**

```python
# 異なる「嗜好」を持つモデルを組み合わせる

ensemble_models = [
    ('v19', aggressive=True, with_refinement=False),   # 検出力重視
    ('v3', conservative=True, with_refinement=True),   # 精度重視
    ('v15_alt', alpha_grad=0.32, with_refinement=False), # 中庸
]

# 異なるパラメータ嗜好を持つモデルを融合
# → 相補的な強みが期待できる
```

### 汎用的な知見：パイプラインアーキテクチャの最適化原理

```
【一般的な法則】

単段階パイプライン（モデル単体）:
  ✅ 出力の「広さ」を優先
  → aggressive パラメータ（高しきい値）
  → 多くの候補を検出

多段階パイプライン（検出 → 精密化 → 後処理）:
  ✅ 第1段階は「狭さ」を優先
  → conservative パラメータ（低しきい値）
  → ノイズが少ない候補のみ
  
  ✅ 第2段階以降で「精密化」
  → 信頼できる候補から詳細抽出
```

**応用例**:
- DINOv2 + SAM2（本研究）: Stage 1 conservative, Stage 2 refinement
- 物体検出 + Tracking: Stage 1 aggressive（多くの候補）, Stage 2 フィルタ
- 画像分類 + Explainability: Stage 1 broad（多クラス），Stage 2 局所化

### チューニングの優先順位（モデル嗜好に基づく）

| 優先度 | タスク | 判定基準 | 推奨アクション |
|:---:|:---|:---|:---|
| **P0** | v19 + v3 ensemble | 嗜好が互補的 → LB 0.32+ | 即座に実装 |
| **P1** | 新 refinement モデル導入 | conservative 嗜好を確認 | ALPHA_GRAD < 0.35 で実験 |
| **P2** | パラメータグリッド探索 | モデル構成ごとに最適値異なる | validate で分離指標監視 |
| **P3** | クラス別最適化 | forged/authentic で嗜好が異なる可能性 | 将来の研究課題 |

---

## EXP004E / EXP005T 実験知見（2025-01）

### 📊 実験結果サマリー

| Exp | アーキテクチャ | LB Score | CV Score | 主要発見 |
|:---|:---|:---:|:---:|:---|
| **EXP005T v0** | DINOv2-only | **0.318** | N/A | **シリーズ最高スコア** |
| EXP005T v1 | DINOv2 + SAM2 pipeline | 0.314 | N/A | SAM2追加で微減 |
| EXP005T v3 | DINOv2 + SAM2 (conservative) | 0.314 | 0.5458 | FP=2.52%、保守的設定 |
| EXP005T v4 | DINOv2 + SAM2 (aggressive) | 0.313 | 0.5633 | パラメータ嗜好不一致 |
| EXP005T v2 | SAM2 fine-tuning | 0.104 | N/A | **完全失敗**（大量FP） |
| EXP004E v3 | DINOv2 + DeepLabV3+ Ensemble | 0.306 | 0.3135 | Ensemble最高 |
| EXP004E v2 | ResNet50 + Multi-Head Ensemble | 0.277 | 0.2334 | Ensemble効果薄 |
| EXP004E v1 | DINOv2 + DeepLabV3+ v1 | 0.304 | N/A | ベースライン |

### 🔑 主要知見

#### 1. SAM2 Refinement Pipeline は**効果なし**

**発見**: DINOv2-only（v0 = 0.318）> DINOv2 + SAM2（v3 = 0.314）

**原因分析**:
- SAM2は「明確なオブジェクト境界」を期待するが、forgery痕跡は曖昧
- DINOv2の勾配マップ + SAM2 refinement で**情報が損失**
- conservative（α=0.30）でも改善せず、むしろ微減

**結論**: SAM2は**推論時の refinement には不向き**
- 代替案: SAM2を**データ生成（augmentation）**に活用する方向へ転換

#### 2. Ensemble 効果の限界（同系統モデル問題）

**発見**: DINOv2 + DeepLabV3+ Ensemble（v3 = 0.306）< DINOv2-only（v0 = 0.318）

**原因分析**:
- DeepLabV3+（ResNet50ベース）と DINOv2（ViT-gベース）は**CNN特徴の相関が高い**
- 両モデルの失敗パターンが類似 → 相補性が低い
- EXP004E/v2 の visual analysis で確認：
  - 45% の forged 画像で両モデルとも検出失敗
  - 特に「高周波テクスチャ保存型forgery」に弱い

**結論**: **アーキテクチャ多様性**が Ensemble 効果の鍵
- ❌ CNN + CNN（ResNet系 × 2）: 効果薄
- ❌ CNN + ViT特徴ベース（DINOv2 + DeepLabV3+）: 効果限定的
- ⭕ 推奨: **異なる入力表現**（RGB + 周波数領域 + 勾配）

#### 3. SAM2 Fine-tuning の失敗パターン

**発見**: SAM2 fine-tuning（v2 = 0.104）は**大失敗**

**原因分析**:
- SAM2 の事前学習は「明確なオブジェクト境界」に最適化
- forgery detection（微妙なテクスチャ差異）への fine-tuning で**全面崩壊**
- Authentic 画像を大量に Forged と誤判定（Massive False Positive）

**結論**: SAM2 の fine-tuning は**避けるべき**
- SAM2 は zero-shot / few-shot で使用するか、データ生成に限定

### 📈 パラメータ嗜好の知見

#### DINOv2 + SAM2 Pipeline の最適パラメータ

| パラメータ | Conservative (v3) | Aggressive (v4) | 推奨 |
|:---|:---:|:---:|:---:|
| alpha_grad | 0.30 | 0.40 | **0.30** |
| min_area | 100 | 50 | 100 |
| LB Score | 0.314 | 0.313 | conservative |
| CV Score | 0.5458 | 0.5633 | aggressive |

**発見**: CV嗜好 ≠ LB嗜好（過学習の兆候）
- CV では aggressive（0.5633）が優位
- LB では conservative（0.314）が同等以上
- **CV-LB Gap に注意**

### 🚀 次のアクション推奨

#### 短期（即座に実施可能）
1. **DINOv2-only（EXP005T v0 相当）を最終ベースラインとして固定**
   - SAM2 refinement は削除
   - 期待スコア: LB 0.318+

2. **異なる入力表現でのEnsemble検討**
   - RGB + 周波数領域（DCT/FFT特徴）
   - RGB + 勾配マップ（Sobel/Laplacian）

#### 中期（次の実験シリーズ）
3. **SAM2をデータ生成に活用（EXP006T候補）**
   - Authentic画像から SAM2 でオブジェクトマスク生成
   - マスク領域に人工的なforgeryを適用 → training data augmentation
   - 期待効果: より多様なforgeryパターンへの対応力向上

4. **周波数領域ベースのモデル追加**
   - DCT/FFT 特徴を入力とする軽量CNN
   - DINOv2（空間特徴）との相補性が期待できる

### 📝 Visual Analysis からの詳細知見

#### EXP004E/v3 vs v2 比較（Ensemble効果の検証）

| 指標 | v2 (ResNet50) | v3 (DINOv2+DeepLabV3+) | 改善率 |
|:---|:---:|:---:|:---:|
| Macro F1 | 0.2334 | 0.3135 | **+34.3%** |
| Forged 検出率 | 55% | 78% | +23pp |
| Authentic 正答率 | 100% | 100% | 維持 |

**v3 で改善されたパターン**:
- 小領域forgery（50-200px）の検出向上
- 低コントラストforgeryの検出向上

**v3 でも失敗するパターン**:
- 高周波テクスチャ保存型forgery
- 極めて小さな領域（< 30px）

---

