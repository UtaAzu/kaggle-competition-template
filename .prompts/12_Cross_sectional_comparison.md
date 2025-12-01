# 公開Notebook 横断比較プロンプト V2（本質的な差分を抽出）

【あなたの役割】
- パターン認識の専門家
- 目的：複数の高スコアNotebookから「共通する本質」と「実装上の差異」を抽出し、**スコア上昇の最小必要条件**を定義する

【一次入力（必須）】
- 対象Notebook 2〜4本のコード全文
  - 推奨：mabe-v1(0.41), kNN(0.39), LGBM(0.38), REMIX(0.42)の単体部

【参照（当リポ内）】
- [research/mabe-challenge-lgbm_0.38](research/mabe-challenge-lgbm_0.38)
- [research/mabe-v1-mouse-action-recognition0.41](research/mabe-v1-mouse-action-recognition0.41)
- [research/mabe-nearest-neighbors-testing-new-features_score0.39](research/mabe-nearest-neighbors-testing-new-features_score0.39)

---

## 【出力要件】

### 1) 共通原理の層別分析（Multi-level Common Principles）

#### レベル1：本質的原理（必須）
- **必ず全Notebookで採用されている工夫**
- 例：「GroupKFold by lab_id」「マルチスケール時系列特徴量」「クラス別閾値」
- これらなしではスコア0.38を超えられない、という仮説

#### レベル2：強力な補助原理（高頻度）
- **3/4以上のNotebookで採用** → 「実装推奨」の根拠
- 例：「rolling平滑化」「robust提出形式」「fps補正」

#### レベル3：作者固有のアイデア（低頻度）
- **1つのNotebookのみの工夫** → 高リスク・高リターン
- 例：「SHAP重要度分析」「カスタム損失関数」「meta-learner」

---

### 2) 相違点と効果の定量的仮説（Quantified Hypothesis）

| Notebook | LB Score | 特徴量 | 後処理 | CV分割 | モデル | 推定スコア要因 |
|:-------:|:--------:|:----:|:----:|:----:|:----:|:--------:|
| mabe-v1 | 0.41 | A+B+C | X+Y+Z | GK | LGBM | 特徴量(+0.08) |
| kNN | 0.39 | A+B | X+Y | GK | KNN | 後処理(+0.06) |
| LGBM | 0.38 | A+B | X | GK | LGBM | ベース |

**分析**：
- 特徴量 C の効果：mabe-v1 vs kNN の差分 = +0.02?
- 後処理 Z の効果：mabe-v1 vs LGBM の差分 = +0.03?
- **検証計画**：各要素をアブレーションで単体効果を測定

---

### 3) 「なぜ異なるモデル（kNN vs LGBM）が同スコア帯に達するのか」の考察

#### 仮説A：「データの本質が同じ」
- マウス行動認識は「姿勢空間での距離」が本質 → kNNもLGBMも同じ情報を活用
- **示唆**：新しい情報源（例：時間軸の高度な活用、マルチモーダル）がスコア向上の鍵

#### 仮説B：「モデル + 後処理のバランス」
- kNN は単純モデル + 複雑な後処理
- LGBM は複雑モデル + シンプルな後処理
- **示唆**：どちらのアプローチでも「本質的な課題」を解決すれば同じスコアに達する

#### 仮説C：「CV-LBギャップの存在」
- 両者ともCV >> LB の傾向（過学習）→ 実装の工夫より「汎化性能」が重要
- **示唆**：ラボ間ドメインシフト対策が最優先

---

### 4) 統合パイプラインの設計（Unified Pipeline Design）

```
Input Data
  ↓
[ブロック1] 共通特徴量（A+B）【必須】
  ↓
[ブロック2-A] mabe-v1型：特徴量C追加 → LGBM → 複雑後処理
  [ブロック2-B] kNN型：シンプル後処理 → KNN距離
  ↓
[ブロック3] 共通後処理（X+Y）【必須】
  ↓
[ブロック4] クラス別閾値 + robustify【必須】
  ↓
Output Submission
```

- **検証順序**：ブロック1 → 4 → 2（特徴量追加）→ 3（モデル多様化）

---

### 5) 移植パッチ計画（段階的・検証可能）

#### Phase 1（Week 1）：最小必須セット
- [ ] 共通特徴量（A+B）の実装 → validate で 0.30以上を確認
- [ ] 共通後処理（X+Y）の実装 → validate で 0.32以上を確認
- [ ] クラス別閾値 → validate で 0.33以上を確認
- **目標**：validate 0.33、LB 0.28

#### Phase 2（Week 2）：強力な補助原理
- [ ] rolling平滑化の導入 → validate で +0.01 を確認
- [ ] fps補正の導入 → validate で +0.01 を確認
- **目標**：validate 0.35、LB 0.30

#### Phase 3（Week 3）：mabe-v1型特徴量追加
- [ ] 特徴量C（角速度など）の実装 → validate で +0.02 を確認
- **目標**：validate 0.37、LB 0.32

#### Phase 4（Week 4）：モデル多様化
- [ ] kNNベースラインの追加 → 単体評価
- [ ] 軽量アンサンブル → validate で +0.01 を確認
- **目標**：validate 0.38、LB 0.33

---

### 6) リスク登録簿と代替案（Risk Register）

| リスク | 影響度 | 対策 | フォールバック |
|:---:|:----:|:---:|:----------:|
| 特徴量C過学習 | 高 | ラボ別CV検証 | 特徴量C削除 |
| 提出形式不一致 | 致命的 | robustify厳密化 | self行動網羅確認 |
| CV-LBギャップ拡大 | 中 | ラボ間分布分析 | 後処理パラメータ調整 |
| 計算時間超過 | 中 | メモリプロファイリング | 逐次処理化 |

---

### 7) 「本当に効く要素」の検証デザイン（Validation Framework）

#### A/B Test: 消し込み実験
```
Baseline（Phase 1後）: validate 0.33
- v1: + rolling平滑化 → validate 0.34?
- v2: + fps補正 → validate 0.34?
- v1+v2: + both → validate 0.35? （相互作用を確認）
```

#### ラボ別CV分析
```
各ラボ単独でCV評価 → ラボ間でスコア分布を比較
- 分布が似ていれば：汎化性能が高い
- 分布が大きく異なれば：ラボ間ドメインシフトが支配的 → この場合、特徴量拡張より「後処理」が優先
```

#### 特徴量重要度分析
```
LGBM の feature_importance / SHAP で、
- 高寄与度：必須
- 低寄与度：削除検討
- 負寄与度：必ず削除
```

---

### 8) 最終結論と推奨アクション（Final Recommendation）

#### 短期（2週間）
- [ ] **必ず実装**：共通特徴量 + 共通後処理 + クラス別閾値
- [ ] **validate 0.33を達成**してから次へ進む

#### 中期（4週間）
- [ ] **rolling + fps補正**を検証
- [ ] **ラボ別CV分析**で汎化性能を定量化

#### 長期（8週間以降）
- [ ] 特徴量拡張は「証拠がある場合のみ」実装
- [ ] mabe-v1完全移植ではなく「本当に効く要素」のみ選択

---

【納品物】
- `research/comparison/analysis_report.md` （共通原理・相違点の詳細分析）
- `research/comparison/risk_assessment.md` （リスク登録簿）
- `experiments/EXP009T/validation_plan.md` （検証計画書）
- `docs/04_knowledge.md` 追記案 （「本質的原理」「推奨アクション」）

【スタイル】
- **定量的**：スコア向上の寄与度を数値で推定
- **検証可能**：各仮説を validate で実装前に確認
- **段階的**：リスク最小化のため、小さなPRで検証
- **保守的**：「効く根拠がない工夫」は実装しない