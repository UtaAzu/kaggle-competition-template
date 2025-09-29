# 初期戦略プラン（Jigsaw - Agile Community Rules Classification）

本プランは、以下の情報に基づいて作成しています。
- 公式ドキュメント（docs/01_overview.md, docs/02_data_description.md）
- 評価: column-averaged AUC
- タスク: 行単位で「特定のルールに対して、コメントが違反した確率」を推定するバイナリ分類
- 重要制約: trainは「2つのルール」のみ、testは「未見の追加ルール」を含む。行には rule と 例示（positive_example_{1,2}, negative_example_{1,2}）が付く

---

## 1. ドメイン知識の分析 (Domain Knowledge Analysis)

- [P0] ルール記述と事例に基づく「ルール条件付き推論」
  - なぜ重要か: 未見ルールへの一般化が本質課題。コメント単体の毒性判定ではなく、「与えられたルール（＋正負の少数例）」に対する違反確率を推論する必要がある。入力設計（テンプレート化）と「例示の取り込み」が精度を左右。
- [P0] サブレディット間の文体・規範の多様性（分布ずれ）
  - なぜ重要か: subredditにより言い回し・語彙・規範が異なる。CVはサブレディットやルールでリークを防ぎつつ、分布ずれに強い汎化を測る設計が必要（GroupKFoldや擬似ルールホールドアウトなど）。
- [P1] クラス不均衡と確率キャリブレーション
  - なぜ重要か: 違反は相対的に少ない想定。AUC最適を主眼としつつ、学習安定化（class_weight/損失設計）と、fold間のスコアばらつきを抑える再現性確保が重要。

---

## 2. 機械学習アプローチの分析 (Machine Learning Approach Analysis)

- 最優先アプローチ: ルール条件付きクロスエンコーダ（BERT/DeBERTa/RoBERTa系）
  - 方式: 「Rule + 例示 + Comment」を単一入力シーケンスとしてエンコードし、違反確率を出す2値分類を微調整
  - 入力テンプレ例:
    - [RULE] {rule}
    - [POS1] {positive_example_1}
    - [POS2] {positive_example_2}
    - [NEG1] {negative_example_1}
    - [NEG2] {negative_example_2}
    - [COMMENT] {body}
    - [Q] Does the comment violate the rule? → [LABEL]
  - 技術ポイント:
    - モデル: roberta-base or deberta-v3-base（多様語彙に強い）。長文なら longformer/distil系も検討
    - 損失/不均衡: BCEWithLogits + pos_weight or focal（まずはBCE＋class_weight）
    - 正則化: ラベルスムージング、勾配クリッピング、multi-sample dropout
    - CV: 2系統で検証
      1) StratifiedKFold（target比率）＋GroupKFold（subreddit）併用の優先設計
      2) 疑似LOO-Rule検証（片方のルールで学習→もう片方で評価）で未見ルール耐性をチェック
    - 推論: 行ごとの提供例示を必ず取り込む（train/test整合）
- 次点アプローチ: TF-IDF/BM25 + ロジスティック回帰（あるいはLightGBM）の高速ベースライン
  - 方式: 
    - テキストTF-IDF（word 1-2, char 3-5）をbodyに適用
    - ルール記述・例示に対するBM25/コサイン類似度など「ルール適合度」特徴量を追加（comment↔positive/negative例の距離・最大/平均）
    - メタ特徴: 文字長/記号率/サブレディットOneHot（高次元になりすぎない範囲）
  - 理由: 超高速でCV設計やリーク点検を回す土台。AUCを素早く確保し、上位モデルの改善幅を測る。
- 挑戦的アプローチ: デュアルエンコーダ＋原型（プロトタイプ）類似度 or PEFT-LoRAでの指示追従LLM微調整
  - 方式:
    - デュアル: f_comment(body) と f_rule(rule+ex) を別エンコーダで埋め込み、cos類似度→確率
    - LLM-PEFT: ルールと例示をプロンプトに埋め込み、LoRAで2値ヘッド学習（VRAM制約に注意）
  - 利点/欠点: スケールしやすく未見ルールへ頑健だが、データサイズ少のため過学習/不安定性に注意。まずはクロスエンコーダで堅実に。

---

## 3. 過去コンペからの知見 (Insights from Past Competitions)

- 類似コンペ1: Jigsaw/Conversation AI（Toxic/Unintended Bias）
  - 応用可能なアイデア:
    - char n-gramや正規化の効きやすさ（スラング/伏字耐性）
    - しきい値最適化・確率較正・外部データ利用の慎重な扱い
    - サブグループ間の分布差に敏感なCV設計（今回はsubredditやrule単位）
- 類似コンペ2: Contradictory, My Dear Watson（テキストペア推論）
  - 応用可能なアイデア:
    - テキストペア（仮説-前提）のジョイントエンコード（クロスエンコーダ）が効果的
    - テンプレ設計と前処理（区切りトークン、セクション見出し）が安定化に寄与
- 補足: Quora Insincere Questions（不均衡バイナリ）
  - 応用可能なアイデア:
    - 不均衡データへのclass_weightや適切な負例サンプリング
    - 単純TF-IDF＋線形の強さと、リーク点検の重要性

---

## 4. 初期アクションプラン (Initial Action Plan)

- Step 1: 信頼できるCVと評価関数を用意（P0）
  - 目的: 未見ルール・サブレディット分布差に対して過学習を防ぎ、LBと整合的なローカル評価を確立
  - アクション:
    - データロードと列名固定（body, rule, subreddit, positive_example_{1,2}, negative_example_{1,2}, rule_violation）
    - CV-1: StratifiedKFold(5, shuffle, seed) ただしGroup=subreddit（GroupStratifiedがなければ「Stratified後にfold内でGroup崩さない工夫」）
    - CV-2: 疑似LOO-Rule検証（trainに2ルールしかないため、Rule Aで学習→Rule B評価、Rule Bで学習→Rule A評価）を併記
    - 指標: AUC（fold別/平均）、ルール毎AUC、subreddit毎AUC（監視用）
    - ロギング: OOF保存・cv_summary.json化・乱数固定
  - 成果物: ベース評価パイプ（OOF出力、fold別指標、疑似LOO-Rule指標）
- Step 2: 高速ベースライン（TF-IDF + LR/LightGBM）（P0）
  - 目的: 数分で回る強固な土台を作り、以降の改善でCV/LBのブースト量を可視化
  - アクション:
    - bodyのTF-IDF（word 1-2 + char 3-5）、最小限のクリーニング（URL/mention正規化、連続記号圧縮）
    - 「ルール適合度」特徴: 
      - bodyとruleのBM25/コサイン類似度
      - bodyとpositive_example_{1,2}の類似度（max/mean）、negative_example_{1,2}との差
    - 分類器: ロジスティック回帰(class_weight="balanced") をまず採用、代替でLightGBM
    - 出力: OOF, cv_summary.json, 重要特徴の確認（木モデル時）
  - 成果物: ベンチマークAUC（fold平均/ルール別/Subreddit別）
- Step 3: クロスエンコーダの導入（RoBERTa/DeBERTa）（P1）
  - 目的: 未見ルール対応の本命モデルを早期に立ち上げ、CV-1/CV-2の両軸で優位性を確認
  - アクション:
    - テンプレ構築（RULE, POS/NEG例, COMMENTのセクション化）、最大長設定（例示含むため512〜1024で調整）
    - 学習設定: 2-3エポック, lr=2e-5〜5e-5, バッチ小＋勾配累積, BCE+pos_weight, seed固定を複数回
    - 早期停止/モデル選択はOOF AUC基準、foldごと保存
    - 推論はfold平均、必要に応じ温度スケーリング等を追加
  - 成果物: Transformer系OOFと比較レポート（TF-IDF対比、疑似LOO-Ruleでの優位性）

---

## リスクと対策

- 未見ルールへの過適合
  - 対策: 疑似LOO-Ruleで常時監視。テンプレ過剰最適化を避け、例示の順序/表現ランダム化でロバスト化。
- サブレディット偏り
  - 対策: Group考慮CV、adversarial validationで分布差の可視化、重み付け検討。
- 入力長溢れ（例示＋本文でトークン超過）
  - 対策: 優先順位付きトリミング（まず本文、次に例示の順で長さ調整ポリシーを明確化）、long-seqモデル検討。

---

## ToDoチェックリスト（初心者向け）

- [ ] データ列名の確認と固定（body/rule/subreddit/pos/neg/target）
- [ ] CV-1（Stratified+Group）とCV-2（疑似LOO-Rule）を実装、AUC出力
- [ ] TF-IDF+LR/LightGBMでOOF作成・保存（ベースライン確立）
- [ ] クロスエンコーダ学習（小さめモデルから、OOFとLOO-Rule比較）
- [ ] 提出スクリプト雛形（row_id, rule_violation）準備
- [ ] 実験ログ（cv_summary.json, READMEのScoreboard更新）

以上。必要に応じ、ベースライン実装・PR作成・Issue切り出しまで自動化します.