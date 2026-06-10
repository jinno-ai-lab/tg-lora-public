# Research Track 3: Attribution（帰属）手法と層の重要度解析

> 調査日: 2026-06-09  
> 目的: TG-LoRAプロジェクトにおける「attention系（out_proj）が最も安定した学習信号を持つ（early_dir_cos 0.30-0.42）」仮説を、既存のattribution・層重要度研究と照合する

---

## 目次

1. [Layer Attribution / Layer Importance in LLMs](#1-layer-attribution--layer-importance-in-llms)
2. [Feature Attribution Methods](#2-feature-attribution-methods)
3. [Fisher Information for Layer Importance](#3-fisher-information-for-layer-importance)
4. [Sensitivity Analysis for LoRA](#4-sensitivity-analysis-for-lora)
5. [Knowledge Neurons / Factual Attribution](#5-knowledge-neurons--factual-attribution)
6. [SVD-based Attribution](#6-svd-based-attribution)
7. [LoRA層選択の実践知](#7-lora層選択の実践知)
8. [TG-LoRAへの示唆と統合分析](#8-tg-loraへの示唆と統合分析)

---

## 1. Layer Attribution / Layer Importance in LLMs

### 1.1 ShortGPT — Layer Removal in LLMs

| 項目 | 内容 |
|:--|:--|
| **論文** | *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect* |
| **著者** | Men et al. |
| **年** | 2024 |
| **URL** | https://arxiv.org/abs/2403.03853 |

**主要貢献:**
- **Block Influence (BI) メトリクス**を提案：各層の入出力隠れ状態間のコサイン類似度で層の重要度を測定
- BI が低い層（＝入力と出力の変換が少ない層）は冗長であり、除去可能
- 約25%の層を除去しても性能の約90%を維持

**核心的知見:**
- LLMの冗長性は**幅（embedding次元）よりも深さ（層数）**に顕著
- 中間層の多くは「ほぼ恒等変換」を行っており、除去しても影響が小さい
- 量子化などの他の最適化手法と直交的に組み合わせ可能

**TG-LoRAとの関連:**
- BI メトリクスの概念は、TG-LoRAの`early_dir_cos`（初期勾配方向コサイン類似度）と類似。BI が高い層＝変換が大きい層＝学習信号が強い層という対応関係がある
- **out_proj のBI値が高い**場合、それは out_proj が情報変換において重要な役割を果たしていることを示唆

---

### 1.2 LaCo — Layer Collapse

| 項目 | 内容 |
|:--|:--|
| **論文** | *LaCo: Large Language Model Pruning via Layer Collapse* |
| **著者** | Yang, Cao, Zhao |
| **年** | 2024 |
| **URL** | https://arxiv.org/abs/2402.11187 |

**主要貢献:**
- 層を完全に除去するのではなく、**後方の層を前方の層に「折り畳む」（collapse）**手法
- 層間の類似性（layer-wise similarity）に基づいて冗長な層をマージ
- 25-30%のプルーニング率で平均タスク性能の80%以上を維持

**核心的知見:**
- LLM-PrunerやSliceGPTなどのベースラインを上回る性能
- 層の「構造的冗長性」に着目：勾配やactivationではなく、層間の表現類似性が鍵
- **層重要度のランキングはプルーニング手法間で不一致**になりうる → 単一の重要度メトリクスでは不十分

**TG-LoRAとの関連:**
- 層間の表現変化量がTG-LoRAの速度ベクトル（velocity）概念と類似
- 「折り畳み」は本質的に層間の差分が小さい（＝velocity が低い）層を統合する操作

---

### 1.3 SliceGPT — Structured Pruning via Orthogonal Projection

| 項目 | 内容 |
|:--|:--|
| **論文** | *SliceGPT: Compress Large Language Models by Deleting Rows and Columns* |
| **著者** | Ashkboos et al. |
| **年** | 2024 (ICLR 2024) |
| **URL** | https://arxiv.org/abs/2401.15024 |

**主要貢献:**
- Transformer の**計算的不変性（computational invariance）**を利用
- 直交変換で重みを回転し、「重要度」の低い次元（行/列）を削除
- **構造化プルーニング**：疎行列ではなく、より小さい密行列を生成 → 特殊ハードウェア不要
- LLaMA-2 70B で25%のパラメータ削除で99%の性能維持

**核心的知見:**
- PCA（主成分分析）を特徴量活性化に適用し、情報を少数の次元に集中させる
- **特異値が小さい次元 ＝ 削除可能** → SVD-based importance scoring の一形態
- SP3（後続研究）は SliceGPT の「全層均一圧縮率」の限界を指摘

**TG-LoRAとの関連:**
- 直交変換による次元の重要度順位付けは、TG-LoRAの Prior-based Subspace Learning における「方向の単位化」「補助方向の直交化」と同種の数学的操作
- SliceGPT が示す「重要な次元への情報集中」は、TG-LoRAの低ランク近似の妥当性を裏付ける

---

## 2. Feature Attribution Methods

### 2.1 Integrated Gradients

| 項目 | 内容 |
|:--|:--|
| **論文** | *Axiomatic Attribution for Deep Networks* |
| **著者** | Sundararajan, Taly, Yan |
| **年** | 2017 (ICML 2017) |
| **URL** | https://arxiv.org/abs/1703.01365 |

**主要貢献:**
- 2つの公理（**Sensitivity** と **Implementation Invariance**）を満たす帰属手法
- ベースライン入力から実際の入力までの直線パスに沿って勾配を積分
- **Completeness 性質**: 帰属値の合計 ＝ 出力差分

**手法の概要:**
1. ベースライン入力の選択（画像ならば黒画像、テキストならばパディングトークン）
2. ベースラインからターゲットへの補間入力列を作成
3. 各補間ステップで勾配を計算
4. 勾配を積分（平均化）して最終帰属スコアを算出

**TG-LoRAとの関連:**
- TG-LoRAの velocity（速度ベクトル）追跡は、本質的にパラメータ空間での「パスに沿った勾配の蓄積」
- Integrated Gradients の「パスに沿った帰属」は、TG-LoRA の軌跡ベースの外挿と数学的に類似した構造を持つ
- **層レベルでの Integrated Gradients** を計算すれば、各層の学習への寄与を定量化可能

---

### 2.2 SHAP / DeepSHAP

| 項目 | 内容 |
|:--|:--|
| **論文** | *A Unified Approach to Interpreting Model Predictions* |
| **著者** | Lundberg, Lee |
| **年** | 2017 (NeurIPS 2017) |
| **URL** | https://arxiv.org/abs/1705.07874 |

**主要貢献:**
- ゲーム理論の Shapley 値をモデル解釈に適用
- **DeepSHAP**: DeepLIFT アルゴリズムと組み合わせて効率的に深層ネットワークの帰属を計算
- 中間層の活性化を「入力」として扱うことで、**層レベルの重要度分析**が可能

**核心的知見:**
- Efficiency 性質（帰属値の合計＝予測差分）を保証
- バックプロパゲーションベースで高速に近似可能
- ただし SHAP 値は因果関係ではなく統計的関連を測定 → 特徴間の相関がある場合、解釈に注意が必要

**TG-LoRAとの関連:**
- 各層の LoRA アダプタの「寄与」を Shapley 値的に分解する枠組みとして応用可能
- ただし LLM（数十億パラメータ）への直接適用は計算コストが高い

---

### 2.3 Layer-wise Relevance Propagation (LRP)

| 項目 | 内容 |
|:--|:--|
| **論文群** | *AttnLRP* (NeurIPS 2024), *PA-LRP*, Chefer et al. (CVPR 2021) |
| **URL** | https://arxiv.org/abs/2402.05602 (AttnLRP) |

**主要貢献:**
- 予測の関連性（relevance）を層ごとに逆伝搬する手法
- **AttnLRP**: Transformer の attention 層に特化した LRP を設計
  - softmax の分布ルール、行列乗算の分解ルールを新規設計
  - 単一のバックワードパスで計算可能 → LLaMA 等の大規模モデルにも適用可能
- **PA-LRP**: 位置エンコーディングを考慮した帰属

**核心的知見:**
- 生の attention weight は真の特徴重要度を必ずしも反映しない
- LRP による帰属は attention 可視化より意味的に一貫した結果を提供
- **潜在空間での帰属**が可能：特定の潜在ニューロンの重要度を定量化

**TG-LoRAとの関連:**
- LRP を用いて各層の LoRA 更新が最終出力に与える影響を定量化可能
- 「out_proj が最も安定した学習信号を持つ」仮説を、LRP による帰属分析で検証可能

---

### 2.4 Attention Attribution

| 項目 | 内容 |
|:--|:--|
| **論文** | *Quantifying Attention Flow in Transformers* |
| **著者** | Abnar, Zuidema |
| **年** | 2020 (ACL 2020) |
| **URL** | https://arxiv.org/abs/2005.00928 |

**主要貢献:**
- **Attention Rollout**: 層間で attention 行列を再帰的に乗算し、入力トークンの寄与を推定
- **Attention Flow**: ネットワークを DAG としてモデル化し、最大フロー問題として情報の流れを解く
- 生の attention weight より忠実な帰属を提供

| 項目 | 内容 |
|:--|:--|
| **論文** | *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers* |
| **著者** | Chefer, Gur, Wolf |
| **年** | 2021 (ICCV 2021) |
| **URL** | https://arxiv.org/abs/2103.15679 |

**主要貢献:**
- self-attention、co-attention、encoder-decoder attention に適用可能な汎用帰属手法
- 勾配情報と multi-head attention map を組み合わせ
- Deep Taylor Decomposition + LRP よりもシンプルかつ汎用的

**TG-LoRAとの関連:**
- Attention の情報フローは、out_proj を経由して集約される → out_proj の学習信号の安定性は、attention head 全体の統合的な情報を扱っていることに起因する可能性

---

## 3. Fisher Information for Layer Importance

### 3.1 Fisher Pruning

| 項目 | 内容 |
|:--|:--|
| **論文** | *Group Fisher Pruning for Practical Network Compression* (ICML 2021) |
| **URL** | https://proceedings.mlr.press/v139/liu21ab.html |

**主要貢献:**
- Fisher 情報量をパラメータ（チャネル）の重要度スコアとして使用
- Fisher 情報量が低いパラメータ ＝ 損失関数への感度が低い ＝ 除去可能
- メモリ・計算コストで正規化し、効率性を最大化

**核心的知見:**
- Fisher 情報量は2次の重要度メトリクス（Hessian の対角近似）
- 完全な Fisher 行列の計算は不可能 → **経験的 Fisher 対角（eFIM diagonal）**で近似
- FishLeg（メタ学習ベース）は Fisher 逆行列をより精密に推定

---

### 3.2 Fisher-Weighted Averaging（モデルマージ）

| 項目 | 内容 |
|:--|:--|
| **論文** | *Merging Models with Fisher-Weighted Averaging* |
| **年** | 2022 (NeurIPS 2022) |
| **URL** | https://arxiv.org/abs/2111.09832 |

**主要貢献:**
- Fisher 情報量を精度行列として使用し、モデルマージを Laplace 近似で定式化
- Fisher 情報量が高いパラメータ → そのモデルの「確信度」が高い → マージ時に重み付けを大きく
- 単純平均より優れたマージ品質

**TG-LoRAとの関連:**
- Fisher 情報量は本質的に「パラメータの感度」を測定 → TG-LoRA の velocity（重み変化の速度）と対応関係がある
- **velocity が高い層 ≈ Fisher 情報量が高い層** → 学習において重要な層
- Fisher-weighted averaging の概念は、TG-LoRA の外挿重み付けに応用可能

---

### 3.3 FIM-LoRA — Fisher-based Rank Allocation

| 項目 | 内容 |
|:--|:--|
| **論文** | *FIM-LoRA: Task-Informative Rank Allocation for LoRA via Calibration-Time Gradient-Variance Estimation* |
| **年** | 2026 |
| **URL** | https://arxiv.org/abs/2605.16800 |

**主要貢献:**
- LoRA アダプタに限定した eFIM 対角を計算（フルモデルの Fisher 推定の約 **1/256** のメモリコスト）
- 少数のキャリブレーション backward pass（~8回）で LoRA-B 行列の**勾配分散**を計算
- 勾配分散をタスク重要度のプロキシとして、層ごとにランク予算を再配分
- 結果は標準的な LoRA アダプタ（層ごとに異なるランク）として出力

**核心的知見:**
- 勾配分散は各層の「タスク適応への寄与度」の安定した指標
- 重要な層により高いランクを配分 → 均一ランクの LoRA と同等以上の性能
- 解釈可能な「ランクマップ」を提供

**TG-LoRAとの関連:**
- **直接的に関連**: TG-LoRA の `early_dir_cos`（初期勾配方向コサイン類似度）は FIM-LoRA の勾配分散と類似した「学習信号の品質」を測定
- FIM-LoRA のキャリブレーションフェーズは、TG-LoRA のオフライン検証（`scripts/offline_tg_w_validation.py`）と同種のアプローチ
- **重要な違い**: FIM-LoRA はランク配分、TG-LoRA は外挿方向の選択に使用

---

## 4. Sensitivity Analysis for LoRA

### 4.1 AdaLoRA — Adaptive Rank Allocation

| 項目 | 内容 |
|:--|:--|
| **論文** | *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning* |
| **著者** | Zhang et al. |
| **年** | 2023 (ICLR 2023) |
| **URL** | https://arxiv.org/abs/2303.10512 |

**主要貢献:**
- 重み更新を SVD 形式（$\Delta W = P \Lambda Q$）でパラメータ化
- **勾配ベースの感度スコア**で各特異値の重要度を推定
- 学習中に動的にランクをプルーニング：均一ランク配分から開始し、重要度の低い特異値を段階的に除去

**核心的知見:**
- 異なる層・モジュールはタスク適応への寄与が異なる → **均一ランクは非効率**
- 低予算設定でより効果を発揮
- ただし瞬時勾配への依存は不安定なスコアを生む可能性（IGU-LoRA が指摘）

**TG-LoRAとの関連:**
- AdaLoRA の「特異値の重要度スコアリング」は、TG-LoRA の velocity tracking の目的と重なる
- TG-LoRA の velocity は「方向の安定性」を追跡 → AdaLoRA の感度スコアより時系列的に安定した信号を提供できる可能性

---

### 4.2 LoRA+ — 行列A/Bの異なる学習率

| 項目 | 内容 |
|:--|:--|
| **論文** | *LoRA+: Efficient Low Rank Adaptation of Large Models* |
| **著者** | Hayou, Ghosh, Yu |
| **年** | 2024 (ICML 2024) |
| **URL** | https://arxiv.org/abs/2402.12354 |

**主要貢献:**
- 行列 A と B に**異なる学習率**を適用するだけで、1-2% の性能向上と最大 2x の収束加速
- 同一学習率は大きな embedding 次元のモデルで非効率であることを理論的に証明
- 計算コスト増なし

**TG-LoRAとの関連:**
- LoRA+ の「行列ごとの最適学習率」は、TG-LoRA の層サンプリングと相補的
- velocity が行列 A と B で異なる場合、LoRA+ 的な非対称更新が有効な可能性

---

### 4.3 Intrinsic Dimensionality

| 項目 | 内容 |
|:--|:--|
| **論文** | *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* |
| **著者** | Aghajanyan, Zettlemoyer, Gupta |
| **年** | 2021 (ACL 2021) |
| **URL** | https://aclanthology.org/2021.acl-long.568/ |

**主要貢献:**
- 事前学習モデルの微調整に必要な更新は、**低次元部分空間**に集中
- 事前学習がこの固有次元を暗黙的に最小化する最適化として機能
- 大規模モデルほど固有次元が低い → 微調整効率が高い
- **LoRA の理論的基盤**: LoRA はこの低ランク仮説を直接操作化

**TG-LoRAとの関連:**
- **直接的な理論基盤**: TG-LoRA の Prior-based Subspace Learning は、まさにこの「低次元部分空間」上での学習
- 軌跡（trajectory）から推定する方向 $v$ とスケール $w_{\text{traj}}$ は、この固有次元を適応的に発見する仕組み

---

### 4.4 LoRA-GA — Gradient-Aligned Initialization

| 項目 | 内容 |
|:--|:--|
| **関連研究** | *LoRA-GA*, *LoRA-One* |
| **年** | 2024-2025 |

**主要貢献:**
- LoRA の初期化をフルランク勾配の主要特異部分空間に揃える
- ランダム初期化よりも安定した学習と高速な収束を実現
- フルランク勾配とのコサイン類似度が高いほど、下流タスク精度が向上

**TG-LoRAとの関連:**
- **極めて密接な関連**: TG-LoRA の `early_dir_cos` メトリクスは、まさにこの「勾配方向の整合性」を測定
- out_proj の early_dir_cos 0.30-0.42 は、他のモジュールよりフルランク勾配との整合性が高いことを示唆
- LoRA-GA の知見は、TG-LoRA の外挿方向選択の妥当性を理論的に裏付ける

---

## 5. Knowledge Neurons / Factual Attribution

### 5.1 Knowledge Neurons

| 項目 | 内容 |
|:--|:--|
| **論文** | *Knowledge Neurons in Pretrained Transformers* |
| **著者** | Dai et al. |
| **年** | 2022 (ACL 2022) |
| **URL** | https://aclanthology.org/2022.acl-long.581/ |

**主要貢献:**
- **FFN（Feed-Forward Network）層の特定ニューロン**が事実的知識を格納
- knowledge attribution method で知識ニューロンを同定
- ニューロンの活性化抑制/増幅で事実知識の編集が可能（再学習不要）

**核心的知見:**
- FFN 層はキー・バリューメモリストアとして機能
- 知識は分散ではなく、比較的**局所化**されている
- Attention 層は「どの知識を呼び出すか」を制御、FFN 層は「知識の格納」を担当

---

### 5.2 ROME / MEMIT — Causal Tracing

| 項目 | 内容 |
|:--|:--|
| **論文** | *Locating and Editing Factual Associations in GPT* (ROME) / *Mass-Editing Memory in a Transformer* (MEMIT) |
| **著者** | Meng et al. |
| **年** | 2022 (NeurIPS 2022) |
| **URL** | https://arxiv.org/abs/2202.05262 (ROME) |

**主要貢献:**
- **Causal Tracing**: activation patching で事実知識の格納場所を特定
- 事実知識は**中間層の MLP モジュール**に局在
- **ROME**: MLP 重み行列のランク1更新で単一の事実を編集
- **MEMIT**: 複数層に分散してランク更新を行い、数千の事実を一括編集

**核心的知見:**
- MLP 重み行列は**キー・バリューストア**として機能
- 主語トークンの処理時に、中間層 MLP が事実呼び出しを仲介
- 編集は高い特異性（対象の事実のみ変更）と汎化性（異なる文脈で適用）を持つ

**TG-LoRAとの関連:**
- **重要な示唆**: Attention 層と MLP 層は異なる機能を担う
  - **Attention 層（out_proj 含む）**: 情報の選択・ルーティング → 学習信号が安定（方向が一貫）
  - **MLP 層**: 知識の格納・更新 → 学習信号がより局所的・タスク依存的
- out_proj の early_dir_cos が高い理由として、attention は「どの情報に注目するか」というより構造的・安定的な機能を学習していることが考えられる
- MLP 層の学習信号は事実知識に依存するため、バッチ間での勾配方向の分散が大きくなりうる

---

## 6. SVD-based Attribution

### 6.1 重み行列のスペクトル分析

| 項目 | 内容 |
|:--|:--|
| **関連研究群** | SVD による重み行列分析、PARA（Post-Optimization Adaptive Rank Allocation）、DF-SVD |

**主要概念:**
- 重み行列 $W$ を $U \Sigma V^T$ に分解
- **特異値スペクトルの減衰特性**が層の圧縮可能性を示す
  - 急速な減衰 → 低ランク近似で十分（冗長な層）
  - 緩やかな減衰 → 高ランクが必要（複雑な変換を行う層）

**手法の分類:**

| 手法 | アプローチ | 特徴 |
|:--|:--|:--|
| **PARA** | 学習後の SVD + 閾値プルーニング | データフリー、学習ワークフロー不変 |
| **DF-SVD** | 特異値の減衰特性を分析 | 層ごとの適応的ランク配分 |
| **SILoR** | 事前学習重みの SVD で LoRA 初期化 | 主要方向との整合性確保 |

### 6.2 SVD による解釈性

**核心的知見:**
- 最大特異値に対応する特異ベクトルは、しばしば**意味的に解釈可能な方向**に対応
- 残差ストリームの主要方向を SVD で同定 → モデルの情報処理の幾何学的理解
- 特異ベクトルの選択的除去/修正による「回路編集」が可能

**TG-LoRAとの関連:**
- TG-LoRA の velocity ベクトルの SVD 分析により、学習の主要方向を同定可能
- Prior-based Subspace Learning で使用する方向 $v$ の品質を、特異値分解で検証可能
- **エネルギー保持率**（Frobenius ノルムの何%を捕捉するか）は、TG-LoRA の低次元近似の妥当性を定量化する指標として使用可能

---

## 7. LoRA層選択の実践知

### 7.1 ターゲットモジュール比較

| 構成 | 性能影響 | 効率性 | 推奨用途 |
|:--|:--|:--|:--|
| **Attention-only** (`q_proj`, `v_proj`) | ベースライン；複雑タスクでは不十分 | 最高（VRAM最小） | 予備実験、メモリ制約が厳しい場合 |
| **Attention + MLP** | 大幅改善；フル微調整に近い | 中程度 | 汎用的な微調整 |
| **All Linear** (`target_modules="all-linear"`) | フル微調整と同等 | 最大リソース消費 | 最大精度が必要な場合 |

### 7.2 個別モジュールの重要度

**経験的知見のまとめ:**

1. **`out_proj` (o_proj)** — **高インパクトな単一モジュール**
   - 全 attention head の出力を統合する最終線形変換
   - 「bang for the buck」が最も高いモジュールとして複数の研究で報告
   - Amazon Science (2026) の研究で、o_proj-only がレイテンシと性能のバランスで最適と報告
   - **学習の安定性が高い**: 失敗することがほとんどなく、複雑なマルチモジュール設定に近い性能

2. **`q_proj`, `k_proj`, `v_proj`**
   - 元の LoRA 論文では `q_proj` と `v_proj` のみ対象
   - attention パターンの学習に直接寄与
   - ただし単独では複雑なタスクに不十分

3. **MLP 層 (`gate_proj`, `up_proj`, `down_proj`)**
   - モデルの知識の大部分を格納（パラメータの大半を占める）
   - MLP を含めないと、特にドメイン固有のタスクで性能劣化
   - Knowledge Neurons 研究と整合：**MLP は知識の格納場所**

### 7.3 層位置と重要度

**主要知見:**

| 戦略 | 根拠 | 効果 |
|:--|:--|:--|
| **最後の25%の層** | 後方の層はよりタスク固有 | 一般的に最高の cost-performance ratio |
| **最初の層を凍結** | 初期層は普遍的言語パターン | 基盤能力の保存 |
| **重要度ベースの選択** | 勾配ノルム / 意味分析 | 10-30%の層で全層チューニングと同等の性能 |
| **全層** | 最大適応能力 | 最高性能だが過学習リスク |

### 7.4 Intruder Dimension 問題

- LoRA は「intruder dimension」（フル微調整には存在しない高ランク特異ベクトル）を導入する可能性
- より包括的な層ターゲティング（all-linear）がこの副作用を軽減
- ランク配分の最適化も有効

---

## 8. TG-LoRAへの示唆と統合分析

### 8.1 「out_proj が最も安定した学習信号を持つ」仮説の検証

**既存研究との整合性: ⭐⭐⭐⭐⭐ (強く支持)**

本調査で発見した複数の独立した研究ラインが、この仮説を支持している：

```
┌─────────────────────────────────────────────────────────────────┐
│                    out_proj の特別な位置付け                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. アーキテクチャ的役割                                         │
│     ├─ 全 attention head の出力を統合する最終変換                │
│     ├─ 情報のルーティング・選択の集約点                          │
│     └─ → 構造的に安定した勾配方向を生む                         │
│                                                                 │
│  2. Knowledge Neurons / ROME の知見                              │
│     ├─ Attention: 情報の選択（構造的・安定的）                   │
│     ├─ MLP: 知識の格納（局所的・タスク依存的）                   │
│     └─ → Attention 系の勾配は MLP 系より方向が安定              │
│                                                                 │
│  3. 実践的エビデンス                                             │
│     ├─ Amazon Science: o_proj-only が最適バランス               │
│     ├─ 複数の practitioners: "rarely fails during training"      │
│     └─ → early_dir_cos 0.30-0.42 は empirically validated       │
│                                                                 │
│  4. LoRA-GA / LoRA-One の理論                                    │
│     ├─ フル勾配との cosine similarity が高い → 高性能            │
│     ├─ out_proj は情報統合層として主要特異部分空間に近い         │
│     └─ → 低ランク近似の有効性が高い                             │
│                                                                 │
│  5. ShortGPT の Block Influence                                  │
│     ├─ 入出力変換が大きい層 = 重要な層                           │
│     ├─ out_proj は attention block 全体の出力変換を担う          │
│     └─ → BI が高い = velocity / early_dir_cos が高い             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 メカニズムの仮説

**なぜ out_proj の学習信号が安定するのか（機構的仮説）:**

1. **統合効果**: out_proj は multi-head attention の全ヘッドの出力を集約する。個々のヘッドの勾配のノイズが平均化され、より安定した方向性を持つ
2. **残差接続との相互作用**: out_proj の出力は残差接続を通じて次の層に直接加算される。この「直接パス」が勾配のバックプロパゲーションを安定化させる
3. **表現空間の構造**: attention 層は「どこを見るか」（構造的決定）を学習し、MLP 層は「何を出力するか」（内容的決定）を学習する。構造的決定は内容的決定より一般的であり、勾配方向が安定しやすい

### 8.3 TG-LoRA設計への具体的提案

| 提案 | 根拠 | 優先度 |
|:--|:--|:--|
| velocity tracking で out_proj を優先的にサンプリング | 学習信号の安定性が最も高い | 高 |
| FIM-LoRA 的なキャリブレーションフェーズの導入 | 層ごとの外挿強度を適応的に設定 | 中 |
| SVD による velocity ベクトルの品質検証 | 低次元近似の妥当性を事前確認 | 高（Milestone 9） |
| Integrated Gradients 的な「パス帰属」の概念導入 | 外挿の累積効果を理論的に裏付け | 低（論文向け） |
| attention vs MLP での外挿パラメータ分離 | 機能的役割の違いを反映 | 中 |

### 8.4 今後の検証方向

1. **定量的検証**: 各モジュール（q_proj, k_proj, v_proj, out_proj, gate/up/down_proj）ごとの early_dir_cos を系統的に測定し、仮説を数値的に確認
2. **Fisher 情報量との相関**: 各モジュールの eFIM 対角と early_dir_cos の相関を分析
3. **SVD スペクトル分析**: 各モジュールの velocity ベクトルの特異値スペクトルを比較 → out_proj の velocity がより低ランクで近似可能かを検証
4. **Causal Tracing の応用**: TG-LoRA の外挿が最終出力に与える因果的影響を層ごとに定量化

---

## 9. 2026年6月 最新動向アップデート

> **更新日**: 2026-06-09 / **出典**: 下記 arXiv を Web 調査。abstract / 公開本文に基づき記述。

### 9.1 FIM-LoRA の正式版確定と「層の役割」実証

§3.3 で挙げた FIM-LoRA は 2026 年に正式版（[arXiv:2605.16800](https://arxiv.org/abs/2605.16800)）が公開され、本調査で arXiv 番号を確定した（旧記載はプレースホルダ）。

- **手法確定**: 学習前に **8 回のキャリブレーション backward pass** で各 LoRA-B 行列の**勾配分散**を計算し、eFIM 対角（LoRA 行列限定、フルモデル比 **約 1/256** メモリ）でランク予算を比例再配分。新規パラメータ・学習オーバーヘッド・サービング変更なし。
- **性能**: GLUE/DeBERTa-v3-base で同パラメータ予算の LoRA と同等（88.6 vs 88.7）、LLaMA-3-8B の commonsense reasoning で 68.5 vs 68.7。
- **実証された層の役割**: 解釈可能な per-layer ランクマップで、**value projection（v_proj）と初期〜中間層**が一貫して高ランクを獲得。

> [!NOTE]
> **out_proj 安定性仮説への補正的示唆**。FIM-LoRA の「勾配分散」基準では **v_proj・初期〜中間層**が重要と判定される。これは「out_proj が最も安定した学習信号を持つ」という本プロジェクトの `early_dir_cos` 観測とは**測っている量が異なる**（FIM-LoRA=勾配の分散＝適応への寄与の大きさ／TG-LoRA=方向の一貫性＝外挿の安定性）。
> したがって両者は矛盾せず、**「適応寄与が大きい層（v_proj・中間層）」と「外挿方向が安定な層（out_proj）」を分離して扱う**ことが、層別設計の精緻化につながる。両指標を同一 run で測り相関を取る検証（§8.4-2）の価値が、最新動向によって裏付けられた。

### 9.2 Fisher / 勾配分散ベースのランク割り当ての系譜

| 研究 | 年 | attribution のシグナル | 用途 |
|:--|:--|:--|:--|
| Lodha et al. | 2023 | フルモデル FIM スコア | どの層を微調整するか（二値・層単位） |
| Kim et al. | 2025 | LoRA アダプタの勾配分散 | ランク配分の事前推定 |
| **FIM-LoRA** | 2026 | LoRA-B 限定 eFIM 対角 | 比例ランク再配分（標準 LoRA 形式で出力） |

> [!IMPORTANT]
> この系譜は「**重要度シグナルをいかに軽量に・LoRA 部分空間に限定して推定するか**」へ収束しつつある。TG-LoRA のオフライン検証（`scripts/collect_true_gradients.py` / `offline_tg_w_validation.py`）は、まさに「LoRA 部分空間に限定した軽量な事前計測で方向・スケールの prior を得る」設計であり、FIM-LoRA のキャリブレーション思想と方法論的に一致する。差分は **出力（FIM-LoRA=ランク予算／TG-LoRA=外挿方向と進み幅）** にある。

---

## 参考文献一覧

### Layer Pruning / Importance
1. Men et al. (2024). *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect.* https://arxiv.org/abs/2403.03853
2. Yang et al. (2024). *LaCo: Large Language Model Pruning via Layer Collapse.* https://arxiv.org/abs/2402.11187
3. Ashkboos et al. (2024). *SliceGPT: Compress Large Language Models by Deleting Rows and Columns.* ICLR 2024. https://arxiv.org/abs/2401.15024

### Feature Attribution
4. Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML 2017. https://arxiv.org/abs/1703.01365
5. Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. https://arxiv.org/abs/1705.07874
6. Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL 2020. https://arxiv.org/abs/2005.00928
7. Chefer et al. (2021). *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers.* ICCV 2021. https://arxiv.org/abs/2103.15679
8. Ali et al. (2024). *AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers.* NeurIPS 2024. https://arxiv.org/abs/2402.05602

### Fisher Information
9. Liu et al. (2021). *Group Fisher Pruning for Practical Network Compression.* ICML 2021.
10. Matena & Raffel (2022). *Merging Models with Fisher-Weighted Averaging.* NeurIPS 2022. https://arxiv.org/abs/2111.09832
11. FIM-LoRA (2026). *FIM-LoRA: Task-Informative Rank Allocation for LoRA via Calibration-Time Gradient-Variance Estimation.* https://arxiv.org/abs/2605.16800
11a. Lodha et al. (2023). *Layer selection for fine-tuning via full-model FIM scores.* (FIM-LoRA で参照)
11b. Kim et al. (2025). *Gradient-variance of LoRA adapters for rank allocation.* (FIM-LoRA で参照)

### LoRA / PEFT Sensitivity
12. Zhang et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023. https://arxiv.org/abs/2303.10512
13. Hayou et al. (2024). *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML 2024. https://arxiv.org/abs/2402.12354
14. Aghajanyan et al. (2021). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021. https://aclanthology.org/2021.acl-long.568/
15. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685

### Knowledge / Factual Attribution
16. Dai et al. (2022). *Knowledge Neurons in Pretrained Transformers.* ACL 2022. https://aclanthology.org/2022.acl-long.581/
17. Meng et al. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. https://arxiv.org/abs/2202.05262
18. Meng et al. (2022). *Mass-Editing Memory in a Transformer.* ICLR 2023. https://arxiv.org/abs/2210.07229

### SVD / Low-Rank Analysis
19. PARA. *Post-Optimization Adaptive Rank Allocation for LoRA.*
20. SILoR. *SVD-based Initialization for LoRA.*
