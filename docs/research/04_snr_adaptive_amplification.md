# Research Track 4: SNRマップ・適応的増幅・選択的学習 — 先行研究調査

> **調査日**: 2026-06-09  
> **目的**: 各層のA/B行列のモード構造から「安定な層・モードにゲインをかけ、ノイズ通過層を抑える」適応的増幅の地図を構築するための理論的基盤を整理する  
> **TG-LoRA文脈**:
> - Bフィルタ仮説: `B_{t-1} @ ΔA_t` の安定性が0.99と極めて高い
> - rank-1支配度が0.78で安定 → 各テンソル内の更新は実質1次元に集中
> - 目標: SNRマップに基づく層ごと・モードごとの適応的ゲイン制御

---

## 目次

1. [Layer-wise Adaptive Learning Rate](#1-layer-wise-adaptive-learning-rate)
2. [Gradient Signal-to-Noise Ratio (SNR)](#2-gradient-signal-to-noise-ratio-snr)
3. [Selective / Sparse Training](#3-selective--sparse-training)
4. [Layer Freezing / Progressive Training](#4-layer-freezing--progressive-training)
5. [Mode Analysis in Neural Networks](#5-mode-analysis-in-neural-networks)
6. [Noise Filtering in Gradient Updates](#6-noise-filtering-in-gradient-updates)
7. [Adaptive Rank Methods Beyond LoRA](#7-adaptive-rank-methods-beyond-lora)
8. [LoRA B-matrix as Filter](#8-lora-b-matrix-as-filter)
9. [TG-LoRAへの統合提案](#9-tg-loraへの統合提案)

---

## 1. Layer-wise Adaptive Learning Rate

### 1.1 LARS (Layer-wise Adaptive Rate Scaling)

| 項目 | 内容 |
|------|------|
| **論文** | "Large Batch Training of Convolutional Networks" |
| **著者** | Yang You, Igor Gitman, Boris Ginsburg |
| **年** | 2017 |
| **URL** | https://arxiv.org/abs/1708.03888 |
| **会議** | AAAI 2017 / arXiv preprint |

**核心的アイデア**: 層ごとの重みノルムと勾配ノルムの比率が大きく異なることを観測。単一のグローバル学習率では、ある層には大きすぎ、別の層には小さすぎる問題を解決。

**手法**:
- 各層 $l$ の学習率を $\eta_l = \eta_{global} \times \frac{\|W_l\|}{\|\nabla W_l\|}$ でスケーリング
- 勾配を正規化し重みノルムでスケーリングすることで、重み更新の大きさを勾配の大きさから切り離す
- ResNet-50でバッチサイズ32Kまでの大規模バッチ学習を安定化

**TG-LoRAへの関連性**: ★★★★★
- **直接的に応用可能**: 各LoRA層の `||B_l||/||∇B_l||` 比率をSNRマップの一部として利用できる
- 層ごとの勾配-重みノルム比は、我々の「安定な層 vs ノイズ通過層」の判別に直結する
- rank-1支配度が高い層 = 更新方向が安定 = LARS的な適応率を大きくできる層

---

### 1.2 LAMB (Layer-wise Adaptive Moments for Batch training)

| 項目 | 内容 |
|------|------|
| **論文** | "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" |
| **著者** | Yang You et al. |
| **年** | 2020 (ICLR 2020) |
| **URL** | https://arxiv.org/abs/1904.00962 |

**核心的アイデア**: LARSをAdamに統合。層ごとの信頼比率（trust ratio）とAdamの適応的モーメントを組み合わせ。

**手法**:
- LARSとは異なり、SGDではなくAdam/AdamWベース
- 各層の trust ratio = `||W_l|| / ||update_l||` で更新をスケーリング
- BERT事前学習のバッチサイズを32K以上に拡大、学習時間を3日→76分に短縮

**TG-LoRAへの関連性**: ★★★★☆
- TG-LoRAのAdamW最適化器に層ごとのtrust ratioを導入する具体的な先例
- 安定な層にはより積極的なステップを許容、不安定な層は抑制する制御則

---

### 1.3 Layer-wise Learning Rate Decay (LLRD)

| 項目 | 内容 |
|------|------|
| **手法** | Transformer Fine-tuning Training Recipe |
| **起源** | BERTfine-tuning best practices; BEiT (Bao et al., 2022) |

**核心的アイデア**: モデルの深さに応じて異なる学習率を割り当て。最上層（出力側）に高い学習率、最下層（入力側）に低い学習率を設定。

**計算式**: $LR_{layer} = LR_{base} \times decay^{distance\_from\_top}$

**典型的設定**:
- Decay rate: 0.9–0.95
- Base LR: 1e-5 ~ 1e-4
- Cosine decay scheduleと組み合わせるのが標準的

**TG-LoRAへの関連性**: ★★★★☆
- 現在の均一学習率から層ごとの適応学習率への移行の最もシンプルな出発点
- SNRマップに基づくdecay rateの動的決定が自然な拡張方向
- 我々の観測: 層ごとのrank-1支配度の差異 → decay rateの自動調整に利用可能

---

### 1.4 LoRA+ (Different Learning Rates for A and B)

| 項目 | 内容 |
|------|------|
| **論文** | "LoRA+: Efficient Low Rank Adaptation of Large Models" |
| **著者** | Soufiane Hayou et al. |
| **年** | 2024 (ICML 2024) |
| **URL** | https://arxiv.org/abs/2402.12354 |

**核心的アイデア**: LoRAのA行列とB行列に異なる学習率を適用すべき。標準LoRAの同一学習率は最適ではない。

**手法**:
- $\eta_B \gg \eta_A$ が推奨（B行列により大きな学習率）
- ハイパーパラメータ $\lambda = \eta_B / \eta_A$ を導入
- 性能1-2%向上、収束速度最大2倍高速化、計算コスト増加なし

**TG-LoRAへの関連性**: ★★★★★
- **Bフィルタ仮説と直接的に整合**: B行列がフィルタとして機能するなら、Bの学習率をAより大きくすることでフィルタの適応を優先する理論的根拠
- SNRマップでA/B行列それぞれのSNRを計算し、異なるゲインを適用する設計に直結
- rank-1支配度が高い = 信号方向が明確 → $\lambda$ を大きく設定可能

---

### 1.5 ALLoRA (Adaptive Learning Rate for LoRA)

| 項目 | 内容 |
|------|------|
| **論文** | "ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws" |
| **年** | 2024 |
| **URL** | https://arxiv.org/abs/2407.11502 |

**核心的アイデア**: 勾配のℓ2ノルムに基づき、サンプルごと・パラメータごとに学習率を適応的にスケーリング。スケーリングファクターやドロップアウト率の手動チューニングを不要にする。

**TG-LoRAへの関連性**: ★★★★☆
- SNRに基づく自動ゲイン制御の直接的な先行研究
- TG-LoRAではさらにモード構造（rank-1方向）に基づくゲイン制御を追加可能

---

## 2. Gradient Signal-to-Noise Ratio (SNR)

### 2.1 GSNR and Generalization

| 項目 | 内容 |
|------|------|
| **論文** | "Understanding Why Neural Networks Generalize Well Through GSNR of Parameters" |
| **著者** | Liu et al. |
| **年** | 2020 (ICLR 2020) |
| **URL** | https://openreview.net/forum?id=HyevIJStwH |

**核心的アイデア**: 勾配のSignal-to-Noise Ratio (GSNR) = 勾配期待値の二乗 / 勾配の分散。高いGSNRは良い汎化性能と理論的・実験的に関連。

**定義**: $GSNR = \frac{(\mathbb{E}[\nabla L])^2}{Var[\nabla L]}$

**主要知見**:
- 高GSNRパラメータは汎化に重要な「信号」を持つ
- 低GSNRパラメータはノイズに支配され、更新が効果的でない
- GSNRはバッチサイズ、学習率、アーキテクチャに依存

**TG-LoRAへの関連性**: ★★★★★
- **SNRマップの理論的基盤**: 各LoRA層・各モードのGSNRを計算し、高GSNRにゲインをかける設計の直接的根拠
- Bフィルタ安定性 0.99 → B行列を通過した信号のGSNRが高い（フィルタリング効果の定量化）
- rank-1支配度 0.78 → 更新方向の分散が小さい = 高GSNR

---

### 2.2 Gradient Noise Scale / Critical Batch Size

| 項目 | 内容 |
|------|------|
| **論文** | "An Empirical Model of Large-Batch Training" |
| **著者** | Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI |
| **年** | 2018 |
| **URL** | https://arxiv.org/abs/1812.06162 |

**核心的アイデア**: 勾配ノイズスケール（GNS）を定義し、最適バッチサイズ（臨界バッチサイズ, CBS）を予測。

**定義**: $B_{noise} = \frac{tr(Cov[\nabla L])}{||\mathbb{E}[\nabla L]||^2}$

**主要知見**:
- GNSは学習が進むにつれて増加（低lossになるほどSNR低下）
- CBS未満ではバッチサイズ増加がほぼ線形にスピードアップ
- CBS超過では追加のサンプルがデータ効率を犠牲にする
- GPT-3等の大規模学習でバッチサイズウォームアップの根拠

**TG-LoRAへの関連性**: ★★★★☆
- 層ごとのGNSが異なる → 層ごとに最適な「有効バッチサイズ」が異なる概念
- SNRが低い層（ノイズ通過層）は勾配累積ステップを増やすか、学習率を下げるべき
- TG-LoRAの外挿ステップ数（K）を層ごとのGNSで適応的に決定する設計

---

### 2.3 Understanding the Difficulty of Training Transformers

| 項目 | 内容 |
|------|------|
| **論文** | "Understanding the Difficulty of Training Transformers" |
| **著者** | Liyuan Liu et al. |
| **年** | 2020 (ACL 2020) |
| **URL** | https://arxiv.org/abs/2004.08249 |

**核心的アイデア**: Transformerの学習不安定性の根本原因は不均衡な勾配ではなく、残差枝への過度な依存による「増幅効果」。

**主要知見**:
- 初期化時のresidual branch依存が小さなパラメータ更新をモデル出力に大きく増幅
- **Admin** (Adaptive Model Initialization) を提案: 残差枝依存を初期化時に制御
- 勾配のバランスよりも、信号の伝搬パスの安定性が重要

**TG-LoRAへの関連性**: ★★★☆☆
- 増幅効果の概念はTG-LoRAの「安定な層にゲインをかける」設計と対照的
- 不安定な層への増幅はかえって害になる → SNRマップで増幅対象を選別する必要性

---

### 2.4 Spectrum: SNR-based Selective Layer Training

| 項目 | 内容 |
|------|------|
| **論文** | "Spectrum: Targeted Training on Signal to Noise Ratio" |
| **年** | 2024 |
| **URL** | https://arxiv.org/abs/2406.06623 |

**核心的アイデア**: ランダム行列理論（Marchenko-Pastur分布）を用いてモジュールレベルのSNRを事前計算し、「情報的な」層のみを学習対象として選択。

**手法**:
- 各重み行列の特異値分布をMP分布と比較
- MP分布から逸脱する特異値を「信号」、分布内を「ノイズ」と判定
- 信号リッチな層のみ更新、冗長な層は凍結
- フルファインチューニングと同等の性能を低GPUメモリで達成

**TG-LoRAへの関連性**: ★★★★★
- **最も直接的な先行研究**: TG-LoRAのSNRマップ構想とほぼ同一のアプローチ
- SpectrumはベースモデルのSNRを見るが、TG-LoRAではLoRA更新のSNR（BΔA）をターゲット
- ランダム行列理論的アプローチはrank-1支配度の理論的裏付けにも利用可能

---

### 2.5 MoLS: Module-wise Learning Rate Scaling via SNR

| 項目 | 内容 |
|------|------|
| **論文** | "Revealing Modular Gradient Noise Imbalance in LLMs: Calibrating Module-wise Learning Rate via SNR" (MoLS) |
| **年** | 2026 |
| **URL** | https://arxiv.org/abs/2605.05794 |

**核心的アイデア**: Adam(W)がモジュールレベルの勾配異質性を明示的に考慮しない問題を解決。SNRに基づき各モジュールの学習率を自動校正。

**TG-LoRAへの関連性**: ★★★★★
- SNRマップ → 層ごとのゲイン制御の最も近い先行研究
- TG-LoRAではさらにモード（特異値方向）レベルの粒度で制御を行う差別化

---

## 3. Selective / Sparse Training

### 3.1 Lottery Ticket Hypothesis

| 項目 | 内容 |
|------|------|
| **論文** | "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" |
| **著者** | Jonathan Frankle, Michael Carbin |
| **年** | 2019 (ICLR 2019 Best Paper) |
| **URL** | https://arxiv.org/abs/1803.03635 |

**核心的アイデア**: 大規模な密なネットワークには、単独で初期値から学習させても元のネットワークと同等の性能を達成できる小さなサブネットワーク（「当選チケット」）が含まれる。

**手法**: Iterative Magnitude Pruning (IMP)
1. 密なネットワークを学習
2. 最小重みのコネクションを剪定
3. 残りの重みを**元の初期値**にリセット
4. スパースなサブネットワークを再学習

**TG-LoRAへの関連性**: ★★★☆☆
- rank-1支配度0.78 = LoRA更新の大部分が1次元に集中 → 「当選チケット」的な構造が更新空間にも存在
- 有効なモード（特異値方向）のみを学習に使用するアプローチの理論的支持
- ただし、LTHは静的マスク、TG-LoRAは動的なモード選択

---

### 3.2 Sparse Fine-tuning

| 項目 | 内容 |
|------|------|
| **論文** | "Training Neural Networks with Fixed Sparse Masks" (FISH Mask) |
| **著者** | Yi-Lin Sung, Varun Nair, Colin Raffel |
| **年** | 2021 (NeurIPS 2021) |
| **URL** | https://arxiv.org/abs/2111.09839 |

| 項目 | 内容 |
|------|------|
| **論文** | "Scaling Sparse Fine-Tuning to Large Language Models" |
| **著者** | Alan Ansell et al. |
| **年** | 2022/2024 |
| **URL** | https://arxiv.org/abs/2401.16405 |

**核心的アイデア**:
- **Sung et al.**: Fisher情報量に基づく静的スパーシティマスクで更新パラメータを選択
- **Ansell et al.**: LLMスケールでの効率的なスパース更新。"drop-and-grow"戦略で動的トポロジー変更

**TG-LoRAへの関連性**: ★★★★☆
- スパースファインチューニングの「重要度に基づくパラメータ選択」はSNRマップの層選択と等価
- TG-LoRAでは「モード」レベルでのスパース性（低ランク＋モード選択）を実現
- Fisher情報量はGSNRと密接に関連 → 統一的な重要度指標としてSNRが使える

---

### 3.3 BitFit (Bias-only Fine-tuning)

| 項目 | 内容 |
|------|------|
| **論文** | "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models" |
| **著者** | Elad Ben-Zaken, Yoav Goldberg, Shauli Ravfogel |
| **年** | 2022 (ACL 2022) |
| **URL** | https://arxiv.org/abs/2106.10199 |

**核心的アイデア**: バイアス項のみをファインチューニング（全パラメータの<0.1%）。驚くほどフルファインチューニングに匹敵する性能。

**含意**: ファインチューニングは「新しい知識の学習」ではなく「既存知識の引き出し」がメイン。

**TG-LoRAへの関連性**: ★★★☆☆
- 極端に少ないパラメータでも十分 → 更新の本質的な次元が非常に低い
- rank-1支配度の高さと整合: 更新は低次元部分空間に集中

---

### 3.4 Diff Pruning

| 項目 | 内容 |
|------|------|
| **論文** | "Parameter-Efficient Transfer Learning with Diff Pruning" |
| **著者** | Demi Guo, Alexander Rush, Yoon Kim |
| **年** | 2021 (ACL 2021) |
| **URL** | https://arxiv.org/abs/2012.07514 |

**核心的アイデア**: タスク固有の「差分ベクトル」を学習し、L0ノルムの微分可能近似でスパース化。

**手法**:
- 凍結されたベースモデルパラメータに対し、スパースな差分ベクトルを追加
- 学習中にどのパラメータを更新するかをデータ駆動で決定
- 非ゼロ値とそのインデックスのみを保存

**TG-LoRAへの関連性**: ★★★★☆
- TG-LoRAのSNRマップによるモード選択は、Diff Pruningの「データ駆動の更新パラメータ選択」の連続ランク版
- L0正則化 → モードごとの重要度に基づくソフトマスキングに一般化可能

---

## 4. Layer Freezing / Progressive Training

### 4.1 ULMFiT: Progressive Layer Unfreezing

| 項目 | 内容 |
|------|------|
| **論文** | "Universal Language Model Fine-tuning for Text Classification" |
| **著者** | Jeremy Howard, Sebastian Ruder |
| **年** | 2018 (ACL 2018) |
| **URL** | https://arxiv.org/abs/1801.06146 |

**核心的アイデア**: 3つのテクニックの組み合わせによる効果的な転移学習:

1. **Discriminative Fine-tuning**: 層ごとに異なる学習率（下層は低く、上層は高く）
2. **Slanted Triangular Learning Rates (STLR)**: 急速な線形ウォームアップ + 緩やかな減衰
3. **Gradual Unfreezing**: 最上層から順に段階的に解凍

**TG-LoRAへの関連性**: ★★★★☆
- Gradual Unfreezingは「段階的にSNRマップを拡大する」概念に対応
- Discriminative Fine-tuningはLLRDの原型であり、SNR駆動学習率の先駆け
- 初期は安定な（高SNR）上位層のみ学習 → 徐々にSNR閾値を下げて下位層も解凍

---

### 4.2 Freeze-and-Thaw Strategies

| 項目 | 内容 |
|------|------|
| **アプローチ** | Neural Network Layer Freezing |
| **関連研究** | ifBO (In-context Freeze-Thaw BO, 2024) |

**2つの文脈**:
1. **学習戦略としてのLayer Freezing**: 転移学習で特定の層の重みを凍結し、タスク固有の層のみ更新
2. **ベイズ最適化としてのFreeze-Thaw**: 複数のモデル設定を「一時停止」「再開」して効率的にハイパーパラメータ探索

**TG-LoRAへの関連性**: ★★★☆☆
- SNRマップが動的に変化する場合、低SNR層を一時的にフリーズし、SNR改善後にアンフリーズする戦略
- ランダムウォークコントローラーとの統合: 現在のフリーズ/アンフリーズ決定をRWで探索

---

## 5. Mode Analysis in Neural Networks

### 5.1 Singular Value Analysis of Weight Updates

| 項目 | 内容 |
|------|------|
| **テーマ** | 重み行列のスペクトル進化分析 |
| **主要文献** | Martin & Mahoney (2021), "Implicit Self-Regularization in DNNs" |

**主要知見**:
- 重み行列の特異値スペクトルは「バルク + テール」分布を示す
- 最大特異値方向は意味的に解釈可能な方向に整列
- 確率微分方程式（SDE）フレームワークで、二乗特異値はDyson Brown motionに従う
- **固有値反発**: 特異値が崩壊することを防ぐメカニズム

**TG-LoRAへの関連性**: ★★★★★
- rank-1支配度0.78 = 最大特異値が「テール」部分を支配
- B_{t-1}ΔA_tの安定性 = フィルタされた更新のスペクトルが安定
- バルク部分のノイズをカットし、テール（信号）のみを増幅する設計の理論的根拠

---

### 5.2 PCA of Training Trajectories

| 項目 | 内容 |
|------|------|
| **テーマ** | 学習軌跡の低次元可視化 |
| **主要文献** | Li et al. (2018); Gur-Ari et al. (2018) |

**主要知見**:
- ニューラルネットワークの学習は、パラメータ空間の驚くほど低次元の多様体上で進行
- 軌跡はドリフト付きランダムウォークに似る
- PCAで2D/3D投影すると、異なるloss landscape領域への収束パスが可視化可能

**TG-LoRAへの関連性**: ★★★★☆
- TG-LoRAの外挿アルゴリズムは本質的に「学習軌跡の低次元性」を仮定
- rank-1支配は軌跡のPCA第1成分が圧倒的に大きいことを示唆
- velocity方向 ≈ PCA第1主成分方向 → 外挿はこの方向に沿った予測

---

### 5.3 Intrinsic Dimensionality

| 項目 | 内容 |
|------|------|
| **論文** | "Measuring the Intrinsic Dimension of Objective Landscapes" |
| **著者** | Li et al. |
| **年** | 2018 (ICLR 2018) |
| **URL** | https://arxiv.org/abs/1804.08838 |

**核心的アイデア**: 目的関数景観の固有次元を測定。ネットワークはランダムに方向付けされた低次元部分空間内で効果的に学習可能。

**手法**:
- パラメータ空間を $\theta = \theta_0 + P\theta'$ でd次元部分空間に射影
- dを変えて学習可能な最小次元を探索
- 固有次元はタスク依存、モデルアーキテクチャにはあまり依存しない

**TG-LoRAへの関連性**: ★★★★★
- **LoRAの理論的基盤そのもの**: ΔWの固有次元が低い → 低ランク近似が有効
- rank-1支配度 = 有効固有次元がほぼ1
- TG-LoRAの外挿はこの1次元方向に沿った予測として解釈可能

---

### 5.4 Mode Connectivity and Loss Landscape

| 項目 | 内容 |
|------|------|
| **論文** | "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs" |
| **著者** | Timur Garipov, Pavel Izmailov, Andrew Gordon Wilson et al. |
| **年** | 2018 (NeurIPS 2018) |
| **URL** | https://arxiv.org/abs/1802.10026 |

**核心的アイデア**: 独立に学習された最適解（モード）間を、低lossの連続パスで接続可能。

**主要知見**:
- 損失関数のランドスケープは従来思われていたより「つながっている」
- Linear Mode Connectivity (LMC): 2つのモデルを重み空間の直線で接続しても損失バリアが存在しない場合がある
- Fast Geometric Ensembling (FGE): 低lossパス上のモデルを集約

**TG-LoRAへの関連性**: ★★★☆☆
- 外挿による重み予測が有効な理由: loss landscapeが局所的に「平坦」で接続されている
- SNRが高い方向 = loss landscapeの「谷」に沿った方向 → 外挿に適した方向

---

## 6. Noise Filtering in Gradient Updates

### 6.1 SAM: Sharpness-Aware Minimization

| 項目 | 内容 |
|------|------|
| **論文** | "Sharpness-Aware Minimization for Efficiently Improving Generalization" |
| **著者** | Pierre Foret et al. |
| **年** | 2021 (ICLR 2021) |
| **URL** | https://arxiv.org/abs/2010.01412 |

**核心的アイデア**: 損失値が低いだけでなく、周囲の損失も均一に低い「平坦な最小値」を探索。

**手法**: 2パスのforward-backward:
1. 敵対的摂動を見つけるための勾配計算
2. 摂動後の勾配で実際のパラメータ更新

**主要バリアント**:
| バリアント | 特徴 |
|-----------|------|
| **ASAM** | スケール不変のsharpness。パラメータ再スケーリングに対してロバスト |
| **GSAM** | Hessianの支配的固有値でsharpnessを近似。surrogate gap最小化 |
| **SSAM** | 再正規化によりsaddle pointを回避 |
| **F-SAM** | 敵対的摂動からフル勾配成分を除去、確率的ノイズのみ使用 |

**TG-LoRAへの関連性**: ★★★☆☆
- SAMの「平坦な最小値」探索はSNRマップの「安定な方向」と関連
- TG-LoRAの外挿がSAM的な「ノイズ耐性のある更新」を暗黙的に実現している可能性
- ただし計算コストが2倍 → TG-LoRAのSNRベースのゲイン制御はより効率的なアプローチ

---

### 6.2 Gradient Clipping Strategies

| 項目 | 内容 |
|------|------|
| **テーマ** | 勾配のノルムクリッピング、値クリッピング |
| **標準実装** | PyTorch `torch.nn.utils.clip_grad_norm_` |

**勾配クリッピングの種類**:
- **Norm clipping**: 全パラメータの勾配ノルムを上限に制限
- **Value clipping**: 個々の勾配値を範囲内に制限
- **Adaptive clipping**: 勾配統計に基づく動的閾値

**TG-LoRAへの関連性**: ★★★☆☆
- SNRが低い層の勾配を積極的にクリッピング → ノイズ抑制
- 層ごとの適応的クリッピング閾値 = SNRマップの直接的応用

---

### 6.3 Gradient Accumulation as Noise Reduction

**原理**: ミニバッチ勾配の平均化は $SNR \propto \sqrt{n}$ でノイズを低減。

**TG-LoRAへの関連性**: ★★★★☆
- SNRが低い層に対して勾配累積ステップを増やす適応戦略
- 外挿ステップ数Kをcritical batch sizeと関連付ける設計

---

### 6.4 ISTA / Proximal Methods

| 項目 | 内容 |
|------|------|
| **テーマ** | 近接勾配法による構造化スパース更新 |
| **標準参考** | Beck & Teboulle (2009); Parikh & Boyd (2014) |

**原理**:
$$\theta^{(k+1)} = \text{prox}_{\alpha R} \left( \theta^{(k)} - \alpha \nabla f(\theta^{(k)}) \right)$$

- 微分可能な損失 $f$ + 非微分可能な正則化 $R$ の最小化
- 近接演算子がスパース性制約を直接適用
- Group Lasso → ブロック単位のソフトスレッショルディング
- 2:4構造化スパーシティのためのカスタム近接演算子

**TG-LoRAへの関連性**: ★★★★☆
- SNRマップに基づくソフトスレッショルディング: 低SNRモードを近接演算で減衰
- 「モードごとのソフトマスキング」は近接演算の一般化と解釈可能
- $R(\theta) = \sum_l \lambda_l ||\theta_l||$ で層ごとの正則化をSNRで制御

---

## 7. Adaptive Rank Methods Beyond LoRA

### 7.1 AdaLoRA: SVD-based Adaptive Rank

| 項目 | 内容 |
|------|------|
| **論文** | "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" |
| **著者** | Qingru Zhang et al. |
| **年** | 2023 (ICLR 2023) |
| **URL** | https://arxiv.org/abs/2303.10512 |

**核心的アイデア**: LoRAの均一ランクの代わりに、重要度に基づいてランクを動的に割り当て。

**手法**:
- 更新を $\Delta W = P \Lambda Q$ でパラメータ化（$\Lambda$: 特異値の対角行列）
- 各特異値に重要度スコアを割り当て
- 低重要度の特異値を剪定 → 重要な更新のランクを維持/増加

| 特徴 | LoRA | AdaLoRA |
|------|------|---------|
| ランク割り当て | 固定/均一 | 適応的（動的） |
| アプローチ | $A, B$ 分解 | SVDベース $P, \Lambda, Q$ |
| パラメータ効率 | 準最適 | 重要度ベースの剪定で最適化 |

**TG-LoRAへの関連性**: ★★★★★
- **rank-1支配度との直接的な関係**: rank-1支配度0.78 → 実質ランク1でも十分な情報量
- AdaLoRAの重要度スコア ≈ TG-LoRAのSNRスコア
- TG-LoRAでは特異値方向ごとのSNRでAdaLoRA的なランク割り当てを実現可能

---

### 7.2 GaLore: Gradient Low-Rank Projection

| 項目 | 内容 |
|------|------|
| **論文** | "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection" |
| **年** | 2024 (ICML 2024 Oral) |
| **URL** | https://arxiv.org/abs/2403.03507 |

**核心的アイデア**: 勾配自体を低ランク部分空間に射影してオプティマイザ状態のメモリを削減。フルパラメータ学習を維持しつつLoRA以上のメモリ効率。

**主要成果**:
- 7Bモデルを24GB VRAM（RTX 4090）で事前学習可能
- オプティマイザメモリ65.5%削減
- 8-bit GaLoreで82.5%削減

**TG-LoRAへの関連性**: ★★★★☆
- GaLoreは「勾配の低ランク性」を利用 → TG-LoRAの「更新の低ランク性」と同根
- SNRマップを勾配の射影品質で定義する可能性
- 勾配空間のSVDで主要方向を特定 → SNRマップの計算に利用可能

---

### 7.3 PiSSA: SVD-based LoRA Initialization

| 項目 | 内容 |
|------|------|
| **論文** | "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models" |
| **年** | 2024 (NeurIPS 2024 Spotlight) |
| **URL** | https://arxiv.org/abs/2404.02948 |

**核心的アイデア**: LoRAのA/B行列をベースモデル重みのSVD主成分で初期化。ノイズではなく主成分を直接更新。

**手法**:
- $W = U\Sigma V^T$ のSVDを計算
- 上位ランクrの成分でA/Bを初期化
- 残りの成分は凍結された残差行列 $W^{res}$ に格納
- Fast SVDで初期化は数秒

**TG-LoRAへの関連性**: ★★★★★
- PiSSAの「主成分直接更新」はTG-LoRAのrank-1支配方向の更新と直結
- SNRマップ: PiSSA的初期化 + rank-1方向のSNR監視 → 適応的増幅の組み合わせ
- Bフィルタ仮説: PiSSAのB行列はベースモデルの主方向を表現 → フィルタとしてより効果的に機能

---

### 7.4 DoRA: Weight-Decomposed Low-Rank Adaptation

| 項目 | 内容 |
|------|------|
| **論文** | "DoRA: Weight-Decomposed Low-Rank Adaptation" |
| **年** | 2024 (ICML 2024) |
| **URL** | https://arxiv.org/abs/2402.09353 |

**核心的アイデア**: 重みを大きさ（magnitude）と方向（direction）に分解し、LoRAは方向成分のみ更新、大きさは別途ファインチューニング。

**TG-LoRAへの関連性**: ★★★★☆
- rank-1支配度 = 方向成分が1次元に集中
- SNRマップでmagnitudeとdirectionのSNRを個別に評価する設計の先行事例

---

### 7.5 LoRA-FA: Freeze A Matrix

| 項目 | 内容 |
|------|------|
| **論文** | "LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning" |
| **年** | 2023 |
| **URL** | https://arxiv.org/abs/2308.03303 |

**核心的アイデア**: A行列（projection-down）を凍結し、B行列（projection-up）のみ学習。活性化メモリを大幅に削減。

**TG-LoRAへの関連性**: ★★★★★
- **Bフィルタ仮説の直接的な検証**: A凍結 + B学習でも同等性能 = Bが情報フィルタリングの主体
- B_{t-1}ΔA_tの安定性0.99 → Aの変化がBでフィルタされている証拠
- TG-LoRA設計: A方向は軌跡priorで固定、Bのゲインのみ適応調整

---

### 7.6 Tensor Decomposition Methods

| 手法 | 主な用途 | 特徴 |
|------|---------|------|
| **Tensor Train (TT)** | モデル圧縮 | 高次元テンソルを低階コアのチェーンに分解 |
| **Tucker分解** | 畳み込み層圧縮 | コアテンソル + モード因子行列 |
| **KFAC** | 最適化 | Fisher行列をKronecker積で近似、2次最適化 |

**TG-LoRAへの関連性**: ★★☆☆☆
- テンソル分解は圧縮目的が主。TG-LoRAの適応的増幅とは間接的な関連
- KFACの層ごとの曲率推定はSNRマップの代替指標として利用可能

---

## 8. LoRA B-matrix as Filter

### 8.1 Low Intrinsic Rank Hypothesis

| 項目 | 内容 |
|------|------|
| **論文** | "LoRA: Low-Rank Adaptation of Large Language Models" |
| **著者** | Edward Hu et al. |
| **年** | 2022 (ICLR 2022) |
| **URL** | https://arxiv.org/abs/2106.09685 |

**基本構造**: $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$

**初期化**: A = Gaussian random, B = zeros → ΔW = 0 at initialization

**暗黙的正則化**:
- 低ランク制約 ($r \ll \min(d,k)$) がオーバーフィッティングを抑制
- 更新を低次元部分空間に制限 = アーキテクチャ的な正則化

---

### 8.2 B行列のフィルタリング機能に関する分析

**TG-LoRAの観測事実**:
1. **B_{t-1} @ ΔA_t の安定性 = 0.99**: B行列が一貫した方向にΔAを射影
2. **rank-1支配度 = 0.78**: 更新の情報量の大部分が1次元方向に集中
3. **Bの更新はAの更新より緩やか**: B行列が「ゆっくり変化するフィルタ」として機能

**関連する理論的枠組み**:

#### Subspace Learning / Manifold Learning
- B行列の列空間が更新の「許容部分空間」を定義
- Aの更新はこの部分空間内でのみ効果的に反映される
- B行列の特異値構造がフィルタの通過帯域/阻止帯域を決定

#### Orthogonal Subspace Methods (Continual Learning)
| 手法 | 概要 | 関連性 |
|------|------|--------|
| **O-LoRA** | タスク間で直交LoRA部分空間を維持 | B行列がタスク知識の「方向フィルタ」 |
| **CLoRA** | 過去のタスク部分空間の直交補空間で学習 | フィルタ干渉の防止 |

#### Intruder Dimensions (Sharma et al., 2024)
- LoRAはフルFTに現れない「侵入次元」（高ランク特異値方向）を導入
- これらはB行列のフィルタ特性に起因する可能性
- 適応的なランク制御（SNRベース）でintruder dimensionを抑制可能

---

### 8.3 SalientLoRA

| 項目 | 内容 |
|------|------|
| **論文** | "Unveiling LoRA Intrinsic Ranks via Salience Analysis" (SalientLoRA) |
| **年** | 2024 (NeurIPS 2024) |
| **URL** | https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed9f00cb7dd5fbdc2175d55e2fdf1b05-Abstract-Conference.html （code: https://github.com/ginobilinie/SalientLoRA） |

**核心的アイデア**: 特異値の「顕著性」（salience）を時間経過とともに測定し、動的にランクを剪定/保持。

**TG-LoRAへの関連性**: ★★★★☆
- 顕著性 = 時間平均されたSNRの一種
- TG-LoRAのモードごとのSNR追跡と直接的に対応
- rank-1が支配的な場合、rank-2以降のモードの顕著性が低い → 剪定対象

---

### 8.4 GeLoRA: Geometry-based Rank Estimation

| 項目 | 内容 |
|------|------|
| **論文** | "GeLoRA: Geometric Low-Rank Adaptation" |
| **年** | 2024 |

**核心的アイデア**: 隠れ表現の幾何的性質から固有次元を推定し、LoRAランクを原理的に設定。

**TG-LoRAへの関連性**: ★★★☆☆
- rank-1支配度の理論的裏付け: 幾何的固有次元 ≈ 1
- SNRマップとの統合: 幾何的推定 + SNR測定の二重基準

---

## 9. TG-LoRAへの統合提案

### 9.1 先行研究との差別化マップ

```
                    パラメータ選択の粒度
                    層 ← ─ ─ ─ ─ ─ ─ ─ → パラメータ
                    │                      │
    静  Spectrum     │  BitFit              │  Diff Pruning
    的  LLRD         │  LoRA-FA             │  Lottery Ticket
    │               │                      │
    ↕               │  ★TG-LoRA★           │
    │               │  (層×モード粒度,      │
    動  MoLS         │   動的SNR制御)        │  Ansell (動的マスク)
    的  ALLoRA       │  AdaLoRA             │  SpIEL
                    │                      │
```

### 9.2 TG-LoRAのSNRマップ設計への具体的示唆

| 先行研究 | TG-LoRAへの適用 |
|---------|----------------|
| **LARS/LAMB** | 層ごとの `\|\|B_l\|\| / \|\|∇B_l\|\|` trust ratioをSNRの一部に |
| **GSNR (Liu 2020)** | モードごとの $GSNR_m = E[g_m]^2 / Var[g_m]$ を計算 |
| **McCandlish GNS** | 層ごとの臨界バッチサイズで外挿ステップKを適応 |
| **Spectrum** | MP分布での信号/ノイズ分離をΔW = BΔAに適用 |
| **LoRA+** | A/B行列に異なるゲインを適用（$\eta_B > \eta_A$） |
| **AdaLoRA** | 特異値重要度でモードごとのランクを動的調整 |
| **PiSSA** | SVD初期化でrank-1方向を事前特定 |
| **LoRA-FA** | Bフィルタ仮説の直接的支持証拠 |
| **ISTA/Proximal** | SNRに基づくモードごとのソフトスレッショルディング |

### 9.3 提案するSNRマップの定義

```python
# 各層 l、各モード m のSNR
SNR[l, m] = (E[σ_m(B_l @ ΔA_l)])^2 / Var[σ_m(B_l @ ΔA_l)]

# 適応ゲイン
gain[l, m] = clip(SNR[l, m] / SNR_threshold, min=0.1, max=10.0)

# 実効学習率
η_eff[l, m] = η_base * gain[l, m] * trust_ratio[l]
```

**理論的裏付け**:
- `SNR[l, m]` が高い → 信号が安定 → ゲインを上げて学習を加速（LARS/LAMB理論）
- `SNR[l, m]` が低い → ノイズ支配 → ゲインを下げて安定化（McCandlish理論）
- rank-1支配度が高い場合、m=0のSNRが支配的 → 実質1次元の適応制御

### 9.4 Bフィルタ仮説の位置づけ

```mermaid
graph LR
    A[ΔA_t: 生の更新] --> B_filter[B_{t-1}: フィルタ行列]
    B_filter --> S[B_{t-1}@ΔA_t: フィルタ済み更新]
    S --> SVD[SVD分析]
    SVD --> SNR[モードごとのSNR計算]
    SNR --> Gain[適応的ゲイン]
    Gain --> Update[重み更新 with gain]
    
    style B_filter fill:#ff9,stroke:#333,stroke-width:2px
    style SNR fill:#9f9,stroke:#333,stroke-width:2px
```

**先行研究との関係**:
- LoRA-FAのA凍結がBフィルタ仮説を支持（Bのみの学習でも性能維持）
- LoRA+のη_B > η_AはBフィルタの適応を優先する設計と整合
- PiSSAのSVD初期化はBのフィルタ特性を最適化する手段
- rank-1支配度はBフィルタの通過帯域幅（≈1次元）を示す

---

## 10. 2026年6月 最新動向アップデート

> **更新日**: 2026-06-09 / **出典**: 下記 arXiv を Web 調査。abstract / 公開本文に基づき記述。

本トラックの中核「**層・モジュールごとの勾配 SNR に基づく適応ゲイン**」は、2026 年に正式版が登場し、構想の正しさが裏付けられた。

### 10.1 MoLS の正式版確定 — モジュール間の勾配ノイズ不均衡を SNR で校正

§2.5 で placeholder だった MoLS は、正式版 [arXiv:2605.05794](https://arxiv.org/abs/2605.05794)（*Revealing Modular Gradient Noise Imbalance in LLMs*）として公開され、本調査で arXiv 番号を確定した。

- **核心**: Adam(W) は per-parameter 適応性を持つが**モジュール間の勾配ノイズの不均衡（heterogeneity）を補正しない**点を指摘。各モジュールの **SNR を基準モジュールで正規化**し、module-wise 学習率を校正。
- **TG-LoRA への含意**: 本トラックが提案する「SNR マップ → 層・モードごとのゲイン制御」（§9.3）の **最も近い 2026 先行研究**。MoLS が module 粒度で行うことを、TG-LoRA は **層×モード（特異値方向）粒度**で行う点が差別化軸として明確になった。

### 10.2 層別適応学習率の継続的進展

| 論文 | 年 | 知見 |
|------|----|------|
| **La-LoRA** (Layer-wise adaptive LoRA) | 2025 (Neurocomputing/ScienceDirect) | AdaLoRA 系の SVD ランク調整に**層ごとの適応学習率**を組み合わせ |
| **Learning Rate Matters: Vanilla LoRA May Suffice** | 2026 / [2602.04998](https://arxiv.org/abs/2602.04998) | 学習率を適切に設定すれば素の LoRA で十分という反証的主張。LoRA+ 等の非対称 LR の効果は LR 探索に吸収されうると示唆 |

> [!IMPORTANT]
> [2602.04998](https://arxiv.org/abs/2602.04998) は「**適応ゲイン/非対称 LR の利得は、十分な学習率探索で消える可能性**」という反証を提示する。TG-LoRA の SNR ベース適応ゲインを設計する際は、**素の LoRA + よく調整した LR を強いベースライン**として置き、ゲイン制御の純粋な上乗せ効果を ablation で示すことが、最新動向に照らして必須である。無駄な増幅ギミックの実装を避けるための重要なガード。

---

## 参考文献一覧

### Layer-wise Adaptive Learning Rate
1. You, Y., Gitman, I., & Ginsburg, B. (2017). "Large Batch Training of Convolutional Networks." arXiv:1708.03888
2. You, Y. et al. (2020). "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." ICLR 2020. arXiv:1904.00962
3. Hayou, S. et al. (2024). "LoRA+: Efficient Low Rank Adaptation of Large Models." ICML 2024. arXiv:2402.12354
4. "ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws." (2024). arXiv:2407.11502

### Gradient SNR
5. Liu, J. et al. (2020). "Understanding Why Neural Networks Generalize Well Through GSNR of Parameters." ICLR 2020
6. McCandlish, S., Kaplan, J., & Amodei, D. (2018). "An Empirical Model of Large-Batch Training." arXiv:1812.06162
7. Liu, L. et al. (2020). "Understanding the Difficulty of Training Transformers." ACL 2020. arXiv:2004.08249
8. "Spectrum: Targeted Training on Signal to Noise Ratio." (2024). arXiv:2406.06623
9. (2026). "Revealing Modular Gradient Noise Imbalance in LLMs: Calibrating Module-wise Learning Rate via SNR" (MoLS). https://arxiv.org/abs/2605.05794

### Selective / Sparse Training
10. Frankle, J. & Carlin, M. (2019). "The Lottery Ticket Hypothesis." ICLR 2019. arXiv:1803.03635
11. Sung, Y.-L., Nair, V., & Raffel, C. (2021). "Training Neural Networks with Fixed Sparse Masks" (FISH Mask). NeurIPS 2021. arXiv:2111.09839. https://arxiv.org/abs/2111.09839
12. Ansell, A. et al. (2024). "Scaling Sparse Fine-Tuning to Large Language Models." arXiv:2401.16405
13. Ben-Zaken, E. et al. (2022). "BitFit: Simple Parameter-efficient Fine-tuning." ACL 2022. arXiv:2106.10199
14. Guo, D., Rush, A., & Kim, Y. (2021). "Parameter-Efficient Transfer Learning with Diff Pruning." ACL 2021. arXiv:2012.07514

### Layer Freezing / Progressive Training
15. Howard, J. & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL 2018. arXiv:1801.06146

### Mode Analysis
16. Li, C. et al. (2018). "Measuring the Intrinsic Dimension of Objective Landscapes." ICLR 2018. arXiv:1804.08838
17. Garipov, T., Izmailov, P., et al. (2018). "Loss Surfaces, Mode Connectivity, and Fast Ensembling." NeurIPS 2018. arXiv:1802.10026

### Noise Filtering
18. Foret, P. et al. (2021). "Sharpness-Aware Minimization." ICLR 2021. arXiv:2010.01412

### Adaptive Rank Methods
19. Zhang, Q. et al. (2023). "AdaLoRA: Adaptive Budget Allocation for PEFT." ICLR 2023. arXiv:2303.10512
20. Zhao, J. et al. (2024). "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection." ICML 2024. arXiv:2403.03507
21. Meng, F. et al. (2024). "PiSSA: Principal Singular Values and Singular Vectors Adaptation." NeurIPS 2024. arXiv:2404.02948
22. Liu, S. et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML 2024. arXiv:2402.09353
23. Zhang, L. et al. (2023). "LoRA-FA: Memory-efficient Low-rank Adaptation." arXiv:2308.03303
23b. (2024). "Unveiling LoRA Intrinsic Ranks via Salience Analysis" (SalientLoRA). NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed9f00cb7dd5fbdc2175d55e2fdf1b05-Abstract-Conference.html

### LoRA Core
24. Hu, E. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685
