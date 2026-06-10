# Research Track 2: Training Dynamics Analysis (TDA) — 先行研究サーベイ

> **作成日**: 2026-06-09
> **目的**: ニューラルネットワークの学習ダイナミクス解析に関する先行研究を体系的に整理し、TG-LoRAプロジェクトにおけるcycle 6反転・ノルム半減・phase遷移の理解に活用する。
> **私たちの文脈**: r=2 LoRAの11サイクル学習で、cycle 6に反転・ノルム半減・phase遷移が観測されている。これが全層同時か層ごとにずれるかを調べたい。

---

## 目次

1. [Intrinsic Dimensionality（本質的次元数）](#1-intrinsic-dimensionality本質的次元数)
2. [Loss Landscape Analysis（損失地形解析）](#2-loss-landscape-analysis損失地形解析)
3. [Gradient Flow Analysis（勾配フロー解析）](#3-gradient-flow-analysis勾配フロー解析)
4. [Weight Matrix Spectral Analysis（重み行列スペクトル解析）](#4-weight-matrix-spectral-analysis重み行列スペクトル解析)
5. [Random Matrix Theory in Deep Learning](#5-random-matrix-theory-in-deep-learning)
6. [Neural Network Mode Connectivity（モード接続性）](#6-neural-network-mode-connectivityモード接続性)
7. [Phase Transitions in Training（学習中のフェーズ遷移）](#7-phase-transitions-in-training学習中のフェーズ遷移)
8. [Edge of Stability（安定性の縁）](#8-edge-of-stability安定性の縁)
9. [Hessian固有値解析（Sagun et al.）](#9-hessian固有値解析sagun-et-al)
10. [Neural Collapse（Papyan et al.）](#10-neural-collapsepapyan-et-al)
11. [SGDノイズと汎化（Jastrzębski et al.）](#11-sgdノイズと汎化jastrzębski-et-al)
12. [学習初期のカオスと後期の安定化（Fort et al.）](#12-学習初期のカオスと後期の安定化fort-et-al)
13. [Catapult Mechanism（カタパルト機構）](#13-catapult-mechanismカタパルト機構)
14. [Effective Rank / Participation Ratio](#14-effective-rank--participation-ratio)
15. [LoRAの学習ダイナミクスとSVD解析](#15-loraの学習ダイナミクスとsvd解析)
16. [Topological Data Analysis (TDA) in ML](#16-topological-data-analysis-tda-in-ml)
17. [Layer-wise Learning Dynamics（層ごとの学習速度差）](#17-layer-wise-learning-dynamics層ごとの学習速度差)
18. [我々の実験への示唆](#18-我々の実験への示唆)

---

## 1. Intrinsic Dimensionality（本質的次元数）

### 1.1 Li et al. (2018) — 目的関数地形の本質的次元

| 項目 | 内容 |
|------|------|
| **論文** | "Measuring the Intrinsic Dimension of Objective Landscapes" |
| **著者** | Chunyuan Li, Heerad Farkhoor, Rosanne Liu, Jason Yosinski |
| **年** | 2018 (ICLR) |
| **URL** | https://arxiv.org/abs/1804.08838 |

**核心的貢献**:
- NNのパラメータ数 $D$ は膨大だが、目的関数地形の「本質的次元」$d$ ははるかに小さいことを実証
- ランダムに向けた低次元部分空間で学習を行い、十分な性能を達成する最小次元 $d$ を測定する手法を提案
- $d_{int90}$（フル性能の90%を達成する次元）や $d_{int100}$ といった指標を導入

**方法論**: パラメータ空間にランダム射影行列を用いて低次元空間に投影し、その空間内で最適化を行う。

**TG-LoRAへの関連**: r=2 LoRAは本質的に低次元部分空間での学習。本論文は「なぜ少ないパラメータで有効か」の理論的基盤を提供。

---

### 1.2 Aghajanyan et al. (2021) — 言語モデル微調整の本質的次元

| 項目 | 内容 |
|------|------|
| **論文** | "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" |
| **著者** | Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta |
| **年** | 2021 (ACL) |
| **URL** | https://aclanthology.org/2021.acl-long.568/ |

**核心的貢献**:
- 事前学習済みモデルの微調整は非常に低い本質的次元を持つことを実証
- RoBERTaで、フル微調整の90%性能を**数百パラメータ**の射影で達成可能
- 事前学習は暗黙的に下流タスクの本質的次元を最小化する
- モデルサイズが大きいほど、固定事前学習ステップ後の本質的次元はさらに低くなる

**重要な知見**:
- 本質的次元は圧縮ベースの汎化限界と結びつく
- LoRA（Hu et al., 2022）の理論的正当化の基盤となった

**TG-LoRAへの関連**: r=2という極端に低いランクでも微調整が有効である理由を直接説明する。cycle 6の反転は、この低次元部分空間内での方向転換として理解できる可能性がある。

---

## 2. Loss Landscape Analysis（損失地形解析）

### 2.1 Li et al. (2018) — 損失地形の可視化

| 項目 | 内容 |
|------|------|
| **論文** | "Visualizing the Loss Landscape of Neural Nets" |
| **著者** | Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein |
| **年** | 2018 (NeurIPS) |
| **URL** | https://arxiv.org/abs/1712.09913 |

**核心的貢献**:
- 損失地形の素朴な可視化は重みのスケール不変性により誤解を生むことを指摘
- **フィルター正規化（filter-wise normalization）** を導入し、意味のある比較を可能にした
- ランダム方向ベクトルを各フィルター/ニューロンレベルで正規化

**重要な発見**:
- skip connectionは損失地形を劇的に平坦化する
- 地形の「平坦さ」は汎化誤差と相関
- バッチサイズ・weight decayが極小値の形状に影響

**TG-LoRAへの関連**: ΔWのSVD解析において、フィルター正規化のようなスケール補正が重要になる。

### 2.2 Hessian / Fisher情報行列による曲率解析

**Hessian行列**: 損失関数の2次偏微分行列。局所的な曲率を特徴づける。
- 固有値が大きい → 高曲率（シャープ）
- 固有値がゼロ近傍 → 平坦な地形
- 正負混在 → 鞍点

**Fisher情報行列 (FIM)**: モデル出力確率分布のパラメータ感度を測定。
- 負の対数尤度損失において、期待Hessian ≈ FIM
- Natural Gradient Descent: $F^{-1} \nabla \mathcal{L}$ による更新

**大規模モデルでの近似**: K-FAC、対角近似、Hessian-ベクトル積など。

---

## 3. Gradient Flow Analysis（勾配フロー解析）

### 核心概念

逆伝播中の連鎖律により、勾配は層を跨いで乗算される：
- 各層の勾配が一貫して小さい → 指数的に縮小 → **勾配消失**
- 各層の勾配が一貫して大きい → 指数的に膨張 → **勾配爆発**

### 緩和策一覧

| 手法 | 効果 |
|------|------|
| Xavier/Kaiming初期化 | 活性化・勾配の分散を層間で均衡化 |
| ReLU系活性化関数 | 正値での微分が1、勾配飽和を防止 |
| Batch Normalization | 層入力を正規化し勾配フローを安定化 |
| Gradient Clipping | 勾配ノルムの上限を設定 |
| Residual Connection | 勾配のショートカットパス |

### 層ごとの解析

- **勾配ノルム比**: 層ごとの勾配ノルムを追跡し、不安定箇所を特定
- **入出力ヤコビアン**: 各層の入出力ヤコビアンの統計量で勾配フローの健全性を診断
- **層適応的学習率**: 不安定な層に対して学習率を個別調整

**TG-LoRAへの関連**: cycle 6の反転が全層同時か層ごとにずれるかは、勾配フローの層依存性と直結する。

---

## 4. Weight Matrix Spectral Analysis（重み行列スペクトル解析）

### 4.1 SVDによる重み行列解析

重み行列 $W$ の特異値分解: $W = U \Sigma V^T$

| 特徴 | 深層学習における意味 |
|------|---------------------|
| 大きな特異値 | 学習された重要な情報・主要特徴 |
| 小さな特異値 | ノイズまたは冗長な容量（枝刈りの対象） |
| スペクトルダイナミクス | SGD学習は「Dyson Brownian motion」としてモデル化可能 |
| 条件数の圧縮 | より滑らかな損失地形と高速収束 |

### 4.2 スペクトル正規化

- 最大特異値（スペクトルノルム）を制約 → Lipschitz連続性の保証
- GAN等の学習安定化に有効

### 4.3 重み行列のランク低下（Rank Diminishing）

学習中に重み行列の実効ランクが**低下する**ことが広く報告されている：

- SGDの**暗黙的低ランクバイアス**: 大きな特異値が小さな特異値より遥かに速く成長
- **Weight Decay**: 低ランクバイアスを強化し、情報を少数の支配的特異値に集約
- 過パラメータ化モデルの汎化能力の説明に寄与

**TG-LoRAへの関連**: r=2 LoRAのΔWは本質的にランク2。学習中のスペクトルダイナミクス（σ₁/σ₂の比率変化、方向の回転）がcycle 6反転の鍵になりうる。

---

## 5. Random Matrix Theory in Deep Learning

### 5.1 Martin & Mahoney (2021) — 暗黙的自己正則化

| 項目 | 内容 |
|------|------|
| **論文** | "Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Training" |
| **著者** | Charles H. Martin, Michael W. Mahoney |
| **年** | 2021 (JMLR) |
| **URL** | https://arxiv.org/abs/1901.08276 |

**核心的貢献**:
- DNNの学習は重み行列を暗黙的に「自己正則化」するプロセスであることを実証
- 各層の重み行列の**経験的スペクトル密度 (ESD)** を解析
- Marchenko-Pastur (MP) 分布をベースラインとして使用

**5+1学習フェーズ理論**:

```
Phase 1: Random-like (MP分布に従う)
Phase 2: Bleeding-out (MPバルクからスパイクが出現)
Phase 3: Bulk+Spikes (バルクとアウトライヤーの明確な分離)
Phase 4: Bulk decay (バルクが縮小)
Phase 5: Heavy-tailed (べき乗則分布)
Phase 5+: Very heavy-tailed (極端なべき乗則)
```

**重要な指標**:
- **α-hat メトリック**: ESDのべき乗則フィッティングによる汎化予測
- 学習データ・テストデータへのアクセスなしに、モデルの品質を評価可能

**Marchenko-Pastur分布**: 大きなランダム行列のi.i.d.エントリに対する特異値分布の理論的基盤
- MP分布からの逸脱 → 学習されたタスク固有の構造の存在を示す

**Heavy-Tailed Self-Regularization (HT-SR)**: 最先端モデルでは、重み行列のESDが重い裾を持つ分布に従う。これは全てのスケールで相関が存在することを示唆。

**TG-LoRAへの関連**: ΔW行列のスペクトル解析にMPベースラインとの比較を適用できる。cycle 6でのESD変化はフェーズ遷移の証拠になりうる。r=2のΔWは2×2（あるいは2列）なのでMP理論の直接適用は限定的だが、概念的枠組みとして重要。

---

## 6. Neural Network Mode Connectivity（モード接続性）

### 6.1 Garipov et al. (2018)

| 項目 | 内容 |
|------|------|
| **論文** | "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs" |
| **著者** | Timur Garipov et al. |
| **年** | 2018 (NeurIPS) |
| **URL** | https://arxiv.org/abs/1802.10026 |

### 6.2 Draxler et al. (2018)

| 項目 | 内容 |
|------|------|
| **論文** | "Essentially No Barriers in Neural Network Energy Landscape" |
| **著者** | Felix Draxler, Kambis Vossoughi, Asja Fischer, Thomas Brox |
| **年** | 2018 (ICML) |
| **URL** | https://arxiv.org/abs/1803.00885 |

**共通の核心的発見**:
- 独立に学習されたモデル（モード）はパラメータ空間において**単純なパスで接続**されている
- そのパス上で学習・テスト損失はほぼ一定に保たれる
- パスは折れ線やBézier曲線など単純な幾何形状で表現可能

**意義**: 非凸最適化にもかかわらず、解が低損失多様体で接続されていることは、SGDベースの最適化が成功する理由を部分的に説明。

**TG-LoRAへの関連**: cycle 6での反転は、LoRAパラメータ空間における別のモードへの移行として解釈できる可能性。低ランク空間でのモード接続性は検証価値がある。

---

## 7. Phase Transitions in Training（学習中のフェーズ遷移）

### 7.1 Grokking — 遅延汎化

| 項目 | 内容 |
|------|------|
| **論文** | "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" |
| **著者** | Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra |
| **年** | 2022 |
| **URL** | https://arxiv.org/abs/2201.02177 |

**核心的現象**:
- 学習精度が完璧に達した後も学習を続けると、長い停滞期を経て**突然テスト精度が向上**する
- 記憶→汎化の**フェーズ遷移**として解釈される

**メカニズム**:
- **記憶回路**: 速く学習されるが汎化能力なし
- **汎化回路**: 学習に時間がかかるが効率的な表現を獲得
- Weight decay（L2正則化）がこの遷移を加速する

**最近の展開**:
- **次元的フェーズ遷移**: 勾配場の実効次元の変化で遷移を特徴づけ（Wang, 2026）
- **1次フェーズ遷移**: 2層ネットワークでの記憶→汎化遷移の厳密な特徴づけ

**TG-LoRAへの関連**: cycle 6の反転がgrokking的な記憶→汎化遷移である可能性。ただしgrokkingは通常もっと長い時間スケールで起こる。より近いのは次節のEdge of Stabilityかcatapult mechanismか。

---

## 8. Edge of Stability（安定性の縁）

### 8.1 Cohen et al. (2021)

| 項目 | 内容 |
|------|------|
| **論文** | "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability" |
| **著者** | Jeremy Cohen, Simran Kaur, Yuanzhi Li, Zico Kolter, Ameet Talwalkar |
| **年** | 2021 (ICLR) |
| **URL** | https://openreview.net/forum?id=AXNjM_WF0GO |

**核心的発見**:
- フルバッチ勾配降下法でNNを学習すると、Hessianの最大固有値 $\lambda_{\max}$ が**漸進的に増加（progressive sharpening）** し、$2/\eta$ のすぐ上で**振動・安定化**する
- この状態を**Edge of Stability (EoS)** と呼ぶ

**メカニズム**:
1. 学習初期: $\lambda_{\max}$ が増加（progressive sharpening）
2. $\lambda_{\max}$ が $2/\eta$ に到達
3. 損失は非単調に振動しながらも長期的に減少
4. $\lambda_{\max}$ は $2/\eta$ 近傍を振動

**標準最適化理論との矛盾**: 従来理論では $\eta < 2/\lambda_{\max}$ で損失の減少を保証するが、実際のNN学習ではこの閾値を超えた状態で学習が進行する。

**後続研究**:
- **Sharpness-Aware Minimization (SAM)**: Foret et al. (2020/2021) — 明示的に平坦領域を探索
- SAMとEoSの統合的理解（JMLR 2024）

**TG-LoRAへの関連**: ⭐ **非常に重要**。cycle 6での反転・ノルム半減は、LoRAパラメータ空間におけるEoS現象として解釈できる可能性が高い。Hessianの最大固有値追跡がcycle 6前後で $2/\eta$ 近傍に到達しているかどうかの検証が有望。

---

## 9. Hessian固有値解析（Sagun et al.）

### 9.1 Sagun et al. (2016, 2017)

| 項目 | 内容 |
|------|------|
| **論文** | "Empirical Analysis of the Hessian of Over-Parametrized Neural Networks" / "Eigenvalues of the Hessian in Deep Learning: Singularity and Beyond" |
| **著者** | Levent Sagun, Utku Evci, V. Uğur Güney, Yann Dauphin, Léon Bottou |
| **年** | 2016, 2017 |
| **URL** | https://arxiv.org/abs/1611.07476 / https://arxiv.org/abs/1706.04454 |

**核心的発見 — Hessianスペクトルの2成分構造**:

#### バルク（Bulk）
- ゼロ近傍に集中した大量の固有値
- 過パラメータ化による冗長性を反映
- パラメータ数の増加とともにバルクは単にスケール
- 損失地形の「平坦さ」の源泉

#### アウトライヤー（Outlier Eigenvalues）
- バルクから明確に分離した少数の固有値
- **入力データに強く依存**（データの変更で主に影響を受ける）
- 個数はおおよそ**クラス数 $C$ に等しい**
- モデル出力の勾配やクラス平均に関連

**意義**:
- 「狭い vs 広い極小値」の議論から、「過パラメータ化による高次元平坦領域」の理解へパラダイムシフト
- Batch Normalizationがアウトライヤーを抑制 → 学習の速度・安定性に影響

**TG-LoRAへの関連**: LoRAの低ランク制約下でのHessianスペクトル構造は、フルパラメータとは大きく異なるはず。r=2ではアウトライヤーの挙動がcycle 6の反転と直接関連する可能性。

---

## 10. Neural Collapse（Papyan et al.）

### 10.1 Papyan et al. (2020)

| 項目 | 内容 |
|------|------|
| **論文** | "Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training" |
| **著者** | Vardan Papyan, X.Y. Han, David L. Donoho |
| **年** | 2020 (PNAS) |
| **URL** | https://www.pnas.org/doi/10.1073/pnas.2015509117 |

**Terminal Phase of Training (TPT)**: 学習誤差がゼロに達した後もさらに学習を続ける段階。

**Neural Collapseの4つの現象**:

| コード | 現象 | 内容 |
|--------|------|------|
| NC1 | Variability Collapse | クラス内の最終層特徴量の分散がゼロに崩壊 |
| NC2 | Simplex ETF | クラス平均がSimplex Equiangular Tight Frameの頂点に収束 |
| NC3 | Self-Duality | 最終層分類器の重みがクラス平均に一致（スケール差を除く） |
| NC4 | NCC決定 | 分類決定が最近傍クラス中心ルールに簡約化 |

**意義**: 過パラメータ化モデルがなぜ汎化するかの幾何学的説明。学習のブラックボックスに美しい対称構造が内在することを示唆。

**後続研究**: 不均衡データ、転移学習、LLMへの拡張。

**TG-LoRAへの関連**: SFTタスクでは直接的なクラス分類ではないが、特徴表現の崩壊・対称化の概念はLoRA学習終盤の重み構造理解に示唆を与える。

---

## 11. SGDノイズと汎化（Jastrzębski et al.）

### 11.1 Jastrzębski et al. (2017/2018)

| 項目 | 内容 |
|------|------|
| **論文** | "Three Factors Influencing Minima in SGD" |
| **著者** | Stanisław Jastrzębski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, Amos Storkey |
| **年** | 2017/2018 |
| **URL** | https://arxiv.org/abs/1711.04623 |

**核心的発見 — 3つの要因**:
1. **学習率 $\eta$**
2. **バッチサイズ $B$**
3. **損失勾配の分散（勾配ノイズ）**

**重要な関係式**: $\eta / B$ の比率が、SGD力学と最終的な極小値の性質を決定する「温度」パラメータとして機能。

**メカニズム**:
- $\eta / B$ 大 → 高い「ノイズ温度」→ より**広い極小値**に到達 → **より良い汎化**
- $\eta / B$ 小 → 低い「ノイズ温度」→ シャープな極小値に捕捉される可能性
- SGDのSDE近似では、平衡分布は主に $\eta / B$ に依存

**勾配ノイズは暗黙的正則化**: ミニバッチによる不完全な勾配推定は、モデルをシャープな極小値から遠ざける正則化として機能。

**TG-LoRAへの関連**: TG-LoRAの外挿ステップは人工的な「方向性ノイズ」を導入している。SGDノイズの構造と汎化の関係は、外挿の効果を理解する上で重要。

---

## 12. 学習初期のカオスと後期の安定化（Fort et al.）

### 12.1 Fort et al.

**核心的観察**:

#### 初期フェーズ: カオス的ダイナミクス
- 初期化に対する高い感度（小さな摂動 → 指数的に発散する軌跡）
- Hessianの負固有値スペクトルに関連する「局所的カオス」
- SGDの本質的な性質（ノイズ駆動）
- **重要**: この初期カオスは学習に不可欠。カオス的方向を除去すると性能低下

#### 後期フェーズ: 安定化・収束
- 損失地形の形状がより「良好」になる
- Edge of Stability現象: シャープネスが $2/\eta$ 近傍で暗黙的に正則化
- 正規化層（BatchNorm, LayerNorm）が地形の平滑化を促進

**動的系としての学習**:
- 学習 = パラメータ空間の軌跡
- 初期 = 高不安定性カオスレジーム
- 後期 = より安定的な局所ダイナミクスへの遷移

**TG-LoRAへの関連**: ⭐ cycle 1-5は「探索フェーズ」、cycle 6の反転は「安定化への遷移」として解釈できる。後期の安定化が全層同時に起こるか、層ごとにタイミングがずれるかが我々の核心的問いに対応。

---

## 13. Catapult Mechanism（カタパルト機構）

### 13.1 Lewkowycz et al. (2020)

| 項目 | 内容 |
|------|------|
| **論文** | "The Large Learning Rate Phase of Deep Learning: the Catapult Mechanism" |
| **著者** | Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, Guy Gur-Ari |
| **年** | 2020 (NeurIPS) |
| **URL** | https://arxiv.org/abs/2003.02218 |

**核心的発見 — 2つのレジーム**:

#### Lazy Phase（小学習率）
- $\eta < 2/\lambda_0$（$\lambda_0$ = 初期Hessian最大固有値）
- Neural Tangent Kernel (NTK) レジーム
- 線形化モデルとして記述可能

#### Catapult Phase（大学習率）
- $\eta > 2/\lambda_0$
- **学習損失が初期に指数的に増大**
- その間にHessianの最大固有値（曲率）が急速に減少
- 初期のシャープな盆地から「カタパルト」されて平坦な極小値に着地
- **より良い汎化性能**

**メカニズムの詳細**:
1. 大学習率 → 初期の高曲率盆地が不安定
2. 重みの急激な変動 → 曲率の再構成（sharpness低下）
3. 新しい低曲率領域に収束 → 損失が安定的に減少

**TG-LoRAへの関連**: ⭐⭐ **最も直接的に関連**。cycle 6での反転・ノルム半減は、catapult mechanismの低ランク版として理解できる可能性が非常に高い。特に：
- ノルム半減 ≈ 曲率の再構成による重みの「リセット」
- 方向反転 ≈ 高曲率盆地からの脱出時の振動
- cycle 7以降の安定化 ≈ 平坦領域への着地

**検証方法**: ΔWのノルムと方向（cosine similarity）をcycle 5→6→7で追跡し、catapultパターンとの整合性を確認。

---

## 14. Effective Rank / Participation Ratio

### 14.1 定義と使用法

#### Participation Ratio (PR)

$$\text{PR} = \frac{(\sum_i \sigma_i^2)^2}{\sum_i \sigma_i^4}$$

- 全固有値（特異値の2乗）が等しい場合: PR = 次元数（最大）
- 1つの固有値が支配的: PR → 1（最小）
- 「スペクトルがどれだけ分散しているか」の指標

#### Effective Rank

- PRはeffective rankの一つの定式化
- Shannon entropy ベースの定義もある: $\text{erank} = \exp\left(-\sum_i p_i \log p_i\right)$ where $p_i = \sigma_i / \sum_j \sigma_j$
- 代数的ランク（整数値）のノイズに対してロバストな実数値拡張

### 14.2 NNにおける使用例

| 文脈 | 使用法 |
|------|--------|
| 重み行列ダイナミクス | 学習中のeffective rankの減少（rank diminishing）の追跡 |
| 層の容量測定 | 高いeffective rank → 高い情報処理容量 |
| 汎化予測 | 低effective rank → 正則化された簡潔な表現 |
| ニューラル表現の次元性 | 集団活動の次元性測定（neuroscience由来） |

**TG-LoRAへの関連**: ⭐ r=2 LoRAのΔW = BA行列において：
- $\sigma_1, \sigma_2$ の比率でPR = $(σ_1^2+σ_2^2)^2 / (σ_1^4+σ_2^4)$
- PR ∈ [1, 2]の範囲で変動
- PR ≈ 1 → ほぼランク1 → 1方向への集中
- PR ≈ 2 → ランク2を十分活用 → 2方向への分散
- cycle 6でPRが変化するかどうかが重要な観測ポイント

---

## 15. LoRAの学習ダイナミクスとSVD解析

### 15.1 LoRA重み更新のスペクトル特性

**スペクトル疎性**: LoRA更新は低周波成分に集中（DCT解析）
- 約33%のDCT係数で全エネルギーの約90%を捕捉
- 高周波成分はノイズ → 除去しても性能維持

### 15.2 学習フェーズの構造

**勾配フロー解析による2フェーズ**:

1. **Alignment Phase（整列フェーズ）**: 学習初期、LoRA重みの特異ベクトルがタスク固有の部分空間に整列
2. **Fitting Phase**: 整列後、スケール調整による性能向上

**初期化スケールの影響**: 小さな初期化 → より良いalignment → より効果的な学習

### 15.3 層ごとの学習ダイナミクス差異

- **Centred Kernel Alignment (CKA)**: 入出力表現の類似度で層の重要度を測定
- 類似度が低い層 → タスク適応に重要
- **MoLA**: 重要な層により多くのLoRAエキスパートを配分
- **中間層が最も重要**: LoRA容量の適応的配分で性能向上

### 15.4 スペクトル幾何学による診断

- ΔWの特異値エントロピー、effective rank、cosine alignmentが学習目的の「指紋」として機能
- 異なるfine-tuning objective（DPOバリエーション等）のスペクトル幾何学的同定が可能

**TG-LoRAへの関連**: ⭐⭐ 直接的に方法論を適用可能。
- 各cycleでのΔW = BA のSVD → σ₁, σ₂, u₁, u₂, v₁, v₂の追跡
- cycle間のcosine similarity（方向変化）
- effective rank（PR）の時系列
- 層ごとのalignment進行度の差

---

## 16. Topological Data Analysis (TDA) in ML

### 16.1 Persistent Homologyの応用

**損失地形の位相的形状**: 
- Persistence diagramで地形の「穴」の生死を追跡
- 鞍点の数は平均persistenceに反比例
- 学習済みNNはデータのトポロジーを「単純化」する（Betti数の減少）

**層ごとの位相変化**:
- 深いネットワークは位相変化を層間でより均等に分散
- 浅いネットワークでは特定の層に位相変化が集中

**微分可能TDA**: Persistent homologyをNN層として統合（differentiable TDA layer）

### 16.2 ベクトル化手法

- Persistence Landscapes
- Persistence Images
- PersLay

**TG-LoRAへの関連**: cycle 6の反転を損失地形の位相的変化として捉える視点を提供。ただし計算コストが高く、r=2 LoRAの低次元空間では他の手法の方が効率的。

---

## 17. Layer-wise Learning Dynamics（層ごとの学習速度差）

### 17.1 層依存性の観察

- 浅い層（入力側）と深い層（出力側）で収束特性が異なる
- 転移学習: 初期層に低学習率（事前学習特徴の保持）、新規層に高学習率
- 浅い層は時に深い層より速く収束

### 17.2 適応的学習率スキーム

| 手法 | 基準 |
|------|------|
| 重み相関 / スペクトル密度 | Heavy-tailedness が弱い層に大学習率 |
| 曲率 / ノイズ適応 | 局所曲率・勾配分散に基づく動的調整 |
| CKA | 入出力表現変化が大きい層を重視 |

### 17.3 Martin & Mahoney のスペクトルベース層適応学習率

重み行列のESD（経験的スペクトル密度）のheavy-tailedness指標 $\hat{\alpha}$ を用いて、層ごとの学習率を適応的に設定する手法。

**TG-LoRAへの関連**: ⭐ Qwen3.5-9Bの32層（24層DeltaNet + 8層Attention）で、cycle 6の反転が：
- 全層同時 → global phenomenonとして解釈
- 層タイプ別（DeltaNet vs Attention）にずれ → アーキテクチャ依存性
- 深さ方向に伝播 → 波動的な遷移

この区別は我々の核心的な問いに直結。

---

## 18. 我々の実験への示唆

### 18.1 Cycle 6反転の理論的候補

我々が観測したcycle 6での「反転・ノルム半減・phase遷移」に最も近い先行事例：

| 順位 | 理論 | 適合度 | 理由 |
|------|------|--------|------|
| 1 | **Catapult Mechanism** | ⭐⭐⭐ | ノルム半減（曲率再構成）、方向反転（盆地脱出時の振動）、後続の安定化 |
| 2 | **Edge of Stability** | ⭐⭐⭐ | progressive sharpeningの後の振動・安定化パターン |
| 3 | **LoRA Alignment Phase遷移** | ⭐⭐ | 整列フェーズ→フィッティングフェーズの遷移点 |
| 4 | **Grokking的フェーズ遷移** | ⭐ | 記憶→汎化の遷移（ただし時間スケールが異なる可能性） |

### 18.2 検証すべき実験

#### A. 層ごとの同期性解析
```
観測: 各層ごとに以下を追跡
- ΔW = BA のノルム時系列
- ΔW の SVD → σ₁, σ₂, PR(= (σ₁²+σ₂²)² / (σ₁⁴+σ₂⁴))
- 連続cycleの ΔW 間のcosine similarity
- 層タイプ別集約（DeltaNet 24層 vs Attention 8層）
```

#### B. Edge of Stability検証
```
観測: cycle 5→6→7での
- 学習損失の非単調性（一時的増加→減少パターン）
- 勾配ノルムのスパイクと安定化
```

#### C. スペクトル遷移の追跡
```
観測: 各cycleでの
- σ₁/σ₂ 比率の変化（ランク1化 vs ランク2維持）
- 主特異ベクトル方向の回転角
- Martin-Mahoney α-hat の層ごと計算（可能であれば）
```

#### D. Catapult検証
```
予測: catapultモデルが正しければ
- cycle 5-6: ΔWノルムの急変＋方向反転
- cycle 7以降: ノルムの安定化＋方向の固定化
- effective rankの変化パターン
```

### 18.3 推奨ツール・手法

| 分析 | 手法 | 計算コスト |
|------|------|-----------|
| ΔW SVD | `torch.linalg.svd` | 低（r=2なので2特異値のみ） |
| Cosine similarity | 連続cycleの主特異ベクトル間 | 低 |
| Participation Ratio | $(σ₁²+σ₂²)² / (σ₁⁴+σ₂⁴)$ | 極低 |
| 層間同期性 | 全層のPR/ノルム/方向のcycle別ヒートマップ | 中 |
| 損失地形可視化 | Filter-normalized random directions (Li et al., 2018) | 高 |

---

## 19. 2026年6月 最新動向アップデート

> **更新日**: 2026-06-09 / **出典**: 下記 arXiv を Web 調査。abstract / 公開本文に基づき記述。

cycle 6 の Phase 遷移（反転・ノルム半減）と「スペクトル幾何で学習を診断する」という本トラックの方法論（§15.4）に対し、2025 後半〜2026 で重要な進展があった。

### 19.1 スペクトル幾何が「学習目的」を符号化する（§15.4 の強力な裏付け）

| 論文 | 年/arXiv | 知見 |
|------|----------|------|
| **Spectral Geometry of LoRA Adapters Encodes Training Objective** | 2026 / [2604.08844](https://arxiv.org/abs/2604.08844) | LoRA アダプタの**特異値エントロピー・effective rank・cosine alignment** が学習目的（objective）を符号化することを実証。教師なしで fine-tune の性質を監視・制御可能 |
| **The Primacy of Magnitude in Low-Rank Adaptation** | 2025 / [2507.06558](https://arxiv.org/abs/2507.06558) | スペクトル初期化（PiSSA等）の効果の本質は方向よりも**スケール（magnitude）**にあると分析。LoRA の学習ダイナミクスは非凸で予測困難と明言 |
| **Spectral Surgery: Training-Free Refinement of LoRA** | 2026 / [2603.03995](https://arxiv.org/abs/2603.03995) | 学習後の LoRA に**学習不要のスペクトル補正**を施し性能改善。ΔW の事後 SVD 操作という本プロジェクトの artifact 解析と同系統 |

> [!IMPORTANT]
> [2604.08844](https://arxiv.org/abs/2604.08844) は、本トラック §15.4 で挙げた「ΔW のスペクトル幾何が学習目的の指紋になる」という仮説を 2026 年に正面から実証した最新研究である。
> **cycle 6 の Phase 遷移検証（§18.2-C）に直接転用可能**: 各 cycle の特異値エントロピー・effective rank・cosine alignment の時系列が、遷移点で不連続に変化するかを見れば、Catapult/EoS と「学習目的の符号化変化」のどちらが起きているかを切り分けられる。

### 19.2 rank-1 軌跡の非線形性（cycle 6 反転との直接的整合）

トラック01 §21.2 で詳述した [arXiv:2604.11446](https://arxiv.org/abs/2604.11446)（rank-1 部分空間は線形進化せず、支配度が時間変化）は、本トラックの **cycle 6 反転・ノルム半減**と強く整合する。

> [!WARNING]
> 「rank-1 支配度が時間変化する」という最新の実証は、本プロジェクトの cycle 6 観測（rank-1 支配 0.78 が一定ではなく Phase 遷移で揺らぐ可能性）と符合する。**§18.2-C のスペクトル遷移追跡で支配度 σ₁/Σσ の時系列を必ず記録すべき**。

### 19.3 2026 サーベイ

- **A Unified Study of LoRA Variants: Taxonomy, Review** ([arXiv:2601.22708](https://arxiv.org/abs/2601.22708), 2026): LoRA 変種を4つの設計軸で体系化した最新サーベイ。トラック01/04 の分類更新の参照元として有用。

---

## 20. TDA総合：現時点の方向性（最重要）

> **更新日**: 2026-06-09 / 全6トラックの調査と既存実験事実（`docs/MEMO.txt`, GOAL.md）を統合した、学習ダイナミクス解析（TDA）からの結論。

### 20.0 出発点：中心問題は「相（regime）」であって「大域的線形性」ではない

TG-LoRA初期設計の前提「局所線形性 ⇒ 外挿でbackwardスキップ」は、TDAの全証拠と衝突する。学習は単一の滑らかな軌跡ではなく、**質的に異なる相の連なり**である。

- **Edge of Stability / Catapult**（§8, §13）: 大LR域での曲率再構成・損失の一時増大→平坦域への射出という**非線形イベント**。
- **Martin-Mahoney 5+1相**（§5）: スペクトルのランダム→重い裾への**相転移**。
- **Leap+Verify**（[Track05](file:///home/jinno/tg-lora/docs/research/05_speculative_extrapolation_zeroth_order.md)）: chaotic相では外挿受理率ほぼ0。
- **低ランク軌跡モデリング**（[2604.11446](https://arxiv.org/abs/2604.11446)）: rank-1部分空間は**線形進化せず**、支配度も時間変化。

→ **結論1: 外挿の妥当性は相に依存する。「いつでも外挿できる」は誤り。**

### 20.1 cycle 6 の正体はほぼ確定：グローバルCatapult相転移

`docs/MEMO.txt` の既存事実：**cycle6/cycle5 のノルム比は全層で一様に ~0.51、位相同期はグローバル相転移（非同期伝播ではない）**。

これを先行研究に重ねると：
- 全層同時・ノルム半減・方向反転 = **Catapult機構の低ランク版**（§13、適合度★★★）。
- Leap+Verifyの「相境界は地形の性質でseed間±50stepで一致」と符合 → **初期化のアーティファクトでなく構造的境界**。

→ **結論2: cycle 6 はグローバルなCatapult/EoS相転移。層ごと調整の対象でなく全体で一度起きるイベント。単一のグローバル相検出器で十分（安価）。**

### 20.2 導かれる設計の方向転換

| # | 転換 | 根拠 |
|---|------|------|
| **A** | 「常時外挿」→「**相ゲート付き外挿**」（stable/transitionのみ。cycle 6型の転移は強制pilot+rollback） | 相転移を跨ぐ線形外挿は方向反転で破綻 |
| **B** | prior方向 $v_0$ は**相転移の後で必ず再推定**（regime-local PSA。転移検出でsubspaceリセット） | cycle 6で方向反転＝転移前velocityは転移後無効 |
| **C** | 予測器の相応じた切替（stable:linear / transition近傍:quadratic）。**モメンタム外挿は永久不採用** | Leap+Verify §2（モメンタムは100〜10,000倍爆発） |
| **D** | 効率上限を「**相の在庫**」で再定義 | Qwen系はstable相が稀（Leap+Verify）→1.24×頭打ちは地形律速の可能性 |

### 20.3 最優先で測るべき計器盤（forward/SVDのみ、4bitで実装可）

| 計測 | 何が分かるか | コスト |
|---|---|---|
| activation-fingerprint cosineの時系列→3相分類 | **外挿可能局面の総量＝効率上限** | 低 |
| ΔW SVD: σ₁/Σσ の時系列 | rank-1支配度が一定か時間変化か（[2604.11446](https://arxiv.org/abs/2604.11446)検証） | 極低 |
| 連続cycle主特異ベクトルの回転角(cos) | 方向反転の発生cycleと急峻さ | 極低 |
| 全層ノルム比/PRのcycle×層ヒートマップ | cycle 6が波か同期か（MEMO既に同期と確認→裏取り） | 中 |

### 20.4 一言で言う方向性

> **TG-LoRA = 「低ランクLoRA学習の stable相における正則化非線形加速（RNA, [Track06](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md)）＋ グローバル相転移をforward信号で検出して外挿をゲート/priorをリセットする phase-aware制御」**。効率主張は **stable相の割合という地形的上限**に対してhonestに行う。cycle 6 はバグでも障害でもなく、設計が尊重すべきグローバルCatapult相転移である。

### 20.5 TDAが明示的に警告する罠

- **相転移を跨ぐ線形外挿** → 方向反転で破綻（cycle 6）。
- **rank-1支配度を定数と仮定** → 時間変化する（測れ）。
- **1.24×をバグと決めつけ実装を弄り続ける** → 地形律速なら徒労。**まず相の在庫を測れ**。

---

## 参考文献一覧

### 本質的次元
1. Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. *ICLR 2018*. https://arxiv.org/abs/1804.08838
2. Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*. https://aclanthology.org/2021.acl-long.568/

### 損失地形
3. Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. *NeurIPS 2018*. https://arxiv.org/abs/1712.09913

### モード接続性
4. Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A.G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. *NeurIPS 2018*. https://arxiv.org/abs/1802.10026
5. Draxler, F., Vossoughi, K., Fischer, A., & Brox, T. (2018). Essentially No Barriers in Neural Network Energy Landscape. *ICML 2018*. https://arxiv.org/abs/1803.00885

### フェーズ遷移
6. Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. https://arxiv.org/abs/2201.02177

### Edge of Stability / Catapult
7. Cohen, J., Kaur, S., Li, Y., Kolter, J.Z., & Talwalkar, A. (2021). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. *ICLR 2021*. https://openreview.net/forum?id=AXNjM_WF0GO
8. Lewkowycz, A., Bahri, Y., Dyer, E., Sohl-Dickstein, J., & Gur-Ari, G. (2020). The Large Learning Rate Phase of Deep Learning: the Catapult Mechanism. *NeurIPS 2020*. https://arxiv.org/abs/2003.02218
9. Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR 2021*.

### Hessian解析
10. Sagun, L., Evci, U., Güney, V.U., Dauphin, Y., & Bottou, L. (2016/2017). Empirical Analysis of the Hessian of Over-Parametrized Neural Networks. https://arxiv.org/abs/1706.04454

### Neural Collapse
11. Papyan, V., Han, X.Y., & Donoho, D.L. (2020). Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training. *PNAS*. https://www.pnas.org/doi/10.1073/pnas.2015509117

### SGDノイズ
12. Jastrzębski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Bengio, Y., & Storkey, A. (2018). Three Factors Influencing Minima in SGD. https://arxiv.org/abs/1711.04623

### Random Matrix Theory
13. Martin, C.H. & Mahoney, M.W. (2021). Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Training. *JMLR*. https://arxiv.org/abs/1901.08276

### LoRA
14. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. https://arxiv.org/abs/2106.09685

### 2026年6月 最新動向アップデート分（§19）
15. (2026). Spectral Geometry of LoRA Adapters Encodes Training Objective and Generalization. https://arxiv.org/abs/2604.08844
16. (2025). The Primacy of Magnitude in Low-Rank Adaptation. https://arxiv.org/abs/2507.06558
17. (2026). Spectral Surgery: Training-Free Refinement of LoRA via Gradient-based Spectral Editing. https://arxiv.org/abs/2603.03995
18. (2026). Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration (rank-1 非線形性). https://arxiv.org/abs/2604.11446
19. (2026). A Unified Study of LoRA Variants: Taxonomy, Review. https://arxiv.org/abs/2601.22708

---

> **次のステップ**: この文献調査に基づき、cycle 6反転の層ごと解析実験（§18.2）を設計・実行する。
