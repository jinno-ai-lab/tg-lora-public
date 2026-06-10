# Research Track 6: 系列加速・重み平均・Forward Gradient・適応HPO — 先行研究調査

> **調査日**: 2026-06-09 / **出典**: 下記 arXiv を Web 調査。abstract / 公開本文に基づき記述。
> **目的**: TG-LoRA の数学的核心（軌跡 iterate の外挿、有限差分による方向微分、ハイパーパラメータの適応探索）が、最適化・数値解析の**古典的かつ確立された理論体系**のどこに位置づくかを明確化し、車輪の再発明と既知の落とし穴を避ける。
> **TG-LoRA 文脈**:
> - velocity 外挿 = 最適化 iterate の系列加速（sequence acceleration）。
> - JVP 非サポート（4bit/bitsandbytes）→ 有限差分で方向微分（GOAL.md §2.1）。これは forward-mode AD / forward gradient の代替。
> - random_walk_controller が $K, N, \alpha, \beta$ を適応探索 → オンライン HPO。

---

## 目次

1. [系列加速：Anderson Acceleration / RNA（外挿の理論的故郷）](#1-sequence-acceleration)
2. [重み平均加速：SWA / LAWA / EMA（外挿の代替・補完）](#2-weight-averaging)
3. [Forward Gradient / Forward-mode AD（FD採用の理論的背景）](#3-forward-gradient)
4. [適応ハイパーパラメータ探索：PBT / Population-Based Bandits](#4-adaptive-hpo)
5. [TG-LoRA への統合提案](#5-tg-lora-integration)

---

## 1. 系列加速：Anderson Acceleration / RNA（外挿の理論的故郷）<a id="1-sequence-acceleration"></a>

TG-LoRA の「軌跡から方向 $v_0$ を推定し外挿する」操作は、数値解析の **sequence/vector extrapolation**（系列・ベクトル外挿）そのものである。最適化 iterate $\{\theta_t\}$ を後処理して不動点（最小点）を推定する古典体系。

### 1.1 Anderson Acceleration (AA)

| 項目 | 内容 |
|------|------|
| **基礎** | Walker & Ni (2011), "Anderson acceleration for fixed-point iterations." *SIAM J. Numer. Anal.* 49(4) |
| **核心** | 直近 $m$ 個の iterate と update step を**線形結合**して不動点反復の収束を加速。残差を最小化する係数を最小二乗で決定 |

### 1.2 Regularized Nonlinear Acceleration (RNA)

| 項目 | 内容 |
|------|------|
| **論文** | Scieur, d'Aspremont, Bach (2016), "Regularized Nonlinear Acceleration." *NeurIPS 2016* |
| **Online版** | "Online Regularized Nonlinear Acceleration." [arXiv:1805.09639](https://arxiv.org/abs/1805.09639) |
| **核心** | 勾配法等の iterate を後処理して関数の最小点を推定。**Anderson acceleration の正則化版**。係数推定の不安定性を Tikhonov 正則化で抑える |

### 1.3 DNN への最近の応用

| 論文 | 年/arXiv | 知見 |
|------|----------|------|
| **Anderson-type acceleration for DNN training** | 2025 / [2510.20254](https://arxiv.org/abs/2510.20254) | AA を DNN 学習の収束加速に適用。直近 iterate と update を結合 |
| **Regularized Anderson Acceleration for Off-Policy Deep RL** | 2019 / [1909.03245](https://arxiv.org/abs/1909.03245) | AA を RL のサンプル効率改善に適用 |

> [!IMPORTANT]
> **TG-LoRA の外挿は RNA/Anderson の特殊形として理論的に位置づけられる**。
> - TG-LoRA の「方向 $v_0$ ＝ 軌跡平均方向」「低次元係数 $(\alpha, \beta_1, \beta_2)$ のフィット」は、AA の「直近 iterate の線形結合係数を残差最小化で決定」と数学的に同型。
> - **正則化の必要性**: RNA が示す通り、係数推定は**本質的に不安定**であり Tikhonov 正則化（L2）が必須。GOAL.md §2.2 が FD 係数 $\alpha$ の不安定性に対し「L2 等の対策は後で判断」としているが、**RNA の理論は L2 正則化が原理的に必要だと示している**。後回しにせず最初から入れるべき。
> - 論文での位置づけ: TG-LoRA を「**深層学習・低ランク部分空間への RNA/Anderson の適用**」とフレーミングすれば、強固な理論的系譜に接続できる。

---

## 2. 重み平均加速：SWA / LAWA / EMA（外挿の代替・補完）<a id="2-weight-averaging"></a>

外挿が「軌跡の先を予測」するのに対し、重み平均は「軌跡上の点を平均」して加速・汎化する。TG-LoRA の競合・補完技術。

| 手法 | 年 | 核心 |
|------|----|------|
| **SWA** (Stochastic Weight Averaging) | 2018 | 後期の複数チェックポイントを平均し、平坦な最小値へ |
| **EMA** (Exponential Moving Average) | — | 重みの指数移動平均。安定化の定番 |
| **Lookahead Optimizer** | 2019 | fast weights を $k$ ステップ進めた後 slow weights を補間更新 |
| **LAWA / Early Weight Averaging** | COLM 2024 / [2306.03241](https://arxiv.org/abs/2306.03241) | **高学習率下で初期から直近チェックポイントを平均** → LLM 事前学習で収束高速化＋汎化改善。Pythia/GPT-2 で実証 |

**LAWA の理論（Wang et al., 2024）**: 小さな定数学習率下で平均が有効、減衰スケジュールでは効果が部分的に相殺される。

> [!IMPORTANT]
> **Lookahead は TG-LoRA の pilot/extrapolation 構造とほぼ同型**: fast weights（pilot step で進める）→ slow weights（補間で安定化）。TG-LoRA の外挿に **Lookahead 的な「補間係数 $\alpha$ で戻す」安定化**を組み込めば、大 $N$ での崩壊（GOAL.md §2.2）を緩和できる可能性。
> **LAWA との比較実験は必須のベースライン**: 「外挿（予測）」が「平均（LAWA）」より速いことを示せなければ、TG-LoRA の存在意義が問われる。LAWA は実装が極めて単純で強力なので、**強いベースライン**として置くべき。

---

## 3. Forward Gradient / Forward-mode AD（FD採用の理論的背景）<a id="3-forward-gradient"></a>

TG-LoRA が JVP 非サポートのため有限差分を使う背景には、**forward-mode AD で方向微分を求める**という本来の選択肢がある。その理論と限界。

| 論文 | 年/arXiv | 知見 |
|------|----------|------|
| **Gradients without Backpropagation** (Baydin et al.) | 2022 / [2202.08587](https://arxiv.org/abs/2202.08587) | forward-mode AD で**接ベクトル方向の方向微分（JVP）**を1回の forward で計算し、不偏勾配推定（forward gradient）を構成。backward 不要 |
| **Second-Order Forward-Mode AD for Optimization** | 2024 / [2408.10419](https://arxiv.org/abs/2408.10419) | forward-mode で2次情報（曲率）も取得し最適化に利用 |
| **Forward gradient guidance** | 2024 / [2410.17764](https://arxiv.org/abs/2410.17764) | forward gradient の分散を抑える誘導手法 |

**核心**: forward gradient $g = (\nabla f \cdot v)\, v$（$v$ はランダム接ベクトル）は不偏だが**高分散**。分散は次元に比例して増大する。

> [!IMPORTANT]
> **TG-LoRA の FD 採用の正確な理論的位置づけ**。
> - JVP（forward-mode AD）が使えれば、有限差分の数値誤差なしに方向微分が得られる。**TG-LoRA が FD を使うのは bitsandbytes 4bit が JVP 非サポートだから**（GOAL.md §2.1）であり、これは正しい回避策。
> - ただし forward gradient 文献の核心的教訓は「**方向微分ベースの勾配推定は高分散で、分散は次元に比例**」。TG-LoRA が**低次元部分空間（3次元）に射影してから FD する**設計は、まさにこの高分散を抑える正攻法であり、forward gradient 研究と整合する。
> - **2408.10419 の2次 forward-mode** は、TG-LoRA が外挿の曲率（quadratic 予測、トラック05 §1.2）を forward only で取得する具体的手段になりうる。

---

## 4. 適応ハイパーパラメータ探索：PBT / Population-Based Bandits<a id="4-adaptive-hpo"></a>

TG-LoRA の `random_walk_controller`（$K, N, \alpha, \beta$ をランダムウォークで適応探索）は、オンライン HPO の確立された枠組みに位置づく。

| 手法 | 年/arXiv | 核心 |
|------|----------|------|
| **PBT** (Population-Based Training) | 2017 | 複数モデルを並列学習し、性能の良い個体の重み＋ハイパラを periodically 継承・摂動（exploit + explore） |
| **PB2** (Population-Based Bandits) | 2020 | PBT の explore を **GP-bandit** に置換し、**証明付きで効率的な**オンライン HPO を実現。少ない個体数で有効 |
| **Generalized PBT (GPBT)** | 2024 / [2404.08233](https://arxiv.org/abs/2404.08233) | PBT の適応性と計算効率を改善。RL ベンチで従来 PBT を上回る |
| **Dynamic LR via Bandit (RL)** | 2024 (OpenReview) | 学習率をバンディットでオンライン調整 |

> [!IMPORTANT]
> **random_walk_controller は PB2（Population-Based Bandits）に置換することで原理的に強化できる**。
> - 現状のランダムウォークは**無誘導の探索**であり、サンプル効率が悪い。PB2 の GP-bandit は「過去の $(K,N,\alpha,\beta)$ → 報酬（loss改善/効率）」を回帰し、**次に試す価値の高い設定を証明付きで選ぶ**。
> - 単一 run 内での適応なら PB2 の time-varying GP bandit が、TG-LoRA の「サイクルごとにハイパラを動かす」設定と完全に一致する。
> - **無駄な設定探索を避ける**直接的処方。ランダムウォークの当てずっぽうを、報酬モデルベースの誘導探索へ。

---

## 5. TG-LoRA への統合提案<a id="5-tg-lora-integration"></a>

本トラックは TG-LoRA の各構成要素を、確立された理論体系へ接続する。

| TG-LoRA 構成要素 | 理論的故郷 | 即時アクション |
|------------------|-----------|----------------|
| velocity 外挿（方向 $v_0$＋係数フィット） | **RNA / Anderson Acceleration** | **L2 正則化を最初から導入**（RNA が原理的必要性を示す） |
| 有限差分による方向微分 | **Forward Gradient / forward-mode AD** | 低次元射影で高分散を抑える設計を維持。2次forward-modeで曲率取得を検討 |
| 大 $N$ での崩壊の安定化 | **Lookahead Optimizer** | 補間係数 $\alpha$ で slow-weight 的に戻す機構を追加 |
| 外挿そのものの有効性検証 | **LAWA / SWA** | **LAWA を強いベースライン**に。外挿が平均に勝つことを示す |
| random_walk_controller | **PB2 / GPBT** | ランダムウォークを GP-bandit へ置換し探索効率化 |

### 5.1 論文フレーミングの示唆

TG-LoRA は「**低ランク部分空間における正則化非線形加速（RNA）＋ regime-adaptive な投機検証（トラック05 Leap+Verify）**」と位置づけると、(a) 系列加速の古典理論、(b) 2026 の投機的重み予測、の両方に接続でき、新規性（低ランク×LoRA×phase-aware）が明確になる。

### 5.2 避けるべき落とし穴（既知）

> [!WARNING]
> - **無正則化の係数フィットは不安定**（RNA）→ L2 必須。
> - **方向微分推定は高分散**（forward gradient）→ 低次元射影・多方向平均・分散削減が必須。
> - **外挿は平均（LAWA）に負けうる**→ 必ず LAWA ベースラインと比較。
> - **無誘導ランダム探索は非効率**（PB2）→ bandit 化。

---

## 参考文献一覧

### 系列加速
1. Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM J. Numer. Anal.* 49(4).
2. Scieur, A., d'Aspremont, A., & Bach, F. (2016). Regularized Nonlinear Acceleration. *NeurIPS 2016*.
3. (2018). Online Regularized Nonlinear Acceleration. arXiv:1805.09639. https://arxiv.org/abs/1805.09639
4. (2025). Anderson-type acceleration method for Deep Neural Network training. arXiv:2510.20254. https://arxiv.org/abs/2510.20254
5. (2019). Regularized Anderson Acceleration for Off-Policy Deep RL. arXiv:1909.03245. https://arxiv.org/abs/1909.03245

### 重み平均
6. Izmailov, P. et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization (SWA). UAI 2018.
7. Zhang, M. et al. (2019). Lookahead Optimizer: k steps forward, 1 step back. NeurIPS 2019.
8. Sanyal, S. et al. (2024). Early Weight Averaging meets High Learning Rates for LLM Pre-training (LAWA). *COLM 2024*. arXiv:2306.03241. https://arxiv.org/abs/2306.03241

### Forward Gradient
9. Baydin, A.G. et al. (2022). Gradients without Backpropagation. arXiv:2202.08587. https://arxiv.org/abs/2202.08587
10. (2024). Second-Order Forward-Mode Automatic Differentiation for Optimization. arXiv:2408.10419. https://arxiv.org/abs/2408.10419
11. (2024). Forward gradient guidance. arXiv:2410.17764. https://arxiv.org/abs/2410.17764

### 適応HPO
12. Jaderberg, M. et al. (2017). Population Based Training of Neural Networks.
13. Parker-Holder, J. et al. (2020). Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits (PB2). NeurIPS 2020.
14. (2024). Generalized Population-Based Training for Hyperparameter Optimization in RL. arXiv:2404.08233. https://arxiv.org/abs/2404.08233

> **次のステップ**: §5 の即時アクションのうち、(1) 外挿係数への L2 正則化、(2) LAWA ベースライン比較、を最小実装して TG-LoRA の効率主張の頑健性を確認する。
