# Research Track 5: 投機的重み外挿 & ゼロ次最適化 — 先行研究調査

> **調査日**: 2026-06-09 / **出典**: 下記 arXiv を Web 調査。abstract / 公開本文に基づき記述。
> **目的**: TG-LoRA のコア機構（速度ベクトルの**外挿**による backward スキップ ＋ **rollback による検証**、および JVP 非サポート環境での**有限差分**による方向微分）に最も近い 2025-2026 の先行研究を整理し、設計上の生死を分ける知見を抽出する。
> **TG-LoRA 文脈**:
> - M9 (Prior-based Subspace Learning): 低次元部分空間で有限差分勾配法により係数 $(\alpha, \beta_1, \beta_2)$ をフィット（GOAL.md §2.2）。
> - 観測された効率は **1.24 倍で頭打ち**（GOAL.md）。原因は FD の高分散（batch=1）と lr 崩落と推定されてきた。
> - cycle 6 で方向反転・ノルム半減の **Phase 遷移** が観測されている。

---

## 目次

1. [Leap+Verify — 投機的重み予測（TG-LoRA の双子）](#1-leapverify)
2. [Universal Momentum Catastrophe と有限差分の優位性](#2-momentum-catastrophe)
3. [Regime Detection（相検出）の方法論](#3-regime-detection)
4. [Weight Nowcasting（学習型重み予測）](#4-weight-nowcasting)
5. [ゼロ次（ZO）最適化](#5-zeroth-order)
6. [大規模 SFT 実証](#6-massive-sft)
7. [TG-LoRA への統合提案（最重要）](#7-tg-lora-integration)

---

## 1. Leap+Verify — 投機的重み予測（TG-LoRA の双子）<a id="1-leapverify"></a>

| 項目 | 内容 |
|------|------|
| **論文** | "Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training" |
| **年/arXiv** | 2026-02 / [2602.19580](https://arxiv.org/abs/2602.19580) |

### 1.1 全体像

LLM 推論の **speculative decoding**（投機実行：予測 → 検証 → 受理/棄却）を**学習加速**に適用。チェックポイント step $t$ の重み $\theta_t$ から $K$ ステップ先の $\theta_{t+K}$ を解析的予測器で予測し、**held-out loss で検証してから受理**する。受理時は $K$ ステップの勾配更新をスキップ、棄却時は副作用なしで通常学習を継続。

> **TG-LoRA との対応**: 予測 = TG-LoRA の **外挿（Extrapolation Step）**、検証/棄却 = TG-LoRA の **rollback**、$K$ = 投機ステップ数 $N$。**機構が事実上同型**であり、本論文の実証結果は TG-LoRA に直接転用できる。

### 1.2 3 つの予測器と受理基準

| 予測器 | 構成 | 備考 |
|--------|------|------|
| **Momentum** | Adam の $m_t/\sqrt{v_t}$ で等速外挿 | optimizer state ベース |
| **Linear** | 連続2チェックポイントから線形外挿（$\theta_t - \theta_{t-\Delta}$ の有限差分速度） | **trajectory-bounded** |
| **Quadratic** | 連続3点から放物線フィット（曲率＝加減速を捕捉） | **trajectory-bounded** |

**受理基準（3種）**:
- **Strict**: $\hat{L}_{t+K} < L_t$（予測が改善を示す場合のみ）
- **Adaptive**: $\hat{L}_{t+K} < L_t + \sigma_L$（直近 loss の1標準偏差以内）
- **Proximity (pct)**: $|\hat{L}_{t+K} - L_t| < \epsilon \cdot L_t$

**Cascade**: 予測を $D$ 回連鎖して最大 $D \times K$ ステップ前進。**stable regime からのみ**評価。

### 1.3 主要な実証結果

- **受理率**: GPT-2 124M で K=5・stable regime に **24%** strict 受理。Qwen 1.5B で K=5・transition regime に **37%**。
- **モメンタム外挿は全スケールで破滅**（§2）。
- **ボトルネックは予測精度ではなく regime availability**（§3）。

---

## 2. Universal Momentum Catastrophe と有限差分の優位性<a id="2-momentum-catastrophe"></a>

本論文の**最重要発見**:

| 予測器 | 124M (K=5 → K=100) | 1.5B (K=5 → K=100) |
|--------|--------------------|--------------------|
| **Momentum** | loss が実測の **122× → 10,764×** に爆発 | **173× → 3,009×** |
| **Linear/Quadratic (FD)** | 成功（有界） | 成功（有界） |

**メカニズム**:
- モメンタム予測 $K \cdot m_t/\sqrt{v_t}$ は、勾配履歴の指数移動平均を外挿するため、**現在点まわりの妥当領域（region of validity）を遥かに超える変位**を生み、ノルム爆発する。
- 有限差分予測器は**実際に取られたステップ幅で本質的に有界**であり、モメンタムに欠ける**自然な正則化**を持つ。学習率スケジュール・勾配ノイズ・地形曲率の正味効果を暗黙に取り込む。

> [!IMPORTANT]
> **TG-LoRA の FD 採用は理論・実証的に正しい**。GOAL.md §2.1 で JVP 非サポートのため有限差分を選んだのは「制約による妥協」と位置づけられていたが、Leap+Verify は **FD こそが投機的重み予測で唯一機能する外挿である**ことを実証した。モメンタム/optimizer-state 外挿（WNN, NiNo 等の学習型も含む）は本質的に不適と明言されている。**TG-LoRA は二度とモメンタム外挿に手を出すべきでない。**

---

## 3. Regime Detection（相検出）の方法論<a id="3-regime-detection"></a>

学習軌跡を **chaotic / transition / stable** の3相に実時間分類する。

### 3.1 Activation Fingerprinting

- 固定した **100 個のプローブ文**を毎チェックポイント（$\Delta = 50$ steps ごと）に通し、最終隠れ状態を連結した **activation fingerprint $\mathbf{a}_t$** を作る。
- 連続する fingerprint の **cosine similarity $s_t = \cos(\mathbf{a}_t, \mathbf{a}_{t-\Delta})$** を、学習軌跡の**局所 Lyapunov 指数の代理**として使う。
- 閾値 $\tau_{\text{high}}, \tau_{\text{low}}$（初期 seed から平均）で3相に分類。$s_t$ 高 = stable、低 = chaotic。

**利点**:
- Hessian 固有値計算（Cohen et al., 2021 = EoS）と違い、**forward pass のみ**で済む（2次勾配不要）。
- 重み空間の線形補間と違い、**表現空間**で機能的類似度を直接測る。

### 3.2 相検出は「予測の前提条件」

- chaotic 相では**どの予測器も意味ある受理率を達成できない**。相検出器が chaotic 中の予測試行を抑制し、**無駄な計算を防ぐ**。
- chaotic → stable の遷移は seed 間で **±50 step** と極めて一貫 → **相境界は最適化地形の性質であり、初期化のアーティファクトではない**。

> [!IMPORTANT]
> **cycle 6 Phase 遷移との直接対応**。TG-LoRA の cycle 6 反転は、Leap+Verify の **chaotic→stable（あるいは transition）相境界**に相当する可能性が高い。トラック02 §8/§13（EoS/Catapult）が Hessian ベースの説明だったのに対し、Leap+Verify は **forward only の activation cosine 信号**で同じ現象を捉えており、**TG-LoRA の 4bit 量子化環境（2次勾配が困難）でそのまま実装可能**。

---

## 4. Weight Nowcasting（学習型重み予測）<a id="4-weight-nowcasting"></a>

| 手法 | 概要 | Leap+Verify の評価 |
|------|------|--------------------|
| **WNN** (Jang & Han, 2023) | 重み空間で将来重みを学習予測 | optimizer-state 外挿に依存する手法は**本質的に不適**と指摘 |
| **NiNo** (Knyazev et al., 2025) | ニューラルネットで重み更新を予測 | 同上。trajectory-bounded 外挿への移行が必要 |

> **示唆**: 学習型の重み予測は魅力的だが、**観測された重み差分（trajectory）からの外挿に限定**しなければ破綻する。TG-LoRA の prior（軌跡から方向 $v_0$・スケール $w_{\text{traj}}$ を推定）は、まさにこの trajectory-bounded 原則に沿っている。

---

## 5. ゼロ次（ZO）最適化<a id="5-zeroth-order"></a>

TG-LoRA の有限差分は、本質的に **ZO（zeroth-order）勾配推定**（forward pass のみで勾配近似）と同じファミリー。ZO 文献の分散削減技法が、TG-LoRA の **batch=1 FD 高分散問題（GOAL.md §2.2）** に直接効く。

| 論文 | 年/arXiv | 知見 |
|------|----------|------|
| **MeZO / Revisiting ZO for Memory-Efficient LLM Fine-Tuning** | 2024 / [2402.11592](https://arxiv.org/abs/2402.11592) | ZO は backward 不要で推論レベルのメモリ。**LoRA チューニングは ZO のノイズに対して頑健**。momentum・適応 LR 等の分散削減で改善 |
| **Harmony in Divergence** | 2025 / [2502.03304](https://arxiv.org/abs/2502.03304) | 高速・高精度・省メモリな ZO の改良 |
| **ZO-Finetuner: Learning a Zeroth-Order Optimizer** | 2025 / [2510.00419](https://arxiv.org/abs/2510.00419) | **摂動戦略を学習**する ZO 最適化器。base LLM ごとに一度学習し全タスクで再利用。4 LLM×7 データセットで 82.1% の組合せで既存 ZO を上回る |

> [!IMPORTANT]
> **TG-LoRA の FD 分散問題への処方箋**。GOAL.md §2.2 は bf16 下で FD perturbation がノイズ域に入り $\alpha$ が不安定化、batch=1 の FD 勾配分散が大 $N$ 崩壊の主因と分析している。ZO 文献は同じ問題に対し **(1) 複数方向の平均化、(2) momentum/適応 LR による分散削減、(3) 学習型摂動（ZO-Finetuner）** という確立した処方を提供する。fp32 化・バッチ拡大の前に、ZO 流の分散削減を試す価値がある。

---

## 6. 大規模 SFT 実証<a id="6-massive-sft"></a>

| 論文 | 年/arXiv | 知見 |
|------|----------|------|
| **Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, ... Affect ...** | 2025 / [2506.14681](https://arxiv.org/abs/2506.14681) | データ規模×層×学習レジームの大規模実証。層別の寄与と学習レジームの相互作用を網羅的に測定 |

> **示唆**: TG-LoRA の層サンプリング（layer_sampler）設計の経験的根拠として参照可能。

---

## 7. TG-LoRA への統合提案（最重要）<a id="7-tg-lora-integration"></a>

Leap+Verify は TG-LoRA と同型でありながら、**先に大規模に検証した結果**を提供する。これを踏まえた設計判断:

### 7.1 確定事項（実験で再確認する必要が低い）

| 判断 | 根拠 |
|------|------|
| **FD（linear/quadratic）外挿を維持。モメンタム外挿は採用しない** | §2 Universal Momentum Catastrophe（全スケールで 100〜10,000× 爆発） |
| **検証（rollback）を必須とする。strict/adaptive/proximity の受理基準を導入** | §1.2。投機は副作用なしで安全 |
| **2次（quadratic）外挿を選択肢に追加** | 曲率を捉え、cycle 6 のような加減速局面に強い |

### 7.2 最優先で追加すべき機構：Regime Detector

> [!WARNING]
> **これが効率頭打ち（1.24×）を破る最有力候補**。TG-LoRA は現状「いつ外挿するか」を warmup 相の固定境界で決めているが、Leap+Verify は **chaotic 相では外挿が原理的に無効**であり、相検出なしの外挿は無駄計算だと示した。

具体案:
1. 固定プローブ文集合（~100 件、検証セットから）の **activation fingerprint** を各 cycle で計算。
2. 連続 fingerprint の **cosine similarity** を相信号とし、stable/transition でのみ外挿を許可、chaotic では pilot（通常学習）に徹する。
3. forward only なので **Qwen3.5-9B の 4bit 量子化環境でそのまま実装可能**（Hessian/JVP 不要）。

### 7.3 効率頭打ちの再解釈（無駄な実験を避けるための核心）

> [!WARNING]
> Leap+Verify は **Qwen 1.5B が学習の 64% を chaotic 相、stable 到達は 2.5% のみ**と報告。「大きいモデルは予測可能なときはより予測可能だが、予測可能な局面が稀」。
> → TG-LoRA が使う **Qwen3.5-9B でも、外挿が妥当な局面が本質的に少ない可能性が高い**。**1.24× の頭打ちは実装バグではなく regime-availability の構造的限界**かもしれない。
> **推奨**: fp32 化・バッチ拡大・クリッピング等の FD 安定化チューニングに工数を投じる前に、まず **TG-LoRA の各 cycle がどの相にいるか（activation cosine の時系列）を計測**し、stable/transition がどれだけ存在するかを確認する。stable がほぼ無いなら、FD をいくら安定化しても効率は上がらない。

### 7.4 検証実験の優先順位

| 優先 | 実験 | 目的 |
|------|------|------|
| ★★★ | 各 cycle の activation fingerprint cosine 時系列を計測し3相分類 | 外挿可能局面の量（効率上限）を先に把握 |
| ★★★ | cycle 6 が相境界（chaotic→stable）と一致するか確認 | Phase 遷移の正体特定（トラック02 と接続） |
| ★★ | linear vs quadratic 外挿の受理率比較 | 加減速局面での予測器選択 |
| ★★ | ZO 流の分散削減（多方向平均・momentum）を FD に適用 | batch=1 FD の大 $N$ 崩壊の緩和 |

---

## 参考文献一覧

1. (2026). Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training. arXiv:2602.19580. https://arxiv.org/abs/2602.19580
2. Jang & Han (2023). Weight Nowcaster Network (WNN).
3. Knyazev et al. (2025). NiNo: Neural network weight prediction.
4. Zhang, Y. et al. (2024). Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning. arXiv:2402.11592. https://arxiv.org/abs/2402.11592
5. (2025). Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-Order LLM Fine-Tuning. arXiv:2502.03304. https://arxiv.org/abs/2502.03304
6. (2025). Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs (ZO-Finetuner). arXiv:2510.00419. https://arxiv.org/abs/2510.00419
7. (2025). Massive Supervised Fine-tuning Experiments Reveal How Data, Layer ... Affect ... arXiv:2506.14681. https://arxiv.org/abs/2506.14681
8. Cohen, J. et al. (2021). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. ICLR 2021.

> **次のステップ**: §7.4 の最優先実験（activation cosine による相検出）を設計し、TG-LoRA の効率上限が regime-availability に律速されているかを先に判定する。
