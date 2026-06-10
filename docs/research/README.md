# TG-LoRA 先行研究・関連文献調査ハブ

本ディレクトリには、TG-LoRA (Tangent-Gradient LoRA) の理論的基盤を強化し、観測された実験的事実に対する科学的解釈を与えるための、先行研究・関連文献調査の結果がまとめされています。

---

## 調査トラックと関連資料一覧

| ドキュメント | 調査テーマ | カバーする主要なキーワード |
|:---|:---|:---|
| [01_lora_variants_layer_analysis.md](file:///home/jinno/tg-lora/docs/research/01_lora_variants_layer_analysis.md) | **LoRA変種と層別・A/B行列構造解析** | AdaLoRA, LoRA-FA, DoRA, PiSSA, LoRA+, GaLore, ReLoRA, rsLoRA, ランク適応, SVD初期化 |
| [02_training_dynamics_analysis.md](file:///home/jinno/tg-lora/docs/research/02_training_dynamics_analysis.md) | **学習ダイナミクス解析 (TDA)** | 本質的次元数 (Intrinsic Dimensionality), Loss Landscape, 勾配フロー, ランダム行列理論 (RMT), Mode Connectivity, Edge of Stability |
| [03_attribution_layer_importance.md](file:///home/jinno/tg-lora/docs/research/03_attribution_layer_importance.md) | **帰属手法と層の重要度解析** | ShortGPT (Block Influence), SliceGPT, 統合勾配 (Integrated Gradients), Fisher情報量, 知識ニューロン (ROME/MEMIT), Attention vs MLP |
| [04_snr_adaptive_amplification.md](file:///home/jinno/tg-lora/docs/research/04_snr_adaptive_amplification.md) | **SNRマップと適応的増幅・選択的学習** | LARS/LAMB, 勾配ノイズスケール, 選択的学習 (Lottery Ticket, BitFit), 漸進的アンフリージング, 勾配フィルタリング (SAM), B行列フィルタ仮説 |
| [05_speculative_extrapolation_zeroth_order.md](file:///home/jinno/tg-lora/docs/research/05_speculative_extrapolation_zeroth_order.md) | **投機的重み外挿 & ゼロ次最適化（TG-LoRAコア機構の直系）** | Leap+Verify, speculative weight prediction, regime detection (activation fingerprint), 有限差分 vs モメンタム外挿, Weight Nowcasting (WNN/NiNo), ゼロ次最適化 (MeZO/ZO-Finetuner) |
| [06_sequence_acceleration_forward_gradient.md](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md) | **系列加速・重み平均・Forward Gradient・適応HPO（TG-LoRAの数学的故郷）** | Anderson Acceleration, RNA (Regularized Nonlinear Acceleration), SWA/LAWA/Lookahead, Forward Gradient (Baydin), forward-mode AD, PBT/PB2 (Population-Based Bandits) |
| [07_architecture_lowprecision_specifics.md](file:///home/jinno/tg-lora/docs/research/07_architecture_lowprecision_specifics.md) | **TG-LoRA固有性：Gated DeltaNetアーキ & 低精度数値条件** | Gated DeltaNet (Qwen3.5の24/32層), 線形注意, 層タイプ別解析, QLoRA/4bit, Stochastic Rounding, FP4学習, FD数値条件 |
| [08_target_model_structure_prestudy.md](file:///home/jinno/tg-lora/docs/research/08_target_model_structure_prestudy.md) | **（番外）研究対象モデル構造 & ギミック投入点の事前検討** | Qwen3.5-9B層構造(32層/24GDN+8Attn), Qwen3.6-35B-A3B(40層/256experts MoE), LoRA適用点, ギミック投入点マップ, Track A/B差分, 検証仮説の優先順位 |

---

## 2026年6月 最新動向アップデート

2026-06-09 の Web 調査で追補した最新研究（**ADA**=適応ランク／**Attribution**=帰属の2軸）。詳細は各トラックの末尾セクションを参照。

| キーワード | 手法 | arXiv | 所在 |
|:---|:---|:---|:---|
| **ADA**（適応ランク） | **ARD-LoRA**（学習可能スケーリングでper-head動的ランク, TV正則化） | [2506.18267](https://arxiv.org/abs/2506.18267) | [01 §21.1](file:///home/jinno/tg-lora/docs/research/01_lora_variants_layer_analysis.md) |
| **ADA**（初期化） | **LoRA-DA**（漸近解析ベースのデータ考慮初期化） | [2510.24561](https://arxiv.org/abs/2510.24561) | [01 §21.1](file:///home/jinno/tg-lora/docs/research/01_lora_variants_layer_analysis.md) |
| **軌跡外挿** | **低ランク軌跡モデリング**（rank-1 は線形進化しない＝線形外挿への反証。NExt同一性[UNVERIFIED]） | [2604.11446](https://arxiv.org/abs/2604.11446) | [01 §21.2](file:///home/jinno/tg-lora/docs/research/01_lora_variants_layer_analysis.md) |
| **軌跡外挿** | **RELEX**（軌跡レベルSVD+rank-1閉形式線形フィット） | [2605.21468](https://arxiv.org/abs/2605.21468) | [01 §21.2](file:///home/jinno/tg-lora/docs/research/01_lora_variants_layer_analysis.md) |
| **Attribution** | **FIM-LoRA**（LoRA-B勾配分散eFIMで層別ランク, v_proj・中間層が高ランク） | [2605.16800](https://arxiv.org/abs/2605.16800) | [03 §9](file:///home/jinno/tg-lora/docs/research/03_attribution_layer_importance.md) |
| **学習ダイナミクス** | **Spectral Geometry of LoRA**（特異値エントロピー/effective rankが学習目的を符号化） | [2604.08844](https://arxiv.org/abs/2604.08844) | [02 §19.1](file:///home/jinno/tg-lora/docs/research/02_training_dynamics_analysis.md) |
| **学習ダイナミクス** | **The Primacy of Magnitude**（スペクトル初期化の本質はスケール） | [2507.06558](https://arxiv.org/abs/2507.06558) | [02 §19.1](file:///home/jinno/tg-lora/docs/research/02_training_dynamics_analysis.md) |
| **SNR/適応ゲイン** | **MoLS**（モジュール間勾配ノイズ不均衡をSNRで校正, §2.5プレースホルダID確定） | [2605.05794](https://arxiv.org/abs/2605.05794) | [04 §10.1](file:///home/jinno/tg-lora/docs/research/04_snr_adaptive_amplification.md) |
| **SNR/反証** | **Learning Rate Matters**（適切なLRなら素のLoRAで十分＝適応ゲインの上乗せ効果に注意） | [2602.04998](https://arxiv.org/abs/2602.04998) | [04 §10.2](file:///home/jinno/tg-lora/docs/research/04_snr_adaptive_amplification.md) |
| **サーベイ** | **A Unified Study of LoRA Variants**（2026 LoRA変種体系化） | [2601.22708](https://arxiv.org/abs/2601.22708) | [02 §19.3](file:///home/jinno/tg-lora/docs/research/02_training_dynamics_analysis.md) |
| **投機的外挿 ★最重要** | **Leap+Verify**（投機的重み予測＝TG-LoRA同型。モメンタム外挿は破滅、FDのみ機能、相検出が前提） | [2602.19580](https://arxiv.org/abs/2602.19580) | [05](file:///home/jinno/tg-lora/docs/research/05_speculative_extrapolation_zeroth_order.md) |
| **ゼロ次最適化** | **ZO-Finetuner**（摂動戦略を学習。FD分散削減の処方） | [2510.00419](https://arxiv.org/abs/2510.00419) | [05 §5](file:///home/jinno/tg-lora/docs/research/05_speculative_extrapolation_zeroth_order.md) |
| **系列加速 ★理論基盤** | **RNA / Anderson Acceleration**（iterate外挿の古典理論。係数フィットにL2正則化が原理的に必要） | [1805.09639](https://arxiv.org/abs/1805.09639) | [06 §1](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md) |
| **重み平均** | **LAWA**（直近チェックポイント平均で高速化＝外挿の強いベースライン） | [2306.03241](https://arxiv.org/abs/2306.03241) | [06 §2](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md) |
| **Forward Gradient** | **Gradients without Backpropagation**（方向微分による勾配＝FD採用の理論的背景、高分散） | [2202.08587](https://arxiv.org/abs/2202.08587) | [06 §3](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md) |
| **適応HPO** | **PB2 / GPBT**（GP-banditでオンラインHPO＝random_walk_controllerの強化） | [2404.08233](https://arxiv.org/abs/2404.08233) | [06 §4](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md) |

> **TG-LoRA への含意（要注意）**:
> - ★ [1805.09639](https://arxiv.org/abs/1805.09639)（RNA）: TG-LoRA の外挿は**正則化非線形加速の特殊形**。係数フィットは原理的に不安定で **L2正則化は後回しにせず最初から必須**（GOAL.md §2.2 の「後で判断」を是正）。
> - [2306.03241](https://arxiv.org/abs/2306.03241)（LAWA）: 「外挿（予測）」が「平均」に勝てるかは自明でない。**LAWA を強いベースラインに据えた比較なしに効率主張をしてはいけない**。
> - [2202.08587](https://arxiv.org/abs/2202.08587)（Forward Gradient）: 方向微分ベース勾配は**次元比例で高分散**。TG-LoRA の低次元射影＋多方向平均は正攻法。
> - ★ [2602.19580](https://arxiv.org/abs/2602.19580)（Leap+Verify）は TG-LoRA と**同型機構**で、(1) **モメンタム外挿は全スケールで破滅（100〜10,000×）／有限差分のみ機能** → TG-LoRA の FD 採用は正しい、(2) **ボトルネックは予測精度ではなく regime availability** → 効率1.24×頭打ちは構造的限界の可能性、(3) **activation cosine による相検出が forward only で実装可能** → cycle 6 診断と「いつ外挿するか」の制御に直結。
> - [2604.11446](https://arxiv.org/abs/2604.11446) は「rank-1 部分空間は線形に進化しない／支配度が時間変化する」と報告しており、**cycle 6 の Phase 遷移**および TG-LoRA の局所線形外挿仮説の適用範囲（線形窓の外では非線形/区分線形へ切替）を再検討すべき根拠となる。
> - [2602.04998](https://arxiv.org/abs/2602.04998) は「適応ゲイン/非対称LRの利得は十分なLR探索で消えうる」と反証。**無駄な増幅ギミック実装を避けるため、素のLoRA+調整済みLRを強いベースラインに据えた ablation が必須**。
> - [2604.08844](https://arxiv.org/abs/2604.08844) は ΔW のスペクトル幾何が学習目的を符号化することを実証 → cycle 6 診断（特異値エントロピー時系列）に直接転用可能。

---

## 核心的な発見と TG-LoRA 実験事実との整合

これまでの実験で観測された TG-LoRA 固有の現象について、先行研究に基づき以下のように位置づけと解釈を与えることができます。

### 1. 各テンソルの更新が実質1次元に集中する現象（rank-1 支配度 0.78）
- **先行研究の知見**: **AdaLoRA** (Zhang et al., 2023) や **PiSSA** (Meng et al., 2024)、および **Intrinsic Dimensionality** (Li et al., 2018) の研究は、LLMの微調整に必要な空間が非常に低次元であることを示しています。
- **解釈**: 各テンソルの $\Delta W$ の特異値分解における第1特異値が支配的（~0.78）であることは、極低ランク（$r=2$）への射影において大部分の有効な勾配情報が保持されていることを数学的に裏付けています。層内集中・層間非整列という構造は、大局的な勾配方向の予測を阻むものの、層別の局所的な固有空間上での最適化や増幅ギミックの余地を残しています。

### 2. out_proj が最も安定した学習信号を持つ（early_dir_cos 0.30 - 0.42）
- **先行研究の知見**:
  - **ShortGPT** (Men et al., 2024) や **ROME/MEMIT** (Meng et al., 2022) によれば、Attention層とMLP層は役割が異なり、Attentionは情報をルーティング（構造的・安定）、MLPは知識を格納（局所的・不安定）します。
  - 特別に `out_proj`（Attentionの最終射影層）は、各ヘッドで抽出された特徴を統合するボトルネック層であり、勾配のノイズが平均化されやすく、最も高い「Block Influence」を持ちます。
- **解釈**: `out_proj` において初期速度ベクトルの類似度（early_dir_cos）が一貫して高く出るのは、この層がタスク依存の重要なルーティング軸を決定しているためであり、外挿やゲイン制御（増幅）の最有力候補として理論的に極めて妥当です。

### 3. B行列がA行列の更新信号をフィルタしている仮説（Bフィルタ仮説 0.99）
- **先行研究の知見**:
  - **LoRA-FA** (Zhang et al., 2023) や **LoRA+** (Hayou et al., 2024) 等の非対称LoRA研究、および低ランク射影による暗慢的自己正則化の理論。
  - $B_{t-1} @ \Delta A_t$ の安定性が $0.99$ と極めて高い現象。
- **解釈**: $A$ 行列は入力情報の低次元圧縮軸（射影基底）を学習し、高次元への復元を担う $B$ 行列が一種の「直交空間への通過ゲート」として機能している可能性があります。シャッフルサロゲート検定（[04_snr_adaptive_amplification.md](file:///home/jinno/tg-lora/docs/research/04_snr_adaptive_amplification.md#8-lora-b-matrix-as-filter)）によって、これが「B行列そのものの慣性（静的な射影構造）」なのか「$\Delta A$ との時間的な動的協調（真のフィルタ作用）」なのかを判別することが、設計の生死を分ける決定打になります。

### 4. サイクル6における反転・ノルム半減・Phase遷移
- **先行研究の知見**:
  - **Edge of Stability** (Cohen et al., 2021) や **Catapult Mechanism** (Lewkowycz et al., 2020) は、学習率が曲率の最大固有値 $\lambda_{max}(H)$ を超えた際に、損失が一時的にスパイク/揺らぎ、その後モデルが鋭い谷から平坦な谷へと脱出（ノルムの再調整と方向変化）する現象を記述しています。
- **解釈**: cycle 6 で起きたノルムの半減と方向の急変は、モデルが学習初期の「不安定性の縁（Edge of Stability）」に達し、局所的な曲率の壁に跳ね返されてより平坦な（安定な）領域へ遷移した物理的 Phase 遷移である可能性が高いです。これが全層で同期しているか、層ごとに時間差（伝播）があるかを見ることで、LLMの学習がグローバルにカスケードしているかローカルに調整されているかの境界が明らかになります。

---

## 提案される次のアクション

調査結果に基づき、TG-LoRAプロジェクトが次の意思決定を行うための具体的なアプローチを提案します。

1. **Bフィルタ仮説のシャッフル検定（最優先）**:
   - `B_{t-1}` を別サイクルの $B$ に差し替えた際の $B_{rand} @ \Delta A_t$ の安定性を測定し、真の協調フィルタ作用が存在するかどうかを白黒つけます（[03_attribution_layer_importance.md](file:///home/jinno/tg-lora/docs/research/03_attribution_layer_importance.md) / [04_snr_adaptive_amplification.md](file:///home/jinno/tg-lora/docs/research/04_snr_adaptive_amplification.md) に詳細手順あり）。
2. **層別SNRマップに基づく「適応ゲイン/ゲイン抑制」の実装設計**:
   - 勾配の安定性が高い `out_proj` などの層に対しては、**LoRA+** 的アプローチでB行列側の学習率をさらに高める（ゲインをかける）、逆にノイズ度合いが高い（支配特異値の平坦な）層やMLPの深層に対しては、勾配をアテニュエーション（減衰）する適応的増幅ギミックのプロトタイプ作成。
3. **時間×層 Phase マップの可視化による同期性の検証**:
   - cycle 6 の挙動が、入力層から順に伝播していく「波（wave）」なのか、最適化器の制約による「グローバル同期」なのかを、層別の gradient norm / SVDスペクトル変化率の時系列ヒートマップによって可視化します。
