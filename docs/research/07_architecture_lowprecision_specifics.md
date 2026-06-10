# Research Track 7: TG-LoRA固有性 — Gated DeltaNetアーキテクチャ & 低精度数値条件

> **調査日**: 2026-06-09 / **出典**: 下記を Web 調査。abstract / 公開情報に基づき記述。未確認は明示。
> **目的**: トラック01-06が扱った「手法・理論の一般論」では落ちる、**TG-LoRA がまさに動かしている具体物**（Qwen3.5 のハイブリッド構造、4bit/bf16 下の有限差分）に固有の落とし穴を閉じる。
> **位置づけ**: 本トラックは「見落としを残さない」ための補完であり、効率/安定性に直結する**実装レベルの最終チェックリスト**。

---

## 目次

1. [Gated DeltaNet — Qwen3.5 が実際に使うアーキテクチャ](#1-gated-deltanet)
2. [低精度（4bit/bf16）下の有限差分・勾配推定の数値条件](#2-low-precision)
3. [TG-LoRA への含意](#3-implications)

---

## 1. Gated DeltaNet — Qwen3.5 が実際に使うアーキテクチャ<a id="1-gated-deltanet"></a>

| 項目 | 内容 |
|------|------|
| **論文** | "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang et al.) |
| **会議/実装** | ICLR 2025 / NVlabs [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) |
| **採用** | NVlabs リポジトリに「2026-02-17: Gated DeltaNet が Qwen3.5 を駆動」と明記 |

### 1.1 仕組み

- **線形注意（linear attention）系の状態更新**に **Delta Rule（誤差訂正的なメモリ更新）** と **Gating（適応的メモリ消去）** を組み合わせる。
- Mamba2 を Delta Rule で改善。retrieval/長文脈での線形注意の弱点を補う。
- **ハイブリッド構造**: 大半の層を Gated DeltaNet 状態更新にし、**数層ごとに重い full-attention 層**を挟む（Qwen3.5/Qwen3-Next/Kimi Linear 系の共通パターン）。
- AGENTS.md の記載と一致: **Qwen3.5-9B は 32層中24層が Gated DeltaNet、8層が標準 Attention**。

### 1.2 なぜ TG-LoRA に重要か

TG-LoRA の velocity/ΔW 解析・層サンプリングは、**機能的に異質な2種類の層**を横断している:

- **Gated DeltaNet 層（線形注意・状態空間的）**: 再帰的な状態更新。勾配の時間相関構造が full-attention と異なる。状態の gating により**実効的な時定数が層・チャネルで変動**しうる。
- **Full-attention 層（out_proj 等）**: トラック03 で「最も安定した学習信号（early_dir_cos 0.30-0.42）」と観測された層はこちら側。

> [!IMPORTANT]
> **out_proj 安定性仮説（トラック03）も、rank-1 支配（トラック01）も、"層タイプ別" に層別化して再解釈する必要がある。**
> `docs/MEMO.txt` の既存事実「Attention A の temporal mode count PR=5.78 / MLP PR=7.33」は、まさに**層タイプで軌跡の単純さが違う**ことを示している。Gated DeltaNet 層と Attention 層では、(a) ΔW の rank-1 支配度、(b) velocity の線形性、(c) 外挿可能性が**系統的に異なる**と予想すべき。
> **層サンプリング（layer_sampler）は「DeltaNet 24層 vs Attention 8層」を層タイプとして区別して設計すべき**であり、全層を等質に扱うと外挿の妥当性を見誤る。

### 1.3 関連

- **Gated DeltaNet-2** (NVlabs): channel-wise gate を追加し、erase gate が利得の大半を占めると ablation で確認。
- **Olmo Hybrid** も Gated DeltaNet を採用（2026-03）。線形注意×full-attention のハイブリッドが主流化。

> **[UNVERIFIED]**: Gated DeltaNet 層への LoRA 適用時の ΔW 構造・学習ダイナミクスに特化した先行研究は、本調査時点で明確なものを特定できていない。**TG-LoRA の層タイプ別解析は、この空白を埋める独自貢献になりうる**（論文上の差別化ポイント）。

---

## 2. 低精度（4bit/bf16）下の有限差分・勾配推定の数値条件<a id="2-low-precision"></a>

TG-LoRA は QLoRA（4bit 凍結ベース + bf16 LoRA）上で有限差分により方向微分を取る。GOAL.md §2.2 は「bf16 下で FD perturbation（~$w_{\text{traj}} \times 2\times10^{-5}$）がノイズ域に入り $\alpha$ が不安定」と分析済み。これに対する確立技法群:

| 論文 / 技法 | 年/arXiv | 知見 |
|------|----------|------|
| **QLoRA** | 2023 / [2305.14314](https://arxiv.org/abs/2305.14314) | 4bit NF4 + double quantization。16bit 微調整性能を保持。TG-LoRA の基盤 |
| **"Give Me BF16 or Give Me Death?"** (Kurtic et al.) | 2025 | 量子化の精度-性能トレードオフを体系評価。bf16 が安全圏という実務指針 |
| **FP4 All the Way: Fully Quantized Training** | 2025 / [2505.19115](https://arxiv.org/abs/2505.19115) | **Stochastic Rounding (SR)**, Differentiable Gradient Estimator (DGE), Random Hadamard Transform (RHT) で低精度学習の勾配ノイズを抑制 |

### 2.0 config 確定事実：DeltaNet 状態は fp32 保持

Qwen3.5-9B / Qwen3.6-35B-A3B の config.json は **`mamba_ssm_dtype: float32`** を持つ（本体は bf16）。線形注意（DeltaNet）の再帰状態は高精度で計算される。

> **FD への含意**: TG-LoRA の FD perturbation が bf16 丸めで消える問題は、**層タイプで深刻度が異なる**。DeltaNet 層の再帰路は fp32 で相対的に頑健、**Attention/FFN の bf16 経路がボトルネック**。→ Stochastic Rounding / fp32 化は **bf16 経路（Attention/FFN）に優先適用**するのが合理的（詳細は [Track08 §5](file:///home/jinno/tg-lora/docs/research/08_target_model_structure_prestudy.md)）。

### 2.1 核心的処方：Stochastic Rounding

- 低精度（bf16/4bit）では、微小な FD perturbation が**丸め誤差に飲まれて0になる/バイアスする**のが本質問題（決定的丸め）。
- **Stochastic Rounding** は期待値で真値を保つ丸めであり、**微小摂動の情報を確率的に保存**する。FP4/FP8 学習で標準的に有効性が示されている。

> [!IMPORTANT]
> **TG-LoRA の FD 不安定（GOAL.md §2.2）への直接的処方**:
> 1. **FD のロス評価とperturbationを fp32 で行う**（"Give Me BF16" の指針：致命的経路は高精度に）。
> 2. **Stochastic Rounding** を FD perturbation 適用時に使い、bf16 丸めで摂動が消える問題を回避。
> 3. **多方向平均**（トラック06 forward gradient）で分散低減。
> これらは「fp32化・バッチ拡大は後で判断」（GOAL.md §2.2）とされた対策に、**SR という低コストで効果的な選択肢**を追加する。

---

## 3. TG-LoRA への含意（最終チェックリスト）<a id="3-implications"></a>

| 領域 | アクション | 優先 |
|------|-----------|------|
| アーキ | rank-1支配/out_proj安定性/外挿可能性を **DeltaNet層 vs Attention層** で層別化して再測定 | ★★★ |
| アーキ | layer_sampler を層タイプ認識に（24 DeltaNet / 8 Attention を別プールに） | ★★ |
| 数値 | FD の評価・perturbation を **fp32 + Stochastic Rounding** に | ★★★ |
| 数値 | perturbation スケールを bf16 の ULP（unit in last place）以上に設定しているか確認 | ★★★ |

> [!WARNING]
> **最大の見落としリスク**: TG-LoRA の全ての層別結論（out_proj が安定、rank-1 支配 0.78 等）は、**DeltaNet 層と Attention 層を混ぜた平均**で語られている可能性がある。Qwen3.5 は 75% が DeltaNet 層なので、**「平均的傾向」が実は DeltaNet 層に支配されている**かもしれない。層タイプ別の分離は、誤った設計判断（例：Attention 用の戦略を DeltaNet 層に誤適用）を防ぐために必須。

---

## 参考文献一覧

1. Yang, S. et al. (2025). "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025. https://github.com/NVlabs/GatedDeltaNet
2. Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314. https://arxiv.org/abs/2305.14314
3. Kurtic, E. et al. (2025). "Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization."
4. (2025). "FP4 All the Way: Fully Quantized Training of LLMs." arXiv:2505.19115. https://arxiv.org/abs/2505.19115

> **次のステップ**: 層タイプ別（DeltaNet/Attention）の ΔW 解析と、FD への Stochastic Rounding 導入を、トラック02 §20 の計器盤計測と同時に実施する。
