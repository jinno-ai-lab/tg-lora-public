# Research Track 8（番外編）: 研究対象モデルの構造解析 & ギミック投入点の事前検討

> **調査日**: 2026-06-09 / **出典**: HuggingFace モデルカード（Qwen3.5-9B / Qwen3.6-35B-A3B）, apxml, mlabonne blog。確証できない箇所は **[UNVERIFIED]** を付し、`make inspect`（`scripts/inspect_model.py`）での実測確認を推奨。
> **目的**: TG-LoRA が実際に学習する2モデル（Track A: 9B / Track B: 35B-A3B）の**正確なレイヤ構造**を確定し、トラック01-07の知見を踏まえて「**どの層・どのモジュールに、どんなギミックを入れると成果が出そうか**」を実装前に設計する。
> **重要前提**: 両モデルとも **3:1 ハイブリッド**（Gated DeltaNet 線形注意 : Gated Attention フル注意）。**機能的に異質な層を TG-LoRA は横断する**（トラック07 §1）。

---

## 目次

1. [Qwen3.5-9B（Track A）の構造](#1-qwen35-9b)
2. [Qwen3.6-35B-A3B（Track B, MoE）の構造](#2-qwen36-35b)
3. [LoRA適用点（all-linear が触る実際のモジュール）](#3-lora-targets)
4. [ギミック投入点マップ（どこに何を入れるか）](#4-gimmick-map)
5. [Track A / Track B の設計差分](#5-track-diff)
6. [成果が出そうな仮説の優先順位](#6-hypotheses)

---

## 1. Qwen3.5-9B（Track A）の構造<a id="1-qwen35-9b"></a>

| 項目 | 値 |
|------|-----|
| Type | Causal LM + Vision Encoder（ネイティブマルチモーダル） |
| パラメータ | 9B |
| Hidden Dimension | 4096 |
| **層数** | **32** |
| **レイアウト** | **8 × ( 3×(Gated DeltaNet→FFN) → 1×(Gated Attention→FFN) )** |
| → DeltaNet層 | **24層**（線形注意） |
| → Attention層 | **8層**（フル注意、各ブロックの4層目） |
| Gated DeltaNet | V head 32 / QK head 16, head dim 128 |
| Gated Attention | Q head 16 / KV head 4（GQA）, head dim 256, RoPE dim 64 |
| FFN | Intermediate 12288（**dense / MoEではない**。config.json: `model_type: qwen3_5`, `num_experts`なし ✅確定） |
| Vocab | 248,320 |
| MTP | `mtp_num_hidden_layers: 1`（dedicated embeddingsなし） |
| Context | 262,144（YaRNで最大1,010,000） |
| その他（config確定） | `attn_output_gate: true`（出力ゲートでattention sink抑制）, **`mamba_ssm_dtype: float32`**（DeltaNet状態はfp32保持）, `linear_conv_kernel_dim: 4`（短conv）, `partial_rotary_factor: 0.25`（RoPE dim=256×0.25=64） |

### 1.1 層インデックスと層タイプ（0-indexed）

4層ごとのブロック構造。**各ブロックの末尾（index 3,7,11,15,19,23,27,31）が Gated Attention 層**、残り24層が Gated DeltaNet 層。

```
[0 1 2]GDN  [3]Attn | [4 5 6]GDN [7]Attn | ... | [28 29 30]GDN [31]Attn
```

> **TG-LoRA への直結事実**: トラック03 で「out_proj が最安定（early_dir_cos 0.30-0.42）」と観測された層は、この **8つの Attention 層** に属する可能性が高い。`docs/MEMO.txt` の「Attention A の PR=5.78 / MLP PR=7.33」も層タイプ差を示す。**全層別結論は DeltaNet/Attention で分離して再解釈すべき**（トラック07 §1.2）。

---

## 2. Qwen3.6-35B-A3B（Track B, MoE）の構造<a id="2-qwen36-35b"></a>

| 項目 | 値 |
|------|-----|
| Architecture | Hybrid sparse MoE（Gated DeltaNet + Gated Attention + MoE FFN） |
| 総パラメータ | 35B |
| **アクティブパラメータ** | **3B**（A3B） |
| Hidden Dimension | 2048 |
| **層数** | **40** |
| Attention | GQA, Q head 16 / KV head 2, head dim 256, RoPE θ=10,000,000 |
| **Experts** | **256（8 routed + 1 shared = 9 active）** |
| FFN Intermediate（per Expert） | 512 |
| Vocab | 248,320 |
| MTP | head 1 |
| Context | 262,144（YaRNで最大1M） |
| Release | 2026-04-15, Apache 2.0 |
| **DeltaNet:Attention 比** | **30 DeltaNet : 10 Attention**（config.json `layer_types` で full_attention が index 3,7,...,39。`full_attention_interval: 4` ✅確定） |
| MoE詳細（config確定） | `moe_intermediate_size: 512`, `num_experts: 256`, `num_experts_per_tok: 8`（+shared expert 1, `shared_expert_intermediate_size: 512`）, `router_aux_loss_coef: 0.001` |
| その他（config確定） | `attn_output_gate: true`, `mamba_ssm_dtype: float32`, `num_key_value_heads: 2`（GQA） |

### 2.1 MoE がもたらす TG-LoRA 固有の論点

- **専門家（expert）ごとに活性化頻度が大きく異なる**: 256 experts のうちトークンごとに 8 routed のみ発火。**稀にしか発火しない expert は勾配がスパースでノイジー** → velocity 推定が不安定。
- **router（gating linear）はモデル全体の挙動を支配**するが、パラメータは小さい。

> [!IMPORTANT]
> **MoE では「外挿していい expert」と「外挿してはいけない expert」が分かれる**。頻繁に routed される expert は velocity が安定し外挿向き、cold expert はサンプル不足で外挿が破綻する（トラック06 forward gradient の高分散がさらに悪化）。**routing 統計（各 expert の発火回数）を SNR の代理として使い、hot expert のみ外挿**するのが Track B 固有の設計。

---

## 3. LoRA適用点（all-linear が触る実際のモジュール）<a id="3-lora-targets"></a>

AGENTS.md の方針 `target_modules="all-linear"` で PEFT が自動検出する Linear を層タイプ別に整理。

| 層タイプ | 主な Linear（LoRA対象） | 備考 |
|----------|-------------------------|------|
| **Gated DeltaNet** | q_proj, k_proj, v_proj, 各種 gate proj（decay/erase gate）, β proj, out_proj 等 [UNVERIFIED:正確な命名は実測] | 再帰的状態更新。時間相関が強い |
| **Gated Attention** | q_proj, k_proj, v_proj, o_proj, output gate proj | GQA（KV少）。o_proj が安定候補 |
| **FFN（9B: dense）** | gate_proj, up_proj, down_proj（SwiGLU） | intermediate 12288 |
| **MoE FFN（35B）** | 各 expert の gate/up/down（×256）, router gate | expert数膨大→LoRA対象の選別必須 |

> **注意**: Vision Encoder 部分は SFT 対象外（`AutoModelForCausalLM` で言語モデル部分のみ、AGENTS.md）。LoRA も言語モデル側のみ。

---

## 4. ギミック投入点マップ（どこに何を入れるか）<a id="4-gimmick-map"></a>

トラック01-07の知見を、構造上の投入点に割り当てる。**スコープ（global / 層タイプ別 / モジュール別）が成否を分ける**。

| ギミック | 投入スコープ | 投入点 | 出典トラック | 期待効果 |
|----------|-------------|--------|-------------|----------|
| **相検出（regime detector）** | **Global**（1個） | モデル出力の activation fingerprint | 02 §20 / 05 | 「いつ外挿するか」を決定。最大の効率レバー |
| **phase-aware 外挿ゲート** | Global | 外挿スケジューラ | 02 §20 / 05 | cycle 6 型転移を跨ぐ外挿を禁止 |
| **prior subspace リセット** | Global（転移検出時） | extrapolator / lora_state | 02 §20-B | 方向反転後の v0 無効化を回避 |
| **L2正則化（外挿係数）** | Global | extrapolator の係数フィット | 06 §1 (RNA) | 係数不安定の原理的対策。**最初から必須** |
| **Stochastic Rounding FD** | Global（数値） | velocity の FD perturbation | 07 §2 | bf16でperturbationが消える問題の解消 |
| **層タイプ別 layer_sampler** | **層タイプ別** | DeltaNet/Attention/FFN を別プール | 07 §1 | 異質な層を等質に扱う誤りを回避 |
| **SNRベース適応ゲイン** | **モジュール別** | 各 LoRA モジュール | 04 (MoLS) | 高SNRモジュールを増幅 |
| **動的ランク（ARD/Salient）** | モジュール別 | rank割当 | 01 §21 | rank-1支配層はr=1、高ランク要求層はr↑ |
| **hot-expert限定外挿** | **expert別（Track Bのみ）** | MoE expert の velocity | 08 §2.1 | cold expertの外挿破綻を回避 |

### 4.1 「どこに入れると成果が出そうか」の構造的直感

- **Attention 層（8層 / 10層）= 外挿の主戦場**: フル注意 o_proj は最も学習信号が安定（トラック03）。**まずこの少数の安定層で外挿の有効性を実証**し、効率を稼ぐ。
- **DeltaNet 層（24層 / 30層）= 慎重に**: 再帰状態の gating で実効時定数が変動。velocity の線形性が崩れやすい → **相検出と組み合わせ、stable 相でのみ外挿**。
- **FFN/MoE = ランク適応の対象**: rank-1支配が強い層は r=1 に削減して計算節約（SalientLoRA）。MoE は hot expert に資源集中。

---

## 5. Track A / Track B の設計差分<a id="5-track-diff"></a>

| 観点 | Track A (9B, CUDA) | Track B (35B-A3B, MLX) |
|------|--------------------|------------------------|
| FFN | Dense（12288） | **MoE（256 experts, 9 active）** |
| 量子化基盤 | bitsandbytes 4bit（JVP非サポート→FD） | MLX 4bit |
| FD実装 | bf16 + Stochastic Rounding 推奨（07 §2） | MLX の数値挙動を別途確認 [UNVERIFIED] |
| 外挿の難所 | DeltaNet層の非線形性 | **expert routing のスパース性** |
| layer_sampler | 層タイプ別（DeltaNet/Attn/FFN） | 層タイプ別 **＋ expert発火頻度別** |
| 相検出 | activation cosine（共通） | 同左（MoEでも出力空間で測れる） |

> [!WARNING]
> **Track B 固有の最大リスク**: MoE の expert は発火がスパースなため、TG-LoRA の velocity 追跡が **「外挿に十分なサンプル」を得られない expert が大量に出る**。Track A で確立した外挿ロジックを MoE にそのまま適用すると、cold expert で破綻する。**routing 統計に基づく外挿可否ゲートが Track B の前提条件**。

> [!IMPORTANT]
> **config 発見の FD への含意（`mamba_ssm_dtype: float32`）**: 両モデルとも DeltaNet の状態更新は **fp32 保持**（本体は bf16）。つまり DeltaNet 層の再帰計算路は高精度。TG-LoRA の FD perturbation が bf16 丸めで消える問題（トラック07 §2）は **DeltaNet 層では fp32 状態のおかげで軽減される可能性**があり、逆に **Attention/FFN の bf16 経路では深刻**。→ Stochastic Rounding はまず bf16 経路（Attention/FFN）に優先適用するのが合理的。

---

## 6. 成果が出そうな仮説の優先順位<a id="6-hypotheses"></a>

| 優先 | 仮説 | 検証方法 | 期待 |
|------|------|----------|------|
| ★★★ | **8つのAttention層のo_projで外挿が最も有効**（安定・線形窓が広い） | 層タイプ別に外挿受理率を比較 | 少数層で効率を稼ぐ実証 |
| ★★★ | **効率上限はstable相の割合で律速**（DeltaNet層が大半なので相依存が強い） | activation cosineで相在庫を計測（02 §20.3） | 1.24×頭打ちの真因特定 |
| ★★ | **DeltaNet層はrank-1支配がさらに強い**（再帰状態の低次元性） | 層タイプ別ΔW SVD | rank削減で計算節約 |
| ★★ | **Track BはMoEのhot expertでのみ外挿が成立** | expert発火頻度×外挿受理率 | MoE外挿の設計指針 |
| ★ | **MTPヘッド（両モデル）は補助損失として相検出の信号源になりうる** | MTP予測の安定性を相信号に | 追加の相検出チャネル [UNVERIFIED] |

### 6.1 実装前の必須確認（`make inspect`）

**config.json で確定済み（実測不要）**:
> - ✅ **2. 9B は dense FFN**（`model_type: qwen3_5`, `num_experts` なし）。MoEではない。
> - ✅ **3. 35B は 30 DeltaNet : 10 Attention**（`layer_types` 配列, `full_attention_interval: 4`）。9B は 24:8。
> - ✅ **層タイプ判別は config 駆動**: `config.text_config.layer_types[i]` が `linear_attention` / `full_attention` を明示。TG-LoRA は実行時にこの配列を読めば層タイプを確実に分類できる。

> [!IMPORTANT]
> **実機ランタイムでの `make inspect`（`scripts/inspect_model.py`）が必要な残項目**:
> 1. **Linear モジュールの正確な命名**: Qwen3-Next系慣例から推定（linear_attn層: `in_proj_qkvz`/`in_proj_ba`/`conv1d`(非Linear)/`out_proj`、attn層: `q_proj`/`k_proj`/`v_proj`/`o_proj`、MoE: `mlp.gate`(router)/`mlp.experts.{i}.{gate,up,down}_proj`/`mlp.shared_expert.*`）だが、**正式名は modeling ソースで要確認** [UNVERIFIED]。
> 4. **PEFT `all-linear` の MoE 実挙動**: 原則は「lm_head を除く全 `nn.Linear` を対象」。Qwen MoE の expert が per-expert `nn.Linear`（`Qwen3MoeMLP` 形式）なら **256 expert 全てに LoRA が付き、router(`gate`)にも付く**（routerへのLoRAは routing 不安定化リスク → 明示除外を検討）。expert が融合3Dテンソル実装なら検出されず付かない。**どちらかは実機で要確認**。

---

## 参考（構造情報の出典）

1. HuggingFace: Qwen/Qwen3.5-9B モデルカード（Model Overview）. https://huggingface.co/Qwen/Qwen3.5-9B
2. HuggingFace: Qwen/Qwen3.6-35B-A3B モデルカード. https://huggingface.co/Qwen/Qwen3.6-35B-A3B
3. apxml: Qwen3.6 35B A3B Specifications. https://apxml.com/models/qwen36-35b-a3b
4. M. Labonne, "Qwen3.5: Nobody Agrees on Attention Anymore." HuggingFace blog. https://huggingface.co/blog/mlabonne/qwen35
5. Yang, S. et al. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025. arXiv:2412.06464
6. **config.json（権威ソース、本調査で取得）**: https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json , https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json
7. 本リポジトリ: AGENTS.md（Qwen固有の注意点）, docs/research/07（アーキ固有性）, docs/MEMO.txt（層タイプ別観測）

> **次のステップ**: `make inspect` で §6.1 を実測確認し、本mdの [UNVERIFIED] を確定値に更新。その後、§6 の★★★仮説（Attention層o_projでの外挿実証＋相在庫計測）から着手する。
