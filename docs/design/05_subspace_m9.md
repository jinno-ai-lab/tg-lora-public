# 05_subspace_m9.md — Prior-based Subspace Learning (M9)

## 1. 事実 (Facts)

### 1.1 部分空間（v0/u1/u2）の構成と Gram-Schmidt 直交化

M9 では、ヒストリー軌跡から $3$ 次元部分空間を構成する。

*   **支配方向 $v_0$**: ヒストリー平均の単位化ベクトル。
    *   *コード引用*: [src/tg_lora/extrapolator.py:L930-936](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L930-L936)
*   **直交 PC 方向 $u_1, u_2$**: 中心化した軌跡に `pca_lowrank` を適用して得られた第1・第2主成分 $pc_1, pc_2$。
    *   *コード引用*: [src/tg_lora/extrapolator.py:L939-952](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L939-L952)
*   **グラム・シュミット（Gram-Schmidt）直交化**: $v_0 \rightarrow pc_1 \rightarrow pc_2$ の順に直交化され、最終的に直交基底 $\{v_0, u_1, u_2\}$ が得られる。
    *   *コード引用*: [src/tg_lora/extrapolator.py:L954-972](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L954-L972)

### 1.2 有限差分（Finite Difference）フィットと外挿適用

*   **パラメータ構成式 $W_{\text{extrap}}$**:
    $$W_{\text{extrap}}(\theta) = W_t + N \cdot \alpha \cdot w_{\text{traj}} \cdot v_0 + \beta_1 \cdot u_1 + \beta_2 \cdot u_2$$
    *   *コード引用*: [src/tg_lora/extrapolator.py:L983, L1024](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L983-L1024)
*   **有限差分フィット**:
    *   `fd_epsilon` (勾配差分幅): `0.001` (`configs/9b_tg_lora_m9.yaml` の `subspace_m9_fd_eps`)
    *   `fit_steps` (ステップ数): `1` (`configs/9b_tg_lora_m9.yaml` の `subspace_m9_steps`)
    *   `fit_lr` (学習率): `0.5` (`configs/9b_tg_lora_m9.yaml` の `subspace_m9_lr`)
    *   *コード引用*: [src/tg_lora/extrapolator.py:L994-1018](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L994-L1018)
*   **外挿の適用構造**:
    フィットされた $\theta = (\alpha, \beta_1, \beta_2)^T$ に基づいて、1回のみパラメータを大きく更新し、独立した検証セット `valid_quick_loader` にて答え合わせ評価（`loss_after`）および承認判定を行う。中間ステップでの逐次チェックは行われない。
    *   *コード引用*: [src/training/train_tg_lora.py:L2750-2819](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2750-L2819)

### 1.3 次元数アサート (Dimension Assertion)

外挿を行う際、ヒストリーの次元数と trainable な LoRA パラメータ数が完全に一致していることを検証する二重のアサートが実装されている。

*   **次元数一致チェック**:
    ヒストリー要素数がモデルの trainable なパラメータ数と完全に一致するか検証する。
    *   *コード引用*: [src/tg_lora/extrapolator.py:L906-911](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L906-L911)
*   **ゼロ要素率チェック**:
    ヒストリー内の 0 の割合が 50% 以上である場合にアサート例外を投げる。これにより、trainable_scope の不一致や不要なパラメータの混入を防止する。
    *   *コード引用*: [src/tg_lora/extrapolator.py:L914-919](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L914-L919)
*   **Trainable Scope の設定**:
    設定ファイルでは `trainable_lora_scope: last_25_percent` が指定されており、Qwen3.5-9B の 32層中、上部 8層（24〜31層）の全 linear 層のみが最適化対象（trainable）に設定される。これにより、実際の `num_active_params` は全 LoRA パラメータ数（43,278,336 個）の 25% である **10,819,584 個**に制限される。

---

## 2. 設計意図 (Rationale)

*   **なぜ M9 では $\beta_1, \beta_2$ などの直交成分（ステアリング）を追加した部分空間（Subspace）で最適化するのか**:
    *   固定された直線方向 $v_0$ の上のみで探索を行うと、最適化軌跡の曲がり角（ステアリングが必要な局面）に対応できず、外挿効率が頭打ちになってしまうため。
*   **なぜ次元一致とゼロ要素率の厳しいアサートを設けているのか**:
    *   過去履歴 delta とモデルパラメータのアクティブ次元（`num_active_params`）の整合性を保証し、`trainable_lora_scope` の不一致や不要なゼロパディングを起動時に即座に検知することで、データリークや不整合な外挿による崩壊を未然に防止するため（[extrapolator.py:L906-919](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L906-L919) の検証アサートに対応）。
*   **なぜ v0 のみ N 倍し u1/u2 を据え置くのか**:
    *   v0 は履歴平均=支配方向であり N サイクル分の一貫した進行を表すため N 倍が原理に忠実。u1/u2 は PCA 残差成分で各サイクル固有の揺らぎを表すため、N 倍すると過大変位となり崩壊を招く。よって u1/u2 は据え置く（[extrapolator.py:L1024](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L1024) の修正後式に対応）。

---

## 3. 設計思想 (Design Philosophy)

*   **根幹思想 (当時の設計意図)**:
    M9 外挿は「$N$ サイクル分を逐次的に勾配降下する」手法ではない。勾配の更新方向を低次元（1次元ないし3次元 $\{v_0, u_1, u_2\}$）の部分空間に圧縮し、その方向上で進み幅（係数）を実データに対してフィッティングして決める、という発想に基づく。フィッティングは厳密な勾配法ではなく、原始的な探索（モンテカルロ的サンプリング/有限差分）で係数を最適化する想定であった。
*   **なぜ逐次更新にしないか (計算量トレードオフ)**:
    $N$ サイクルを逐次に勾配降下すると backward を $N$ 回分消費し、TG-LoRA の存在意義である backward 削減（`reduction_rate`）が原理的に成立しない。そこで「進む方向は過去の accept 履歴から作った低次元部分空間に固定し、その上で進み幅だけをデータにフィットして決める」ことで、backward を追加せず forward 主体（報告では約6回）で外挿点を決定する設計を選んだ。これが効率化の本質である。
*   **この設計が成立する前提と既知の限界**:
    一発で大きく進む外挿は「損失地形が当該方向に $N$ サイクル先までほぼ直線」という前提に依存する。$N$ が小さい範囲では前提が概ね成立するが、$N$ が大きいと地形の曲率により着地点が谷から外れうる。途中サイクルのデータは個別の勾配としては使われず、低次元空間内の係数フィットの目的関数を通じてのみ寄与する（すなわち、途中各点での方向の測り直しは行わない）。

---

## 4. 既知の計測上の注意 (Known Measurement Caveats)

### 4.1 raw_delta_cosine_sim の比較対象のズレ (確認1, 2026-06-07)

*   **事実**: run_metrics.jsonl の `tg_lora_raw_delta_cosine_sim` は `cos(velocity._state, pilot_delta)` を測定しており、設計意図である `cos(velocity, M9_delta)` ではない。
    *   *コード引用*: `src/training/train_tg_lora.py:L1787` — `raw_delta_cos_sim = velocity.cosine_similarity(dW)`。`dW` は pilot delta (`W_K - W_0`) であり、M9 delta ではない。
    *   M9 delta と velocity の cosine は現在どこでも計算されていない。
*   **影響**: この metric は「pilot delta と velocity EMA の方向一致度」を示すものであり、「M9 外挿方向の品質」を直接反映しない。0.02〜0.06 の低い値は pilot delta のノイズまたは勾配方向の変化を示すだけで、M9 の v0 が悪い方向であることを意味しない（Sweep B で v0 方向の安定性が確認済み）。
*   **現在のステータス**: **未修正・要判断**。lr 崩落修正後の run 結果を見て、metric の意味を再評価し、比較対象を M9 delta に変更するかどうかを判断する。

---

## 5. 固定係数化と履歴全捨て純粋外挿方式 (2026-06-07)

### 5.1 事実 (Facts)

*   **FD フィットのバイパス**: `subspace_m9_fit_step` にて、FD 摂動ループ（fit_steps 回の反復による grad_alpha, grad_beta1, grad_beta2 の計算）を完全にバイパスし、固定係数 alpha=1.0, beta1=0.0, beta2=0.0 を返すよう変更した。
    *   *コード引用*: `src/tg_lora/extrapolator.py:L989-994`
*   **外挿公式の簡略化**: 上記により外挿式は $W_{\text{extrap}} = W_t + N \cdot w_{\text{traj}} \cdot v_0$ に簡略化される（$u_1, u_2$ 項は $\beta = 0$ により消滅）。
*   **accept 時の履歴全捨て**: M9 accept 後、`delta_tracker._history` および `_norm_history` をクリアし、`warmup_released` を False に、`warmup_cos_consecutive` を 0 にリセットする。
    *   *コード引用*: `src/training/train_tg_lora.py` M9 accept path
*   **reject 時の非クリア**: reject 時には履歴のクリアは行わない。accept のみがリセットのトリガーとなる。
*   **再ウォームアップ要件**: リセット後、cos >= 0.75 が 3 サイクル連続で再蓄積されるまで外挿は再開しない。

### 5.2 動機 (Motivation — 前回 run での事実)

*   前回 120 サイクル run における alpha の範囲: [-31.66, +7.17]、std=4.46（極端なノイズ）。
*   accept された 3 サイクルはいずれも alpha ≒ 0（v0 の寄与ほぼゼロ）であり、accept は偶然によるものであった。
*   3 件中 2 件の accept で loss が悪化した。
*   batch_size=1 かつ bfloat16 での FD フィットは、ノイズ支配の係数推定を生む。
*   これは純粋な v0 外挿の有効性を検証するための**診断実験**であり、FD フィットの恒久的な廃止を決定するものではない。

### 5.3 設計意図 (Rationale)

*   **alpha=1.0 固定の意図**: $N \cdot w_{\text{traj}} \cdot v_0$ という公式を純粋な形でテストする。もし N=5 や N=10 で loss 崩落が起きるなら、問題は方向またはスケールにあり、フィッティングではない。
*   **履歴全捨ての意図**: 大きな外挿ジャンプは蓄積された勾配方向履歴を無効化するという前提に基づく。再ウォームアップにより新鮮な方向を蓄積する。
*   **分離の意図**: 「v0 外挿はそもそも有効か？」という問いを「正しい係数をフィットできるか？」から独立して検証する。

### 5.4 既知のトレードオフ

*   **accept 頻度の低下**: accept ごとに約 20 サイクルの再ウォームアップが発生するため、reduction_rate の上限が下がる。
*   これはクリーンな診断のための受容済みトレードオフである。結果が良好であれば、部分的な履歴保持などのハイブリッド手法を検討できる。
