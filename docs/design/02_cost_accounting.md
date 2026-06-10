# 02_cost_accounting.md — 効率会計の正本

## 1. 事実 (Facts)

### 1.1 状態（Flag）別のコスト計算

TG-LoRA の学習ループにおいて、1サイクルあたりの実 backward および forward コストは、サイクルの最終状態に応じて以下のように計算・累積される。

1.  **通常フォールバック (Flag: B)**
    *   コサイン類似度閾値チェックまたは Linearity Guard により外挿がスキップされた状態。
    *   実 backward 消費数: `pilot_K * grad_accum`
    *   等価（削減）backward 数: `0`
    *   Forward 消費数: `loss_pilot` 評価用（32サンプル、評価バッチサイズ 1、すなわち **`32` 回の Forward**）。
    *   該当コード: [src/training/train_tg_lora.py:L1841-1851](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1841-L1851) ( `speculative_optimizer_steps=0`, `speculative_equivalent_backward_passes=0` )
2.  **外挿承認 (Flag: Y)**
    *   外挿が適用され、かつ独立した検証セット（`valid_quick_loader`）による評価にて改善が認められ承認された状態。
    *   実 backward 消費数: `pilot_K * grad_accum`
    *   等価（削減）backward 数: `proposal.N * grad_accum`
    *   Forward 消費数: M9フィットでの `6` 回（フィットバッチサイズ 1 での Forward） ＋ 答え合わせ評価（`valid_quick_loader` 32サンプル、評価バッチサイズ 1、すなわち **`32` 回の Forward**） ＝ **`38` 回の Forward**。
    *   該当コード: [train_tg_lora.py:L2831-2839](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2831-L2839) ( `speculative_optimizer_steps=proposal.N`, `speculative_equivalent_backward_passes=proposal.N * grad_accum`, `accepted=True` )
3.  **外挿拒否 / ロールバック (Flag: N)**
    *   外挿が適用されたが、独立した検証セット（`valid_quick_loader`）による評価にて悪化し、pilot 後の状態へロールバックされた状態。
    *   実 backward 消費数: `pilot_K * grad_accum` （無駄になった pilot の実 backward はすべて消費として累積される）。
    *   等価（削減）backward 数: `0`
    *   Forward 消費数: M9フィットでの `6` 回（フィットバッチサイズ 1 での Forward） ＋ 答え合わせ評価（`valid_quick_loader` 32サンプル、評価バッチサイズ 1、すなわち **`32` 回の Forward**） ＝ **`38` 回の Forward**。
    *   該当コード: [train_tg_lora.py:L2831-2839](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2831-L2839) ( `speculative_optimizer_steps=0`, `speculative_equivalent_backward_passes=0`, `accepted=False` )

### 1.2 `reduction_rate` の計算式とコード対応

削減率 `reduction_rate` は、[src/tg_lora/cycle_state.py](file:///home/jinno/tg-lora/src/tg_lora/cycle_state.py) の `CycleState` クラスのプロパティとして定義されている。

*   **実際のコード実装**:
    [src/tg_lora/cycle_state.py:L204-210](file:///home/jinno/tg-lora/src/tg_lora/cycle_state.py#L204-L210)
    ```python
    @property
    def reduction_rate(self) -> float:
        total = (
            self.full_backward_passes + self.speculative_equivalent_backward_passes
        )
        if total == 0:
            return 0.0
        return 1.0 - self.full_backward_passes / total
```

*   **分子と分母の対応**:
    *   `self.full_backward_passes` (分子):
        これまでに実際に行われた累積実 backward パス数（すべてのサイクルでの `actual_backward_passes` すなわち pilot step 分の backward パスの累積値）。
    *   `total` (分母):
        実際に行われた累積実 backward パス数（分子）と、外挿承認によってスキップ（削減）された累積等価 backward パス数（`speculative_equivalent_backward_passes`）の合算値。
    *   **物理的変位との乖離に関する事実（M9での修正と整合性）**:
        かつての Original TG-LoRA では、`speculative_equivalent_backward_passes` の計算に `proposal.N * grad_accum` が乗算されているにもかかわらず、物理的な重みの外挿変位には `N` が適用されておらず、移動距離が常に約1サイクル相当（$w_{\text{traj}}$）であるという乖離（事実上のバグ）が存在した。
        しかし、M9設計への移行（Fixed）に伴い、[src/tg_lora/extrapolator.py:L983, L1024](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L983) にて外挿変位 `m9_delta` の $v_0$（支配方向）成分のスケールに `selected_N` を直接乗算する修正が行われた。これにより、物理的な移動距離と削減率上の削減ステップ数（等価 backward 数）の定義が一致し、乖離が完全に解消された。

---

## 2. 設計意図 (Rationale)

*   **なぜ実 backward パスの削減率を主要な評価指標とし、wall-clock時間（実行速度）を前面に出さないのか**:
    *   wall-clock は GPU・実装・バッチサイズに強く依存し再現性と公平性を担保しにくいため主要指標から除外する（[06_experiment_protocol.md](file:///home/jinno/tg-lora/docs/design/06_experiment_protocol.md) の wall-clock 非主張方針と整合）。backward パス数はアルゴリズムが要求する勾配計算回数そのものでハードウェア非依存の計算量本質指標であるため、これを正本とする。
*   **なぜ外挿に要した Forward コスト（M9フィットの6回、チェックの32回）を `reduction_rate` の分母・分子に反映させないのか**:
    *   forward と backward は計算特性が異なるため単一指標への合算は誤解を招く。よって reduction_rate は backward のみを対象とする。ただし eval_batch_size=1 下では1サイクル38 forward(fit 6 + 評価 32)のオーバーヘッドが無視できず backward 節約を相殺しうるため、これは reduction_rate とは別に「forward オーバーヘッド vs backward 節約の総コスト」指標として併記する。
