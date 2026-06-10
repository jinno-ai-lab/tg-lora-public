# 03_gating_and_N_selection.md — 投機判定とN決定

## 1. 事実 (Facts)

### 1.1 コサイン類似度による $N$ 制限ゲート

投機ステップ数 $N$ は、短期方向（現在のパイロットステップの重み差分 $dW$）と長期的な速度ベクトル $v_0$ の類似度に基づいて決定される。

*   **閾値テーブル (Config)**:
    [configs/9b_tg_lora_m9.yaml:L83-88](file:///home/jinno/tg-lora/configs/9b_tg_lora_m9.yaml#L83-L88) の `cosine_n_selection_thresholds` に規定されている。
    ```yaml
      cosine_n_selection_thresholds:
        1: 0.70
        3: 0.70
        5: 0.75
        10: 0.80
        20: 0.90
    ```
*   **決定ロジック (`choose_N`)**:
    [src/tg_lora/velocity.py:L237-256](file:///home/jinno/tg-lora/src/tg_lora/velocity.py#L237-L256)
    ```python
    def choose_N(
        self,
        N_candidates: list[int],
        c_threshold_map: dict[int, float],
    ) -> int:
        ...
        consistency = self.predicted_consistency()
        candidates = sorted(set(int(n) for n in N_candidates), reverse=True)
        thresholds = {int(n): float(c) for n, c in c_threshold_map.items()}
        for n_steps in candidates:
            c_min = thresholds.get(n_steps)
            if c_min is not None and consistency >= c_min:
                return n_steps
                
        # If consistency is below the absolute minimum threshold, return 0 (no extrapolation)
        min_threshold = min(thresholds.values()) if thresholds else 0.70
        if consistency < min_threshold:
            return 0
            
        return min(candidates)
```
    *   コサイン類似度 `consistency` が閾値マップ内の最小閾値（デフォルト `0.70`）を下回る場合は、**`0` を返す**。

### 1.2 外挿適用のフォールバック

*   **`proposal.N <= 0` 時のスキップ挙動**:
    `choose_N` が `0` を返した場合、または何らかの要因で `proposal.N <= 0` となった場合、[src/training/train_tg_lora.py](file:///home/jinno/tg-lora/src/training/train_tg_lora.py) の `_should_fallback_to_baseline_like` は `True` を返す。
    *   該当コード: [src/training/train_tg_lora.py:L551-552](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L551-L552)
        ```python
        if proposal_N <= 0:
            return True, "no_extrapolation_requested"
```
    *   これによって外挿は行われず、通常LoRA（Flag: B、ログ上は `no_extrapolation_requested`）としてそのサイクルが完了する。

### 1.3 逐次チェックや中間チェックの有無と検証スキップ

*   **適用構造の実態**:
    外挿は、決定された $N$ に対して「1回の大きなジャンプ（パラメータ一括適用）を行い、独立した検証セット `valid_quick_loader` に対する Forward ロスを測定して着地チェック（答え合わせ）を1回行う」構造になっている。
    *   外挿中の「適応的中間チェック」や「逐次的にチェックしながら $N$ を進める」ロジックは、コード上の外挿および答え合わせ部分には存在しない。
    *   該当コード: [train_tg_lora.py:L2750-2819](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2750-L2819)
*   **検証のスキップ (validation skip) の廃止と独立検証の実施**:
    かつては M9 有効時に `m9_batch` によるその場での即座の判定が行われ、独立した検証セットでの答え合わせはスキップされる仕様であった。しかし、データリーク防止のためこれが廃止され、M9 有効時にも `post_extrapolation_eval = True` および `post_extrapolation_eval_skipped = False` としてマークされ、後半の独立した検証セット `valid_quick_loader` での評価が必ず行われるようになった。
    *   該当コード: [train_tg_lora.py:L2383-2386](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2383-L2386) (M9適用時のバイパス無効化), [train_tg_lora.py:L2750-2790](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2750-L2790) (検証セットによる評価)

---

## 2. 設計意図 (Rationale)

*   **なぜ $N$ の中間ステップに逐次的なチェックや早期確定処理を設けていないのか**:
    *   TBD（人間確定待ち）
*   **なぜコサイン類似度の閾値を 0.70〜0.90 という厳しいマップ設定にしたのか**:
    *   TBD（人間確定待ち）
