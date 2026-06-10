# 01_algorithm_dataflow.md — 1サイクルの完全データフロー

## 1. 事実 (Facts)

TG-LoRA の1サイクルは、[src/training/train_tg_lora.py](file:///home/jinno/tg-lora/src/training/train_tg_lora.py) の `train_tg_lora` 関数 (L726) 内のメインループ `for cycle in pbar:` (L1412) において定義される。
以下に、1サイクル内で実行される処理を順序に沿って列挙する。

### 1.1 局所学習フェーズ (Pilot Step)
*   **処理**:
    現在の重み $W_0$ から $K$ ステップの通常の QLoRA 更新を行う。
    *   `K` の値は `proposal.K` （初期値は `3`。`configs/9b_tg_lora_m9.yaml` の `K_initial` で指定され、コントローラにより `K_candidates: [2, 3, 5, 8]` から適応的に選択される）。
*   **実 backward の消費**:
    ステップごとに `grad_accumulation` (8) 回の `forward_backward` を行う。
    オプティマイザの更新も含めて合計で **`K * grad_accumulation` 回（K=3, accum=8 なら 24回）の実 backward** が行われる。
    該当コード: [src/training/train_tg_lora.py:L1482](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1482) の `for _ in range(pilot_K):`
*   **Pilot ロス（loss_pilot）の評価**:
    パイロットステップ完了直後に、検証用データセット（`valid_quick_loader`）の 32サンプルで Forward 損失を評価する。
    *   消費コスト: **`32` サンプル評価（評価バッチサイズ 1、すなわち `32` 回の Forward）**。
    *   該当コード: [src/training/train_tg_lora.py:L1540-1580](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1540-L1580)
 
### 1.2 勾配速度更新と投機判定
*   **処理**:
    パイロット更新差分 $dW = W_K - W_0$ を計算し、[src/tg_lora/velocity.py](file:///home/jinno/tg-lora/src/tg_lora/velocity.py) の `Velocity` 状態を更新してコサイン類似度 `consistency = predicted_consistency` を算出する。
    該当コード: [src/training/train_tg_lora.py:L1786-1788](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1786-L1788)
*   **投機ステップ数 $N$ の決定**:
    `choose_N` によりコサイン類似度に基づいて $N$ を選択する。
    *   コサイン類似度が `0.70` 未満の場合、`choose_N` は `0` を返す。
    *   該当コード: [src/tg_lora/velocity.py:L237-256](file:///home/jinno/tg-lora/src/tg_lora/velocity.py#L237-L256)
*   **無駄な投機の回避 (Skip Route)**:
    `proposal.N <= 0` である場合、`_should_fallback_to_baseline_like` は `True` を返す。プログラムは外挿処理に入らず、このサイクルを通常LoRA（Flag=B）として終了する。
    *   該当コード: [src/training/train_tg_lora.py:L529](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L529) および [L1816](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1816)

### 1.3 部分空間フィットフェーズ (M9 Fit)
*   **処理**:
    `proposal.N > 0` かつガードによるスキップが発生しない場合、[src/tg_lora/extrapolator.py](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py) の `subspace_m9_fit_step` を呼び出す。
    M9設計への移行に伴い、引数として `selected_N` を渡し、平均方向 $v_0$ のスケールを $N$ 倍してフィットを行う。
*   **データの消化量と消費コスト**:
    フィット用のミニバッチ `fit_batch`（オプティマイザの1ステップ分）を使用する。
    有限差分 gradient descent のため、偏微分の計算（`compute_loss_at`）を複数回呼び出す。
    *   `fit_steps = 1` のとき、初期・最終評価を含めて **合計 `6` 回の Forward（実 backward は 0回）** を消費する。
    *   該当コード: [src/tg_lora/extrapolator.py:L980-1019](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L980-L1019)

### 1.4 外挿の適用 (Extrapolation)
*   **処理**:
    フィットされた座標係数（$\alpha, \beta_1, \beta_2$）に基づき、部分空間上の $N$ ステップ先の外挿重みを算出して適用する。
    *   計算上の backward コスト: **`0` 回**。
    *   外挿の構造: 1回の大きなジャンプを行う。中間ステップでの逐次チェックや適応的チェックは実装されていない。
    *   該当コード: [src/tg_lora/extrapolator.py:L1020-1027](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py#L1020-1027)

### 1.5 答え合わせと承認/拒否判定 (Accept/Reject)
*   **処理**:
    データリーク防止のため、フィットに使用したトレーニングミニバッチ `m9_batch` での判定は廃止され、独立した検証セット（`valid_quick_loader`、32サンプル）を用いた Forward 損失評価により承認判定を行う。
*   **検証のスキップを行わない経路**:
    M9フィット後、[train_tg_lora.py:L2383-2386](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2383-L2386) にて `post_extrapolation_eval = True` および `post_extrapolation_eval_skipped = False` が設定され、検証スキップは行わずにループ後半の共通判定ロジックへ移行する。
*   **独立した validation による答え合わせと承認/拒否**:
    検証セット（`valid_quick_loader`、32サンプル、評価バッチサイズ 1、すなわち `32` 回の Forward）を用いて外挿適用後の `loss_after` を計算する ([train_tg_lora.py:L2750-2790](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2750-L2790))。
    `loss_after <= loss_pilot + rollback_tolerance` ならば承認（`accepted = True`）。悪化している場合は拒否（`accepted = False`）し、保存されていた pilot 後の状態へロールバックする ([train_tg_lora.py:L2800-2819](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2800-L2819))。

### 1.6 チェックポイント保存判定
*   **処理**:
    サイクルの最後で、等価ステップ数が $250, 500, 750, 1000, 1250, 1500$ に到達したか判定する。
    *   到達している場合、フル評価データセット `valid_full_loader`（493件）でのフル validation loss を測定し、`checkpoint-{step}` ディレクトリにチェックポイントを保存するとともに、`_check_and_save_linearity_budget_checkpoint` を呼び出して metrics の記録を行う。
    *   該当コード: [src/training/train_tg_lora.py:L576](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L576) (関数定義) および [L1924](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1924), [L3065](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L3065) (呼び出し)

---

## 2. 設計意図 (Rationale)

*   **なぜ $N$ 外挿は「1回の大きなジャンプ（着地チェック1回のみ）」なのか**:
    *   TBD（人間確定待ち）
*   **なぜ M9 フィットに有限差分（FD）を採用しているのか**:
    *   TBD（人間確定待ち）
