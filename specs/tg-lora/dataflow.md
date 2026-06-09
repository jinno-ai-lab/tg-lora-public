# TG-LoRA データフロー図

**作成日**: 2026-06-10
**関連アーキテクチャ**: [architecture.md](architecture.md)
**関連要件定義**: [requirements.md](requirements.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 既存実装・要件定義書を参考にした確実なフロー
- 🟡 **黄信号**: 既存実装から妥当な推測によるフロー
- 🔴 **赤信号**: 参照資料にない自動推定によるフロー

---

## システム全体のデータフロー 🔵

**信頼性**: 🔵 *README.md Algorithm セクション・全ソースコードより*

```mermaid
flowchart TD
    subgraph "ユーザー学習ループ"
        UL[学習ループ制御]
    end

    subgraph "TG-LoRA コンポーネント"
        RWC[RandomWalkController]
        LS[LoRAState]
        DT[DeltaTracker]
        VEL[Velocity]
        RB[RollbackManager]
        EXT[Extrapolator]
        LSAMP[LayerSampler]
        CS[CycleState]
        TA[TrajectoryAnalyzer]
    end

    subgraph "外部"
        MODEL[PyTorch Model<br/>+ LoRA Adapter]
        OPT[Optimizer]
        EVAL[Eval Function]
    end

    UL -->|1. propose| RWC
    RWC -->|Proposal| UL
    UL -->|2. snapshot| LS
    LS -->|W0 dict| UL
    UL -->|3. K steps| OPT
    OPT -->|grad update| MODEL
    UL -->|4. snapshot| LS
    LS -->|WK dict| UL
    UL -->|5. compute_and_record| DT
    DT -->|delta dict| UL
    UL -->|6. update| VEL
    VEL -->|velocity state| UL
    UL -->|7. eval| EVAL
    EVAL -->|loss_pilot| UL
    UL -->|8. save| RB
    UL -->|9. select_active_layers| LSAMP
    LSAMP -->|active_names| UL
    UL -->|10. apply_extrapolation| EXT
    EXT -->|weight update| MODEL
    UL -->|11. eval| EVAL
    EVAL -->|loss_after| UL
    UL -->|12. accept/reject| RWC
    UL -->|13. record_cycle| CS
    UL -->|14. early_stop_advice| TA
```

## 主要機能のデータフロー

### 機能1: サイクル実行（Pilot → Extrapolate → Accept/Rollback） 🔵

**信頼性**: 🔵 *README.md Algorithm・全モジュール API より*

**関連要件**: REQ-001, REQ-003, REQ-005, REQ-006

```mermaid
sequenceDiagram
    participant UL as 学習ループ
    participant RWC as RandomWalkController
    participant LS as LoRAState
    participant DT as DeltaTracker
    participant VEL as Velocity
    participant RB as RollbackManager
    participant EXT as Extrapolator
    participant MODEL as Model

    Note over UL: === サイクル開始 ===

    UL->>RWC: propose()
    RWC-->>UL: Proposal(K, N, alpha, beta, lr)

    Note over UL: === Pilot Phase ===
    UL->>LS: snapshot_lora(model)
    LS-->>UL: W0: dict[str, Tensor]
    UL->>UL: K real optimizer steps
    UL->>LS: snapshot_lora(model)
    LS-->>UL: WK: dict[str, Tensor]

    Note over UL: === Delta & Velocity ===
    UL->>DT: compute_and_record(WK, W0, K)
    Note over DT: compute_mean_delta(WK, W0, K)<br/>→ DeltaStats(total_norm, per_layer_norm, ...)
    DT-->>UL: delta: dict[str, Tensor]
    UL->>VEL: update(delta, beta)
    Note over VEL: EMA: v ← beta*v + (1-beta)*delta<br/>_record_magnitude()<br/>cosine_similarity(), magnitude_trend()
    VEL-->>UL: velocity.state: dict[str, Tensor]

    Note over UL: === Extrapolation Phase ===
    UL->>RB: save(model)
    Note over RB: snapshot_lora(model) → history[-1]
    UL->>EXT: apply_extrapolation(model, velocity, active_names, alpha, N)
    Note over EXT: 各 active パラメータ:<br/>update = N × alpha × v<br/>cap_update(update, p, relative_update_cap)<br/>p.add_(capped_update)
    EXT->>MODEL: in-place weight update

    Note over UL: === Accept/Rollback ===
    alt loss_after ≤ loss_pilot + tol
        UL->>RWC: reward(loss_pilot, loss_after)
        Note over RWC: alpha *= boost, lr *= boost<br/>K decrease prob, N increase prob
    else loss_after > loss_pilot + tol
        UL->>RB: rollback(model)
        Note over RB: load_lora_snapshot(model, history[index])
        RB->>MODEL: restore W to rollback point
        UL->>RWC: penalize(loss_pilot, loss_after)
        Note over RWC: alpha *= decay, lr *= decay<br/>K increase, N decrease prob
    end
```

**データ変換**:

1. `W0` / `WK`: `dict[str, Tensor]` — LoRA パラメータ名 → CPU Tensor のマッピング
2. `delta`: `dict[str, Tensor]` — `(WK - W0) / K` の平均差分
3. `velocity.state`: `dict[str, Tensor]` — EMA 平滑化された差分
4. `Proposal`: dataclass `(K, N, alpha, beta, lr, active_layer_strategy, relative_update_cap)`
5. `DeltaStats`: dataclass `(total_norm, per_layer_norm, max_component, mean_abs)`

### 機能2: 層選択戦略 🔵

**信頼性**: 🔵 *layer_sampler.py 実装より*

**関連要件**: REQ-009

```mermaid
flowchart TD
    A[select_active_layers] --> B{strategy?}
    B -->|last_25_percent| C[最終 25% の層インデックス]
    B -->|last_25_percent_plus_random_2| D[最終 25% + ランダム 2 層]
    B -->|middle_random| E[全層から 1/3 ランダム選択]
    B -->|lisa_like_weighted| F{layer_scores あり?}
    F -->|あり| G[softmax 重み付きサンプリング]
    F -->|なし| C

    C --> H[active_names: set of str<br/>active_indices: set of int]
    D --> H
    E --> H
    G --> H
```

**入力**: `model`, `strategy`, `random_middle=2`, `layer_scores`, `temperature`
**出力**: `tuple[set[str], set[int]]` — アクティブなパラメータ名の集合と層インデックスの集合

### 機能3: ハイパーパラメータ適応 🔵

**信頼性**: 🔵 *random_walk_controller.py 実装より*

**関連要件**: REQ-006

```mermaid
flowchart TD
    subgraph "propose()"
        P1[現在の state から出発]
        P1 --> P2[alpha: log-normal random walk]
        P1 --> P3[K: 隣接候補を確率 k_explore_prob で探索]
        P1 --> P4[N: 隣接候補を確率 n_explore_prob で探索]
        P1 --> P5[beta: 確率 beta_explore_prob で候補切替]
        P1 --> P6[strategy: 確率 strategy_explore_prob で切替]
        P1 --> P7[lr: 確率 lr_explore_prob で log-normal walk]
        P2 & P3 & P4 & P5 & P6 & P7 --> P8[Proposal 生成]
    end

    subgraph "reward()"
        R1[accept 時] --> R2[alpha *= alpha_accept_boost]
        R1 --> R3[lr *= lr_accept_boost]
        R1 --> R4[K: 確率 0.2 で減少]
        R1 --> R5[N: 確率 0.3 で増加]
    end

    subgraph "penalize()"
        PE1[reject 時] --> PE2[alpha *= alpha_reject_decay]
        PE1 --> PE3[lr *= lr_reject_decay]
        PE1 --> PE4[K: 確定的に増加]
        PE1 --> PE5[N: 確率 0.5 で減少]
        PE1 --> PE6[strategy: 確率 0.1 で変更]
    end

    subgraph "事前適応"
        A1[adapt_to_convergence] -->|trend >= 0| A2[lr decay, K increase]
        A3[adapt_to_acceleration] -->|accel > deadzone| A2
        A3 -->|accel < -deadzone| A4[lr boost]
    end
```

## データ処理パターン

### 同期処理 🔵

**信頼性**: 🔵 *全モジュール実装より*

全コンポーネントは同期的に動作する。非同期処理やバッチ処理は存在しない。ユーザーの学習ループ内で順次呼び出される設計。

- スナップショット取得・差分計算・外挿適用はすべて `@torch.no_grad()` コンテキストで実行
- velocity 更新は in-place 演算（`mul_`, `add_`）でメモリ効率を確保

### スナップショットのライフサイクル 🔵

**信頼性**: 🔵 *lora_state.py・rollback_manager.py 実装より*

```mermaid
stateDiagram-v2
    [*] --> Taken: snapshot_lora(model)
    Taken --> Diffed: diff_lora(after, before)
    Taken --> Stored: RollbackManager.save()
    Diffed --> UsedAsDelta: DeltaTracker
    Stored --> Restored: RollbackManager.rollback()
    Restored --> [*]
    Stored --> Discarded: history overflow (maxlen=100)
    Discarded --> [*]
```

- **snapshot_lora**: `iter_lora_params(model)` → `{name: p.detach().cpu().clone()}`
- **diff_lora**: `{name: after[name] - before[name]}` (オプションで scale 適用)
- **snapshot_lora_delta**: base からの差分のみ保存（メモリ効率化）
- **load_lora_snapshot**: `p.copy_(saved.to(device, dtype))` で in-place 復元

## エラーハンドリングフロー 🔵

**信頼性**: 🔵 *全モジュールの ValueError / RuntimeError 実装より*

```mermaid
flowchart TD
    A[入力] --> B{バリデーション}
    B -->|引数範囲外| C[ValueError]
    B -->|正規の入力| D[処理実行]
    D --> E{計算結果}
    E -->|NaN/Inf 検出| F[警告ログ + 安全処理]
    E -->|有限値| G[正常終了]
    F --> G

    subgraph "安全処理の例"
        F1[cap_update: ゼロ化]
        F2[_sanitize_snapshot: nan_to_num]
        F3[is_magnitude_anomalous: skip non-finite]
    end
```

- **引数検証**: 全モジュールでコンストラクタ・メソッド引数を検証し `ValueError` を送出
- **NaN/Inf 対処**: `cap_update` は非有限値をゼロ化、`_sanitize_snapshot` は `nan_to_num` で置換
- **ログ**: `logging.getLogger("tg-lora")` で警告・エラーを記録

## 状態管理フロー

### コンポーネント状態管理 🔵

**信頼性**: 🔵 *ControllerState・CycleState dataclass 実装より*

各コンポーネントは独自の状態を管理し、`summary()` / `from_dict()` でシリアライズ可能。

```mermaid
flowchart LR
    subgraph "RandomWalkController"
        CS1[ControllerState<br/>K, N, alpha, beta, lr<br/>layer_scores, counts]
    end

    subgraph "CycleState"
        CS2[cycle, optimizer_steps<br/>full_backward_passes<br/>best_loss, stale_cycles]
    end

    subgraph "Velocity"
        CS3[_state: dict<br/>_magnitude_history: deque]
    end

    subgraph "DeltaTracker"
        CS4[_history: list<br/>_norm_history: deque<br/>_last_stats: DeltaStats]
    end

    subgraph "RollbackManager"
        CS5[_history: list<br/>maxlen=100]
    end

    CS1 -->|summary() → from_dict()| PERSIST[永続化<br/>JSON/checkpoint]
    CS2 -->|summary() → from_dict()| PERSIST
```

### チェックポイント復元 🟡

**信頼性**: 🟡 *summary()/from_dict() 実装から妥当な推測*

- `ControllerState.summary()` → JSON 保存 → `ControllerState.from_dict()` で復元
- `CycleState.summary()` → JSON 保存 → `CycleState.from_dict()` で復元
- `RandomWalkController.restore_state(state)` で保存済み状態を適用
- velocity と rollback 履歴はメモリ上のみ（永続化なし）

## データ整合性の保証 🔵

**信頼性**: 🔵 *rollback_manager.py・cap_update() 実装より*

- **スナップショット整合性**: `detach().cpu().clone()` で GPU 計算グラフから分離された独立コピーを保持
- **外挿安全制約**: `relative_update_cap` で更新量を現在の重みノルムに対する比率に制限
- **ロールバック保証**: `_sanitize_snapshot` で NaN/Inf を排除した状態を履歴に保存
- **差分キー整合性**: `compute_and_record` で after/before のキー不一致を ValueError で検出

## 関連文書

- **アーキテクチャ**: [architecture.md](architecture.md)
- **分析記録**: [interview-record.md](interview-record.md)
- **要件定義**: [requirements.md](requirements.md)

## 信頼性レベルサマリー

- 🔵 青信号: 10 件 (91%)
- 🟡 黄信号: 1 件 (9%)
- 🔴 赤信号: 0 件 (0%)

**品質評価**: 高品質 — 全データフローが既存実装に基づいている
