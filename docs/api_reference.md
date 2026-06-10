# TG-LoRA API Reference

`src/tg_lora/__init__.py` が公開する全APIのリファレンス。各クラス・関数のシグネチャ、パラメータ、戻り値をソースコードと一致させる。

---

## 目次

- [Velocity](#velocity)
- [apply_extrapolation](#apply_extrapolation)
- [cap_update](#cap_update)
- [DeltaTracker](#deltatracker)
- [compute_mean_delta](#compute_mean_delta)
- [CycleState](#cyclestate)
- [select_active_layers](#select_active_layers)
- [get_num_layers](#get_num_layers)
- [StrategyName](#strategyname)
- [RollbackManager](#rollbackmanager)
- [RandomWalkController](#randomwalkcontroller)
- [snapshot_lora](#snapshot_lora)
- [load_lora_snapshot](#load_lora_snapshot)
- [diff_lora](#diff_lora)
- [cosine_similarity](#cosine_similarity)
- [total_norm](#total_norm)
- [per_layer_norms](#per_layer_norms)

---

## Velocity

```python
from src.tg_lora import Velocity
```

LoRA 重み更新の速度ベクトルを指数移動平均 (EMA) で追跡する。

### Velocity コンストラクタ

```python
Velocity(max_history: int = 100)
```

| パラメータ    | 型  | デフォルト | 説明                          |
| ------------- | --- | ---------- | ----------------------------- |
| `max_history` | int | 100        | 保持する magnitude 履歴の上限 |

### Velocity プロパティ

```python
@property
state -> dict[str, torch.Tensor] | None
```

現在の速度ベクトル。未初期化時は `None`。

```python
@property
magnitudes -> list[float]
```

速度ベクトルのノルム履歴（新しい順）。

### Velocity メソッド

```python
update(delta: dict[str, torch.Tensor], beta: float) -> dict[str, torch.Tensor]
```

速度の EMA 更新: `state = beta * state + (1 - beta) * delta`。初回は `delta` をそのまま設定。

| パラメータ | 型                        | 説明                            |
| ---------- | ------------------------- | ------------------------------- |
| `delta`    | `dict[str, torch.Tensor]` | レイヤー名→差分テンソルのマップ |
| `beta`     | float                     | EMA 平滑化係数 (0-1)            |
| **戻り値** | `dict[str, torch.Tensor]` | 更新後の速度ベクトル            |

```python
reset() -> None
```

速度と magnitude 履歴をクリア。

```python
is_magnitude_anomalous(threshold_sigma: float = 3.0) -> bool
```

最新 magnitude が `mean + threshold_sigma * std` を超えた場合に `True`。履歴 3 件未満の場合は `False`。

```python
magnitude_trend(window: int = 5) -> float
```

直近 `window` 件の magnitude の線形回帰傾き。負 = 収束傾向、正 = 発散傾向。

```python
cosine_similarity(delta: dict[str, torch.Tensor]) -> float
```

速度ベクトルと `delta` のコサイン類似度 (-1〜1)。未初期化時は `0.0`。

---

## apply_extrapolation

```python
from src.tg_lora import apply_extrapolation
```

速度ベクトルから次ステップの LoRA 重みを外挿予測し、モデルに反映する。

```python
apply_extrapolation(
    model: torch.nn.Module,
    velocity: dict[str, torch.Tensor],
    active_names: set[str],
    alpha_by_name: dict[str, float],
    default_alpha: float,
    n_steps: int,
    relative_update_cap: float = 0.005,
) -> None
```

| パラメータ            | 型                        | 説明                                   |
| --------------------- | ------------------------- | -------------------------------------- |
| `model`               | `torch.nn.Module`         | LoRA 適用済みモデル                    |
| `velocity`            | `dict[str, torch.Tensor]` | 速度ベクトル                           |
| `active_names`        | `set[str]`                | 外挿対象のパラメータ名集合             |
| `alpha_by_name`       | `dict[str, float]`        | パラメータごとのステップサイズ         |
| `default_alpha`       | float                     | `alpha_by_name` にない場合のデフォルト |
| `n_steps`             | int                       | 外挿ステップ数 (N)                     |
| `relative_update_cap` | float                     | 更新の相対ノルム上限 (0.005 = 0.5%)    |

`@torch.no_grad()` で実行。更新は `cap_update()` で安全にクリップされる。

---

## cap_update

```python
from src.tg_lora import cap_update
```

更新テンソルのノルムを基準テンソルに対する相対比で制限する。

```python
cap_update(
    update: torch.Tensor,
    ref: torch.Tensor,
    max_ratio: float = 0.01,
    eps: float = 1e-8,
) -> torch.Tensor
```

| パラメータ  | 型             | 説明                                 |
| ----------- | -------------- | ------------------------------------ |
| `update`    | `torch.Tensor` | 制限対象の更新テンソル               |
| `ref`       | `torch.Tensor` | 基準テンソル（パラメータの現在地）   |
| `max_ratio` | float          | 許容する相対ノルム比 (デフォルト 1%) |
| `eps`       | float          | ゼロ除算防止                         |
| **戻り値**  | `torch.Tensor` | クリップ後の更新テンソル             |

`update` に NaN/Inf が含まれる場合はゼロテンソルを返す。

---

## DeltaTracker

```python
from src.tg_lora import DeltaTracker
```

学習サイクル間の重み差分を追跡し、統計量と異常検知を提供する。

### DeltaTracker コンストラクタ

```python
DeltaTracker(max_history: int = 100)
```

| パラメータ    | 型  | デフォルト | 説明                   |
| ------------- | --- | ---------- | ---------------------- |
| `max_history` | int | 100        | 保持する差分履歴の上限 |

### DeltaTracker プロパティ

```python
@property
last_stats -> DeltaStats | None
```

最新の差分統計。未記録時は `None`。

```python
@property
norm_history -> list[float]
```

差分ノルムの履歴。

### DeltaTracker メソッド

```python
compute_and_record(
    after: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    K: int,
) -> dict[str, torch.Tensor]
```

`(after - before) / K` を計算し履歴に記録。

```python
is_anomalous(threshold_sigma: float = 3.0) -> bool
```

最新差分ノルムが `mean + threshold_sigma * std` を超えた場合 `True`。

```python
convergence_trend(window: int = 5) -> float
```

直近 `window` 件の差分ノルムの傾き。負 = 収束。

```python
summary() -> dict
```

統計サマリーを辞書で返す（`total_norm`, `max_component`, `mean_abs`, `anomalous`, `convergence_trend`, `history_length`）。

---

## compute_mean_delta

```python
from src.tg_lora import compute_mean_delta
```

```python
compute_mean_delta(
    after: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    K: int,
) -> dict[str, torch.Tensor]
```

| パラメータ | 型                        | 説明                           |
| ---------- | ------------------------- | ------------------------------ |
| `after`    | `dict[str, torch.Tensor]` | 更新後の LoRA スナップショット |
| `before`   | `dict[str, torch.Tensor]` | 更新前の LoRA スナップショット |
| `K`        | int                       | パイロットステップ数           |
| **戻り値** | `dict[str, torch.Tensor]` | `(after - before) / K`         |

---

## CycleState

```python
from src.tg_lora import CycleState
```

TG-LoRA 学習サイクル全体の進捗とメトリクスを管理するデータクラス。

```python
@dataclass
class CycleState:
    cycle: int = 0
    full_backward_passes: int = 0
    extrapolation_steps: int = 0
    best_loss: float = float("inf")
    best_step: int = 0
    stale_cycles: int = 0
    last_train_loss: float = 0.0
    accepted_count: int = 0
    rejected_count: int = 0
```

### CycleState メソッド

```python
record_cycle(
    K: int,
    N: int,
    grad_accum: int,
    train_loss: float,
    valid_loss: float | None = None,
    accepted: bool = True,
) -> None
```

サイクル結果を記録。`full_backward_passes += K * grad_accum`、`extrapolation_steps += N`。

```python
should_stop(patience: int | None = None, min_cycles: int = 10) -> bool
```

`stale_cycles >= patience` かつ `cycle >= min_cycles` で `True`。

```python
record_full_eval(full_loss: float) -> None
```

フル評価の結果で `best_loss` / `stale_cycles` を更新。

### CycleState プロパティ

| プロパティ        | 型    | 説明                                     |
| ----------------- | ----- | ---------------------------------------- |
| `reduction_rate`  | float | `1 - full_backward_passes / total_steps` |
| `acceptance_rate` | float | `accepted_count / total_cycles`          |
| `total_cycles`    | int   | `accepted_count + rejected_count`        |

```python
summary() -> dict
```

全メトリクスの辞書を返す。

---

## select_active_layers

```python
from src.tg_lora import select_active_layers
```

```python
select_active_layers(
    model: torch.nn.Module,
    strategy: StrategyName,
    random_middle: int = 2,
    layer_scores: dict[int, float] | None = None,
    temperature: float = 1.0,
) -> tuple[set[str], set[int]]
```

| パラメータ      | 型                 | 説明                                               |
| --------------- | ------------------ | -------------------------------------------------- |
| `model`         | `torch.nn.Module`  | LoRA 適用済みモデル                                |
| `strategy`      | `StrategyName`     | 層選択戦略                                         |
| `random_middle` | int                | `last_25_percent_plus_random_2` で使う追加中間層数 |
| `layer_scores`  | `dict[int, float]` | `lisa_like_weighted` 用の層スコア                  |
| `temperature`   | float              | `lisa_like_weighted` の softmax 温度               |
| **戻り値**      | `tuple[set, set]`  | (アクティブパラメータ名, 層インデックス)           |

### 戦略一覧

| 戦略                              | 説明                             |
| --------------------------------- | -------------------------------- |
| `"last_25_percent"`               | 出力側の上位 25% の層のみ        |
| `"last_25_percent_plus_random_2"` | 上位 25% + ランダム中間層 2 つ   |
| `"middle_random"`                 | 全層から 1/3 をランダム選択      |
| `"lisa_like_weighted"`            | スコアベース重み付きサンプリング |

`select_active_layers` 自体は旧来のランダム戦略も保持しているが、提供中の paper-PoC config では `last_25_percent` と実験レベルの `force_top_layers_only` を組み合わせて deterministic に運用する。

---

## get_num_layers

```python
from src.tg_lora import get_num_layers
```

```python
get_num_layers(model: torch.nn.Module) -> int
```

LoRA パラメータを持つ層数を返す。

---

## StrategyName

```python
from src.tg_lora import StrategyName
```

```python
StrategyName = Literal[
    "last_25_percent",
    "last_25_percent_plus_random_2",
    "middle_random",
    "lisa_like_weighted",
]
```

層サンプリング戦略の型エイリアス。

---

## RollbackManager

```python
from src.tg_lora import RollbackManager
```

学習不安定時にモデルの LoRA 状態を復元する。

### RollbackManager コンストラクタ

```python
RollbackManager(max_history: int = 100)
```

### RollbackManager メソッド

```python
save(model: torch.nn.Module) -> int
```

現在の LoRA スナップショットを保存。スナップショットインデックスを返す。NaN/Inf は `_sanitize_snapshot` で置換される。

```python
rollback(model: torch.nn.Module, index: int = -1) -> None
```

指定インデックスのスナップショットに復元。デフォルトは直近の保存。履歴が空の場合 `RuntimeError`、範囲外の場合 `IndexError`。

```python
pop() -> None
```

直近のスナップショットを履歴から削除。

```python
clear() -> None
```

全スナップショットを削除。

---

## RandomWalkController

```python
from src.tg_lora import RandomWalkController
```

ハイパーパラメータ (K, N, alpha, beta, lr, strategy) をランダムウォークで適応探索する。

### RandomWalkController コンストラクタ

```python
RandomWalkController(
    K_initial: int = 3,
    K_candidates: list[int] | None = None,          # デフォルト: [2, 3, 5, 8]
    N_initial: int = 5,
    N_candidates: list[int] | None = None,          # デフォルト: [1, 3, 5, 10, 20]
    alpha_initial: float = 0.3,
    alpha_min: float = 0.03,
    alpha_max: float = 1.5,
    alpha_log_sigma: float = 0.15,
    beta_initial: float = 0.8,
    beta_candidates: list[float] | None = None,     # デフォルト: [0.5, 0.8, 0.9, 0.95]
    lr_initial: float = 5e-4,
    lr_min: float = 1e-5,
    lr_max: float = 1e-3,
    lr_accept_boost: float = 1.2,
    lr_reject_decay: float = 0.5,
    active_layer_strategy: StrategyName = "last_25_percent_plus_random_2",
    relative_update_cap: float = 0.005,
    rollback_tolerance: float = 0.005,
    enable_random_walk: bool = True,
    enable_convergence_adaptation: bool = True,
    k_explore_prob: float | None = None,            # デフォルト: 0.4
    n_explore_prob: float | None = None,            # デフォルト: 0.4
    beta_explore_prob: float | None = None,         # デフォルト: 0.15
    strategy_explore_prob: float | None = None,     # デフォルト: 0.08
) -> None
```

### RandomWalkController メソッド

```python
propose() -> Proposal
```

現在の状態から新しいハイパーパラメータを提案。`enable_random_walk=True` なら alpha は対数正規分布、K/N/beta は候補リストからの隣接ステップ、strategy は確率的切替。`False` なら現在値をそのまま返す。

```python
accept(loss_pilot: float, loss_after: float) -> bool
```

外挿結果が受け入れ可能か判定。`loss_after <= loss_pilot` または相対劣化が `rollback_tolerance` 以下なら `True`。NaN/Inf は `False`。

```python
reward(loss_pilot: float, loss_after: float) -> None
```

accept 時に呼ぶ。`enable_random_walk=True` なら alpha と lr を増加、K を減少、N を増加。`False` ならカウンタ更新のみ。

```python
penalize(loss_pilot: float, loss_after: float) -> None
```

reject 時に呼ぶ。`enable_random_walk=True` なら alpha と lr を減少、K を増加、N を減少、確率的に strategy を変更。`False` ならカウンタ更新のみ。

```python
adapt_to_convergence(convergence_trend: float) -> None
```

収束トレンド（DeltaTracker から取得）に基づいてプロアクティブに調整。`enable_random_walk=True` かつ `enable_convergence_adaptation=True` の時だけ有効で、トレンド >= 0 の場合は lr を減衰、K を増加。

```python
update_layer_scores(active_layer_indices: list[int], reward: float) -> None
```

`lisa_like_weighted` 戦略用の層スコアを更新。

```python
acceptance_rate() -> float
```

これまでの acceptance rate を返す。

```python
summary() -> dict
```

現在の全ハイパーパラメータと統計を辞書で返す。

### データクラス

```python
@dataclass
class Proposal:
    K: int
    N: int
    alpha: float
    beta: float
    lr: float
    active_layer_strategy: StrategyName
    relative_update_cap: float

@dataclass
class ControllerState:
    K: int
    N: int
    alpha: float
    beta: float
    lr: float
    active_layer_strategy: StrategyName
    relative_update_cap: float
    layer_scores: dict[int, float] = {}
    total_cycles: int = 0
    accepted_count: int = 0
    rolled_back_count: int = 0
    alpha_accept_boost: float = 1.1
    alpha_reject_decay: float = 0.5
    lr_accept_boost: float = 1.2
    lr_reject_decay: float = 0.5
```

---

## snapshot_lora

```python
from src.tg_lora import snapshot_lora
```

```python
snapshot_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]
```

モデルの全 LoRA パラメータを `{名前: detach.cpu().clone()}` でスナップショット。

---

## load_lora_snapshot

```python
from src.tg_lora import load_lora_snapshot
```

```python
load_lora_snapshot(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None
```

スナップショットから LoRA パラメータを復元。`@torch.no_grad()` で実行。

---

## diff_lora

```python
from src.tg_lora import diff_lora
```

```python
diff_lora(
    after: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]
```

2 つのスナップショットの差分を計算: `(after - before) * scale`。

---

## cosine_similarity

```python
from src.tg_lora import cosine_similarity
```

```python
cosine_similarity(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
) -> float
```

2 つの状態ベクトル間のコサイン類似度 (-1〜1)。共通キーのみ計算。

---

## total_norm

```python
from src.tg_lora import total_norm
```

```python
total_norm(state: dict[str, torch.Tensor]) -> float
```

全テンソルの連結ノルム: `sqrt(sum(norm(t)^2))`。非有限値はスキップ。

---

## per_layer_norms

```python
from src.tg_lora import per_layer_norms
```

```python
per_layer_norms(state: dict[str, torch.Tensor]) -> dict[str, float]
```

層ごとのノルム。`layers.<N>.` パターンで層を抽出し `layer_<N>` で集計。非有限値はスキップ。
