# TG-LoRA Operations Runbook

本番運用・障害対応のためのランブック。OOM / CUDA error / NaN・Inf 検出時の対応手順、ハイパーパラメータチューニングガイド、MLflow 運用手順をまとめる。

---

## 目次

- [1. 障害対応ランブック](#1-障害対応ランブック)
  - [1.1 OOM (Out of Memory)](#11-oom-out-of-memory)
  - [1.2 CUDA Error](#12-cuda-error)
  - [1.3 NaN / Inf 検出](#13-nan--inf-検出)
  - [1.4 チェックポイントからの復旧](#14-チェックポイントからの復旧)
- [2. ハイパーパラメータチューニングガイド](#2-ハイパーパラメータチューニングガイド)
  - [2.1 パラメータ概要](#21-パラメータ概要)
  - [2.2 推奨初期値と探索範囲](#22-推奨初期値と探索範囲)
  - [2.3 シナリオ別チューニング](#23-シナリオ別チューニング)
  - [2.4 学習率 (learning rate)](#24-学習率-learning-rate)
- [3. MLflow 運用手順](#3-mlflow-運用手順)
  - [3.1 サーバー設定](#31-サーバー設定)
  - [3.2 ロガー設定 (config)](#32-ロガー設定-config)
  - [3.3 ラン比較](#33-ラン比較)
  - [3.4 ダッシュボード使用法](#34-ダッシュボード使用法)

---

## 1. 障害対応ランブック

### 1.1 OOM (Out of Memory)

**症状**: `torch.cuda.OutOfMemoryError` が発生し学習が停止する。

**自動挙動**: `train_tg_lora.py` は OOM をキャッチし、自動的にフォールトチェックポイントを保存して終了する。

**手動対応手順**:

1. **ログを確認** — どのサイクル・ステップで OOM が起きたかを特定:

   ```bash
   grep -n "OutOfMemoryError" runs/<experiment>/train.log
   ```

2. **フォールトチェックポイントの確認**:

   ```bash
   ls -la runs/<experiment>/fault_checkpoint/
   ```

   保存内容: モデルアダプタ + `training_state.pt` (CycleState, ControllerState, Velocity, DeltaTracker を含む)。

3. **seq_len を短縮** — `max_seq_len` が 2048 の場合、1024 に減らす:

   ```yaml
   data:
     max_seq_len: 1024
   ```

4. **バッチサイズを調整** — `grad_accumulation` を維持しつつ実バッチを減らす余地はない（すでに `batch_size: 1`）。代わりに `grad_accumulation` を減らすことでトータルメモリ使用量を下げられる:

   ```yaml
   training:
     grad_accumulation: 4 # デフォルト: 8
   ```

5. **層サンプリング戦略を変更** — サンプリング対象を減らす:

   ```yaml
   tg_lora:
     active_layer_strategy: last_25_percent # ランダム中間層を含めない
   ```

6. **学習を再開**:

   ```bash
   # train_tg_lora.py は --resume フラグで復旧
   python -m src.training.train_tg_lora --config configs/9b_tg_lora.yaml \
       --resume runs/<experiment>/fault_checkpoint
   ```

**予防策**:

- `gradient_checkpointing: true` を維持する（30-40% メモリ削減）
- `quick_eval_examples` を小さくする（デフォルト: 32）
- 12GB GPU の場合は `max_seq_len: 1024` で運用

### 1.2 CUDA Error

**症状**: `RuntimeError` に "CUDA" を含むメッセージ（例: `CUDA error: device-side assert triggered`）。

**自動挙動**: `_is_cuda_error()` が CUDA 由来の `RuntimeError` を検出し、フォールトチェックポイントを保存。

**手動対応手順**:

1. **エラーの種類を特定**:

   ```bash
   grep -n "CUDA" runs/<experiment>/train.log | tail -20
   ```

2. **device-side assert の場合**:
   - データに異常なトークン ID が含まれていないか確認
   - `max_seq_len` がモデルの最大長を超えていないか確認
   - CUDA リセット後に再実行:

     ```bash
     python -c "import torch; torch.cuda.empty_cache()"
     ```

3. **illegal memory access の場合**:
   - GPU ドライバーと CUDA バージョンの互換性を確認
   - `docker compose run --rm tg-lora` で隔離環境で再実行

4. **再発する場合**:
   - ホストマシンを再起動
   - `nvidia-smi` で GPU ステータスを確認
   - Docker コンテナで再実行して環境差異を排除

### 1.3 NaN / Inf 検出

**症状**: loss 値が `NaN` または `Inf` になる。TG-LoRA の外挿後のパラメータに非有限値が混入する。

**自動挙動**:

- `check_lora_params_finite()`: 外挿後にすべての LoRA パラメータの有限性を検証
- `cap_update()`: 更新ベクトルに NaN/Inf が含まれる場合はゼロベクトルを返す（更新をスキップ）
- `_sanitize_snapshot()`: ロールバック用スナップショットの NaN/Inf を置換（NaN→0.0, Inf→1e6, -Inf→-1e6）
- `NumericalInstabilityError`: forward/backward 中に NaN/Inf loss を検出した場合に送出

**手動対応手順**:

1. **ログから NaN 発生箇所を特定**:

   ```bash
   grep -n "non-finite\|NaN\|Inf\|instability" runs/<experiment>/train.log
   ```

2. **外挿の alpha を下げる** — 外挿ステップサイズが大きすぎる場合:

   ```yaml
   tg_lora:
     alpha_initial: 0.1 # デフォルト: 0.3
     alpha_max: 0.5 # デフォルト: 1.5
   ```

3. **relative_update_cap を厳しくする**:

   ```yaml
   tg_lora:
     relative_update_cap: 0.001 # デフォルト: 0.005
   ```

4. **学習率を下げる**:

   ```yaml
   tg_lora:
     lr_initial: 1.0e-4 # デフォルト: 5e-4
     lr_max: 5.0e-4 # デフォルト: 1e-3
   ```

5. **N（外挿ステップ数）を減らす**:

   ```yaml
   tg_lora:
     N_initial: 1 # デフォルト: 5
     N_candidates: [1, 3] # デフォルト: [1, 3, 5, 10, 20]
   ```

### 1.4 チェックポイントからの復旧

TG-LoRA はフォールト発生時に完全な学習状態を保存する:

```text
fault_checkpoint/
├── adapter_model.safetensors   # LoRA アダプタ重み
├── adapter_config.json
├── tokenizer.json (+ related files)
└── training_state.pt           # CycleState, ControllerState, Velocity, DeltaTracker
```

**復旧コマンド**:

```bash
python -m src.training.train_tg_lora \
    --config configs/9b_tg_lora.yaml \
    --resume runs/<experiment>/fault_checkpoint
```

`--resume` を指定すると:

1. モデルアダプタを読み込み
2. `training_state.pt` から CycleState, ControllerState, Velocity, DeltaTracker を復元
3. 中断サイクルから学習を再開（`cycle_offset` が適用される）

---

## 2. ハイパーパラメータチューニングガイド

### 2.1 パラメータ概要

現行の `configs/9b_tg_lora.yaml` は、論文向け PoC を優先して deterministic な設定を使う。具体的には、出力側の上位層のみを外挿対象に固定し、ランダムウォーク探索はデフォルトで無効化している。Random Walk Controller 自体は残っており、ablation や exploratory run で再度有効化できる。

### 2.1.1 推奨 config surfaces

- `configs/9b_tg_lora.yaml`: current mainline。deterministic な paper-PoC 既定値。
- `configs/9b_tg_lora_paper_poc.yaml`: named/frozen な paper-PoC 設定。比較実験で moving target を避けたい時に使う。
- `configs/9b_tg_lora_adaptive_k5.yaml`: historical adaptive branch。random walk とランダム層戦略を有効化した branch。
- `configs/9b_tg_lora_adaptive_k5_no_conv.yaml`: adaptive branch から `enable_convergence_adaptation` だけを切り、lr/K の能動収束制御を ablate する設定。
- `scripts/run_ablation_suite.sh`: baseline / deterministic / adaptive / adaptive-no-conv を同じ eval hygiene で走らせる launcher。

| パラメータ   | 役割                 | デフォルト      | 探索範囲              |
| ------------ | -------------------- | --------------- | --------------------- |
| **K**        | パイロットステップ数 | 3               | [2, 3, 5, 8]          |
| **N**        | 外挿ステップ数       | 5               | [1, 3, 5, 10, 20]     |
| **alpha**    | 外挿ステップサイズ   | 0.3             | [0.03, 1.5]           |
| **beta**     | 速度平滑化係数 (EMA) | 0.8             | [0.5, 0.8, 0.9, 0.95] |
| **lr**       | 学習率               | 5e-4            | [1e-5, 1e-3]          |
| **strategy** | 層サンプリング戦略   | last_25_percent | 4 種                  |

### 2.2 推奨初期値と探索範囲

**標準設定（RTX 3060 12GB、paper-PoC / 3k データセット）**:

```yaml
tg_lora:
  K_initial: 3
  K_candidates: [2, 3, 5, 8]
  N_initial: 5
  N_candidates: [1, 3, 5, 10, 20]
  alpha_initial: 0.3
  alpha_min: 0.03
  alpha_max: 1.5
  beta_initial: 0.8
  beta_candidates: [0.5, 0.8, 0.9, 0.95]
  lr_initial: 0.0005
  lr_min: 1.0e-5
  lr_max: 0.001
  relative_update_cap: 0.005
  active_layer_strategy: last_25_percent
  force_top_layers_only: true
  enable_random_walk: false
  enable_convergence_adaptation: true
  k_explore_prob: 0.0
  n_explore_prob: 0.0
  beta_explore_prob: 0.0
  strategy_explore_prob: 0.0
```

**探索確率（exploratory mode のみ。paper-PoC の既定値はすべて 0）**:

```yaml
k_explore_prob: 0.4 # K の変更確率
n_explore_prob: 0.4 # N の変更確率
beta_explore_prob: 0.15 # beta の変更確率
strategy_explore_prob: 0.08 # 層戦略の変更確率
```

### 2.3 シナリオ別チューニング

**高速プロトタイプ（動作確認）**:

```yaml
tg_lora:
  K_initial: 2
  K_candidates: [2, 3]
  N_initial: 3
  N_candidates: [1, 3]
  alpha_initial: 0.2
  lr_initial: 5.0e-4
training:
  max_cycles: 50
eval:
  quick_eval_examples: 16
  full_eval_every_cycles: 5
```

**高精度（本番品質）**:

```yaml
tg_lora:
  K_initial: 5
  K_candidates: [3, 5, 8]
  N_initial: 10
  N_candidates: [5, 10, 20]
  alpha_initial: 0.15
  alpha_max: 0.8
  lr_initial: 2.0e-4
  lr_max: 5.0e-4
  relative_update_cap: 0.003
training:
  max_cycles: 500
  early_stopping_patience: 30
eval:
  quick_eval_examples: 64
  full_eval_every_cycles: 5
```

**高学習率安定性テスト**:

```yaml
tg_lora:
  lr_initial: 0.002 # デフォルトの 4 倍
  lr_max: 0.01
  alpha_initial: 0.1 # 外挿を控えめに
  relative_update_cap: 0.002
eval:
  rollback_tolerance: 0.003
```

### 2.4 学習率 (learning rate)

`enable_random_walk: true` の場合のみ、Random Walk Controller は accept/reject フィードバックで学習率を適応的に調整する。paper-PoC の既定設定では無効で、`lr_initial` がそのまま使われる。

`enable_convergence_adaptation` は、random walk を有効にしたまま `convergence_trend >= 0` の時の lr 減衰 / K 増加だけを切り離すためのフラグ。`configs/9b_tg_lora_adaptive_k5_no_conv.yaml` は、この経路が lr の単調減少や K=8 収束を作っていたかを切り分けるための ablation 用設定になっている。

- **accept 時**: `lr *= lr_accept_boost`（デフォルト: 1.2 倍、上限 `lr_max`）
- **reject 時**: `lr *= lr_reject_decay`（デフォルト: 0.5 倍、下限 `lr_min`）

| 推奨初期 lr | シナリオ                       |
| ----------- | ------------------------------ |
| 5e-4        | 標準（Dolly 3k, Capybara）     |
| 2e-4        | 高精度・長期学習               |
| 1e-3        | 高速プロトタイプ               |
| 2e-3 ~ 1e-2 | 安定性テスト（高 lr 耐性確認） |

---

## 3. MLflow 運用手順

### 3.1 サーバー設定

**ローカルファイルベース（デフォルト）**:

設定不要。MLflow は `./mlruns` ディレクトリに自動記録する。

```bash
# UI の起動
mlflow ui --port 5000
# ブラウザで http://localhost:5000 を開く
```

**リモートトラッキングサーバー**:

```bash
# サーバーの起動
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts
```

**config での指定**:

```yaml
logging:
  backend: mlflow
  mlflow:
    enabled: true
    tracking_uri: "http://<host>:5000"
    experiment_name: "tg-lora-experiments"
```

`tracking_uri: ""`（空文字）の場合、ローカルファイルストアが使用される。

### 3.2 ロガー設定 (config)

`MLflowLogger` は `src/utils/mlflow_logger.py` に定義され、以下の機能を持つ:

- **graceful degradation**: mlflow がインストールされていない、または `enabled: false` の場合、すべてのメソッドが no-op になる（学習に影響なし）
- **自動リトライ**: 一時的なネットワークエラー（`ConnectionError`, `TimeoutError`, `OSError`）に対して指数バックオフで最大 3 回リトライ
- **自動メタデータ**: 実行開始時に K, N, alpha, beta, lr をタグとして自動記録
- **コンテキストマネージャ**: `with MLflowLogger(...) as mlflow:` 形式で使用し、終了時に run を自動クローズ

```python
from src.utils.mlflow_logger import MLflowLogger

with MLflowLogger(
    enabled=True,
    tracking_uri="http://localhost:5000",
    experiment_name="tg-lora-experiments",
    run_name="my-run",
    config={"K": 3, "N": 5, "alpha": 0.3, "lr": 5e-4},
) as mlf:
    mlf.log_params({"model": "Qwen3.5-9B", "lora_r": 16})
    mlf.log_metrics({"train_loss": 1.23, "valid_loss": 1.45}, step=10)
    mlf.log_artifact("runs/my-experiment/report.md")
    mlf.set_tag("status", "completed")
```

### 3.3 ラン比較

**MLflow UI での比較**:

1. `mlflow ui` を起動
2. Experiment を選択
3. 比較したい run のチェックボックスをオン
4. 「Compare」ボタンをクリック
5. パラメータ・メトリクスの差分を表形式で確認

**CLI での比較**:

```bash
# experiment 内の run 一覧
mlflow runs list --experiment-id 0

# 特定 run のメトリクス取得
mlflow runs describe --run-id <run_id>
```

**スクリプトでの比較**:

```bash
python scripts/compare_runs.py \
    --baseline runs/qlora_9b_baseline/run_metrics.jsonl \
    --tg-lora runs/tg_lora_9b_mvp/run_metrics.jsonl \
    --output-dir reports

# マルチランダッシュボード
python scripts/compare_runs.py dashboard runs/
```

**スイープ結果の集計**:

```bash
python scripts/summarize_sweep.py --sweep-dir runs/sweep_*/
```

### 3.4 ダッシュボード使用法

**MLflow UI メイン画面**:

| タブ        | 用途                                                |
| ----------- | --------------------------------------------------- |
| Experiments | 実験一覧・フィルタリング                            |
| Runs        | 個別 run のパラメータ・メトリクス・アーティファクト |
| Metrics     | メトリクスのグラフ表示・複数 run の重ね合わせ       |
| Artifacts   | チェックポイント・レポートのダウンロード            |
| Tags        | ハイパーパラメータタグによる検索                    |

**活用ポイント**:

1. **タグフィルタ**: `hp.K = 3` や `hp.alpha > 0.2` で run を絞り込み
2. **メトリクス比較**: train_loss と valid_loss のカーブを重ねて過学習を確認
3. **アーティファクト**: サイクルごとのレポートをダウンロードして外挿品質を確認
4. **自動記録タグ**: `mlflow.note` に experiment 名と主要ハイパーパラメータが自動設定される

**記録されるメトリクス一覧**:

| メトリクス            | 説明                                 |
| --------------------- | ------------------------------------ |
| `train_loss`          | サイクル平均学習 loss                |
| `valid_loss`          | 検証 loss（quick/full eval 別）      |
| `grad_norm`           | 勾配ノルム                           |
| `velocity_magnitude`  | 速度ベクトルの大きさ                 |
| `acceptance_rate`     | 外挿 acceptance rate                 |
| `extrapolation_error` | 外挿予測と実際の差                   |
| `reduction_rate`      | compute 削減率（1 - backward/total） |
