# Linearity Guard (Baseline-like Fallback) ロジック設計書

## 1. 事実 (Facts)

### 1.1 目的と現状のステータス
- **目的**: 局所的な線形性が崩れているシグナルをリアルタイムで検知し、外挿を完全にスキップして通常の QLoRA と同一の挙動（$N=0$、実質的な Forward/Backward による手堅い更新）へとフォールバックさせるための安全装置。
- **現状の実装と設定**:
  - 本番の実験設定ファイル `configs/9b_tg_lora_m9.yaml` ([9b_tg_lora_m9.yaml:L109-114](file:///home/jinno/tg-lora/configs/9b_tg_lora_m9.yaml#L109-114)) では、`linearity_guard_enabled` が `false` に設定されており、本番実行において安全装置は無効化されている。また、`linearity_guard_min_acceptance_rate` も `0.0` に設定されている。したがって、現状の本番コード上ではアブレーション（比較検証用）の機能となっている。

### 1.2 システムパラメータ（Config）
`configs/9b_tg_lora_m9.yaml` ([9b_tg_lora_m9.yaml:L109-114](file:///home/jinno/tg-lora/configs/9b_tg_lora_m9.yaml#L109-114)):
```yaml
  # Linearity Guard config
  linearity_guard_enabled: false
  linearity_guard_min_acceptance_rate: 0.0
  linearity_guard_warmup_cycles: 5
  linearity_guard_max_positive_acceleration: 0.02
  linearity_guard_pilot_margin: 0.01
```

### 1.3 ガード判定の論理設計
判定関数である `_should_fallback_to_baseline_like` は `src/training/train_tg_lora.py` ([train_tg_lora.py:L529-573](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L529-573)) に実装されている。

- **事前回避条件 (Warmup & Minimal Checks)** ([train_tg_lora.py:L551-556](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L551-556)):
  - 要求されている外挿ステップ数がゼロ以下（`proposal_N <= 0`）の場合は、フォールバックを発動させずに `True` を返す（理由: `"no_extrapolation_requested"`）。
  - ガードが無効化されている（`enabled == False`）場合は、判定を行わずに `False` を返す（理由: `"disabled"`）。
  - 現在の累積サイクル数が `warmup_cycles` 未満の場合、判定を行わずに `False` を返す（理由: `"warmup"`）。

- **パイロットステップ安定性 (`stable_pilot`) の定義** ([train_tg_lora.py:L558-560](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L558-560)):
  局所的な更新である pilot step 自体が前回の検証ロス（`previous_valid_loss`）を大きく悪化させていないかを評価する。
  ```python
  stable_pilot = not math.isfinite(
      previous_valid_loss
  ) or pilot_loss <= previous_valid_loss * (1.0 + pilot_margin)
  ```

- **フォールバック発動トリガー** ([train_tg_lora.py:L562-572](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L562-572)):
  - **Trigger A: 速度アノマリーの検知**
    - 最新サイクルの速度ベクトルの大きさ（ノルム）が、過去の EMA 履歴から統計的に外れている（スパイクしている）場合。
    - `velocity_anomalous` (実引数には `velocity.is_magnitude_anomalous()` の評価値が渡される) が `True` のとき。
    - 返り値: `True, "velocity_anomaly"`
  - **Trigger B: 異常な正の加速度 ＋ パイロット不安定**
    - 勾配速度の加速が限界値を超えて急加速しており、かつ pilot step 自体が不安定な場合。
    - `math.isfinite(acceleration) and acceleration > positive_acceleration_limit and not stable_pilot` のとき。
    - 返り値: `True, f"positive_acceleration:{acceleration:.6f}"`
  - **Trigger C: 外挿承認率の低下 ＋ パイロット不安定**
    - 過去の外挿の成功率が低く、かつ pilot step 自体も不安定である場合。
    - `acceptance_rate < min_acceptance_rate and not stable_pilot` のとき。
    - 返り値: `True, "low_acceptance_and_unstable_pilot"`

### 1.4 フォールバック発動時の挙動
判定が `True` の場合、`src/training/train_tg_lora.py` ([train_tg_lora.py:L1836-1940](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1836-L1940)) において以下の処理が実行される。

1. **外挿試行（M9 Fit & Extrapolation）の完全なスキップ**:
   - 有限差分による部分空間フィットや外挿先の予測ロスの評価を一切行わない。これにより、無駄な Forward パスを回避する。
2. **通常LoRAステップの確定**:
   - そのサイクルで実行された pilot step の重み更新のみをモデルに確定する。
3. **統計情報の記録**:
   - ログおよびメトリクスに `tg_lora_N = 0`、`accepted = None`、`speculative_optimizer_steps = 0`、フラグを `B` として記録する。
4. **段階的制御の有無**:
   - 現状の実装には、「1サイクル目N=0→以降N=1→回復で通常N」のような段階的制御（ステップ関数）は存在しない。発動したサイクルについてその場で外挿をスキップ（$N=0$ 固定）する挙動のみが実装されている。

---

## 2. 設計意図 (Rationale)

- **なぜ本番で Linearity Guard が無効化（`enabled: false`）されているのか**:
  - TBD（要・人間の確定）
- **なぜ Trigger C の閾値（`min_acceptance_rate`）が 0.0（無効化）なのか**:
  - TBD（要・人間の確定）
- **なぜ段階的制御（ステップ関数）を導入せず、即時スキップのみの挙動としているのか**:
  - TBD（要・人間の確定）
