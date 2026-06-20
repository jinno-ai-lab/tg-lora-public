# 10_guard_experiment.md — TG-LoRA Guard: 選択的学習による高速フィット検証

## 1. 主張

学習軌道を r_A で監視し、収束した層を出力側から連続して凍結し、上流から可逆に解放することで、全層学習(baseline)と同等の下流性能に、より短い壁時計(総GPU秒)で到達する。質は到達条件として固定し、競うのは速度。

## 2. 制御信号 r_A

各サイクル t、各 series s について:
- A(s,t) = LoRA ΔW(=B·A) の Frobenius ノルム (trace trick: O(r³))
- dA/dt = 直前サイクルとの一次差分
- r_A(s,t) = |dA/dt| / (A + ε), ε = 全series A 中央値の 0.01 倍
- A 立ち上がり前(初期 W サイクル)は判定対象外

層 L の r_A = 当該層の全 series の領域平均。直近 W=5 サイクルの移動領域平均を r_A_window(L) とする。

### 計算方法

```
||B·A||_F² = trace((B^T·B) @ (A·A^T))    # ともに (r×r)、O(r³)
```

r=16 なので、r³=4096 FLOP/series × 8層×16series ≈ 0.5M FLOP。1サイクルあたりミリ秒以下。

## 2.5 必須前提：prefix/suffix 分割 + prefix cache（速度PASS成立条件 — 絶対に忘れないこと）

凍結で「backward/forward 計算を削減」できるのは、**loss が prefix/suffix 境界（`split_layer_idx`）で計算され、凍結 prefix の forward 出力がキャッシュされて N 回の eval（外挿線探索）で再利用される**場合に限る。これは **TG-LoRA の基本ロジック（pilot loss at split boundary + prefix cache）** であり、Guard 実験はこれを前提とする。

PyTorch 経路での三位一体:
- `src/tg_lora/activation_cache.py` — `ActivationCache.eval_and_cache`（prefix forward のキャッシュと再利用）
- `src/tg_lora/extrapolator.py` — `split_layer_idx` / `output_split_layer_idx`（pilot loss を境界で計算）
- `src/tg_lora/prefix_runtime_offload.py` — `split_layer_idx` での prefix オフロード

plain LoRA（loss がモデル最終出力）で `freeze` だけ移植しても、凍結層の `W×x` forward matmul は output `y` を作るために不可避、`∂L/∂x` の backward matmul は上流 LoRA への勾配伝播のために不可避。凍結で消えるのは `∂L/∂W`（重み勾配）の VJP だけで、削減は数%が上限（ノイズ内）。**速度 PASS は原理的に検証不可能**。

**Guard 実験を他経路へ移植する際は `dynamic_freeze` だけでなく `activation_cache` + `split_layer` 機構も必ず移植すること。片方だけ移植した freeze 実験は構造的に無効。**（2026-06 MLX 移植で `dynamic_freeze` だけ移植して `activation_cache` を脱落させた教訓）

## 3. 固定ロジック (出力側連続)

§3: 各サイクル終了時、r_A_window(L) < τ(=0.015) を「静か」とする。

- 出力側 L31 から入力側 L24 へ走査
- 静かな層を連続して固定塊に入れる
- 最初の騒がしい層で打ち切る (orphan は作らない)
- 固定塊 = {L31, ..., L_k} の連続範囲
- requires_grad=False → backward をこの塊の境界で打ち切る

**出力側連続の理由**: backward 打ち切りが連続塊でのみ FLOP/壁時計削減になるため。

## 4. 解放ロジック (上流側逐次・可逆)

固定層は ΔW が更新されず r_A を計算できないため、解放トリガは固定層自身の r_A では引かない。

上流端 (L_k 側) から 1 層ずつ解放:
- **(a) 強制攪拌**: 固定塊が R=10 サイクル維持されたら上流端 1 層を解放
- **(b) 上流活性**: 固定塊直上流の層の r_A_window が τ×1.5(=0.0225) を超えたら上流端 1 層を解放

出力側 (L31 側) は最後まで固定を保ち連続塊を守る。

## 5. 二段停止プロトコル

### 5.1 条件 A (baseline先行)
- 固定なし・全層学習
- 保険の valid 早期停止 (§5.3) 込み
- 終了後: L* = baseline 最良 valid_full loss, G* = L* 時点の gold_test 性能

### 5.2 条件 B (Guard)
- §3-4 の動的固定/解放を有効化
- 一次トリガ: valid_full loss ≤ L*+0.02 で gold_test 発火
- 最終判定: gold ≥ G* なら停止 (=baseline 同等の下流性能に到達)
- 未達なら学習継続、次の gold 発火は 5 サイクルのクールダウン後

### 5.3 保険の valid 早期停止 (両条件共通)
- valid_full loss が 10 サイクル連続で改善幅 < 0.01 なら打ち切り

## 6. 壁時計の正味削減実測

各サイクルで forward / backward / optimizer / 制御の時間を分離計測:
- 固定塊サイズと backward 時間の負相関を確認
- gradient_checkpointing 併用時の固定層 forward 再計算スキップ確認
- 制御オーバーヘッド (requires_grad 切替・param group 再構成・r_A 計算) のミリ秒計測

### 6.1 外挿信頼度上限（proxy → scale）— proxy 数の過信を防ぐ境界

`FreezeCostAccountant.reduction_rate` はモデル非依存の**厳密な算術**であり、幅 h によらない（凍結 suffix の割合そのもの）。しかし算術値を実スケールの**実現恩恵**（wall-clock・固定オーバーヘッド・スケールでの品質）へ翻訳する検証は、proxy モデル（h ≤ `PROXY_VALIDATED_MAX_WIDTH` = **2048**。in-vivo 検証 `tests/test_progressive_freeze_invivo.py` は h=24）でしか行われていない（cf. `10_progressive_freezing.md` §7 `[UNVERIFIED]`）。

したがって §7 第一関門は、9B（h=4096）のような検証範囲外の幅に対して **proxy の削減率をそのまま信じてはならない**。`freeze_cost.py` の `extrapolation_confidence` / `gate_reduction` がこの境界を与える:

```
confidence = min(1, validated_max_width / target_width)
effective_reduction = proxy_reduction × confidence
requires_scale_measurement = (confidence < floor)   # 既定 floor = 0.5（= 2× 幅）
```

- **検証範囲内（h ≤ 2048）**: `confidence = 1.0`。proxy をそのまま信用し、第一関門は通常通り判定。
- **9B（h=4096, 2× 幅）**: `confidence = 0.5`。proxy +30% → effective **+15%** に割り引かれ、閾値 10% を超えれば**条件付き PASS**（生の proxy 数を黙って信じたのではない）。proxy が +15% なら effective +7.5% < 10% で**正しく FAIL** し、9B 実測を要求する。
- **4× 幅以上（h ≥ 8192）**: `confidence < 0.5` → `requires_scale_measurement = True`。proxy のみでは**関門を通さず**、実 CUDA/scale 計測を必須とする。

これは steering フィードバックの二者択一（実 CUDA/9B 計測を行う **or** proxy 数を過信しない境界を置く）の後者。CUDA が利用できない反復では、境界によって「proxy 数が関門を素通りする」ことを構造的に防ぐ。実 9B 計測は依然 next iteration の推奨であり、それが得られれば `validated_max_width` を引き上げて `confidence` を回復させる。

### 6.2 実現性補正（Level-1 過大評価排除）— 算術削減率を過信しない第二の境界

§6.1 が「幅（proxy → scale）」の境界なら、本節は直交する「**実現性（算術 → in-vivo）**」の境界である。`freeze_cost.py` の `realizable_reduction` は、`FreezeCostAccountant.reduction_rate` の**算術値**が in-vivo で**実際に実現する**削減に一致するかを補正する。

in-vivo 検証（`tests/test_progressive_freeze_invivo.py`、CPU-proxy h=24, L=8）が empirically に示したこと:

- **Level 1（freeze-only, `requires_grad=False`）**: 重み勾配 FLOP は会計上「削減」と計上されるが、活性勾配は凍結層を貫通して伝播するため、実際の backward 通過数は 1 件も減らない。**実現削減率 ≈ 0**（`test_accountant_level1_overstates_realizable_savings_in_vivo`）。
- **Level 2（trio: `activation_cache` + `split_layer` + `dynamic_freeze`）**: 境界 local loss が凍結 suffix への逆伝播を物理的に断つため、**実現削減率 == 算術予測に完全一致**（`test_in_vivo_level2_matches_accountant_prediction_exactly`）。

したがって §7 第一関門は Level-1 の削減率を**信用してはならない**。`gate_reduction(level=1)` は算術値を保持（`proxy_reduction`、透明性のため）しつつ、`realized_reduction` を `LEVEL1_REALIZED_REDUCTION_CEILING = 0.0` に据え、`effective_reduction` を 0 にする。結果として Level-1 はいかなる幅でも第一関門（10% 短縮）を通らない。これは CPU-proxy in-vivo 証拠に基づく境界であり、実 9B 計測ではない（§6.1 の幅境界と同じ honesty）。将来の in-vivo 計測で Level-1 に非零の実現削減が観測されれば（例: gradient checkpointing 併用下）、この ceiling を引き上げて回復させる。

**回復の着地点（§6.2 証拠配管）**: 上記の「ceiling を引き上げて回復させる」を、定数の手書き修正ではなく監査可能な証拠経路で実現する着地点が `freeze_cost.py` に存在する。`Level1RealizationRecord(observed_reduction, num_runs, source)` が計測1件の証拠を運び、`resolve_level1_ceiling(record)` が関門が信用する ceiling 値を解決する——record が薄い証拠（`MIN_SAMPLE_FOR_CONFIDENCE_BAND` 未満 = §6.3 と同値の3件未満）なら検証済み `0.0` を維持し、bar を超えれば観測値を返す。解決された ceiling は `realizable_reduction(level1_ceiling=)` → `gate_reduction` → `speed_gate_verdict` → `compare_freeze_levels(level1_record=)` へ直列に伝播し、実現削減率は `min(ceiling, proxy)` で**算術上限を超えて信用しない**。非薄の record が供給されれば Level-1 判定は `FAIL` から `PASS`/`PROVISIONAL_PASS` へ回復し得る。2026-06 時点では**実 9B 計測は未供給**で既定は `0.0` のままであり、関門の判定は一切変化しない——これは gate の厳格化ではなく、実測値が得られた時にそれを反映するための pluggable な着地点である。

### 6.3 分散較正バンド（観測分散 → band 幅）— 薄い証拠で band を名乗らない第三の境界

§6.1（幅: proxy → scale）と §6.2（実現性: 算術 → in-vivo）は関門が信用する**点推定** `effective_reduction` を段階的に補正する。本節は直交する第三の次元「**観測分散**」を扱う: 関門が提示する削減率を、その**測定されたばらつき**とともに記録し、band 幅を**測定分散**から較正する。 steering フィードバックが指摘した失敗モード — "中央値の2回再現は confidence band と呼ぶには薄い証拠" — を、判断ではなく**強制可能な監査可能ルール**として退場させる。

`freeze_cost.py` の 3 つ組がこれを実装する:

- **`ReductionSample`** — 観測された実現削減率（サイクル毎・ラン毎・測定条件毎）を生の値のまま保持し、`n` / `min` / `max` / `mean` / `stddev`（標本標準偏差, ddof=1）を算出する。点推定と違い**測定されたばらつき**を記録する。
- **`calibrate_reduction_band(sample, *, method, z)`** — band 幅を標本の**測定分散**から較正する:
  - `method="empirical_envelope"`（既定）: band = `[min, max]`。観測された全範囲・ノンパラメトリック。幅は観測 spread そのもので、推測を含まない。
  - `method="normal"`: band = `mean ± z·stddev`（`z=1.96` ≈ 95% 正規区間）。観測を一つの削減率の雑音付き測定とみなす。低平均・高分散の標本ではゼロを下回りうる（削減率は非負のため、その場合は empirical_envelope を推奨）。`format_reduction_band()` は下限がゼロを下回ったとき、負の削減率が達成可能と読まれないよう監査行にその旨を明示する（既定の empirical_envelope は `lower=min(observations)≥0` でこの注記は発火せず、パイプライン出力は byte-identical）。
- **`MIN_SAMPLE_FOR_CONFIDENCE_BAND = 3`** — 標本がこれ未満のとき band は `is_thin_evidence=True` になる。統計量は監査のため計算されるが、関門はこれを「較正済み confidence band」として提示してはならない（1-2 点の再現は band の名に値しない）。

`ConfidenceBand` は `effective_reduction` 周りの**不確実性報告**であり、§7 の判定を反転させない（関門の honesty は既に §6.1 + §6.2 が担う）。実観測から band に供給する系列は `per_cycle_realized_reductions(accountant, level)` が生成する: サイクル `t` 毎に「`t` までに凍結した層だけ」でスケジュールを切り詰め、§6.2 の実現削減率を報告する。これは実現削減率がラン全体で**どう推移したか**を 1 本のヘッドライン数に潰さずに記録する。複数ランにわたる spread は、系列を結合してから標本を組めば同じ枠で扱える。

**パイプライン統合**: `scripts/analyze_dynfreeze_experiment.py` の `_proxy_speed_gate_section` は、§7 proxy 判定（`speed_gate_verdict`）の直後に、実現削減率を信用する Level-2 のときだけこの band を出力する。観測サイクル数が少ないランは `THIN_EVIDENCE` と明記され、confidence band の体裁をとらない。Level-1 / 凍結なしが信用するものはゼロなので、そこでは band を出力しない。

これは steering フィードバックの「実 CUDA/9B 計測を行う **or** 分散に対して band を較正する」の後者。CUDA が利用できない反復では、band が少なくとも**測定された spread に対して正直**になる。実 9B 計測が得られれば、その per-cycle / per-run 観測を同じ `ReductionSample` に蓄積するだけで band が真の測定分散に較正される。

**比較ヘッドラインの再生産 bracket 着地点 (§6.3 → Phase 3)**: 上記の band は一つの level の実現削減率の per-cycle spread を較正するが、Phase 3 の A/B ヘッドライン `additional_realized_reduction`（Level 2 の suffix 切断が Level 1 基盤の上で追加実現する後方削減）は、`compare_freeze_levels()` では proxy 算術からの**点推定**としてしか報告されず、再生産をまたぐ測定分散を受け取る着地点がなかった。`ReproductionRecord`（N 件の実測ヘッドライン削減 + `source`）と `calibrate_reproduction_bracket()` がその着地点である: N 件の実 A/B 再生産観測を `ReductionSample` / `calibrate_reduction_band` と同じ枠で §6.3 `ConfidenceBand` に較正し、`LevelComparison.reproduction_bracket` に載せ、`format_level_comparison()` が監査行として出力する。これは §7 判定を反転させない**不確実性報告**であり（per-level band と同じ honesty 役割をヘッドラインについて担う）、関門の厳格化ではない — steering が「既に厳格な gate をこれ以上硬化するな」と明示した点にも合致する。既定（record なし）は `None` で出力は byte-identical、薄い証拠（`MIN_SAMPLE_FOR_CONFIDENCE_BAND` 未満）は `THIN_EVIDENCE` + 件数を明示し confidence 区間の体裁をとらない。これが steering bullet 2 の「thin (N=2) bracket を厚くする」着地点: 実 CUDA A/B run が N 件の観測を供給すれば、点だったヘッドラインは正直で再生産件数付きの bracket になる。実測値の供給は GPU 依存の別作業であり、本着地点はそれを受け取る pluggable な口のみを提供する（実 9B 計測を一切偽造しない）。

## 7. 判定基準

- **第一関門 (速度) PASS**: B の総GPU秒 ≤ A × 0.90 (10%以上短縮) + 正味削減確認
  - 削減量を proxy（h ≤ 2048）の会計から判定する場合、§6.1 の外挿信頼度で割り引いた `effective_reduction` で判定すること。検証範囲を外れる幅では proxy 数をそのまま信じない。
  - **実現性**: `effective_reduction` は §6.2 の実現性補正を経た値を使うこと。Level-1（freeze-only）の算術削減率は in-vivo で実現しないため、関門はこれを信用せず Level-2（trio）のみを信用する。
  - **判定の実装**: 上記2補正を経た proxy 判定は `freeze_cost.speed_gate_verdict()`（閾値 `SPEED_GATE_THRESHOLD = 0.10`）がそのまま出力する。生の `passes()` 真偽値では表現できない段階的判定 (`PASS` / `PROVISIONAL_PASS` / `REQUIRES_SCALE_MEASUREMENT` / `FAIL`) を返し、proxy 数が関門を黙り越しで通過することを構造的に防ぐ。Level-1 はいかなる幅でも `FAIL`、9B(h=4096) は条件付き `PROVISIONAL_PASS`、4×幅以上は実計測を要求する。
  - **不確実性の報告**: 判定を反転させない第三の honesty 次元として、信用した実現削減率の**測定された per-cycle spread** を §6.3 の分散較正バンド（`freeze_cost.calibrate_reduction_band`）で併記する。観測が薄い（`MIN_SAMPLE_FOR_CONFIDENCE_BAND` 未満）場合は `THIN_EVIDENCE` と明記し、confidence band の体裁をとらない。これは PASS/FAIL を変えるものではなく、ヘッドライン数をその分散抜きで提示しないための報告である。
  - **Level-1 vs Level-2 定量比較 (GOAL §5 / Phase 3)**: 二段階の実装レベルを横断する GOAL §5 の効率会計・Phase 3 の定量比較を `freeze_cost.compare_freeze_levels()` が一つの対象幅で出力する。Level 1（progressive freeze・weight-grad 停止・確実な基盤）と Level 2（activation matching の suffix 切断・発展実験）それぞれの §7 verdict を**同一の閾値/幅/床**で構築し、suffix 切断が Level 1 基盤の上に**追加で**得る算術 / 実現 / effective 削減（`additional_*_reduction`）を明示する。Level 2 は Level 1 が飛ばす作業の上位集合を飛ばすため、各 delta は構造上非負。§6.2 の ceiling が Level 1 の in-vivo 実現を 0 に据えるため、追加実現削減は Level 2 の実現削減そのものになる（suffix 切断だけが実現する後方削減を担う）。`additional_passes` は Level 1 が `FAIL` の幅で Level 2 だけが関門を通る（`PASS` / `PROVISIONAL_PASS`）場合に真であり、これが Phase 3 の activation matching 実験が追加削減を proxy-loss 品質リスク（GOAL §1.6.5）と天秤にかけるための数量である。`format_level_comparison()` が判定ファイル向けの監査ブロックを吐く。
  - **パイプライン出力の保証**: 上記の段階的判定と band を実際に `gate_decision.txt` へ吐く `scripts/analyze_dynfreeze_experiment.py::_proxy_speed_gate_section`（CUDA-less パス）は、単体テスト `tests/test_freeze_cost.py` が押さえる算術の*外*に位置する配線 — guard ブロックログ `block_layers` の構文解析・global→local 層 idx 再マップ・`num_cycles` 導出・「信用する削減があるときだけ band を出す」§6.3 の出力可否 — を担う。ここは単体テストが拾えないsilent な破損点（再マップ落ち・cycle 数誤り・Level-1 で band を誤出力）が入りうる唯一の接続部なので、`tests/test_analyze_dynfreeze_gate.py` が統合テストで固定する: 各幅/level でパイプライン内部と*同じ* `freeze_cost` 公開 API で会計士と verdict を再構築し、吐出ブロックが正準 formatter 出力（`format_speed_gate_verdict` / `format_reduction_band`）に**行単位で一致**すること、Level-2 では band が出て Level-1 では出ないこと、観測なし/範囲外は SKIP することを検証する。これにより算術の honesty が判定ファイルまで途切れず届く。
- **第二関門 (質)**: gold ≥ G* (§5.2 停止規則に統合済み)
- 両関門 PASS で主張成立

## 8. パラメータ一覧 (確定値)

| パラメータ | 値 | 備考 |
|---|---|---|
| τ | 0.015 | 固定閾値 |
| W | 5 | 移動窓幅 |
| R | 10 | 強制攪拌間隔 |
| 上流活性倍率 | 1.5 | τ×1.5=0.0225 |
| ε | A中央値×0.01 |適応 epsilon |
| 一次トリガ余裕 | L*+0.02 | gold 発火閾値 |
| gold クールダウン | 5 cycles | 再評価間隔 |
| 保険 patience | 10 cycles (Δ<0.01) | 早期停止 |
| 第一関門ライン | 総GPU秒 10% 短縮 | |

## 9. 実装マッピング

| 機能 | ファイル | クラス/関数 |
|---|---|---|
| コントローラ | `src/tg_lora/dynamic_freeze.py` | `DynamicFreezeController` |
| 設定 | `src/training/config_schema.py` | `TGLoRAParams.dynfreeze_*` |
| 統合 | `src/training/train_tg_lora.py` | 初期化・判定・スキップ・メトリクス |
| チェックポイント | `src/utils/checkpoint.py` | `TrainingState.dynfreeze_state` |
| 設定 (有効) | `configs/9b_tg_lora_m10_dynfreeze.yaml` | experiment: tg_lora_9b_m10_guard |
| 設定 (baseline) | `configs/9b_tg_lora_m10_dynfreeze_baseline.yaml` | dynfreeze_enabled: false |
| 分析 | `scripts/analyze_dynfreeze_experiment.py` | グラフ・判定出力 |

## 10. 事前分析結果 (M9 110チェックポイント)

- r_A は唯一の識別軸 (cos, r_S は無関係)
- 全 8 層 (L24-L31) が同時動 (深度勾配なし)
- r_A 平均の時系列が固定/攪拌フェーズを分離
- p90/mean 比 = 1.6 (安定、平均だけで代表可能)
- descent (cycles 1-15), settling (15-50), plateau (50+)
