# PURPOSE.md — TG-LoRA 目標・マイルストーン状況

> **位置づけ**: 自律開発ループ（`25_purpose_driven_executor`）が Phase 2 で読む
> 「未達 deliverable」の**ルート固定名エントリポイント**。正本は
> [docs/GOAL.md](docs/GOAL.md) §3（Forward design）・§4（Execution plan）。本ファイルは
> GOAL.md の Phase 構造をマイルストーン状況として蒸留したもので、GOAL.md と矛盾すれば
> **GOAL.md を正**とする。 charter.yml（`.concept/charter.yml`）は本 repo には存在せず、
> executor は Phase 3 をスキップして Phase 4 へ進む（GOAL の Phase 0–4 が事実上の charter）。

---

## 達成済みマイルストーン（skip 対象）

| ID | マイルストーン | 状態 | 根拠 |
|----|---------------|------|------|
| MS-R1 | 第1期〜第5期 研究路線の帰無検証と棄却（velocity 外挿 / 漸進ランク ZO / B-filter / PSA 転換） | done | GOAL §1.1–§1.5。PSA 本体は `src/tg_lora/psa.py` に実装保留。 |
| MS-PF0 | Phase 0: full backprop ベースライン確定 | done | GOAL §3.1 Phase 0。`baseline_plain` best_valid=**1.0565**、`accum16`=**1.0704** と整合確認済み。 |
| MS-PF1 | Phase 1: 単層フリーズゲート機構（Level 1）実装 | done | `src/tg_lora/progressive_freeze.py` `ProgressiveFreezeController`（single-shot `should_freeze`/`cache_xin`/`apply_freeze` + progressive `layers_due_at`/`apply_freeze_layer`/`progress`）。`tests/test_progressive_freeze*.py`。 |
| MS-PF2-CPU | Phase 2: フリーズスケジュール設計の **CPU 予測面** | done | `freeze_schedule.py`（3 policy: output_first / convergence_order / compromise）+ `freeze_cost.py`（GOAL §5 会計）+ `freeze_frontier.py`（深度→FLOPs 削減フロンティア）。全て pure-Python・GPU 不要。 |
| MS-PF2-INVIVO | Progressive Freezing in-vivo ベンチマーク（proxy） | done | `tests/test_progressive_freeze_invivo.py`（h=24 proxy）。スケールでの [UNVERIFIED] は `extrapolation_confidence` で割引。 |
| MS-HONESTY | 科学誠実性インフラ（GOAL §7 鉄則の符号化） | done | `scripts/evaluate_paper_gates.py`（G0–G4, 3-state PASS/FAIL/**INSUFFICIENT**）+ `consolidate_paper_results.py` + `scripts/check_spine_anchors.py`（159 anchor 整合性・CI gate）。TASK-0144〜0151。 |

> **設計上の制約（GOAL §1.6.3）**: 学習ループは **Level 1（基本）のみ** を prod path として駆動。
> Level 2（Activation Matching）は Phase 3 の発展実験。Level 2 を prod loop に組み込まないこと。

---

## 未達マイルストーン（deliverable 候補）

### MS-PF2: Phase 2 実スイープ（valid_loss 軸のフロンティア曲線）— **優先: P1**

GOAL §3.1 Phase 2 / §4 step 3。3 自由度（順序 / 深度 / タイミング）をスイープし、
valid_loss 劣化 vs FLOPs 削減のフロンティア曲線を描く。FLOPs 軸は MS-PF2-CPU で完了、
**valid_loss 軸は GPU run 必須（分類 C）**。

- **分類 C（外部依存・GPU）**: 実 run での valid_loss 計測、frontier sweep 実行。
- **分類 A 変換（コードで解決可能）**:
  - [x] **再現可能な random-order freeze surrogate** — GOAL §4「ランダム順フリーズ対照を超えた削減だけを有効」+「複数シード」が要求するサロゲート null の**再現可能生成器**。`freeze_schedule.py` は「shuffled permutation を渡せば surrogate になる」と文書化していたが生成器が無かった。→ **本イテレーションで `random_freeze_order(layers, seed)` を追加して閉包**。
  - [x] **frontier sweep の CLI exposition** — `freeze_frontier.frontier()` を bare CLI から起動できるようにし、Phase 2 計画を GPU run 前に再現可能に吐く。→ 本イテレーションで `scripts/run_freeze_frontier.py`（homogeneous-stack first-order cost model・table/JSON/`--output` 出力・Level-1 既定＝GOAL §1.6.3）+ `tests/test_run_freeze_frontier_cli.py`（9 tests・import health/`--help`/単調性検証）を追加して閉包。
  - [x] **サロゲート超過の判定ヘルパ** — candidate schedule の削減/性能が random-order surrogate を超えるかを 1 関数で判定（valid_loss 軸は GPU 依存だが、構造と seed 管理は CPU で固定可能）。→ `src/tg_lora/freeze_surrogate_gate.py` `surrogate_exceedance()`（seeded surrogate 分布 vs candidate・SURPASSES/TIES/UNDERSHOOTS 3 値・valid_loss 軸は構造化スレッド・honesty keystone: 均質スタックでは順序無効なので TIES）+ `tests/test_freeze_surrogate_gate.py`（11 tests）で閉包。（MS-PF2 Cat-A 3/3 時点。）
  - [x] **valid_loss 差のブートストラップ CI ヘルパ** — GOAL §4 統計の歯止め（"valid_loss 差はブートストラップ CI で評価"）が未実装。`surrogate_exceedance()` は valid_loss 軸を「構造化スレッド」として残し、有意性判定を持たなかった（少数 seed の 1 比較は逸話で非有意性証明）。→ `src/tg_lora/freeze_surrogate_ci.py`（`surrogate_valid_loss_ci()`：candidate vs surrogate の valid_loss 差 `mean(surrogate)−mean(candidate)` のパーセンタイル bootstrap CI〔10k resample・`numpy` 既存依存のみ・seed で完全再現〕・CI が 0 を上に除外→SURPASSES／下→UNDERSHOOTS／跨→TIES で構造判定を**有意性付きに昇格**・GOAL §7 鉄則の分離として significance と materiality を別軸〔`material_margin`・2.6σ-but-tiny を win と呼ばせない〕・薄サンプルは `is_thin_evidence`〔=`freeze_cost.MIN_SAMPLE_FOR_CONFIDENCE_BAND`〕・verdict 定数は `freeze_surrogate_gate` から import で共有=promotion not rename〕+ `tests/test_freeze_surrogate_ci.py`（26 tests・有意性判定／§7 分離／thin-evidence／seed 再現性／CI 幅の confidence 依存／validation）で閉包。`freeze_surrogate_gate.py`（構造）と `evaluate_paper_gates.py`（G0–G4）を結ぶ統計層。**MS-PF2 Category-A 4/4 完了**。

### MS-PF3: Phase 3 Activation Matching（Level 2 発展実験）— **優先: P2**

GOAL §3.1 Phase 3 / §4 step 4。最適スケジュール固定後、後段貫通も省けばさらに削れるか。

- **分類 C**: Level 1 vs Level 2 の定量比較 run（GPU）。
- **分類 A 変換**:
  - [x] local-loss の per-arm breakdown 観測（`progressive_freeze.local_loss_breakdown` / `activation_matching`）— 実装済み。
  - [x] 損失関数アブレーションハーネス（MSE 単独 / MSE+cos / 分布一致）を CPU で切り替え可能にし、各 arm の重み付けを config 駆動にする。→ `src/tg_lora/loss_ablation.py`（`LOSS_ARMS` 3 preset: `mse`/`mse_cos`/`dist`・MSE 常に base・各 arm は項を1つ加える factorial 設計 + `LossArmConfig` で arm 名＋重み上書きの **config 駆動** + `build_matching_loss()` 橋渡し + `run_loss_ablation()` 同一入力で全 arm を並走させる side-by-side harness・各 arm のスカラー loss と per-term breakdown を detached で返す）+ `tests/test_loss_ablation.py`（17 tests・named-arm 重み固定・config 上書き・harness の breakdown は combiner と byte-identical・distribution arm の置換不変性を MSE と対比して観測）で閉包。**MS-PF3 Category-A 2/2 完了**。

### MS-PF4: Phase 4 跨条件検証（スケジュール汎用性）— **優先: P3（副次）**

GOAL §3.1 Phase 4 / §4 step 5。最適スケジュールを LR/データ/r/シード で変えて有効半径を地図化。

- **分類 C**: 複数条件での GPU run。
- **分類 A 変換**:
  - [x] **schedule-portability の手順プリミティブ＋単体テスト** — GOAL §3.1 Phase 4 line 177 が「target xin の使い回し」ではなく「スケジュール（いつ何層固めるかという手順）の使い回し」を検証対象と明記。層セット非依存の手順 `ScheduleProcedure`（policy/depth/timing）＋ `bind()` で異なる active_layer 構成へ再適用するプリミティブが無かった。→ `src/tg_lora/freeze_schedule.py` `ScheduleProcedure.bind()`（手順は不変・convergence_order/stability_epoch は条件ごと再供給・小さいセットでは明確な ValueError で安全縮退）+ `tests/test_freeze_schedule_portability.py`（13 tests・等サイズ集合で freeze-epoch 列が一致=手順不変・output 側 suffix の汎用・`FreezeCostAccountant` の reduction_rate が等サイズ一様コスト条件で一致=指数非依存パイプライン）で閉包。**MS-PF4 Category-A 1/1 完了**。

---

## ブロッキング条件（外部依存・自動化不可 = 分類 C）

1. **GPU compute（RTX 3060 12GB 級）** — MS-PF2/PF3/PF4 の valid_loss 実測全てが依存。
   変換ルール（instruction §4）により、上記「分類 A 変換」で代替コード作業を生成し**待機しない**。
2. **公開ミラーに private `src.data` pipeline が欠損** — data-dependent test 約 130 件が
   pre-existing 失敗（[[public-mirror-preexisting-test-failures]] 既知制約・非回帰）。
3. **`make lint`（ruff）の `scripts/` 既存負債** — pre-existing（[[public-mirror-preexisting-lint-debt]]・非回帰）。
   自ファイルは `ruff check <file>` で隔離検証すること。

---

## 課題セクション（コードベース分析から抽出）

- **【高】成功条件（定量）が [UNVERIFIED]**: MS-PF2/PF3/PF4 の valid_loss 実測が GPU 依存で未実施。
  GOAL §4「成功の定義」は構造は完成しているが数値検証が未完了。→ 分類 A 変換で足場を固め、
  GPU 利用可能次第に即座に検証できる状態を維持する。
- **【中】目的駆動ループの空転（本イテレーションで解消）**: `25_purpose_driven_executor` が
  Phase 1/2 で読む `SYSTEM_CONSTITUTION.md` / `PURPOSE.md` が**ルートに存在せず**、executor が
  毎回 skip → `run-pws` が no-output で exit 4 になっていた（直近イテレーションの却下理由）。
  GOAL.md を正本として両ファイルを蒸留・配置し、ループを始動可能にした。
- **【中】`src.data` 欠損による data-dependent test の不活性**: 公開ミラー制約。本 repo の
  品質 canary は `tests/test_cli_help_smoke.py`（37 passed / 3 xfailed）で回帰監視する。

---

## 次の一手（next execution）

1. **MS-PF2/PF3/PF4 の Category-A は全て閉包**（MS-PF2 4/4・MS-PF3 2/2・MS-PF4 1/1）。
   GOAL §4 統計の歯止め（bootstrap-CI）も CPU 実装済み。未達は分類 C（GPU run での
   valid_loss 実測）のみ。GPU ブロックを理由に停止せず、CPU で足場付けできる次の
   Category-A を設定する。
2. **次候補 / 分類 A**: bootstrap-CI の G0–G4 ゲートへの接続 —
   `freeze_surrogate_ci.surrogate_valid_loss_ci()`（有意性付き valid_loss 軸判定）を
   `scripts/evaluate_paper_gates.py` が消費できるようにする。現状ゲートは proxy FLOP 会計
   （G0–G4）と構造的 surrogate-exceedance だけで、valid_loss 軸の**有意性**を問う口が無い。
   GPU run が candidate/surrogate の valid_loss 標本を置いた瞬間に G シリーズが CI 判定
   （SURPASSES/TIES/UNDERSHOOTS）で quality 軸を grade し、標本未供給なら INSUFFICIENT
   に落とす接続が、`freeze_surrogate_gate.py`（構造）→`freeze_surrogate_ci.py`（有意性）→
   `evaluate_paper_gates.py`（G0–G4 認定）の橋を完成させる。
3. 理由: 手順スケジュールの CPU 側（plan / cost / frontier / surrogate-gate /
   **surrogate-CI** / loss-ablation / schedule-portability）は完全に枯れた。残る未達は
   valid_loss の数値検証（GPU・分類 C）と、その検証結果をゲートに取り込む配線（分類 A）
   のみ。分類 A 変換ルール（instruction §4）に従い、有意性判定層をゲートへ接続して
   「GPU が来たら即判定」の状態を維持する。
