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
  - [x] **valid_loss 軸の significance を問う GPU 実行導線（proxy-scale）** —
    `surrogate_valid_loss_ci()` は構築済みだったが**実 run の valid_loss を一度も消費していなかった**（テストは構成定数のみ）。→
    `scripts/run_freeze_validloss_ci.py` + `make freeze-validloss-ci`（学習可能 proxy で candidate=`output_first` vs surrogate=`random_freeze_order` を**実 GPU 学習**し、得た real valid_loss 標本を `surrogate_valid_loss_ci()` に流す最初の導線・device auto=CUDA・seed 完全再現・`proxy_scale=True` 誠実ラベル）+ `tests/test_run_freeze_validloss_ci.py`（11 tests・実学習確認/seed 再現/CI 境界との自己一貫性/誠実ラベル）。
    **RTX 3060 実測（2026-06-25）**: device=cuda・candidate_mean=**0.4254** vs surrogate_mean=**0.4349**（共に uniform≈3.47 から真に学習）・改善点=0.0096・bootstrap CI[95%]=[−0.096, +0.109]・判定 **TIES**（n=5/5・non-thin）。
    科学意味: homogeneous proxy stack では順序は有効でない、という構造キーストン（`freeze_surrogate_gate`）が**実 GPU valid_loss で初めて経験的に確認**された。target-scale 判定は同一関数に 9B の標本を流すだけで昇格（`proxy_scale` フラグがそれを示す）。
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
   **本イテレーションで Category-C を具体コマンドに縮約**: `make freeze-validloss-ci` が
   実 GPU で valid_loss 標本を生成し `surrogate_valid_loss_ci()` に流す（proxy-scale TIES 判定を
   RTX 3060 で実測済み・上記 MS-PF2 参照）。target-scale 判定は同一導線に 9B 標本を流すのみで、
   残る外部依存は下記 #2 の private `src.data` pipeline のみ（足場コードの追加ではない）。
2. **公開ミラーに private `src.data` pipeline が欠損** — data-dependent test 約 130 件が
   pre-existing 失敗（[[public-mirror-preexisting-test-failures]] 既知制約・非回帰）。
3. **`make lint`（ruff）の `scripts/` 既存負債** — pre-existing（[[public-mirror-preexisting-lint-debt]]・非回帰）。
   自ファイルは `ruff check <file>` で隔離検証すること。

---

## 課題セクション（コードベース分析から抽出）

- **【高】成功条件（定量）が [UNVERIFIED]**: MS-PF2/PF3/PF4 の valid_loss 実測が GPU 依存。
  GOAL §4「成功の定義」は構造は完成。**proxy-scale valid_loss 証拠は本イテレーションで初取得**
  （`make freeze-validloss-ci` → TIES、RTX 3060 実測）。target-scale 数値検証のみが残り、それは
  private `src.data` pipeline 依存（分類 C・外部依存）。
- **【中】目的駆動ループの空転（本イテレーションで解消）**: `25_purpose_driven_executor` が
  Phase 1/2 で読む `SYSTEM_CONSTITUTION.md` / `PURPOSE.md` が**ルートに存在せず**、executor が
  毎回 skip → `run-pws` が no-output で exit 4 になっていた（直近イテレーションの却下理由）。
  GOAL.md を正本として両ファイルを蒸留・配置し、ループを始動可能にした。
- **【中】`src.data` 欠損による data-dependent test の不活性**: 公開ミラー制約。本 repo の
  品質 canary は `tests/test_cli_help_smoke.py`（37 passed / 3 xfailed）で回帰監視する。

---

## 次の一手（next execution）

> **方針転換（AI-Hub feedback 2026-06-25）**: Category-A（CPU-only 足場）は**枯竭**。
> 次イテレーションで**足場ヘルパーをこれ以上追加しない**こと（収益逓減・"indefinitely
> deferring the actual research result while accumulating orthogonal CPU scaffolding"）。
> 代わりに Category-C（GPU）ブロックを直接叩く — **本イテレーションでそれを実行した**。

### Category-A vs Category-C 台帳（quantification）

- **Category-A helpers 残数: 0**（MS-PF2 4/4・MS-PF3 2/2・MS-PF4 1/1 = 計 **7/7 完了**）:
  plan / cost / frontier / surrogate-gate / **surrogate-CI** / loss-ablation / schedule-portability。
- **Category-C run 残数: 1**（target-scale valid_loss 実測）— ただし**具体コマンドに縮約済み**:
  `make freeze-validloss-ci`（proxy-scale 判定は実測済み = **TIES**）。

### 本イテレーションで完了した Category-C 攻撃

1. **初の実 GPU valid_loss 判定を取得** — `make freeze-validloss-ci` が RTX 3060 で
   candidate(`output_first`) vs surrogate(`random_freeze_order`) を**実学習**し、real valid_loss 標本を
   `surrogate_valid_loss_ci()` に流入 → **TIES**（CI[95%]=[−0.096, +0.109]・n=5/5・non-thin）。
   homogeneous proxy stack では順序無効、という構造キーストン（`freeze_surrogate_gate`）を
   **実 GPU valid_loss で初めて経験的に確認**。`surrogate_valid_loss_ci()` が初めて構成定数ではなく
   実 run の標本を消費した。
2. **Category-C を具体コマンドに縮約** — target-scale 判定は同一導線に 9B の標本を流すのみ
   （`proxy_scale` フラグが target 昇格を示す）。残る外部依存は private `src.data` pipeline 単体
   （足場コードの追加では解決しない本物の外部依存）。

### 次候補（足場追加ではない）

1. **target-scale valid_loss 判定** — private `src.data` pipeline 利用可能次第、9B QLoRA run の
   candidate/surrogate valid_loss 標本を `surrogate_valid_loss_ci()` へ（導線は既存）。**唯一の真の
   研究結果だが、外部依存のため本 mirror 単独では実施不可**（[[public-mirror-preexisting-test-failures]]）。
2. **proxy 証拠の拡張（任意・低優先）** — heterogeneous stack（層幅/重要度の非対称）で順序が
   有意に効くかを probe し、`SURPASSES` 判定を proxy で観測できるか確認（パイプライン感度証明）。
   ただし target-scale の代用にはならない。
3. **避ける**: bootstrap-CI → G0–G4 ゲート配線などの**追加 Category-A ヘルパー**（feedback が収益逓減と
   明示）。target-scale 標本が無い段階でのゲート統合は空転になる。
