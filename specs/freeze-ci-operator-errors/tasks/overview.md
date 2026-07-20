# freeze-ci-operator-errors タスク概要

<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-07-20
**最終更新**: 2026-07-20（TASK-0179..0183 計画・freeze-ci-operator-errors feature の leaf module + 3 entrypoint wire-up + NFR consolidation を 5 TASK に分割）
**プロジェクト期間**: 2026-07-20 - 2026-07-20（5 TASK・1 iter 内）
**推定工数**: 5.5時間（TASK-0179: 1.5h + TASK-0180: 1.5h + TASK-0181: 1.5h + TASK-0182: 0.5h + TASK-0183: 0.5h）
**総タスク数**: 5件（TASK-0179..0183）
**要件数**: 32件（REQ-001..003 / REQ-101..102 / REQ-201..202 / REQ-301..303 + REQ-301a / REQ-401..402 / REQ-501..502 / REQ-601..602 / REQ-701..705 + NFR-101..102 / NFR-201..203 / NFR-301..302 + EDGE-001..004 / EDGE-101..103）
**次回タスク番号**: TASK-0184

## 関連文書

- **要件定義書**: [📋 requirements.md](../requirements.md)
- **設計文書**: [📐 architecture.md](../architecture.md)
- **データフロー**: [🔄 dataflow.md](../dataflow.md)
- **受け入れ基準**: [✅ acceptance-criteria.md](../acceptance-criteria.md)
- **ユーザストーリー**: [👤 user-stories.md](../user-stories.md)
- **実装分析**: [📝 interview-record.md](interview-record.md)
- **設計分析**: [📝 design-interview.md](../design-interview.md)
- **型定義**: [🔌 interfaces.py](../interfaces.py)
- **コンテキスト**: [📝 note.md](../note.md)
- **正本**: [📜 docs/GOAL.md](../../../docs/GOAL.md) §7

## フェーズ構成

| フェーズ | 期間 | 成果物 | タスク数 | 工数 | ファイル |
|---------|------|--------|----------|------|----------|
| Phase 81.1 | 1.5h | leaf module + 4 subtype + 4 wrapper + 1 emitter + 22 unit test | 1件 | 1.5h | [TASK-0179](#phase-811-leaf-module) |
| Phase 81.2 | 1.5h | replay entrypoint wire-up + 17 integration test | 1件 | 1.5h | [TASK-0180](#phase-812-replay-entrypoint-wire-up) |
| Phase 81.3 | 1.5h | producer entrypoint wire-up + `--json` mode + 15 integration test | 1件 | 1.5h | [TASK-0181](#phase-813-producer-entrypoint-wire-up) |
| Phase 81.4 | 0.5h | launcher exit 78 → FATAL 分類 + 7 contract / integration test | 1件 | 0.5h | [TASK-0182](#phase-814-launcher-exit-78--fatal-分類) |
| Phase 81.5 | 0.5h | NFR / zero-regression consolidation + 11 axis-wide test | 1件 | 0.5h | [TASK-0183](#phase-815-nfr--zero-regression-consolidation) |

## タスク番号管理

**使用済みタスク番号**: TASK-0179 ~ TASK-0183
**次回開始番号**: TASK-0184
**前 TASK 番号**: TASK-0178（`9737ace`・composite `passes` boolean bind・9 axis bind family の最終 commit）

## 全体進捗

- [x] Phase 81.1: leaf module + 4 subtype + 4 wrapper + 1 emitter + 22 unit test ✅ TASK-0179
- [x] Phase 81.2: replay entrypoint wire-up + 17 integration test ✅ TASK-0180
- [x] Phase 81.3: producer entrypoint wire-up + `--json` mode + 15 integration test ✅ TASK-0181
- [x] Phase 81.4: launcher exit 78 → FATAL 分類 + 7 contract / integration test ✅ TASK-0182
- [x] Phase 81.5: NFR / zero-regression consolidation + 11 axis-wide test ✅ TASK-0183

## マイルストーン

- **M1: leaf module 成立** (2026-07-20): `src/utils/cli_errors.py` 新設 + 4 subtype + 4 wrapper + 1 emitter + 22 unit test green・既存 9 axis bind family（TASK-0171..0178）byte-identical 維持
- **M2: replay wire-up 成立** (2026-07-20): `scripts/replay_freeze_validloss_ci.py` の 4 raise site 置換 + `main()` outer try/except + 17 integration test green
- **M3: producer wire-up 成立** (2026-07-20): `scripts/run_freeze_validloss_ci_9b.py` の 4 call site 置換 + `--json` flag 新設 + `main()` outer try/except + 15 integration test green
- **M4: launcher wire-up 成立** (2026-07-20): `scripts/launch_freeze_ci_9b_full.py::classify_exit_code` の 78 → FATAL 分岐追加 + 5 個目 worker exit code contract pin + 7 test green
- **M5: axis consolidation 成立** (2026-07-20): 5 cluster（replay / producer / launch_honesty / worker_launcher_exit_contract / config_launchability）byte-identical 緑 + NFR-101 axis-wide mutation 証明 + 11 axis-wide test green

---

## Phase 81.1: leaf module

**期間**: 1.5時間
**目標**: operator-error axis の chokepoint 最小核（leaf module）を新設・4 subtype 階層 + 4 wrapper + 1 emitter + `EXIT_OPERATOR_ERROR=78` 定数 + 22 unit test（mutation 証明含む）を commit
**成果物**: `src/utils/cli_errors.py`（NEW・interfaces.py 逐語実装）+ `tests/test_cli_operator_errors.py::TestOperatorErrorHierarchy` + `TestWrapperMutation` + `TestEmitter` + `TestLeafIndependence`（22 test）

### タスク一覧

- [x] [TASK-0179: leaf module `src/utils/cli_errors.py` 新設 + 4 subtype 階層 + 4 wrapper + 1 emitter + 22 unit test](TASK-0179.md) - 1.5h (feat) 🔵

### 依存関係

```
TASK-0179 → TASK-0180
TASK-0179 → TASK-0181
TASK-0179 → TASK-0182
```

---

## Phase 81.2: replay entrypoint wire-up

**期間**: 1.5時間
**目標**: TASK-0179 leaf を使って `scripts/replay_freeze_validloss_ci.py` を wire・`load_samples()` 内の 4 raise site（`FileNotFoundError` / `JSONDecodeError` / 必須 key 欠落 / 型不一致）を leaf wrapper に置換 + `main()` outer try/except + 17 integration test commit
**成果物**: `scripts/replay_freeze_validloss_ci.py`（修正）+ `tests/test_replay_freeze_validloss_ci.py::TestReplayEntrypointIntegration`（17 test）

### タスク一覧

- [x] [TASK-0180: replay entrypoint wire-up + 17 integration test](TASK-0180.md) - 1.5h (fix) 🔵

### 依存関係

```
TASK-0179 → TASK-0180
TASK-0180 → TASK-0183
```

---

## Phase 81.3: producer entrypoint wire-up

**期間**: 1.5時間
**目標**: TASK-0179 leaf を使って `scripts/run_freeze_validloss_ci_9b.py` を wire・`OmegaConf.load()` 2 call site + `load_and_validate_config()` 1 call site の 4 raise site を leaf wrapper に置換 + `--json` flag 新設（REQ-502）+ `main()` outer try/except + 15 integration test commit
**成果物**: `scripts/run_freeze_validloss_ci_9b.py`（修正）+ `tests/test_run_freeze_validloss_ci_9b_producer_operator_errors.py::TestProducerEntrypointIntegration`（15 test）

### タスク一覧

- [x] [TASK-0181: producer entrypoint wire-up + `--json` mode + 15 integration test](TASK-0181.md) - 1.5h (fix) 🔵

### 依存関係

```
TASK-0179 → TASK-0181
TASK-0181 → TASK-0183
```

---

## Phase 81.4: launcher exit 78 → FATAL 分類

**期間**: 0.5時間
**目標**: `scripts/launch_freeze_ci_9b_full.py::classify_exit_code` に `code == 78 → Action.FATAL("operator_error", 0.0)` 分岐追加 + `tests/test_worker_launcher_exit_contract.py` に 5 個目 contract pin 追加 + `TestLauncherExit78Classification` 5 test + `TestLauncherExitCodeIntegration` 1 test = 7 test commit
**成果物**: `scripts/launch_freeze_ci_9b_full.py`（修正）+ `tests/test_worker_launcher_exit_contract.py`（5 個目 contract pin）+ `tests/test_launch_freeze_ci_9b_full.py::TestLauncherExit78Classification`（5 test）+ `tests/test_cli_operator_errors.py::TestLauncherExitCodeIntegration`（1 test）

### タスク一覧

- [x] [TASK-0182: launcher exit 78 → FATAL 分類 + 7 contract / integration test](TASK-0182.md) - 0.5h (fix) 🔵

### 依存関係

```
TASK-0179 → TASK-0182
TASK-0182 → TASK-0183
```

---

## Phase 81.5: NFR / zero-regression consolidation

**期間**: 0.5時間
**目標**: 直前 4 TASK が個別に打った pin を **統合 test layer** として consolidate・4 wrapper × 1 entrypoint ずつの NFR-101 axis-wide mutation 証明 + 5 cluster（replay / producer / launch_honesty / worker_launcher_exit_contract / config_launchability）zero-regression 統合 pin + NFR-201/202/203 message 品質 + EDGE-102/103 boundary の 4 class・11 test commit
**成果物**: `tests/test_cli_operator_errors.py`（4 class・11 test 追加・**新 production code なし**）

### タスク一覧

- [x] [TASK-0183: NFR / zero-regression consolidation + 11 axis-wide test](TASK-0183.md) - 0.5h (test) 🔵

### 依存関係

```
TASK-0179 → TASK-0183
TASK-0180 → TASK-0183
TASK-0181 → TASK-0183
TASK-0182 → TASK-0183
```

---

## 信頼性レベルサマリー

### 全タスク統計

- **総タスク数**: 5件
- 🔵 **青信号**: 5件 (100%)
- 🟡 **黄信号**: 0件 (0%)
- 🔴 **赤信号**: 0件 (0%)

### フェーズ別信頼性

| フェーズ | 🔵 青 | 🟡 黄 | 🔴 赤 | 合計 |
|---------|-------|-------|-------|------|
| Phase 81.1 (leaf) | 1 | 0 | 0 | 1 |
| Phase 81.2 (replay) | 1 | 0 | 0 | 1 |
| Phase 81.3 (producer) | 1 | 0 | 0 | 1 |
| Phase 81.4 (launcher) | 1 | 0 | 0 | 1 |
| Phase 81.5 (consolidation) | 1 | 0 | 0 | 1 |

**品質評価**: 高品質（全 5 TASK が 🔵 青信号・interfaces.py の signature 凍結 + acceptance-criteria.md の TC-* pin 対象 によって全 TASK の実装・test 範囲が spec で一意に決定）

### タスク別 test 数

| TASK | 新規 test | 既存 test 維持 | mutation 証明 |
|------|-----------|----------------|----------------|
| TASK-0179 | 22 | (なし・新規 leaf) | 8 wrapper mutation |
| TASK-0180 | 17 | 157 (TASK-0171..0178) | 4 raise site mutation |
| TASK-0181 | 15 | producer smoke cluster | 4 call site mutation |
| TASK-0182 | 7 | 4 worker exit code (ad8c84a) | 1 launcher 78 mutation |
| TASK-0183 | 11 | 22 (TASK-0179) + 174 (TASK-0180) + 15 (TASK-0181) + 7 (TASK-0182) | 4 axis-wide wrapper mutation |
| **合計** | **72** | (上記 5 cluster) | (21 mutation 証明) |

### タスク別 production code 変更

| TASK | 変更 file | 種別 |
|------|-----------|------|
| TASK-0179 | `src/utils/cli_errors.py` | NEW (interfaces.py 逐語実装) |
| TASK-0180 | `scripts/replay_freeze_validloss_ci.py` | 修正 (4 raise site 置換 + outer try/except) |
| TASK-0181 | `scripts/run_freeze_validloss_ci_9b.py` | 修正 (4 call site 置換 + `--json` flag + outer try/except) |
| TASK-0182 | `scripts/launch_freeze_ci_9b_full.py` | 修正 (1 分岐追加) |
| TASK-0183 | なし | (test 追加のみ) |

## クリティカルパス

```
TASK-0179 → TASK-0180 → TASK-0183
TASK-0179 → TASK-0181 → TASK-0183
TASK-0179 → TASK-0182 → TASK-0183
```

**クリティカルパス工数**: 3.5時間（TASK-0179 → TASK-0180 → TASK-0183 = 1.5h + 1.5h + 0.5h = 3.5h）
**並行作業可能工数**: 2.0時間（TASK-0181 と TASK-0182 は TASK-0179 完了後に並行可能 = 1.5h + 0.5h = 2.0h、ただし 1.5h が 0.5h より長いため実並行時間は 1.5h）
**実最短系列**: 4.5時間（TASK-0179 → max(TASK-0180, TASK-0181, TASK-0182) → TASK-0183 = 1.5h + 1.5h + 0.5h）

## 直交 axis との独立性

| Axis | 直交性 | 備考 |
|------|--------|------|
| 9 axis bind family (TASK-0171..0178) | 🔵 完全直交 | operator error handling は stored boolean axis とは別 chokepoint・本 TASK は `format_replay` / `replay_to_json` 内の stale check に **触れない** |
| CUDA OOM path (`4afc5e9`) | 🔵 完全直交 | 9B target-scale 実行失敗の distinct path・本 TASK は GPU 実行 path に **触れない** |
| 4 worker exit code (ad8c84a) | 🔵 拡張のみ | 本 TASK は 5 個目 contract pin を **追加**（既存 4 個は無変更） |
| 5 launch-honesty invariants (b8ee35c) | 🔵 拡張のみ | invariant 4 = `worker_exit_code_contract_pinned` が 5 個目に grow しても invariant 自体は不変 |
| Atomic torch.save axis (d827507) | 🔵 完全直交 | torch-free leaf + entrypoint 統合のみ |
| Checkpoint load-side integrity axis (59781e5) | 🔵 完全直交 | leaf は `checkpoint_integrity.py` と無関係 |
| Atomic JSON deposit write axis (54a4cd8) | 🔵 完全直交 | operator error 出力は `--json` mode の stdout 1 行のみ・deposit 書き込みとは無関係 |
| 9 axis verdict honesty gate (8edf287, b8ee35c) | 🔵 完全直交 | committed corpus byte-identical 維持 |

## 次のステップ

タスクを実装するには:

- **全タスク順番に実装**: 各 TASK を 1 commit で独立に merge
  - TASK-0179（leaf 単体・既存 code path に触れない）→ TASK-0180（replay wire-up）→ TASK-0181（producer wire-up・TASK-0180 と並行可）→ TASK-0182（launcher wire-up・TASK-0180/0181 と並行可）→ TASK-0183（consolidation・最後）
- **特定タスクを実装**: `/tsumiki:kairo-implement TASK-0179` のように指定

## AI-Hub feedback 系への誠実応答（freeze-ci-operator-errors 計画全体）

- **bind family 完全性**: 本 feature は bind 軸ではなく **operator 入力 vs 内部 state** の chokepoint 化 = 9 axis bind family（TASK-0171..0178）とは **直交 axis**。feedback の "bind family is structurally complete" 原則と矛盾しない。
- **test-only witness 警告**: TASK-0179/0180/0181/0182 は **production code 変更を含む**（leaf 新設 + 3 entrypoint 修正）= test-only ではない。TASK-0183 のみ test 追加だが、`b8ee35c` assembled launch-honesty dry-run pattern の **同形** として axis-wide integrated regression net を pin する consolidation であり、個別 test-only witness とは性質が異なる。
- **doc_spine.yml references block**: 本 feature 全体（TASK-0179..0183）で `specs/_doc_spine.yml` には **触れない**（per-anchor approach 維持）。
- **phantom lever**（recursive.enabled / max_passes / rfb eval / Kendall τ 等）: 本 feature 全体では **使用しない**。
- **two fixes (jsonl defer + async_cache guard)**: 本 feature 全体は **別 axis**（operator error handling ≠ 9B producer 学習 loop 内部の jsonl/async_cache）= 受理は維持しつつ本 feature の scope には含めない。
- **doc-only commit 警告**: 本 feature 全体は **production code + test code の双方を変更** する = doc-only ではない。
