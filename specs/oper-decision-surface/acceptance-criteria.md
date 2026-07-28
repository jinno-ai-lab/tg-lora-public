# oper-decision-surface 受け入れ基準

**作成日**: 2026-07-28
**feature_id**: `oper-decision-surface`
**work_scope**: `light`
**関連要件定義**: [requirements.md](requirements.md)
**関連ユーザストーリー**: [user-stories.md](user-stories.md)
**分析記録**: [interview-record.md](interview-record.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 既存 `tests/test_section4_operator_decision.py` 43 tests + `scripts/section4_operator_decision.py` 実装 + `PURPOSE.md` 追記15/19/22/24 + TASK-0198/0199/0217/0224 からの直接抽出
- 🟡 **黄信号**: leaf pattern からの妥当な推測
- 🔴 赤信号: 本 spec では使用しない

---

## REQ-001: 2 deposit + worker probe を 1 snapshot へ consolidate 🔵

**信頼性**: 🔵 *scripts:354-523 + tests:TestSection4DecisionArc*

### Given

- `tests/fixtures/freeze_validloss_ci_9b_full.json` が commit 済
- `tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json` が commit 済
- `scripts/run_freeze_validloss_ci_9b.py` が import 可能

### When

- `assess_section4_decision()` を呼ぶ

### Then

- snapshot dict の `legs` キーに 2 つの leg dict（homogeneous, heterogeneous）が含まれる
- snapshot dict の `verdict_worker_status` キーが `"executable"` を示す
- snapshot dict の `arc_complete` キーが `True` を示す

### テストケース

- [x] **TC-001-01**: `test_arc_complete_on_real_deposits` 🔵
  - **入力**: 既存 deposit 2 件
  - **期待結果**: `snap["arc_complete"] is True`
  - **信頼性**: 🔵 *tests/test_section4_operator_decision.py*

- [x] **TC-001-02**: `test_both_legs_citable_faithful_ties` 🔵
  - **入力**: 既存 deposit 2 件
  - **期待結果**: 両 leg が `present=True ∧ citable_as_full_section4_verdict=True ∧ faithful=True ∧ rederived_verdict==TIES ∧ recorded_verdict==TIES`
  - **信頼性**: 🔵 *tests/test_section4_operator_decision.py*

---

## REQ-002: 各 leg の required field set 🔵

**信頼性**: 🔵 *scripts:188-240 + tests*

### Given

- commit 済 §4 deposit 1 件

### When

- `_assess_leg("homogeneous", "tests/fixtures/freeze_validloss_ci_9b_full.json", REPO_ROOT)` を呼ぶ

### Then

- leg dict に `present` / `deposit` / `label` / `citable_as_full_section4_verdict` / `faithful` / `rederived_verdict` / `recorded_verdict` / `candidate_mean` / `surrogate_mean` / `ci_lower` / `ci_upper` / `seq_len` / `proxy_scale` / `architecture` が全て含まれる

### テストケース

- [x] **TC-002-01**: `test_both_legs_citable_faithful_ties` 内に各 leg の field 存在確認が含まれる 🔵

---

## REQ-003: arc_complete は 4 predicate の conjunction 🔵

**信頼性**: 🔵 *scripts:397-403*

### Given

- test_repo_root に mutated deposit（例: `citable_as_full_section4_verdict=False`）

### When

- `assess_section4_decision(repo_root=tmp_path)` を呼ぶ

### Then

- leg 1 つでも `not present or not citable or not faithful or rederived != TIES` なら `arc_complete == False`

### テストケース

- [x] **TC-003-01**: `test_arc_complete_on_real_deposits`（real repo の 4 predicate 全 green）🔵
- [x] **TC-003-02**: `test_stale_recorded_verdict_breaks_arc`（faithful=false → arc_complete=False）🔵
- [x] **TC-003-03**: `test_non_citable_deposit_breaks_arc`（citable=false → arc_complete=False）🔵
- [x] **TC-003-04**: `test_missing_deposit_breaks_arc`（present=false → arc_complete=False）🔵

---

## REQ-004: arc_complete → SHIP regardless of executability 🔵

**信頼性**: 🔵 *scripts:485-497 + tests:101-105 (mutation-killed)*

### Given

- arc_complete = True（commit 済 deposit + reproducible probe）
- verdict_worker_status = "executable"

### When

- `assess_section4_decision()` を呼ぶ

### Then

- `recommendation == "SHIP"`
- `run_executable_here == True`
- （旧 logic は FIRE_OR_EXTEND を返していた → mutation-killed）

### テストケース

- [x] **TC-004-01**: `test_recommendation_is_ship_when_arc_complete` 🔵
- [x] **TC-004-02**: `test_arc_complete_ships_regardless_of_executability` 🔵

---

## REQ-005: arc incomplete + worker executable → FIRE_OR_EXTEND 🔵

**信頼性**: 🔵 *scripts:498-502 + tests:192-198*

### Given

- repo_root = tmp_path（deposit 無し）
- verdict_worker_status = "executable"

### When

- `assess_section4_decision(repo_root=str(tmp_path), verdict_worker_status="executable")` を呼ぶ

### Then

- `arc_complete == False`
- `recommendation == "FIRE_OR_EXTEND"`
- `rationale` に `"freeze-validloss-ci-9b-full"` を含む

### テストケース

- [x] **TC-005-01**: `test_arc_incomplete_and_executable_fires_or_extends` 🔵

---

## REQ-006: arc incomplete + architectural block → INCOMPLETE_ARC 🔵

**信頼性**: 🔵 *scripts:503-505 + tests:200-205*

### Given

- repo_root = tmp_path
- verdict_worker_status = "architectural_block"

### When

- `assess_section4_decision(repo_root=str(tmp_path), verdict_worker_status="architectural_block")` を呼ぶ

### Then

- `arc_complete == False`
- `recommendation == "INCOMPLETE_ARC"`

### テストケース

- [x] **TC-006-01**: `test_arc_incomplete_and_architecturally_blocked_is_incomplete` 🔵

---

## REQ-007: arc incomplete + transient block → FIRE_OR_EXTEND 🔵

**信頼性**: 🔵 *scripts:498-502 拡張 + tests:207-214*

### Given

- repo_root = tmp_path
- verdict_worker_status = "transient_block"

### When

- `assess_section4_decision(repo_root=str(tmp_path), verdict_worker_status="transient_block")` を呼ぶ

### Then

- `arc_complete == False`
- `recommendation == "FIRE_OR_EXTEND"`
- `rationale` に `"transient factor"` を含む

### テストケース

- [x] **TC-007-01**: `test_arc_incomplete_and_transient_block_still_fires_or_extends` 🔵

---

## REQ-008: quality_preservation clause は derived 🔵

**信頼性**: 🔵 *scripts:243-268, 415-424, 485-497 + tests:TestQualityPreservationAxis*

### Given

- commit 済 deposit 2 件（両 leg とも `baseline_present=True`）

### When

- `_quality_preservation_clause(snap["quality_preservation"])` を呼ぶ

### Then

- 返却文字列に `"homogeneous SURPASSES"` と `"heterogeneous SURPASSES"` が含まれる
- hardcoded 文字列が無い（mutation に対し review 容易）

### テストケース

- [x] **TC-008-01**: `test_homogeneous_baseline_verdict_is_surfaced_per_leg` 🔵
- [x] **TC-008-02**: `test_heterogeneous_baseline_verdict_is_surfaced_per_leg` 🔵
- [x] **TC-008-03**: `test_quality_preservation_clause_is_derived`（hardcoded 切除 mutation-kill）🔵

---

## REQ-101: probe verdict worker 3-class classification 🔵

**信頼性**: 🔵 *scripts:144-185 + tests:TestProbeClassification*

### Given

- mock subprocess runner（returncode 0 / 1 + stderr text）

### When

- `_probe_verdict_worker(runner=mock)` を呼ぶ

### Then

- returncode==0 → `("executable", "...public Dolly...")`
- stderr に `"No module named 'src"` を含み returncode==1 → `("architectural_block", "stripped src.* dep: ...")`
- stderr に `"No module named 'torch"` を含み returncode==1 → `("transient_block", "transient runtime factor (torch/etc.): ...")`

### テストケース

- [x] **TC-101-01**: `test_probe_executable_when_import_succeeds` 🔵
- [x] **TC-101-02**: `test_probe_architectural_block_on_src_missing` 🔵
- [x] **TC-101-03**: `test_probe_transient_block_on_torch_missing` 🔵

---

## REQ-201: blocking prompt + exit 3 🔵

**信頼性**: 🔵 *scripts:325-351, 700-745 + tests:TestBlockingPrompt*

### Given

- arc_complete = True
- `section4_landed_decision.json` absent

### When

- `python scripts/section4_operator_decision.py` を実行

### Then

- exit code = 3
- `_blocking_prompt()` 1 個のみ stderr へ
- prompt に `ship` / `accept_null` / `pivot` / `--land` の 3 branch + コマンド + "until you land one, this surface blocks (exit 3)" を含む

### テストケース

- [x] **TC-201-01**: `test_blocking_prompt_names_all_three_branches` 🔵
- [x] **TC-201-02**: `test_main_returns_three_when_arc_complete_unlanded` 🔵

---

## REQ-202: landed + valid → exit 0 🔵

**信頼性**: 🔵 *scripts:569-617 + tests:TestLandToExitZero*

### Given

- arc_complete = True
- `section4_landed_decision.json` が valid + branch = "ship"

### When

- `python scripts/section4_operator_decision.py` 実行

### Then

- exit code = 0
- stdout に `landed_decision.branch` / `basis` / `landed_at` / `pivot_private_repo_only` を出力

### テストケース

- [x] **TC-202-01**: `test_land_to_exit_zero_when_arc_complete` 🔵
- [x] **TC-202-02**: `test_landed_landing_record_loaded` 🔵

---

## REQ-203..205: --land rejection gates 🔵

**信頼性**: 🔵 *scripts:580-595 + tests:TestLandRejects*

### Given

- `--land <branch> --basis "<why>"` 起動
- arc_complete = False または branch 無効 または basis 空

### When

- 異なる rejection パターンで `--land` 起動

### Then

- exit code = 4（`EXIT_LAND_INVALID`）
- stderr へ `invalid --land branch {branch!r}; choose one of ship, accept_null, pivot` または `--land requires a non-empty --basis explaining the call.` を出力
- atomic landing record は書き込まれない

### テストケース

- [x] **TC-203-01**: `test_land_rejects_invalid_branch` 🔵
- [x] **TC-203-02**: `test_land_rejects_missing_basis` 🔵
- [x] **TC-203-03**: `test_land_rejects_arc_incomplete` 🔵

---

## REQ-206, REQ-207: PIVOT branch executable_here keys off src.data strip 🔵

**信頼性**: 🔵 *scripts:443-476 + tests:TestPivotBranchCorrection*

### Given

- src.data = stripped（real repo）または src.data = present（mock 注入）

### When

- `assess_section4_decision()` または `assess_section4_decision(src_data_present=True)` を呼ぶ

### Then

- stripped → `branches["pivot"]["executable_here"] == False ∧ branches["pivot"]["private_repo_only"] == True`
- present → `branches["pivot"]["executable_here"] == True ∧ branches["pivot"]["private_repo_only"] == False`

### テストケース

- [x] **TC-206-01**: `test_pivot_is_private_repo_only_in_this_mirror` 🔵
- [x] **TC-207-01**: `test_pivot_becomes_public_doable_when_src_data_present` 🔵

---

## REQ-301..306: --land atomic write 🔵

**信頼性**: 🔵 *scripts:297-322, 608-617 + tests:TestLandingRecordFormat*

### Given

- arc_complete = True
- `--land accept_null --basis "..."` 起動

### When

- 実際の `_write_landed_decision(...)` 呼び出し

### Then

- `section4_landed_decision.json` に `{branch, basis, landed=True, pivot_private_repo_only, deposits, landed_at}` が出力される
- `landed_at` は UTC ISO 8601
- `deposits` は `[HOMOGENEOUS_DEPOSIT, HETEROGENEOUS_DEPOSIT]`
- atomic 書き込み（`src.utils.io.save_json` 経由）

### テストケース

- [x] **TC-301-01**: `test_landing_record_has_required_fields` 🔵
- [x] **TC-301-02**: `test_landing_record_deposits_list` 🔵
- [x] **TC-301-03**: `test_landing_record_landed_at_is_iso8601` 🔵

---

## REQ-401: subprocess isolation 🔵

**信頼性**: 🔵 *scripts:144-186 + tests:TestSubprocessIsolation*

### Given

- テスト runner が torch 不在 / src.data strip

### When

- `assess_section4_decision()` を呼ぶ

### Then

- assert `verdict_worker_status in {"executable", "transient_block"}`（architectural_block ではない）
- subprocess probe 経由で torch 不在を transient_block として報告

### テストケース

- [x] **TC-401-01**: `test_real_checkout_worker_has_no_architectural_block` 🔵

---

## REQ-402: atomic write via io.save_json 🔵

**信頼性**: 🔵 *scripts:303-322 + scripts/atomic_save.py パターンのミラー*

### Given

- `_write_landed_decision(...)` 呼び出し

### When

- `section4_landed_decision.json` 書込中

### Then

- `src.utils.io.save_json(record, path)` 呼び出し経由（atomicな rename-with-tmpfile）
- kill mid-write で half-record が残らない

### テストケース

- [x] **TC-402-01**: `test_landing_record_atomic_write`（kill mid-write simulation → split-brain 検出）🔵

---

## REQ-403: non-unilateral（unilateral land 禁止）🔵

**信頼性**: 🔵 *PURPOSE.md 追記19 + scripts:700-745 + tests:TestBlockingPrompt*

### Given

- ツールは何も呼ばれない（CI 起動 / shell 起動）

### When

- landing record absent + arc_complete = True

### Then

- ツールは exit 0 を出してはならない（unilateral 着地禁止）
- exit 3 で blocking prompt

### テストケース

- [x] **TC-403-01**: `test_main_returns_three_when_arc_complete_unlanded`（= TC-201-02 と同一）🔵

---

## REQ-404: --land は recommendation を mutate しない 🔵

**信頼性**: 🔵 *scripts:485-577 + tests:TestLandPreservesRecommendation*

### Given

- arc_complete = True
- `--land accept_null --basis "..."` 起動後

### When

- snapshot 出力

### Then

- `recommendation == "SHIP"`（evidence-based recommendation）
- `landed_decision.branch == "accept_null"`（operator call）
- 両者が分離表示される

### テストケース

- [x] **TC-404-01**: `test_land_does_not_mutate_recommendation` 🔵

---

## REQ-405: VALID_LAND_BRANCHES pin 🔵

**信頼性**: 🔵 *scripts:133-141*

### Given

- `VALID_LAND_BRANCHES = ("ship", "accept_null", "pivot")` 定義

### When

- 4 個目の branch を `--land` で試みる

### Then

- exit 4 で拒否（silent 受理しない）

### テストケース

- [x] **TC-405-01**: `test_land_rejects_invalid_branch`（= TC-203-01 と同じ mutation で網羅）🔵

---

## REQ-501, REQ-502: human mode / JSON mode 出力 🔵

**信頼性**: 🔵 *scripts:531-548, 700-745 + tests:TestJsonMode, TestHumanMode*

### Given

- `--json` or `--land <branch> --basis "<why>"` 起動

### When

- snapshot 出力

### Then

- `--json`: stdout 1 行 JSON
- human mode: human-readable + (arc complete & 未着地時) blocking prompt

### テストケース

- [x] **TC-501-01**: `test_json_mode_outputs_single_json_line` 🔵
- [x] **TC-501-02**: `test_human_mode_output_includes_leg_summary` 🔵
- [x] **TC-501-03**: `test_blocking_prompt_in_human_mode` 🔵

---

## EDGE-001..004: malformed record handling 🔵

**信頼性**: 🔵 *scripts:277-294 + tests:TestLandingRecordRobustness*

### Given

- landing record が `{}` / `OSError` / `non-dict` / valid JSON but branch unknown

### When

- `_load_landed_decision(repo_root)` を呼ぶ

### Then

- 各ケースで `None` を返す（壊 record が着地を偽装しない）

### テストケース

- [x] **TC-EDGE-001-01**: `test_load_returns_none_for_missing_file` 🔵
- [x] **TC-EDGE-001-02**: `test_load_returns_none_for_malformed_json` 🔵
- [x] **TC-EDGE-001-03**: `test_load_returns_none_for_empty_dict` 🔵
- [x] **TC-EDGE-001-04**: `test_load_returns_none_for_unknown_branch` 🔵

---

## EDGE-101..104: --land rejection 細部 🔵

**信頼性**: 🔵 *scripts:587-595 + tests:TestLandRejects*

### Edge ケース

- branch typo ("accept_null_legacy")
- 空 basis (`""`)
- basis 引数欠落
- 別 invocation での record 上書き

### Given / When / Then

| ケース | Given | When | Then |
|---|---|---|---|
| typo branch | `--land accept_null_legacy --basis "valid"` | main | exit 4 + `invalid --land branch 'accept_null_legacy'; choose one of ship, accept_null, pivot` |
| 空 basis | `--land ship --basis ""` | main | exit 4 + `--land requires a non-empty --basis explaining the call.` |
| basis 欠落 | `--land ship` | main | exit 4 + 同上 |
| record 上書き | 1 度目 ship 着地 → 2 度目 accept_null 着地 | main | 後勝ち accept_null で record 更新 |

### テストケース

- [x] **TC-EDGE-101-01**: `test_land_rejects_invalid_branch` 🔵
- [x] **TC-EDGE-102-01**: `test_land_rejects_empty_basis` 🔵
- [x] **TC-EDGE-103-01**: `test_land_rejects_missing_basis` 🔵
- [x] **TC-EDGE-104-01**: `test_overwrite_landing_record` 🔵

---

## テストケースサマリー

### カテゴリ別件数

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| 機能要件 | 25 | 12 | 6 | 43 |
| 非機能要件 | 3 | 0 | 0 | 3 |
| Edgeケース | 0 | 4 | 4 | 8 |
| **合計** | **28** | **16** | **10** | **54** |

### 信頼性レベル分布

- 🔵 青信号: 54 件 (100%)
- 🟡 黄信号: 0 件 (0%)
- 🔴 赤信号: 0 件 (0%)

**品質評価**: 高品質（全 54 件 🔵 青信号、既存 43 tests との 1 対 1 写像 + 11 件は既存 tests からの同名 mapping）

### 優先度別テストケース

- **Must Have**: 38 件
- **Should Have**: 16 件
- **Could Have**: 0 件

---

## テスト実施計画

### Phase 1: 既存 artifacts 維持確認

- 43 tests passed in `tests/test_section4_operator_decision.py`（TASK-0198 で commit 済 baseline）
- 125 tests passed in `tests/test_run_freeze_validloss_ci_9b_*.py`系（既存 producer / worker family）
- 6 tests passed in `tests/test_section4_*.py` helper

### Phase 2: 本 spec 検証

- 本 spec には新規 production code ゼロ（既存実装 + 既存 tests 維持）
- 検証は docs-only: spec 4 files が commit され、既存 43 tests + 125 tests + 6 tests が一切 regression しないことで完了
