# oper-decision-surface 要件定義書


<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-07-28
**feature_id**: `oper-decision-surface`
**work_scope**: `light` — 機能追加（artifact）は既に commit 済（TASK-0198 / 0217）+ 43 tests green。本 spec は**既存の operator-decision 表面を EARS 形式で形式化**し、フィードバックループが「operator `--land` 以外の次手」を提案する前に踏むべき停止点を 1 つの正本へ蒸留する目的。

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 既存 `scripts/section4_operator_decision.py` 実装 + `tests/test_section4_operator_decision.py` 43 tests + `PURPOSE.md` 追記15/19/22/24/26 + GOAL.md §4/§7 + TASK-0198/0217 を直接参照
- 🟡 **黄信号**: 既存 leaf pattern（`atomic_save.py` / `checkpoint_integrity.py` / `cli_errors.py`）からの妥当な推測
- 🔴 **赤信号**: 将来拡張のための推測（本 spec では使用しない）

## 概要

`scripts/section4_operator_decision.py` が提供する **§4 verdict result（homogeneous + heterogeneous）の SHIP / ACCEPT-NULL / PIVOT 決定を operator が landing する機械可検証 surface**。AI-Hub feedback が「operator-facing `--land` 決定 + 9B target-scale 実行可能性」を PURPOSEM.md 残 open の 1 番目として永続指摘してきた経緯から、`PURPOSE.md` §4-247（P1 品質保持）と `GOAL.md` §4（Execution plan）/ §7（科学誠実性鉄則）に属する evidence-based recommendation を **non-unilateral** で operator に提示する。

正本は `scripts/section4_operator_decision.py`（commit 済）+ `tests/test_section4_operator_decision.py` + `PURPOSE.md` 追記15/19/22/24/26。本 spec はその EARS 形式による書き起こし。

## 関連文書

- **分析記録**: [💬 interview-record.md](interview-record.md)
- **ユーザストーリー**: [👤 user-stories.md](user-stories.md)
- **受け入れ基準**: [✅ acceptance-criteria.md](acceptance-criteria.md)
- **正本実装**: [`scripts/section4_operator_decision.py`](../../scripts/section4_operator_decision.py)
- **正本テスト**: [`tests/test_section4_operator_decision.py`](../../tests/test_section4_operator_decision.py)
- **コミット記録**: `PURPOSE.md` 追記15（TASK-0198）/ 追記19（TASK-0199）/ 追記22（TASK-0217）/ 追記24（TASK-0211）/ 追記26（TASK-0224）
- **直交 spec**: [`specs/freeze-ci-operator-errors/`](../../specs/freeze-ci-operator-errors/) — operator **error** 別 surface（exit 78）であり本 spec とは別軸
- **.audit 出典**: `[[ai-hub-feedback-infra-vs-this-repo]]` `[[silent-success-best-model-min-delta-axis]]` `[[progressive-freeze-level2-trio-orphaned]]`

## 機能要件（EARS記法）

### 通常要件

- **REQ-001**: システムは §4 verdict result の 2 つの committed deposit（`tests/fixtures/freeze_validloss_ci_9b_full.json` と `tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json`）と verdict worker（`scripts.run_freeze_validloss_ci_9b`）の import 可能性を単一の `assess_section4_decision()` snapshot へ consolidate しなければならない 🔵 *scripts/section4_operator_decision.py:354-523*

- **REQ-002**: システムは各 leg（homogeneous / heterogeneous）に対し `present` / `citable_as_full_section4_verdict` / `faithful` / `rederived_verdict` / `recorded_verdict` / `candidate_mean` / `surrogate_mean` / `ci_lower` / `ci_upper` / `seq_len` / `proxy_scale` / `architecture` を 1 leg 1 dict として surface しなければならない 🔵 *scripts:188-240 + tests/test_section4_operator_decision.py::TestSection4DecisionArc*

- **REQ-003**: システムは `arc_complete` を「両 leg が `present` ∧ `citable_as_full_section4_verdict` ∧ `faithful` ∧ `rederived_verdict == TIES`」の conjunction として算出しなければならない 🔵 *scripts:397-403 + tests:83-99*

- **REQ-004**: システムは `arc_complete == True` のとき、無条件に `recommendation = "SHIP"` を返さなければならない（verdict worker の executability に関わらず） 🔵 *scripts:485-497 + tests:101-105（mutation-killed: 旧 logic は executable 時に FIRE_OR_EXTEND を返していた）*

- **REQ-005**: システムは `arc_complete == False` かつ `verdict_worker_status == "executable"` のとき、`recommendation = "FIRE_OR_EXTEND"`（`rationale` に `freeze-validloss-ci-9b-full` を含む）を返さなければならない 🔵 *scripts:498-502 + tests:192-198*

- **REQ-006**: システムは `arc_complete == False` かつ `verdict_worker_status == "architectural_block"` のとき、`recommendation = "INCOMPLETE_ARC"` を返さなければならない 🔵 *scripts:503-505 + tests:200-205*

- **REQ-007**: システムは `arc_complete == False` かつ `verdict_worker_status == "transient_block"` のとき、`recommendation = "FIRE_OR_EXTEND"`（`rationale` に "transient factor" を含む）を返さなければならない 🔵 *scripts:498-502 拡張 + tests:207-214*

- **REQ-008**: システムは `quality_preservation` を各 leg の `baseline_present` / `baseline_verdict` / `candidate_mean` / `baseline_mean` / `n_baseline` から派生し、SH IPP rationale の `_quality_preservation_clause(quality_preservation)` 出力へ文字列連結しなければならない（hardcoded "heterogeneous unanswered" 禁止） 🔵 *scripts:243-268, 415-424, 485-497 + tests:TestQualityPreservationAxis*

### 条件付き要件

- **REQ-101**: `_probe_verdict_worker()` が subprocess として `python -c "import scripts.run_freeze_validloss_ci_9b"` を spawn し `returncode == 0` のとき `status = "executable"` とし、`stderr` に `"No module named 'src"` が出現すれば `status = "architectural_block"`、それ以外の import failure を `status = "transient_block"` として分類しなければならない 🔵 *scripts:144-185 + tests:TestProbeClassification*

- **REQ-102**: ツール起動時の `sys.executable` で verdict worker の import に失敗し、`stderr` が空または "No module named 'src" を含まないとき、`reason` 末尾は `f"transient runtime factor (torch/etc.): {last}"` 形式としなければならない 🔵 *scripts:181-185*

- **REQ-103**: `--json` mode は `assess_section4_decision()` の snapshot dict を `format_decision()` 経由で stdout 1 行 JSON として出力しなければならない 🔵 *scripts:531-548 + tests:TestJsonMode*

- **REQ-104**: human mode は `format_decision()` の人間可読形式を stdout に出力し、arc complete & 未着地時は `_blocking_prompt()` を続けて stderr へ出力しなければならない 🔵 *scripts:531-548 + scripts:325-351*

- **REQ-105**: landing record 読込時、absent / `OSError` / `ValueError` / `record.get("branch") not in VALID_LAND_BRANCHES` のいずれかであれば `None` を返し、**壊 record が着地を偽装してはならない** 🔵 *scripts:277-294 + tests:TestLandingRecordRobustness*

### 状態要件

- **REQ-201**: landing record が absent / invalid のとき、ツールは **blocking**（exit `EXIT_AWAITING_DECISION = 3`）し、`_blocking_prompt()` 1 個のみを emit しなければならない 🔵 *scripts:107-141, 325-351, 700-745 + tests:TestBlockingPrompt*

- **REQ-202**: landing record が valid（branch ∈ `("ship", "accept_null", "pivot")` ∧ basis ≠ "" ∧ arc complete）のとき、ツールは exit 0 とし、recommendation / landed_branch / basis を分けて出力しなければならない 🔵 *scripts:569-617 + tests:TestLandToExitZero*

- **REQ-203**: landing record が valid かつ arc complete が False（壊 record や陳腐化 record）のとき、ツールは exit `EXIT_LAND_INVALID = 4` で拒否しなければならない 🔵 *scripts:580-595 + tests:TestLandRejects*

- **REQ-204**: landing record が valid かつ branch が `VALID_LAND_BRANCHES` 以外のとき、ツールは exit `EXIT_LAND_INVALID = 4` で拒否し stderr へ `"invalid --land branch {branch!r}; choose one of ..."` を出力しなければならない 🔵 *scripts:587-595*

- **REQ-205**: landing record が valid かつ basis が空文字列または欠落のとき、ツールは exit `EXIT_LAND_INVALID = 4` で `--land requires a non-empty --basis explaining the call.` メッセージを stderr へ出力しなければならない 🔵 *scripts:593-595*

- **REQ-206**: src.data が `stripped_deliberate` のとき、`branches["pivot"]["executable_here"] = False` ∧ `branches["pivot"]["private_repo_only"] = True` としなければならない 🔵 *scripts:443-476 + tests:TestPivotBranchCorrection*

- **REQ-207**: src.data が `present` のとき、`branches["pivot"]["executable_here"] = True` ∧ `branches["pivot"]["private_repo_only"] = False` としなければならない 🔵 *scripts:466-475 + tests:228-233*

- **REQ-208**: ツールは verdict worker importability を **直接**（`scripts.run_freeze_validloss_ci_9b`）probe し、recover.py `--rerun`/`train_tg_lora` 経由の src.data import 失敗を verdict worker の architectural block として誤分類してはならない 🔵 *scripts:84-100, 144-186, 432-437 + tests:TestVerdictWorkerExecutability*

### オプション要件

- **REQ-301**: `--land <branch> --basis "<why>"` 起動時、ツールは `_write_landed_decision(repo_root, branch=branch, basis=basis, pivot_private_repo_only=…)` 経由で `section4_landed_decision.json` を atomic 書き込みしなければならない 🔵 *scripts:297-322, 608-617*

- **REQ-302**: `--land` 起動時、`branch ∈ VALID_LAND_BRANCHES` かつ `basis` が non-empty かつ `arc_complete == True` かつ `branch != "pivot" or src_data_present`（PIVOT 自動分岐）を全て gate として pass しなければならない 🔵 *scripts:580-595 + tests:TestLandRejectionGates*

- **REQ-303**: `--land` 起動時、`pivot` branch 選択で `src_data_status == "stripped_deliberate"` ならば `pivot_private_repo_only = True` フラグを record に付与しなければならない 🔵 *scripts:301-322, 588-595*

- **REQ-304**: `--land` 起動時、`pivot` branch 選択で `src_data_status == "present"` ならば `pivot_private_repo_only = False` フラグを record に付与しなければならない 🔵 *scripts:301-322*

- **REQ-305**: `--land` 起動時、`landed_at` フィールドは `datetime.now(timezone.utc).isoformat()`（UTC ISO 8601）として record に書き込まなければならない 🔵 *scripts:311-321*

- **REQ-306**: `--land` 起動時、`deposits` フィールドは `[HOMOGENEOUS_DEPOSIT, HETEROGENEOUS_DEPOSIT]` として record に書き込まなければならない 🔵 *scripts:316-321*

- **REQ-501**: human mode 出力は `recommendation` / `rationale` / 各 leg の `label` / `deposit` / `citable_as_full_section4_verdict` / `faithful` / `rederived_verdict` / `recorded_verdict` / `candidate_mean` / `surrogate_mean` / `ci_lower` / `ci_upper` / `seq_len` を含む list rendering 形式としなければならない 🔵 *scripts:531-548 + tests:TestHumanMode*

- **REQ-502**: `--land` 完了後の stdout 出力は `landed_decision.branch` / `landed_decision.basis` / `landed_decision.landed_at` / `landed_decision.pivot_private_repo_only` を含み、`recommendation` は変えずに出力しなければならない（"call is landed and binding"） 🔵 *scripts:606-617 + tests:TestLandPreservesRecommendation*

### 制約要件

- **REQ-401**: ツールは verdict worker の probe を subprocess 経由（`subprocess.run`）で行い、本体 process は torch を import してはならない 🔵 *scripts:144-185, 166-174 + tests:TestSubprocessIsolation*

- **REQ-402**: ツールは landing record の書き込みに `src.utils.io.save_json` を使用し、kill mid-write が half-record を残して「着地偽装」を生むことを防がなければならない 🔵 *scripts:303-322 + scripts/atomic_save.py パターンのミラー*

- **REQ-403**: ツールは `--land` を **unilateral に** 実行してはならない — landing record の存在しない間は blocking（exit 3）で operator の明示的呼びかけを待たなければならない 🔵 *PURPOSE.md 追記19 + scripts:700-745 + operator-facing decision-making 原則*

- **REQ-404**: ツールは `assessment_section4_decision()` に `recommendation` を **land** によって mutate してはならない — `reco=SHIP` + `landed_branch=accept_null` の同時表示が invariant 🔵 *scripts:485-577 + tests:TestLandPreservesRecommendation*

- **REQ-405**: ツールは `VALID_LAND_BRANCHES = ("ship", "accept_null", "pivot")` を唯一の branch 集合として pin しなければならない — 4 個目の branch は landing 拒否で decision space の silent drift を防ぐ 🔵 *scripts:133-141*

## 非機能要件

### パフォーマンス

- **NFR-001**: ツールの snapshot 出力（`assess_section4_decision()`）は GPU/torch なしで動作しなければならず、平均実行時間は 5 秒未満（probe が `cwd=REPO_ROOT` で subprocess を 1 回 spawn するコスト） 🔵 *scripts:144-186, 392-477 + tests:TestSubprocessIsolation*

- **NFR-002**: ツールの `--land` 書込は atomic 1 ファイル（`section4_landed_decision.json`）の 1 操作で完了しなければならず、landed 後の snapshot 出力は追加の subprocess を spawn してはならない 🔵 *scripts:297-322, 569-617*

### セキュリティ

- **NFR-101**: ツールは landing record を `cwd` ではなく `repo_root` 配下に書き込まなければならない（任意 `cwd` への書込による filesystem pollution 防止） 🔵 *scripts:271-274, 297-322*

- **NFR-102**: ツールは `branch` / `basis` の文字列長を無制限に受理するが、`section4_landed_decision.json` のサイズは documented 容量（最大 ~1 KB）以内に収まらなければならない（証拠は basis に何を載せるか次第） 🔵 *scripts:297-322*

- **NFR-103**: ツールは `--land` 時の `--basis` に **API key 風文字列**が混入しても commit してしまうが、その record は repo 内 1 ファイルに閉じ込まなければならず、外部 service へ send してはならない 🔵 *scripts:297-322*

### ユーザビリティ

- **NFR-201**: ツールは `blocking prompt` を **1 個のみ** 出力し、3 branch 名 + 正確な `--land` コマンド + "until you land one, this surface blocks (exit 3)" を含む plain text 形式としなければならない 🔵 *scripts:325-351*

- **NFR-202**: ツールは `--json` mode と human mode の両方をサポートし、CI / scripts から `--json`、operator の対話実行から human mode を使い分けられるようにしなければならない 🔵 *scripts:531-548, 719-745*

- **NFR-203**: ツールの `main` 入口は **4 つの exit code の意味**を docstring に明文化しなければならない（0 = landed / 2 = arc incomplete / 3 = arc complete but awaiting / 4 = --land rejected） 🔵 *scripts:55-57, 139-141*

## Edgeケース

### エラー処理

- **EDGE-001**: landing record が valid JSON だが `{}`（branch 欠落）のとき、`_load_landed_decision()` は `None` を返し、ツールは blocking（exit 3）しなければならない 🔵 *scripts:277-294 + tests:TestLandingRecordRobustness*

- **EDGE-002**: landing record が valid JSON で `branch = "ship"`（valid）だが `arc_complete == False` のとき、`--land` は exit 4 で拒否され、`landed_decision` は file system に書き込まれてはならない 🔵 *scripts:580-595*

- **EDGE-003**: landing record 書込直後に process が kill され、read 側で `OSError` が出るとき、ツールは blocking（exit 3）にフォールバックしなければならない 🔵 *scripts:286-294*

- **EDGE-004**: subprocess probe が `returncode == 0` でも `stderr` に `"No module named 'src"` が出る corner case（verdict worker 自体が src.data を import する壊れた実装になった場合）、`status = "architectural_block"` として分類しなければならない 🔵 *scripts:175-186*

### 境界値

- **EDGE-101**: `branch = "ship"` / `branch = "accept_null"` / `branch = "pivot"` 以外の `"accept_null_legacy"` 等の typo 入力時、exit 4 で拒否し stderr へ `"invalid --land branch 'accept_null_legacy'; choose one of ship, accept_null, pivot"` を出力しなければならない 🔵 *scripts:587-595*

- **EDGE-102**: `--land` 起動時 `--basis` を `--basis ""` のように空文字で指定したとき、exit 4 で `--land requires a non-empty --basis explaining the call.` を出力しなければならない 🔵 *scripts:593-595*

- **EDGE-103**: `--land` 起動時 `--basis` フラグ自体を欠落したとき、exit 4 で上記メッセージを出力しなければならない 🔵 *scripts:593-595*

- **EDGE-104**: shell で `--land ship --basis "Because the relative verdict is done."` のように両フラグを指定しても `--land accept_null --basis "..."` を別 invocation で再上書きしたとき、**後勝ち**で record を atomic 書込しなければならない（history は保持しない、これは "record the latest decision" 仕様） 🔵 *scripts:297-322*

## 信頼性レベル分布

- 🔵 青信号: **全 38 件**（100%）
- 🟡 黄信号: 0 件
- 🔴 赤信号: 0 件

## 既存 spec との切り分け

| 軸 | `freeze-ci-operator-errors` | `oper-decision-surface`（本 spec） |
|---|---|---|
| 主題 | operator **error** handling | operator **decision** landing |
| 失敗 source | missing config / malformed YAML / validation / malformed eval | 不要（operator が正常呼び出しした後の decision） |
| Exit code | 78（`EX_CONFIG`） | 0/2/3/4（landed / incomplete / awaiting / invalid） |
| 動作 | Exception → structured stderr / stdout JSON | snapshot + blocking prompt → atomic record |
| 起動契機 | entrypoint script の operator error 発生時 | `make section4-operator-decision` または `python scripts/section4_operator_decision.py` |
| 直交性 | 両 spec は**完全直交**（involve する script / leaf / exit / branch 全て別） | （同上） |

## 既存 TASK との対応

- **TASK-0198**: `scripts/section4_operator_decision.py` の 3-way exit 契約 + `_land_decision` + `_blocking_prompt` 追加（43 tests のうち 29 → 43 / 14 追加）
- **TASK-0199**: `_blocking_prompt` の live 確認 + decision tool が non-unilateral であることを live 再確認
- **TASK-0217**: 4bullet closeout（decision 機構は既に TASK-0198 で完成、bullet 4 = operator-facing surface は (a) 9B 両 leg TIES + (b) blocking prompt 稼働で既充足）
- **TASK-0224**: 本 surface は cache fingerprint 軸とは独立（`TestCacheFingerprintCompleteness` とは zero overlap）

## 関連文書

- **分析記録**: [interview-record.md](interview-record.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
