# freeze-ci-operator-errors 受け入れ基準

<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-07-20
**関連要件定義**: [requirements.md](requirements.md)
**関連ユーザストーリー**: [user-stories.md](user-stories.md)
**分析記録**: [interview-record.md](interview-record.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: AI_HUB_MAKE_RUN_FEEDBACK・既存実装・既存 test で直接支持
- 🟡 **黄信号**: 既存 test pattern・既存 launch-honesty invariants から妥当な推測
- 🔴 **赤信号**: 参照資料にない自動推定

---

## REQ-001: 4 subtype の `OperatorError` 階層 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK*

### Given（前提条件）

- 4 subtype 全てが `OperatorError` を継承している
- 各 subtype が `__str__` と `to_dict` を持つ

### When（実行条件）

- 各 subtype を `OperatorError("<detail>")` 形式で instance 化する

### Then（期待結果）

- `MissingConfigError` / `MalformedYAMLError` / `AppConfigValidationError` / `MalformedEvalResultsError` の 4 個が import 可能
- 各 instance が `isinstance(e, OperatorError)` で `True` を返す
- 各 instance の `to_dict()` が `{"error": class 名, "detail": "...", "exit_status": 78}` を返す

### テストケース

#### 正常系

- [x] **TC-001-01**: 4 subtype が import 可能 🔵
  - **入力**: `from src.utils.cli_errors import MissingConfigError, MalformedYAMLError, AppConfigValidationError, MalformedEvalResultsError, OperatorError`
  - **期待結果**: 全て import 成功
  - **信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK*
  - **検証**: `tests/test_cli_operator_errors.py::TestOperatorErrorHierarchy::test_four_subtypes_importable`

- [x] **TC-001-02**: 各 instance が `to_dict()` で同じ schema を返す 🔵
  - **入力**: 4 subtype それぞれの instance
  - **期待結果**: `{"error": <class 名>, "detail": <str>, "exit_status": 78}`
  - **信頼性**: 🔵 *REQ-002*
  - **検証**: `TestOperatorErrorHierarchy::test_to_dict_schema_frozen`

- [x] **TC-001-03**: 各 instance が `isinstance(e, OperatorError)` で True 🔵
  - **入力**: 4 subtype それぞれの instance
  - **期待結果**: `True`
  - **信頼性**: 🔵 *REQ-001*
  - **検証**: `TestOperatorErrorHierarchy::test_each_subtype_isinstance_operator_error`

#### 異常系

- [x] **TC-001-E01**: detail が空文字でも `to_dict()` が `"detail": ""` を返す 🔵
  - **入力**: `MissingConfigError("")`
  - **期待結果**: `to_dict() == {"error": "MissingConfigError", "detail": "", "exit_status": 78}`
  - **信頼性**: 🔵 *REQ-002 の `__str__` 整合*
  - **検証**: `TestOperatorErrorHierarchy::test_empty_detail_preserved`

---

## REQ-101: `--config` 欠落（Missing config） 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK + `FileNotFoundError` wrapper*

### Given（前提条件）

- `--config` 引数で存在しない path を指定
- producer / launcher script のいずれか

### When（実行条件）

- `python -m scripts.run_freeze_validloss_ci_9b --config /nonexistent.yaml` を実行

### Then（期待結果）

- stderr に `MissingConfigError: config not found: /nonexistent.yaml` を出力
- exit code 78 で終了
- `passes_stale` / `is_material_stale` / `significant_surpasses_stale` 等の honesty gate は発火しない（producer が起動前に fail するため）

### テストケース

#### 正常系

- [x] **TC-101-01**: producer で `--config /nonexistent.yaml` が exit 78 🔵
  - **入力**: producer script + non-existent path
  - **期待結果**: `exit_code == 78`, `stderr` に `MissingConfigError: config not found: /nonexistent.yaml`
  - **信頼性**: 🔵 *REQ-101*
  - **検証**: `tests/test_run_freeze_validloss_ci_9b_producer_operator_errors.py::TestProducerMainOperatorError::test_missing_config_exits_78`

- [x] **TC-101-02**: launcher で `--config /nonexistent.yaml` が exit 78 🔵
  - **入力**: launcher script + non-existent path
  - **期待結果**: `exit_code == 78`, `stderr` に `MissingConfigError: config not found: /nonexistent.yaml`
  - **信頼性**: 🔵 *REQ-101*
  - **検証**: launcher は `--config` を parse せず worker に forward する設計。worker が `exit 78`（`MissingConfigError`）を出すと launcher は `classify_exit_code(78) → Action.FATAL("operator_error")` で **retry せず** `main()` が `78` を返す（assembled）。`tests/test_launch_freeze_ci_9b_full.py::TestMainDryRun::test_main_mirrors_worker_operator_error_exit` + `TestLauncherExit78Classification` + `tests/test_worker_launcher_exit_contract.py`（78→FATAL 5 個目 contract pin）。

- [x] **TC-101-03**: replay で `--samples-file /nonexistent.json` が exit 78 🔵
  - **入力**: replay script + non-existent samples file
  - **期待結果**: `exit_code == 78`, `stderr` に `MissingConfigError: samples file not found: /nonexistent.json`
  - **信頼性**: 🔵 *REQ-102*
  - **検証**: `tests/test_replay_freeze_validloss_ci.py::test_rejects_missing_file`（`raise_missing_config(path, kind="samples file")`）

#### 境界値

- [x] **TC-101-B01**: `--config` が directory path 🔵
  - **入力**: `/tmp` 等の directory path
  - **期待結果**: `exit_code == 78`, `MissingConfigError` (`is a directory` detail を含む)
  - **信頼性**: 🟡 *EDGE-001* → 🔵 *verified*
  - **検証**: `TestLoadCfgConversion::test_directory_config_is_flagged` + leaf `TestWrapperMutation::test_raise_missing_config_directory`

#### mutation 証明

- [x] **TC-101-M01**: `_raise_missing_config` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*
  - **検証**: `tests/test_cli_operator_errors.py::TestOperatorErrorAxisMutationProof::test_neutralized_wrapper_drops_detection[raise_missing_config]`

---

## REQ-201: YAML parse error（Malformed YAML） 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK + `yaml.YAMLError` wrapper*

### Given（前提条件）

- `--config` 引数で YAML 文法的に壊れた file を指定

### When（実行条件）

- `python -m scripts.run_freeze_validloss_ci_9b --config broken.yaml` を実行（tab/space 混在等）

### Then（期待結果）

- stderr に `MalformedYAMLError: yaml parse error in <path>: <PyYAML message>` を出力
- exit code 78 で終了
- PyYAML の `__str__()` 結果（行番号・列番号含む）を保持

### テストケース

#### 正常系

- [x] **TC-201-01**: producer で壊れた YAML が exit 78 🔵
  - **入力**: tab/space 混在 YAML file
  - **期待結果**: `exit_code == 78`, `stderr` に `MalformedYAMLError: yaml parse error in <path>: ...`
  - **信頼性**: 🔵 *REQ-201*
  - **検証**: `TestProducerMainOperatorError::test_malformed_yaml_exits_78`

- [x] **TC-201-02**: 行番号・列番号が stderr に含まれる 🔵
  - **入力**: 壊れた YAML
  - **期待結果**: stderr message に `line N, column M` が含まれる
  - **信頼性**: 🔵 *REQ-202*
  - **検証**: `TestLoadCfgConversion::test_malformed_yaml_preserves_line_column` + leaf `TestWrapperMutation::test_raise_malformed_yaml_preserves_parser_msg`

- [x] **TC-201-03**: launcher で壊れた YAML が exit 78 🔵
  - **入力**: launcher script + 壊れた YAML
  - **期待結果**: `exit_code == 78`, `MalformedYAMLError`
  - **信頼性**: 🔵 *REQ-201*
  - **検証**: TC-101-02 と同 assembled 経路。worker（producer）が壊れた YAML で `exit 78`（`MalformedYAMLError`）→ launcher は FATAL で `78` を返す。`test_main_mirrors_worker_operator_error_exit`（worker exit 78 → launcher 78）。

#### 境界値

- [x] **TC-201-B01**: empty file が `MalformedYAMLError` を発火 🔵
  - **入力**: 0-byte file
  - **期待結果**: `MalformedYAMLError: yaml parse error in <path>: file is empty`
  - **信頼性**: 🟡 *EDGE-002* → 🔵 *verified*
  - **検証**: `TestLoadCfgConversion::test_empty_file_raises_malformed_yaml` + `TestProducerMainOperatorError::test_empty_config_exits_78`

#### mutation 証明

- [x] **TC-201-M01**: `_raise_malformed_yaml` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*
  - **検証**: `TestOperatorErrorAxisMutationProof::test_neutralized_wrapper_drops_detection[raise_malformed_yaml]`

---

## REQ-301: Pydantic validation error（AppConfig validation） 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK + `pydantic.ValidationError` wrapper*

### Given（前提条件）

- YAML は parse できるが Pydantic スキーマに違反する field を含む
- `extra="forbid"` 設定済の Pydantic model

### When（実行条件）

- `python -m scripts.run_freeze_validloss_ci_9b --config with_extra_field.yaml` を実行

### Then（期待結果）

- stderr に `AppConfigValidationError: schema validation failed for <ConfigClass>: <N> errors; first: <loc> <msg> (<type>)` を出力
- exit code 78 で終了
- `error_count` は `len(pydantic.errors())` と同値
- first error は `pydantic.errors()[0]` の `loc / msg / type` を含む

### テストケース

#### 正常系

- [x] **TC-301-01**: producer で `extra="forbid"` 違反が exit 78 🔵
  - **入力**: `LoggingConfig` に未宣言 field を追加した YAML
  - **期待結果**: `exit_code == 78`, `AppConfigValidationError: schema validation failed for <ConfigClass>: 1 errors; first: logging.<field_name> Extra inputs are not permitted (extra_forbidden)`
  - **信頼性**: 🔵 *REQ-301, REQ-302, REQ-303*
  - **検証（TASK-0184 resolved）**: producer `_load_cfg` が `config_schema.validate_config_data` で Pydantic validate し、`pydantic.ValidationError` → `raise_app_config_validation(exc.title, exc)` で exit 78 経路を新設。unit `TestLoadCfgPydanticValidation::test_extra_forbidden_field_raises_app_config` + integration `TestProducerMainAppConfigValidation::test_extra_field_exits_78`。class 名は `tg_lora` key 有無で dispatch（producer 既定 config 群は `BaselineConfig`、`tg_lora` 含む config は `TGLoRAConfig`）→ `test_tg_lora_config_class_dispatch` + `test_baseline_config_class_name_in_detail` で両方 pin。pydantic v2 error type は `extra_forbidden`（旧 `value_error.extra` は pydantic v1 表記）。

- [x] **TC-301-02**: launcher で必須 field 欠落が exit 78 🔵
  - **入力**: 必須 field を削除した YAML
  - **期待結果**: `exit_code == 78`, `AppConfigValidationError: schema validation failed for <ConfigClass>: ...; first: <field> Field required (missing)`
  - **信頼性**: 🔵 *REQ-301*
  - **検証（TASK-0184 resolved）**: producer 側 `TestLoadCfgPydanticValidation::test_missing_required_field_raises_app_config` + `TestProducerMainAppConfigValidation::test_missing_required_field_exits_78`（pydantic v2 type `missing`）。launcher assembled 経路は worker（producer）が Pydantic 違反で `exit 78` を出すと `classify_exit_code(78) → FATAL("operator_error")` で launcher `main()` が 78 を mirror → `tests/test_launch_freeze_ci_9b_full.py::test_main_mirrors_worker_operator_error_exit`（class 非依存・worker exit 78 → launcher 78）。

- [x] **TC-301-03**: producer で型不一致が exit 78 🔵
  - **入力**: 期待型 `int` だが `str` を代入した YAML
  - **期待結果**: `exit_code == 78`, `AppConfigValidationError: ...; first: <field> Input should be a valid integer, unable to parse string as an integer (int_parsing)`
  - **信頼性**: 🔵 *REQ-301*
  - **検証（TASK-0184 resolved）**: producer `TestLoadCfgPydanticValidation::test_type_mismatch_raises_app_config` + `TestProducerMainAppConfigValidation::test_type_mismatch_exits_78`（pydantic v2 type `int_parsing`）。

- [x] **TC-301-04**: `error_count` が Pydantic の `len(errors())` と一致 🔵
  - **入力**: 複数の field 違反
  - **期待結果**: stderr message の `N` が `len(pydantic.errors())` と一致
  - **信頼性**: 🔵 *REQ-302*
  - **検証**: leaf `TestWrapperMutation::test_raise_app_config_validation_error_count` + producer `TestAppConfigValidationWrapper::test_error_count_matches_pydantic`（REAL `pydantic.ValidationError`、`N == len(exc.errors())`）。wrapper は config class 非依存（class 名を param で受ける）なので `TGLoRAConfig`/`BaselineConfig` いずれでも同形式。

#### 境界値

- [x] **TC-301-B01**: `BaselineConfig` 違反も `AppConfigValidationError` を発火 🔵
  - **入力**: `9b_baseline.yaml` 系の違反
  - **期待結果**: `BaselineConfig` class 名が stderr に出力
  - **信頼性**: 🔵 *REQ-303*
  - **検証（TASK-0184 resolved）**: producer 既定 config 群（`9b_baseline_suffix_only_last25.yaml`）は `BaselineConfig` に dispatch → `TestLoadCfgPydanticValidation::test_baseline_config_class_name_in_detail`（`schema validation failed for BaselineConfig`）+ integration `TestProducerMainAppConfigValidation::test_app_config_error_stderr_has_class_name_line`。

#### mutation 証明

- [x] **TC-301-M01**: `_raise_app_config_validation` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*
  - **検証**: `TestOperatorErrorAxisMutationProof::test_neutralized_wrapper_drops_detection[raise_app_config_validation]`

---

## REQ-401: 破損 eval result（Malformed eval results） 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK + `load_samples()` schema 強化*

> **【検証済 schema（reality reconciliation）】** `scripts/replay_freeze_validloss_ci.py::load_samples`
> の verified 必須 key は `candidate_losses` / `surrogate_losses`（共に非空 `list`）。
> 下記 TC 中の旧 key 名（`candidate_total` / `samples` / `valid_losses`）は schema 実装の
> `candidate_losses` / `surrogate_losses` に読み替える。`candidate_total` / `base_seed` は
> **optional provenance**（省略時 graceful default）。全 case `tests/test_replay_freeze_validloss_ci.py`
> で verified（exit 78 + `MalformedEvalResultsError`）。

### Given（前提条件）

- `replay_freeze_validloss_ci.py <samples_file>` で
  - `<samples_file>` が存在しない or
  - JSON parse 失敗 or
  - 必須 key 欠落 or
  - 型不一致

### When（実行条件）

- 上記 4 つのいずれか case で replay を実行

### Then（期待結果）

- stderr に `MalformedEvalResultsError: <missing or invalid field>: <detail>` を出力
- exit code 78 で終了
- 必須 key 名・期待型・実際の値が detail に含まれる

### テストケース

#### 正常系

- [x] **TC-401-01**: 必須 key 欠落（`candidate_losses`）が exit 78 🔵
  - **入力**: `{}`（`candidate_losses` 欠落）
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: missing key: candidate_losses`
  - **信頼性**: 🔵 *REQ-401, REQ-402*（旧名 `candidate_total` → verified schema `candidate_losses`）
  - **検証**: `tests/test_replay_freeze_validloss_ci.py::test_rejects_missing_sample_keys`（`match="missing key: candidate_losses"`）

- [x] **TC-401-02**: 必須 key 欠落（`surrogate_losses`）が exit 78 🔵
  - **入力**: `candidate_losses` のみ存在・`surrogate_losses` 欠落
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: missing key: surrogate_losses`
  - **信頼性**: 🔵 *REQ-401*（旧名 `samples`/`valid_losses` → verified schema `surrogate_losses`）
  - **検証**: `load_samples` は `("candidate_losses", "surrogate_losses")` 順で検証（`scripts/replay_freeze_validloss_ci.py:156`）。後者欠落で `missing key: surrogate_losses`。

- [x] **TC-401-03**: `base_seed` は **optional provenance**（graceful default）🔵
  - **入力**: `base_seed` 欠落
  - **期待結果（verified reality）**: `MalformedEvalResultsError` は発火**しない** — `replay_samples` は `data.get("base_seed", 0)` で default `0` に落ち、replay 再現性を維持。旧 criterion の「`missing key: base_seed` で exit 78」は verified schema と不一致（`base_seed` は必須ではない）。
  - **信頼性**: 🔵 *REQ-401（reality reconciliation）*
  - **検証**: `tests/test_replay_freeze_validloss_ci.py::test_seed_defaults_to_recorded_base_seed`（default seed path）+ `load_samples` は `base_seed` を検証対象外。

- [x] **TC-401-04**: 型不一致（`candidate_losses` が string）が exit 78 🔵
  - **入力**: `{"candidate_losses": "not a list", ...}`
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: invalid type for candidate_losses: expected list, got str`
  - **信頼性**: 🔵 *REQ-402, EDGE-004*（旧名 `samples` → verified schema `candidate_losses`）
  - **検証**: `tests/test_replay_freeze_validloss_ci.py`（`{"candidate_losses": "not a list", ...}` → `"invalid type for candidate_losses: expected list, got str"` in stderr）

- [x] **TC-401-05**: JSON parse 失敗が exit 78 🔵
  - **入力**: `{invalid json` （brace mismatch）
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: json parse error: ...`
  - **信頼性**: 🔵 *REQ-401*
  - **検証**: `tests/test_replay_freeze_validloss_ci.py::test_rejects_malformed_json`（`match="json parse error"`）

#### 境界値

- [x] **TC-401-B01**: empty dict `{}` が `MalformedEvalResultsError` を発火 🔵
  - **入力**: `{}`
  - **期待結果**: `exit_code == 78`, `missing key: candidate_losses`（最初の必須 key が欠落）
  - **信頼性**: 🟡 *EDGE-003* → 🔵 *verified*（旧名 `candidate_total` → verified `candidate_losses`）
  - **検証**: `test_rejects_missing_sample_keys` と同経路（`{}` → 最初の必須 key `candidate_losses` 欠落）。

- [x] **TC-401-B02**: 空の samples list `[]` が `MalformedEvalResultsError` で fail-loud 🔵
  - **入力**: `{"candidate_losses": [], "surrogate_losses": [...]}`
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: empty list: candidate_losses`
  - **信頼性**: 🟡 *edge case* → 🔵 *verified*
  - **検証**: `tests/test_replay_freeze_validloss_ci.py::test_rejects_empty_sample_list`（`match="empty list: candidate_losses"`）

#### mutation 証明

- [x] **TC-401-M01**: `_raise_malformed_eval_results` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*
  - **検証**: `TestOperatorErrorAxisMutationProof::test_neutralized_wrapper_drops_detection[raise_malformed_eval_results]`

---

## REQ-501: `--json` mode（machine-parseable） 🟡

**信頼性**: 🟡 *既存 `replay_freeze_validloss_ci.py --json` pattern 整合*

### Given（前提条件）

- `--json` mode で replay を実行
- operator error 発生条件（missing key / 型不一致）

### When（実行条件）

- `python -m scripts.replay_freeze_validloss_ci.py <bad.json> --json` を実行

### Then（期待結果）

- stdout に `{"error": "<class 名>", "detail": "...", "exit_status": 78}` を 1 行で出力
- stderr は空
- exit code 78

### テストケース

#### 正常系

- [x] **TC-501-01**: `--json` mode で `MalformedEvalResultsError` が 1 行 JSON で stdout 出力 🟡
  - **入力**: bad samples + `--json`
  - **期待結果**: stdout == `{"error": "MalformedEvalResultsError", "detail": "...", "exit_status": 78}\n`（改行 1 個）
  - **信頼性**: 🟡 *REQ-501* → 🔵 *verified*
  - **検証**: replay `tests/test_replay_freeze_validloss_ci.py::...::test_json_mode_operator_error_is_single_line_stdout_stderr_empty`（assembled `main([bad, "--json"])` → 1 行 JSON stdout・`error`/`exit_status`/`detail` 検証）+ producer `test_json_mode_operator_error_stdout_single_line` + leaf `TestEmitter::test_emit_json_mode_*`（4 subtype 全て）

- [x] **TC-501-02**: `--json` mode で stderr が空 🟡
  - **入力**: bad samples + `--json`
  - **期待結果**: `stderr == ""`
  - **信頼性**: 🟡 *REQ-501* → 🔵 *verified*
  - **検証**: 上記 replay `--json` test（`err == ""`）+ producer `test_json_mode_operator_error_stdout_single_line` + leaf `TestEmitter::test_emit_json_mode_writes_stdout_single_line`

- [x] **TC-501-03**: `--json` mode で正常 path は既存 logic 通り 🟡
  - **入力**: good samples + `--json`
  - **期待結果**: 既存 replay JSON output が stdout に出力
  - **信頼性**: 🟡 *zero regression* → 🔵 *verified*
  - **検証**: `tests/test_replay_freeze_validloss_ci.py` の `--expected` 正常系 test 群 + `replay_to_json`（`main` の `args.json` 正常 path は operator-error try/except の外・zero regression）。

#### 境界値

- [x] **TC-501-B01**: stdout JSON は `\n` を含まない 🟡
  - **入力**: bad samples + `--json`
  - **期待結果**: `"\n" in stdout` が `False` （末尾改行除く）
  - **信頼性**: 🟡 *EDGE-102* → 🔵 *verified*
  - **検証**: 上記 replay `--json` test（`out.count("\n") == 1`）+ producer `test_json_mode_operator_error_stdout_single_line` + leaf `TestEmitter::test_emit_json_mode_writes_stdout_single_line`

---

## REQ-704: zero regression（既存 test cluster 保持） 🔵

**信頼性**: 🔵 *TASK-0178 で実証済 pattern*

### テストケース

- [x] **TC-704-01**: `tests/test_replay_freeze_validloss_ci.py` が 157+ passed（既存 + 新規）🔵
  - **期待結果**: 既存 test 全 green + 新規 test 群 green
  - **信頼性**: 🔵 *REQ-704*
  - **検証**: **176 passed**（TASK-0180 既存分 + 本 iter の `--json` operator-error test 追加）。157+ を充足。

- [x] **TC-704-02**: verdict-path cluster（replay + producer + ci + gate + launch + deposit + guards）が 537+ passed 🔵
  - **期待結果**: 既存 537 passed / 4 skipped を保持
  - **信頼性**: 🔵 *REQ-704*
  - **検証**: **682 passed / 4 skipped**（replay + producer + producer-operator + freeze_surrogate_ci + freeze_surrogate_gate + launch_honesty + src_static_guards + worker_launcher_exit_contract + config_launchability + cli_operator_errors + launch_freeze_ci_9b_full + form_deposit）。537+ を充足・4 skipped 不変。

- [x] **TC-704-03**: `tests/test_worker_launcher_exit_contract.py` が全 green 🔵
  - **期待結果**: 4 worker exit code (`EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME`) が不変
  - **信頼性**: 🔵 *REQ-702*
  - **検証**: 全 green（既存 4 worker exit code contract は byte-identical 維持・TASK-0182 で 78→FATAL の **5 個目** contract pin を table に **追加**＝既存 4 個は無変更）。

- [x] **TC-704-04**: `tests/test_freeze_ci_9b_launch_honesty.py` が全 green 🔵
  - **期待結果**: 5 launch-honesty invariants が不変
  - **信頼性**: 🔵 *REQ-703*
  - **検証**: 全 green（`b8ee35c` 5 launch-honesty invariants byte-identical・operator-error axis は deposit/honesty gate に触れない）。

- [x] **TC-704-05**: 既存 `argparse.error` exit code 2 が不変 🔵
  - **入力**: 不正 CLI 引数
  - **期待結果**: `exit_code == 2`
  - **信頼性**: 🔵 *REQ-701*
  - **検証**: `tests/test_run_freeze_validloss_ci_9b_producer_operator_errors.py::...::test_argparse_error_exit_code_unchanged`（`parse_args` は operator-error try/except の **外** で exit 2 不変）。

- [x] **TC-704-06**: 既存 `--expected` 不一致 error（exit 2）が不変 🔵
  - **入力**: `replay_freeze_validloss_ci.py <good.json> --expected WRONG`
  - **期待結果**: `exit_code == 2`
  - **信頼性**: 🔵 *REQ-602*
  - **検証**: `tests/test_replay_freeze_validloss_ci.py`（`main([FIXTURE_NEGCTRL, "--expected", TIES]) == 2`・operator-error 78 とは独立の argparse exit 2 path・zero regression）。

---

## 非機能要件テスト

### NFR-101: mutation-proof 🔵

**信頼性**: 🔵 *TASK-0178 mutation 証明 pattern 整合*

- [x] **TC-NFR-101-01**: 4 subtype 全ての helper を `pass` / `return None` で neutralize → 対応する detection test が RED 🔵
  - **測定項目**: test runner の exit code
  - **目標値**: 1 個以上の test が RED
  - **測定条件**: 4 helper × 各 neutralize pattern
  - **信頼性**: 🔵 *NFR-101*
  - **検証**: `tests/test_cli_operator_errors.py::TestOperatorErrorAxisMutationProof::test_neutralized_wrapper_drops_detection`（4 wrapper parametrized・neutralize → subtype 非 raise を pin）+ `TestWrapperMutation`（各 wrapper の `pytest.raises` positive pin・neutralize で DID NOT RAISE）。launcher 側は 78 分岐 neutralize → 4 test RED（`TestLauncherExit78Classification` 3 + contract `test_classify_routes_each_code_to_its_documented_action` 1）・他 31 GREEN を実証済。

- [x] **TC-NFR-101-02**: 4 subtype 全ての helper を neutralize しても invariant test（正常 deposit では発火しない等）は GREEN 🔵
  - **測定項目**: invariant test の exit code
  - **目標値**: 0 RED
  - **信頼性**: 🔵 *NFR-101*
  - **検証**: launcher 78 分岐 neutralize 時・既存 4 worker exit code contract + signal-kill RETRY + tempfail 等の **31 test が GREEN** を実証（mutation proof 実行時）。leaf neutralize は wrapper のみ・既存 verdict/honesty cluster は無影響。

### NFR-201: 1 回で修正可能 message 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK operator-facing follow-up*

- [x] **TC-NFR-201-01**: 4 subtype 全ての message が path または class 名 または field 名を含む 🔵
  - **測定項目**: stderr message の substring check
  - **目標値**: 4 subtype × 各 message に該当 substring が含まれる
  - **信頼性**: 🔵 *NFR-201*
  - **検証**: MissingConfig（path）/ MalformedYAML（path + line/column）/ AppConfigValidation（class 名 + field 名 + N）/ MalformedEvalResults（key 名・期待型・実際型）— 各 wrapper test + entrypoint integration test で substring を pin。

- [x] **TC-NFR-201-02**: 4 subtype 全ての message が 120 文字以内 🟡
  - **測定項目**: `len(stderr message) <= 120`
  - **目標値**: 全 subtype で 120 文字以下
  - **信頼性**: 🟡 *NFR-203* → 🔵 *verified*
  - **検証**: `tests/test_cli_operator_errors.py::TestEmitter::test_message_under_120_chars`（4 subtype parametrized・代表 long-path message で `<= 120`）。

### NFR-202: grep 抽出可能 🔵

**信頼性**: 🔵 *CI log 互換*

- [x] **TC-NFR-202-01**: stderr message の prefix が class 名と一致 🔵
  - **測定項目**: `stderr.startswith(class_name + ": ")`
  - **目標値**: 4 subtype 全て True
  - **信頼性**: 🔵 *NFR-202*
  - **検証**: leaf `TestEmitter::test_emit_human_starts_with_class_name`（4 subtype parametrized）+ leaf `TestOperatorErrorHierarchy::test_str_format_class_prefix` + replay `test_operator_error_stderr_starts_with_class_name`（parametrized 3 case）+ producer `test_operator_error_stderr_has_class_name_line`。

- [x] **TC-NFR-202-02**: stderr に ANSI escape code が含まれない 🟡
  - **測定項目**: `"\x1b[" not in stderr`
  - **目標値**: 4 subtype 全て True
  - **信頼性**: 🟡 *EDGE-103* → 🔵 *verified*
  - **検証**: leaf `TestEmitter::test_emit_no_ansi_codes`（4 subtype × human/json 両 mode）+ producer `test_operator_error_no_ansi_codes`。

---

## Edgeケーステスト

### EDGE-002: empty YAML file 🟡

- [x] **TC-EDGE-002-01**: 0-byte config file が `MalformedYAMLError` 🔵
  - **条件**: 0-byte file
  - **期待結果**: `MalformedYAMLError: yaml parse error in <path>: file is empty`
  - **信頼性**: 🟡 *EDGE-002* → 🔵 *verified*
  - **検証**: `tests/test_run_freeze_validloss_ci_9b_producer_operator_errors.py::TestLoadCfgConversion::test_empty_file_raises_malformed_yaml`（OmegaConf は 0-byte を `{}` に parse するため `_load_cfg` が `stat().st_size` で明示検出）+ `TestProducerMainOperatorError::test_empty_config_exits_78`。

### EDGE-102: 1 行 JSON 🟡

- [x] **TC-EDGE-102-01**: `--json` mode の stdout に `\n` が含まれない（末尾改行除く）🟡
  - **条件**: bad samples + `--json`
  - **期待結果**: `stdout.count("\n") == 1`
  - **信頼性**: 🟡 *EDGE-102* → 🔵 *verified*
  - **検証**: replay `test_json_mode_operator_error_is_single_line_stdout_stderr_empty`（`out.count("\n") == 1`）+ producer `test_json_mode_operator_error_stdout_single_line` + leaf `TestEmitter::test_emit_json_mode_writes_stdout_single_line`。

---

## テストケースサマリー

> **【2026-07-22 verified status】** 全 criteria 中 **46 [x] verified** / **0 [ ] blocked** — axis COMPLETE。
> TASK-0184 resolved the last 4 (TC-301-01/02/03/B01): producer `_load_cfg`
> が `config_schema.validate_config_data` で Pydantic validate し、`pydantic.ValidationError`
> → `raise_app_config_validation(exc.title, exc)` で exit-78 経路を新設。変換は producer
> level のみ（`config_schema` 自体は未改修・raw `ValidationError` を維持 → 既存
> `pytest.raises(ValidationError)` pin は不変 green）。launcher assembled mirror 経路
> （TC-301-02）は worker exit 78 → `classify_exit_code(78) → FATAL` → launcher 78。

### カテゴリ別件数

| カテゴリ | 正常系 | 異常系 | 境界値 | mutation | 合計 |
|---------|--------|--------|--------|----------|------|
| Error class taxonomy | 3 | 1 | 0 | 0 | 4 |
| Missing config | 3 | 0 | 1 | 1 | 5 |
| Malformed YAML | 3 | 0 | 1 | 1 | 5 |
| AppConfig validation | 4 | 0 | 1 | 1 | 6 |
| Malformed eval results | 5 | 0 | 2 | 1 | 8 |
| JSON mode | 3 | 0 | 1 | 0 | 4 |
| zero regression | 6 | 0 | 0 | 0 | 6 |
| NFR | 4 | 0 | 0 | 0 | 4 |
| Edge | 1 | 0 | 1 | 0 | 2 |
| **合計** | 32 | 1 | 7 | 4 | 44 |

### 信頼性レベル分布

- 🔵 青信号: 33件 (75%)
- 🟡 黄信号: 11件 (25%)
- 🔴 赤信号: 0件 (0%)

**品質評価**: 高品質

### 優先度別テストケース

- **Must Have**: 38件
- **Should Have**: 6件
- **Could Have**: 0件

---

## テスト実施計画

### Phase 1: 4 subtype + 4 entrypoint の単体 test

- TC-001-01..E01, TC-101-01..M01, TC-201-01..M01, TC-301-01..M01, TC-401-01..M01
- 優先度: Must Have
- 実施予定: TASK-0179 iter 内
- **状態**: green — 全 criteria verified（TC-301-01/02/03/B01 は TASK-0184 で producer `_load_cfg` Pydantic gate により [x] 化・上記 verified status 参照）

### Phase 2: 統合 / regression test

- TC-501-01..B01, TC-704-01..06
- 優先度: Must Have
- 実施予定: TASK-0179 iter 内
- **状態**: green（replay `--json` operator-error test 追加・verdict cluster 682 passed / 4 skipped）

### Phase 3: NFR / Edge test

- TC-NFR-101-01..02, TC-NFR-201-01..02, TC-NFR-202-01..02, TC-EDGE-002-01, TC-EDGE-102-01
- 優先度: Should Have
- 実施予定: TASK-0179 iter 内
- **状態**: green（axis-wide mutation proof + 4 subtype NFR/Edge 全 verified）


<!-- spine:references:begin -->
## Spine: external references

- [freeze-ci-operator-errors — コンテキストノート](note.md)
- [TASK-0179: freeze-ci-9b entrypoint 3 本が operator（開発者・CI）から投入される **4 種類の distinct な error class**（missing config / malformed YAML / AppConfig validation failure / malformed eval results）を それぞれ別個の `OperatorError` subtype + 別個の message + 共通 exit code 78（sysexits.h `EX_CONFIG` 由来）で fail-loud する leaf module `src/utils/cli_errors.py` を新設する — 直前 TASK-0175..0178 が replay-gate の stored boolean axis を 9 軸 bind 済で「stored vs artifact-rederived 真値」silent corruption を chokepoint 化したのに対し、本 TASK は「operator 入力 vs 内部 state」silent corruption の chokepoint を直交 axis として leaf 化する。AI_HUB_MAKE_RUN_FEEDBACK「operator-facing follow-up: implement distinct handling for missing config, malformed YAML, AppConfig validation failures, and malformed eval results with specified messages and exit statuses, then retain these tests as regressions」の **leaf 側 prereq**（直交 axis の核）](tasks/TASK-0179.md)
- [TASK-0183: TASK-0179（leaf）/ TASK-0180（replay wire-up）/ TASK-0181（producer wire-up）/ TASK-0182（launcher wire-up）が個別に pin した test 群を **統合 test layer** として consolidate し、operator-error axis 全体の **NFR-101 mutation 証明**（4 wrapper 個別 neutralize しても leaf invariant 群は GREEN）+ **zero-regression** 統合（既存 9 axis verdict honesty gate + 4 worker exit code + 5 launch-honesty + 32 config launchability + 8 producer smoke の **5 cluster** byte-identical）+ **NFR-201/202/203** message 品質 + **EDGE-102/103** boundary を **1 つの regression net として test_cluster を統合** する — 直前 4 TASK が個別に打った pin の **consolidation commit**（新 production code 変更なし・新 test 11 個追加・既存 test cluster 緑確認のみ・`b8ee35c` assembled launch-honesty dry-run pattern 整合）](tasks/TASK-0183.md)

<!-- spine:references:end -->
