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

- [ ] **TC-001-01**: 4 subtype が import 可能 🔵
  - **入力**: `from src.utils.cli_errors import MissingConfigError, MalformedYAMLError, AppConfigValidationError, MalformedEvalResultsError, OperatorError`
  - **期待結果**: 全て import 成功
  - **信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK*

- [ ] **TC-001-02**: 各 instance が `to_dict()` で同じ schema を返す 🔵
  - **入力**: 4 subtype それぞれの instance
  - **期待結果**: `{"error": <class 名>, "detail": <str>, "exit_status": 78}`
  - **信頼性**: 🔵 *REQ-002*

- [ ] **TC-001-03**: 各 instance が `isinstance(e, OperatorError)` で True 🔵
  - **入力**: 4 subtype それぞれの instance
  - **期待結果**: `True`
  - **信頼性**: 🔵 *REQ-001*

#### 異常系

- [ ] **TC-001-E01**: detail が空文字でも `to_dict()` が `"detail": ""` を返す 🔵
  - **入力**: `MissingConfigError("")`
  - **期待結果**: `to_dict() == {"error": "MissingConfigError", "detail": "", "exit_status": 78}`
  - **信頼性**: 🔵 *REQ-002 の `__str__` 整合*

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

- [ ] **TC-101-01**: producer で `--config /nonexistent.yaml` が exit 78 🔵
  - **入力**: producer script + non-existent path
  - **期待結果**: `exit_code == 78`, `stderr` に `MissingConfigError: config not found: /nonexistent.yaml`
  - **信頼性**: 🔵 *REQ-101*

- [ ] **TC-101-02**: launcher で `--config /nonexistent.yaml` が exit 78 🔵
  - **入力**: launcher script + non-existent path
  - **期待結果**: `exit_code == 78`, `stderr` に `MissingConfigError: config not found: /nonexistent.yaml`
  - **信頼性**: 🔵 *REQ-101*

- [ ] **TC-101-03**: replay で `--samples-file /nonexistent.json` が exit 78 🔵
  - **入力**: replay script + non-existent samples file
  - **期待結果**: `exit_code == 78`, `stderr` に `MissingConfigError: samples file not found: /nonexistent.json`
  - **信頼性**: 🔵 *REQ-102*

#### 境界値

- [ ] **TC-101-B01**: `--config` が directory path 🔵
  - **入力**: `/tmp` 等の directory path
  - **期待結果**: `exit_code == 78`, `MissingConfigError` (`is a directory` detail を含む)
  - **信頼性**: 🟡 *EDGE-001*

#### mutation 証明

- [ ] **TC-101-M01**: `_raise_missing_config` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*

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

- [ ] **TC-201-01**: producer で壊れた YAML が exit 78 🔵
  - **入力**: tab/space 混在 YAML file
  - **期待結果**: `exit_code == 78`, `stderr` に `MalformedYAMLError: yaml parse error in <path>: ...`
  - **信頼性**: 🔵 *REQ-201*

- [ ] **TC-201-02**: 行番号・列番号が stderr に含まれる 🔵
  - **入力**: 壊れた YAML
  - **期待結果**: stderr message に `line N, column M` が含まれる
  - **信頼性**: 🔵 *REQ-202*

- [ ] **TC-201-03**: launcher で壊れた YAML が exit 78 🔵
  - **入力**: launcher script + 壊れた YAML
  - **期待結果**: `exit_code == 78`, `MalformedYAMLError`
  - **信頼性**: 🔵 *REQ-201*

#### 境界値

- [ ] **TC-201-B01**: empty file が `MalformedYAMLError` を発火 🔵
  - **入力**: 0-byte file
  - **期待結果**: `MalformedYAMLError: yaml parse error in <path>: file is empty`
  - **信頼性**: 🟡 *EDGE-002*

#### mutation 証明

- [ ] **TC-201-M01**: `_raise_malformed_yaml` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*

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

- [ ] **TC-301-01**: producer で `extra="forbid"` 違反が exit 78 🔵
  - **入力**: `LoggingConfig` に未宣言 field を追加した YAML
  - **期待結果**: `exit_code == 78`, `AppConfigValidationError: schema validation failed for TGLoRAConfig: 1 errors; first: logging <field_name> extra fields not permitted (value_error.extra)`
  - **信頼性**: 🔵 *REQ-301, REQ-302, REQ-303*

- [ ] **TC-301-02**: launcher で必須 field 欠落が exit 78 🔵
  - **入力**: 必須 field を削除した YAML
  - **期待結果**: `exit_code == 78`, `AppConfigValidationError: schema validation failed for <ConfigClass>: ...; first: <field> field required (value_error.missing)`
  - **信頼性**: 🔵 *REQ-301*

- [ ] **TC-301-03**: producer で型不一致が exit 78 🔵
  - **入力**: 期待型 `int` だが `str` を代入した YAML
  - **期待結果**: `exit_code == 78`, `AppConfigValidationError: ...; first: <field> ... (type_error)`
  - **信頼性**: 🔵 *REQ-301*

- [ ] **TC-301-04**: `error_count` が Pydantic の `len(errors())` と一致 🔵
  - **入力**: 複数の field 違反
  - **期待結果**: stderr message の `N` が `len(pydantic.errors())` と一致
  - **信頼性**: 🔵 *REQ-302*

#### 境界値

- [ ] **TC-301-B01**: `BaselineConfig` 違反も `AppConfigValidationError` を発火 🔵
  - **入力**: `9b_baseline.yaml` 系の違反
  - **期待結果**: `BaselineConfig` class 名が stderr に出力
  - **信頼性**: 🔵 *REQ-303*

#### mutation 証明

- [ ] **TC-301-M01**: `_raise_app_config_validation` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*

---

## REQ-401: 破損 eval result（Malformed eval results） 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK + `load_samples()` schema 強化*

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

- [ ] **TC-401-01**: 必須 key 欠落（`candidate_total`）が exit 78 🔵
  - **入力**: `{}` または `{"surrogate_total": 100, ...}`（`candidate_total` 欠落）
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: missing key: candidate_total`
  - **信頼性**: 🔵 *REQ-401, REQ-402*

- [ ] **TC-401-02**: 必須 key 欠落（`samples` / `valid_losses`）が exit 78 🔵
  - **入力**: 必須 sample 配列欠落
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: missing key: samples`（or `valid_losses`・legacy 互換）
  - **信頼性**: 🔵 *REQ-401*

- [ ] **TC-401-03**: 必須 key 欠落（`base_seed`）が exit 78 🔵
  - **入力**: `base_seed` 欠落
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: missing key: base_seed`
  - **信頼性**: 🔵 *REQ-401*

- [ ] **TC-401-04**: 型不一致（`samples` が string）が exit 78 🔵
  - **入力**: `{"samples": "not a list", ...}`
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: invalid type for samples: expected list, got str`
  - **信頼性**: 🔵 *REQ-402, EDGE-004*

- [ ] **TC-401-05**: JSON parse 失敗が exit 78 🔵
  - **入力**: `{invalid json` （brace mismatch）
  - **期待結果**: `exit_code == 78`, `MalformedEvalResultsError: json parse error: ...`
  - **信頼性**: 🔵 *REQ-401*

#### 境界値

- [ ] **TC-401-B01**: empty dict `{}` が `MalformedEvalResultsError` を発火 🔵
  - **入力**: `{}`
  - **期待結果**: `exit_code == 78`, `missing key: candidate_total`（最初の必須 key が欠落）
  - **信頼性**: 🟡 *EDGE-003*

- [ ] **TC-401-B02**: 空の samples list `[]` が `MalformedEvalResultsError` または replay 失敗のいずれかで fail-loud 🔵
  - **入力**: `{"candidate_total": 100, "samples": [], ...}`
  - **期待結果**: `exit_code == 78` または replay 失敗（既存 logic 整合）
  - **信頼性**: 🟡 *edge case*

#### mutation 証明

- [ ] **TC-401-M01**: `_validate_eval_samples_schema` helper を `pass` で neutralize → detection test RED
  - **信頼性**: 🔵 *NFR-101 整合*

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

- [ ] **TC-501-01**: `--json` mode で `MalformedEvalResultsError` が 1 行 JSON で stdout 出力 🟡
  - **入力**: bad samples + `--json`
  - **期待結果**: stdout == `{"error": "MalformedEvalResultsError", "detail": "...", "exit_status": 78}\n`（改行 1 個）
  - **信頼性**: 🟡 *REQ-501*

- [ ] **TC-501-02**: `--json` mode で stderr が空 🟡
  - **入力**: bad samples + `--json`
  - **期待結果**: `stderr == ""`
  - **信頼性**: 🟡 *REQ-501*

- [ ] **TC-501-03**: `--json` mode で正常 path は既存 logic 通り 🟡
  - **入力**: good samples + `--json`
  - **期待結果**: 既存 replay JSON output が stdout に出力
  - **信頼性**: 🟡 *zero regression*

#### 境界値

- [ ] **TC-501-B01**: stdout JSON は `\n` を含まない 🟡
  - **入力**: bad samples + `--json`
  - **期待結果**: `"\n" in stdout` が `False` （末尾改行除く）
  - **信頼性**: 🟡 *EDGE-102*

---

## REQ-704: zero regression（既存 test cluster 保持） 🔵

**信頼性**: 🔵 *TASK-0178 で実証済 pattern*

### テストケース

- [ ] **TC-704-01**: `tests/test_replay_freeze_validloss_ci.py` が 157+ passed（既存 + 新規）🔵
  - **期待結果**: 既存 test 全 green + 新規 test 群 green
  - **信頼性**: 🔵 *REQ-704*

- [ ] **TC-704-02**: verdict-path cluster（replay + producer + ci + gate + launch + deposit + guards）が 537+ passed 🔵
  - **期待結果**: 既存 537 passed / 4 skipped を保持
  - **信頼性**: 🔵 *REQ-704*

- [ ] **TC-704-03**: `tests/test_worker_launcher_exit_contract.py` が全 green 🔵
  - **期待結果**: 4 worker exit code (`EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME`) が不変
  - **信頼性**: 🔵 *REQ-702*

- [ ] **TC-704-04**: `tests/test_freeze_ci_9b_launch_honesty.py` が全 green 🔵
  - **期待結果**: 5 launch-honesty invariants が不変
  - **信頼性**: 🔵 *REQ-703*

- [ ] **TC-704-05**: 既存 `argparse.error` exit code 2 が不変 🔵
  - **入力**: 不正 CLI 引数
  - **期待結果**: `exit_code == 2`
  - **信頼性**: 🔵 *REQ-701*

- [ ] **TC-704-06**: 既存 `--expected` 不一致 error（exit 2）が不変 🔵
  - **入力**: `replay_freeze_validloss_ci.py <good.json> --expected WRONG`
  - **期待結果**: `exit_code == 2`
  - **信頼性**: 🔵 *REQ-602*

---

## 非機能要件テスト

### NFR-101: mutation-proof 🔵

**信頼性**: 🔵 *TASK-0178 mutation 証明 pattern 整合*

- [ ] **TC-NFR-101-01**: 4 subtype 全ての helper を `pass` / `return None` で neutralize → 対応する detection test が RED 🔵
  - **測定項目**: test runner の exit code
  - **目標値**: 1 個以上の test が RED
  - **測定条件**: 4 helper × 各 neutralize pattern
  - **信頼性**: 🔵 *NFR-101*

- [ ] **TC-NFR-101-02**: 4 subtype 全ての helper を neutralize しても invariant test（正常 deposit では発火しない等）は GREEN 🔵
  - **測定項目**: invariant test の exit code
  - **目標値**: 0 RED
  - **信頼性**: 🔵 *NFR-101*

### NFR-201: 1 回で修正可能 message 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK operator-facing follow-up*

- [ ] **TC-NFR-201-01**: 4 subtype 全ての message が path または class 名 または field 名を含む 🔵
  - **測定項目**: stderr message の substring check
  - **目標値**: 4 subtype × 各 message に該当 substring が含まれる
  - **信頼性**: 🔵 *NFR-201*

- [ ] **TC-NFR-201-02**: 4 subtype 全ての message が 120 文字以内 🟡
  - **測定項目**: `len(stderr message) <= 120`
  - **目標値**: 全 subtype で 120 文字以下
  - **信頼性**: 🟡 *NFR-203*

### NFR-202: grep 抽出可能 🔵

**信頼性**: 🔵 *CI log 互換*

- [ ] **TC-NFR-202-01**: stderr message の prefix が class 名と一致 🔵
  - **測定項目**: `stderr.startswith(class_name + ": ")`
  - **目標値**: 4 subtype 全て True
  - **信頼性**: 🔵 *NFR-202*

- [ ] **TC-NFR-202-02**: stderr に ANSI escape code が含まれない 🟡
  - **測定項目**: `"\x1b[" not in stderr`
  - **目標値**: 4 subtype 全て True
  - **信頼性**: 🟡 *EDGE-103*

---

## Edgeケーステスト

### EDGE-002: empty YAML file 🟡

- [ ] **TC-EDGE-002-01**: 0-byte config file が `MalformedYAMLError` 🔵
  - **条件**: 0-byte file
  - **期待結果**: `MalformedYAMLError: yaml parse error in <path>: file is empty`
  - **信頼性**: 🟡 *EDGE-002*

### EDGE-102: 1 行 JSON 🟡

- [ ] **TC-EDGE-102-01**: `--json` mode の stdout に `\n` が含まれない（末尾改行除く）🟡
  - **条件**: bad samples + `--json`
  - **期待結果**: `stdout.count("\n") == 1`
  - **信頼性**: 🟡 *EDGE-102*

---

## テストケースサマリー

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

### Phase 2: 統合 / regression test

- TC-501-01..B01, TC-704-01..06
- 優先度: Must Have
- 実施予定: TASK-0179 iter 内

### Phase 3: NFR / Edge test

- TC-NFR-101-01..02, TC-NFR-201-01..02, TC-NFR-202-01..02, TC-EDGE-002-01, TC-EDGE-102-01
- 優先度: Should Have
- 実施予定: TASK-0179 iter 内


<!-- spine:references:begin -->
## Spine: external references

- [freeze-ci-operator-errors — コンテキストノート](note.md)
- [TASK-0179: freeze-ci-9b entrypoint 3 本が operator（開発者・CI）から投入される **4 種類の distinct な error class**（missing config / malformed YAML / AppConfig validation failure / malformed eval results）を それぞれ別個の `OperatorError` subtype + 別個の message + 共通 exit code 78（sysexits.h `EX_CONFIG` 由来）で fail-loud する leaf module `src/utils/cli_errors.py` を新設する — 直前 TASK-0175..0178 が replay-gate の stored boolean axis を 9 軸 bind 済で「stored vs artifact-rederived 真値」silent corruption を chokepoint 化したのに対し、本 TASK は「operator 入力 vs 内部 state」silent corruption の chokepoint を直交 axis として leaf 化する。AI_HUB_MAKE_RUN_FEEDBACK「operator-facing follow-up: implement distinct handling for missing config, malformed YAML, AppConfig validation failures, and malformed eval results with specified messages and exit statuses, then retain these tests as regressions」の **leaf 側 prereq**（直交 axis の核）](tasks/TASK-0179.md)
- [TASK-0183: TASK-0179（leaf）/ TASK-0180（replay wire-up）/ TASK-0181（producer wire-up）/ TASK-0182（launcher wire-up）が個別に pin した test 群を **統合 test layer** として consolidate し、operator-error axis 全体の **NFR-101 mutation 証明**（4 wrapper 個別 neutralize しても leaf invariant 群は GREEN）+ **zero-regression** 統合（既存 9 axis verdict honesty gate + 4 worker exit code + 5 launch-honesty + 32 config launchability + 8 producer smoke の **5 cluster** byte-identical）+ **NFR-201/202/203** message 品質 + **EDGE-102/103** boundary を **1 つの regression net として test_cluster を統合** する — 直前 4 TASK が個別に打った pin の **consolidation commit**（新 production code 変更なし・新 test 11 個追加・既存 test cluster 緑確認のみ・`b8ee35c` assembled launch-honesty dry-run pattern 整合）](tasks/TASK-0183.md)

<!-- spine:references:end -->
