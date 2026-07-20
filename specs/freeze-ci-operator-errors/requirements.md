# freeze-ci-operator-errors 要件定義書

<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**最終更新**: 2026-07-20（Phase 81 追加: operator-facing distinct error handling for freeze-ci-9b entrypoints — TASK-0175..0178 で成立した stored-boolean chokepoint の operator-side follow-up）

## 概要

freeze-ci-9b 系の 3 entrypoint script
（`scripts/replay_freeze_validloss_ci.py` /
 `scripts/run_freeze_validloss_ci_9b.py` /
 `scripts/launch_freeze_ci_9b_full.py`）が **4 種類の distinct な operator
 error class** に対して、**それぞれ別個の message と exit status** を
 提供しなければならない。

4 class:

1. **Missing config** — `--config X` で `X` が存在しない
2. **Malformed YAML** — `X` は存在するが YAML parse に失敗
3. **AppConfig validation failure** — YAML は parse できるが Pydantic スキーマ違反
4. **Malformed eval results** — `replay_freeze_validloss_ci.py samples.json` の `samples.json` が期待 schema と一致しない

各 class は独立した `OperatorError` subtype として表現され、stderr
（human mode）または stdout JSON（`--json` mode）で class-specific な
message + 新規 exit code `78`（`sysexits.h` `EX_CONFIG` 由来）で
fail-loud する。既存 `argparse.error` の exit code 2、
`ad8c84a` で pin 済の 4 worker exit code（`EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME`）、
`4afc5e9` の CUDA OOM path、はいずれも不変。

## 関連文書

- **分析記録**: [interview-record.md](interview-record.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **コンテキストノート**: [note.md](note.md)
- **正本**: [docs/GOAL.md](../../docs/GOAL.md) §7

## 機能要件（EARS記法）

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 既存実装・既存 test・AI_HUB_MAKE_RUN_FEEDBACK で直接支持される要件
- 🟡 **黄信号**: 既存 test pattern・既存 launch-honesty invariants から妥当な推測
- 🔴 **赤信号**: 参照資料にない自動推定

### 通常要件

#### Error class taxonomy

- REQ-001: システムは operator error を次の 4 subtype として表現しなければならない:
  `MissingConfigError` / `MalformedYAMLError` / `AppConfigValidationError` /
  `MalformedEvalResultsError`（全て `OperatorError` の subtype）🔵
  *AI_HUB_MAKE_RUN_FEEDBACK「distinct handling for missing config, malformed
  YAML, AppConfig validation failures, and malformed eval results」*
- REQ-002: `OperatorError` は基底 `Exception` であり、`__str__()` で class 名 +
  人間可読 detail を返さなければならない。`to_dict()` は `{"error": class 名,
  "detail": str, "exit_status": 78}` を返さなければならない 🔵
  *sysexits.h EX_CONFIG=78 規約・既存 `to_json()` pattern 整合*
- REQ-003: システムは operator error 発生時、stderr（human mode）または
  stdout JSON（`--json` mode）に class-specific な message を出力し、
  exit code 78 で終了しなければならない 🔵
  *AI_HUB_MAKE_RUN_FEEDBACK「specified messages and exit statuses」*

#### Missing config（class 1）

- REQ-101: `--config` 引数で指定された path が存在しない場合、システムは
  `MissingConfigError("config not found: <path>")` を発生させ、exit code
  78 で終了しなければならない。stderr 出力は `MissingConfigError: config
  not found: <path>` でなければならない 🔵
  *AI_HUB_MAKE_RUN_FEEDBACK・`FileNotFoundError` の wrapper 化*
- REQ-102: `--samples-file` 引数で指定された path が存在しない場合も
  `MissingConfigError` として扱わなければならない（`replay` script 固有）🔵
  *AI_HUB_MAKE_RUN_FEEDBACK の "malformed eval results" とは別軸だが
  「file がない」は同 class*

#### Malformed YAML（class 2）

- REQ-201: `--config` 引数で指定された path が存在するが YAML parse に
  失敗した場合、システムは `MalformedYAMLError("yaml parse error in <path>:
  <parser message>")` を発生させ、exit code 78 で終了しなければならない
  🔵 *AI_HUB_MAKE_RUN_FEEDBACK・`yaml.YAMLError` の wrapper 化*
- REQ-202: `MalformedYAMLError` の detail は `yaml.YAMLError` の
  `__str__()` 結果（line / column info 含む）を含まなければならない 🔵
  *PyYAML 標準 error message の保持*

#### AppConfig validation failure（class 3）

- REQ-301: YAML は parse できるが Pydantic スキーマに違反する場合
  （必須 field 欠落・型不一致・`extra="forbid"` 違反）は
  `AppConfigValidationError("schema validation failed for <config_class>:
  <error_count> errors; first: <pydantic error>")` を発生させ、exit
  code 78 で終了しなければならない 🔵
  *AI_HUB_MAKE_RUN_FEEDBACK・`pydantic.ValidationError` の wrapper 化*
- REQ-302: `AppConfigValidationError` の detail は Pydantic の
  `errors()` 出力（loc / msg / type の最初の 1 件）を含まなければ
  ならない。`error_count` は `len(errors())` と同値でなければならない 🔵
  *Pydantic v2 標準 error schema*
- REQ-303: `AppConfigValidationError` は `BaselineConfig` /
  `TGLoRAConfig` のいずれの schema 違反でも発火しなければならない
  （`load_and_validate_config` の dispatch 結果に依存）🔵
  *既存 `test_config_launchability_gate.py` のスキーマ範囲*

#### Malformed eval results（class 4）

- REQ-401: `replay_freeze_validloss_ci.py <samples_file>` で
  `<samples_file>` が JSON として parse できない、または必須 key
  （`candidate_total` / `surrogate_total` /
  `samples` または legacy `valid_losses` /
  `base_seed` 等）が欠落している場合、システムは
  `MalformedEvalResultsError("<missing or invalid field>: <detail>")`
  を発生させ、exit code 78 で終了しなければならない 🔵
  *AI_HUB_MAKE_RUN_FEEDBACK・`load_samples()` の schema 強化*
- REQ-402: `MalformedEvalResultsError` は欠落 field 名の特定を含む
  メッセージでなければならない（"missing key: candidate_total" /
  "invalid type for samples: expected list, got str" 等）🔵
  *operator が即座に修正できる message*

### 条件付き要件

- REQ-501: `--json` mode が指定されている operator error は、stdout に
  `{"error": class 名, "detail": "...", "exit_status": 78}` を 1 行
  JSON で出力し、stderr は空でなければならない 🟡
  *既存 `--json` mode pattern 整合（replay のみ・`--json` flag なし）*
- REQ-502: 3 entrypoint 全てに `--json` mode が渡された場合
  （producer / launcher 側で新規対応）、REQ-501 と同じ contract
  で fail-loud しなければならない 🟡
  *test 用に producer/launcher にも `--json` 拡張*

### 状態要件

- REQ-601: operator error handling は **他の正常 path 実行中に例外が
  発生した場合のみ** 発火し、program 起動時の引数 parse 段階
  （`argparse.error`）では発火してはならない 🟡
  *既存 `argparse.error` の exit code 2 を保持*
- REQ-602: 既存 `replay_freeze_validloss_ci.py` の `--expected` 不一致
  error（exit code 2）は operator error とは別 path として維持し、
  本要件の operator error には含めない 🟡
  *TASK-0178 で pin 済の verdict / honesty gate は scope 外*

### 制約要件

- REQ-701: 既存 `argparse.error` の exit code 2 は不変でなければならない 🔵
  *既存 test 群が `assert exit_code == 2` で pin している*
- REQ-702: 既存 `ad8c84a` で pin された worker exit code
  `EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME` は不変でなければならない 🔵
  *`tests/test_worker_launcher_exit_contract.py` 整合*
- REQ-703: 既存 CUDA OOM path（`4afc5e9`）の distinct な
  handling は不変でなければならない 🔵
  *`test_freeze_ci_9b_launch_honesty.py` 整合*
- REQ-704: 既存 `test_replay_freeze_validloss_ci.py` 157 passed
  + verdict-path cluster 537 passed / 4 skipped を保持しなければならない 🔵
  *zero regression*
- REQ-705: 新規 exit code 78 は `sysexits.h` `EX_CONFIG`（configuration
  file error）由来とし、operator が `man sysexits` で意味を引ける値で
  なければならない 🟡
  *POSIX 規約*

### オプション要件

- REQ-301a: 任意で `OperatorError.to_dict()` の `exit_status` を
  instance 生成時に override できるようにする（class ごとに固定値
  78 で構わないが、将来 class 別 exit code 追加に備えた hook）🔴
  *将来拡張・本 TASK では未使用*

## 非機能要件

### ユーザビリティ

- NFR-201: operator error message は **非技術的 operator でも 1 回目で
  修正可能** なレベルでなければならない（field 名・行番号・期待型
  を含む）🔵
  *AI_HUB_MAKE_RUN_FEEDBACK・operator-facing follow-up の本質*
- NFR-202: operator error message は **CI ログの grep** で抽出
  しやすい形式でなければならない（class 名 prefix を含む）🔵
  *machine-parseable*
- NFR-203: operator error message は **path・class 名・欠落 field 名**
  の 3 つを含む場合、**120 文字以内** に収まらなければならない 🟡
  *terminal width 想定*

### 信頼性

- NFR-101: 4 subtype 全てが **mutation-proof** な test で守られなければ
  ならない。helper を `pass` 化 / `return None` 化した mutation で
  detection test が RED になること 🔵
  *既存 test pattern（TASK-0178 の `_passes_stale` mutation 証明）整合*
- NFR-102: 4 subtype 全てが **既存 test cluster の zero regression** を
  示さなければならない 🔵
  *`b8ee35c` assembled dry-run pattern 整合*

### 保守性

- NFR-301: 4 subtype は `src/utils/cli_errors.py`（または
  `scripts/_cli_errors.py`）に集約し、3 entrypoint から import する
  形でなければならない 🔵
  *DRY・`atomic_save.py` / `checkpoint_integrity.py` の leaf 化 pattern*
- NFR-302: 4 subtype の `__str__` / `to_dict` の message format は
  frozen された（test で pin される）文字列でなければならない 🟡
  *operator が log を grep する想定*

## Edgeケース

### エラー処理

- EDGE-001: `--config` が directory（file でない）だった場合、
  `MissingConfigError`（"config not found: <path> (is a directory)"）
  を発火しなければならない 🟡
  *path validation の最低限*
- EDGE-002: `--config` が empty file だった場合、
  `MalformedYAMLError`（"yaml parse error in <path>: file is empty"）
  を発火しなければならない 🟡
  *PyYAML の `YAMLError` message 整合*
- EDGE-003: `--samples-file` が valid JSON だが `{}`（空 dict）
  だった場合、`MalformedEvalResultsError`（"missing key:
  candidate_total"）を発火しなければならない 🟡
  *必須 key 欠落の代表 case*
- EDGE-004: `--samples-file` が valid JSON だが `samples` field が
  string だった場合、`MalformedEvalResultsError`（"invalid type for
  samples: expected list, got str"）を発火しなければならない 🟡
  *型不一致の代表 case*

### 境界値

- EDGE-101: `MalformedEvalResultsError` の detail message に **PII
  候補 token**（API key 風文字列）が混入しないこと（operator error
  message は opaque payload ではなく schema 違反箇所のみ出力）🔴
  *将来拡張・本 TASK では最低限の sanity check のみ*
- EDGE-102: `--json` mode で operator error 発生時、stdout の JSON
  は **必ず 1 行** で出力しなければならない（`\n` を含まない）🔵
  *`json.dumps(..., indent=2)` ではなく `json.dumps(...)`*
- EDGE-103: stderr（human mode）の operator error message は **ANSI
  color code を含まない** こと 🟡
  *CI log 互換*

## 受け入れ基準サマリ

| ID | カテゴリ | 概要 | 信頼性 |
|----|----------|------|--------|
| REQ-001 | Error class taxonomy | 4 subtype 定義 | 🔵 |
| REQ-002 | Error class taxonomy | `OperatorError` 基底仕様 | 🔵 |
| REQ-003 | Error class taxonomy | exit code 78 | 🔵 |
| REQ-101 | Missing config | `--config` 欠落 | 🔵 |
| REQ-102 | Missing config | `--samples-file` 欠落 | 🔵 |
| REQ-201 | Malformed YAML | YAML parse error | 🔵 |
| REQ-202 | Malformed YAML | parser message 保持 | 🔵 |
| REQ-301 | AppConfig validation | Pydantic 違反 | 🔵 |
| REQ-302 | AppConfig validation | error_count + first error | 🔵 |
| REQ-303 | AppConfig validation | Baseline/TGLoRA 両方 | 🔵 |
| REQ-401 | Malformed eval results | schema 違反 | 🔵 |
| REQ-402 | Malformed eval results | field 名の特定 | 🔵 |
| REQ-501 | JSON mode | stdout 1 行 JSON | 🟡 |
| REQ-502 | JSON mode | producer/launcher 対応 | 🟡 |
| REQ-601 | 状態 | 起動時引数 parse は scope 外 | 🟡 |
| REQ-602 | 状態 | `--expected` 不一致は scope 外 | 🟡 |
| REQ-701 | 制約 | argparse.error 不変 | 🔵 |
| REQ-702 | 制約 | 4 worker exit code 不変 | 🔵 |
| REQ-703 | 制約 | CUDA OOM path 不変 | 🔵 |
| REQ-704 | 制約 | zero regression | 🔵 |
| REQ-705 | 制約 | exit code 78 = EX_CONFIG | 🟡 |
| REQ-301a | オプション | exit_status override hook | 🔴 |
| NFR-201 | ユーザビリティ | 1 回で修正可能 message | 🔵 |
| NFR-202 | ユーザビリティ | grep 抽出可能 | 🔵 |
| NFR-203 | ユーザビリティ | 120 文字以内 | 🟡 |
| NFR-101 | 信頼性 | mutation-proof | 🔵 |
| NFR-102 | 信頼性 | zero regression | 🔵 |
| NFR-301 | 保守性 | leaf module 集約 | 🔵 |
| NFR-302 | 保守性 | message format pin | 🟡 |
| EDGE-001 | エラー | directory path | 🟡 |
| EDGE-002 | エラー | empty file | 🟡 |
| EDGE-003 | エラー | empty dict | 🟡 |
| EDGE-004 | エラー | type mismatch | 🟡 |
| EDGE-101 | 境界 | PII 非混入 | 🔴 |
| EDGE-102 | 境界 | 1 行 JSON | 🔵 |
| EDGE-103 | 境界 | ANSI color なし | 🟡 |
