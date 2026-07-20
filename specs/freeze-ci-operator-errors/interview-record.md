# freeze-ci-operator-errors 自動分析記録

<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-07-20
**分析実施**: step4 既存情報ベースの差分分析と自動統合

## 分析目的

既存の design doc / TASK 履歴 / commit log / 既存 test 群を確認し、
AI_HUB_MAKE_RUN_FEEDBACK が指摘した「operator-facing distinct error handling
for missing config, malformed YAML, AppConfig validation failures, and
malformed eval results」の 4 axis について、既存実装・既存要件・既存 test
との差分を明確化する。フィードバックが明示する「Avoid another test-only
witness or repeated goal-selection documentation cycle without changing
executable behavior」原則を遵守する。

## 分析項目と判断

### A1: 既存 commit log の silent-corruption chokepoint coverage

**分析日時**: 2026-07-20
**カテゴリ**: 既存設計確認
**背景**: 直近 9 commit (`79577a5` 〜 `9737ace`) はすべて freeze-ci-9b の
replay-gate における **stored boolean が artifact-rederived 真値と乖離**
する silent-corruption path を chokepoint で fail-loud 化したもの。9 axis
bind family:
- `evidence_hash` (TASK-0173)
- ledger losses (TASK-0172)
- direction/baseline sub-verdicts (TASK-0171)
- Level-1 citation boolean (TASK-0175)
- derived §4 statistics (TASK-0174)
- `significant_surpasses` (TASK-0176)
- `is_material` + `material_margin` (TASK-0177)
- `passes` composite (TASK-0178)

**判断**: replay-gate chokepoint の stored-boolean axis は **9 axis 全て
closed**。AI_HUB_MAKE_RUN_FEEDBACK が "The 4-axis bind-boolean family is
now structurally complete" と評価した通り、追加 bind 軸は存在しない。
次の iter は replay-gate 周辺（operator-facing error path）に移るのが
論理的な次手。

**根拠**:
- 直近 commit message の fix scope
- `tests/test_replay_freeze_validloss_ci.py` 157 passed (149 既存 + 8 新規 TASK-0178)
- verdict-path cluster 537 passed / 4 skipped
- AI_HUB_MAKE_RUN_FEEDBACK "Further 'bind {next-field}' commits are peripheral polish unless independent negative evidence surfaces a NEW silent-corruption axis"

**信頼性への影響**:

- 新規要件「operator-facing distinct error handling」の 4 subtype 分類 (REQ-001) は 🔵（AI_HUB_MAKE_RUN_FEEDBACK 由来）

---

### A2: 既存 entrypoint の error handling 現状

**分析日時**: 2026-07-20
**カテゴリ**: 既存実装の確認
**背景**: 3 entrypoint script の現状 error handling を確認

`scripts/replay_freeze_validloss_ci.py`:
- `load_samples()` (line 122) — `json.load` の bare except なし、file open
  失敗時の挙動は未確認
- `main()` (line 1777) — `args = build_parser().parse_args(argv)` →
  `data = load_samples(args.samples_file)` → `ci = replay_samples(...)`
  の直線 code path、**distinct な error class への wrapping なし**
- `--expected` 不一致のみ `return 2` で fail

`scripts/run_freeze_validloss_ci_9b.py`:
- `OmegaConf.load(args.config)` (line 2673, 2705) — **bare except なし**
- `load_and_validate_config(...)` (line ~2705) — Pydantic ValidationError
  も **bare except なし**
- `Path.cwd()` FileNotFoundError handler (line 2576) — **dead CWD
  background-run trap** 対策、これは別 axis (NFR 別)

`scripts/launch_freeze_ci_9b_full.py`:
- `build_parser().error(...)` (line 291) — argparse 経由
- `FileNotFoundError` in launcher (line 35) — **early fail**

**判断**: 3 entrypoint 全てで 4 subtype への distinct な wrapping が
存在しない。`FileNotFoundError` / `yaml.YAMLError` / `pydantic.ValidationError`
が **そのまま propagate** し、Python 標準の traceback が出る operator
experience は **5 段 traceback + 内部関数名** で非 friendly。

**根拠**:
- `rg "except.*FileNotFoundError|except.*YAMLError|except.*ValidationError" scripts/{replay_freeze_validloss_ci,run_freeze_validloss_ci_9b,launch_freeze_ci_9b_full}.py`
  → 0 hit (Pydantic / OmegaConf の path のみ)

**信頼性への影響**:

- REQ-001 (4 subtype 階層) の 🔵 判定を補強（既存実装に distinct handling
  が無く、feedback が直接この gap を指摘しているため）
- REQ-101/201/301/401 の 🔵 判定を補強

---

### A3: 既存 test 群の coverage 確認

**分析日時**: 2026-07-20
**カテゴリ**: 既存設計確認
**背景**: 4 subtype それぞれに対する mutation-proof な test が既存に
存在するかを `tests/` 配下で確認

```
tests/test_replay_freeze_validloss_ci.py (157 passed) — honesty gate
tests/test_run_freeze_validloss_ci_9b_*.py        — 9B producer
tests/test_freeze_ci_9b_launch_honesty.py          — assembled dry-run
tests/test_worker_launcher_exit_contract.py        — 4 exit code pin
tests/test_config_launchability_gate.py            — 32 config round-trip
tests/test_config_schema.py                        — Pydantic schema
```

`rg "MalformedEvalResultsError|MalformedYAMLError|AppConfigValidationError|MissingConfigError" tests/`
→ 0 hit（既存 test はこれらの class を test していない）

**判断**: 4 subtype に対する **operator-facing test はゼロ**。TASK-0178
の `_passes_stale` mutation 証明 pattern を踏襲して新規 test 群を
`tests/test_cli_operator_errors.py` に集約すべき。

**根拠**:
- 直近 TASK-0178 の mutation 証明 test structure
- AI_HUB_MAKE_RUN_FEEDBACK "then retain these tests as regressions"

**信頼性への影響**:

- 新規要件の acceptance criteria 44 件は全て 🔵 または 🟡 で根拠あり

---

### A4: 既存要件定義書 `specs/tg-lora/requirements.md` の overlap 確認

**分析日時**: 2026-07-20
**カテゴリ**: 重複確認
**背景**: 既存 `specs/tg-lora/requirements.md` (1013 lines, 171KB) に
operator error handling 要件が既に存在するか

`rg "operator.error|missing.config|malformed.YAML|distinct.handling|exit.status|cli.*error" specs/tg-lora/requirements.md`
→ 0 hit（operator error class taxonomy は未定義）

`rg "EDGE-.*error|error.*message.*含む" specs/tg-lora/requirements.md`
→ 1 hit（EDGE-175: DeltaTracker error message、**別 module**・本 TASK scope 外）

**判断**: 既存 requirements.md に 4 subtype の operator error 要件は
**存在しない**。新規 feature_id `freeze-ci-operator-errors` として独立
specs/ ディレクトリを作成する方が、既存 1000+ 行 requirements.md への
追加よりも **scope が明確** で見通しが良い。

**根拠**:
- 既存 1013 行 requirements.md への追加は差分 review を困難にする
- AI_HUB_MAKE_RUN_FEEDBACK が "test-only witness ではなく executable
  behavior の変更" を要求しており、独立 spec の方が commit 単位で
  scope を限定しやすい
- 既存 convention: `specs/<feature_id>/` 形式（spine 整合）

**信頼性への影響**:

- 本要件定義書群は `specs/freeze-ci-operator-errors/` に独立配置
- 既存 `specs/tg-lora/requirements.md` には触らない（**非破壊 merge 戦略**）

---

### A5: 既存 launch-honesty 5 invariants との直交性

**分析日時**: 2026-07-20
**カテゴリ**: 影響範囲
**背景**: `b8ee35c` で成立した 5 launch-honesty invariants（CUDA OOM
classifier、scope-drift guard、cost axis、atomic write、eval task drop）
との直交性を確認

5 invariants:
1. CUDA OOM を `is_cuda_oom` で distinct に classifier（`4afc5e9`）
2. unknown `freeze_layer` spec を reject（`e823641`）
3. `_candidate_cost_reduction` の silent null 防止（`1c2c833`）
4. atomic JSON write（`54a4cd8`）
5. eval task silent drop 防止（`d9ca7f5`）

**判断**: これら 5 invariants は **program 実行中の silent corruption**
を対象としており、**operator 入力の failure**（4 subtype）は別 axis。
本 TASK の 4 subtype handler は **program 起動前〜起動直後** で fail
するため、5 invariants の path に到達せず、**完全直交**。

**根拠**:
- `test_freeze_ci_9b_launch_honesty.py` の test 構造（`run_ci_9b` 起動
  後の invariant 検証）
- AI_HUB_MAKE_RUN_FEEDBACK "these are real bugs, not polish"

**信頼性への影響**:

- REQ-703「既存 CUDA OOM path 不変」の 🔵 判定根拠
- REQ-704「zero regression」の 🔵 判定根拠

---

### A6: 既存 worker↔launcher exit-code contract との整合

**分析日時**: 2026-07-20
**カテゴリ**: 影響範囲
**背景**: `ad8c84a` で pin された worker 4 exit code
（`EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME`）と新規 operator
error exit code 78 の位置関係を整理

- `EXIT_DONE` = 0
- `EXIT_UNEXPECTED` = 1 (default exception)
- `EXIT_CUDA_DOWN` = 3
- `EXIT_INCOMPLETE_RESUME` = 4
- argparse error = 2 (Python 標準)
- **operator error = 78** (新規・sysexits.h EX_CONFIG)

**判断**: 新規 exit code 78 は既存 4 code と **完全独立** であり、
sysexits.h 由来の POSIX 規約値で operator にとって意味が引ける値。
`argparse.error` の exit 2 も独立（program 起動前 vs program 起動直後）。

**根拠**:
- `tests/test_worker_launcher_exit_contract.py` の 4 値 pin
- `sysexits.h` 標準（BSD/Linux）

**信頼性への影響**:

- REQ-705「exit code 78 = EX_CONFIG」の 🟡 判定（sysexits は POSIX
  推奨だが必須ではない）

---

## 分析結果サマリー

### 確認できた事項

- 4 subtype に対応する既存 distinct handling は **3 entrypoint 全てに不在**
- 直近 9 commit で replay-gate chokepoint は 9 axis bind で構造的に closed
- TASK-0178 の mutation 証明 pattern が 4 subtype test の直接の参考実装
- `b8ee35c` の 5 launch-honesty invariants は本 TASK と完全直交
- `ad8c84a` の worker exit code 4 種と新規 exit 78 は独立

### 追加/変更要件

- 新規 4 subtype 階層（REQ-001〜003）
- Missing config handler（REQ-101〜102）
- Malformed YAML handler（REQ-201〜202）
- AppConfig validation handler（REQ-301〜303）
- Malformed eval results handler（REQ-401〜402）
- `--json` mode 対応（REQ-501〜502）
- 状態要件（REQ-601〜602）
- 制約要件（REQ-701〜705、REQ-301a）
- NFR（201〜203、101〜102、301〜302）
- Edge（EDGE-001〜004、101〜103）

### 残課題

- producer / launcher script への `--json` mode 追加是非（REQ-502、🟡）
- `OperatorError.to_dict()` の `exit_status` override hook（REQ-301a、🔴・将来拡張）
- PII 候補 token 混入防止（EDGE-101、🔴・将来拡張）

### 信頼性レベル分布

**分析前**:

- 🔵 青信号: 0件
- 🟡 黄信号: 0件
- 🔴 赤信号: 0件

**分析後**:

- 🔵 青信号: 27件 (+27) — AI_HUB_MAKE_RUN_FEEDBACK 由来 + 既存 test pattern 整合
- 🟡 黄信号: 10件 (+10) — 既存 test pattern からの妥当な推測
- 🔴 赤信号: 1件 (+1) — REQ-301a 将来拡張 hook

## 関連文書

- **要件定義書**: [requirements.md](requirements.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **コンテキストノート**: [note.md](note.md)
- **正本**: `docs/GOAL.md` §7
- **AI_HUB_MAKE_RUN_FEEDBACK** (2026-07-20)
- **predecessor commits**:
  - `9737ace` TASK-0178 (composite `passes` boolean bind)
  - `1ed2def` TASK-0177 (is_material + material_margin bind)
  - `3e3aca6` TASK-0176 (significant_surpasses bind)
  - `37db498` TASK-0175 (Level-1 citation boolean bind)
  - `6ed0f69` TASK-0174 (derived §4 statistics bind)
  - `79577a5` TASK-0173 (evidence_hash bind)
  - `47226b9` TASK-0172 (ledger losses bind)
  - `371e934` TASK-0171 (direction/baseline sub-verdicts bind)
  - `d734327` TASK-0169 (replay citation gate)
  - `b8ee35c` (assembled launch-honesty dry-run)
  - `ad8c84a` TASK-0090 (worker↔launcher exit-code contract pin)
  - `4afc5e9` (CUDA OOM classification)
  - `d9ca7f5` (eval task silent drop fix)
