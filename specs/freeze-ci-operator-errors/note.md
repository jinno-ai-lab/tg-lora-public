# freeze-ci-operator-errors — コンテキストノート

**作成日**: 2026-07-20
**関連要件定義**: [requirements.md](requirements.md)
**分析記録**: [interview-record.md](interview-record.md)

## 背景

GOAL §7「科学誠実性インフラ」の freeze-ci-9b 系の chokepoint（TASK-0175..0178 で
stored boolean を artifact-rederived 真値に bind 済）は **produces 側と
replay 側の双方で** operator（開発者・CI）が 4 種類の distinct な error
class を experience する：

1. **Missing config** — `python -m scripts.run_freeze_validloss_ci_9b --config X` で
   `X` が存在しない（`FileNotFoundError`）
2. **Malformed YAML** — `X` は存在するが `yaml.YAMLError`（壊れた YAML）
3. **AppConfig validation failure** — YAML は parse できるが Pydantic スキーマに
   合わない（必須 field 欠落・型不一致・`extra="forbid"` 違反）
4. **Malformed eval results** — `replay_freeze_validloss_ci.py samples.json` で
   `samples.json` が期待 schema と一致しない（必須 key 欠落・型不一致）

これら 4 つの class は現コードで **distinct な handler が無く**、
`argparse.error` / `OmegaConf.load` / `load_and_validate_config` /
`load_samples` / `json.JSONDecodeError` の混在で stderr に「結局何が
悪かったのか」が分からない。AI_HUB_MAKE_RUN_FEEDBACK が
"operator-facing follow-up: implement distinct handling for missing config,
malformed YAML, AppConfig validation failures, and malformed eval results
with specified messages and exit statuses, then retain these tests as
regressions" と明示した axis。

## 範囲

### 対象 entrypoint

- `scripts/replay_freeze_validloss_ci.py` — `load_samples()` の周辺（eval results）
- `scripts/run_freeze_validloss_ci_9b.py` — `OmegaConf.load()` / `load_and_validate_config()` / `pydantic.ValidationError` 周辺（missing config / malformed YAML / AppConfig validation）
- `scripts/launch_freeze_ci_9b_full.py` — 上流 launcher の error path 統合

### 非対象

- 9B target-scale の GPU 実行失敗（`is_cuda_oom` 系） — TASK-0094 / `4afc5e9` の範囲、既に対応済
- `replay_freeze_validloss_ci.py` の verdict / honesty gate（`passes` / `significant_surpasses` / `is_material` / `evidence_hash` / `ci_stats` / `citation_label` / `ledger_losses` / `citable_as_full_section4_verdict`） — TASK-0171..0178 範囲、既に対応済
- 既存の `argparse` の usage error — 既に `argparse.error` 経由で exit code 2 を返している
- private `src.data` 系の絶対 loss comparability — DATA axis、別 issue

## 関連コミット（先行 chokepoint）

- `9737ace` TASK-0178 — composite `passes` boolean bind
- `1ed2def` TASK-0177 — `is_material` + `material_margin` bind
- `3e3aca6` TASK-0176 — `significant_surpasses` bind
- `37db498` TASK-0175 — Level-1 citation boolean bind
- `6ed0f69` TASK-0174 — derived §4 statistics bind
- `79577a5` TASK-0173 — `evidence_hash` bind
- `47226b9` TASK-0172 — ledger losses bind
- `371e934` TASK-0171 — direction/baseline sub-verdicts bind
- `d734327` TASK-0169 — replay citation gate honors producer's 4 honesty axes
- `b8ee35c` — assembled launch-honesty dry-run (5 invariants end-to-end)
- `ad8c84a` TASK-0090 — worker↔launcher exit-code contract pin (4 codes)

## 設計上の重要ポイント

- **既存 test-only witness の蓄積禁止** — feedback が明示する "Avoid another
  test-only witness or repeated goal-selection documentation cycle without
  changing executable behavior" に対応する。tests ではなく **executable
  behavior** を変更する。
- **既存 launch-honesty 5 invariants との直交性** — `b8ee35c` の
  `test_freeze_ci_9b_launch_honesty.py` は GPU/CUDA OOM + 5 silent-corruption
  site の assembled 経路を fix している。本 TASK は operator error path の
  distinct handling を追加するもので、既存 5 invariants を壊さない。
- **既存 worker↔launcher exit-code contract** — `ad8c84a` は
  `EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME` の 4 コードを pin 済。
  operator error はこれらに**追加**する（破壊しない）。新規 exit code は
  `EXIT_OPERATOR_ERROR = 78`（sysexits.h `EX_CONFIG` 由来）を採用。
- **`extra="forbid"` スキーマとの整合** — `test_config_launchability_gate.py`
  は configs 追加時に schema 違反を検出する。operator が手で
  `9b_tg_lora.yaml` に不正 field を追加した場合、本 TASK の distinct handler
  が「AppConfig validation failure」として exit 78 で fail-loud。
- **`sys.exit()` の message 一貫性** — `--json` mode では stderr に 1 行
  JSON で `{"error": "<class>", "detail": "<message>", "exit_status": 78}` を
  出力。human mode では `"<class>: <message>"` 形式。

## テスト戦略

- 4 error class × 3 entrypoint で **mutation-proof** な test を追加
- 既存 test cluster の regression なし（`tests/test_replay_freeze_validloss_ci.py`,
  `tests/test_run_freeze_validloss_ci_9b_*.py`, `tests/test_freeze_ci_9b_launch_honesty.py`）
- 既存 `argparse.error` の exit code 2 は不変（本 TASK の scope 外）

## 影響範囲

- 新規 file: `src/utils/cli_errors.py` (or `scripts/_cli_errors.py`) — error class + handler 共通化
- 変更 file: 3 entrypoint script + 各 test file
- 既存 test count: ~157 passed（replay cluster）+ ~537 passed（verdict-path cluster） を保持

## 関連文書

- **要件定義書**: [requirements.md](requirements.md)
- **分析記録**: [interview-record.md](interview-record.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **正本**: `docs/GOAL.md` §7
- **task 仮 ID**: TASK-0179（次 iter で確定）
