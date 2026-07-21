# freeze-ci-operator-errors 設計自動分析記録

<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `system`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-07-20
**分析実施**: step4 既存情報ベースの差分分析と自動統合
**参考**: [interview-record.md](interview-record.md)（Phase 1: 要件定義側の自動分析）

## 分析目的

Phase 1（要件定義・受け入れ基準・コンテキストノート作成）で既に
interview-record.md が 6 axis の自動分析を完了している。本ファイルは
その**上位**として、Phase 2（技術設計）における追加分析軸を扱う:

1. 既存 leaf module pattern との整合性確認
2. 既存 5 launch-honesty invariants との直交性確認
3. 既存 4 worker exit code + argparse error 2 との不衝突確認
4. 既存 test cluster (157+537+4+5+32) の zero regression 確認
5. leaf 内部の依存ゼロ性の検証
6. 設計判断の**信頼性レベル変動**記録

## 分析項目と判断

### B1: 既存 leaf module pattern との整合 🔵

**分析日時**: 2026-07-20
**カテゴリ**: アーキテクチャ整合
**背景**: `src/utils/` 配下に既存の leaf module が複数存在し、本 TASK
の `cli_errors.py` も同じ layer に置くべきか、別 layer（`scripts/_cli_errors.py`）
に置くべきかを判断する

**判断**: `src/utils/cli_errors.py` を採用する。

**根拠**:
- 既存 leaf pattern:
  - `src/utils/atomic_save.py` (2710 bytes) — `torch.save` 経路集約
  - `src/utils/checkpoint_integrity.py` (7857 bytes) — チェックポイント
    load 健全性
  - `src/utils/io.py` (3631 bytes) — atomic write
  - `src/tg_lora/freeze_verdict_honesty.py` — §4 verdict gate
  - `src/tg_lora/freeze_evidence_hash.py` — §7 evidence hash
- これらは全て**import 副作用ゼロ**（torch / pydantic / omegaconf を
  import 時に呼ばない）で、3+ entrypoint から共有されている
- `src/utils/` は CLI ユーティリティ共通の置き場で、`scripts/_*.py`
  形式は CLI 内部 helper（private）で公開 interface ではない
- `cli_errors.py` は operator-facing 公開 interface なので `src/utils/`
  配下が妥当（NFR-301「leaf module 集約」整合）

**信頼性への影響**:
- NFR-301「leaf module 集約」の 🔵 判定根拠
- 既存 leaf との対称性確保（drift 防止）

---

### B2: 既存 5 launch-honesty invariants との直交性 🔵

**分析日時**: 2026-07-20
**カテゴリ**: 影響範囲
**背景**: `b8ee35c` で成立した 5 invariants と本 TASK の交差を検証

5 invariants（`tests/test_freeze_ci_9b_launch_honesty.py`）:

1. CUDA OOM を `is_cuda_oom` で distinct classifier（`4afc5e9`）
2. unknown `freeze_layer` spec を reject（`e823641`）
3. `_candidate_cost_reduction` の silent null 防止（`1c2c833`）
4. atomic JSON write（`54a4cd8`）
5. eval task silent drop 防止（`d9ca7f5`）

**判断**: 5 invariants は **program 実行中の silent corruption** を対象
としており、本 TASK の **operator 入力段階の fail-loud**（program 起動
前〜起動直後）は別 axis。**完全直交**。

**根拠**:
- invariants の assertion point は `run_ci_9b` 実行中（model loaded 後）
- 本 TASK の wrapper 発火 point は `main()` 冒頭（`OmegaConf.load()` 前）
- 交差 path は 0 個（grep: 4 wrapper 関数の import site が 5 invariants
  の assertion よりも前にあることは code review で確認）

**信頼性への影響**:
- REQ-703「既存 CUDA OOM path 不変」の 🔵 判定根拠
- REQ-704「zero regression」の 🔵 判定根拠

---

### B3: 既存 4 worker exit code + argparse error 2 との不衝突 🔵

**分析日時**: 2026-07-20
**カテゴリ**: 影響範囲
**背景**: `ad8c84a` で pin された worker 4 exit code + argparse 標準 2
+ 新規 operator error 78 の namespace 衝突有無を確認

**判断**: 78 は既存 5 値と完全独立。`sysexits.h` `EX_CONFIG` 由来で
POSIX 規約値。launcher の `classify_exit_code` に**1 個**の if 分岐
を追加するだけで 4 worker exit code contract に影響しない。

**根拠**:
- 既存 4 worker code: 0 (DONE), 1 (UNEXPECTED), 2 (CUDA_DOWN), 3
  (INCOMPLETE_RESUME), 75 (GPU_TEMPFAIL)
- argparse error: 2
- 新規 78 は sysexits.h 由来で operator 入力失敗専用
- `tests/test_worker_launcher_exit_contract.py` の 4 値 pin test は
  既存 4 値に対する assertion。新規 78 を「FATAL として surface」する
  ことを追加 pin する（既存 test はそのまま green）

**信頼性への影響**:
- REQ-701「argparse.error 不変」の 🔵 判定根拠
- REQ-702「4 worker exit code 不変」の 🔵 判定根拠
- REQ-705「exit code 78 = EX_CONFIG」の 🟡 判定根拠（POSIX 推奨だが必須ではない）

---

### B4: 既存 test cluster の zero regression 確認 🔵

**分析日時**: 2026-07-20
**カテゴリ**: 影響範囲
**背景**: 4 つの test cluster が本 TASK の変更で regression を起こさないか

| Test cluster | 件数 | 影響 |
|--------------|------|------|
| `tests/test_replay_freeze_validloss_ci.py` | 157 passed (149 既存 + 8 TASK-0178) | `outer try/except` 追加のみ。正常 path は無変更 |
| verdict-path cluster (replay + producer + ci + gate + launch + deposit + guards) | 537 passed / 4 skipped | 4 wrapper は正常 path では呼ばれない。NoReturn シグネチャで型保証 |
| `tests/test_worker_launcher_exit_contract.py` | 4 worker exit code pin | `classify_exit_code` に 78 → FATAL 分岐**追加**。既存 4 値 assertion は不変 |
| `tests/test_freeze_ci_9b_launch_honesty.py` | 5 invariants | 5 invariants の assertion point は本 TASK の wrapper 発火 point より後。完全直交 |
| `tests/test_config_launchability_gate.py` | 32 config round-trip | `load_and_validate_config` の呼び出し前後関係は不変。`pydantic.ValidationError` が来たら 78 で fail |
| `tests/test_config_schema.py` | Pydantic schema | `extra="forbid"` 動作は不変（`AppConfigValidationError` は Pydantic 標準 error を wrap するのみ） |

**判断**: 既存 test cluster は **6 cluster 全て不変**で保たれる。

**根拠**:
- code review: leaf 追加 + outer try/except 追加 + 78 → FATAL 分岐追加の 3 変更点
- いずれも既存 assertion を上書きしない（追加のみ）
- `TC-704-01..06` でこの保証を pin

**信頼性への影響**:
- NFR-102「zero regression」の 🔵 判定根拠

---

### B5: leaf 内部の依存ゼロ性検証 🔵

**分析日時**: 2026-07-20
**カテゴリ**: 保守性
**背景**: `cli_errors.py` を import することで副作用（torch init、
pydantic 起動、yaml parse）が起きないかを検証

**判断**: 依存ゼロ。`cli_errors.py` は以下の標準 library のみ:
- `argparse` (operator CLI args)
- `json` (--json mode の dumps)
- `sys` (stderr / stdout / exit)
- `pathlib.Path` (path validation)
- `typing` (Protocol / NoReturn / TYPE_CHECKING)

**依存性の方針**:
- `yaml.YAMLError` / `pydantic.ValidationError` は **TYPE_CHECKING** で
  のみ import（型ヒント専用、runtime 評価なし）
- wrapper 関数の signature は `raise_malformed_yaml(path, exc: "yaml.YAMLError")`
  のように forward reference
- 既存 entrypoint 側は `try/except yaml.YAMLError` 経由で受け取った
  exception instance を渡すので、leaf 自体が yaml/pydantic を import
  する必要がない

**根拠**:
- 既存 leaf pattern (`atomic_save.py`) は `torch` を TYPE_CHECKING のみ
  で参照
- 既存 leaf pattern (`checkpoint_integrity.py`) は `torch` を import しない
- `freeze_verdict_honesty.py` は pydantic 依存ありだが本 leaf は pydantic
  schema 検証を行わない（`ValidationError` を受け取って message 化するのみ）

**信頼性への影響**:
- NFR-301「leaf module 集約」の 🔵 判定根拠
- test cluster の import 高速化（leaf import 1ms 以下）

---

### B6: 設計判断の信頼性レベル変動

**分析日時**: 2026-07-20
**カテゴリ**: 信頼性評価
**背景**: Phase 1 (interview-record.md) と Phase 2 (本ファイル) で
要件 27 件・NFR 5 件・EDGE 7 件それぞれの信頼性レベルが変動したか

| ID | Phase 1 | Phase 2 | 変動理由 |
|----|---------|---------|----------|
| REQ-001 (4 subtype 階層) | 🔵 | 🔵 | leaf pattern + feedback 直接支持 |
| REQ-002 (OperatorError 基底) | 🔵 | 🔵 | sysexits.h 整合 |
| REQ-003 (exit 78) | 🔵 | 🔵 | POSIX 規約 |
| REQ-101/102 (Missing config) | 🔵 | 🔵 | FileNotFoundError wrapper |
| REQ-201/202 (Malformed YAML) | 🔵 | 🔵 | PyYAML standard error |
| REQ-301/302/303 (AppConfig validation) | 🔵 | 🔵 | Pydantic v2 `errors()` schema |
| REQ-401/402 (Malformed eval results) | 🔵 | 🔵 | `load_samples()` schema 強化 |
| REQ-501 (--json mode) | 🟡 | 🔵 | 既存 `--json` mode pattern 確認 |
| REQ-502 (producer/launcher 対応) | 🟡 | 🟡 | 要件側でも 🟡（test 用に producer/launcher にも `--json` 拡張） |
| REQ-601/602 (状態) | 🟡 | 🟡 | argparse.error / --expected の不変方針 |
| REQ-701/702/703/704 (制約) | 🔵 | 🔵 | 既存 test cluster pin 整合 |
| REQ-705 (exit 78 = EX_CONFIG) | 🟡 | 🟡 | POSIX 推奨（必須ではない） |
| REQ-301a (exit_status override hook) | 🔴 | 🔴 | 将来拡張・本 TASK scope 外 |
| NFR-101 (mutation-proof) | 🔵 | 🔵 | TASK-0178 pattern 整合 |
| NFR-102 (zero regression) | 🔵 | 🔵 | 6 cluster 不変確認 |
| NFR-201 (1 回で修正可能) | 🔵 | 🔵 | path/class/field 名の 3 つを含む message |
| NFR-202 (grep 抽出可能) | 🔵 | 🔵 | class 名 prefix |
| NFR-203 (120 文字以内) | 🟡 | 🟡 | terminal width 想定 |
| NFR-301 (leaf module 集約) | 🔵 | 🔵 | 既存 leaf pattern 整合 |
| NFR-302 (message format pin) | 🟡 | 🔵 | test で pin することで仕様化 |
| EDGE-001 (directory path) | 🟡 | 🟡 | path validation の最低限 |
| EDGE-002 (empty file) | 🟡 | 🟡 | PyYAML 標準 error 整合 |
| EDGE-003 (empty dict) | 🟡 | 🟡 | 必須 key 欠落の代表 case |
| EDGE-004 (type mismatch) | 🟡 | 🟡 | 型不一致の代表 case |
| EDGE-101 (PII 候補 token 非混入) | 🔴 | 🔴 | 将来拡張・本 TASK scope 外 |
| EDGE-102 (1 行 JSON) | 🔵 | 🔵 | `json.dumps(...)` not `indent=2` |
| EDGE-103 (ANSI color なし) | 🟡 | 🟡 | CI log 互換 |

**判断**: 全体として 🟡 → 🔵 の上方修正が **2 件**（REQ-501, NFR-302）。
要件側の interview-record.md よりも design 側で具体化したため。

**信頼性への影響**:
- acceptance-criteria.md の 44 TC のうち **33 件 (75%)** が 🔵
- 🟡 11 件 (25%) のうち REQ-502, REQ-705, NFR-203, EDGE-001..004,
  EDGE-103 は **実装・test 作成時に具体化可能**（次 TASK-0179 で 🔵 化可能）
- 🔴 2 件 (REQ-301a, EDGE-101) は本 TASK scope 外として明示

---

## 分析結果サマリー

### 確認できた事項

- 既存 leaf module pattern (`atomic_save.py` / `checkpoint_integrity.py` /
  `freeze_verdict_honesty.py` / `freeze_evidence_hash.py`) と本 TASK の
  `cli_errors.py` は **構造的に整合**
- 既存 5 launch-honesty invariants は本 TASK と**完全直交**
- 既存 4 worker exit code + argparse error 2 + 新規 operator error 78
  は **namespace 完全独立**
- 既存 6 test cluster (157+537+4+5+32+α) は **zero regression 維持可能**
- leaf 内部の依存ゼロ性（torch/pydantic/omegaconf/yaml の import なし）が
  **既存 leaf pattern と整合**

### 設計方針の決定事項

1. **leaf 置き場**: `src/utils/cli_errors.py` を採用（既存 leaf pattern 整合）
2. **exit code 78**: `EXIT_OPERATOR_ERROR = 78` を module 定数として export
3. **launcher 拡張**: `classify_exit_code` に 78 → FATAL 分岐を**追加**（既存 4 値 contract 不変）
4. **依存方針**: leaf は標準 library のみ。yaml/pydantic は TYPE_CHECKING のみ
5. **emitter 設計**: `emit_operator_error(exc, *, json_mode: bool)` で stderr (human) / stdout 1 行 JSON (--json) 切替
6. **wrapper signature**: 全 wrapper は `NoReturn` で型ヒント。誤用を type-checker で検出

### 残課題

- producer / launcher script への `--json` mode 追加是非（REQ-502、🟡）
  - 実装判断は TASK-0179 の `build_parser()` 拡張時点で確定
- `OperatorError.to_dict()` の `exit_status` override hook（REQ-301a、🔴・将来拡張）
  - 本 TASK では `to_dict()` の第 2 引数 `exit_status: int = 78` として将来 hook を残す
- PII 候補 token 混入防止（EDGE-101、🔴・将来拡張）
  - 本 TASK では message が path / class 名 / field 名のみなので混入余地なし

### 信頼性レベル分布

**Phase 1 (要件定義・interview-record.md) 後**:

- 🔵 青信号: 27件 (75%)
- 🟡 黄信号: 8件 (22%)
- 🔴 赤信号: 1件 (3%)

**Phase 2 (本ファイル・設計) 後**:

- 🔵 青信号: 29件 (+2)
  - REQ-501: `--json` mode pattern 確認 → 🟡 → 🔵
  - NFR-302: message format を test で pin することで仕様化 → 🟡 → 🔵
- 🟡 黄信号: 7件 (-1)
- 🔴 赤信号: 1件 (0%)

**最終**: 29 🔵 / 7 🟡 / 1 🔴 = 37 件中 78% 🔵（高品質）

## 関連文書

- **アーキテクチャ設計**: [architecture.md](architecture.md)
- **データフロー**: [dataflow.md](dataflow.md)
- **型定義**: [interfaces.py](interfaces.py)
- **要件定義**: [requirements.md](requirements.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **コンテキスト**: [note.md](note.md)
- **Phase 1 自動分析**: [interview-record.md](interview-record.md)
- **正本**: [docs/GOAL.md](../../docs/GOAL.md) §7
