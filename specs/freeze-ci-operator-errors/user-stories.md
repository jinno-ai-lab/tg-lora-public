# freeze-ci-operator-errors ユーザストーリー

<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](../tg-lora/architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-07-20
**関連要件定義**: [requirements.md](requirements.md)
**分析記録**: [interview-record.md](interview-record.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: AI_HUB_MAKE_RUN_FEEDBACK・既存実装・既存 test で直接支持されるストーリー
- 🟡 **黄信号**: 既存 test pattern・既存 launch-honesty invariants から妥当な推測
- 🔴 **赤信号**: 参照資料にない自動推定

---

## エピック1: Operator error 診断の高速化

### ストーリー 1.1: 設定ファイル typo を即座に発見する 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「missing config」+ 既存 `argparse` 経由の path validation*

**私は** TG-LoRA 開発者 **として**
**`python -m scripts.run_freeze_validloss_ci_9b --config configs/9b_tg_lora.yamlx` のように typo した時、即座に `MissingConfigError: config not found: configs/9b_tg_lora.yamlx` で fail-loud してほしい**
**そうすることで** 「設定ファイルの問題なのか、コードの問題なのか、GPU の問題なのか」を切り分けず 1 回目で修正できる

**関連要件**: REQ-001, REQ-003, REQ-101, NFR-201

**詳細シナリオ**:

1. typo した path を `--config` に渡す
2. system が `FileNotFoundError` を内部で `MissingConfigError` に wrap する
3. stderr に `MissingConfigError: config not found: <path>` を出力
4. exit code 78 で終了

**前提条件**:

- 3 entrypoint script のいずれかを実行

**制約事項**:

- 既存 `argparse.error` の exit code 2 は不変（REQ-701）

**優先度**: Must Have

---

### ストーリー 1.2: YAML 文法エラーの位置を特定する 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「malformed YAML」+ PyYAML 標準 error message 整合*

**私は** TG-LoRA 開発者 **として**
**`configs/9b_tg_lora.yaml` の tab/space 混在で YAML parse に失敗した時、**何行目の何列目**で失敗したかを即座に知りたい**
**そうすることで** 該当行を editor で開いてすぐ修正できる

**関連要件**: REQ-001, REQ-003, REQ-201, REQ-202, NFR-201

**詳細シナリオ**:

1. 壊れた YAML を `--config` に渡す
2. system が `yaml.YAMLError` を内部で `MalformedYAMLError` に wrap する
3. stderr に `MalformedYAMLError: yaml parse error in <path>: <PyYAML message>` を出力（行番号・列番号を含む）
4. exit code 78 で終了

**前提条件**:

- YAML 文法的に壊れた file がある

**制約事項**:

- PyYAML の `YAMLError.__str__()` 形式を保持（REQ-202）

**優先度**: Must Have

---

### ストーリー 1.3: Pydantic スキーマ違反を field 単位で診断する 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「AppConfig validation failures」+ Pydantic v2 `errors()` 整合*

**私は** TG-LoRA 開発者 **として**
**`configs/9b_tg_lora.yaml` に存在しない field（例: `keep_last_checkpoints:` が `LoggingConfig` に未宣言）を追加した時、**どの field が違反** したかを即座に知りたい**
**そうすることで** GOAL §6.2「設定変更で動作が変わる」原則と `extra="forbid"` の整合を取りつつ、自分の typo を即座に修正できる

**関連要件**: REQ-001, REQ-003, REQ-301, REQ-302, REQ-303, NFR-201

**詳細シナリオ**:

1. YAML は valid だが Pydantic スキーマに違反する field を追加
2. system が `pydantic.ValidationError` を `AppConfigValidationError` に wrap
3. stderr に `AppConfigValidationError: schema validation failed for <ConfigClass>: <N> errors; first: <loc> <msg> (<type>)` を出力
4. exit code 78 で終了

**前提条件**:

- `extra="forbid"` 設定済の Pydantic model がある

**制約事項**:

- 既存 `test_config_launchability_gate.py` の schema 範囲と整合（REQ-303）

**優先度**: Must Have

---

### ストーリー 1.4: 破損した eval result deposit を即座に発見する 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「malformed eval results」+ `load_samples()` schema 強化*

**私は** 9B target-scale run 後の reviewer **として**
**`replay_freeze_validloss_ci.py <deposit.json>` で deposit が壊れている
（必須 key 欠落 / 型不一致）時、**どの field が壊れているか** を即座に知りたい**
**そうすることで** 9B 実行をやり直すか deposit を修正するか判断できる

**関連要件**: REQ-001, REQ-003, REQ-401, REQ-402, NFR-201

**詳細シナリオ**:

1. 9B run が異常終了して deposit が partial な状態で残存
2. reviewer が `replay_freeze_validloss_ci.py <deposit.json>` を実行
3. system が `load_samples()` の schema check で `MalformedEvalResultsError` を発火
4. stderr に `MalformedEvalResultsError: missing key: candidate_total` 等を出力
5. exit code 78 で終了

**前提条件**:

- `samples.json` または `deposit.json` が partial / 破損

**制約事項**:

- 既存 `--expected` 不一致 error（exit 2）は別 path（REQ-602）

**優先度**: Must Have

---

## エピック2: CI 統合の安定化

### ストーリー 2.1: CI が operator error と他 error を区別できる 🔵

**信頼性**: 🔵 *既存 `argparse.error` exit 2 + 既存 worker exit 4 との差別化*

**私は** CI エンジニア **として**
**freeze-ci-9b の CI script が `exit 78`（operator error）で fail した時に、**GPU 再実行ではなく設定修正が必要** と判定したい**
**そうすることで** GPU ジョブを浪費せず、PR コメントで「設定ファイル修正求む」と即座に報告できる

**関連要件**: REQ-003, REQ-705, NFR-202

**詳細シナリオ**:

1. CI で `python -m scripts.run_freeze_validloss_ci_9b --config <bad>` を実行
2. system が exit 78 で fail
3. CI script が `if exit_code == 78: post_pr_comment("config error")` ブランチで handling
4. 開発者が即座に config を修正

**前提条件**:

- CI script が exit code 78 を特別扱いする

**制約事項**:

- sysexits.h `EX_CONFIG` = 78 由来（REQ-705）

**優先度**: Must Have

---

### ストーリー 2.2: `--json` mode で機械抽出できる operator error を得る 🟡

**信頼性**: 🟡 *既存 `replay_freeze_validloss_ci.py --json` pattern 整合*

**私は** 下流 evaluator（外部 script） **として**
**`--json` mode で replay を実行した時、operator error 発生時も JSON 形式で
 1 行出力してほしい**
**そうすることで** stderr 文字列を grep せず、機械的に `error class` を判定できる

**関連要件**: REQ-003, REQ-501, REQ-502, NFR-202

**詳細シナリオ**:

1. 下流 evaluator が `replay_freeze_validloss_ci.py <bad.json> --json` を実行
2. system が `{"error": "MalformedEvalResultsError", "detail": "...", "exit_status": 78}` を 1 行で stdout に出力
3. evaluator が `json.loads(stdout)` で class を取得

**前提条件**:

- `--json` mode が replay script に存在

**制約事項**:

- EDGE-102: stdout JSON は 1 行（`\n` なし）
- 既存 `--json` mode の正常 path は不変

**優先度**: Should Have

---

## ストーリーマップ

```
エピック1: Operator error 診断の高速化
├── ストーリー 1.1 (🔵 Must Have) - typo / missing config
├── ストーリー 1.2 (🔵 Must Have) - YAML 位置特定
├── ストーリー 1.3 (🔵 Must Have) - Pydantic field 単位
└── ストーリー 1.4 (🔵 Must Have) - eval result 破損

エピック2: CI 統合の安定化
├── ストーリー 2.1 (🔵 Must Have) - exit 78 差別化
└── ストーリー 2.2 (🟡 Should Have) - --json mode
```

## 信頼性レベルサマリー

- 🔵 青信号: 7件 (87.5%)
- 🟡 黄信号: 1件 (12.5%)
- 🔴 赤信号: 0件 (0%)

**品質評価**: 高品質
