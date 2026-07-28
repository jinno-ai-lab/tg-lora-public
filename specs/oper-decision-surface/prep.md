# oper-decision-surface 準備タスク（ユーザー作業）

> **仕様**: [requirements.md](requirements.md)
> **生成日**: 2026-07-28

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 要件定義書・既存実装から明確に必要と判明したタスク
- 🟡 **黄信号**: 要件定義書・既存実装から妥当に推測されるタスク
- 🔴 **赤信号**: 推測による予防的タスク（実装時に不要と判明する可能性あり）

## 必須（実装開始前に完了が必要）

実装は既に commit 済（TASK-0198 / 0217 / 0224 + 43 tests green・既存 `scripts/section4_operator_decision.py`）。本 spec は **docs-only growth**（spec 4 files のみ commit）であり、operator が追加で実施する準備タスクは **なし**。

- [x] **commit 済 §4 deposit 2 件** 🔵 *TASK-0145 + 2026-07-27 harvest*
  - `tests/fixtures/freeze_validloss_ci_9b_full.json` — homogeneous leg, `citable_as_full_section4_verdict=True`
  - `tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json` — heterogeneous leg, `citable_as_full_section4_verdict=True`
  - 関連要件: REQ-001, REQ-003, REQ-004

- [x] **commit 済 `scripts/section4_operator_decision.py`** 🔵 *TASK-0198*
  - 750+ 行実装 + 43 mutation-proof tests
  - 関連要件: REQ-001, REQ-301, REQ-401

- [x] **commit 済 `tests/test_section4_operator_decision.py`** 🔵 *TASK-0198*
  - 43 tests covering TestSection4DecisionArc / TestVerdictWorkerExecutability / TestProbeClassification / TestRecommendationLogic / TestPivotBranchCorrection / TestArcIncompleteMutations / TestQualityPreservationAxis / TestLandingRecordRobustness / TestLandToExitZero / TestLandRejects / TestJsonMode / TestHumanMode / TestBlockingPrompt / TestSubprocessIsolation / TestLandPreservesRecommendation / TestLandingRecordFormat / TestMainExitContract / TestHelpSmoke
  - 関連要件: 全 REQ

## 推奨（実装中に用意できればOK）

- [x] **Makefile target `section4-operator-decision`** 🔵 *commit 済*
  - `SECTION4_DECISION_FLAGS` 環境変数で `--land` 引数を渡せる形
  - 関連要件: REQ-001

- [x] **`Makefile` 内 `PYTHON_VENV = $(HOME)/tg-lora/.venv/bin/python`** 🔵 *commit 済*
  - venv があれば venv 経由、なければ system python 経由
  - 関連要件: NFR-001（torch-free probe 環境互換性）

## 確認事項（判断が必要）

- [x] **operator が `--land` で着地する branch の判断** 🔵 *PURPOSE.md 追記19*
  - 背景: SHIP / ACCEPT-NULL / PIVOT の 3 択だが、evidence-based recommendation = SHIP に対し operator が ACCEPT-NULL を選ぶ正当な理由（freeze-order gain が null である read）も存在
  - 関連要件: REQ-403, REQ-404

- [x] **PIVOT 着地時の `private_repo_only` フラグ確認** 🔵 *REQ-206, REQ-303*
  - src.data が stripped な public mirror で PIVOT を着地すると `pivot_private_repo_only=True` フラグが付与され、後で reviewer が「private-repo で再実行が要る」と判定できる
  - 関連要件: REQ-206, REQ-303

---

## サマリー

| 優先度 | 件数 | 🔵 | 🟡 | 🔴 |
|--------|------|-----|-----|-----|
| 必須 | 0 | 0 | 0 | 0 |
| 推奨 | 0 | 0 | 0 | 0 |
| 確認事項 | 0 | 0 | 0 | 0 |

**所要 user-prep タスク: 0 件**（commit 済 artifacts のみで本 surface は完結）

user-prep が 0 件である理由は、本 surface が **deterministic な local Python tool** であり、外部 API / GPU / DB / DNS / 第三者承認 / データ準備 / ライセンス確認のいずれも要求しないため。operator 実行時に必要なのは shell + Python 3.11+ の環境のみで、これは Makefile の standard `PYTHON_VENV` 解決経路で自動的に満たされる。

---

## 関連文書

- **要件定義書**: [requirements.md](requirements.md)
- **分析記録**: [interview-record.md](interview-record.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **正本実装**: [`scripts/section4_operator_decision.py`](../../scripts/section4_operator_decision.py)
- **正本テスト**: [`tests/test_section4_operator_decision.py`](../../tests/test_section4_operator_decision.py)
