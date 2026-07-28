# oper-decision-surface ユーザストーリー

**作成日**: 2026-07-28
**feature_id**: `oper-decision-surface`
**work_scope**: `light`
**関連要件定義**: [requirements.md](requirements.md)
**分析記録**: [interview-record.md](interview-record.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 既存 `scripts/section4_operator_decision.py` 実装 + `tests/test_section4_operator_decision.py` + `PURPOSE.md` 追記15/19 からの確実な抽出
- 🟡 **黄信号**: 既存 surface pattern からの妥当な推測
- 🔴 **赤信号**: 推測による追加（本 spec では使用しない）

---

## エピック1: operator が §4 verdict を landing する

### ストーリー 1.1: shell から snapshot を確認する 🔵

**信頼性**: 🔵 *scripts:50-50, 700-745 + tests:TestSection4DecisionArc*

**私は** 開発者 **として**
**`python scripts/section4_operator_decision.py` を実行したい**
**そうすることで** Currently arc が complete なのか、complete なら recommendation が何か、landed 済みかを 1 コマンドで把握できる

**関連要件**: REQ-001, REQ-002, REQ-201, REQ-202

**詳細シナリオ**:

1. shell で `python scripts/section4_operator_decision.py` を実行
2. ツールが 2 つの §4 deposit を読み、各 leg の `present` / `citable_as_full_section4_verdict` / `faithful` / `rederived_verdict` を print
3. landing record が absent なら `arc_complete` / `recommendation` / `awaiting_operator_decision` を print し、続けて `_blocking_prompt()` 1 個を stderr へ
4. exit code 3 で blocking（landed していない場合）

**前提条件**:

- `tests/fixtures/freeze_validloss_ci_9b_full.json` と `tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json` が commit 済
- `scripts/run_freeze_validloss_ci_9b.py` が import 可能（public Dolly path、no src.data）
- `section4_landed_decision.json` が repo root に存在しない（未着地）

**優先度**: Must Have

---

### ストーリー 1.2: CI から machine-readable snapshot を取り込む 🔵

**信頼性**: 🔵 *scripts:51-53, 719-745 + tests:TestJsonMode*

**私は** CI パイプライン **として**
**`python scripts/section4_operator_decision.py --json` を実行したい**
**そうすることで** recommendation / arc_complete / legs / branches / landed_decision を 1 行 JSON で parse でき、CI step の gate として使える

**関連要件**: REQ-103, REQ-501

**詳細シナリオ**:

1. CI runner で `python -m scripts.section4_operator_decision --json` を subprocess 起動
2. stdout から 1 行 JSON を読み、`json.loads()` で dict 化
3. `arc_complete` / `recommendation` / `awaiting_operator_decision` を取り出し、CI step の fail/pass を判定
4. exit code 3 なら CI job は pending へ（blocking 状態）

**前提条件**:

- Python 3.11+ が install 済
- pytest 等の test framework が利用可能

**優先度**: Must Have

---

### ストーリー 1.3: operator が --land で SHIP を着地する 🔵

**信頼性**: 🔵 *scripts:52, 297-322, 569-617 + tests:TestLandToExitZero*

**私は** 開発者 **として**
**`python scripts/section4_operator_decision.py --land ship --basis "..."` で SHIP 決定を record したい**
**そうすることで** 自分の §4 SHIP 判断が atomic に commit され、ツールが以後 exit 0 で動作する

**関連要件**: REQ-202, REQ-301, REQ-302, REQ-305, REQ-306

**詳細シナリオ**:

1. shell で `--land ship --basis "Both §4 legs are citable faithful TIES; the verdict arc is COMPLETE."` 実行
2. ツールが arc_complete を再評価（commit 後に陳腐化していないか gate）
3. `section4_landed_decision.json` を atomic 書込（`src.utils.io.save_json` 経由）
4. exit 0 で stdout へ `landed_decision.branch` / `basis` / `landed_at` / `pivot_private_repo_only` を出力
5. `recommendation` は SHIP のまま（landed による mutate なし）

**前提条件**:

- arc_complete == True（両 leg が citable / faithful / TIES）
- `basis` 文字列が non-empty
- 書込先 `section4_landed_decision.json` の親ディレクトリ（repo root）が writable

**優先度**: Must Have

---

### ストーリー 1.4: operator が --land で ACCEPT-NULL を着地する 🔵

**信頼性**: 🔵 *scripts:50-52, 569-617 + tests:TestLandPreservesRecommendation*

**私は** 開発者 **として**
**`--land accept_null --basis "..."` で TIES を honest null として record したい**
**そうすることで** 自分が Progressive Freezing は random-order surrogate に **勝たない** という read を採用した、という evidence になる

**関連要件**: REQ-202, REQ-301, REQ-404

**詳細シナリオ**:

1. shell で `--land accept_null --basis "The freeze-order gain is null at full budget; SURPASSES only at reduced budget."` 実行
2. ツールが arc_complete を gate
3. `section4_landed_decision.json` を atomic 書込、branch = "accept_null"
4. 標準出力には `reco=SHIP`（evidence-based recommendation）のまま + `landed_branch=accept_null`（operator の call） = 両方が分離表示される

**前提条件**:

- arc_complete == True
- `basis` 文字列が non-empty

**優先度**: Must Have

---

### ストーリー 1.5: operator が PIVOT 着地を試みると private-repo-only 警告が出る 🔵

**信頼性**: 🔵 *scripts:107-141, 569-617 + tests:TestPivotBranchCorrection*

**私は** 開発者 **として**
**`--land pivot --basis "..."` 実行時、src.data が stripped なので private-repo-only フラグ付きで record されることを期待する**
**そうすることで** public mirror 上で PIVOT 着地が「private-repo-only な標識付き」で commit され、後から reviewer が「これは private repo で再実行が要る」と判断できる

**関連要件**: REQ-206, REQ-303

**詳細シナリオ**:

1. shell で `--land pivot --basis "src.data is stripped; pivot is private-repo-only."` 実行
2. ツールが `src_data_status == "stripped_deliberate"` を確認
3. `section4_landed_decision.json` を atomic 書込、branch = "pivot", `pivot_private_repo_only = True`
4. 出力に `pivot_private_repo_only: true` が明示される

**前提条件**:

- arc_complete == True
- `src/data/build_seed_dataset.py` が存在しない（public mirror）

**優先度**: Should Have

---

### ストーリー 1.6: 壊れた landing record を tool が detected して blocking 🔵

**信頼性**: 🔵 *scripts:277-294 + tests:TestLandingRecordRobustness*

**私は** 開発者 **として**
**`section4_landed_decision.json` が壊 JSON / branch 無し / OSError 読み込み失敗のとき、ツールが blocking することを期待する**
**そうすることで** 壊 record が「着地済」を偽装せず、operator に再 `--land` 機会を与える

**関連要件**: REQ-105, EDGE-001, EDGE-003

**詳細シナリオ**:

1. shell で `section4_landed_decision.json` を `{}` に書き換え
2. `python scripts/section4_operator_decision.py` 実行
3. ツールが `_load_landed_decision()` で `None` を返す
4. exit 3 で blocking prompt を出す

**前提条件**:

- arc_complete == True（deposit 側は valid）

**優先度**: Must Have

---

## ストーリーマップ

```
エピック1: operator が §4 verdict を landing する
├── ストーリー 1.1 (🔵 Must Have) — shell snapshot
├── ストーリー 1.2 (🔵 Must Have) — CI JSON
├── ストーリー 1.3 (🔵 Must Have) — SHIP 着地
├── ストーリー 1.4 (🔵 Must Have) — ACCEPT-NULL 着地
├── ストーリー 1.5 (🔵 Should Have) — PIVOT private-repo-only
└── ストーリー 1.6 (🔵 Must Have) — 壊 record blocking
```

## 信頼性レベルサマリー

- 🔵 青信号: 6 件 (100%)
- 🟡 黄信号: 0 件 (0%)
- 🔴 赤信号: 0 件 (0%)

**品質評価**: 高品質（全 6 ストーリーが 🔵 青信号・既存実装 + 既存 43 tests + PURPOSE.md 追記から 1 対 1 抽出、推測 0 件）

## 信頼性レベル分布

- 🔵 青信号: 6 件 (100%)
- 🟡 黄信号: 0 件 (0%)
- 🔴 赤信号: 0 件 (0%)

**品質評価**: 高品質
