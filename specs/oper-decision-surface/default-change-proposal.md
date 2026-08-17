# §4 SHIP 後の default 変更提案 — `progressive_freeze_enabled` 既定値（operator hand-off）

> **位置づけ**: `section4_landed_decision.json`（branch=`ship`, landed=true, 2026-07-29）で
> §4 評決は「**品質保持手法として採用**」として着地済み。しかし着地は**いかなる default も
> 変えていない** — 本提案は、そこに残った唯一の product 面（学習既定値）を operator 決定に
> 回す concrete な hand-off artifact である。`requirements.md` REQ-403 と同じ non-unilateral
> 規律で、本 doc は提案のみを行い、loop 側は default を書き換えない。
> loop-axis 面（MS-008 closeout / MS-009 / halt の ratification）は `loop_axis_state.json`
> の `ratification_options` が既に管理しており、本 doc はそれと**直交する product default 面**
> のみを扱う（halt-doc ではない）。

---

## 1. 現状の default（すべて live 検証 — `tests/test_section4_default_proposal.py`）

| 面 | 現状 | 根拠 |
|---|---|---|
| schema 既定値 | `progressive_freeze_enabled: bool = False` | `src/training/config_schema.py`（Progressive Freezing (Phase 1 gate) ブロック） |
| mainline 設定 | キー未設定（= False 継承）→ `make train-tg-lora` は freeze-**OFF** で走る | `configs/9b_tg_lora.yaml` / `configs/9b_tg_lora_paper_poc.yaml` に同キーなし |
| 明示 OFF 面 | m10 dynfreeze 2 config のみ `progressive_freeze_enabled: false` | `configs/9b_tg_lora_m10_dynfreeze{,_baseline}.yaml`（`freeze_mutual_exclusion` validator との排他） |

つまり「SHIP = 品質保持手法として採用」が着地しても、mainline 既定訓練は SHIP 根拠となった
freeze-ON 経路と**異なる** freeze-OFF のまま。この乖離が本提案が閉じる唯一の残差である。

## 2. 束縛済み証拠（これ以上の測定は不要）

決定に必要な表面はすべて機械検証済み（doc ↔ rail ↔ GPU deposit ↔ JSON record）:

- **品質軸（vs full backprop）**: 両 leg **SURPASSES**（valid_loss は低い方が優位）。
- **surrogate 軸**: 両 leg **TIES**（いずれの CI も 0 を含む）。

| leg | candidate | surrogate | CI (cand − surr) | full-backprop base |
|-----|----------:|----------:|------------------|-------------------:|
| homogeneous    | 1.6947 | 1.6960 | [-0.0001, 0.0027] | 1.8794 |
| heterogeneous  | 1.7180 | 1.7191 | [-0.0011, 0.0028] | 1.8862 |
- **コスト軸**: 当該 prod path（Level-1）の実現 backward 削減 = **0.0**
  （in-vivo 検証: `tests/test_progressive_freeze_invivo.py::test_level1_freeze_only_cuts_no_backward_in_vivo`）。
  名目 `reduction_rate ≈ 0.11` は weight-grad FLOP 算術にすぎない。
- 正本: `section4_landed_decision.json` + `tests/fixtures/freeze_validloss_ci_9b_full.json` +
  `tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json`。
  再導行は `scripts/section4_operator_decision.py`（`tests/test_section4_terminal_verdict.py` が live pin）。

読み替え: default を ON にする根拠は**品質保持のみ**（コスト利得は 0）、OFF を維持する根拠は
**簡素性のみ**。測定は双方とも完了しており、残るは operator の価値判断のみ。

## 3. 決定肢（いずれかを operator が着地）

### D1 — mainline 2 config のみ ON（推奨）

`configs/9b_tg_lora.yaml` と `configs/9b_tg_lora_paper_poc.yaml` の `tg_lora:` ブロックに
`progressive_freeze_enabled: true` を 1 行ずつ追加。schema 既定値は不変。

- 効果: mainline 訓練（`make train-tg-lora` / paper-PoC 比較面）が SHIP 根拠と同じ freeze-ON に揃う。
- blast radius: 2 config のみ。実験 fleet（accel / adaptive / cosine / m9 / prefix / psa 等）は
  schema 既定 False のまま変わらない。
- 検証: `tests/test_config_launchability_gate.py`（config round-trip gate）+
  `tests/test_progressive_freeze_schedule_config.py` + `scripts/loop_halt_guard.py` が依然 77
  （default 変更は loop unblock ではないことの確認）。

### D2 — schema 既定値を True に flip

`src/training/config_schema.py` の `progressive_freeze_enabled: bool = False` を `True` に変更。

- 効果: キー未設定の全 config（fleet 大半）が freeze-ON を継承。
- リスク: 2 leg の証拠が覆うのは mainline 面（Level-1, `seq_len=1024`, Qwen3.5-9B + Dolly）のみ。
  fleet 一斉切替は証拠の外挿であり、`tests/test_config_launchability_gate.py` は launchability は
  担保しても学習挙動の等価性は担保しない。dynfreeze 2 面は明示 false なので
  `freeze_mutual_exclusion` は発火しないが、残る全対象の再検証 = 新たな測定を要求する。
- 検証: D1 に加え fleet 全域の A/B 再確認が必要（feedback が「不要」と断じた領域）。

### D3 — default は False のまま（SHIP を opt-in として記録）

変更ゼロ。`docs/section4_terminal_verdict.md` §1 に「採用 = 手法として利用可能・既定では無効」
の 1 文を追記する doc 修正のみ行う。

- リスク: SHIP 評決の「採用」と mainline 経路の乖離が文書上のみ解消され、実態は残る。

## 4. 推奨と着地手順

**推奨: D1。** 証拠が覆う面と変更面が一致し、コスト軸 null（0.0）ゆえ品質動機以外の副作用がなく、
fleet への外挿を含まない。

着地は operator のみが行う（non-unilateral）:

1. operator が D1–D3 のいずれかを ratify する（AI-Hub feedback thread 等での指示）。
2. ratify 後の TASK が該当 diff を適用し、§3 の検証で green を残す。
3. 着地 diff と同じ commit で本 doc の §1 現状表と `tests/test_section4_default_proposal.py`
   の現状 pin を更新する（guard が RED を出すのは「default が変わったのに本提案書が更新されて
   いない」場合のみ — silent drift の機械的防止）。

## 5. Provenance

- 本 doc の全事実主張は `tests/test_section4_default_proposal.py` が live 検証する
  （schema 既定値行・mainline config キー不在・landed record・数値・in-vivo test の実在・3 決定肢）。
- 数値の出典: `section4_landed_decision.json` の basis + live `assess_section4_decision()` snapshot
  （`tests/test_section4_terminal_verdict.py` と同一の再導行系に対して照合）。
- 関連 spec: [requirements.md](requirements.md)（REQ-403 non-unilateral / REQ-405 branch pin）。
