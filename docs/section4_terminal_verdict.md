# §4 Terminal Verdict — Progressive Freezing at 9B Target Scale

> **位置づけ**: GOAL.md §4（Execution plan）・§7（科学誠実性鉄則）に属する §4 研究 arc の
> **最終評決**を、単一の自己完結した引用可能アーティファクトとして正式化する文書。
> オペレータ決定 landing（`section4_landed_decision.json`）と `scripts/section4_operator_decision.py`
> の snapshot が事実関係の正本。本ファイルの**全数値・全評決は
> `tests/test_section4_terminal_verdict.py` が committed fixture + LIVE 再導出に対して機械検証する**
> （prose が陳腐化したら CI が RED になる — §7 の「評価条件を必ず統一し、予測力は loss 着地で検証するまで信用しない」を、
> docs に対しても適用した形）。

---

## 1. 評決（Verdict）: SHIP — landed

§4 の問い「**9B target scale で Progressive Freezing は random-order surrogate に勝つか?**」への答えは **TIES（引分）**。
両 leg（homogeneous / heterogeneous LoRA-rank）とも、citable な full-budget §4 deposit で
**faithful・evidence-intact な TIES** として再導出される。よってオペレータ決定は **SHIP**
（citable な相対評決 TIES を §4 の結果として採用）として landing 済（`section4_landed_decision.json`,
branch=`ship`, 2026-07-29）。

| leg | candidate mean | surrogate mean | 95% CI (cand − surr) | verdict | full-backprop baseline |
|-----|---------------:|---------------:|----------------------|---------|------------------------|
| homogeneous    | 1.6947 | 1.6960 | [−0.0001, +0.0027] | **TIES** | SURPASSES (base 1.8794) |
| heterogeneous  | 1.7180 | 1.7191 | [−0.0011, +0.0028] | **TIES** | SURPASSES (base 1.8862) |

両 leg: `seq_len=1024`, `proxy_scale=False`, `total_steps=1500`, `model=Qwen/Qwen3.5-9B`,
`dataset=databricks/databricks-dolly-15k`, `citable_as_full_section4_verdict=True`,
`citable_as_target_scale=True`, `n_baseline=3`.

**解釈**: Progressive Freezing は full backprop に対して**品質を保持したコスト削減**（P1 品質保持 = SURPASSES、
両 leg）だが、random-order surrogate に対する loss 改善は target scale では**信号にならない**（TIES）。
すなわち Progressive Freezing の寄与は「品質保持型の効率化」であり、loss 改善手法ではない — という
null 含みの正直な結論。（参考: deposits の `candidate_cost_reduction.realized_reduction_rate = 0.0` は、
名目 reduction_rate≈0.11 に対し実現コスト削除が 0 であることを記録する。別軸の研究課題として残る。）

## 2. §7 自己完結再現性（citable ≠ reproducible — 本 mirror で閉じた）

GOAL §7 の鉄則「予測力は loss 着地で検証するまで信用しない」の docs 版として、
「citable な評決」が「自己完結して再現可能」であることを 3 つの機械検証可能事実で閉じる:

1. **verdict worker は `src.data` 非依存** — `scripts/run_freeze_validloss_ci_9b.py` は自身の SFT adapter で
   public Dolly を扱い、private `src.data` を import しない（`tests/test_9b_verdict_producer_self_contained.py`
   が AST で保証）。よって verdict run に architectural な src.data block は無い。
2. **offline `--data-file` rail が実証済** — `tests/fixtures/freeze_validloss_ci_9b_datafile_repro.json`
   （+ ledger + runlog）は、LOCAL Dolly dump から network/private-`src.data` 無しで real 9B A/B を deposit した
   fired 証拠（`citable_as_target_scale=True`）。ledger header は `data_file=<local .jsonl>` を stamp する。
3. **同一モデル・同一公開データ** — offline rail 証拠と full §4 verdict は同じ `Qwen/Qwen3.5-9B` +
   同じ `databricks/databricks-dolly-15k` を使う。よって §4 verdict は本 public mirror 単独で
   offline 自己完結再現可能。

（上記 1–3 はすべて `tests/test_section4_terminal_verdict.py` が検証する。reduced-budget smoke は
`citable_as_full_section4_verdict=False` を正しく stamp し、§4 honesty gate から安全に除外される。）

## 3. 残る唯一の open（decision B — 本 repo ではスコープ外）

**private `src.data` PRODUCTION-baseline 絶対損失 leg** — `src.data` は本 public mirror で意図的に strip 済み
（public/private 境界）。この leg は private checkout（`/home/jinno/tg-lora`）でのみ actionable。
本 repo では tooling を拡張せず、**operator hand-off** として記録する（AI-Hub feedback decision (B) の通り）。

## 4. Go/no-go — arc 継続の可否

§4 の研究問いは**回答済**（TIES）。proxy scale でも null は硬化しており（banked verdict: ratio=0.000 null at proxy scale）、
target scale でも TIES。よって **funnel/arc の継続は、新たな問い無しには正当化されない**（go/no-go = NO-FURTHER）。

前方路は明示的に 2 つのみ:
- **(B)** private repo で production-baseline 絶対損失 leg を実行（上記 §3）。
- **(C)** §4 arc を離れる **新規 Cat-A** デリバラブル（例: GOAL §3.1 Phase 4 のスケジュール汎用性、
  または PSA §3.2 の再活性化）を定義して loop を pivot する。

## 5. Provenance（機械検証）

- 評決・数値: `scripts/section4_operator_decision.py`（`assess_section4_decision()` が deposited samples から
  GPU-free bootstrap で再導出）+ `tests/fixtures/freeze_validloss_ci_9b_full{,_heterogeneous}.json`。
- §7 自己完結: `tests/fixtures/freeze_validloss_ci_9b_datafile_repro{,_ledger.jsonl,_runlog.json}` +
  `tests/test_9b_verdict_producer_self_contained.py`。
- landing: `section4_landed_decision.json`（branch=`ship`, landed=true）。
- 本 doc の全主张: `tests/test_section4_terminal_verdict.py` が上記ソースに対して pin。
