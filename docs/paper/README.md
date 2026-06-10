# Paper Hub

## Purpose

このフォルダは、論文執筆に必要な入力資料と元データへのリンクを 1 箇所に集約するためのハブです。

ここには数値や本文を複製しません。正本は元の docs / configs / data / runs / scripts に残し、このフォルダは参照入口だけを提供します。

## Folder Rules

1. このフォルダには link registry と section map だけを置く。
2. 結果数値は [../paper_results_snapshot.md](../paper_results_snapshot.md) と raw artifact を正本とする。
3. 実験計画・gate 判定は [../paper_experiment_plan.md](../paper_experiment_plan.md) と [../eval_plan_and_status.md](../eval_plan_and_status.md) を正本とする。
4. config / data / scripts / runs の実体は移動せず、このフォルダからリンクする。

## Read Order

1. [01_inputs.md](01_inputs.md)
   - 論文執筆に必要な docs / configs / data / scripts の正本一覧
2. [02_source_data.md](02_source_data.md)
   - 論文の主張を支える raw experimental artifacts へのリンク集
3. [03_writing_map.md](03_writing_map.md)
   - 論文の各 section ごとに、どの doc と artifact を参照すべきかの地図
4. [04_component2_positioning.md](04_component2_positioning.md)
   - Component 2 を支配的直進方向の外挿として記述するための主張境界

## Canonical Entry Points

- canonical results snapshot: [../paper_results_snapshot.md](../paper_results_snapshot.md)
- canonical experiment plan: [../paper_experiment_plan.md](../paper_experiment_plan.md)
- canonical status sheet: [../eval_plan_and_status.md](../eval_plan_and_status.md)
- abstract draft: [../abstract.md](../abstract.md)

## Current Safe Posture

現時点では、memory frontier claim は G2 PASS として扱えます。
一方、strict wall-clock speedup claim はまだ使いません。Component 2 は
「一般的な軌道予測」ではなく、[04_component2_positioning.md](04_component2_positioning.md)
に従って「支配的直進方向の外挿」として限定的に記述します。

最新の Component 2 状況:

- offline predictability controls は、EMA 方向が未来更新を予測すること、
  ただし時間順序の先読みではなく低周波の支配的直進成分が主因であることを示した。
- runtime cosine-N ablation は、固定Nに対して `reduction_rate` を `0.625`
  から `0.752` に上げ、3-seedで rollback 0、loss 実質同等を確認した。
- validation-skip diagnostic は完了。cosine-driven `N` は skip 条件下でも
  `reduction_rate` を fixed-N `0.549` から `0.714` へ上げたが、wall-clock は
  fixed-N 比 `1.000x` に留まった。
- post-extrapolation eval を skip した cycle で rollback は出ておらず、安全側の挙動は確認済み。
  ただし pilot validation と scheduled full eval が残るため、2x以上の wall-clock
  speedup はまだ実測 claim として使わない。
- seed 42 の final-eval-only smoke (`EVAL_POINTS=1`) では、cosine-N は baseline 比
  `0.967x`、fixed-N 比 `1.001x`。scheduled full eval が主要固定コストだったことは
  強く示されたが、3-seed へ拡張するまで manuscript-level speed claim にはしない。
- 次の作業は、final-eval-only 設定を3-seedへ拡張し、さらに pilot validation の頻度または方式を削る固定コスト分解 ablation。
- **【2026-06-05決定】Priorベース低次元係数学習設計への移行**:
  効率が1.24xで頭打ちになったのは、方向 $v$ を固定してスケールをその場の少サンプルlossで手探りしていたという「実装の退化」が原因と確定。これを是正するため、軌跡から方向 $v$ とスケール $w_{\text{traj}}$ の両方を prior として推定し、その prior の周りの低次元係数 $\{\alpha, \beta_j\}$ のみをデータで緩やかに学習する設計に移行する。
  有限差分フォールバックへの対応として数値条件の正規化（方向の単位化・$w_{\text{traj}}$による無次元化・補助方向の直交化）を適用する。本番実装前に、この設計が成立するかをオフライン検証で確認する（[docs/master_plan.md](../master_plan.md)のMilestone 9に設定）。

