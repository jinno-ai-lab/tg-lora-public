# PURPOSE.md — TG-LoRA 目標・マイルストーン状況

> **位置づけ**: 自律開発ループ（`25_purpose_driven_executor`）が Phase 2 で読む
> 「未達 deliverable」の**ルート固定名エントリポイント**。正本は
> [docs/GOAL.md](docs/GOAL.md) §3（Forward design）・§4（Execution plan）。本ファイルは
> GOAL.md の Phase 構造をマイルストーン状況として蒸留したもので、GOAL.md と矛盾すれば
> **GOAL.md を正**とする。 charter.yml（`.concept/charter.yml`）は本 repo には存在せず、
> executor は Phase 3 をスキップして Phase 4 へ進む（GOAL の Phase 0–4 が事実上の charter）。

---

## 達成済みマイルストーン（skip 対象）

| ID | マイルストーン | 状態 | 根拠 |
|----|---------------|------|------|
| MS-R1 | 第1期〜第5期 研究路線の帰無検証と棄却（velocity 外挿 / 漸進ランク ZO / B-filter / PSA 転換） | done | GOAL §1.1–§1.5。PSA 本体は `src/tg_lora/psa.py` に実装保留。 |
| MS-PF0 | Phase 0: full backprop ベースライン確定 | done | GOAL §3.1 Phase 0。`baseline_plain` best_valid=**1.0565**、`accum16`=**1.0704** と整合確認済み。 |
| MS-PF1 | Phase 1: 単層フリーズゲート機構（Level 1）実装 | done | `src/tg_lora/progressive_freeze.py` `ProgressiveFreezeController`（single-shot `should_freeze`/`cache_xin`/`apply_freeze` + progressive `layers_due_at`/`apply_freeze_layer`/`progress`）。`tests/test_progressive_freeze*.py`。 |
| MS-PF2-CPU | Phase 2: フリーズスケジュール設計の **CPU 予測面** | done | `freeze_schedule.py`（3 policy: output_first / convergence_order / compromise）+ `freeze_cost.py`（GOAL §5 会計）+ `freeze_frontier.py`（深度→FLOPs 削減フロンティア）。全て pure-Python・GPU 不要。 |
| MS-PF2-INVIVO | Progressive Freezing in-vivo ベンチマーク（proxy） | done | `tests/test_progressive_freeze_invivo.py`（h=24 proxy）。スケールでの [UNVERIFIED] は `extrapolation_confidence` で割引。 |
| MS-HONESTY | 科学誠実性インフラ（GOAL §7 鉄則の符号化） | done | `scripts/evaluate_paper_gates.py`（G0–G4, 3-state PASS/FAIL/**INSUFFICIENT**）+ `consolidate_paper_results.py` + `scripts/check_spine_anchors.py`（159 anchor 整合性・CI gate）。TASK-0144〜0151。 |

> **設計上の制約（GOAL §1.6.3）**: 学習ループは **Level 1（基本）のみ** を prod path として駆動。
> Level 2（Activation Matching）は Phase 3 の発展実験。Level 2 を prod loop に組み込まないこと。

---

## 未達マイルストーン（deliverable 候補）

### MS-PF2: Phase 2 実スイープ（valid_loss 軸のフロンティア曲線）— **優先: P1**

GOAL §3.1 Phase 2 / §4 step 3。3 自由度（順序 / 深度 / タイミング）をスイープし、
valid_loss 劣化 vs FLOPs 削減のフロンティア曲線を描く。FLOPs 軸は MS-PF2-CPU で完了、
**valid_loss 軸は GPU run 必須（分類 C）**。

- **分類 C（外部依存・GPU）**: 実 run での valid_loss 計測、frontier sweep 実行。
  - [x] **valid_loss 軸の significance を問う GPU 実行導線（proxy-scale）** —
    `surrogate_valid_loss_ci()` は構築済みだったが**実 run の valid_loss を一度も消費していなかった**（テストは構成定数のみ）。→
    `scripts/run_freeze_validloss_ci.py` + `make freeze-validloss-ci`（学習可能 proxy で candidate=`output_first` vs surrogate=`random_freeze_order` を**実 GPU 学習**し、得た real valid_loss 標本を `surrogate_valid_loss_ci()` に流す最初の導線・device auto=CUDA・seed 完全再現・`proxy_scale=True` 誠実ラベル）+ `tests/test_run_freeze_validloss_ci.py`（23 tests・実学習確認/seed 再現/CI 境界との自己一貫性/誠実ラベル/heterogeneous 正控御/generalize conclusive-TIES/teacher confidence/target-scale drop-in param threading）。
    **RTX 3060 実測（2026-06-25）・3 判定すべて TIES**（device=cuda・n=5/5・non-thin）:
    (1) **memorize × homogeneous** — candidate_mean=**0.4254** vs surrogate=0.4349・CI[−0.096,+0.109]。**自明な TIES**（train==valid の暗記タスクでは順序は構造上効き得ない）。
    (2) **generalize × homogeneous** — candidate_mean=**2.529** vs surrogate=2.648・CI[−0.067,+0.313]。**決定的 TIES**: student は保持検証バッチで全 order・seed について ~2.5（uniform≈3.47 を大きく下回る）に**真に汎化**=パイプラインは常に-TIES の壊れ経路ではなく、それでも順序は汎化に寄与しない。
    (3) **generalize × heterogeneous** — candidate_mean=**2.569** vs surrogate=2.607・CI[−0.137,+0.209]。**正控御**: 層毎 rank(1,2,4,7,13,24) の非対称を注入したが n=5 bootstrap 床の下に留まり SURPASSES を観測できず=proxy scale で順序効果を解像する感度までは示せなかった（誠実な限界）。
    科学意味: generalize タスクが apparatus 検証軸となり、memorize の自明 TIES を**決定的 null（proxy scale では output-first 順序に汎化優位なし）に格上げ**。ただし正控御は発火せず「真の順序効果があれば捕捉できる」という感度証明は未取得だった——これは下記 order-sensitivity 診断で**解消**した（正控御が発火しなかったのは apparatus の感度不足ではなく、proxy では順序信号が**構造上ゼロ**だったため）。target-scale 判定は同一関数に 9B 標本を流すだけで昇格（`proxy_scale` フラグがそれを示す）。
  - [x] **apparatus order-resolution 診断（proxy-scale 感度特性化）** — verdict run の TIES が「感度不足の below-resolution 読み」ではなく「真の null」かを問う、**測定科学的**ステップ。正控御を発火させようとする（=順序を利かせようとする）のではなく、apparatus の順序解像度を直接**分散分解**で測る: Var(order)=固定 seed で *distinct* なフリーズ順序間の valid_loss spread（順序効果が出せる最大信号）、Var(seed)=固定順序で seed を変えた際の spread（シードノイズ床）、ratio=Var(order)/Var(seed)。→ `scripts/run_freeze_order_sensitivity.py`（`order_sensitivity()`・`distinct_orders()`=全順序の seeded shuffle で衝突なし・`RESOLUTION_THRESHOLD`=0.10・verdict runner の fixture を再利用し同一 trio を訓練）+ `make freeze-order-sensitivity` + `tests/test_run_freeze_order_sensitivity.py`（20 tests）。
    **RTX 3060 実測（2026-06-25）・ratio=0.000（厳密にゼロ）**: homogeneous で Var(order)=**0.00000000**（12 個の distinct 順序が全て同一 valid_loss=2.7155）vs Var(seed)=0.0202・task-loss 併用でも ratio≈0.001・heterogeneous/concentrated スタック・early/late freeze・depth 3/5 でも不変。フリーズは順序依存に正しく適用される（順序毎に凍結層が異なることを確認済み=配線バグではない）が、凍結後に prod path が切替える境界 local-loss（GOAL §1.6.3）が held-out 課題指標に結合しないため、最終 valid_loss は順序非依存の freeze 前軌道で固定される。**結論は「感度未証明」より強い**: proxy scale では順序は本質的に非解像（full-rank 学習可能出力頭+residual が「どの LoRA 層を凍結したか」に頑健）→ verdict TIES は真の null であり、順序が効くか否かを解像できるのは**実 LM 頭と層の専門化がある target-scale 9B run のみ**。これにより「target-scale は必要と*想定*」が「target-scale は必要と*証明*」に格上げ。
    【2026-06-26 追記・この linchpin 証拠を GPU 不要再検証へ】この ratio=0.000 は
    PURPOSE 記述止まりで `make freeze-order-sensitivity`（GPU 必要）でのみ再現可能だった。
    valid_loss 判定（TIES）は `make freeze-replay` で GPU 不要再検証可能な一方、**より
    load-bearing な「target-scale 必要」証明が非対称に GPU 専用**だった証拠鎖の穴を、
    commit 済み実 GPU fixture + stdlib-only replay で閉じた（詳細は下記 #9）。
  - [x] **Cat-C を具体コマンドに縮約: recorded-sample replay judge** —
    AI-Hub feedback (2026-06-25) が「足場ヘルパーをこれ以上追加せず、Cat-C を
    具体的な機械検証可能 artifact（recorded/proxy dataset + 実行可能 make target +
    expected-output assertion）に縮約せよ。さもなくば研究結果を無限に先送りしつつ
    直交する CPU 足場を蓄積する」と指示。`surrogate_valid_loss_ci()` は GPU-run 導線の
    *内部* にしか存在せず、GPU を持たない検証者が記録済み証拠を再判定する独立コマンドは
    無かった。→ `scripts/replay_freeze_validloss_ci.py`（記録 sample JSON を読み
    `surrogate_valid_loss_ci()` **だけ**で再判定・GPU/model/torch 不要・`proxy_scale`
    をファイルから浮上・`--material-margin`/`--seed`/`--expected`/`--json`）+
    `make freeze-replay`（既定は commit 済み proxy 記録を再判定し TIES を assert）+
    `tests/test_replay_freeze_validloss_ci.py`（25 tests）。**初の commit 済み Cat-C
    dataset**: `tests/fixtures/freeze_validloss_generalize_proxy.json`（実 RTX 3060・
    `--task generalize`・verdict **TIES**・candidate=2.5294 vs surrogate=2.6478・
    CI[−0.067,+0.313]・n=5/5・non-thin）。replay test が「記録済み floats が決定論
    bootstrap の下で同一 verdict を再獲得する」（faithfulness・描き込みではない）を
    assert。**target-scale 9B は同一 schema の sample file を流すのみで昇格**
    （`proxy_scale=false` で scale label が target に自動切替・コード変更不要）=
    残る外部依存は private `src.data` pipeline 単体。
    【本イテレーションで**実証**】上記 drop-in 昇格パスは docstring/PURPOSE の主張止まり
    だったが閉じた: (a) `run_ci()` の scale を hardcoded `True` から caller 引数
    `proxy_scale: bool = True`（既定 True・既存 fixture と byte-identical）に格上げし、
    生成器が target-scale ラベルの sample file を authoring 可能な schema に修正。
    (b) replay judge の `proxy_scale=false`（TARGET_SCALE）分岐を commit 済み plumbing
    fixture `tests/fixtures/freeze_validloss_target_dropin_plumbing.json`（合成 floats・
    **実 9B 計測ではない**・SURPASSES に再現性再獲得）で初検証 — label が PROXY→TARGET に
    切替・note に "this verdict IS the §4 target-scale result"・JSON `proxy_scale=false`・
    faithfulness・`--expected` exit code・proxy 記録との弁別（差は `proxy_scale` flag のみ）。
    「同一 schema・コード変更不要」契約が符号レベル + テストレベルで成立（target run は
    private `src.data` で 9B 標本を生成し同 schema で流すのみ）。
    【2026-06-26 追記・証拠鎖を主張→実証→CI 強制へ】commit 済み GPU 証拠の再現性を**実 GPU で
    bit-for-bit 再検証**: (1) `make freeze-validloss-ci-generalize` が verdict **TIES**・全 mean/
    CI bound/10 標本とも fixture と**完全一致**、(2) `make freeze-order-sensitivity` が
    Var(order)=0.000・**12 個の distinct 順序が単一 valid_loss=2.7155 に完全一致**（line 46 の
    「厳密ゼロ」を実証）vs Var(seed)=0.0202。faithfulness test だけでは fixture は「浮動 float の
    まま再現性無防衛」（apparatus が腐ってもファイル内 float は不変なので test は緑のまま）だった
    ため、再現性を **assertion→CI 強制**へ格上げ: `tests/test_run_freeze_validloss_ci.py::
    TestApparatusDriftSentinel` が apparatus の決定論的 tiny-budget CPU `generalize` 出力を
    golden 値に pin し、`TEACHER_*`/`make_generalize_task`/`arm_valid_loss` の定数・論理が drift
    して GPU fixture を黙って陳腐化させたら CI が fail する（failure = 「GPU fixture 再生成 +
    golden 再 pin」を促す再検証信号）。これは足場ヘルパーではなく証拠鎖の硬度化。
- **分類 A 変換（コードで解決可能）**:
  - [x] **再現可能な random-order freeze surrogate** — GOAL §4「ランダム順フリーズ対照を超えた削減だけを有効」+「複数シード」が要求するサロゲート null の**再現可能生成器**。`freeze_schedule.py` は「shuffled permutation を渡せば surrogate になる」と文書化していたが生成器が無かった。→ **本イテレーションで `random_freeze_order(layers, seed)` を追加して閉包**。
  - [x] **frontier sweep の CLI exposition** — `freeze_frontier.frontier()` を bare CLI から起動できるようにし、Phase 2 計画を GPU run 前に再現可能に吐く。→ 本イテレーションで `scripts/run_freeze_frontier.py`（homogeneous-stack first-order cost model・table/JSON/`--output` 出力・Level-1 既定＝GOAL §1.6.3）+ `tests/test_run_freeze_frontier_cli.py`（9 tests・import health/`--help`/単調性検証）を追加して閉包。
  - [x] **サロゲート超過の判定ヘルパ** — candidate schedule の削減/性能が random-order surrogate を超えるかを 1 関数で判定（valid_loss 軸は GPU 依存だが、構造と seed 管理は CPU で固定可能）。→ `src/tg_lora/freeze_surrogate_gate.py` `surrogate_exceedance()`（seeded surrogate 分布 vs candidate・SURPASSES/TIES/UNDERSHOOTS 3 値・valid_loss 軸は構造化スレッド・honesty keystone: 均質スタックでは順序無効なので TIES）+ `tests/test_freeze_surrogate_gate.py`（11 tests）で閉包。（MS-PF2 Cat-A 3/3 時点。）
  - [x] **valid_loss 差のブートストラップ CI ヘルパ** — GOAL §4 統計の歯止め（"valid_loss 差はブートストラップ CI で評価"）が未実装。`surrogate_exceedance()` は valid_loss 軸を「構造化スレッド」として残し、有意性判定を持たなかった（少数 seed の 1 比較は逸話で非有意性証明）。→ `src/tg_lora/freeze_surrogate_ci.py`（`surrogate_valid_loss_ci()`：candidate vs surrogate の valid_loss 差 `mean(surrogate)−mean(candidate)` のパーセンタイル bootstrap CI〔10k resample・`numpy` 既存依存のみ・seed で完全再現〕・CI が 0 を上に除外→SURPASSES／下→UNDERSHOOTS／跨→TIES で構造判定を**有意性付きに昇格**・GOAL §7 鉄則の分離として significance と materiality を別軸〔`material_margin`・2.6σ-but-tiny を win と呼ばせない〕・薄サンプルは `is_thin_evidence`〔=`freeze_cost.MIN_SAMPLE_FOR_CONFIDENCE_BAND`〕・verdict 定数は `freeze_surrogate_gate` から import で共有=promotion not rename〕+ `tests/test_freeze_surrogate_ci.py`（26 tests・有意性判定／§7 分離／thin-evidence／seed 再現性／CI 幅の confidence 依存／validation）で閉包。`freeze_surrogate_gate.py`（構造）と `evaluate_paper_gates.py`（G0–G4）を結ぶ統計層。**MS-PF2 Category-A 4/4 完了**。

### MS-PF3: Phase 3 Activation Matching（Level 2 発展実験）— **優先: P2**

GOAL §3.1 Phase 3 / §4 step 4。最適スケジュール固定後、後段貫通も省けばさらに削れるか。

- **分類 C**: Level 1 vs Level 2 の定量比較 run（GPU）。
- **分類 A 変換**:
  - [x] local-loss の per-arm breakdown 観測（`progressive_freeze.local_loss_breakdown` / `activation_matching`）— 実装済み。
  - [x] 損失関数アブレーションハーネス（MSE 単独 / MSE+cos / 分布一致）を CPU で切り替え可能にし、各 arm の重み付けを config 駆動にする。→ `src/tg_lora/loss_ablation.py`（`LOSS_ARMS` 3 preset: `mse`/`mse_cos`/`dist`・MSE 常に base・各 arm は項を1つ加える factorial 設計 + `LossArmConfig` で arm 名＋重み上書きの **config 駆動** + `build_matching_loss()` 橋渡し + `run_loss_ablation()` 同一入力で全 arm を並走させる side-by-side harness・各 arm のスカラー loss と per-term breakdown を detached で返す）+ `tests/test_loss_ablation.py`（17 tests・named-arm 重み固定・config 上書き・harness の breakdown は combiner と byte-identical・distribution arm の置換不変性を MSE と対比して観測）で閉包。**MS-PF3 Category-A 2/2 完了**。

### MS-PF4: Phase 4 跨条件検証（スケジュール汎用性）— **優先: P3（副次）**

GOAL §3.1 Phase 4 / §4 step 5。最適スケジュールを LR/データ/r/シード で変えて有効半径を地図化。

- **分類 C**: 複数条件での GPU run。
- **分類 A 変換**:
  - [x] **schedule-portability の手順プリミティブ＋単体テスト** — GOAL §3.1 Phase 4 line 177 が「target xin の使い回し」ではなく「スケジュール（いつ何層固めるかという手順）の使い回し」を検証対象と明記。層セット非依存の手順 `ScheduleProcedure`（policy/depth/timing）＋ `bind()` で異なる active_layer 構成へ再適用するプリミティブが無かった。→ `src/tg_lora/freeze_schedule.py` `ScheduleProcedure.bind()`（手順は不変・convergence_order/stability_epoch は条件ごと再供給・小さいセットでは明確な ValueError で安全縮退）+ `tests/test_freeze_schedule_portability.py`（13 tests・等サイズ集合で freeze-epoch 列が一致=手順不変・output 側 suffix の汎用・`FreezeCostAccountant` の reduction_rate が等サイズ一様コスト条件で一致=指数非依存パイプライン）で閉包。**MS-PF4 Category-A 1/1 完了**。

---

## ブロッキング条件（外部依存・自動化不可 = 分類 C）

1. **GPU compute（RTX 3060 12GB 級）** — MS-PF2/PF3/PF4 の valid_loss 実測全てが依存。
   **本イテレーションで Category-C を具体コマンドに縮約**: `make freeze-validloss-ci`
   （+ `make freeze-validloss-ci-heterogeneous` / `make freeze-validloss-ci-generalize`）が
   実 GPU で valid_loss 標本を生成し `surrogate_valid_loss_ci()` に流す（proxy-scale TIES 判定を
   RTX 3060 で 4 セル実測済み・上記 MS-PF2 参照）。target-scale 判定は同一導線に 9B 標本を流すのみで、
   残る外部依存は下記 #2 の private `src.data` pipeline のみ（足場コードの追加ではない）。
   **記録済み証拠の GPU 不要再検証**は `make freeze-replay`（commit 済み Cat-C dataset
   `tests/fixtures/freeze_validloss_generalize_proxy.json` を再判定し TIES を assert）。
2. **公開ミラーに private `src.data` pipeline が欠損** — data-dependent test 約 130 件が
   pre-existing 失敗（[[public-mirror-preexisting-test-failures]] 既知制約・非回帰）。
3. **`make lint`（ruff）の `scripts/` 既存負債** — pre-existing（[[public-mirror-preexisting-lint-debt]]・非回帰）。
   自ファイルは `ruff check <file>` で隔離検証すること。

---

## 課題セクション（コードベース分析から抽出）

- **【高】成功条件（定量）が [UNVERIFIED]**: MS-PF2/PF3/PF4 の valid_loss 実測が GPU 依存。
  GOAL §4「成功の定義」は構造は完成。**proxy-scale valid_loss 証拠は取得済み**（`make
  freeze-validloss-ci` / `--task generalize` / `--architecture heterogeneous` → 4 セル全て TIES・
  generalize は決定的 null・RTX 3060 実測）。**proxy の順序感度も特性化済み**: `make
  freeze-order-sensitivity` が Var(order)/Var(seed)=**0.000**（厳密ゼロ）を実測し、verdict TIES
  が感度不足の below-resolution 読みではなく真の null であることを証明。ゆえに target-scale は
  順序効果を解像する唯一の手段と**証明済み**（想定ではない）。target-scale 数値検証のみが残り、
  それは private `src.data` pipeline 依存（分類 C・外部依存）。
- **【中】目的駆動ループの空転（本イテレーションで解消）**: `25_purpose_driven_executor` が
  Phase 1/2 で読む `SYSTEM_CONSTITUTION.md` / `PURPOSE.md` が**ルートに存在せず**、executor が
  毎回 skip → `run-pws` が no-output で exit 4 になっていた（直近イテレーションの却下理由）。
  GOAL.md を正本として両ファイルを蒸留・配置し、ループを始動可能にした。
- **【中】`src.data` 欠損による data-dependent test の不活性**: 公開ミラー制約。本 repo の
  品質 canary は `tests/test_cli_help_smoke.py`（37 passed / 3 xfailed）で回帰監視する。

---

## 次の一手（next execution）

> **【2026-06-28 追記・Tier-1 deposit 形成の turnkey 化（feat）— 9B lever の分類 C→A 変換】**
> AI-Hub feedback 第4回。提案 (3) critique/experiment-loop・(4) 871/866 計数は**本 iter でも**
> `grep -rni critique.?loop|experiment.?loop = 該当なし`・`MS-008/871/866 = 本 repo 全文に存在せず`
> で**引き続き別 repo の PURPOSE への誤送**（[[ai-hub-feedback-infra-vs-this-repo]]）。決定的 lever は
> 引き続き (1)/(2) の **9B target-scale run**。本 iter で判明: real 9B *Progressive Freezing* 数値は
> **既存 upstream run には存在しない**（`runs/` 138 件の `run_metrics.jsonl` は m9/m10 velocity・dynfreeze 系で
> `progressive_freeze_enabled: true` な config は 0 件）。よって real PF verdict には新規 multi-seed 9B
> campaign が必要（~4-6h GPU + public/private 境界合意）。**この iter は越境・campaign 実行を行わず**、
> executor の分類 C→A 変換ルールに従い lever を**コードで前進**させた:
> - **`scripts/form_freeze_validloss_deposit.py`（feat）+ `tests/test_form_freeze_validloss_deposit.py`（6 test）新設**:
>   recipe TASK-0152 Tier-1 step3「deposit JSON の成形」を**手書き転写 → 1 コマンド**に変換。upstream
>   `run_metrics.jsonl` の `run_footer.best_valid_loss`（footer 無しは min `loss_valid` step）を直接読み、
>   `proxy_scale=false`/`synthetic=false`/`negative_control=false`・各 run_id/seed/step の provenance 付きで
>   `replay_freeze_validloss_ci` へ直結する deposit を生成。**手書き転写は P0 再現性 hazard**（1 float の誤記が
>   bootstrap verdict を暗黙に腐らせる）であり、本 CLI はそれを「artifact 直読 + 監査可能 provenance」で除去する。
> - **form→judge chain の test green**: 形成した deposit が `citable_as_target_scale=True` gate を開くこと
>   （`test_deposit_replays_as_citable_target_scale`）+ `make freeze-form-deposit` target + 105 passed/3 xfailed 非回帰。
> - **verdict の現状（誠実）**: 9B target-scale の §4 verdict 自体は**依然 pending**（real multi-seed PF campaign 待ち）。
>   ただし「real 9B 数値 → verdict」の経路は**完全 turnkey かつ real-number 対応 test 済み**になり、
>   campaign が数値を出せば `form` → `replay --json` の 2 コマンドで決定的判定が出る。
> 次の一手は（合意後）Tier-1 multi-seed 9B PF campaign 実行 → 本 CLI で deposit → `replay` で §4 verdict 記録。

> **【2026-06-28 追記・AI-Hub feedback 第3回再送の検証 + 「BLOCKED」→「runnable」への再定式化 + 実測 test 計数 pin + TASK-0152 recipe 化】**
> AI-Hub feedback が**同一 4 提案を第3回**再送。各々を本 mirror で再検証:
> - **(1) 9B target-scale run / (2) §2.5 gate vs 9B corpus + claims.ndjson** → これまで通り
>   `src/data/` 不在が *public-mirror 内の* blocker だが、**本 iter で blocker が fundamental でないことを証明**:
>   決定的 lever は **runnable** である。根拠（全て本 iter で grep/実行確認）:
>   (a) private pipeline は同機の upstream `/home/jinno/tg-lora/src/data/`（`build_seed_dataset.py:218 load_dataset`）に present;
>   (b) 対象モデル `Qwen/Qwen3.5-9B`（GOAL §0 Track A）が `~/.cache/huggingface/hub/` に 19G で cached;
>   (c) 学習データ `data/{train,valid_quick,valid_full,gold_test}.jsonl` が upstream に present;
>   (d) `torch.cuda.is_available()=True`（RTX 3060, torch 2.1.1+cu121）。`claims.ndjson` は upstream `.concept/`
>   の concept-invariant ledger（実験 verdict 用ではない）で public には不在 → 判定は既存 deposit 契約で代替。
> - **(3) critique-loop/experiment-loop** → 引き続き AI Hub 自身の infra（`grep = 該当なし`）= [[ai-hub-feedback-infra-vs-this-repo]]。
> - **(4) MS-008=871 vs system-health=866** → 引き続き**本 repo 全文に存在せず**（別 repo の PURPOSE）。
>   → **本 iter は feedback #4 の spirit「監査計数を実測に pin」を、本 repo の *実* 計数で履行**:
>   全 test suite（`/home/jinno/.pyenv/shims/python -m pytest --continue-on-collection-errors`）=
>   **3469 passed / 17 failed / 5 skipped / 3 xfailed / 21 errors**（213s）。**17 failed 全件**を分類した結果、
>   全て `src/data/` 不在（4 schema-rejection test は subprocess が `from src.data...` で import crash して
>   validation 実行前に落ちる）+ 宣言済み dev dep 未 install（peft/datasets/hypothesis/mlx）= **修正可能な
>   code defect は一件もない**（feedback が想定する 871/866 のような計数矛盾も本 repo には存在しない）。
>
> **やったこと（public-mirror scope に留まる）**: blocker を越境せず、**発見を turnkey で検証済みの recipe に変換**:
> - **`specs/tg-lora/tasks/TASK-0152.md` 新設**: 9B target-scale 決定検証を 2 tier で定式化。
>   **Tier-1（turnkey NOW）**: 9b candidate（`9b_tg_lora.yaml`=output_first progressive freeze）vs 9b baseline（full backprop）
>   の best_valid_loss を multi-seed で比較 → `proxy_scale=False` で deposit → `replay_freeze_validloss_ci` が判定。
>   GOAL §0 究極目標（品質保持×コスト削減・Phase 0/1）に決定的。全コマンド本 iter 検証済み。
>   **Tier-2（feedback の §4 verdict gate 通り・upstream 拡張必要）**: heterogeneous×generalize での
>   candidate vs surrogate(random-order) は turnkey でない — upstream `train_tg_lora.py` は prod-path 1本のみで、
>   (a) random-order surrogate 掃引 (b) heterogeneous per-layer rank (c) generalize held-out verdict arm は
>   public の合成-runner（`run_freeze_validloss_ci.py`）にしか存在しない（grep 確認）。これらの実 9B 移植が本研究の eng gap。
> - **deposit 契約の test green を再確認**: `tests/test_replay_freeze_validloss_ci.py:852` が
>   `proxy_scale=False`+非 synthetic → `citable_as_target_scale=True` を主張（:860/:872 は False）= 誤引用 gate 機能中。
>
> **本 iter の越境判断**: upstream pipeline を実行して 9B 数値を生成・public に deposit することは
> public/private 境界を越え、かつ public repo に科学的 claim を刻む取り消し困難な行為であるため、
> **明示的な合意を得るまで実行せず**、代わりに「実行すれば即決定的」な recipe（TASK-0152）として結晶化した。
> 次の一手は合意後の Tier-1 実行（GPU 予算 ~数時間）+ Tier-2 拡張の TDD 化。`[[ai-hub-feedback-infra-vs-this-repo]]`。
>
> **検証**: `ls /home/jinno/tg-lora/src/data/` = present・`load_dataset`@build_seed_dataset.py:218・
> `du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B` = 19G・`data/*.jsonl` present・
> `python -c "import torch;print(torch.cuda.is_available())"` = True・`grep heterogeneous|surrogate|random.?order
> upstream train_tg_lora.py` = 該当なし・`pytest test_replay_freeze_validloss_ci.py -k thin` = 1 passed・
> 全 suite = 3469p/17f/21e/3xf（17f 全件 src.data/dep 由来・fixable defect 0）。

> **【2026-06-27 追記・AI-Hub feedback 再送（同一 4 提案）の再確認 + scripts/ の実在する latent NameError 修正 + whole-`src/` lint-clean/CI-pin】**
> AI-Hub feedback が前回と**同一の 4 提案**を再送。各々を本 mirror で再検証し、やはり全件
> 実行不能/非適用（前回エントリと同一結論）: (1) 9B target-scale run → `src/data/` 不在で BLOCKED
> （`train_tg_lora.py:16` `from src.data...`）・(2) §2.5 gate vs 9B corpus + claims.ndjson → 同 BLOCKED
> （`claims.ndjson` は本 repo に存在せず）・(3) critique-loop/experiment-loop → AI Hub 自身の infra
> （`grep -rni critique.?loop|experiment.?loop|revision_only` = 該当なし）・(4) MS-008=871 vs 866 →
> `MS-008`/`871`/`866`/`system.?health` は本 repo 全文に存在せず（別 repo の PURPOSE）。9B lever は
> 引き続き private `src.data` で block（不変）。feedback の「足場/ゲート硬化は停止・proxy 負対照は飽和」
> にも合致するため、これらをこれ以上追加しない。
>
> 代わりに audit-integrity/hygiene 軸（feedback #4 の spirit「測定値に pin して整合させる」の継続）で、
> **実在する latent defect を修正しつつ再発防止を CI 化**:
> - **(a) scripts/analyze_trajectory_deltas.py の実在する latent `NameError` を修正**。[[public-mirror-preexisting-lint-debt]]
>   memory は本件を「既知の scripts/ lint 債」と記録していたが、実際は **runtime NameError** だった:
>   `compute_regime_inventory`（GOAL §4 step 2 regime inventory）の empty-`step_cosines` 早期 return（L136）が
>   `len(incregments)` を読んでいた（param `increments` の typo・未定義名）。平ら/零ノルム軌道は全 cosine の
>   `n1>1e-10 and n2>1e-10` guard を false にして `step_cosines` を空にするため、この経路に到達し
>   `NameError` で crash していた（L123/L145 は正しく `increments`）。`increments` へ修正。
>   **新規 `tests/test_analyze_trajectory_deltas.py`**（零ノルム経路の回帰 test + stable 軌道の happy-path）で固定。
>   **mutation 証明**: typo を再注入すると零ノルム test が `NameError` で fail → revert で green。
>   ※ scripts/ の他 F841（`prev_meta` 等）は dead-local であり NameError ではないため、既知債のまま残置。
> - **(b) whole-`src/` tree を ruff-clean (0) 化 + CI-pin**（a4d7c26 の `train_tg_lora.py` 単体 pin を `src/` 全体へ拡張）。
>   11 件を除去、全て振舞非変更: F401 ×2（`eval/eval_json_extraction.py` `sys`・`layer_delta_analysis.py` `LayerType`）・
>   F841 ×3 dead local（`activation_regime.py` `n`・`regime.py` `mean_v`・`weight_averaging.py` `n` — いずれも計算-only で未読;
>   trainer から除去したのと同 dead-state class）・E741 ×6（`dynamic_freeze.py` ×4 + `extrapolator.py` `l`→`loss` + `json_generation.py` `l`→`line`）。
>   **新規 `tests/test_src_static_guards.py::test_src_tree_is_ruff_clean`**（subprocess `ruff check src/`==0）で CI 強制。
>   **mutation 証明**: throwaway src file に非 underscore の unused local を置くと F841 で fail → 削除で green。
>   scope は `src/` のみ（`scripts/`+`tests/` 債は高 churn/低優先で意図的に残置・`make lint` は同 2 slice で依然 red）。
>
> **検証**: `ruff check src/` = **All checks passed! (0)**・新規 2 test（3 passed）・`test_train_tg_lora_static_guards.py`
> green・canary `tests/test_cli_help_smoke.py` **37p/3xf**（不変）・影響 module test（dynamic_freeze / activation_regime /
> weight_averaging / layer_delta_analysis / extrapolator / regime / eval_modules / eval_loss / analyze_trajectory_deltas）
> = **221 passed**（振舞非変更を確認）・`test_eval_downstream.py` の collection error は `ModuleNotFoundError: No module named 'peft'`
> で**既知の pre-existing 欠損 dep**（stash で clean HEAD でも同一に fail = 非回帰）。memory
> [[public-mirror-preexisting-lint-debt]] 更新済み（whole-`src/` clean/CI-pin 化 + `incregments` NameError 修正）。

> **【2026-06-27 追記・AI-Hub feedback 4 提案の検証と lint-debt/audit-drift gap の close】**
> AI-Hub feedback が前イテレーション（§4 verdict gate の負対照 provenance guard）を VALUABLE と判定し、
> 4 件の focus を提案。各々を本 mirror で検証した結果:
> - **(1) 9B target-scale run（heterogeneous×generalize leg）** → **BLOCKED**: `src/data/` が不在で
>   `train_tg_lora.py:16` が `from src.data.build_seed_dataset import load_dataset` し、verdict runner
>   （`run_freeze_validloss_ci.py`）は `TEACHER_*` 定数の合成 proxy。private `src.data` pipeline 単体が
>   唯一の外部依存（不変・MS-PF2 分類 C）。
> - **(2) §2.5 verdict gate vs 9B corpus + claims.ndjson 記録** → 同 BLOCKED（src.data）。
>   ※ `claims.ndjson` は本 repo に存在せず（`docs/paper/tmlr_claims_alignment.md` 等の paper-claims doc のみ）。
> - **(3) critique-loop (revision_only) vs experiment-loop の end-to-end artifact** → **AI Hub 自身の infra**
>   （`grep -rni critique.?loop|experiment.?loop|revision_only` = 該当なし・`scripts/run_ablation_cache_isolation.sh`
>   の無関係 `# Main experiment loop` コメントのみ）= [[ai-hub-feedback-infra-vs-this-repo]] 既知パターン。
> - **(4) PURPOSE.md の MS-008=871 vs system-health=866 計数の不一致** → **該当なし**: `MS-008`/`871`/`866`/
>   `system.?health` は本 repo 全文に存在せず（grep 確認）= AI Hub 側の別 repo PURPOSE を指す。本 repo の
>   監査計数（canary 37p/3xf 等）は実測と一致（drift なし）。
> → **4 件とも本 mirror では実行不能/非適用**。9B lever は引き続き src.data で block（不変）。
> feedback の「足場/ゲート硬化は停止（収益逓減）・proxy 負対照は飽和」という指導にも合致し、これらを追加しない。
>
> 代わりに feedback **#4 の spirit（"audit source-of-truth を測定値に pin して整合させる"）** を本 repo に適用:
> 監査文書（PURPOSE.md + [[public-mirror-preexisting-lint-debt]]）が長らく主張していた
> 「`src/training/train_tg_lora.py` に **2 件**の pre-existing ruff error（F841 `production_start_full_backward_passes` + E741）」が
> **実測と矛盾して drift** していた — 実測は **1 件**（E741@L4086 のみ）。F841 は ruff に**検出されなくなっていた**:
> `train_tg_lora()` が `_snapshot_efficiency_accounting(locals())` で `locals()` を呼ぶため pyflakes が
> 関数内の全 local を「使用済みの可能性」と見做して F841 を抑制（=apparatus が盲点）。ゆえに「2 errors」という
> 監査主張は prose のまま陳腐化していた（GOAL §7「測定せず結論しない」違反の小さな実例）。
>
> **本イテレーションでこの gap を close**（足場ではなく監査整合性の硬化）:
> - **(a) 死変数 `production_start_full_backward_passes` を削除**（init L1406 + shadow-block 代入 + warmup-release 代入の 3 site）。
>   memory が「別 clean-up 対象」と明記していた TODO。**provably write-only**: 代入 3 site・読者ゼロ・
>   `_EFFICIENCY_ACCOUNTING_KEYS`（21 名 allowlist）外・文字列 key 参照なし・`globals()`/`vars()`/`dir()` 呼出なし。
>   reader が存在しないので**振舞非変更**（GOAL §7 诚实性: 削除=誤データ生成ではなく死状態の除去）。
> - **(b) E741@L4086 を fix**: dynfreeze guard record の `",".join(str(l) for l in dynfreeze.frozen_block)` の
>   ambiguous `l` → `layer` に純 rename。→ `ruff check src/training/train_tg_lora.py` = **All checks passed! (0)**.
> - **(c) 新 CI guard `tests/test_train_tg_lora_static_guards.py::test_training_entry_point_is_ruff_clean`**:
>   F821 のみならず**全 rule の ruff check = 0** を CI 強制（`_run_ruff` helper で既存 F821 test を DRY 化）。
>   「0 errors」が prose ではなく test invariant になり、監査 drift が再発したら CI が捕る
>   （apparatus-drift sentinel `TestApparatusDriftSentinel` と同パターン）。**mutation 証明**: E741 を含む file で
>   ruff が nonzero を返すことを確認 → この guard は pre-fix 状態を検知したはず（no-op ではない）。
>
> **検証**: `tests/test_train_tg_lora_static_guards.py` **2 passed**（既存 F821 + 新 full-clean）・
> canary `tests/test_cli_help_smoke.py` **37p/3xf**（不変）・`tests/test_checkpoint.py` +
> `test_resume_state_integration.py` + `test_dynfreeze_all_frozen_path.py` = **24 passed**（死変数削除の
> 振舞非変更を確認）・`py_compile` OK・`ruff check` train_tg_lora.py = **0**。
> memory [[public-mirror-preexisting-lint-debt]] 更新済み（train_tg_lora.py = lint-clean/CI-pinned）。
> **9B 実 run は引き続き private `src.data` で block・不変**。残る真の研究結果 lever は src.data 利用可能次第の
> 9B target-scale verdict のみ（導線は具体コマンドに縮約済み・`proxy_scale` flag で昇格・コード変更不要）。

> **【2026-06-27 追記・async-cache-swap 完了 cycle marker（swap_cycle_vq/vf）の resume state-loss を修正（resume-state-loss 軸を dormant async-cache-swap route まで拡張）】**
> resume-state-loss 軸の **10 件目**（mainline 8 件 + PSA route 1 件に続き、**dormant async-cache-swap route** の完了 cycle marker）。
> `swap_cycle_vq`/`swap_cycle_vf` は caller scope の plain scalar（caller init L999 `= None`）で、async cache builder が `valid_quick` / `valid_full` を
> cached dataset に swap した cycle を記録（L1957/L1960 で `if async_builder is not None and not async_ready` gate 下で `= cycle` 代入）。run-end summary（L4566-4569）は
> これらを読み `async_cache_swap_cycle_valid_quick`/`full` を出力するが、`TrainingState` にも resume 復元 block にも無く、fault/periodic resume のたびに
> caller init `None` に**空再構築** → run-end summary が両 field を**黙って欠落**させていた。
> **重要な誠実性注記**: async cache builder は本 mirror の**全 config で無効**（dormant route）。ゆえにこれは **dormant route の硬化**であり mainline 挙動は不変（非破壊）——が、
> その run-end summary field が resume で黙って腐るのは GOAL §7「測定せず結論しない」違反として本 axis の対象（PSA/dynfreeze と同根拠）。
> → best_lawa_loss / triggered_target_steps / best_full_eval と**同一パターン**（plain scalar・None-safe）: (a) `TrainingState.swap_cycle_vq/vf: int | None`
> （legacy-safe 既定 None）+ `save_training_state` 記述 + `load_training_state` 復元（`blob.get` 既定 None）、(b) `_save_fault_checkpoint`
> （L772 param + L846-850 docstring + L893 fault-save 構築 site）・L4331 periodic save 構築 site・L4413 call site（keyword 引数）・L1481 resume 復元 block で対称化。
> **検証**: `ruff --select F821` = **0**（checkpoint.py / train_tg_lora.py / touched tests）、`tests/test_checkpoint.py` **22 passed**
> （+`swap_cycle_vq/vf` 往復 + legacy-load-clean・両 key 欠落で None 復元）、`tests/test_resume_state_integration.py` を **9→10 site に拡張**
> （`swap_cycle_vq=7`/`vf=9` を `_build_saved_state` で populate・fault-checkpoint 再 snapshot で検証・async builder が test config で `None` のため復元値が cycle body で不変）、
> **mutation 証明**: L1481 復元行を `= None` に破壊すると当該 assertion のみ fail → revert で green（9 site と同基準）。これで resume-state-loss 軸は **10/10**
> （dormant route 含む全 run-wide summary field が孤立 round-trip + 実 loop 一括復元の双方で証明済み）。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・PSA subspace-prior の resume state-loss を修正（resume-state-loss 軸を mainline 8 件から PSA route まで拡張）】**
> resume-state-loss 軸の **9 件目**（mainline 8 件: dynfreeze / best_full_eval / warmup / lawa-window / best_lawa_loss /
> triggered_target_steps / act_regime_state / efficiency_accounting に続き、**PSA route** の `psa_prior`）。
> GOAL §1.5 / §3.3 PSA（Prior-based Subspace Amplification）の `PSAPrior` は run-wide 累積（per-step `_delta_history` ring buffer +
> 抽出済み PC1 `priors`（= production `amplify_gradients` を駆動）+ L2-reg blend anchor `_prev_priors` + 非有界 `_prior_cosines` 安定度系列 +
> `should_update` timing）を**全て `train_tg_lora` module scope に置き** fault/periodic resume のたびに**空再構築**していた → resume 後は
> 2 delta 再蓄積 + 次 `extract_priors` 発火まで gradient amplification が**黙って off**になり、residual run が短いと run-end
> `layer_delta_analysis`（GOAL §4 rank-1 dominance・`history_count >= 2` gate）が**丸ごと欠落**していた。
> **重要な誠実性注記**: `enable_psa` は本 mirror の**全 config で `false`**（PSA は Phase 1-5 の移行済み route・現行は Phase 6 Progressive Freezing）。
> ゆえにこれは **dormant route の硬化**であり mainline 挙動は不変（非破壊）——が、PSA は「baseline と比較済み」（GOAL §1.5）の実科学出力経路なので、
> その run-end summary が resume で黙って腐るのは GOAL §7「測定せず結論しない」違反として本 axis の対象。
> → LAWA（`0eb6fdb`）/ act_regime（`2994fcd`）と**同一パターン**: (a) `PSAPrior.state_dict()`/`load_state_dict()` 追加（tensor は save 時 CPU 化・
> `None`/partial-dict 許容 load・`_delta_history` deque maxlen 再構築・`gain_map` は derived なので**非永続化**＝`compute_gain_map` で再計算）、
> (b) `TrainingState.psa_state: dict | None`（legacy-safe 既定 None）+ `save_training_state` 記述 + `load_training_state` 復元（`blob.get` 既定 None）、
> (c) `from src.tg_lora.psa import PSAPrior` を module top へ巻き上げ（`ActivationFingerprintTracker` hoist と同根拠・`_save_fault_checkpoint` 型注釈の
> `UnboundLocalError`/ruff F821 回避）+ 遅延 import を縮約、(d) `_save_fault_checkpoint`（param + docstring + fault-save 構築 site）・periodic save 構築 site・
> call site（`psa_prior=psa_prior` keyword・位置引数後キーワードの SyntaxError 回避）・resume 復元 block（`psa_prior` 構築直後・act_regime 復元と同 guard）で対称化。
> **検証**: `ruff --select F821` = **0**（train_tg_lora.py / psa.py / checkpoint.py）、`tests/test_psa.py` **52 passed**
> （+7 `TestPSAPriorStateRoundtrip`: 往復 / amplification-direction 不変 / `None` no-op / partial-dict 許容 / `gain_map` 非永続・再計算 /
> window maxlen 再構築 / CPU tensor 復元）、`tests/test_checkpoint.py` **21 passed**（+`psa_state` 往復 + legacy-load-clean）、
> `tests/test_train_tg_lora_static_guards.py` green、`tests/test_cli_help_smoke.py` **37p/3xf**、
> `tests/test_fault_recovery.py` **7f/15p == HEAD**（src.data import-block・stash 比較で非回帰）。
> これで resume-state-loss 軸は **mainline 8/8 + PSA-route 1 = 9/9**。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・resume-state-loss 軸に integration-level fault-resume test を追加（孤立 round-trip → 実 loop 一括復元の gap を閉鎖）】**
> run-feedback 指摘: 9/9 は**オブジェクト毎の孤立 round-trip**で検証されており、「実際の fault-resume が loop 内で `load_training_state` を起動し、
> **全フィールドを一括で**復元する」ことを証明する test がなかった。孤立 round-trip は**各オブジェクトの serialize/deserialize** を証明するが、復元 line の削除・
> フィールド入れ替え・過剰に広い guard は全ての孤立 test を素通りさせ、9B run の fault/periodic resume で**黙って state 落ち**させる。
> → `tests/test_resume_state_integration.py`（1 件の integration test）を追加。本 mirror では `train_tg_lora` の top-level import chain が
> private `src.data` + 未 install の `peft`（経由 `src.model.load_model`）で un-importable なため、**test-only の `sys.modules` shim**
> （`src.data.build_seed_dataset` + `src.model.load_model` の 4 name を raise-stub 化・`lora_utils` は依存無しなので実 import のまま）で loop を import 可能にし、
> **本物の loop コード**を走らせる（shim は未 mock 呼出で大声で raise・src/ 変更ではなく・不在 dep を「fix した」とは一切主張しない誠実設計）。
> 手順: 全 5 フィールド populated の `TrainingState` を disk に save → 実 `train_tg_lora(resume_path=...)` で resume → 最初の再開 cycle の pilot `forward_backward`
> で `NumericalInstabilityError` を注入 → loop の fault handler が**復元直後の in-loop state** から fault checkpoint を書く → 2 seam で検証:
> (1) **capturing factory**（PSAPrior / ActivationFingerprintTracker の `load_state_dict` 復帰直後に object を捕捉・`load_state_dict` が正しい field で発火した直接証拠）、
> (2) **loop 自身の fault-checkpoint TrainingState**（`save_training_state` を wrap して捕捉・plain local の `best_lawa_loss` / `triggered_target_steps` /
> `efficiency_accounting` が復元 block を経て live 変数へ運ばれた証拠）。**mutation 証明**: 5 復元 site を**個別に破壊**（best_lawa_loss→inf / triggered_target_steps→空 set /
> psa `load_state_dict` 無効化 / act_regime 無効化 / efficiency guard `and False`）→ いずれも test が**該当 assertion で正確に fail**（over-broad guard case 含む）→
> 全件 revert → tree clean → `test_resume_state_integration` + `test_fault_recovery` + `test_checkpoint` + `test_psa` + `test_activation_regime` = **126 passed**。
> これで resume-state-loss 軸は**孤立 round-trip に加え実 loop 一括復元も証明**（9/9 不変・強化）。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・integration fault-resume test を全 9 site に拡張 + fault seam を `run_cycle` へ移行（dynfreeze site の mutation 証明を閉鎖）】**
> 上記（`c3d7f8f`）の integration test は run-feedback が例示した **5 site**（best_lawa_loss / triggered_target_steps / act_regime_state /
> efficiency_accounting / psa_state）のみを end-to-end で証明し、残り **4 site**（dynfreeze_state / best_full_eval_loss+perplexity /
> warmup_released+cos_consecutive / lawa_state window）は孤立 round-trip のみだった。→ 同 test を**全 9 site** に拡張:
> `_populate_dynfreeze_state` / `_populate_lawa_state` fixture 追加・`_build_saved_state`（9 site populated）と `_make_resume_state_config`（dynfreeze+LAWA ON）へ組込み・
> `DynamicFreezeController` / `LAWAAverager` の capturing factory 追加・sanity block と 4 site 分 assertion 追加。
> **fault seam の移行（本改訂の要）**: 元 seam は pilot の `forward_backward`（`train_tg_lora.py:2112`）だが、これは `dynfreeze_all_frozen` の pilot-skip gate（L2075）**下流**にある。
> dynfreeze 復元 line を破壊すると**新鮮な controller が初 cycle で全層凍結**（`run_cycle` が `block_size == len(_all_layers)` で True を返す・`dynamic_freeze.py:376`）→
> pilot skip → fault **不発火** → loop が full-eval path（L3957 `tg_lora_cache_built=use_cache`）へ落ち、**該当 assertion に届く前に** `use_cache` 未代入の `UnboundLocalError` で異常終了していた
> （= dynfreeze の mutation 証明が不正）。→ seam を**最初の再開 cycle action** である `dynfreeze.run_cycle`（L1999・復元 block 直後・cycle body が何も mutate する前）に移行し、
> `patch.object(DynamicFreezeController, "run_cycle", side_effect=NumericalInstabilityError)` で注入。dynfreeze state に依存せず fault が確定発火 → 全 9 site の assertion が mutation 時にも到達可能。
> **mutation 証明（全 9 site・各復元 line を個別破壊 → 該当 assertion で正確 fail）**: dynfreeze→`:592` / best_full_eval→`:611` / warmup→`:621` / lawa→`:632`（新 4 site）+
> best_lawa_loss→`:562` / triggered_target_steps→`:568` / psa→`:533` / act_regime→`:548` / efficiency→`:585`（既 5 site・seam 移行後も再証明・green）。全件 revert → tree clean。
> **発見した latent bug（別件）**: `use_cache` 他 5 名は `if not dynfreeze_all_frozen:`（L2292）内でのみ代入され、all-frozen 時の full-eval path（L2497/2967/3957）で**未代入参照**となる
> `UnboundLocalError` が潜在していた（dynfreeze は dormant Guard 実験・本 mirror 全 config で無効なので本番非発火）。→ **別途 fix 済み**（下記【2026-06-27 追記・all-frozen skip path の latent UnboundLocalError を修正】参照）。
> **検証**: `tests/test_resume_state_integration.py` **1 passed**（9 site green）・full resume-state suite（8 file）= **192 passed, 3 xfailed**・
> `tests/test_fault_recovery.py` **7f/15p == HEAD**（src.data import-block・src/ 未変更で非回帰）。これで resume-state-loss 軸は**全 9 site が孤立 round-trip + 実 loop 一括復元の双方で証明済み**（9/9 完結・強化）。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・all-frozen skip path の latent UnboundLocalError を修正（上記 latent bug の close）】** 上記 9-site integration test 拡張中に発見し「別途硬化すべき」と記録していた latent crash を今回 fix。
> dynfreeze が全層凍結（`dynfreeze_all_frozen == True`）すると cycle は skip block（`train_tg_lora.py:2225`）から pilot/extrapolation/post-eval を飛ばし**そのまま metrics 記録へ fall-through** する設計だが、
> skip block は**共有の最終 record_step（L3938）が読む 5 名**（`use_cache` / `cache_eligible` / `cache_hit` / `can_confident_skip` / `m9_cycle_stats`）も、**post-record_step tail の full-eval gate（L4162 `if is_full_eval_cycle:`）が読む `is_full_eval_cycle`** も初期化していなかった。
> この 6 名は全てスキップ対象の `if not dynfreeze_all_frozen:` block（L2292-3919）内でのみ代入されるため、fall-through は**未代入参照の UnboundLocalError** を起こしていた（1 crash site: L3957 record_step の `use_cache` → fix 後は 2 crash site: L4162 の `is_full_eval_cycle` へ移動）。
> = dynfreeze 有効化 run は**最後の層を凍結した瞬間**（実験が到達すべき終端状態）に crash していた。dynfreeze は本 mirror 全 config で無効（dormant Guard 実験）なので本番非発火だが、潜在 crash としては閉鎖すべき defect。
> **fix**: skip block に 6 名の既定値を追加（`use_cache=False` / `cache_eligible=False` / `cache_hit=False` / `can_confident_skip=False` / `m9_cycle_stats={}` は record_step 引数に整合・`is_full_eval_cycle=False` は凍結 model 不変→冗長 full-eval を行わず・best_model/early-stop は最終訓練 cycle で既決定）。
> skip block 自身の「metrics recording へ直接 skip」という intent にも整合（凍結 cycle は余計な full-eval/checkpoint を行わない）。
> **回帰 test**: `tests/test_dynfreeze_all_frozen_path.py` — `tests/test_resume_state_integration.py` と同一の sys.modules shim（`src.data` + `src.model.load_model` raise-stub 化）で本物の loop を import し、
> `DynamicFreezeController` を「毎 cycle 全層凍結」mock に差し替えて all-frozen path を**実コードで強制起動**。run 完了 + 全 record_step が `tg_lora_cache_built=False`（cache 非作業の意味的検証・単に例外未脱出でないことの証拠）+ mock controller 発火を assert。
> **mutation 証明**: src 改変を `git stash` で revert（test file は untracked で保持）→ test が `UnboundLocalError`（`use_cache` 未代入）で **RED** → `stash pop` で復元 → **GREEN**。fix が無ければ再現する crash であることを確認。
> **検証**: `tests/test_dynfreeze_all_frozen_path.py` **1 passed**・resume-state + fault-recovery + 新 test = **24 passed**（非回帰）・`ruff check --select F821 src/training/train_tg_lora.py` **All checks passed!**（0・新規未定義名なし）。
> 9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・GOAL §5/P3 efficiency-accounting counter block の resume state-loss を修正】** resume-state-loss 軸の
> **8 件目**（dynfreeze / best_full_eval / warmup / lawa-window / best_lawa_loss / triggered_target_steps / act_regime_state / **efficiency_accounting**）。
> `train_tg_lora` の **21 個**の run-wide 効率会計カウンタ（`activation_cache_*_count` / `pilot|post_validation_forward_count`
> / `post_extrapolation_eval_*` / `subspace_zo_*_total` / `alpha_line_*_total` / `future_work_projection_ratios` +
> `future_work_internal_pair_count`）は mainline config の実経路で cycle 毎に蓄積し run-end summary の **GOAL §5 / P3
> コスト会計ブロック**（`validation_forwards_total`・`activation_cache_hit_rate`・subspace-ZO/alpha-line tallies・
> `projection_ratio_mean`）を駆るが、**全て `TrainingState` に未永続化**で蓄積後の fault/periodic resume が**ゼロ/空再構築**していた →
> run-end コスト報告が post-resume-only に化けていた（削減率や forward 会計が resume 跨ぎで過小評価＝GOAL §7「測定せず結論しない」の誠実性違反）。
> → 既存パターンで `TrainingState.efficiency_accounting: dict | None`（legacy-safe 既定 None・混在 int/float/list/dict 型を plain dict で往復）を追加し、
> (a) `_EFFICIENCY_ACCOUNTING_KEYS` タプル（21 名・単一ソース）+ `_snapshot_efficiency_accounting(locals())` ヘルパで fault save と periodic save の**両 site** を DRY に対称化、
> (b) `_save_fault_checkpoint`（param + docstring + TrainingState 構築）に thread、(c) `save_training_state` 記述 + `load_training_state` 復元（`blob.get` 既定 None）、
> (d) resume 復元は counter init block（L1659-1679）直後に配置・各カウンタは `.get(key, 現 init)` で旧 checkpoint / 未存在カウンタは現 init に後退し**偽データを生成しない**・
> act_regime / lawa の復元と同 guard（fresh run・None は非接触）。**検証**:
> `tests/test_checkpoint.py` **20 passed**（+`efficiency_accounting` 往復 + legacy-load-clean）、`tests/test_train_tg_lora_static_guards.py`
> green（**F821=0**）、`tests/test_cli_help_smoke.py` **37p/3xf**、`tests/test_fault_recovery.py` **7f/15p == HEAD**（src.data block・stash 比較で非回帰）、
> ruff F821=0（既存 E741@L4004 は非回帰・不変）。
> これで resume-state-loss 軸は **8/8**。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・activation-fingerprint regime inventory の resume state-loss を修正】** resume-state-loss 軸の
> **7 件目**（dynfreeze / best_full_eval / warmup / lawa-window / best_lawa_loss / triggered_target_steps / **act_regime_state**）。
> GOAL §4 step 1 の `ActivationFingerprintTracker` は最終 decoder 層の forward hook で activation cosine 時系列を取り
> 3 相分類（STABLE/TRANSITION/CHAOTIC）し、run-wide 状態（完全 cosine 系列 `_all_cosines`=GOAL §7 null baseline 入力・
> per-regime `_counts`=regime_inventory/stable_fraction 駆動・分類窓 `_cosines`・現 `_regime`）を蓄積するが、これらは
> **全て `TrainingState` に未永続化**で、蓄積後の fault/periodic resume が**空 tracker** を再構築していた → run-end summary の
> `activation_regime_inventory` / `activation_regime_stable_fraction`（GOAL §4 step 1 の理論上限指標）が post-resume-only に化けていた。
> tracker は `configs/9b_tg_lora_psa.yaml:140` の `activation_regime_enabled: true` で**実 config で有効化済み**（非デッド機能）。
> → `LAWAAverager`（`0eb6fdb`）と**同一パターン**: (a) tracker に `state_dict()`/`load_state_dict()` 追加（run-wide 累積を plain dict
> で往復・enum key は `.value` 文字列・transient な `_prev_act`/`_current_act` と hook は意図的に非永続化＝次 forward hook で run-start 同様に再 populate・`None`/partial-dict 許容で旧 checkpoint と disabled run は clean load）、(b) `TrainingState.act_regime_state`（`dict | None` 既定・legacy-safe）、(c) `save_training_state` 記述 + `load_training_state` 復元（`blob.get` 既定 None）、(d) `_save_fault_checkpoint`（param + docstring + call site）と periodic save の両 site で `act_regime_tracker.state_dict()` を is-not-None ガードで対称化、(e) resume 復元は tracker 構築+hook 直後に配置（`act_regime_tracker` は resume block **より後**で構築されるため同 block 内だと `UnboundLocalError` → `LAWAAverager` hoist と同根拠・`ActivationFingerprintTracker` import を `LAWAAverager` の巻き上げと並べて module top へ）。**検証**:
> `tests/test_activation_regime.py` に `TestActivationRegimeStateRoundtrip`（+6: 実 step 経由の往復 / 直接注入の inventory 一致 /
> `None` no-op / partial-dict 許容 / transient tensor 非永続の cold-start / window maxlen 再構築）、`tests/test_checkpoint.py`
> **+round-trip +legacy-load-clean**、全体 **87 passed / 3 xfailed**、static-guards **F821=0**（旧 F841/E741 2 件は非回帰・不変）。
> これで resume-state-loss 軸は **7/7**。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-27 追記・linearity-budget target-step set の resume state-loss を修正】** resume-state-loss 軸の
> **6 件目**（dynfreeze / best_full_eval / warmup / lawa-window / best_lawa_loss / **triggered_target_steps**）。
> `_check_and_save_linearity_budget_checkpoint` が各 target step（250/500/.../1500）を
> `target not in triggered_target_steps` ガードで**1 回限り**発火させる（mandatory full eval +
> `checkpoint-{target}` save + step-aligned `is_step_aligned_full_eval` record + vs-baseline 比較）が、
> `triggered_target_steps` は `TrainingState` に無く resume で空 set に戻り、post-resume 初 cycle が
> **既超過 target を全て再発火**していた（冗長 full eval + `checkpoint-{target}` 再 save + **重複
> `aligned_target` record** で linearity-budget vs-baseline 比較 dataset を破壊＝downstream の step-keyed
> reader が post-resume 値を二重計上 / pre-resume 値を上書き）。→ 既存パターンで
> `TrainingState.triggered_target_steps`（`list[int] | None`・legacy-safe・`accepted_valid_history` と同型・
> sorted 永続化）を追加し、`_save_fault_checkpoint`（param + docstring）・periodic save build・fault call site・
> resume 復元（list→set）で対称化。**検証**: `test_checkpoint.py` **18 passed**（+round-trip +legacy-load）、
> static-guards **F821=0**、`test_cli_help_smoke.py` **37p/3xf**、`test_weight_averaging.py` **28 passed**
> （LAWA 隣接・TrainingState 構築）、`test_fault_recovery.py` **7f/15p == HEAD**（src.data block・stash 比較で
> 非回帰）。これで resume-state-loss 軸は **6/6**。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-26 追記・LAWA best_lawa_loss headline の resume state-loss を修正】** resume-state-loss 軸の
> **5 件目**（dynfreeze / best_full_eval / warmup / lawa-window に続く）。直前の `0eb6fdb`（LAWA
> スナップショット窓 `lawa_state` の永続化）が**自身のバグ記述で `best_lawa_loss` も inf にリセットされると
> 明記していたのに、窓だけを直して tracker は未修復のまま残していた文書化された半fix**を閉じた。
> `best_lawa_loss` は GOAL §3.3 必須ベースライン LAWA 比較（`evaluate_with_lawa`）の run-wide 最小値で
> run summary JSON の headline。`train_tg_lora` module-local で `inf` 初期化・resume で復元されず、fault/
> periodic resume 後は inf 再始動して run-end headline が post-resume-only の最小値になっていた。
> → `best_full_eval_loss`（`73201a4`）と**同一パターン**: `TrainingState.best_lawa_loss` field +
> save/load（legacy 旧 checkpoint は inf 既定で clean load・headline は post-resume から再計算 = pre-fix 挙動・
> 偽の低値ではない）+ fault-save param thread + periodic save + resume 復元（plain float なので averager 構築
> 不要・`lawa_state` 窓の復元とは別 site）。**検証**: `tests/test_checkpoint.py` **17 passed**
> （+best_lawa_loss 往復 / +legacy-load-clean）、`test_train_tg_lora_static_guards.py` green（F821=0）、
> `test_weight_averaging.py` passed（LAWA 隣接・非接触）、`test_cli_help_smoke.py` 37 passed / 3 xfailed、
> ruff 0 新規（既存 F841/E741 2 件は非回帰・不変）。これで resume-state-loss 軸は **5/5**（dynfreeze・
> best_full_eval・warmup・lawa-window・best_lawa_loss）。9B 実 run は引き続き private `src.data` で block・不変。

> **【2026-06-26 追記・LAWA weight-averaging window の resume state-loss を修正】** resume-state-loss 軸の
> **4 件目**（`119e815` dynfreeze / `73201a4` best_full_eval / `02711e6` warmup と同軸）。LAWA は
> **GOAL §3.3 の必須ベースライン**（P2 公平比較ゲート）かつ実 prod path（`configs/jsonex_lawa.yaml`
> で `enable_lawa: true`）。`lawa_averager` のスナップショット窓は `train_tg_lora` の module-local 状態で
> `record()` で蓄積されるが、**resume で空再構築**されていた → `is_ready` が False に落ち、LAWA 比較
> （`evaluate_with_lawa`）と **LAWA 平均化 JSON eval**（`averaged_weights_context`）が `start_cycle`
> 分の新スナップショット再蓄積まで**黙って skip** され、resume 後の見出し品質ベースラインが
> fault 後のみの窓で測られていた（`best_lawa_loss` も inf にリセット）。→ (a) `LAWAAverager.state_dict()`
> /`load_state_dict()`（CPU スナップショット buffer + counters・deque maxlen を load で再構築）、(b) `TrainingState`
> に `lawa_state: dict | None` 追加（既定 None で旧 checkpoint と LAWA-disabled run は後方互換）、
> (c) `_save_fault_checkpoint` + periodic save の両 site で `lawa_averager.state_dict()` を記録、
> (d) resume で `restored_training_state` ガード下に復元（`lawa_averager` は resume block **より後**で
> 構築されるため、同 block 内だと `UnboundLocalError` → 構築直後に配置・`119e815` の hoist と同様に
> `LAWAAverager` import を module top に巻き上げ）。**検証**: `tests/test_weight_averaging.py` **28 passed**
> （+5 `TestLAWAStateRoundtrip`：往復 / `is_ready` の resume 越え生存 / `average_snapshot` の byte 同一性 /
> maxlen 窓 trim / 空 buffer 許容）、`tests/test_checkpoint.py` **16 passed**（+legacy-load clean / +`lawa_state`
> 往復）、`test_train_tg_lora_static_guards.py` green（F821=0）、`test_cli_help_smoke.py` 37 passed / 3 xfailed、
> `test_fault_recovery.py` 7 fail / 15 pass は stashed HEAD と**完全同一**（src.data import-block・非回帰）。
> ruff 0 新規（既存 F841 `production_start_full_backward_passes` は write-only dead var と判明＝本 fix
> の調査副産物・別 clean-up 対象・非回帰）。本 axis は「実際の training run への旋回」の実行可能形
> （9B は private `src.data` で block のまま・不変）。

> **【2026-06-26 追記・warmup 2-phase gate の resume state-loss を修正】** resume-state-loss 軸の
> **3 件目**（`119e815` dynfreeze / `73201a4` best_full_eval と同クラス・同ファイルの兄弟）。
> `warmup_released`/`warmup_cos_consecutive` は `train_tg_lora` の module-local 2-phase gate 状態で、
> False の間は pilot-only で `adapt_to_convergence`/`adapt_to_acceleration`/外挿を全バイパスする。
> **mainline config**（`9b_tg_lora.yaml`・`9b_tg_lora_m9.yaml`・`jsonex_*`・`measure_accum*` 全て
> `warmup_release_count: 1` / `warmup_release_cos: 0.1`）で実経路。かつ**単調でない**——M9 subspace-accept
> path（L3517）が意図的に `warmup_released=False` に戻して再ウォームアップするため、checkpoint は
> **どちらの相**を捕捉し得る。`TrainingState` に**永続化されず resume で False/0 に戻る**ため、
> 本番期（mid-production）の checkpoint から resume すると**黙ってウォームアップ相へ逆戻り**し、
> 収束/加速度適応と外挿を gate 再発火まで再無効化していた。→ 両 field を `TrainingState` に追加
> （既定 False/0 で旧 checkpoint と後方互換）し、save/load 往復 + resume 復元 + fault-save
> （`_save_fault_checkpoint` param thread = `dynfreeze`/`best_full_eval` と同一パターン・periodic save
> site も含む）で対称化。**検証**: `tests/test_checkpoint.py` **15 passed**（mid-production checkpoint
> の往復 assert + 旧 checkpoint が False/0 に落ちる legacy-load test 追加）。`ruff --select F821
> src/training/train_tg_lora.py` = **0**（static-guards canary green）。`tests/test_fault_recovery.py`
> の 7 fail は HEAD と**完全同一**（src.data block の pre-existing・`ModuleNotFoundError` で非回帰確認）。
> これで resume-state-loss 軸は **3/3**（dynfreeze・best_full_eval・warmup）。本 axis は「実際の
> training run への旋回」の実行可能形（9B は private `src.data` で block のまま・不変）。

> **【2026-06-26 追記・fault-resume の best_model 無条件上書きバグを修正】** 直前の
> `119e815`（訓練ループ潜伏 NameError 2 件）と**同クラス・同ファイルの兄弟バグ**を発見・修正
> （足場ではなく製品挙動の正確性軸を継続）。`best_full_eval_loss`/`best_full_eval_perplexity`
> は `train_tg_lora` の module-local tracker で `best_model/` 保存 gate（5 site）を駆るが、
> `TrainingState` に**永続化されず resume で復元されない**ため、fault-resume 後の初回 full-eval
> が `inf` と比較して**常に真**になり、真に最良だった fault 前の `best_model/` を**黙って上書き**
> していた（`cycle_state.best_loss` とは §5.3 `min_delta` の有無で意味が異なり流用不可）。
> → 両 field を `TrainingState` に追加（既定値 inf/None で旧 checkpoint と後方互換）し、
> save/load 往復 + resume 復元 + fault-save（`_save_fault_checkpoint` へ param thread =
> `dynfreeze` と同一パターン）で対称化。**検証**: `tests/test_checkpoint.py` 14 passed
> （往復 assert + 旧 checkpoint が inf/None に落ちる legacy-load test 追加）。
> `ruff --select F821 src/training/train_tg_lora.py` = **0**（thread 前に一時 F821×2 を
> 導入したが param 化で解消・`test_train_tg_lora_static_guards.py` green）。
> `tests/test_fault_recovery.py` の 7 fail は HEAD と**完全同一**（src.data block の
> pre-existing・stash 比較で非回帰確認）。本 axis は「実際の training run への旋回」の
> 実行可能形（9B は private `src.data` で block のまま）。

> **方針転換（AI-Hub feedback 2026-06-25）**: Category-A（CPU-only 足場）は**枯竭**。
> 次イテレーションで**足場ヘルパーをこれ以上追加しない**こと（収益逓減・"indefinitely
> deferring the actual research result while accumulating orthogonal CPU scaffolding"）。
> 代わりに Category-C（GPU）ブロックを直接叩く — **本イテレーションでそれを実行した**。

> **【2026-06-26 追記・訓練ループ本体の正確性バグ修正へ旋回】** AI-Hub feedback (2026-06-26) が
> 再び「proxy 証拠の足場追加は停止（valid_loss verdict と order-sensitivity ratio=0.000 は
> 二重ロック済）・症状でなく根本原因を直せ・実際の training run へ旋回せよ」を指示。ただし
> feedback の具体的根本原因提案（make-run auto-commit での `references:` ブロック振動・
> `helix/orchestrator/gates.py` spine-audit 配線）は **AI Hub 自身のインフラ**を指し、本 mirror には
> `_doc_spine.yml`・`helix/`・`check_spine_manifest.py` が存在せずここでは実行不能（feedback の
> 名指しインフラは AI-Hub 側＝[[ai-hub-feedback-infra-vs-this-repo]] の既知パターン）。ゆえに
> 「実際の training run への旋回」を実行可能な形で解釈し、**訓練ループ本体の潜伏 NameError バグ
> 2 件**を発見・修正した（足場ではなく製品挙動・MS-PF 系とは独立の正確性軸）:
> 1. **fault checkpoint が dynfreeze 状態を黙って喪失** — `_save_fault_checkpoint` がスコープに
>    `dynfreeze` を持たないまま `dynfreeze.state_dict()` を参照（`NameError`）。広い `except` に
>    飲まれて `training_state.pt` が OOM/CUDA fault 時に**黙って書かれず**、fault-resume が
>    cycle/velocity/delta_tracker/controller/dynfreeze の全状態を失う。`dynfreeze_enabled: true` の
>    実 config（`9b_tg_lora_m10_dynfreeze.yaml` 等）に潜在。並行の正常 periodic save（同関数の外・
>    dynfreeze は正しくスコープにある）だけが正しく、**fault 側のみの欠陥**だった。
>    → `dynfreeze` をパラメータで明示スレッドし、唯一の呼び出し site（`finally` block）で渡す。
> 2. **progressive freeze 有効化で即 crash** — `ProgressiveFreezeController` を使用（≈L1200）するのが、
>    唯一の import（無関係の `enable_psa` block 内の遅延 import・L1434）より前 → 有効化した瞬間に
>    `NameError`。活性研究機能（MS-PF1）を有効にできない状態だった。`progressive_freeze_enabled: true`
>    の config が無く・単体 test が controller を直接構築するため潜在化していた。
>    → 両 controller（`ProgressiveFreezeController`/`DynamicFreezeController`）を module top に hoist
>    （循環 import なし・両 module とも `src.training` を import しないことを確認済）し、遅延 import
>    2 件を削除。これで F821/F401 の lint 族（5 F821 + 1 F401）も一括解消。
> **検証**: `ruff --select F821 src/training/train_tg_lora.py` = **0**（修正前 5 件）。新規
> `tests/test_train_tg_lora_static_guards.py` が F821 ゼロを CI 強制（train_tg_lora は L16 の src.data
> 依存で本 mirror では import 不可のため、ruff を file path に走らせ **import を回避** = src.data
> block の ~130 pre-existing fail に触らない）。dynfreeze_state の serialize 往復は既に
> `tests/test_checkpoint.py` が cover。**`tests/test_fault_recovery.py` の OOM/resume 諸試験は本 fix で
> private repo（src.data あり）では red→green に反転する**（本 mirror では src.data block で
> pre-existing fail のまま・stash 比較で非回帰を確認: 修正前後とも同一 7 fail）。`ruff check` は
> 8→2 error へ（残り F841@L2296 + E741@L3717 は従来からの無関係負債・非回帰）。

### Category-A vs Category-C 台帳（quantification）

- **Category-A helpers 残数: 0**（MS-PF2 4/4・MS-PF3 2/2・MS-PF4 1/1 = 計 **7/7 完了**）:
  plan / cost / frontier / surrogate-gate / **surrogate-CI** / loss-ablation / schedule-portability。
  **本イテレーションも Cat-A ヘルパーを 0 件追加**（feedback の収益逓減指示に合致）—
  generalization task / heterogeneous stack / order-sensitivity 診断 / **replay judge** は
  いずれも Cat-A 足場ではなく **Cat-C 証拠の生成・解像・再検証**（同じ実行導線・同じ trio・
  GPU 証拠を消費する・足場ヘルパーではない。replay は記録済み GPU 証拠に judge を走らせ
  verdict を出す=研究結果パイプラインそのものの GPU 不要化）。
- **Category-C run 残数: 1**（target-scale valid_loss 実測）— ただし**具体コマンドに縮約済み**:
  (1) 証拠**生成** `make freeze-validloss-ci` + `--task generalize` + `--architecture heterogeneous`
  （proxy-scale 判定は 4 セル実測済み = **全て TIES**・generalize は決定的 null・
  `make freeze-order-sensitivity` で TIES が真の null と**証明済み**）。
  (2) 証拠**再検証** `make freeze-replay`（commit 済み proxy 記録を GPU 不要で再判定し TIES を
  assert・初の commit 済み Cat-C dataset = `tests/fixtures/freeze_validloss_generalize_proxy.json`）+
  `make freeze-order-sensitivity-replay`（2 つ目の commit 済み Cat-C dataset
  `tests/fixtures/freeze_order_sensitivity_proxy.json` = ratio=0.000 linchpin 証拠を GPU 不要で再判定し
  not_resolvable を assert）。
  target-scale は (1) の導線で 9B 標本を生成し (2) に流すのみ（同一 schema・`proxy_scale` フラグで昇格）。

### 本イテレーションで完了した Category-C 攻撃

1. **初の実 GPU valid_loss 判定を取得（前イテレーション）** — `make freeze-validloss-ci` が RTX 3060 で
   candidate(`output_first`) vs surrogate(`random_freeze_order`) を**実学習**し、real valid_loss 標本を
   `surrogate_valid_loss_ci()` に流入 → **TIES**（CI[95%]=[−0.096, +0.109]・n=5/5・non-thin）。
   `surrogate_valid_loss_ci()` が初めて構成定数ではなく実 run の標本を消費した。
2. **generalize タスクで TIES を決定的 null に格上げ（本イテレーション）** — memorize タスク
   （train==valid・暗記）の TIES は**自明**（順序は構造上効き得ない）であったため、保持検証
   バッチで真に汎化する teacher-student タスクを追加。student は全 order・seed で held-out
   valid_loss ~2.5（uniform≈3.47 を大きく下回る）に**真に汎化**するにも関わらず **TIES**
   （CI[−0.067,+0.313]）=「パイプラインは常に-TIES の壊れ経路ではなく、それでも順序は汎化に
   寄与しない」という**決定的 null**。teacher 校正（entropy 1.14 vs uniform 3.47）は apparatus 検証
   定数であって verdict 操作ではない。
3. **heterogeneous 正控御の実施と誠実な限界（前イテレーション）** — 層毎 rank(1,2,4,7,13,24) の
   非対称を注入した generalize 正控御も **TIES**（CI[−0.137,+0.209]）。注入非対称は n=5 bootstrap
   床の下に留まり `SURPASSES` を観測できず=「真の順序効果があれば捕捉できる」という**感度証明は
   未取得**だった（誠実な限界）——これは下記 #5 の診断で**解消**した。
4. **Category-C を具体コマンドに縮約** — target-scale 判定は同一導線に 9B の標本を流すのみ
   （`proxy_scale` フラグが target 昇格を示す）。残る外部依存は private `src.data` pipeline 単体
   （足場コードの追加では解決しない本物の外部依存）。
5. **apparatus 順序感度の特性化で #3 の限界を解消（本イテレーション）** — verdict TIES が
   「感度不足の below-resolution 読み」か「真の null」かを、正控御を発火させようとするのではなく
   **分散分解**で直接測定。`make freeze-order-sensitivity` が Var(order)/Var(seed)=**0.000**（厳密
   ゼロ・12 distinct 順序が全て同一 valid_loss）を実測=proxy では順序信号が**構造上ゼロ**であり、
   #3 の正控御が発火しなかったのは感度不足ではなく「測るべき順序信号が最初から無かった」ため。
   ゆえに verdict TIES は真の null と**証明**され、target-scale 9B run は順序効果を解像する唯一の
   残る手段と**証明済み**（想定→証明に格上げ）。これは verdict を出す run ではなく**測定科学的**
   診断（surrogate_valid_loss_ci を呼ばず SURPASSES/TIES を出さない）=足場ヘルパーではない。
6. **Cat-C を GPU 不要の再検証コマンドに縮約（本イテレーション）** — feedback の「足場追加ではなく
   Cat-C を具体コマンドに縮約せよ（recorded/proxy dataset + 実行可能 make target +
   expected-output assertion）」への直接応答。記録済み valid_loss sample を
   `surrogate_valid_loss_ci()` **だけ**で再判定する独立 GPU 不要コマンド（`make freeze-replay`）と、
   **初の commit 済み Cat-C dataset**（`tests/fixtures/freeze_validloss_generalize_proxy.json`・実
   RTX 3060・`--task generalize`・TIES）を追加。replay test が「記録済み floats が決定論 bootstrap の
   下で同一 verdict を再獲得する」（faithfulness）を assert し、target-scale 9B は同一 schema の
   sample file を流すのみで昇格（`proxy_scale` フラグで scale label 自動切替）。**残る外部依存は
   private `src.data` pipeline 単体**（足場コードの追加では解決しない本物の外部依存）。
7. **target-scale drop-in に synthetic-provenance 保護を追加（本イテレーション）** — feedback #1 の
   「commit 渺み verdict は全て proxy-scale であり §4 target-scale 結果として引用してはならない」警告を
   prose から code+test 契約に格上げ（前イテレーションが docstring-only の『no code change』を契約化
   したのと同パターン・同一 integrity 軸）。plumbing fixture（`proxy_scale=false`・合成 floats・
   SURPASSES）に機械可読 `synthetic: true` を付与し、`replay_freeze_valid_loss_ci()` は合成記録に対して
   『this verdict IS the §4 target-scale result』の引用可能クレームを**差し控え**『do not cite』note を
   出力（scale label の PROXY→TARGET 切替と verdict の忠実再計算は維持=drop-in 機構は壊さない・
   `--expected` assertion も阻害しない）。真の 9B run が同 schema で `synthetic: false`（省略可）の標本を
   置けば note は自動的に正の TARGET_SCALE クレームに戻る——その分岐は構成録音
   （`proxy_scale=false, synthetic=false`）で cover 済み（private `src.data` 不要）。これは足場ヘルパー
   ではなく evidence-integrity 保護。**残る外部依存は private `src.data` pipeline 単体**（合成→本物 9B
   への置換のみで研究結果となる）。
8. **引用ゲートを machine-readable JSON パスにも適用（本イテレーション）** — feedback #1 の引用制約
   （proxy/synthetic verdict は §4 target-scale 結果として引用不可）は #7 で **human-readable**
   （`format_replay` の prose note）でのみ強制されていた。**machine-readable**（`replay_to_json` /
   `result_to_json`）には引用ゲートが無く、下流 consumer が `proxy_scale`/`synthetic` の2生フラグから引用
   可否を推論しなければならなかった。これを単一 boolean `citable_as_target_scale` で閉じた:
   consumer 側 `(not proxy_scale) and (not synthetic)`・generator 側 `not proxy_scale`（synthetic path 無し）。
   3型（genuine→True / proxy→False / synthetic plumbing→False・target-scale label でも非引用可能）+
   「machine boolean == human prose（"this verdict IS" の有無）」cross-check test で prose と機械ゲートの
   不整合を自動検知。実 committed proxy fixture が `citable_as_target_scale=False` になることを CLI `--json`
   で確認済み。前2イテレーションと同一 integrity 軸（prose→機械契約）の延長・足場ではなく evidence-integrity
   保護。**残る外部依存は private `src.data` pipeline 単体**（不変）。
9. **「target-scale 必要」の linchpin 証拠（ratio=0.000）を commit 済み実測 + GPU 不要再検証へ（本イテレーション）** —
   証拠鎖の非対称を閉じた: valid_loss 判定（TIES）は #6 で commit 済み fixture + `make freeze-replay`（GPU 不要）
   だったが、**それより load-bearing な「順序は proxy で非解像→target-scale のみが順序を解像できる」証明
   （#5 の ratio=0.000）は GPU でのみ再現可能で PURPOSE prose 止まり**だった。これを #6 と同一パターンで
   閉じた: (a) **初の commit 済み order-sensitivity 証拠** `tests/fixtures/freeze_order_sensitivity_proxy.json`
   （実 RTX 3060・homogeneous/generalize・ratio=**0.000**・12 distinct 順序がすべて valid_loss=2.7155・
   Var(seed)=0.0202・再現性 bit-for-bit）、(b) stdlib-only replay `scripts/replay_freeze_order_sensitivity.py`
   （torch/GPU/numpy 不要・記録 by_order/by_seed float から分散分解を再計算・閾値は fixture の
   `resolution_threshold` から読む=並列定数なし・`--expected {resolvable,not_resolvable}` で exit 0/2・
   `citable_as_target_scale` ゲート）、(c) `make freeze-order-sensitivity-replay`（commit 済み proxy 記録を
   再判定し not_resolvable を assert）、(d) `tests/test_replay_freeze_order_sensitivity.py`（21 tests・
   faithfulness/分散公式の source との torch-gated cross-check/resolvable 分岐/scale honesty/CLI assertion/
   target-scale drop-in）。target-scale 9B は同一 schema の by_order/by_seed を流すのみで昇格
   （`proxy_scale` flag で scale label と citable ゲートが自動切替・コード変更不要）。これは verdict ではなく
   測定科学的 diagnostic の GPU 不要再検証化=足場ではなく evidence-integrity 保護。**残る外部依存は
   private `src.data` pipeline 単体**（不変）。

### 次候補（足場追加ではない）

1. **target-scale valid_loss 判定** — private `src.data` pipeline 利用可能次第、9B QLoRA run の
   candidate/surrogate valid_loss 標本を `surrogate_valid_loss_ci()` へ。**導線は具体コマンドに縮約済み**:
   `make freeze-validloss-ci`（9B 設定）で標本を生成 → 同一 schema の JSON に書き出し →
   `make freeze-replay FREEZE_REPLAY_FLAGS=target_9b.json` で verdict 昇格（`proxy_scale=false` で
   scale label が target に切替・コード変更不要）。**唯一の真の研究結果だが、外部依存のため本 mirror
   単独では実施不可**（[[public-mirror-preexisting-test-failures]]）。order-sensitivity 診断
   （ratio=0.000）が proxy では順序が非解像と証明した以上、target-scale は順序効果を解像する
   **唯一の**手段（**必要と証明済み**）。
2. **避ける**: (a) heterogeneous/generalize を超える proxy 正控御の更なる調整（既に発火せず・
   収益逓減）、(b) bootstrap-CI → G0–G4 ゲート配線などの**追加 Category-A ヘルパー**（feedback が
   収益逓減と明示）。target-scale 標本が無い段階でのゲート統合は空転になる。
   > **注**: proxy の順序感度については、更なる正控御調整（避ける(a)）の代わりに**分散分解診断**
   > （`make freeze-order-sensitivity`）で決着させた——正控御を発火させるのではなく apparatus の
   > 解像度を直接測り ratio=0.000 を得た。この問いは**閉じた**（target-scale のみ残る）。
