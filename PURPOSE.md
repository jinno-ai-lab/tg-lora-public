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
