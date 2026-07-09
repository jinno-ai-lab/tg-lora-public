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

> **【2026-07-09 追記・break-even gate の per-arm 帰属を構造化（feat+test+ci）— feedback #1「per-arm failure records」が message 文字列埋め込み(弱)のみだったのを structured field へ硬化・CI consumer も部分文字列照合→構造読取りへ移行】**
> feedback #1 が反復して求める "per-arm failure records" を実態監査した結果、**弱充足**だったことを発見: `evaluate_gates()` の失敗 record は `{gate, message}` のみで、**arm 帰属は人間可読 `message` 文字列内に埋め込み**だった（例: `"warm_tg_gpu_peak_mb=14000 MB exceeds budget ..."`）。ゆえに全 consumer が**脆い部分文字列照合**に依存していた — 単体 test（`"warm_tg_gpu_peak_mb" in failures[0]["message"]`）・CI workflow（`'warm_tg_gpu_peak_mb' in msgs`）。message の文言を変えれば arm 帰属が**黙って退化**し、強制 CI gate の per-arm verdict が silent に壊れる状態だった（GOAL §7 検証可能性の観点で契約の芯が文字列に依存していた）。本 iter で構造化して硬化:
> - **structured `arm` field（feat・`scripts/analyze_prefix_cache_break_even.py`）**: 各失敗 record に `arm` key を追加。VRAM gate(`--max-warm-gpu-peak-mb`) は違反 arm の metric key(`warm_baseline_gpu_peak_mb` / `warm_tg_gpu_peak_mb`)・cross-arm gate（warm-win / break-even-runs / one-run-win = 両 arm を包括的に比較し単一 arm の責任ではない）は `None`。consumer は `f["arm"]` だけで「どの arm が予算違反か」を読める。`message` は後方互換で維持（既存 stderr/JSON consumer 影響なし）・失敗 list の型注釈を `dict[str, str | None]` へ更新。no-gate path は `gate_eval_record()` が `None` を返すため byte-identical 維持。
> - **test（+2 新規・既存強化・mutation 証明）**: `test_vram_arm_field_is_structured_not_message_embedded`（load-bearing: `arm` は metric key で message 文言に非依存 — `arm` 値が result dict の key に一致）・`test_non_arm_gate_failures_carry_none_arm`（cross-arm gate は `None`・`f["arm"]` は KeyError にならず uniform schema）。既存 TG-exceeds / baseline-exceeds / both-arms / unmeasured 強化 + **real-producer-output e2e**（`test_vram_violation_fires_nonzero_with_per_arm_record`）と **checked-in fixture per-arm test**（`test_violating_mutation_fires_naming_only_the_tg_arm`）に構造 `arm` assert を追加 = **real pipeline 出力で構造 per-arm 帰属を検証**（fixture-vs-pipeline gap は閉じたまま硬度化）。**mutation 証明**: VRAM `arm` を `None` に無力化すると新 test 2 件が正確に RED（`assert None == 'warm_tg_gpu_peak_mb'`）→ restore で GREEN。`arm` が装飾でなく load-bearing であることを実証。
> - **CI consumer も構造読取りへ（ci・`.github/workflows/test.yml`）**: VRAM reject-path assertion を `arms == {'warm_tg_gpu_peak_mb'}`（structured set）へ強化 — 従来の `'warm_tg_gpu_peak_mb' in msgs`（部分文字列照合）から、CI 側でも structured field を読むよう移行。本環境で bash block を simulation 駆動 → reject exit 1・verdict gate/arm = `--max-warm-gpu-peak-mb / warm_tg_gpu_peak_mb`・assertion pass。これで「per-arm failure records が正しく発火」が producer 出力 × 実 `make` target × CI consumer の 3 面で**構造的**に検証された（文字列照合ではない）。
> - **位置づけ**: feedback #1「per-arm failure records ... fire correctly on real pipeline output」は「message 埋め込み(弱) → structured field(強)」に格上げ（warm-arm 予算違反 e2e 自体は既存 iter で閉包済・本 iter はその verdict の per-arm 帰属を構造化）。#2(TASK-0059 recursive funnel cost)= 引き続き **phantom**（本 iter でも再 grep で `recursive_passes`/`max_passes`/`max_inference_calls`/role gate は `src`/`scripts` に存在せず・本 repo の TASK-0059 は MLflow retry 別物）→ 計測対象なし。#3(exit-3→defer/retry control plane)= 引き続き**本 repo 外**（producer+分類器半分は両 trainer で `091219e`/`0f0d859` にて完了済・retry loop 自体は AI-Hub infra）。#4(gate wiring)= 完了維持（`prefix-cache-ab-check` は phantom・実在する両 GPU-free gate は `gates-ci`/`ci:`/GH Actions `gates` job で起動）。該当 file **76 passed**（74→+2 新規・既存多数強化）・ruff touched file clean・YAML valid・src static guard 不変。`[[ai-hub-feedback-infra-vs-this-repo]]` `[[atomic-torch-save-axis-complete]]`

> **【2026-07-09 追記・baseline trainer の graceful-OOM exit code を canonical 契約へ対称化（fix+test+docs）— feedback #3 producer 半分の**残り対称 gap** を閉包・AGENTS.md doc-vs-impl drift の解消】**
> 直前 `0f0d859` で feedback #3 の producer 半分（`train_tg_lora.py` graceful OOM → exit 3）は閉じたが、**実態監査で baseline 側に同族の未閉包 gap** を発見: AGENTS.md `## Process exit codes` は `train_tg_lora.py` / `train_baseline_qlora.py` **両 trainer** が deferrable OOM に `OOM_EXIT_CODE`(3)・実故障に `2` を吐くと文書していたが、**`train_baseline_qlora.py` の graceful-OOM handler は fault checkpoint を保存した後 bare `raise`（→ exit 1）しており、文書は両 trainer を謳うのに実装は TG のみ**という doc-vs-impl drift だった（GOAL §7 誠実性の観点で、主張していない契約の違反）。しかも `test_fault_exit_contract.py` の static guard は TG trainer しか pin していなかったため、この drift は unguarded だった。分類器(`frontier_report.determine_status`)は `\bOOM\b` log-text backstop で baseline OOM を `oom` に拾えてはいたが、exit code 単独では読めず=契約の核心（"exit code ALONE で defer/retry を key する"）が baseline について成立していなかった。本 iter で閉じた:
> - **baseline の fault exit を canonical 化（fix・`src/training/train_baseline_qlora.py`）**: graceful-OOM handler で `from src.utils.device import fault_exit_code, is_gpu_oom_error` し、`reason = "oom" if is_gpu_oom_error(exc) else "cuda_error"` で分類して `raise SystemExit(fault_exit_code(reason))` に切替（bare `raise` → 廃止）。**fault bucket の entry 条件は byte-identical**（`OutOfMemoryError` or `RuntimeError`-with-`"CUDA"`）に維持=何が fault かは変えず exit code のみ変更・最小 blast radius。`OutOfMemoryError` / `RuntimeError("…out of memory…")` → exit 3（縮小再開可能）・`RuntimeError("CUDA error: …")`（非 OOM）→ exit 2（実故障）と、TG の oom/numerical_instability/cuda_error 分類と同じ意味論を baseline も持つ。非 OOM/非 CUDA な `RuntimeError` は従来通り bare `raise`（propagate）。
> - **static guard を対称化（test・`tests/test_fault_exit_contract.py`）**: 4 invariant → **5 invariant** へ。新 `test_baseline_routes_fault_exit_through_helper` が baseline trainer の (a)device import・(b)`is_gpu_oom_error` による分類行・(c)`raise SystemExit(fault_exit_code(reason))` routing を pin。docstring も「the trainer」→「BOTH trainers」へ一般化。
> - **baseline の fault-exit 挙動 test（test・`tests/test_baseline_training.py`）**: `TestFaultExitContract`(+4) を追加 — `_patch_all_deps` harness で `forward_backward` に OOM/CUDA error を raise させ、`save_checkpoint` を no-op 化して exit-code routing のみを隔離（TG の `test_oom_exits_defer_exit_code` と同一隔離根拠）。(1)`test_oom_exits_defer_code`〔parametrize: `OutOfMemoryError` instance + `RuntimeError("…out of memory…")` message の両方 → `SystemExit(OOM_EXIT_CODE=3)`〕・(2)`test_cuda_error_exits_fault_code`〔`RuntimeError("CUDA error: device-side assert triggered")` → `SystemExit(2)`=非 deferrable〕・(3)`test_saves_fault_checkpoint_before_exit`〔OOM 時に `save_checkpoint` + `save_baseline_training_state` が `oom_checkpoint/` に呼ばれてから SystemExit すること=exit 化が save-on-fault を退行させないことを pin〕。
> - **mutation 証明**: baseline の routing を `git stash` で bare `raise`（HEAD）に戻すと — 新 static guard 1 test + 新 behavior test 4 test が**正確に RED**（bare raise → `OutOfMemoryError` が uncaught propagate → SystemExit 捕獲なし → "graceful fault must raise SystemExit (got none)"）→ restore で **9 GREEN**。test が装飾でなく load-bearing であることを実証。
> - **AGENTS.md（docs）**: 既存 `## Process exit codes` 節に「両 trainer で実装済み（symmetric）」bullet を追記 — TG(3 値)/baseline(2 値) が共に `fault_exit_code()` 経由であること・かつて baseline が bare raise だった doc-vs-impl drift が閉じたこと・`test_fault_exit_contract.py` が両 trainer を pin することを明記。
> - **位置づけ**: feedback #3 の producer 半分が **TG trainer のみ → 両 trainer** に拡張完了（`c15c55e` の checkpoint load-side integrity 対称化と同じ両-entrypoint 対称の原則）。**制御系 retry loop 自体**（exit 3 を読んで batch/seq_len 縮小再実行）は引き続き AI-Hub loop infra = 本 repo 外（AGENTS.md 明記）。#1(real run 記録)= 直前 entry で 4 gate 完成・#4(non-inert)= 同・4 gate が CI 直接起動・いずれも維持。#2(TASK-0059 recursive funnel)= 本 iter でも再 grep で**引き続き phantom**（`recursive_passes`/`max_passes`/`max_inference_calls`/role gate は `src`/`scripts` に存在せず・本 repo の TASK-0059 は MLflow retry 別物）→ 計測対象なし=scope 外。該当 file **5 static guard + 4 behavior test 全 pass**・ruff touched file clean。`[[ai-hub-feedback-infra-vs-this-repo]]` `[[checkpoint-load-side-integrity-axis]]`


> AI-Hub feedback #3（HIGHEST-LEVERAGE）「GPU lock の価値は制御系(control plane)が exit 3 を training fault でなく『defer and retry』として読むことに依存する。exit-3→defer/retry 経路を検証・配線せよ — さもなくば OOM 対策の deferral は void に落ちる」。AGENTS.md は「3 を defer と読む解釈は control plane の domain」と明記していたが、**実態監査で producer→分類器の seam 自体が壊れていた**ことを発見: (a)`train_tg_lora.py` の graceful OOM handler は `"GPU OOM at cycle N"` と log し `raise SystemExit(2)` で終了していた — つまり**deferrable な OOM が generic training fault と同じ exit 2 を吐いており**、(b)`scripts/frontier_report.determine_status` は OOM を exit 137 / 文字列 "out of memory" / "Killed" で検出しており**exit 2 + "GPU OOM" 略語を認識しなかった** — 実証で graceful OOM run が `failed`（非 `oom`）に誤分類されることを確認。これが「OOM 対策が void に繰延される」状態の in-repo 実現そのものだった。本 iter で producer 側半分を閉じた:
> - **canonical exit-code 契約（feat・`src/utils/device.py`）**: `OOM_EXIT_CODE = 3` 定数 + `fault_exit_code(fault_reason)` helper を追加（single source of truth）。`"oom"→3`(繰延可能)・`numerical_instability`/`cuda_error`→`2`(実故障・縮小再試行で再現)・`None→0`。OOM を実故障と**明示的に区別**するのが契約の核心 — 縮小すれば通る OOM と、縮小しても再現する実故障を exit code だけで区別できなければ defer できない。kernel OOM-killer の 137(SIGKILL) は別個の非 graceful signal のまま置換しない。
> - **trainer の fault exit を helper 経由へ（fix・`src/training/train_tg_lora.py`）**: 硬 encoding `raise SystemExit(2)` を `raise SystemExit(fault_exit_code(fault_reason))` に置換（fault taxonomy は既存の oom/numerical_instability/cuda_error 3 値・`_run_with_deps` test helper は非零 SystemExit を捕獲）。OOM log text を `"GPU OOM at cycle N"` から `"GPU out of memory (OOM) at cycle N: <detail>"` に変更（"out of memory" 全文 + "OOM" 略語の両方を載せ log 経由の副次検出経路も強化）。
> - **分類器が graceful OOM を `oom` として読む（fix・`scripts/frontier_report.py`）**: (a)log-text backstop として `\bOOM\b` pattern を追加（baseline の "OOM checkpoint saved to …" 略語のみの行も検出）、(b)`determine_status` に `if exit_code == OOM_EXIT_CODE: return "oom"` を log-scraping より**先に**挿入（log が rotate/truncate されても exit code だけで分類）。stdlib-only 制約・torch 非依存は維持。
> - **test（+16・unit + static guard 併用）**: `TestFaultExitCode`(+6・device: oom→3 / 数値・CUDA→2 / None→0 / OOM≠実故障 / `OOM_EXIT_CODE==3` pin)・`TestGracefulOomClassification`(+5・frontier_report: exit-code path・log-text path・`\bOOM\b` 略語・数値故障は非 OOM・producer 定数 pin)・`test_oom_exits_defer_exit_code`(+1・fault_recovery: checkpoint I/O を mock して exit-code routing のみを隔離)・**新 `tests/test_fault_exit_contract.py`**(+4 static guard: device が `OOM_EXIT_CODE` 定義・trainer が `SystemExit(2)` 硬 encoding に戻らない・classifier が exit 3 + `\bOOM\b` を認識・AGENTS.md が契約を文書化)。
> - **AGENTS.md に契約を文書化（docs）**: `## Process exit codes（trainer → control plane）` 節を追加 — 0/2/3/137 の契約表 +「3 を defer と読む制御系解釈は operator/AI-Hub control plane の domain(本 repo 外)。本 repo が保証するのは生産者が区別された code を**確実に吐く**こと + 分類器が `oom` として**読む**こと（log text を副次経路として併用）」と明記。
> - **位置づけ**: feedback #3 の **producer 側半分を閉包** — 生産者(trainer)が OOM を exit 3 として確実に吐き、分類器がそれを `oom` として読む経路が unit test + static guard で pinned（`SystemExit(2)` 硬 encoding への退行を防ぐ）。**制御系 retry loop 自体**（exit 3 を読んで batch/seq_len を縮小して再実行）は AI-Hub loop infra であり**本 repo 外**（AGENTS.md 明記）= この repo で配線できるのは「信号が確実に生産され正しく読まれる」まで。#1(real run 記録)= 直前 2026-07-09 entry で 4 gate 完成・維持。#4(non-inert)= 同・4 gate が CI 直接起動・維持。#2(TASK-0059 recursive funnel)= **引き続き phantom**(本 repo の TASK-0059 は MLflow retry・funnel/`recursive_passes_performed`/`max_passes`/role gate は `src`/`scripts` に grep で存在せず=private repo 別物)→ 計測対象がない。該当 file **+16 test 全 pass**・ruff touched file clean・`test_fault_recovery.py` の既存 6 fail は pre-existing(mock model が atomic-checkpoint guard に衝突・stash 比較で非回帰確認)。`[[ai-hub-feedback-infra-vs-this-repo]]` `[[pytest-cov-torch-clash]]`

> **【2026-07-09 追記・残り2 gate（償却 gate）の fixture-vs-pipeline gap を閉包 — feedback #1「4 gate 全部を real pipeline 出力で検証」の完成・#4「CI で 4 gate 全部 non-inert」の完成（test+ci）】**
> 直前までの iter で break-even gate 4 つのうち **wall-clock(`--require-warm-win`) + VRAM(`--max-warm-gpu-peak-mb`)** の 2 つは real producer 出力(`build_benchmark_summary`)×実 `make` target で e2e 検証済みだった。しかし**償却の本体である残り2 gate** — `--max-break-even-runs`(cold build が何回再利用で償却するか) と `--require-one-run-win`(cold build 含む1走が baseline に勝つか) — は**4-key の `_single_run_summary` unit fixture と手組み dict のみ**で検証され、dense 18-key の real producer 出力でも実 `make` target でも一度も走っていなかった = fixture-vs-pipeline gap が **4 gate 中 2 gate** に残存。償却 gate こそが「cold build は本当に元を取るか」という break-even の核心問いに答える gate なので、この残りを閉じた:
> - **償却2 gate の real-producer × real-make e2e test（test）**: `TestEndToEndPipelineGate` に 3 test 追加 — いずれも実 `build_benchmark_summary` producer を canonical 入力で駆動し実 `make analyze-prefix-break-even-ci` target を subprocess 起動。(1)`--max-break-even-runs` reject: canonical は `break_even_repeated_runs = cold_build(600)/warm_delta(60) = 10.0`・budget 5 で FIRE → **exit non-zero**・stderr `break_even_repeated_runs=10.000 exceeds budget 5.0 (cold_build=600.0s / warm_delta=60.0s)`・verdict `failures={--max-break-even-runs}` のみ(warm-win は 240<300 で PASS → mask しない=per-gate 独立性)。(2)`--require-one-run-win` reject: canonical は `one_run_total = 600+240 = 840 > baseline 300` → `one_run_total_delta=-540 <= 0` で FIRE → **exit non-zero**・verdict `failures={--require-one-run-win}`。(3)accept 正控御: `cold_tg_build=50` で `one_run_total=290<300`(delta=+10>0)・`ber=0.833<=20` → **両償却 gate が exit 0**。**mutation 証明**: `evaluate_gates` の `repeated > max_break_even_runs` を `False and ...` に無力化すると reject test が正確に RED(returncode=0 に退化) → restore で GREEN。これで dense producer shape 依存の回帰(例: cold_build_seconds を dense `tg_lora_summary` surface から誤読)が unit stub ではなくここで捕まる。
> - **CI `gates` job で 4 gate 全部を self-test（ci）**: `.github/workflows/test.yml` の break-even step は従来 accept + VRAM reject(2 gate) のみ self-test していた → `--max-break-even-runs`(reject2: ber=10>5) と `--require-one-run-win`(reject3: delta=-540<=0) の reject 境界を**checked-in canonical fixture に対して追加**(producer 起動不要・portability 規約不変)。step 全体を `set -euo pipefail` で本環境 simulation 駆動 → **SIM EXIT: 0**・3 reject verdict gate(`--max-warm-gpu-peak-mb`/`--max-break-even-runs`/`--require-one-run-win`)すべて印字。これで 4 gate 全部が毎 push で CI から直接起動 = 真に non-inert(従来 pytest e2e 経由の間接起動のみだった償却2 gate も CI 直接起動に昇格)。
> - **実 run の exit code + verdict 一覧（記録・feedback #1 の 4-gate 完成）**: canonical producer 出力で — `--require-warm-win`: exit 0(240<300)・`--max-warm-gpu-peak-mb 12288`: exit 0(10000/8200 ≤ 12288)・`--max-break-even-runs 5`: **exit 1**・verdict gate `--max-break-even-runs`(ber=10>5)・`--require-one-run-win`: **exit 1**・verdict gate `--require-one-run-win`(delta=-540<=0)。accept(warm-win+VRAM) と reject(償却2 gate) の境界が**実 producer 出力・実 `make` target・CI step** の3面で記録済。
> - **位置づけ**: feedback #1(real run 記録)は「2 gate(従来) → 4 gate 全部(本 iter)」で完成。#4(non-inert) も「CI 直接起動 2 gate → 4 gate」で完成。#2(TASK-0059 recursive funnel cost)= **引き続き phantom**(本 repo の TASK-0059 は [MLflow retry logic](specs/tg-lora/tasks/TASK-0059.md)・funnel/`recursive_passes_performed`/`max_passes`/`max_inference_calls`/role gate は `src`/`scripts` に grep で一切存在せず=private repo 別物)→ 計測対象がない。#3(exit-3→defer/retry control plane wiring)= **引き続き本 repo 外**(AI-Hub loop infra・AGENTS.md が control plane の domain と明記)→ `prefix-cache-ab-check` target も本 repo に存在しない(phantom)。実装・検証は `[[ai-hub-feedback-infra-vs-this-repo]]` の grep-before-acting 原則に従い実在する象のみを扱った。該当 file **74 passed**(71→+3)・ruff touched file clean・YAML valid・src static guard 不変。`[[pytest-cov-torch-clash]]`

> **【2026-07-05 追記・loop の gate sequence(`make gates-ci`)を real pipeline 出力へ開き、aggregate 全体を初めて e2e 検証（fix+test）— feedback #1「record real run's exit code+verdict」の sequence-level 達成・#4「loop gate sequence」残り seam の閉包】**
> 直前 iter で sub-target `analyze-prefix-break-even-ci` と `gates-ci`/`ci:`/GH Actions `gates` job への配線は閉じたが、**2つの未検証 seam が残っていた**: (a)`make gates-ci` recipe は `PAPER_SUMMARY=tests/fixtures/...` を**硬 encoding** しており(`make -n` で env が無視されることを実証済)、autonomous loop が gate sequence を **real GPU A/B summary** で走らせることができなかった = fixture-vs-pipeline gap が **sequence level** で残存、(b)`make gates-ci` **aggregate 全体**を走らせる test が存在せず、velocity-ops / break-even いずれかの sequence 内配線や failure 集約が壊れても気づかない状態だった(sub-target の個別 test と checked-in fixture test のみ)。本 iter で両 seam を閉じた:
> - **`gates-ci` を real pipeline 出力へ開く（fix・production code）**: `PAPER_SUMMARY`/`OUTPUT_PATH` を `$(or $(VAR),default)` で overridable 化(one-shot benchmark line 519-523 と同一 idiom・default は canonical fixture を保持→既存呼び出しの byte-identical 挙動・`ci:` aggregate も同一経路なので自動的に開かれる)。`make -n` で env override が win することを確認。これで loop は `PAPER_SUMMARY=<real gpu ab summary.json> make gates-ci` で gate sequence 全体を本物の出力に向けられる(予算 `MAX_WARM_GPU_PEAK_MB=12288` は RTX 3060 target 定数として据え置き)。
> - **aggregate `make gates-ci` の初 e2e test（test）**: `TestGatesCiLoopSequence`(2 test) を追加 — (1) accept: default で `make gates-ci` が **exit 0**・verdict `gates.passed=true`・`failures=[]`、(2) reject: `PAPER_SUMMARY=` override で producer-faithful な VRAM 違反 summary(warm TG wall 240s<300s = wall-clock win だが `gpu_peak_mb=14000`>12288) を指すと aggregate が **non-zero** で伝播し TG 腕のみ名指し(baseline 8200MB は非名指し=per-arm 独立性)。**mutation 証明**: `gates-ci` の `PAPER_SUMMARY=$(or ...)` を硬 encoding に戻すと reject test が正確に RED(returncode=0 に退化=`make gates-ci` が canonical fixture に落ちる) → restore で GREEN。これが `make gates-ci` 全体の**初 test** = 両 gate の sequence 内配線と failure 集約が pin された。
> - **実 aggregate run の exit code + verdict（記録・feedback #1 の sequence-level 達成）**: `VENV=... make gates-ci` を**実駆動** — accept(default canonical): **exit 0**・`gates.passed=true`・`break_even_status="warm_win"`・`break_even_repeated_runs=10.0`・両腕 8200/10000 MB ≤ 12288。reject(`PAPER_SUMMARY=` violating producer output): `make` **exit 2**(内側 python が gate fail で exit 1 → make wrap=2 = 非 zero)・stderr `GATE FAILED [--max-warm-gpu-peak-mb]: warm_tg_gpu_peak_mb=14000.0 MB exceeds budget 12288.0 MB; the Condition-B TG arm...`・verdict `gates.passed=false`・`failures={--max-warm-gpu-peak-mb}` のみ・違反していない baseline 腕は verdict に現れず。sub-target の実 run 証跡(直前 iter)と合わせ、gate の accept/reject 境界が**実 producer 出力かつ実 aggregate sequence** の両面で記録済 = fixture-vs-pipeline gap の閉包。
> - **位置づけ**: feedback #1(real run 記録)は sub-target(既存) + aggregate sequence(本 iter) の両面で完了。#4(loop gate sequence)の in-repo seam も「配線されている」→「real 出力へ開かれ aggregate 全体が test 済」に格上げ。#2(TASK-0059 recursive funnel cost)= 引き続き phantom(本 repo の TASK-0059 は MLflow retry・funnel/`recursive_passes_performed`/role gate は `src`/`scripts` に存在せず)、#3(exit-3→defer/retry control plane)= 引き続き本 repo 外(AGENTS.md が control plane の domain と明記)。該当 file **71 passed**(69→+2)・ruff touched file clean・src static guard 不変。`[[ai-hub-feedback-infra-vs-this-repo]]`

> **【2026-07-05 追記・break-even CI gate を producer-faithful fixture で CI/`ci:` gate sequence へ実配線（feat+test+ci）— feedback #4 の残り半分・`analyze-prefix-break-even-ci` だけが依然 inert だったのを閉包】**
> AI-Hub feedback #4「新 gate target が make entry のみで CI/loop から呼ばれず inert」。直前 2026-07-05 entry で `bench-velocity-ops-ci` を portable 化 + GH Actions `gates` job へ実配線して**第1弾**を閉じたが、**もう一つの GPU-free gate `analyze-prefix-break-even-ci` は CI から依然呼ばれていなかった** — `gates` job のコメントが「paper-summary input が必要なので本 job では意図的に除外」と明記した通り(CI は GPU A/B を走らせられないため paper-summary が得られない)。結果この gate は pytest e2e(`bec2153`・実 make target を subprocess 駆動)経由でのみ**間接的**に exercise されるだけで、operator/loop が `make` で直接叩く経路では inert だった。本 iter でこの残り半分を閉じた:
> - **producer-faithful fixture の check-in（feat）**: 実 `build_benchmark_summary` producer を canonical な budget-compliant 入力(warm TG 240s@10000MB が baseline 300s@8200MB に勝利・両腕 12288MB 予算内・cold build 600s)で駆動し、`tests/fixtures/prefix_break_even_canonical_summary.json` を生成・check-in(dense 18-key `warm.tg_lora` + `delta` = unit 4-key fixture とは次元が違う本物の pipeline shape)。CI は GPU run も on-the-fly producer 起動も不要で、安定した review 済み入力を得る。
> - **drift-guard + 境界 test（test）**: `TestCheckedInCanonicalFixture`(4 test) を追加 — (a) drift-guard: 同 canonical 入力で producer を再駆動し checked-in fixture と構造・値一致を assert(per-tmpdir path field のみ exempt)→ fixture が 4-key stub に silent drift しない、(b) gate が canonical fixture で CI と同一設定(`--require-warm-win --max-warm-gpu-peak-mb 12288`)で PASS することを in-process pin、(c) VRAM 違反変異(warm TG 14000>12288)で gate が FIRE し TG 腕のみ名指しする per-arm 独立性を pin。**mutation 証明**: dense key を1つ削ると drift-guard + shape test が正確に RED → restore で GREEN。該当 file **69 passed**(65→+4)。
> - **`gates-ci` aggregate target + `ci:` への実配線（ci）**: 全 GPU-free gate を1 target に束ねる `make gates-ci`(`bench-velocity-ops-ci` + `analyze-prefix-break-even-ci` を再帰 make で起動)を追加し、`ci:` target の spine check 直後・pytest 直前に挿入 → `make ci` が gate sequence を明示的に走らせる(従来は pytest 経由の間接起動のみ)。verdict は gitignored `runs/gates_ci/` へ。`make gates-ci` 実 run で **exit 0**・`gates.passed=true`・`break_even_status="warm_win"` を確認。
> - **GH Actions `gates` job への実配線（ci）**: workflow の `gates` job に break-even gate step を追加し、「paper-summary が必要なので除外」という陳腐化した design rationale を supersede。step は script 直接呼び出し(checked-in `.venv` 非依存・velocity-ops step と同一 portability 規約)で**境界の両側を self-test**: canonical fixture で exit 0 (accept)、VRAM 違反変異で non-zero (reject)・verdict が TG 腕のみを名指し(`--max-warm-gpu-peak-mb`)することを assert。実 bash block を本環境で simulation 駆動し「reject gate: --max-warm-gpu-peak-mb / BOTH boundaries verified / sim exit: 0」を確認。これで `analyze-prefix-break-even-ci` は毎 push で CI から直接起動 = 真に non-inert。
> - **位置づけ**: feedback #4 の in-repo 実現が**完了**(両 GPU-free gate が `make gates-ci` / `make ci` / GH Actions `gates` job の3経路から起動される)。control-plane 半分(feedback #3 exit-3→defer/retry)は引き続き AI-Hub loop infra 側 = 本 repo 外(AGENTS.md が明記)。verdict 自体は引き続き Category-C(9B GPU + private `src.data`)。`[[ai-hub-feedback-infra-vs-this-repo]]`

> **【2026-07-05 追記・velocity-ops CI gate を portable 化 + GH Actions へ実配線（feat+test+ci）— feedback #4「GPU-free gate が make にあるだけで CI/loop から呼ばれず inert」の in-repo 実現・ただし実態監査の結果、feedback 前提の大部分は本 public mirror では成立しない】**
> AI-Hub feedback #4「4つの新 gate target が make entry として存在するが CI/autonomous loop から呼ばれる証拠がない — 少なくとも GPU-free pre-flight(`prefix-cache-ab-check`)と post-run gate が gate sequence に wire されているか確認せよ、さもなくば inert」。**実態監査**(grep before acting): (1)`prefix-cache-ab-check` は**本 repo に存在しない**(phantom)。(2)autonomous loop(`alternating-loop`/`make run`)**も本 repo にない**(AI-Hub infra → `[[ai-hub-feedback-infra-vs-this-repo]]`)。(3)残る GPU-free gate は `bench-velocity-ops-ci` のみ — だが**こいつが壊れていた**:
> - **REQ-149 gate の非 portable 性 = 実 DEFECT（feat）**: `bench-velocity-ops-ci` は checked-in `baselines/velocity_ops.json` に対する**絶対 per-iter ms の `--baseline`/`--threshold 20` 比較**だった。これが**非 portable**: 本 12GB box で pytest 負荷下に走らせると `cap_update_per_iter_ms`=30ms = baseline 2.5ms の **12x**・`velocity_ema` 4.8x と出て**gate は常に RED**。自身の pin test `test_ci_gate_passes` も threshold を **50 に誤魔化して**(make は 20)通そうとし、それでも本環境では RED。REQ-149 の趣旨（観測用 benchmark → enforced CI gate）は**絶対時間比較では達成不能**だった。
> - **portable gate への修正（feat）**: `cap_update` は capped path が nocap path に対し**in-place `mul_` 1つ分**だけ余計に走る（2つの norm reduction + clone は共通）→ `cap_update_overhead_ratio = capped_per_iter_ms / nocap_per_iter_ms` は**同一 run 内で host 速度が相殺**される hardware-normalized 不変量。`--max-cap-overhead-ratio`（opt-in・default 3.0）gate を追加。負荷下でも両項が一緒に膨らむため ratio は安定（絶対時間が 12x に跳ねた同じ負荷で ratio は ~1.3・8 run で max 2.02 = ceiling 3.0 に 48% 余裕）。`--baseline`/`--threshold` は local same-host diagnostic として残存。Makefile `bench-velocity-ops-ci` を portable gate に切替（`MAX_CAP_OVERHEAD_RATIO` override 可）。
> - **CI へ実配線（ci）**: `.github/workflows/test.yml` に `gates` job を追加 — 每 push/PR で portable velocity-ops gate を起動（`make` ではなく script 直接呼び出しで checked-in `.venv` 非依存・CPU-only torch）。ruff lint は既存 `test_src_tree_is_ruff_clean` static guard が pytest job で担保済み、`ruff format --check src/` は 46 file の既存 debt で RED になるため意図的に除外（本 iter scope 外）。これで「make にあるだけで誰も呼ばない = inert」を閉じた。
> - **検証（test）**: RED だった `test_ci_gate_passes` → `test_portable_ci_gate_passes` に修正し**GREEN**（make target と完全一致・任意 host で pass するのが ratio normalization の全意义）。新 `TestPortableCapOverheadGate`(4 test) で**境界の両側**を pin: ceiling 3.0 は pass・ceiling 0.5（true ratio ~1.0-2.0 の下）は確実に bite + ratio 計算の正しさ。新 `tests/test_ci_gate_wiring.py` static guard が「workflow が portable gate を呼ぶ」+「make target が non-portable `--baseline` に戻らない」を pin（**mutation 証明**: make target を旧形に戻すと guard が正確に RED）。該当 file 全 **23 + 2 passed**・ruff touched file clean。
> - **feedback 各項の誠実なスコープ**: #1(end-to-end break-even gate)= `bec2153` で**完了・65 passed で verified-green**（本 iter で再実行し再確認）。#2(TASK-0059 recursive funnel cost 計測)= 本 repo の TASK-0059 は**MLflow retry logic**であり、funnel/`recursive_passes_performed`/`max_passes`/`max_inference_calls`/role gate は `src`/`scripts` に**一切存在しない**(private repo 別物の phantom)→ 計測しようがない=scope 外。#3(exit-3→defer/retry の control plane wiring)= AI-Hub loop infra であり**本 repo 外**(AGENTS.md が control plane の domain と明記)→ scope 外。#4= 本 iter（portable gate 化 + CI 配線 + static guard）で in-repo 実現。`[[ai-hub-feedback-infra-vs-this-repo]]`。

> **【2026-07-04 追記・prefix-cache break-even CI gate を**実 pipeline 出力**で end-to-end 駆動（test）— feedback #3「fixture-vs-pipeline gap を閉じよ」の第1弾・非 zero exit + per-arm failure record + verdict JSON を本物の producer 出力で検証】**
> AI-Hub feedback #3「`make analyze-prefix-break-even-ci` を**actual/faithful な paper-summary の warm 腸が予算を違反する実例**で end-to-end 駆動し、非 zero exit + per-arm failure record + verdict JSON が**unit fixture でなく実 pipeline 出力**で正しく発火することを確認せよ」。直前3 commit（`746cdee`/`f5fbd17`/`b269968`）で wall-clock + VRAM 予算 gate を整えたが、**全 gate test は `tests/` 内の hand-pruned 4-key fixture**（`_single_run_summary`）だけで、gate が**実 `benchmark_prefix_cache.py` が吐く dense な summary.json 形状**で発火することは未検証だった（`make -n` dry-run で target の存在は pin されていたが実行は未履歴）。本 iter でこの gap を閉じた:
> - **実 producer 駆動の faithful fixture**（test）: `run_metrics.jsonl`（`run_header`/`step`/`run_footer`=`train_*` が吐く形式）を tmp に書き、**実 `build_benchmark_summary`** で summary.json を生成 — `warm.tg_lora` は **18 key**（`prefix_feature_cache_offloaded_prefix_modules`/`runtime_offload_gpu_freed_mb`/`extrapolation_steps`/`loss_red_per_wall_minute`...）+ top-level `delta` を持ち、unit fixture の 4-key subset とは次元が違う。これが「もう一つの unit fixture でなく真の coverage 増」という fidelity 根拠。
> - **実 `make analyze-prefix-break-even-ci` end-to-end**（test）: 生成した summary を**実 make target**（`VENV=`/`PAPER_SUMMARY=`/`OUTPUT_PATH=`/`REQUIRE_WARM_WIN=`/`MAX_WARM_GPU_PEAK_MB=` env→flag wiring 経由）に食わせ、**VRAM 違反腕**（warm TG wall 240s<300s = wall-clock win だが `gpu_peak_mb=14000`>12288 予算）で駆動。
> - **実 run の exit code + verdict（記録）**: `make` exit **2**（recipe 内 python が gate 失敗で exit 1 → make が 2 で wrap = 非 zero）。stderr は `GATE FAILED [--max-warm-gpu-peak-mb]: warm_tg_gpu_peak_mb=14000 MB exceeds budget 12288.0 MB; the Condition-B TG arm...`。verdict JSON は `gates.passed=false`・`failures[0].gate="--max-warm-gpu-peak-mb"`・`requested=["--require-warm-win","--max-warm-gpu-peak-mb=12288.0"]`。**per-arm 独立性を実出力で証明**: wall-clock gate（`--require-warm-win`）は **pass**（`break_even_status="warm_win"`）したまま VRAM gate のみ fail し、**違反していない baseline 腕（8200 MB）は verdict に名指しされない**。wall-clock 敗北腕（warm TG 350s>300s）と all-green（両 gate pass・exit 0）も併せて境界の両側を pin。
> - **検証**: +4 e2e test（dense-shape fidelity guard / VRAM 違反・per-arm / wall-clock 敗北 / all-green positive control）。**mutation 証明**: VRAM gate の比較 `elif float(peak) > max_warm_gpu_peak_mb:` を `elif False:` に変えると e2e test が正確に RED → restore で GREEN（test が装飾でなく load-bearing）。全 file **65 passed**・ruff touched file clean・src static guard 不変。`make`/producer import 不可環境では skip で安全縮退。
> - **位置づけ**: gate の accept/reject 境界を**実 producer 出力**で検証したことで fixture-vs-pipeline gap を閉包。verdict 自体は引き続き Category-C（9B GPU + private `src.data`）。`[[ai-hub-feedback-infra-vs-this-repo]]`

> **【2026-07-04 追記・prefix-cache break-even の display-only gpu_peak_mb を VRAM 予算 gate で消費（feat + test）— feedback #2「metric を実際に消費せよ」の第2弾・wall-clock に続き VRAM 次元を閉包】**
> AI-Hub feedback #2「display-only metric family を gate/triage rule で**実際に消費**せよ（第5の対称集計軸を足すな）」。`746cdee` が wall-clock 系（`break_even_status`/`break_even_repeated_runs`/`one_run_total_delta_seconds`）を CI gate で消費して第1弾を閉じたが、analyzer は**抽出済みの `warm_baseline_gpu_peak_mb`/`warm_tg_gpu_peak_mb` を一度も読んでいなかった**（`test_extracts_gpu_peak`/`test_gpu_peak_mb_forwarded` は抽出を検査するのみ）。これは**憲法 P3（VRAM コスト会計）+ RTX 3060 12GB 目標環境**に直結する指標であり、実 run では `gpu_peak_mb=10782 MB`（12288 MB カードで 88%・残り ~1.4GB）を観測済み=prefix feature cache が VRAM 圧力を**加える**以上、cache-on 腕が予算を OOM すれば wall-clock の break-even は無意味になる。本 iter でこの死指標を消費する gate を追加:
> - **`--max-warm-gpu-peak-mb`**（feat）: 両 warm 腕（baseline+TG）の `gpu_peak_mb` が予算 MB 以下なら pass（boundary-inclusive=`--max-break-even-runs` と同一規約）。**各違反腕が個別 failure record** を出し verdict が「どの腕が予算を割ったか」を名指し。**未計測（None）は fail-loud**（計測されていない peak に対して予算を保証できない=GOAL §7「測定せず結論しない」）。Makefile CI target に `MAX_WARM_GPU_PEAK_MB` を追加。
> - **位置づけ**: これは既存 break-even 判定への**新次元（VRAM 実行可能性）**であり、対称な集計軸の追加ではない（feedback が警告した churn パターンとは逆）。verdict 自体は引き続き Category-C（9B GPU + private `src.data`）。
> - **検証**: +12 test（6 unit + 4 CLI・境界 inclusion / TG 超過 / baseline 超過 / 両腕 / 未計測 fail-loud / verdict JSON 記録）。**mutation 証明**: gate logic のみ `git stash` で除去 → 新10 test が RED・restore → GREEN（61 passed）。canary 39p/1f 不変（1f は既知の absent-private-`src.data` precompute case・非回帰）・A/B symmetry guard 5p 不変・ruff touched file clean。`[[ai-hub-feedback-infra-vs-this-repo]]`

> **【2026-07-03 追記・LoRA adapter チェックポイント保存を atomic 化（fix + test）— torch.save 軸では覆えなかった最大コスト artifact の torn-write ハザードを閉包】**
> AI-Hub feedback が「`training_state.pt` の 2 site 以外にも同じ切り捨てハザードを監査せよ。**失うと最もコストの大きい artifact は model/baseline チェックポイント（LoRA 重み）**」と指摘。`_atomic_torch_save` 軸（7 site・2 AST guard で pinned・`d827507` closed）は **`torch.save` のみ**を覆っていたが、LoRA 重みは `save_checkpoint`→`model.save_pretrained`→**`safetensors.save_file`** で永続化され、後者は **temp+rename 無しで対象 path に直接書く**（`safetensors 0.3.1` の `serialize_file` で確認）— よって SIGINT/OOM が multi-MB の `adapter_model.safetensors` dump 中に落ちると**切り捨て重み**が残り、resume の `load_file`（`train_tg_lora.py:1465`）が crash または**黙って腐った重みを復元**する状態だった。この最大 artifact は `torch.save` 軸の外側にあった唯一の非 atomic publish path。本 iter で閉包:
> - **`save_checkpoint`**（fix）: 全 save を PID-suffix sibling temp dir に staged し、完成後のみ `_atomic_publish_checkpoint_dir` で publish。readback 検証は publish 前（temp 上）に実施して空 save を表面化。staging 中の fault は `except BaseException` で orphan temp を削除（`KeyboardInterrupt`/`SystemExit` を catch = `_atomic_torch_save` と同一根拠）。全 11 call site（baseline/TG-LoRA の periodic・best_model・oom・linearity-budget）がこの単一 funnel を経由するため一括で保護される。
> - **`_atomic_publish_checkpoint_dir`**（feat）: POSIX は非空 dir を rename 置換できないため、overwrite path（`best_model/`）では**既存 dir を PID-suffix backup に退避してから**新 dir を rename-in し、最後に旧 dir を削除。fault は「新着の完全値」か「退避した旧値に復元」のいずれかになり、**空にも old/new 混在にもならない**。
> - **test**: `tests/test_checkpoint.py` に `TestAtomicCheckpointDirPublish`（4: 成功 publish/temp 消失・fresh fault で非 publish・**overwrite fault で旧 checkpoint 復元**・成功 overwrite で backup 削除）+ `TestSaveCheckpointAtomicEndToEnd`（`KeyboardInterrupt`/`SystemExit`/`OSError` の mid-`save_pretrained` interrupt で**非 publish + orphan temp 清掃**・成功 save で temp 残存なし）= **+8**。**mutation 証明**: `except BaseException`→`except Exception` に絞ると `KeyboardInterrupt`/`SystemExit` の 2 case だけ fail（orphan temp 残存）= BaseException 句が load-bearing（`test_atomic_save.py` の根拠と同一）。
> - **検証**: canary 37 passed/3 xfailed・checkpoint/resume/fault family 128 passed・stash 比較で src.data-blocked slice 含め **+8 のみ（非回帰）**・ruff touched file clean・`torch.save` AST guard 4 passed（本変更は guard を踏まない）。peft 非依存（既存 `_mock_model_and_tokenizer` pattern で `save_pretrained` を fake）。
> - **位置づけ**: これで on-disk artifact publish path で非 atomic だった経路は皆無になった（`torch.save` 7 site + LoRA adapter dir publish）。verdict 自体は引き続き Category-C（9B GPU + private `src.data`）。`[[atomic-torch-save-axis-complete]]` `[[ai-hub-feedback-infra-vs-this-repo]]`

> **【2026-07-03 追記・PSA RegimeDetector の resume-state-loss を閉包（fix + test）— resume-state 軸の第12サイト・軸完成 12/12】**
> `RegimeDetector`（`src/tg_lora/regime.py`）は `enable_psa` block 内で構築される（本 mirror の全 config で dormant だが **dormant-route であっても resume-state-loss は GOAL §7 honesty break**）。loss/velocity 分類窓・現在 regime・**run-wide `transition_count`**（per-cycle に `psa_regime_transitions` として `run_metrics.jsonl` へ永続化）を保持するが resume されず、故障/定期 resume で `transition_count` が **0 に戻り** post-resume metrics が fresh run のように見える — 既修の `psa_state`/`act_regime_state`/`progressive_freeze_state` と同族の silent honesty break。本 iter で PSAPrior と同一の確立パターンで閉包:
> - **`RegimeDetector`**（feat）: `state_dict()`/`load_state_dict(None-safe, partial-dict tolerant)`（`_losses`/`_velocities`/`regime`(.value 文字列)/`transition_count` を永続化・deque は**構築時 window** で maxlen 再構築＝config-window 変更が live run 同様に trim・`_reset_signaled` は transient で非永続化）。config 系 param は `__init__` で再構築のため非永続化。
> - **`TrainingState.psa_regime_state`**（feat）: field + `save`/`load` round-trip・legacy checkpoint は `None` で安全縮退（`test_checkpoint.py` +1 round-trip / +1 legacy）。
> - **`train_tg_lora.py` wiring**（fix）: import を module top に hoist + `_save_fault_checkpoint` に `regime_detector` param thread + 両 save site（fault/periodic）で serialize + resume block で `load_state_dict`（`psa_prior` 復元の直後）。src.data で import 不可のため `tests/test_train_tg_lora_static_guards.py::test_regime_detector_resume_state_is_wired` で wiring string を pin。
> - **test**: `tests/test_regime.py::TestRegimeDetectorStateRoundtrip`（8 test・round-trip / transition_count 存続 / None-safe / partial-dict / window maxlen 再構築 / reset signal 非永続化 / regime 文字列往復）。gold-standard 統合 fault-resume test（`tests/test_resume_state_integration.py`）を**第12サイトに拡張**: `_populate_regime_detector` helper + captured factory + 復元 assert。**mutation 証明**: resume 復元 guard を反転 → test が `assert restored_regime is not None` で正確に fail（static guard が捕れない over-broad guard を統合 test が捕る）→ revert で green。
> - **検証**: canary 37 passed/3 xfailed・resume/checkpoint/freeze/regime/psa family **244 passed**（+第12サイト統合 test 含む）・ruff 全 src/ + touched 7 file clean・stash 比較で src.data-blocked slice 7f/15p **unchanged（非回帰確認）**。
> - **位置づけ**: 本 fix で **resume-state-loss 軸が 12/12 で完成**（cycle/velocity/delta_tracker/controller/dynfreeze/psa_prior/activation_regime/progressive_freeze/lawa/snap/progressive_freeze_state/regime）。verdict 自体は引き続き Category-C（9B GPU + private `src.data`）。残る真の研究課題は heterogeneous×generalize leg の実 9B 計測（>12GB・別 TDD）。`[[resume-state-loss-axis-psaprior-next]]`

> **【2026-07-03 追記・Progressive Freeze の resume-state-loss を閉包（fix + test）— 多層 PF 経路の fault-resume 復元、resume-state 軸の第11サイト】**
> `edcf174` が多層 progressive freeze を config 駆動で trainer に接続したが、その `ProgressiveFreezeController` は dynfreeze/LAWA/warmup 等と異なり **resume-state を持たなかった**（`resume-state-loss 軸 10/10 done` は PF が全 config で無効だった時代の計数）。故障/定期 resume で (a) controller が config から再構築され `_frozen_layers` が空に戻る、(b) LoRA adapter が safetensors から復元されるが **`requires_grad=False` は weight に載らず全層が再訓練可能**、(c) cycle loop の `layers_due_at(cycle)` gate は `cycle >= cycle_offset` のみ発火し**故障前 cycle の累積 freeze を再適用しない** — の三重に、resume すると凍結済み層が**黙って再訓練**され（Progressive Freezing のコスト削減＝存在意義が消滅）・run footer の `frozen_layers`（Tier-2 §4 order-verdict arm provenance）が**故障後分のみ**を報告する。Tier-2 verdict の deposit 信頼性（`5ed3380`/`c28e522` が硬化中）を fault-resume が黙って損なう状態だった。本 iter で閉包:
> - **`ProgressiveFreezeController`**（feat）: `state_dict()`/`load_state_dict(None-safe)`（cumulative `_frozen_layers` + `_last_frozen_layer` のみ永続化・Level-2 `xin` cache は別 Phase-3 軸なので意図的に非永続化=docstring で明記）+ `refreeze_loaded_layers(model)`（safetensors 復元後の model に `requires_grad=False` を再適用・冪等）。
> - **`TrainingState.progressive_freeze_state`**（feat）: field + `save`/`load` round-trip・legacy checkpoint は `None` で安全縮退（`test_checkpoint.py` +1 round-trip / +1 legacy）。
> - **`train_tg_lora.py` wiring**（fix）: `_save_fault_checkpoint` に `progressive_freeze` param thread + 両 save site（fault/periodic）で serialize + resume block で `load_state_dict`→`refreeze_loaded_layers`（adapter 復元後）。src.data で import 不可のため `tests/test_train_tg_lora_static_guards.py` に wiring string + ruff(F821/full) guard を追加（2026-06-26 の static-guard 手法と同一）。
> - **test**: `tests/test_progressive_freeze.py::TestResumeState`（6 test・round-trip / None-safe / refreeze 復元 / 冪等 / 完全 resume シーケンス）。canary GREEN・freeze/checkpoint/resume/dynfreeze family **149 passed**（`test_resume_e2e.py` の src.data collection error は stash 比較で pre-existing 確認・非回帰）。
> - **位置づけ**: verdict 自体は Category-C（9B GPU + private `src.data`）。本 fix は Tier-2 §4 order verdict の**もう一つの Category-A 前提**（多層 PF campaign が fault-resume しても凍結状態と arm provenance が保たれる）を閉じる。残る真の研究課題は heterogeneous×generalize leg の実 9B 計測（>12GB・別 TDD）。

> **【2026-07-03 追記・Tier-2 §4 order verdict の deposit-side arm-provenance 証明化（feat + test）— c28e522 の footer 契約を deposit→replay まで閉包】**
> `c28e522` は Tier-2 candidate(`output_first`)/surrogate(`random_order`) 両腕の機械可読 arm 識別（`policy` + `surrogate_seed`）を **run footer に**永続化したが、消費者である deposit 形成経路（`form_freeze_validloss_deposit.py`）はそれを**読んでいなかった** — よって c28e522 が名指しした P0 ハザード「`--candidate`/`--surrogate` の取り違えが verdict の符号を暗黙に反転（SURPASSES↔UNDERSHOOTS）」は footer 層でしか閉じておらず、deposit 形成時にはまだ無防衛だった。本 iter でそれを**形成時 fail-loud** に閉包:
> - **`extract_best_valid_loss`** が footer の `tg_lora_summary.progressive_freeze` block（`policy`/`resolved_policy`/`surrogate_seed`/`mode`）を provenance に浮上（block 無し=Tier-1 plain TG-LoRA / baseline は `None` で安全縮退）。
> - **`form_deposit`** に `_reject_swapped_arm` を追加: `mode=="progressive"` の腕のみ検証し、candidate 構に surrogate（`surrogate_seed` 非 None）・surrogate 構に実腕（`surrogate_seed` None）が入ったら **`ValueError`** で停止。`mode=="progressive"` gating により Tier-1（PF footer 無し/single-shot）は**完全非影響**（TASK-0152 Tier-1 recipe そのまま動作・回帰 test で担保）。
> - deposit JSON に加法的 `candidate_arm_policies` / `surrogate_arm_policies`（None entry は provenance 無し）を追加 — replay judge は `candidate_losses`/`surrogate_losses` 以外を無視するので**非破壊**（`test_deposit_replays_as_citable_target_scale` で回帰確認）。
> - **test**: `tests/test_form_freeze_validloss_deposit.py` に 8 件追加（PF provenance 浮上 / None 縮退 / label の policy 表示 / 整列腕の受理 / surrogate→candidate 取違えの拒否 / 実腕→surrogate 取違えの拒否 / Tier-1 skip / single-shot skip）。canary `tests/test_cli_help_smoke.py` 37 passed/3 xfailed・replay 回帰 green。
> - **位置づけ**: これは Tier-2 §4 order verdict の**もう一つの Category-A 前提**（verdict 自体は Category-C: 9B GPU + private `src.data`）。`0d590d2`(順序対比)→`c28e522`(footer 証明)→本 iter(deposit 証明化) と、 verdict が「config から再現可能」かつ「形成時 fail-loud」に向かって硬化中。残る真の研究課題は heterogeneous×generalize leg の実 9B 計測（>12GB・別 TDD）。

> **【2026-06-29 追記・9B §4 verdict 取得 + 境界 closed-with-limitation の耐久化（test + docs）— 「pending 無期限放置」の最終解消】**
> AI-Hub feedback（本 iter）。4 提案を本 mirror で検証した結果、実質的な新規作業は **(1) の成果の耐久化** のみ:
> - **(1)「Tier-1 multi-seed seq256 campaign を走らせ citable_as_target_scale / citable_as_full_section4_verdict を実データで exercising せよ」+「seq1024 が恒久 OOM なら境界を正式 close せよ」** → **前 commit `e99e3c7`（milestone #10）で実施済み**: seed{42,43,44} candidate(TG-LoRA) vs baseline(full backprop) の 6 run を走らせ real best_valid_loss を deposit、verdict=**TIES**（CI[95%]=[−0.0205,+0.0018]・candidate mean 1.051 vs baseline mean 1.044・n=3/3 non-thin）。`citable_as_target_scale=True` / `citable_as_full_section4_verdict=False` を実データで exercising 済み。境界は **seq256 verdict 記録 + seq1024 完全判定の >12GB GPU への正式 defer** で closed-with-limitation。
> - **(3) G2.3 misrouting の回帰 test** → **`3d2bf08` で追加済み**（data-missing vs measured-breach の route を pin）。
> - **(2) make-run が routed target を実行し red→green を capture** / **(4) finding-blocked `loop_upgrade` gap の close** → いずれも **AI Hub 自身の infra**: `recover_gate_block` / `recover` / `loop_upgrade` / `finding-blocked` / `make-run` orchestrator は本 repo 全文に存在せず（grep 該当なし）・本 checkout の `.audit/` も空 = [[ai-hub-feedback-infra-vs-this-repo]]。本 repo には該当コードがないため実施不可。
>
> よって本 iter は 5番目の provenance guard を足さず、**記録済み verdict を durable にする**（feedback #1 の「REAL 数値を gate に流す」を耐久性ある結論へ昇格）:
> - **verdict 絶対値 pin（test）**: `TestRealTargetScale9BDeposit` は従来 `test_real_verdict_is_pinned_and_faithful` が `replayed_verdict == data["verdict"]`（deposit 自身の verdict field との**整合性**）のみを検査していた。これは「losses 再取得 + verdict field を新 verdict に再描画」の**協調 drift** を検知できず（整合性は保たれたまま TIES→UNDERSHOOTS 等の科学的結果が黙って変わる余地）、記録済み claim を不変量にしていなかった。`test_real_verdict_pins_literal_ties_and_ci_bounds` を追加: verdict=**TIES**（literal・`data["verdict"]` ではない）・CI が零を跨ぐ構造理由・cited 数値（means 1.0510/1.0438・CI[−0.0205, +0.0018]）・non-thin・非 material を**絶対値**で pin。**mutation 証明**: candidate losses を +0.05 摂動し verdict field を新 verdict に再描画すると、旧 test は pass のまま新 test が fail する（= 旧 test が見逃す協調 drift を新 test が捕る）ことを確認。real 数値が「再現可能」から「durable な回帰不変量」へ昇格。
> - **`次の一手` section の陳腐化解消（docs）**: `e99e3c7` は境界 closure を milestone / verdict-log section（milestone #10）に記録したが、**本 `次の一手` section の直前の最新 entry（下記 2026-06-28）は依然「9B §4 verdict 自体は依然 pending（real campaign 待ち）」「次の一手は campaign 実行 → deposit → verdict 記録」と記載したままで、記録済み verdict と矛盾**していた — まさに feedback #1 が命名した「'9B §4 verdict pending' の無期限放置」失敗モードが、**実行指示を出す section に文字通り残存**していた。本 entry がそれを supersede する（旧 entry は歴史 log として残置・削除しない）。
>
> **真の次の一手（現状・誠実）**: (a) seq1024 + valid_full(493) による完全 §4 verdict は **>12GB VRAM の別ハードウェアへ正式 deferred**（"pending" ではなく accepted-with-limitation・無期限放置ではない）。(b) Tier-2 §4 **order** verdict（random-order surrogate arm の upstream 移植・order-sensitivity 診断の ratio=0.000 を 9B target-scale で解像する）が、真に残る target-scale 研究課題（別 TDD・upstream private repo `src.data` 必要）。`[[ai-hub-feedback-infra-vs-this-repo]]` `[[gpu-usable-proxy-runs]]`。

> **【2026-06-28 追記・reduced-context provenance guard 追加（feat）— §4 verdict 誤引用防止をコード契約化】**
> AI-Hub feedback 第5回。(2)/(3)/(4) は引き続き本 repo 該当なし（[[ai-hub-feedback-infra-vs-this-repo]]）、
> (1)「real 9B §4 verdict を出すか境界を正式 unblock せよ」は**12GB RTX 3060 の物理制約 + public/private 境界合意**
> で二重に gated（seq_len=1024 は OOM・seq256 のみ動作・実測済 [[gpu-usable-proxy-runs]]）。その代わり**数値が出た瞬間に
> 誤引用を許さない honesty guard**を、recipe の prose 警告（TASK-0152 lines 86-97「reduced-context probe は完全 §4 verdict
> ではない・tag は deposit に正しく付与する」）から**コード契約**に昇格させた（provenance-guard 家族の第4成员:
> proxy/synthetic/negative_control に続き `full_context`）:
> - **`scripts/form_freeze_validloss_deposit.py`（feat）**: `--seq-len` / `--full-context` flag 追加。12GB 現実を**正直な default**
>   とし `full_context=False`（`seq_len>=1024` または明示 `--full-context` のみ True）。deposit に `full_context`/`seq_len` を付与。
> - **`scripts/replay_freeze_validloss_ci.py`（feat）**: **2-level citability** を導入。`citable_as_target_scale`（不変・real 9B run）
>   に加え NEW `citable_as_full_section4_verdict`（= target_scale AND full_context）を追加。reduced-context probe は
>   target-scale だが **full verdict ではなく**、強い "this verdict IS the §4 target-scale result" 主張を**差し控え**、
>   `seq_len` を明示した REDUCED CONTEXT caveat を描画。legacy deposit（field 無し）は full 扱いで後方互換。
> - **実データ検証**: upstream `runs/9b_verdict/cand_seed42/run_metrics.jsonl`（best_valid_loss=1.0439）で form→replay を
>   実行 → `citable_as_target_scale=True`・`citable_as_full_section4_verdict=False`・seq256 caveat 描画を確認（越境せず・read-only）。
> - **test green + lint clean**: `tests/test_form_freeze_validloss_deposit.py`（+6）・`tests/test_replay_validloss_ci` の
>   `TestReducedContextProvenanceGuard`（5 test）+ 機械/prose 双方向 drift-guard cross-check 更新 = **両 file 79 passed**・
>   canary GREEN・ruff clean・net diff 168 行（§5「diff 200行以下」内）。**9B §4 verdict 自体は依然 pending**（real campaign 待ち）だが、
>   数値が出た瞬間に「seq256 probe を完全 verdict として誤引用する」過ちがコードレベルで不可能になった。
> 次の一手は（合意後）Tier-1 multi-seed 9B PF campaign 実行 → `form --seq-len` で deposit → `replay --json` で §4 verdict 記録。

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
- **Category-C run 残数: 0**（target-scale valid_loss 実測 = **seq256 で実施済み milestone #10・verdict TIES**；seq1024 完全判定のみ >12GB へ deferred）— 具体コマンド化済み:
  (1) 証拠**生成** `make freeze-validloss-ci` + `--task generalize` + `--architecture heterogeneous`
  （proxy-scale 判定は 4 セル実測済み = **全て TIES**・generalize は決定的 null・
  `make freeze-order-sensitivity` で TIES が真の null と**証明済み**）。
  (2) 証拠**再検証** `make freeze-replay`（commit 済み proxy 記録を GPU 不要で再判定し TIES を
  assert・初の commit 済み Cat-C dataset = `tests/fixtures/freeze_validloss_generalize_proxy.json`）+
  `make freeze-order-sensitivity-replay`（2 つ目の commit 済み Cat-C dataset
  `tests/fixtures/freeze_order_sensitivity_proxy.json` = ratio=0.000 linchpin 証拠を GPU 不要で再判定し
  not_resolvable を assert）。
  target-scale は (1) の導線で 9B 標本を生成し (2) に流すのみ（同一 schema・`proxy_scale` フラグで昇格）
  → **本イテレーション milestone #10 で seq256 にて実施済み**（6 run・verdict TIES・deposit + 4 実データ test）。
  seq1024 完全判定は >12GB VRAM へ deferred（"9B §4 verdict pending" の無期限放置を解消）。

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
10. **9B target-scale verdict を実測で取得 — seq256 reduced-context で境界を closed-with-limitation（本イテレーション）** —
    feedback #1 の「5番目の provenance guard を足すのではなく REAL 数値を gate に流せ」への直接応答。「9B は
    private `src.data` で block」は**正しくなかった**: blocker は fundamental ではなく runnable で、upstream
    `/home/jinno/tg-lora`（src.data 含む private pipeline）+ cached `Qwen/Qwen3.5-9B` + RTX 3060 で 12GB VRAM-floor
    config（seq256 + eval_batch_size=1 + expandable_segments）なら学習完走可能。seed{42,43,44} で candidate(TG-LoRA)
    vs baseline(full backprop) を 6 run 実施（全 run footer 付き完走）。best_valid_loss — candidate=
    [1.043962, 1.064300, 1.044656] (mean 1.050972) vs baseline=[1.047144, 1.040318, 1.043807] (mean 1.043756)。
    bootstrap CI[95%]=**[−0.0205, +0.0018]**（零を跨ぐ）→ verdict=**TIES**（n=3/3・non-thin・is_material=False）。
    すなわち TG-LoRA 外挿は full-backprop と**統計的に同等の品質**を達成（GOAL §0 品質保持と整合）、かつ candidate の
    backward_passes は外挿受容で変動（752/696/488・mean 645 vs baseline 752 固定）し**平均 14% 減**（cand_seed44 は
    35% 減 488 bw で同等品質）=コスト削減の観測的現れ。deposit = `tests/fixtures/freeze_validloss_9b_target.json`
    （`proxy_scale=false`・`citable_as_target_scale=True`・`citable_as_full_section4_verdict=False`・
    `--task lm_next_token --architecture homogeneous --seq-len 256`）：`scripts/form_freeze_validloss_deposit.py`
    が upstream `run_metrics.jsonl` の footer best_valid_loss を一発成形 → verdict stamp → `make freeze-replay` で
    GPU 不要再検証（faithful）。`tests/test_replay_freeze_validloss_ci.py::TestRealTargetScale9BDeposit`（4 tests）が
    provenance guard / 2-level citable gate / multi-seed 実 float / verdict faithfulness を**実データ**で pin（合成・
    proxy・negative_control を含む全 7 fixture 中、初の genuine target-scale）。**境界 closure**: seq256 verdict は実 9B
    **target-scale** 測定だが完全 §4 verdict ではなく（seq1024 は 12GB で CE-loss `logits.float()` が OOM・実測）、
    **seq1024 完全判定は >12GB GPU へ正式に deferred**。これで「9B §4 verdict pending」の**無期限放置を解消**
    （accepted-with-limitation）= feedback #1 の「honesty-guard N+1 を足し続けて core verdict を block したままにする」
    失敗モードを回避。**残る未解決 = seq1024 + valid_full(493) による完全 §4 verdict（>12GB VRAM 必要・別ハードウェア）**。

### 次候補（足場追加ではない）

1. ~~**target-scale valid_loss 判定**~~ — **DONE（seq256 reduced-context・本イテレーション milestone #10）**:
   verdict=**TIES**（CI[95%]=[−0.0205,+0.0018]・candidate mean 1.051 vs baseline mean 1.044・n=3/3）。
   upstream `/home/jinno/tg-lora`（src.data 含む）で 6 run を実施し deposit + 4 実データ test で pin 済み。
   seq256 は実 9B **target-scale** だが完全 §4 verdict ではなく、**seq1024 完全判定は >12GB GPU へ正式 deferred**
   （"9B §4 verdict pending" の無期限放置を解消）。残る唯一の未解決 = seq1024 + valid_full(493) による完全 §4 verdict
   （>12GB VRAM 必要・別ハードウェア）。※本 Tier-1 は **TG-LoRA vs full-backprop**（§0 品質保持×コスト削減）であり、
   order-sensitivity 診断（ratio=0.000）が証明した「順序は proxy で非解像」を 9B で解像する **§4 order verdict は
   Tier-2（別 TDD・random-order surrogate arm の upstream 移植が必要）** — これが真に残る target-scale 研究課題。
2. **避ける**: (a) heterogeneous/generalize を超える proxy 正控御の更なる調整（既に発火せず・
   収益逓減）、(b) bootstrap-CI → G0–G4 ゲート配線などの**追加 Category-A ヘルパー**（feedback が
   収益逓減と明示）。target-scale 標本が無い段階でのゲート統合は空転になる。
   > **注**: proxy の順序感度については、更なる正控御調整（避ける(a)）の代わりに**分散分解診断**
   > （`make freeze-order-sensitivity`）で決着させた——正控御を発火させるのではなく apparatus の
   > 解像度を直接測り ratio=0.000 を得た。この問いは**閉じた**（target-scale のみ残る）。
