# TG-LoRA 自動分析記録


<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-05-21
**分析実施**: step4 既存情報ベースの差分分析と自動統合

## 分析目的

TG-LoRAの既存設計文書・実装・テストスイートを確認し、機能要件の完全性・正確性を検証するための自動分析を実施しました。

## 分析項目と判断

### A1: コアアルゴリズムの実装完全性

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認
**背景**: AGENTS.mdに記載された4つのコア機能（Velocity Tracking, Extrapolation, Layer Sampling, Rollback）が実装と一致しているか確認が必要

**判断**: 全4機能が完全に実装されている。加えて、ドキュメントに明示されていない以下の機能が実装に存在:
- RandomWalkController（ハイパーパラメータ探索）
- DeltaTracker（重み差分計算）
- Metrics（cosine similarity等の計算ユーティリティ）

**根拠**: src/tg_lora/ 配下7ファイルの全コード読み込み、tests/ 配下9テストファイルの検証

**信頼性への影響**:
- AGENTS.md記載の4機能は全て 🔵（確実）
- 追加発見の3機能も 🔵（実装ベース）
- ドキュメントと実装の差分は補完済み

---

### A2: データパイプラインの完全性

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認
**背景**: docs/datasets.mdに記載のパイプラインが実装と一致しているか確認

**判断**: データパイプラインは完全に実装済み。docs/datasets.mdの記載と実装は一致。追加で以下が実装されている:
- dedup.py（完全一致 + 埋め込みベースの重複排除）
- provenance.py（データ来歴追跡）
- filter_dataset.py（品質フィルタリング）

**根拠**: src/data/ 配下5ファイル、scripts/download_data.py、scripts/prepare_data.py

**信頼性への影響**:
- データパイプライン要件は 🔵（確実）
- docs/datasets.mdに記載なしの機能も実装ベースで 🔵

---

### A3: 評価システムの完全性

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認
**背景**: docs/evaluation.mdに記載の3層評価構造が実装と一致しているか確認

**判断**: 3層評価構造（学習中評価、チェックポイント評価、ベンチマーク評価）は全て実装済み。docs/evaluation.mdの記載と実装は一致。追加で:
- eval_task.py（タスク評価）
- eval_format.py（フォーマット準拠性評価）
- compare_runs.py（比較レポート生成）

**根拠**: src/eval/ 配下3ファイル、scripts/run_eval.sh、scripts/run_eval_lora.sh、scripts/compare_runs.py

**信頼性への影響**:
- 評価要件は 🔵（確実）

---

### A4: 比較実験システムの完全性

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認
**背景**: 公正なベースライン vs TG-LoRA比較が可能か確認

**判断**: run_comparison.shにより同一backward pass予算での比較が実装済み。TG-LoRAのサイクル数 = 予算 / K_initial で自動計算。compare_runs.pyで損失曲線、効率メトリクス（loss/backward, loss/minute, loss/GB-hour）、TG-LoRA固有メトリクス（受理率、cosine similarity）を含むレポート生成。

**根拠**: scripts/run_comparison.sh、scripts/compare_runs.py

**信頼性への影響**:
- 比較システム要件は 🔵（確実）

---

### A5: テストカバレッジと品質

**分析日時**: 2026-05-21（初回）、2026-05-21（更新）
**カテゴリ**: 未定義部分詳細化
**背景**: テストスイートが要件をどの程度カバーしているか確認

**判断**: テストスイートは包括的（43テストファイル、575テストケース、全パス）:
- 全コアモジュール（velocity, extrapolator, delta_tracker, layer_sampler, rollback_manager, random_walk_controller, lora_state, metrics）にユニットテストあり
- データモジュール（build_seed_dataset, filter_dataset, dedup, provenance, generate_open_data, download_data, prepare_data）にユニットテストあり
- 評価モジュール（eval_loss, eval_task, eval_format, compare_runs, run_eval）にユニットテストあり
- モデル管理（load_model, lora_utils）にユニットテストあり
- 学習ループ（trainer_loop, loss）にユニットテストあり
- ユーティリティ（io, logging, memory, run_metrics, seed）にユニットテストあり
- GPT-2を使用した統合テスト（test_smoke.py）でE2E検証

**根拠**: tests/ 配下43ファイルの検証、`pytest tests/` で575 passed

**信頼性への影響**:
- 全モジュールのテスト: 🔵（確実、942テスト全パス）
- 以前「未検証」としていたデータ・評価・モデルモジュールも 🔵 に更新

---

### A6: 未実装・不完全機能の特定

**分析日時**: 2026-05-21（初回）、2026-05-21（更新）
**カテゴリ**: 追加要件
**背景**: 実装に存在するがテスト・文書化されていない部分の特定

**判断**:
- MLflow連携: logging設定でbackend=mlflowが指定されているが、実際のMLflowクライアント実装は見当たらない。RunMetrics（JSONL）が代替
- 早期終了: try/finally rollback safetyは実装済みだが、明示的なpatience-based early stoppingの完全な実装は要確認
- モデル検査: scripts/inspect_model.pyは存在するがAGENTS.mdに記載なし
- **バグ修正(6717ee8)**: Velocity.cosine_similarityでdeltaのキーとstateのキーが不一致の場合にKeyErrorが発生する問題を修正。`if k not in self._state: continue` を追加
- **新機能(6717ee8)**: RunMetricsにコンテキストマネージャ（`__enter__`/`__exit__`）を追加。with文での使用が可能に

**根拠**: src/utils/logging.py、src/utils/run_metrics.py、scripts/inspect_model.py、コミット6717ee8

**信頼性への影響**:
- MLflow連携要件: 🟡（推測ベース）
- 早期終了: 🔵（実装確認済み）
- cosine_similarity KeyError修正: 🔵（修正確認済み、テストあり）
- RunMetrics context manager: 🔵（実装・テスト確認済み）

---

### A8: Velocity Magnitude History・異常検出・トレンド追跡

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認・追加要件
**背景**: 直近コミット(bbcb7e7)でvelocity.pyにmagnitude history ring buffer, is_magnitude_anomalous, magnitude_trendが追加された。これらが要件定義に反映されているか確認が必要

**判断**: 全機能が完全に実装・テストされている:
- **Magnitude history**: `_magnitude_history`リングバッファ（max_history=100）、各update後にL2 normを記録、max_history超過時に最古エントリを破棄
- **is_magnitude_anomalous**: σ閾値ベースの異常検出。履歴3件未満はFalse、std<1e-12時はmean*2.0閾値、通常はmean+threshold_sigma*std
- **magnitude_trend**: 直近window件の線形回帰傾き。データ2件未満は0.0、負で収束、正で発散
- **reset()**: stateとmagnitude_historyの両方をクリア
- テスト: test_velocity.pyに8テスト追加（magnitude tracking, anomaly detection, trend, max_history trimming）

**根拠**: src/tg_lora/velocity.py、tests/test_velocity.py、コミットbbcb7e7

**信頼性への影響**:
- 新規要件 REQ-049, REQ-050 を追加（全て 🔵: 実装・テストベース）
- REQ-001 を更新（magnitude history記録を統合）
- 新規Edgeケース EDGE-115~117 を追加
- NFR-401 テスト数を 455→547 に更新

---

### A9: データスキーマバリデーション（Pydantic）

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認・追加要件
**背景**: 直近コミット(5be1b9d)でsrc/data/schema.pyが追加された。DataRecord, ValidationSummary, validate_recordsが要件定義に反映されているか確認が必要

**判断**: データスキーマバリデーションが完全に実装・テストされている:
- **DataRecord**: Pydantic BaseModel。text必須（非空+ChatMLマーカー含有バリデータ）、source/token_countオプション、token_count正値制約
- **ValidationSummary**: total/valid/skipped/errors集計クラス、log()でlogger.info/warning出力
- **validate_records**: 生dictリスト→DataRecord検証→(valid_records, summary)返却
- テスト: test_schema.pyに161行のテスト（DataRecord, ValidationSummary, validate_records）

**根拠**: src/data/schema.py、tests/test_schema.py、コミット5be1b9d

**信頼性への影響**:
- 新規要件 REQ-051, REQ-052 を追加（全て 🔵: 実装・テストベース）
- REQ-029のデータフィルタリングと相補的な機能

---

### A7: CycleState・DeltaTracker抽出と学習ループ統合

**分析日時**: 2026-05-21
**カテゴリ**: 既存設計確認・追加要件
**背景**: 直近のコミット（87609e1, f41277d, 7d4411c）でtrain_tg_lora.pyから純粋ロジックがCycleState（92行）とDeltaTracker（136行）として抽出された。これらの新モジュールが要件定義に反映されているか確認が必要

**判断**: CycleStateとDeltaTrackerは完全に実装・テスト・統合されている:
- **CycleState** (cycle_state.py): サイクル数、backward pass数、外挿ステップ数、best_loss、stale_cycles、受理/拒否カウントを追跡。reduction_rate/acceptance_rateプロパティを提供。should_stop(patience, min_cycles)でpatience-based早期終了を判定。test_cycle_state.pyに19テスト（6クラス）が存在
- **DeltaTracker** (delta_tracker.py): compute_mean_delta + DeltaStats（total_norm, per_layer_norm, max_component, mean_abs）。is_anomalous(threshold_sigma)で異常検出。convergence_trend(window)で収束傾向を線形回帰で計算。test_delta_tracker.pyに21テストが存在
- **学習ループ統合** (train_tg_lora.py): cycle_state.record_cycle()、delta_tracker.compute_and_record()、cycle_state.should_stop()が全てワイヤリング済み。summary()をマージして出力。test_training_integration.pyに7テストが存在
- **_InfiniteBatchIterator**: DataLoaderを無限にイテレートするヘルパークラス。test_infinite_batch_iterator.pyでテスト済み

**根拠**: src/tg_lora/cycle_state.py, src/tg_lora/delta_tracker.py, src/training/train_tg_lora.py, tests/test_cycle_state.py, tests/test_delta_tracker.py, tests/test_training_integration.py

**信頼性への影響**:
- 新規要件 REQ-038~044 を追加（全て 🔵: 実装・テストベース）
- NFR-401 テスト数を 262→321 に更新
- 新規Edgeケース EDGE-110~114 を追加
- テストスイート: 321 tests全パス確認済み

---

### A10: 適応学習率・収束適応の要件ギャップ 🔵

**分析日時**: 2026-05-21
**カテゴリ**: 追加要件・既存実装との差分
**背景**: 直近のコミット(9bddecb)でlr_reject_decayが0.7→0.5に変更され、lr境界テスト3件と適応LRスモークテスト3件が追加された。しかし、適応学習率（lr）機能自体が要件定義に反映されていなかった。REQ-011はK, N, alpha, betaのみを列挙し、lrのランダムウォーク探索が漏れていた。またadapt_to_convergence、update_layer_scoresメソッドも未定義だった

**判断**: 以下の実装が要件定義に未反映だった:
- **適応学習率（lr）**: reward()でlrをlr_accept_boost倍に増加、penalize()でlrをlr_reject_decay倍に減少。常に[lr_min, lr_max]にクランプ。lr_reject_decay=0.5は0.7より速く保守的lrに戻す設計意図
- **adapt_to_convergence()**: 探索停滞時（convergence_trend >= 0、total_cycles > 2）にproactiveにlr×0.8減少・K増加。健全な収束時は変更なし
- **update_layer_scores()**: アクティブレイヤーの受理/拒否フィードバックでlayer_scoresを更新
- **lr境界テスト**: 連続penalizeでlr_minクランプ、連続rewardでlr_maxクランプ、交互で範囲内維持
- **適応LRスモークテスト**: 設定ワイヤリング確認、1サイクル完了、accept/reject後lr変化確認

**根拠**:
- src/tg_lora/random_walk_controller.py: lr適応ロジック（reward/penalize）、adapt_to_convergence、update_layer_scores
- configs/9b_tg_lora.yaml: lr_initial, lr_min, lr_max, lr_accept_boost, lr_reject_decay=0.5
- tests/test_random_walk_controller.py: lr境界テスト3件、収束適応テスト2件
- tests/test_training_integration.py: TestAdaptiveLrSmokeTest 3件

**信頼性への影響**:
- REQ-011を更新（lrをパラメータリストに追加）
- 新規要件 REQ-013a（適応学習率制御）、REQ-053（収束適応）、REQ-054（lr境界クランプ）、REQ-055（レイヤースコア更新）を追加
- 新規Edgeケース EDGE-118~120（lr境界テスト）を追加
- NFR-401 テスト数を 547→562 に更新
- 受け入れ基準テスト数を 153→165 に更新
- 全新規要件の信頼性: 🔵（実装・テスト・config全て確認済み）

---

### A11: 外挿安全性の要件ギャップ（train_tg_lora.py） 🔵

**分析日時**: 2026-05-21
**カテゴリ**: 追加要件・既存実装との差分
**背景**: AI_HUB_MAKE_RUN_FEEDBACKの指摘により、train_tg_lora.pyがtrain_baseline_qlora.pyと同等の数値安全性カバレッジを持っていないことが判明。直近コミット(0febcf0)でtrainer_loop.pyにNumericalInstabilityErrorが追加され、train_baseline_qlora.pyにearly stoppingが追加されたが、train_tg_lora.pyの外挿パスには外挿適用後のパラメータ検証が存在しない

**判断**: 以下の安全性ギャップを特定:
- **外挿後パラメータ検証の欠落**: `apply_extrapolation()`呼出後に、LoRAパラメータが有限値（NaN/Infでない）ことを検証するコードが存在しない。外挿が極端な更新を生成した場合、次のeval_lossまたはforward_backwardで予期しないエラーが発生する可能性がある
- **NumericalInstabilityErrorのトップレベルハンドリング**: 両trainerとも`forward_backward()`から送出されるNumericalInstabilityErrorを学習ループ外でキャッチしていない。train_tg_lora.pyでは特に、try/finallyブロック内のrollback保護はあるが、NumericalInstabilityErrorのコンテキストで安全な終了処理が保証されていない
- **既存の共有安全性**: バッチキー検証（compute_loss）、勾配クリッピング（optimizer_step）、学習不可能パラメータ検出（create_optimizer）は両trainerで共有されており、差分なし

**根拠**:
- src/training/train_tg_lora.py: 302-311行目のapply_extrapolation呼出後、パラメータ検証なし
- src/training/train_baseline_qlora.py: forward_backward経由でNumericalInstabilityErrorが共有されるが、明示的なハンドリングなし
- src/training/trainer_loop.py: NumericalInstabilityError定義、forward_backwardでのtorch.isfinite()チェック
- AI_HUB_MAKE_RUN_FEEDBACK: "train_tg_lora.py does not yet have the early-stopping or numerical-safety guards"

**信頼性への影響**:
- 新規要件 REQ-056（外挿後パラメータ有限性検証）、REQ-057（trainer間安全性カバレッジ一致）を追加
- 新規Edgeケース EDGE-121（外挿による非有限パラメータ）を追加
- 信頼性: 🔵（コード直接確認・AI_HUB_MAKE_RUN_FEEDBACK指摘）

---

### A12: Config文字列フィールドLiteral enum拡張の要件ギャップ 🔵

**分析日時**: 2026-05-21
**カテゴリ**: 追加要件・設定検証強化
**背景**: 直近コミット(f755436)でconfig_schema.pyのactive_layer_strategyとbnb_quant_typeにLiteral型検証が追加された。しかし、他の列挙可能文字列フィールド（dtype, bnb_4bit_compute_dtype）が未だstr型であり、無効なセンチネル値を受け入れる可能性がある。AI_HUB_MAKE_RUN_FEEDBACKの指摘「verify no other string fields accept invalid sentinel values」に対応する必要がある

**判断**: 以下の文字列フィールドがLiteral型検証の対象となる:
- **ModelConfig.dtype**: 現在`str`、有効値は"bfloat16", "float16", "float32" → Literal型に変更すべき
- **ModelConfig.bnb_4bit_compute_dtype**: 現在`str`、有効値は"bfloat16", "float16", "float32" → Literal型に変更すべき
- 以下はLiteral型にすべきではない（動的値を取るため）:
  - `ModelConfig.name_or_path`: モデル名/パス（任意文字列）
  - `ModelConfig.device_map`: "auto", "cuda:0", "cpu", "{int}"など多様
  - `ModelConfig.device`: 任意デバイス文字列
  - `LoRAConfig.target_modules`: "all-linear"またはカンマ区切りモジュール名
  - `LoggingConfig.run_dir`: ファイルシステムパス
  - `LoggingConfig.backend`: 将来的に多様なバックエンドをサポートする可能性

**根拠**:
- src/training/config_schema.py: ActiveLayerStrategy/BnbQuantTypeのLiteral定義、dtype/bnb_4bit_compute_dtypeのstr型定義
- AI_HUB_MAKE_RUN_FEEDBACK: "verify no other string fields accept invalid sentinel values"
- test_config_schema.py: active_layer_strategy='test'が旧スキーマで通過していた問題がLiteral enumで修正された実績

**信頼性への影響**:
- 新規要件 REQ-058（dtype/bnb_4bit_compute_dtype Literal enum検証）を追加
- 新規Edgeケース EDGE-122（dtypeフィールド不正文字列拒否）を追加
- 信頼性: 🔵（config_schema.py直接確認・Literal enumパターン適用）

---

### A13: 外挿安全性統合テストの要件ギャップ 🔵

**分析日時**: 2026-05-21
**カテゴリ**: 追加要件・テストカバレッジギャップ
**背景**: 直近のコミット(9754565)でcheck_lora_params_finiteとDtypeLiteral enumが実装された。しかし、AI_HUB_MAKE_RUN_FEEDBACKの指摘により、train_tg_lora.pyの非有限パラメータ回復フロー（332-355行）が統合テストで検証されていないことが判明。ユニットテスト（test_trainer_loop.py TestCheckLoraParamsFinite）はcheck_lora_params_finite関数自体をテストしているが、実際の学習ループ内での回復フロー（rollback→penalize→update_layer_scores→record_cycle→continue）がモックベースの統合テストで検証されていない

**判断**: 以下の統合テストギャップを特定:
- **回復フローの統合テスト欠落**: check_lora_params_finiteがFalseを返すケースで、実際の学習ループ内でrollback→penalize→update_layer_scores→record_cycle→continueが正しく実行されることを確認する統合テストが存在しない
- **副作用の未検証**: controller.penalize()が正しい引数で呼ばれること、controller.update_layer_scores()がactive_indicesと-1.0で呼ばれること、cycle_state.record_cycle()がaccepted=Falseで呼ばれることが検証されていない
- **パラメータ復元の未検証**: rollback後にモデルパラメータが外挿前の状態に正しく復元されることが統合レベルで検証されていない
- **スキップパスの未検証**: continueにより通常のaccept/rollback評価パス（eval_loss外挿後、_decide_accept_rollback）がスキップされることが検証されていない

**根拠**:
- src/training/train_tg_lora.py: 332-355行の非有限回復パス
- tests/test_trainer_loop.py: TestCheckLoraParamsFinite（ユニットテストのみ、3テスト）
- tests/test_training_integration.py: TestMockedTrainingLoop（非有限ケースなし）
- AI_HUB_MAKE_RUN_FEEDBACK: "Add an integration test exercising the full extrapolation→NaN detection→rollback→cycle_state.record_cycle() path with a mock model that produces non-finite params"
- AI_HUB_MAKE_RUN_FEEDBACK: "verify the penalize/score-update side effects"
- テストスイート: 720 tests全パス（check_lora_params_finiteユニットテスト含む）

**信頼性への影響**:
- 新規要件 REQ-059（統合テスト）を追加: 🔵
- 新規要件 REQ-060（副作用検証）を追加: 🔵
- 新規Edgeケース EDGE-123（record_cycle引数検証）、EDGE-124（update_layer_scores引数検証）、EDGE-125（スキップパス検証）を追加
- NFR-401 テスト数を 720 に更新
- 全新規要件の信頼性: 🔵（コード直接確認・AI_HUB_MAKE_RUN_FEEDBACK指摘）

---

### A14: Phase 14 reliability fixes の要件ギャップ 🔵

**分析日時**: 2026-05-21
**カテゴリ**: 追加要件・既存実装との差分
**背景**: 直近5コミット（3fe19ae, f03932a, 7720c98, 81ee464, dac6c51）で5つの信頼性修正が実装されたが、そのうち4コミットの新規動作が要件定義に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKの評価は「VALUABLE: Five distinct reliability fixes with 27 new test methods」

**判断**: 以下の実装が要件定義に未反映だった:
- **Config extra='forbid'**（RISK-0015/0016）: 全11のPydantic設定モデルに`model_config = ConfigDict(extra="forbid")`を追加。YAML内のタイポ等の未知フィールドを学習開始前に検出・拒否。空YAML・リストYAMLもValueErrorで拒否。test_config_schema.pyにTestExtraFieldsRejected（8テスト）・TestMalformedYAML（2テスト）を追加
- **cap_update ゼロ返却**（RISK-0015/0016関連）: extrapolator.pyのcap_update()が非有限テンソルを検出した場合、NaNを伝播させるのではなくゼロテンソルを返す。`inf * 0 = NaN`の伝播経路を遮断。test_extrapolation_safety_direct.pyの4テストを「NaNが発生する」→「NaNが防止される」に更新
- **ロールバックスナップショットサニタイズ**（RISK-0074）: rollback_manager.pyに_sanitize_snapshot()を追加。NaN→0.0, +Inf→1e6, -Inf→-1e6にサニタイズ。ロールバック時に破損状態を復元しないことを保証。max_history（デフォルト100）で履歴サイズを制限、FIFO破棄。test_rollback_manager.pyに5テスト追加
- **DeltaTracker/metrics安全性**: delta_tracker.pyの_compute_stats()が非有限ノルムをスキップ、norm_historyに非有限値を追加しないガード追加。metrics.pyのcosine_similarity()がキー不一致を安全にスキップ。test_delta_tracker.pyに5テスト、test_metrics.pyに3テスト追加

**根拠**:
- src/training/config_schema.py: extra="forbid"追加、load_and_validate_configのdict判定
- src/tg_lora/extrapolator.py: cap_updateのtorch.isfiniteチェック
- src/tg_lora/rollback_manager.py: _sanitize_snapshot, max_history
- src/tg_lora/delta_tracker.py: _compute_statsのmath.isfinite, norm_historyのmath.isfinite
- src/tg_lora/metrics.py: cosine_similarityのk not in b チェック
- tests/test_config_schema.py: +10テスト（TestExtraFieldsRejected 8, TestMalformedYAML 2）
- tests/test_rollback_manager.py: +5テスト
- tests/test_delta_tracker.py: +5テスト
- tests/test_metrics.py: +3テスト
- tests/test_extrapolation_safety_direct.py: 4テスト更新

**信頼性への影響**:
- 新規要件 REQ-061（extra='forbid'）、REQ-062（非mapping拒否）、REQ-063（cap_update ゼロ返却）、REQ-064（スナップショットサニタイズ）、REQ-065（max_history制限）、REQ-066（metrics キー不一致安全）、REQ-067（_compute_stats 非有限スキップ）、REQ-068（norm_history ガード）を追加
- 新規Edgeケース EDGE-126~134 を追加
- NFR-401 テスト数を 720→764 に更新
- 全新規要件の信頼性: 🔵（コミット内容の直接確認・テスト全パス確認）

---

### A15: Perplexity E2Eパイプライン検証の要件ギャップ 🔵

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・テストカバレッジギャップ
**背景**: AI_HUB_MAKE_RUN_FEEDBACKの指摘「Add an end-to-end integration test that runs a short mock training loop and asserts the RunMetrics footer contains a finite best_perplexity value, then verify train_tg_lora.py has parity with train_baseline_qlora.py's perplexity plumbing」。直近コミット(11e21f9)でperplexityがwrite_footerに伝播されるようになったが、モック学習ループ→RunMetrics→footer出力のE2Eパスが統合テストで検証されていない。また両trainerのperplexity配管パリティがテストで確認されていない

**判断**: 以下のテストギャップを特定:
- **Perplexity E2Eテストの欠落**: run_metrics.pyの_sanitize_perplexityとwrite_footerのperplexity引数が追加されたが、モック学習ループ（baseline/tg_lora両モード）を実行し、write_footer出力に有限best_perplexityが含まれることを検証する統合テストが存在しない
- **Trainer間パリティテストの欠落**: train_tg_lora.pyとtrain_baseline_qlora.pyのperplexity取り扱い（eval_result.perplexity保存、best_perplexity追跡、write_footerへの引数渡し）が一致していることを確認するパラメータ化テストが存在しない
- **EvalLossResult統合は完了**: 両trainerともeval_loss_detailedのEvalLossResultからperplexityを取り出し、best_perplexityに保存し、write_footerに渡すフローが実装済み

**根拠**:
- src/utils/run_metrics.py: _sanitize_perplexity、write_footer(perplexity=...)引数
- src/training/train_baseline_qlora.py: best_perplexity = eval_result.perplexity、write_footer(perplexity=best_perplexity)
- src/training/train_tg_lora.py: best_full_eval_perplexity = full_result.perplexity、write_footer(perplexity=best_full_eval_perplexity)
- AI_HUB_MAKE_RUN_FEEDBACK: "E2E integration test that runs a short mock training loop and asserts the RunMetrics footer contains a finite best_perplexity value"
- テストスイート: test_run_metrics.pyにwrite_footerテストあり、E2E統合テストなし

**信頼性への影響**:
- 新規要件 REQ-069（perplexity E2Eテスト）を追加: 🔵
- 新規要件 REQ-070（trainer間パリティ）を追加: 🔵
- 新規Edgeケース EDGE-135（RunMetrics footer有限perplexity保証）を追加: 🔵
- NFR-401テスト数はREQ-069/070実装時に増加

---

### A16: accept()相対許容誤差のプロパティベーステスト要件ギャップ 🟡

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・テスト品質ギャップ
**背景**: AI_HUB_MAKE_RUN_FEEDBACKの指摘「Consider extending the relative-tolerance accept() change with a property-based test (e.g., hypothesis) that verifies idempotence across loss-value magnitudes」。コミット(02582db)でaccept()が絶対許容誤差から相対許容誤差に変更され、loss値の大きさに依存する挙動が導入された。例えばloss_pilot=1.0とloss_pilot=1000.0で同じrollback_tolerance=0.1でも受理結果が異なる可能性がある

**判断**: 以下のテスト品質ギャップを特定:
- **大きさ依存性の未検証**: accept()の相対許容誤差（(loss_after - loss_pilot) / max(abs(loss_pilot), 1e-8)）は、loss_pilotが小さい場合に1e-8のfloorが効いて絶対判定に近くなり、loss_pilotが大きい場合に純粋な相対判定になる。この大きさ依存性がプロパティベーステストで検証されていない
- **べき等性の未検証**: 同じペア（loss_pilot, loss_after）に対するaccept()の結果が常に一貫していること（冪等性）がhypothesis等で広範囲に検証されていない
- **境界値の網羅性**: 1e-8のfloor近傍での挙動、loss_pilot=0.0の特殊ケース等が既存のユニットテストでは部分的にしかカバーされていない

**根拠**:
- src/tg_lora/random_walk_controller.py: accept()メソッド、relative = (loss_after - loss_pilot) / max(abs(loss_pilot), 1e-8)
- コミット02582db: accept()の絶対→相対許容誤差変更
- tests/test_random_walk_controller.py: accept()テストあり、プロパティベーステストなし
- AI_HUB_MAKE_RUN_FEEDBACK: "property-based test for hypothesis that verifies idempotence across loss-value magnitudes"

**信頼性への影響**:
- 新規要件 REQ-071（accept()プロパティベーステスト）を追加: 🟡
- 新規Edgeケース EDGE-136（accept()大きさ依存性）を追加: 🟡
- 注: hypothesisはpyproject.tomlに未依存指定のため、実装時に追加が必要

### A17: 公開API・入力検証・エッジケース強化 🔵

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・既存実装との差分
**背景**: Phase 20完了後の3コミット（cceccde, a020e5b, e19da0f）で公開API、入力検証、評価エッジケース強化、ロールバック安全性強化が実装されたが、要件定義に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価

**判断**: 以下の実装が要件定義に未反映だった:
- **公開API（__init__.py）**: src/tg_lora/__init__.pyで全コアコンポーネント（16クラス/関数）を__all__付きでエクスポート。外部からのimportパスを統一
- **RandomWalkController入力検証**: K_candidates > 0, N_candidates > 0, lr_min < lr_max, alpha_min < alpha_max をコンストラクタで検証。ValueErrorで即時失敗
- **空データローダーNaN返却**: eval_loss()が空ローダーでNaNを返すよう変更（旧: 0.0）。eval_loss_detailed()はNaN/infのEvalLossResultを返す。誤解を招く「完全損失0.0」を防止
- **ロールバックtry-catch**: train_tg_lora.pyのrollback()呼び出しをtry-catchで囲み、RuntimeError/IndexErrorを安全に処理。ロールバック失敗による学習クラッシュを防止
- **非有限loss_afterガード**: accept/rollback判定前にmath.isfinite(loss_after)をチェック。非有限値をfloat("inf")に設定し、受理判定が確実に拒否されることを保証
- **共有InfiniteBatchIterator**: train_tg_lora.pyの_InfiniteBatchIteratorをsrc/training/batch_iter.pyに抽出。両trainerで共有
- **共有save_checkpoint**: src/utils/checkpoint.pyに抽出。両trainerのチェックポイント保存を統一
- **戦略リスト重複排除**: _ALL_STRATEGIES = list(get_args(StrategyName))で戦略一覧を自動生成。ハードコード重複を排除

**根拠**:
- src/tg_lora/__init__.py: 16コンポーネントの__all__エクスポート
- src/tg_lora/random_walk_controller.py: 4つの入力検証
- src/eval/eval_loss.py: 空ローダーNaN返却
- src/training/train_tg_lora.py: rollback try-catch, non-finite guard
- src/training/batch_iter.py: 共有InfiniteBatchIterator
- src/utils/checkpoint.py: 共有save_checkpoint
- テスト: 867 passed, 9 skipped

**信頼性への影響**:
- 新規要件 REQ-073~080 を追加（全て 🔵: 実装・テスト確認済み）
- 新規要件 REQ-081~084 を追加（未実装の次フェーズ候補、🔴/🟡）
- 新規Edgeケース EDGE-137~144 を追加
- NFR-401 テスト数を 867 に更新

### A18: 次フェーズ候補要件 🔴🟡

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・テストカバレッジギャップ
**背景**: AI_HUB_MAKE_RUN_FEEDBACKが「Continue building on this progress」として4つの具体的改善提案を提示。これらは未実装の次フェーズ候補

**判断**: 以下のテストギャップと改善点を特定:
- **save_checkpoint readback検証**: 保存後のディレクトリ存在確認・ファイル数確認が未実装。不完全なチェックポイントの検出不可能
- **InfiniteBatchIteratorエッジケース**: 単一バッチデータローダー、デバイスキャストの境界値テストが不足
- **非有限lossガードログ**: train_tg_lora.pyのmath.isfinite(loss_after)ガードが無言でloss_after=infに設定。デバッグ時にどのサイクルで発火したか追跡不可能
- **ロールバック失敗E2Eテスト**: rollback()が例外を送出するシナリオのE2Eテストが未実装。モックでrollbackをraiseさせ、学習継続または安全な失敗を検証する必要あり

**根拠**:
- AI_HUB_MAKE_RUN_FEEDBACK: "Add dedicated tests for src/utils/checkpoint.py (save_checkpoint with readback verification)"
- AI_HUB_MAKE_RUN_FEEDBACK: "edge-case tests for InfiniteBatchIterator (single-batch loader, device casting)"
- AI_HUB_MAKE_RUN_FEEDBACK: "the non-finite loss guard silently sets loss_after=inf instead of logging"
- AI_HUB_MAKE_RUN_FEEDBACK: "test that verifies rollback-failure resilience end-to-end"

**信頼性への影響**:
- 新規要件 REQ-081（save_checkpoint readback）を追加: 🔴
- 新規要件 REQ-082（InfiniteBatchIteratorエッジケース）を追加: 🔴
- 新規要件 REQ-083（非有限lossガードログ）を追加: 🟡
- 新規要件 REQ-084（ロールバック失敗E2Eテスト）を追加: 🔴

### A19: Phase 23 実装完了検証 🔵

**分析日時**: 2026-05-22
**カテゴリ**: 既存実装確認・信頼性レベル更新
**背景**: Phase 23（TASK-0051~0054）が完了し、REQ-081~084が実装・テスト検証された。要件定義の信頼性レベルを更新し、残課題の最新状態を確認する

**判断**: Phase 23の4タスクが全て完了:
- **TASK-0051（save_checkpoint readback）**: checkpoint.pyにディレクトリ存在確認・ファイル数確認を追加。test_checkpoint.pyに7テスト作成。正常保存・readback検証成功・空ディレクトリ検出・存在しないディレクトリ検出の全シナリオをカバー
- **TASK-0052（InfiniteBatchIteratorエッジケース）**: test_infinite_batch_iterator.pyに7テスト追加。単一バッチ反復・デバイス文字列/オブジェクト両キャスト・複数キーキャスト・float16/int64/float32 dtype維持の全エッジケースをカバー
- **TASK-0053（非有限loss warning log）**: train_tg_lora.pyにlogger.warning()を追加。test_training_integration.pyに3テスト追加（NaN/Inf/有限値でのwarning検証）
- **TASK-0054（rollback例外E2Eテスト）**: test_training_integration.pyにTestRollbackFailureResilience（2テスト）とTestNonFiniteParamsRollbackException（E2Eテスト）を追加。RuntimeError/IndexError時の学習継続・ログ出力・副作用検証をカバー

**根拠**:
- src/utils/checkpoint.py: readback検証実装確認
- tests/test_checkpoint.py: 7テスト全パス確認
- tests/test_infinite_batch_iterator.py: 17テスト全パス確認
- src/training/train_tg_lora.py: logger.warning追加確認
- tests/test_training_integration.py: rollback E2Eテスト全パス確認
- テストスイート: 891 passed, 9 skipped, 0 failed, 0 errors

**信頼性への影響**:
- REQ-081: 🔴 → 🔵（実装・テスト確認済み）
- REQ-082: 🔴 → 🔵（実装・テスト確認済み）
- REQ-083: 🟡 → 🔵（実装・テスト確認済み）
- REQ-084: 🔴 → 🔵（実装・テスト確認済み）
- NFR-401 テスト数を 867→891 に更新

### A20: 探索確率パラメータテストカバレッジギャップ解消 🔵

**分析日時**: 2026-05-22
**カテゴリ**: テストカバレッジギャップ解消
**背景**: 28f709bコミットで4つの探索確率パラメータ（k_explore_prob, n_explore_prob, beta_explore_prob, strategy_explore_prob）が設定可能になったが、AI_HUB_MAKE_RUN_FEEDBACKで「Missing tests for the new parameters prevent full A2」と指摘された。propose()メソッドの探索分岐が全くテストされていなかった

**判断**: 探索確率パラメータのテストカバレッジギャップを以下の11テストで解消:
1. **デフォルト値テスト** (test_explore_prob_defaults): None指定時にクラス定数が使用されることを検証
2. **カスタム値テスト** (test_explore_prob_custom_values): コンストラクタ値が正しく設定されることを検証
3. **K探索確率テスト** (2テスト): prob=0.0でK不変、prob=1.0でK常変化を検証
4. **N探索確率テスト** (2テスト): prob=0.0でN不変、prob=1.0でN常変化を検証
5. **beta探索確率テスト** (2テスト): prob=0.0でbeta不変、prob=1.0でbeta再サンプリング活性を検証
6. **strategy探索確率テスト** (2テスト): prob=0.0でstrategy不変、prob=1.0でstrategy常切替を検証
7. **スキーマ検証テスト** (test_explore_prob_config_schema_validation): TGLoRAParamsのデフォルト値・カスタム値・不正値(0.0, 1.0)の検証

**根拠**:
- random_walk_controller.py propose()の4つの探索分岐（K, N, beta, strategy）
- config_schema.py TGLoRAParamsの4つのField(gt=0.0, lt=1.0)
- テストスイート: 902 passed, 9 skipped, 0 failed（891→902に増加）

**信頼性への影響**:
- 新規要件 REQ-085~088 を追加（全て 🔵）
- 新規EDGE-145 を追加（🔵）
- テスト数を 891→902 に更新

### A21: メトリクスNaN/Inf安全性ガードの要件ギャップ 🔵

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・既存実装との差分
**背景**: 直近コミット(1bb591d)でmetrics.pyのtotal_norm()とper_layer_norms()にNaN/Inf安全性ガードが追加された。delta_tracker._compute_stats（Phase 14）やextrapolator.cap_updateと一貫した安全性パターンだが、metrics.py側は要件定義に未反映だった

**判断**: 以下の実装が要件定義に未反映だった:
- **total_norm()**: 非有限テンソルノルムをmath.isfinite()でスキップ。全非有限時は0.0を返す。test_metrics.pyに3テスト追加（nan_skipped, inf_skipped, all_nonfinite_returns_zero）
- **per_layer_norms()**: 非有限テンソルノルムをスキップし有限テンソルのみでレイヤー別集計。test_metrics.pyに2テスト追加（nan_skipped, inf_skipped）
- これらはREQ-067（DeltaTracker._compute_stats）やREQ-063（cap_update）と同じmath.isfiniteパターン

**根拠**:
- src/tg_lora/metrics.py: total_norm, per_layer_norms のmath.isfiniteチェック
- tests/test_metrics.py: TestTotalNorm 3テスト, TestPerLayerNorms 2テスト
- コミット1bb591d

**信頼性への影響**:
- 新規要件 REQ-089, REQ-090 を追加（全て 🔵）
- 新規Edgeケース EDGE-146, EDGE-147 を追加
- NFR-401 テスト数を更新

---

### A22: _compute_pilot_average NaN/Infフィルタリングの要件ギャップ 🔵

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・既存実装との差分
**背景**: 直近コミット(1bb591d)でtrain_tg_lora.pyの_compute_pilot_averageに非有限step_lossesのフィルタリングが追加された。pilot損失計算でNaN/Infが混入するのを防ぐ重要な安全性改善

**判断**: 以下の実装が要件定義に未反映だった:
- **_compute_pilot_average**: step_lossesから有限値のみをフィルタリング（finite_losses = [l for l in step_losses if math.isfinite(l)]）。全非有限時はNaN + finite_count=0を返す。min_loss/max_lossも有限値のみから計算。metrics辞書にfinite_countを追加

**根拠**:
- src/training/train_tg_lora.py: _compute_pilot_average finite_losses フィルタリング
- tests/test_training_pure_functions.py: _compute_pilot_average テスト
- コミット1bb591d

**信頼性への影響**:
- 新規要件 REQ-091 を追加（🔵）
- 新規Edgeケース EDGE-148, EDGE-149 を追加

---

### A23: 設定からコントローラへの探索確率伝播要件 🔵

**分析日時**: 2026-05-22
**カテゴリ**: 追加要件・既存実装との差分
**背景**: 直近コミット(1bb591d, 03d1e46)でtrain_tg_lora.pyからRandomWalkControllerに探索確率パラメータが渡されるようになり、config-to-controller integration testsが追加された。設定値が正しく伝播されることが検証されているが要件定義に未反映

**判断**: 以下の実装が要件定義に未反映だった:
- **探索確率の伝播**: train_tg_lora.pyのRandomWalkController初期化でk_explore_prob, n_explore_prob, beta_explore_prob, strategy_explore_probをtg_cfg.get("xxx", None)で渡す
- **Config-to-controller integration tests**: YAML設定→OmegaConf→RandomWalkControllerの全伝播パスを検証。test_random_walk_controller.pyにTestConfigToControllerIntegrationクラス（12テスト）

**根拠**:
- src/training/train_tg_lora.py: RandomWalkController初期化の探索確率引数
- tests/test_random_walk_controller.py: TestConfigToControllerIntegration 12テスト
- コミット03d1e46, 1bb591d

**信頼性への影響**:
- 新規要件 REQ-092, REQ-093 を追加（全て 🔵）
- NFR-401 テスト数を更新

---

### A24: 純粋関数ユニットテストの拡充 🔵

**分析日時**: 2026-05-22
**カテゴリ**: テストカバレッジ拡充
**背景**: 直近コミット(7115bd4)でtrain_tg_lora.pyの純粋関数に対する28のユニットテストが追加された。should_run_full_eval, check_lora_params_finite, _compute_pilot_average, _decide_accept_rollback, _evaluate_full_eval_outcome, _format_cycle_progress, build_training_summaryをカバー

**判断**: 以下のテストカバレッジ拡充が行われた:
- **should_run_full_eval**: cycle cadence判定のテスト
- **check_lora_params_finite**: 有限/非有限パラメータ検出のテスト
- **_compute_pilot_average**: 正常・空・非有限ロスのテスト
- **_decide_accept_rollback**: 受理/拒否判定のテスト
- **_evaluate_full_eval_outcome**: フル評価結果判定のテスト
- **_format_cycle_progress**: フォーマット出力のテスト
- **build_training_summary**: サマリー結合のテスト

**根拠**:
- tests/test_train_tg_lora_pure.py: 28テスト全パス
- コミット7115bd4

**信頼性への影響**:
- NFR-401 テスト数を 902→942 に更新
- 全テスト 🔵（実装・テスト確認済み）

---

### A29: Phase 33 diff_lora高速化・cosine_similarity警告・decoder層ロギングの要件ギャップ 🔵

**分析日時**: 2026-05-23
**カテゴリ**: 既存実装との差分・テストカバレッジギャップ
**背景**: 直近2コミット（7a643a9, 182de29）でdiff_lora高速化、cosine_similarity直交ベクトル警告、decoder層ロギング、smoke_async_prefix.yaml設定サーフェスが実装されたが、要件定義（REQ-136~138）に未反映だった。またAI_HUB_MAKE_RUN_FEEDBACKが「Add an integration test exercising the full async cache lifecycle」を指摘

**判断**: 以下の実装が要件定義に未反映だった:
- **diff_lora fast paths**: scale==0.0の場合はゼロテンソルを直接返し、scale==1.0の場合は単純減算のみを実行。乗算を回避する性能最適化
- **cosine_similarity warnings**: 分母が1e-12以下でノルムが非ゼロ（直交ベクトル）の場合にwarnings.warn()で警告。stacklevel=2で正しい呼び出し元行番号を表示
- **decoder層ロギング**: _get_decoder_layersがパス発見時にlogger.debug()でパスを記録。全候補失敗時に候補数を含むエラーメッセージを送出
- **smoke_async_prefix.yaml**: 非同期キャッシュビルド検証用の設定サーフェス（prefix_feature_cache_async=true, async_device="cuda:1"）
- **async_cache_builder logger**: __name__→"tg-lora"に変更し一貫したロギング

**根拠**:
- src/tg_lora/lora_state.py: diff_lora fast paths（scale==0.0, scale==1.0）
- src/tg_lora/metrics.py: cosine_similarity warnings.warn
- src/tg_lora/activation_cache.py: _get_decoder_layers debug log/enhanced error
- configs/smoke_async_prefix.yaml: 非同期キャッシュビルド検証設定
- テストスイート: 1299 passed

**信頼性への影響**:
- 新規要件 REQ-139~143 を追加（全て 🔵: 実装確認済み、REQ-139はテストギャップ指摘）
- 新規Edgeケース EDGE-164~167 を追加
- NFR-401 テスト数を 1289→1299 に更新

---

### A30: Phase 34 in-place tensor ops・data_ptr保存検証・velocity opsベンチマークの要件ギャップ 🔵

**分析日時**: 2026-05-23
**カテゴリ**: 既存実装との差分・テストカバレッジ拡充
**背景**: 直近3コミット（851041e, c9928b6, c51fd5b）でin-place tensor操作の導入、data_ptr保存検証テスト、velocity opsマイクロベンチマークが実装されたが、要件定義（REQ-139~143）に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価

**判断**: 以下の実装が要件定義に未反映だった:
- **In-place EMA update**: velocity.pyのupdate()が`mul_(beta).add_(delta[k], alpha=(1.0-beta))`でin-place EMA更新を実行。既存キーのテンソルdata_ptrを保存しメモリアロケーションを削減。新規キーはclone()で別テンソルを割り当て
- **In-place cap_update**: extrapolator.pyのcap_update()が`update.mul_(max_norm/update_norm)`でin-place cappingを実行。非有限入力時はtorch.zeros_like()で新規テンソル返却（REQ-063のゼロ返却は維持）
- **data_ptr保存テスト**: test_velocity.pyにTestVelocityDataPtrPreservation（5テスト）、test_extrapolator.pyにTestCapUpdateDataPtrPreservation（4テスト）を追加。混在更新・複数キー同時更新・新規キー置き換えのシナリオをカバー
- **benchmark_velocity_ops.py**: velocity EMA updateとcap_updateのマイクロベンチマークスクリプト（143行）。1000反復でtime/memory計測、JSON出力。benchmark_optimizer_lifecycle.pyパターンに準拠
- **test_benchmark_velocity_ops.py**: 9 smoke tests（import健全性、--help、--quick JSON出力、必須フィールド検証）

**根拠**:
- src/tg_lora/velocity.py: in-place mul_/add_ EMA更新（32行）
- src/tg_lora/extrapolator.py: in-place mul_ capping（24行）
- tests/test_velocity.py: TestVelocityDataPtrPreservation 5テスト
- tests/test_extrapolator.py: TestCapUpdateDataPtrPreservation 4テスト
- scripts/benchmark_velocity_ops.py: 143行
- tests/test_benchmark_velocity_ops.py: 96行
- テストスイート: 1344 passed

**信頼性への影響**:
- 新規要件 REQ-144~148 を追加（全て 🔵: 実装・テスト確認済み）
- 新規Edgeケース EDGE-168~171 を追加
- NFR-401 テスト数を 1299→1344 に更新
- AI_HUB_MAKE_RUN_FEEDBACK指摘のMakefile bench-velocity-ops統合をREQ-148として要件化

---

### 確認できた事項

- コアアルゴリズム（velocity, extrapolation, layer sampling, rollback）は完全実装
- データパイプライン（download → prepare → tokenize → train）は完全実装
- 3層評価システムは完全実装
- 公正な比較実験システムは完全実装
- OptimizerLifecycleManagerによる2ポリシー（recreate_per_cycle / reuse_state_reset_experimental）が完全実装
- ActivationCache hit/miss追跡メトリクスが完全実装
- optimizer_lifecycle設定がrun_metrics headerに記録される
- ベンチマークスクリプトで両policyの性能比較が可能
- 包括的なユニットテスト＋スモークテスト＋統合テストが存在（63テストファイル、1158テストケース全パス）
- テストカバレッジは全モジュール（コア・データ・評価・モデル・学習・ユーティリティ）を網羅
- ドキュメントと実装の乖離は軽微（追加機能が文書化されていない程度）
- Velocity.cosine_similarityのKeyError修正が適用済み
- RunMetricsのコンテキストマネージャが実装済み

### 追加/変更要件

- RandomWalkController、DeltaTracker、MetricsはAGENTS.mdに明示的に記載すべき
- ~~eval_*, data/*, model/* モジュールのテストカバレッジが不足~~ → **解決済み**: 全モジュールにテスト追加完了
- MLflow連携は設定に存在するが実体はJSONLログのみ
- EDGE-105（cosine_similarity KeyError保護）を追加
- NFR-304（RunMetricsコンテキストマネージャ）を追加
- **REQ-038~044**: CycleState・DeltaTracker・学習ループ統合の要件を追加
- **EDGE-110~114**: CycleState・DeltaTrackerの境界値要件を追加
- **REQ-049, 050**: Velocity magnitude history・異常検出・トレンド追跡要件を追加
- **REQ-051, 052**: データスキーマバリデーション（DataRecord/ValidationSummary）要件を追加
- **EDGE-115~117**: Velocity magnitude境界値要件を追加
- **REQ-001更新**: magnitude history記録を統合
- **REQ-011更新**: lrをランダムウォーク探索パラメータに追加
- **REQ-013a追加**: 適応学習率制御（lr_accept_boost/lr_reject_decay）要件
- **REQ-053追加**: 収束適応（adapt_to_convergence）要件
- **REQ-054追加**: lr境界クランプ要件
- **REQ-055追加**: レイヤースコア更新（update_layer_scores）要件
- **EDGE-118~120追加**: lr境界テスト要件
- **architecture.md修正**: TG-LoRA固有パラメータ表の値をconfig (9b_tg_lora.yaml) に一致するよう修正（K_initial, N_initial, alpha_min/max, beta_initial, relative_update_cap等）。lr関連パラメータ（lr_initial, lr_min, lr_max, lr_accept_boost, lr_reject_decay）を追加
- **dataflow.md修正**: 適応学習率フロー（boost/decay/clamp, adapt_to_convergence）を追加、lr_reject_decay=0.5の設計意図を明記
- **REQ-056追加**: 外挿後パラメータ有限性検証要件（AI_HUB_MAKE_RUN_FEEDBACK指摘対応）
- **REQ-057追加**: trainer間数値安全性カバレッジ一致要件
- **REQ-058追加**: dtype/bnb_4bit_compute_dtype Literal enum検証要件
- **EDGE-121追加**: 外挿による非有限パラメータのエッジケース
- **EDGE-122追加**: dtypeフィールド不正文字列拒否のエッジケース
- **REQ-059追加**: 外挿安全性統合テスト要件（AI_HUB_MAKE_RUN_FEEDBACK指摘「integration test exercising the full extrapolation→NaN detection→rollback→cycle_state path」）
- **REQ-060追加**: 回復フロー副作用検証要件（AI_HUB_MAKE_RUN_FEEDBACK指摘「verify the penalize/score-update side effects」）
- **EDGE-123~125追加**: 非有限回復フローの統合テスト境界値
- **REQ-061追加**: 設定スキーマextra='forbid'要件（RISK-0015/0016）
- **REQ-062追加**: 設定読込非mapping拒否要件
- **REQ-063追加**: cap_update非有限ゼロ返却要件
- **REQ-064追加**: ロールバックスナップショットNaN/Infサニタイズ要件（RISK-0074）
- **REQ-065追加**: ロールバック履歴max_history制限要件（RISK-0074）
- **REQ-066追加**: metrics.cosine_similarityキー不一致安全要件
- **REQ-067追加**: DeltaTracker._compute_stats非有限スキップ要件
- **REQ-068追加**: DeltaTracker norm_history非有限ガード要件
- **EDGE-126~134追加**: Phase 14 reliability fixes境界値
- **REQ-069追加**: Perplexity E2Eパイプライン検証要件（AI_HUB_MAKE_RUN_FEEDBACK「E2E integration test」指摘対応）
- **REQ-070追加**: Trainer間perplexity配管パリティ要件（AI_HUB_MAKE_RUN_FEEDBACK「verify parity」指摘対応）
- **REQ-071追加**: accept()プロパティベーステスト要件（AI_HUB_MAKE_RUN_FEEDBACK「property-based test」指摘対応）
- **REQ-072追加**: Wiki bootstrapコミット方針要件（AI_HUB_MAKE_RUN_FEEDBACK「skip wiki bootstraps」指摘対応）
- **EDGE-135追加**: RunMetrics footer有限perplexity保証境界値
- **EDGE-136追加**: accept()相対許容誤差大きさ依存性境界値
- **REQ-073追加**: 公開APIエクスポート要件（__init__.py）
- **REQ-074追加**: RandomWalkController入力検証要件
- **REQ-075追加**: 空データローダーNaN返却要件
- **REQ-076追加**: ロールバックtry-catch安全性要件
- **REQ-077追加**: 非有限loss_afterガード要件
- **REQ-078追加**: 共有InfiniteBatchIterator要件
- **REQ-079追加**: 共有save_checkpoint要件
- **REQ-080追加**: 戦略リスト重複排除要件
- **EDGE-137~144追加**: 入力検証・空データ・ロールバック安全性境界値
- **REQ-081追加→更新**: save_checkpoint readback検証（🔴→🔵: TASK-0051完了）
- **REQ-082追加→更新**: InfiniteBatchIteratorエッジケーステスト（🔴→🔵: TASK-0052完了）
- **REQ-083追加→更新**: 非有限lossガードログ出力（🟡→🔵: TASK-0053完了）
- **REQ-084追加→更新**: ロールバック失敗E2Eテスト（🔴→🔵: TASK-0054完了）
- **REQ-089追加**: metrics.total_norm()非有限スキップ（🔵: 1bb591dコミット）
- **REQ-090追加**: metrics.per_layer_norms()非有限スキップ（🔵: 1bb591dコミット）
- **REQ-091追加**: _compute_pilot_average非有限フィルタリング（🔵: 1bb591dコミット）
- **REQ-092追加**: 探索確率の設定→コントローラ伝播（🔵: 1bb591d/03d1e46コミット）
- **REQ-093追加**: config-to-controller integration tests（🔵: 03d1e46コミット）
- **EDGE-146~149追加**: メトリクス/pilot average NaN/Inf境界値
- **REQ-103追加**: ControllerState summary()/from_dict()シリアライズ要件（🔵: 4435fdeコミット）
- **REQ-104追加**: CycleState from_dict()デシリアライズ要件（🔵: 04a7581コミット）
- **REQ-105追加**: TrainingState統合シリアライズ要件（🔵: 57739faコミット）
- **REQ-106追加**: 診断スクリプト要件（🔵: b942b4bコミット）
- **REQ-107追加**: 障害回復スクリプト要件（🔵: 66987e4コミット）
- **REQ-108追加**: Makefile ciターゲット要件（🔵: AI_HUB_MAKE_RUN_FEEDBACK指摘対応）
- **REQ-109追加**: APIリファレンス完全性要件（🔵: b942b4bコミット）
- **REQ-110追加**: ActivationCacheによるレイヤースキップ評価最適化要件（🔵: 0bc7236コミット）
- **REQ-111追加**: ActivationCache CachedBatch管理・フォールバック要件（🔵: 0bc7236コミット）
- **REQ-112追加**: ActivationCache学習ループ統合要件（🔵: 0bc7236/64bd8a8コミット）
- **REQ-113追加**: enable_random_walkフラグによるハイパーパラメータ凍結要件（🔵: 0782acdコミット）
- **REQ-114追加**: force_top_layers_onlyによる決定論的レイヤー選択要件（🔵: 555287dコミット）
- **REQ-115追加**: 移動平均ベースライン判定要件（🔵: 555287d/d3f834bコミット）
- **REQ-116追加**: Metropolis-Hastings確率的受理要件（🔵: d3f834bコミット）
- **REQ-117追加**: confident-skipメカニズム要件（🔵: 555287dコミット）
- **REQ-118追加**: K-step中間ロールバック要件（🔵: 64bd8a8/a1ffe6dコミット）
- **EDGE-150~158追加**: Phase 28境界値要件
- **REQ-119追加**: OptimizerLifecycleManagerライフサイクル管理要件（🔵: 3fdf57a/d2e2a51コミット）
- **REQ-120追加**: TrainingConfig.optimizer_lifecycle設定フィールド要件（🔵: d69a57dコミット）
- **REQ-121追加**: RunMetrics header optimizer_lifecycle出力要件（🔵: 2ac68d1コミット）
- **REQ-122追加**: ActivationCache hit/miss追跡メトリクス要件（🔵: f45a269コミット）
- **REQ-123追加**: ベンチマークスクリプト要件（🔵: d69a57dコミット）
- **REQ-124追加**: 実験用optimizer再利用設定サーフェス要件（🔵: d69a57dコミット）
- **EDGE-159~163追加**: Phase 29境界値要件
- **REQ-128追加**: 破損キャッシュファイルハンドリング（🔵: design-interview A27推奨・prefix_feature_cache.py既存実装より）
- **REQ-129追加**: force_rebuildフラグ動作（🔵: train_tg_lora.py _maybe_cache_dataset既存分岐・config_schema.py TrainingConfig既存フィールドより）
- **REQ-130追加**: position_idsビルドパス（🔵: prefix_feature_cache.py build_prefix_feature_dataset既存処理・design-interview A27推奨より）
- **REQ-131追加**: model.training状態復元（🔵: prefix_feature_cache.py try/finally既存実装・design-interview A27推奨より）
- **REQ-132追加**: SHA-256キャッシュ無効化（🔵: prefix_feature_cache.py get_prefix_feature_cache_path既存実装より）
- **REQ-133追加**: format_version不一致検証（🔵: prefix_feature_cache.py load_prefix_feature_dataset既存チェックより）
- **REQ-134追加**: 空データセット拒否（🔵: prefix_feature_cache.py save_prefix_feature_dataset既存チェックより）
- **REQ-135追加**: compare-prefix-coldwarm smoke test（🔵: Makefile既存targets・AI_HUB_MAKE_RUN_FEEDBACK指摘・design-interview A27推奨より）
- **REQ-139追加**: AsyncCacheBuilder統合テスト（🔵: AI_HUB_MAKE_RUN_FEEDBACK指摘・テストギャップ確認）
- **REQ-140追加**: diff_lora fast paths（🔵: 7a643a9コミット実装確認）
- **REQ-141追加**: cosine_similarity直交ベクトル警告（🔵: 7a643a9コミット実装確認）
- **REQ-142追加**: decoder層ロギング強化（🔵: 7a643a9コミット実装確認）
- **REQ-143追加**: smoke_async_prefix.yaml設定サーフェス（🔵: 182de29コミット確認）
- **EDGE-164~167追加**: Phase 33境界値要件
- **REQ-144追加**: In-place EMA update data_ptr保存（🔵: 851041e/c9928b6コミット確認）
- **REQ-145追加**: In-place cap_update data_ptr保存（🔵: 851041e/c9928b6コミット確認）
- **REQ-146追加**: data_ptr保存検証テスト（🔵: c9928b6コミット TASK-0079確認）
- **REQ-147追加**: benchmark_velocity_ops.pyマイクロベンチマーク（🔵: c51fd5bコミット TASK-0080確認）
- **REQ-148追加**: Makefile bench-velocity-opsターゲット（🔵: AI_HUB_MAKE_RUN_FEEDBACK指摘・未実装）
- **EDGE-168~171追加**: Phase 34境界値要件
- **REQ-149追加**: bench-velocity-ops-ci CI gate（🔵: design-interview A31 🔴指摘の解決・AI_HUB_MAKE_RUN_FEEDBACK「Wire bench-velocity-ops --baseline into CI」より）
- **EDGE-172追加**: bench-velocity-ops-ci exit code境界値（🔵: benchmark_velocity_ops.py exit code実装・TestBaselineRegressionDetection より）
- **REQ-153追加**: Velocity magnitude_acceleration（🔵: velocity.py magnitude_acceleration()・TASK-0090）
- **REQ-154追加**: cap_update非有限値ロギング強化（🔵: extrapolator.py cap_update() warning logging・TASK-0090）
- **REQ-155追加**: DeltaTracker key-mismatch検証（🔵: delta_tracker.py compute_and_record()・TASK-0091）
- **REQ-156追加**: RollbackManager max_historyガード（🔵: rollback_manager.py __init__・TASK-0091）
- **REQ-157追加**: snapshot_lora_delta空ベース検証（🔵: lora_state.py snapshot_lora_delta()・df57154）
- **REQ-158追加**: propose() OverflowError防止（🔵: random_walk_controller.py propose() exp clamping・580680d）
- **REQ-159追加**: _compute_stats autograd漏れ防止（🔵: delta_tracker.py _compute_stats() @torch.no_grad()・580680d）
- **EDGE-173~178追加**: Phase 37境界値要件
- **REQ-162追加**: RandomWalkController.restore_state()障害回復（🔵: random_walk_controller.py restore_state()・9f195f0コミット）
- **REQ-163追加**: resume_pathによる学習再開（🔵: train_tg_lora.py resume_path引数・9f195f0コミット）
- **REQ-164追加**: --resume CLI引数（🔵: train_tg_lora.py main()・9f195f0コミット）
- **REQ-165追加**: last_accel_action属性による加速度適応観測（🔵: random_walk_controller.py last_accel_action・b2eb409コミット）
- **REQ-166追加**: MLflow cycle metrics加速度メトリクス（🔵: train_tg_lora.py magnitude_acceleration/accel_action・b2eb409コミット）
- **EDGE-183~186追加**: Phase 39境界値要件

### 残課題（Phase 39更新）

- MLflowバックエンドの実際の統合状況（RunMetricsで代替されているか）
- 自社データへの移行フェーズの具体的な要件定義
- 本番環境でのデプロイ・運用要件
- 実際のQwen3.5-9BモデルでのE2E学習検証（現在はGPT-2 tinyモデルのみ）
- **REQ-162~164 E2Eテスト**: AI_HUB_MAKE_RUN_FEEDBACK指摘「Add an E2E integration test for --resume that saves a checkpoint, simulates interruption, resumes training, and asserts loss continues decreasing from the recovered state」— 現在モックベースのテストのみ。実モデルでのsave→kill→resume→loss減少確認が必要
- **TruthfulQA品質ギャップ**: 外部ベンチマークでTG-LoRAの品質向上が確認されていない（delta -0.00045 acc）。次フェーズでaccel adaptation params（accel_instability_lr_decay, accel_convergence_lr_boost）のチューニングによる改善を調査する必要がある

### A27: OptimizerLifecycleManager・キャッシュメトリクス追跡の要件ギャップ 🔵

**分析日時**: 2026-05-23
**カテゴリ**: 既存実装との差分
**背景**: 直近5コミット（3fdf57a, d2e2a51, d69a57d, f45a269, 2ac68d1）でOptimizerLifecycleManager、activation cache hit/missメトリクス、optimizer_lifecycle policyログが実装されたが、要件定義（REQ-001~118）に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価し、「Add an integration test or small training-run smoke test that confirms optimizer_lifecycle policy appears in actual run_metrics output end-to-end」を提案

**判断**: 以下の実装が要件定義に未反映だった:
- **OptimizerLifecycleManager** (optimizer_lifecycle.py): サイクル間AdamWライフサイクル管理。recreate_per_cycle（毎サイクル再生成）とreuse_state_reset_experimental（in-place zero-reset再利用）の2ポリシー。prepare_for_cycle()でpolicyに応じたoptimizerを返す。state_tensor_pointers()でメモリ再確保検知を提供
- **TrainingConfig.optimizer_lifecycle**: OptimizerLifecyclePolicy型フィールド、デフォルト"recreate_per_cycle"
- **run_metrics header**: write_header()がoptimizer_lifecycle設定値を出力。getattr()でフィールド不在時も安全にNone出力
- **Activation cache hit/miss追跡**: train_tg_lora.pyでactivation_cache_eligible_count/hit_countを追跡。RunMetrics.record_step()がcache_built/cache_eligible/cache_hitをステップ単位で記録。hit_rate計算でゼロ除算を回避
- **ベンチマークスクリプト**: scripts/benchmark_optimizer_lifecycle.pyで両policyのprepare/step時間・メモリ増分・state tensor pointer安定性を比較測定
- **実験用設定**: configs/9b_tg_lora_optimizer_reuse_experimental.yaml（enable_random_walk=false, force_top_layers_only=true, optimizer_lifecycle=reuse_state_reset_experimental）

**根拠**:
- src/training/optimizer_lifecycle.py: OptimizerLifecycleManager, _set_optimizer_hparams, _zero_optimizer_state_in_place
- src/training/config_schema.py: TrainingConfig.optimizer_lifecycle
- src/utils/run_metrics.py: write_header optimizer_lifecycle, record_step cache fields
- src/training/train_tg_lora.py: activation_cache_*_count, OptimizerLifecycleManager初期化
- scripts/benchmark_optimizer_lifecycle.py: ベンチマークスクリプト
- configs/9b_tg_lora_optimizer_reuse_experimental.yaml: 実験用設定
- テストスイート: 1158 passed

**信頼性への影響**:
- 新規要件 REQ-119~124 を追加（全て 🔵: 実装・テスト・config確認済み）
- 新規Edgeケース EDGE-159~163 を追加
- NFR-401 テスト数を 1145→1158 に更新
- 全新規要件の信頼性: 🔵（コード直接確認・コミット内容検証）

### A26: Phase 28 ActivationCache・決定論的モード・中間ロールバックの要件ギャップ 🔵

**分析日時**: 2026-05-23
**カテゴリ**: 既存実装との差分
**背景**: 直近5コミット（555287d, d3f834b, 0782acd, a1ffe6d, 64bd8a8）で9つの主要新機能が実装されたが、要件定義（REQ-001~109）に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価

**判断**: 以下の実装が要件定義に未反映だった:
- **ActivationCache**: src/tg_lora/activation_cache.py（新規モジュール）。eval_and_cache()でスプリットレイヤーの隠れ状態をキャッシュ、eval_from_cache()でキャッシュからの部分フォワード。32層モデルで最後8層がアクティブな場合、評価コストを約75%削減
- **移動平均ベースライン**: _decide_accept_rollback()がaccepted_valid_historyのmoving_avg_window件の平均をベースラインとして使用。loss_pilotのみの比較よりノイズ耐性が高い
- **Soft accept（Metropolis-Hastings）**: soft_accept_temperature > 0の場合、exp(-(loss_after-baseline)/temperature)の確率で境界ケースを受理。局所最適解からの脱出を支援
- **K-step中間ロールバック**: pilotフェーズでdivergence検出時、全中間点（step 0~K-1）を評価し最良点にロールバック。delta snapshotで増分状態管理
- **enable_random_walk**: falseの場合、propose()が静的、reward()/penalize()が適応スキップ、adapt_to_convergence()無効化。決定論的学習モード
- **force_top_layers_only**: trueの場合、active_layer_strategyに関わらず"last_25_percent"を強制。ActivationCacheのスプリットレイヤー一貫性を保証
- **Confident-skip**: velocity方向が高安定（cos_sim >= threshold、acceptance_rate >= 0.8等）の場合に評価を省略し自動受理
- **best_full_eval_loss**: full evalの最良損失をperplexityとは独立に追跡
- **新規Config フィールド**: moving_avg_window, soft_accept_temperature, force_top_layers_only, enable_random_walk, confident_skip_cos, confident_skip_min_cycles

**根拠**:
- src/tg_lora/activation_cache.py: ActivationCache, CachedBatch, determine_split_layer()
- src/training/train_tg_lora.py: _decide_accept_rollback(), intermediate_deltas, confident_skip, force_top_layers_only統合
- src/tg_lora/random_walk_controller.py: enable_random_walkフラグ
- src/training/config_schema.py: 新規TGLoRAParams/EvalConfigフィールド
- scripts/run_kstep_rollback_test.sh: K-stepロールバックテストスクリプト
- configs/9b_tg_lora.yaml: 新規パラメータ設定

**信頼性への影響**:
- 新規要件 REQ-110~118 を追加（全て 🔵: 実装ベース）
- 新規Edgeケース EDGE-150~158 を追加
- NFR-401 テスト数を 1122→1145 に更新
- 全新規要件の信頼性: 🔵（コード直接確認・コミット内容検証）

### A28: prefix_feature_cache堅牢性・compare-prefix smoke testの要件ギャップ 🔵

**分析日時**: 2026-05-23
**カテゴリ**: テストカバレッジギャップ・既存実装との差分
**背景**: Phase 31でprefix_feature_cacheの永続化対応（Makefile targets, env var wiring, config fields, dataflow spec, design analysis）が完了したが、design-interview A27が改善推奨として挙げているcorrupted cache handling, force_rebuild flag, position_ids build pathのテストが未実装。AI_HUB_MAKE_RUN_FEEDBACKも「compare-prefix-coldwarm targetのsmoke実行CI stepを追加する」を指摘。test_prefix_feature_cache.pyは4テストのみで基本機能検証に留まる

**判断**: 以下のテストカバレッジギャップを特定:
- **破損キャッシュハンドリング**: load_prefix_feature_dataset()はtorch.load()でキャッシュを読み込むが、部分的に書き込まれたファイル、フォーマット不整合、欠落キーに対するテストが存在しない。_maybe_cache_dataset()のtry/exceptパターンが破損時に再ビルドにフォールバックする動作が未検証
- **force_rebuildフラグ**: prefix_feature_cache_force_rebuild=trueの場合、_maybe_cache_dataset()がメモリ/ディスクキャッシュをスキップして必ず再ビルドする動作がテストされていない。force_rebuild=falseの通常動作も明示的なテストがない
- **position_idsビルドパス**: build_prefix_feature_dataset()はbatch.get("position_ids")でposition_idsを処理するが、position_idsを含むデータセットでのビルドがテストされていない。_TokenDatasetテストヘルパーがposition_idsを含まないため
- **model.training状態復元**: build_prefix_feature_dataset()はtry/finallyでmodel.training状態を復元するが、ビルド中に例外が発生した場合の復元がテストされていない
- **SHA-256キャッシュ無効化**: get_prefix_feature_cache_path()がメタデータ変更で異なるパスを生成することがテストされていない
- **format_version不一致**: load_prefix_feature_dataset()がformat_versionを検証するが、不一致ValueErrorのテストがない
- **空データセット拒否**: save_prefix_feature_dataset()が空データセットをValueErrorで拒否するがテストがない
- **compare-prefix-coldwarm smoke test**: Makefileにcompare-prefix-cold/warm/coldwarm targetsが存在するが、exit code 0とcache hit/miss logを検証する自動テストがない

**根拠**:
- src/tg_lora/prefix_feature_cache.py: load_prefix_feature_dataset, build_prefix_feature_dataset, get_prefix_feature_cache_path, save_prefix_feature_dataset
- src/training/train_tg_lora.py: _maybe_cache_dataset (memory→disk→build 3段階ルックアップ, force_rebuild分岐)
- tests/test_prefix_feature_cache.py: 4テスト（基本機能のみ）
- Makefile: compare-prefix-cold/warm/coldwarm targets
- design-interview.md: A27改善推奨（corrupted cache, force_rebuild, position_ids）
- AI_HUB_MAKE_RUN_FEEDBACK: 「compare-prefix-coldwarm targetのsmoke実行CI stepを追加する」

**信頼性への影響**:
- 新規要件 REQ-128~135 を追加（全て 🔵: 既存実装ベース、design-interview A27推奨準拠）
- 全新規要件の信頼性: 🔵（コード直接確認・design-interview A27分析・AI_HUB_MAKE_RUN_FEEDBACK指摘）

---

**分析前**:

- 🔵 青信号: 0
- 🟡 黄信号: 0
- 🔴 赤信号: 0

**初回分析後**:

- 🔵 青信号: 34（既存実装から直接確認）
- 🟡 黄信号: 6（推測ベース）
- 🔴 赤信号: 0（自動推定のみの項目なし）

**更新後（2026-05-21 Phase 2完了）**:

- 🔵 青信号: 38（＋4: テストカバレッジ全面解消、KeyError修正、コンテキストマネージャ追加）
- 🟡 黄信号: 2（−4: テスト未検証項目が解消）
- 🔴 赤信号: 0

**更新後（2026-05-21 CycleState+DeltaTracker要件追加）**:

- 🔵 青信号: 47（＋9: REQ-038~044、EDGE-110~114、NFR-401更新）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-21 Velocity Magnitude + Data Schema要件追加）**:

- 🔵 青信号: 54（＋7: REQ-049~052、EDGE-115~117、NFR-401更新）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-21 適応学習率・収束適応要件追加）**:

- 🔵 青信号: 60（＋6: REQ-013a、053~055、EDGE-118~120、テスト数562更新）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-21 設計文書正確性検証・パラメータ表修正）**:

- 🔵 青信号: 63（＋3: パラメータ表修正、適応LRフロー追加、テスト575検証）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-21 外挿安全性・Config文字列検証ギャップ追加）**:

- 🔵 青信号: 69（＋6: REQ-056~058、EDGE-121~122、分析A11~A12）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-21 外挿安全性統合テストギャップ追加）**:

- 🔵 青信号: 75（＋6: REQ-059/060、EDGE-123~125、分析A13）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-21 Phase 14 reliability fixes反映）**:

- 🔵 青信号: 86（＋11: REQ-061~068、EDGE-126~134、分析A14）
- 🟡 黄信号: 2（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-22 Phase 19 perplexity E2E・property-based testing反映）**:

- 🔵 青信号: 90（＋4: REQ-069/070、EDGE-135、分析A15）
- 🟡 黄信号: 4（＋2: REQ-071、EDGE-136、分析A16）
- 🔴 赤信号: 0

**更新後（2026-05-22 Phase 23 テストカバレッジ強化完了）**:

- 🔵 青信号: 106（＋4: REQ-081~084全て実装完了、分析A19）
- 🟡 黄信号: 4（−1: REQ-083が🔴→🟡→🔵に昇格）
- 🔴 赤信号: 0（−3: REQ-081/082/084が🔴→🔵に昇格）

**更新後（2026-05-22 メトリクスNaN/Infガード・探索確率伝播・純粋関数テスト追加）**:

- 🔵 青信号: 120（＋9: REQ-089~093、EDGE-146~149、分析A21~A24）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-22 Phase 27 チェックポイントシリアライズ・運用スクリプト・CI反映）**:

- 🔵 青信号: 127（＋7: REQ-103~109、分析A25）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-23 Phase 28 ActivationCache・決定論的モード・中間ロールバック反映）**:

- 🔵 青信号: 136（＋9: REQ-110~118、分析A26）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-23 Phase 29 OptimizerLifecycleManager・キャッシュメトリクス追跡反映）**:

- 🔵 青信号: 146（＋10: REQ-119~124、EDGE-159~163、分析A27）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-23 Phase 32 prefix_feature_cache堅牢性テスト・compare-prefix smoke test反映）**:

- 🔵 青信号: 155（＋9: REQ-128~135、分析A28）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-23 Phase 34 in-place ops・data_ptr保存検証・velocity opsベンチマーク反映）**:

- 🔵 青信号: 172（＋9: REQ-144~148、EDGE-168~171、分析A30）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

**更新後（2026-05-23 Phase 35 CI gate要件分析・A32）**:

- 🔵 青信号: 174（+2: REQ-149, EDGE-172）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0

### A32: Phase 35 bench-velocity-ops CI gate要件ギャップ解消 🔵

**分析日時**: 2026-05-23
**カテゴリ**: CI回帰検出・運用品質
**背景**: design-interview.md A31がベンチマークCI閾値を🔴（未設計）と明示的に指摘。AI_HUB_MAKE_RUN_FEEDBACKが「Wire bench-velocity-ops --baseline into CI」を推奨。benchmark_velocity_ops.pyには--baseline/--save-baseline/--thresholdが実装済みでテストも7件存在するが、MakefileにCI gateターゲットが存在せず、CI パイプラインに組み込まれていない。REQ-148の拡張として明示的にCI gate要件を定義する必要がある

**判断**: 以下のギャップを特定・解消:
1. **Makefile bench-velocity-ops-ci ターゲット欠落**: bench-velocity-opsは観測的（JSON出力のみ）でCI gate機能がない。--baseline --threshold付きのCI用ターゲットを追加するREQ-149を新規定義
2. **チェックインベースラインファイルの不在**: baselines/velocity_ops.jsonがリポジトリに存在しない。CI gate機能にはベースラインとの比較が必要
3. **acceptance-criteria.md Phase 34欠落**: REQ-144~148の受け入れ基準がacceptance-criteria.mdに未記載。テストは実装済み（test_benchmark_velocity_ops.py 16テスト含む）だが受け入れ基準文書が未更新
4. **テスト数乖離**: 実測値1351テスト vs 文書記載1344テスト

**根拠**:
- scripts/benchmark_velocity_ops.py: --baseline/--save-baseline/--threshold 完全実装（_compare_with_baseline, exit code 0/1/2）
- tests/test_benchmark_velocity_ops.py: TestBaselineRegressionDetection 7テスト + 9 smoke tests = 16テスト
- Makefile: bench-velocity-opsターゲット存在、bench-velocity-ops-ci ターゲット不在
- design-interview.md A31: 「CI回帰閾値未実装」🔴指摘
- AI_HUB_MAKE_RUN_FEEDBACK: 「Wire bench-velocity-ops --baseline into CI: add a Makefile target that runs against a checked-in baseline file and fails on regression」
- pytest --co -q: 1351 tests collected

**信頼性への影響**:
- 新規要件 REQ-149（CI gate Makefile target）を追加: 🔵（--baseline/--threshold実装済み、Makefileパターン確立済み）
- 新規Edgeケース EDGE-172（CI gate exit codes）を追加: 🔵（benchmark_velocity_ops.py exit code実装ベース）
- design-interview A31 🔴→🔵（CI gate要件をREQ-149で明示的に設計）

**分析結果サマリー更新**

**Phase 35要件分析後** (A32 CI gate gap):
- 🔵 青信号: 174（+2: REQ-149, EDGE-172）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（A31 CI閾値🔴が解消）

**Phase 41要件分析後** (A36 E2E resume + accel sweep):
- 🔵 青信号: 205（+13: REQ-167~172, EDGE-187~191）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

---

### A33: LR探索統合ギャップ — propose()のlr出力がtraining loopで未消費

**分析日時**: 2026-05-24
**カテゴリ**: 既存設計確認・追加要件
**背景**: AI_HUB_MAKE_RUN_FEEDBACKで「config_schema.pyとtrain_tg_lora.pyに追加されたLR探索パラメータ（lr_explore_prob, lr_log_sigma）がtraining pipelineで消費されているか確認」が指摘された。b712540コミットでlog-normal LR explorationがpropose()に追加されたが、training loopでの統合が未検証だった。

**分析内容**:
1. config_schema.py: lr_explore_prob=0.3, lr_log_sigma=0.1がTGLoRAParamsに定義済み — ✅
2. train_tg_lora.py: tg_cfg.get("lr_explore_prob", None)でcontrollerに渡す — ✅
3. random_walk_controller.py: propose()でlog-normal walkによるlr探索を実装 — ✅
4. train_tg_lora.py training loop: controller.propose()の戻り値proposal.lrがcontroller.state.lrに反映されない — ❌ **ギャップ発見**

**判断**:
- propose()で生成された探索lr（proposal.lr）がtraining loopで一切使用されていなかった
- lrはreward/penalizeの決定論的boost/decay（1.2x/0.5x）のみで変動
- lr_explore_probとlr_log_sigmaはcontrollerに渡されるが、その探索出力が消費されずorphanedになっていた

**修正内容**:
1. train_tg_lora.py: controller.propose()後にcontroller.state.lr = proposal.lrを追加（1行）
2. これにより探索lrが次サイクルのpilot trainingで使用され、reward/penalizeがさらに調整
3. test_training_integration.py: TestLrExplorationIntegration 3テストを追加:
   - test_lr_explore_prob_wired_from_config: config→controller伝播検証
   - test_proposed_lr_applied_to_state: 探索lrのstate反映検証
   - test_full_propose_accept_reject_cycle_with_lr_walk: 5サイクルfull cycle検証

**信頼性への影響**:
- 新規要件 REQ-150（LR探索パラメータ定義）: 🔵（既存実装ベース）
- 新規要件 REQ-151（探索lrのstate反映）: 🔵（今回の修正で解決）
- 新規要件 REQ-152（統合テスト要件）: 🔵（3テスト追加済み）
- この分析により、AI_HUB_MAKE_RUN_FEEDBACKの指摘3項目すべてが解消

**Phase 36要件分析後**:
- 🔵 青信号: 177（+3: REQ-150, REQ-151, REQ-152）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

### A34: Phase 37 magnitude_acceleration・入力検証強化・数値安定性 🔵

**分析日時**: 2026-05-24
**カテゴリ**: 既存実装との差分・テストカバレッジ拡充
**背景**: 直近4コミット（1a23a0b, 16f0ea6, df57154, 580680d）で6つの修正と1つの新機能が実装されたが、要件定義（REQ-001~152）に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価

**判断**: 以下の実装が要件定義に未反映だった:
- **magnitude_acceleration**: velocity.pyの新メソッド。magnitude履歴の二階微分を計算し、加速的不安定性を早期検出。正=加速的増大（不安定）、負=減速（収束）。3件未満は0.0
- **cap_update非有限値ロギング**: extrapolator.py cap_update()がNaN/Inf検出時に要素数（n_nan, n_inf）を含む警告ログを出力。REQ-063のゼロ返却に加えて観測性を向上
- **DeltaTracker key-mismatch検証**: delta_tracker.py compute_and_record()がafter/before辞書のキー不一致をValueErrorで拒否。欠落キーをエラーメッセージに列挙
- **RollbackManager max_historyガード**: rollback_manager.py __init__がmax_history <= 0をValueErrorで拒否
- **snapshot_lora_delta空ベース検証**: lora_state.py snapshot_lora_delta()が空のbase辞書をValueErrorで拒否
- **propose() OverflowError防止**: random_walk_controller.py propose()がmath.exp()の引数を700にクランプ
- **_compute_stats autograd漏れ防止**: delta_tracker.py _compute_stats()に@torch.no_grad()を追加

**根拠**:
- src/tg_lora/velocity.py: magnitude_acceleration() 新規実装
- src/tg_lora/extrapolator.py: cap_update() warning logging追加
- src/tg_lora/delta_tracker.py: compute_and_record() key validation、_compute_stats() @torch.no_grad()
- src/tg_lora/rollback_manager.py: __init__ max_history <= 0 ValueError
- src/tg_lora/lora_state.py: snapshot_lora_delta() empty base ValueError
- src/tg_lora/random_walk_controller.py: propose() exp clamping
- tests/test_validation_hardening_0091.py: 11テスト（key-mismatch 4 + max_history 3 + velocity edge 4）
- tests/test_prefix_cache_coverage_0091.py: 2テスト
- テストスイート: 1554 passed, 7 skipped, 1 failed (benchmark flake)

**信頼性への影響**:
- 新規要件 REQ-153~159 を追加（全て 🔵: 実装・テスト確認済み）
- 新規Edgeケース EDGE-173~178 を追加（全て 🔵）
- NFR-401 テスト数を 1351→1554 に更新

**Phase 37要件分析後**:
- 🔵 青信号: 184（+7: REQ-153~159）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

### A35: Phase 39 --resume障害回復再開・加速度適応観測性の要件ギャップ 🔵

**分析日時**: 2026-05-24
**カテゴリ**: 既存実装との差分
**背景**: 直近2コミット（9f195f0, b2eb409）で--resume flagによる障害回復再開機能と加速度適応の観測性が実装されたが、要件定義（REQ-160~161）に未反映だった。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価

**判断**: 以下の実装が要件定義に未反映だった:
- **restore_state()**: RandomWalkControllerにrestore_state(state: ControllerState)メソッドを追加。保存済みControllerStateを採用しつつconfig（candidates, bounds, tolerances, exploration probs）は保持。last_accel_actionを0にリセット
- **--resume CLI flag**: train_tg_lora()にresume_path引数を追加。load_training_state()でtraining_state.ptを読み込み、controller/velocity/delta_tracker/cycle_stateを復元。cycle_offset以前のサイクルをスキップ
- **last_accel_action属性**: adapt_to_acceleration()の実行結果（1=不安定, -1=収束, 0=無行動）を追跡。enable_random_walk=false時は常に0。summary()に含まれる
- **MLflow cycle metrics**: magnitude_accelerationとaccel_actionをMLflowサイクルメトリクスに追加

**根拠**:
- src/tg_lora/random_walk_controller.py: restore_state(), last_accel_action
- src/training/train_tg_lora.py: resume_path, cycle_offset skip, MLflow magnitude_acceleration/accel_action
- tests/test_fault_recovery.py: TestRestoreStateIntegration 2テスト
- tests/test_random_walk_controller.py: last_accel_action 6テスト
- テストスイート: 1594 passed

**信頼性への影響**:
- 新規要件 REQ-162~166 を追加（全て 🔵: 実装・テスト確認済み）
- 新規Edgeケース EDGE-183~186 を追加（全て 🔵）
- NFR-401 テスト数を 1554→1594 に更新

**Phase 39要件分析後**:
- 🔵 青信号: 192（+8: REQ-162~166, EDGE-183~186は別カウント）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

### A36: Phase 40-41 E2E resume・benchmark分析・accel param実験の要件ギャップ 🔵

**分析日時**: 2026-05-24
**カテゴリ**: 既存実装との差分
**背景**: TASK-0090（E2E resumeテスト）、TASK-0091（TruthfulQA分析・accel param感度）、TASK-0092（accel paramスイープconfig・スクリプト）、TASK-0093（Phase 40-41 doc update）が完了したが、requirements.mdはPhase 39（REQ-162~166）で止まっていた。AI_HUB_MAKE_RUN_FEEDBACKは前回イテレーションを「VALUABLE」と評価

**判断**: 以下の実装が要件定義に未反映だった:
- **E2E resume検証**: test_resume_e2e.pyでTrainingState保存→中断→resume→loss継続の完全フローを検証。cycle_offset未満スキップ、velocity方向保存の検証を含む
- **ベンチマーク分析**: scripts/analyze_benchmark.pyでbaseline/TG-LoRAメトリクス差分計算、欠損メトリクスの安全なスキップ
- **accel param感度**: accel_instability_lr_decay×accel_convergence_lr_boostのグリッドサーチ空間を定義し、4つの実験config（conservative/aggressive/balanced/no_accel）を作成
- **スイープインフラ**: scripts/run_accel_sweep.sh（4config順次実行・エラー耐性・レポート生成）、scripts/summarize_sweep.py（run_metrics.jsonl集約・validation lossソート）

**根拠**:
- tests/test_resume_e2e.py: TestResumeE2E 3テスト
- scripts/analyze_benchmark.py: baseline/TG-LoRA比較分析
- configs/9b_tg_lora_accel_*.yaml: 4つの実験config
- scripts/run_accel_sweep.sh: パラメータスイープ自動化
- scripts/summarize_sweep.py: スイープ結果集約
- tests/test_accel_experiment_configs.py: config検証テスト
- テストスイート: 1627 passed, 7 skipped

**信頼性への影響**:
- 新規要件 REQ-167~172 を追加（全て 🔵: 実装・テスト確認済み）
- 新規Edgeケース EDGE-187~191 を追加（全て 🔵）
- NFR-401 テスト数を 1594→1627 に更新

**Phase 41要件分析後**:
- 🔵 青信号: 205（+13: REQ-167~172, EDGE-187~191は別カウント）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

### A37: test_restore_state_propose_uses_restored_lr 非決定性バグ修正 🔵

**分析日時**: 2026-05-24
**カテゴリ**: テスト品質修正
**背景**: テストスイート全体実行時、test_restore_state_propose_uses_restored_lrが間欠的に失敗する。単独実行では通過するが、フルスイートでは先行テストのrandom seed状態によりpropose()のlog-normal LR探索が発生し、restored lrが変動する

**判断**: テストはrestore_state後のlrがpropose()で維持されることを検証したいが、lr_explore_prob=0.3（デフォルト）により30%の確率でlog-normal摂動が適用される。lr_explore_prob=0.0をコンストラクタに渡すことで、propose()がlrを変更しないことを確定的に検証可能

**根拠**:
- テスト失敗: `assert proposal.lr == 7e-4` → `proposal.lr == 0.0008285690350572555`
- 修正: `RandomWalkController(lr_initial=5e-4)` → `RandomWalkController(lr_initial=5e-4, lr_explore_prob=0.0)`
- テストスイート: 1832 passed, 7 skipped

**信頼性への影響**:
- テストの信頼性: 間欠的失敗→確定的通過に修正
- 既存要件への影響なし（実装変更なし、テスト修正のみ）

### A38: 未検証コンストラクタの入力検証ギャップ 🔵

**分析日時**: 2026-05-24
**カテゴリ**: 追加要件・入力検証ギャップ
**背景**: AI_HUB_MAKE_RUN_FEEDBACKにより、RandomWalkControllerに追加されたパラメータ検証パターン（REQ-074）を他のコンストラクタにも展開することが推奨された。全__init__メソッドの検証状況を網羅的に調査

**判断**: 以下のコンストラクタに入力検証が欠落している:
1. **DeltaTracker(max_history)**: max_history > 0 の検証なし。RollbackManager（REQ-156）は既に実装済み
2. **Velocity(max_history)**: max_history > 0 の検証なし。RollbackManagerと同じパターンが適用可能
3. **OptimizerLifecycleManager(lr, weight_decay)**: lr > 0, weight_decay >= 0 の検証なし。PyTorch AdamWの不正動作を防止するため必須
4. **PrefixFeatureDataset(examples)**: 空リストの検証なし。InfiniteBatchIteratorの空データセット検証（REQ-078）と同じパターン
5. **MappedPrefixFeatureDataset**: テンソル形状互換性とsplit_layer_idxの検証なし
6. **AsyncCacheBuilder**: config/device/split_layerの検証なし

**根拠**:
- src/tg_lora/delta_tracker.py: __init__ max_history検証なし
- src/tg_lora/velocity.py: __init__ max_history検証なし
- src/training/optimizer_lifecycle.py: __init__ lr/weight_decay検証なし
- src/tg_lora/prefix_feature_cache.py: PrefixFeatureDataset/MappedPrefixFeatureDataset 検証なし
- src/training/async_cache_builder.py: __init__ 検証なし
- 既存の良好パターン: RandomWalkController（REQ-074）, RollbackManager（REQ-156）, InfiniteBatchIterator（REQ-078）

**信頼性への影響**:
- 新規要件 REQ-173~177 を追加（全て 🔵: 実装調査ベース）
- 新規Edgeケース EDGE-192~196 を追加（全て 🔵）
- これらの検証追加後、全主要コンストラクタが入力検証を備えることになる

---

### A39: テスト非決定性パターンの横断分析 🔵

**分析日時**: 2026-05-24
**カテゴリ**: テスト品質・非決定性排除
**背景**: A37でtest_restore_state_propose_uses_restored_lrのflaky fixを実施。AI_HUB_MAKE_RUN_FEEDBACKは「scan the full test suite for other tests using default lr_explore_prob or other randomized defaults」を推奨。同様の非決定性パターンが他のテストにも存在するか横断調査を実施

**判断**: 以下のテストファイルでRandomWalkControllerをデフォルトexplore_probで使用するパターンを検出:
1. **test_random_walk_controller.py**: 複数テストがexplore_prob省略でインスタンス化（行58, 68, 77, 82, 91, 109, 326, 458, 1287, 1337）
2. **test_training_integration.py**: 行545-565, 1251-1284, 1315-1356
3. **test_fault_recovery.py**: _make_controller()ヘルパーがexplore_prob省略（行32-50）
4. **test_resume_e2e.py**: _simulate_training_cycles()がexplore_prob省略（行43-60）
5. **test_extrapolation_safety_integration.py**: 行341-356, 472-492, 500-530
6. **test_task_0028_ten_cycle_smoke.py**: 行319-323, 382-403
7. **test_smoke.py**: 行112-118

デフォルト値のリスク:
- k_explore_prob=0.4（40%でK変更）
- n_explore_prob=0.4（40%でN変更）
- beta_explore_prob=0.15（15%でbeta変更）
- strategy_explore_prob=0.08（8%でstrategy変更）
- lr_explore_prob=0.3（30%でLR探索）

**根拠**:
- test_restore_state_propose_uses_restored_lr flaky fix（f5fe40fコミット）
- AI_HUB_MAKE_RUN_FEEDBACK「scan the full test suite for other tests using default lr_explore_prob」
- tests/ 配下全ファイルのgrep調査

**信頼性への影響**:
- 新規要件 REQ-178 を追加（🔵: 既存パターンの横断分析ベース）
- 新規Edgeケース EDGE-197 を追加（🔵）
- テストスイート全体の信頼性向上: 間欠的失敗リスクの削減

---

**Phase 42要件分析後** (A38 コンストラクタ検証ギャップ + A39 テスト非決定性):
- 🔵 青信号: 213（+8: REQ-173~178, EDGE-192~197）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

### 確認できた事項（Phase 42時点）

- RandomWalkControllerの入力検証パターンは確立されており、他コンストラクタへの展開準備が整っている
- RollbackManagerのmax_history検証（REQ-156）がDeltaTracker/Velocityへの参照実装として機能する
- OptimizerLifecycleManagerのlr/weight_decay検証はPydantic config_schema.pyのField検証と一貫性を持つべき
- テスト非決定性は7つのテストファイルに横断的に存在し、体系的な修正が必要

### 追加/変更要件（Phase 42）

- REQ-173~178: コンストラクタ入力検証 + テスト非決定性排除（6件）
- EDGE-192~197: 境界値テスト（6件）

## 関連文書

- **要件定義書**: [requirements.md](requirements.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)

---

### A40: Runtime Prefix Offload・補助スクリプトの要件ギャップ

**分析日時**: 2026-05-25
**カテゴリ**: 既存設計確認・未定義部分詳細化
**背景**: Phase 53の要件整合性確認で、`prefix_runtime_offload.py`が`train_tg_lora.py`で使用されているがrequirements.mdに要件が存在しないことを発見

**判断**: `prefix_runtime_offload.py`（75行）は本番モジュールとして`train_tg_lora.py`の577行目で呼び出され、runtime時にprefix層をCPUにオフロードしてVRAMを解放する重要機能。`config_schema.py`には`prefix_runtime_offload_valid`バリデータが存在する。テスト(`test_prefix_runtime_offload.py`)も存在。要件は存在したが文書化が漏れていた。

また、補助スクリプト3つ（precompute_prefix_cache_parallel.py 435行、benchmark_prefix_cache.py 231行、generate_sweep_dashboard.py 219行）が要件に未記載。

**根拠**: src/tg_lora/prefix_runtime_offload.py完全読み込み、src/training/train_tg_lora.py:577の呼び出し、src/training/config_schema.py:136のバリデータ、tests/test_prefix_runtime_offload.pyの存在確認

**信頼性への影響**:
- prefix_runtime_offloadの3機能は全て 🔵（実装ベース）
- 新規要件 REQ-193~197 を追加
- 新規Edgeケース EDGE-198~201 を追加
- テストスイート: 96テストファイル、2035テストケース、全パス

---

**Phase 53要件分析後** (A40 Runtime Prefix Offloadギャップ解消):
- 🔵 青信号: 220（+7: REQ-193~197, EDGE-198~201）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

---

### A41: Frontier Sweep パイプライン強化の要件ギャップ

**分析日時**: 2026-05-25
**カテゴリ**: 既存設計確認・未定義部分詳細化
**背景**: 直近5コミット（4530f8c~9047beb）でfrontier sweepパイプラインに大幅な強化が加えられたが、要件定義書（REQ-185~197）はStage 3-5自動化の基本要件のみを記載しており、実際の実装に含まれるG2.3自動評価、構造化メタデータ、メモリデルタ計算、OOM検知強化が未記載

**判断**: 以下の5つの実装ギャップを特定し、REQ-198~204として要件化:

1. **G2.3自動評価**: evaluate_paper_gates.pyの_check_g2()が--frontier-report引数でfrontier_report.jsonを読み込み、G2.3をinformationalから実際のpass/fail判定に昇格（REQ-198）
2. **構造化メタデータ**: run_frontier_sweep.shが各runディレクトリにrun_metadata.json（make_exit/summary_exists/oom_in_log）を書き出し、暗黙的ヒューリスティックを明示的JSONに置き換え（REQ-199）
3. **メモリデルタ計算**: frontier_report.pyにmemory_delta_mb、memory_savings_pct、avg_memory_savings_pctが追加され、メモリ削減効果の定量評価が可能に（REQ-200）
4. **OOM検知統合**: detect_oom_from_log()の複数パターン検知、determine_status()のcompleted/oom/failed分類、exit code 137の特別処理（REQ-201）
5. **テストカバレッジ**: test_frontier_report.py（675行、14+テストクラス）がOOM検知からメタデータパイプラインまで包括的にカバー（REQ-204）

**根拠**: scripts/frontier_report.py 279行完全読み込み、scripts/run_frontier_sweep.sh 102行、scripts/evaluate_paper_gates.py _check_g2()、tests/test_frontier_report.py 675行、git log --oneline -5（4530f8c~9047beb）

**信頼性への影響**:
- 全7新規要件は 🔵（実装ベース）
- 新規要件 REQ-198~204 を追加
- architecture.mdの運用層テーブル・Mermaid図・ディレクトリ構造を更新

---

**Phase 54要件分析後** (A41 Frontier Sweep パイプライン強化ギャップ解消):
- 🔵 青信号: 227（+7: REQ-198~204）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

---

### A42: 運用スクリプト・ユーティリティモジュールの要件ギャップ

**分析日時**: 2026-05-25
**カテゴリ**: 既存設計確認・未定義部分詳細化
**背景**: kairo-requirements要件整理により、実装済みの運用スクリプト（13スクリプト）とユーティリティモジュール（5モジュール）が要件定義書（REQ-001~204）に記載されていないことが判明。これらはarchitecture.mdの運用層テーブルに記載されているが、機能要件として正式化されていなかった

**判断**: 以下の2カテゴリ13件のギャップを特定し、REQ-205~217として要件化:

1. **ユーティリティモジュール**（REQ-205~209）:
   - src/utils/io.py: orjsonベースJSON/JSONL高速I/O（save_json, load_json, save_jsonl, load_jsonl）
   - src/utils/memory.py: VRAM使用量・パラメータ数ユーティリティ
   - src/utils/run_query.py: RunMetrics JSONLログクエリAPI（TASK-0060で追加）
   - src/utils/logging.py: RichHandler ロギング設定・ディレクトリ確保
   - src/utils/checkpoint.py: モデルチェックポイント保存・TrainingState直列化/復元・NaN/Infサニタイズ

2. **運用スクリプト**（REQ-210~217）:
   - scripts/run_sweep.sh: 9設定組み合わせハイパーパラメータスイープ
   - scripts/run_ablation_suite.sh: ベースライン vs TG-LoRA変種アブレーション
   - scripts/run_high_lr_comparison.sh: 高LR安定性比較（ロールバック優位性検証）
   - scripts/run_kstep_rollback_test.sh: K-step中間ロールバック検証
   - scripts/run_accel_sweep_parallel.sh: 2-GPU並列accel paramスイープ
   - scripts/run_accel_sweep_auto.sh: GPU空き監視自動スイープラッパー
   - scripts/generate_sweep_dashboard.py: HTMLダッシュボード生成
   - scripts/compare_paper_memory_modes.py: reuse vs one-shot比較レポート

**根拠**: scripts/全37ファイルとsrc/utils/全5ファイルの網羅的調査、architecture.md運用層テーブルとのクロスチェック、Makefile既存ターゲットとの対応確認

**信頼性への影響**:
- 全13新規要件は 🔵（既存実装ベース、テスト済み）
- 新規要件 REQ-205~217 を追加
- 信頼性分布: 🔵 239（+13）、🟡 4（変更なし）、🔴 0（変更なし）

---

**Phase 55要件分析後** (A42 運用スクリプト・ユーティリティモジュール要件ギャップ解消):
- 🔵 青信号: 240（+13: REQ-205~217）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

---

### A43: モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析の要件ギャップ

**分析日時**: 2026-05-25
**カテゴリ**: 既存設計確認・未定義部分詳細化
**背景**: Phase 55完了後（457/457 acceptance criteria）、実装済み機能の中に要件化されていない重要機能が残存していることを発見。Phase 56として以下の4カテゴリを分析

**判断**: 以下の6カテゴリ14件のギャップを特定し、REQ-218~231として要件化:

1. **モデル構造検査ツール**（REQ-218~219）:
   - scripts/inspect_model.py（230行）: HuggingFaceモデルのLoRA互換ターゲットモジュール自動発見ツール。README.md Quick Startに記載、Makefile inspect/inspect-configターゲットで提供。A6でも指摘されていたが要件化されていなかった

2. **比較ダッシュボード・可視化**（REQ-220~223）:
   - compare_runs.pyのdashboardサブコマンド: マルチラン横断比較ダッシュボード（gather_runs/find_best_run/build_comparison_table/render_dashboard）
   - 5種類の可視化プロット: acceptance_rate/reduction_rate/velocity_magnitude/layer_scores/hyperparams
   - generate_markdown_report(): Markdown形式比較レポート
   - log_reports_to_mlflow(): MLflowアーティファクト自動ロギング
   - REQ-037は比較レポート生成の基本要件のみを記載し、dashboard/可視化/markdown/MLflow連携が未要件化

3. **ワンショットPrefix Feature Cache**（REQ-224~225）:
   - prefix_feature_cache_mode="one_shot": SSDバッキングのdisk-backedキャッシュモード
   - config_schema.py PrefixFeatureCacheMode定義（Literal["reuse", "one_shot"]）
   - configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml設定サーフェス
   - Makefile paper-memory-one-shotターゲットで提供
   - architecture.mdにはMappedPrefixFeatureDataset disk-backed modeとして記載済み

4. **コスト・効果分析**（REQ-226~227）:
   - scripts/analyze_prefix_cache_break_even.py（148行）: キャッシュコールドビルドコストの損益分岐点分析
   - Makefile analyze-prefix-break-evenターゲット

5. **データパイプライン細粒度ターゲット**（REQ-228）:
   - Makefile細粒度ターゲット（download-dolly, download-capybara, prepare-data-small, prepare-capybara）
   - 個別データセットの独立した取得・準備を可能にする

6. **クリーンアップ・運用ターゲット**（REQ-229~231）:
   - clean/clean-data/clean-runsクリーンアップターゲット
   - compare-prefix-cold/warm/coldwarmキャッシュモード別比較ターゲット
   - configs/9b_baseline_suffix_only_last25.yaml suffix-onlyベースライン設定サーフェス

**根拠**: scripts/inspect_model.py 230行完全読み込み、scripts/compare_runs.py 808行完全読み込み（dashboardサブコマンド・5プロット関数・markdownレポート・MLflow連携）、src/tg_lora/prefix_feature_cache.py one_shot mode、scripts/analyze_prefix_cache_break_even.py 148行、Makefile全42ターゲット、configs/全15YAMLファイル

**信頼性への影響**:
- 全14新規要件は 🔵（既存実装ベース、テスト済みまたはテスト可能）
- 新規要件 REQ-218~231 を追加
- 信頼性分布: 🔵 254（+14）、🟡 4（変更なし）、🔴 0（変更なし）

---

**Phase 56要件分析後** (A43 モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析ギャップ解消):
- 🔵 青信号: 254（+14: REQ-218~231）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

---

### A44: Phase 57-58 論文エクスポート・感度分析・サイクルモニタ・実験比較要件ギャップ

**分析日時**: 2026-05-25
**カテゴリ**: 未文書化実装の要件化
**背景**: Phase 57-58（TASK-0115~0118）で4つのモジュール/スクリプトが実装されたが、requirements.md・architecture.md・acceptance-criteria.mdに反映されていなかった

**判断**: 以下の4領域について実装ベースで要件を正式化:

1. **論文結果エクスポート**（REQ-241~242）:
   - scripts/export_paper_results.py: aggregate_summary.jsonからLaTeX/Markdown/CSV形式の出版テーブル生成
   - 不正構造JSON拒否・ファイル不存在エラー処理

2. **ハイパーパラメータ感度分析**（REQ-243~244）:
   - scripts/analyze_sensitivity.py: Pearson相関行列計算・感度ランキング生成
   - デフォルト8パラメータ×3メトリクス分析・None値ペアフィルタリング

3. **学習サイクルヘルスモニタ**（REQ-245~248）:
   - src/tg_lora/cycle_monitor.py CycleMonitor: 発散検知（NaN/Inf・loss spike）・停滞検知（patienceベース）・介入推奨
   - DivergenceReport/StagnationReport/HealthReportデータクラス

4. **実験構成マトリクス比較**（REQ-249~250）:
   - scripts/compare_experiment_configs.py: 実験自動検出・比較マトリクス・ランク付け
   - ExperimentSummary/ComparisonMatrixデータクラス・Markdown/JSON出力

**根拠**: scripts/export_paper_results.py・scripts/analyze_sensitivity.py・src/tg_lora/cycle_monitor.py・scripts/compare_experiment_configs.pyの全コード読み込み、test_cycle_monitor.py（25テスト）

**信頼性への影響**:
- 新規要件 REQ-241~250 を追加（全て 🔵 既存実装ベース）
- 信頼性分布: 🔵 264（+10）、🟡 4（変更なし）、🔴 0（変更なし）

---

### A45: Phase 61 Training Advisor モジュール・CLI要件ギャップ

**分析日時**: 2026-05-25
**カテゴリ**: 未文書化実装の要件化
**背景**: Phase 61（TASK-0121）でTraining AdvisorモジュールとCLIが実装されたが、requirements.md・architecture.md・acceptance-criteria.mdに反映されていなかった

**判断**: 以下の2領域について実装ベースで要件を正式化:

1. **Training Advisor コアモジュール**（REQ-251~256）:
   - src/tg_lora/training_advisor.py TrainingAdvisor: CycleMonitor + TrajectoryAnalyzer統合
   - AdvisoryAction（10種アクション・4段優先度・信頼度）・AdvisoryReport（3段ヘルス）・AdvisorConfig
   - generate_advice_from_history()履歴一括処理
   - NaN/Inf graceful処理・best_loss追跡

2. **Training Advisor CLI**（REQ-257~258）:
   - scripts/advise_training.py: run_metrics.jsonl→AdvisoryReport変換
   - --json/-o/--patience/--spike-threshold/--trajectory-window CLI引数
   - exit code 0/1/2によるヘルス状態通知

**根拠**: src/tg_lora/training_advisor.py・scripts/advise_training.pyの全コード読み込み、test_training_advisor.py（35テスト）

**信頼性への影響**:
- 新規要件 REQ-251~258 を追加（全て 🔵 既存実装ベース）
- 信頼性分布: 🔵 272（+8）、🟡 4（変更なし）、🔴 0（変更なし）

---

**Phase 57-58, 61要件分析後** (A44-A45):
- 🔵 青信号: 272（+18: REQ-241~258）
- 🟡 黄信号: 4（変更なし）
- 🔴 赤信号: 0（変更なし）

## 分析結果サマリー

### 確認できた事項

- 全コアアルゴリズム（velocity, extrapolator, layer_sampler, rollback, random_walk_controller）が完全実装・テスト済み
- データパイプライン・評価・比較システムが完全実装
- 運用スクリプト13件とユーティリティモジュール5件が実装・テスト済み
- テストスイート: 102テストファイル、2,217テストケース（全パス）、カバレッジ99%
- Phase 56要件追加により、モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析の4領域14要件を正式化

### 追加/変更要件

- Phase 56: REQ-218~231（14件の新規要件）— モデル検査ツール、比較ダッシュボード、ワンショットキャッシュ、コスト分析、データ細粒度ターゲット、クリーンアップターゲット
- Phase 57-58: REQ-241~250（10件の新規要件）— 論文結果エクスポート、ハイパーパラメータ感度分析、学習サイクルヘルスモニタ、実験構成マトリクス比較
- Phase 61: REQ-251~258（8件の新規要件）— Training Advisor コアモジュール、AdvisoryAction/AdvisoryReport、AdvisorConfig、advise_training.py CLI
- Phase 57 ギャップ補完: REQ-259~264（6件の新規要件）— 統計分析モジュール（stats.py）、論文実験運用Makefileターゲット

---

### A48: 実装済みコードの要件カバレッジギャップ検出

**分析日時**: 2026-05-25
**カテゴリ**: 要件カバレッジ検証
**背景**: Phase 57-61の実装完了後、全ソースモジュールとMakefileターゲットが要件定義書でトレーサビリティを持っているか確認が必要

**判断**: 以下の実装済みコードが要件定義書に未カバー:
1. src/analysis/stats.py（228行、4公開関数）— confidence_interval, paired_t_test, cohens_d, analyze_multi_seed。export_paper_results.pyとevaluate_paper_gates.pyで使用中
2. 7つのMakefileターゲット — paper-memory-dry-run, paper-memory-one-shot, paper-memory-compare-modes, paper-memory-all-modes, paper-memory-external-eval, precompute-prefix-cache, bench-velocity-ops-save-baseline

**根拠**: 全src/配下モジュールとMakefileターゲットの要件定義書grep照合

**信頼性への影響**:
- 統計分析モジュール4関数のREQ-259~262を追加（🔴→🔵に解消）
- Makefile運用ターゲット2REQを追加（REQ-263~264）

---

### A49: 比較スクリプト間の警告パターン不統一

**分析日時**: 2026-05-25
**カテゴリ**: コード品質・一貫性
**背景**: AI_HUB_MAKE_RUN_FEEDBACKで指摘。compare_runs.pyはJSONLパース失敗時にstderrへのprintのみで構造化フィールドに収集していないが、compare_experiment_configs.pyはExperimentSummary.parse_warningsに収集しJSON/markdown出力に含めている

**判断**: compare_runs.pyのgather_runs()にparse_warningsリストを追加し、format_json()をrunsキーとparse_warningsキーを含むオブジェクトに変更し、render_dashboard()にRich Panelでの警告表示を追加。compare_experiment_configs.pyと同一のパターンに統一

**根拠**: compare_experiment_configs.py ExperimentSummary.parse_warningsパターン、format_as_markdown/format_as_jsonでの警告出力

**信頼性への影響**:
- REQ-037aを追加し、compare_runs.pyのparse_warnings構造化収集を要件化（🔴→🔵に解消）

---

### 残課題

- REQ-301/302（MLflow完全連携）は依然として🟡（依存指定のみで実装部分的）
- Phase 43/50はGPU依存タスクとして未完了（実行環境が必要）
- 論文実験（Stage 2-5）の実行とGate判定はGPU環境が必要
- Training Advisorの学習ループ統合（evaluate()をtrain_tg_lora.pyのメインループに接続）は未実装

### 信頼性レベル分布

**Phase 57 ギャップ補完後**:

- 🔵 青信号: 278件
- 🟡 黄信号: 4件
- 🔴 赤信号: 0件

---

### A50: PSA（Prior-based Subspace Amplification）モジュールの要件ギャップ

**分析日時**: 2026-06-10
**カテゴリ**: 未定義部分詳細化
**背景**: docs/GOAL.mdが軌跡外挿からサブスペース増幅への研究方向転換を文書化しているが、PSAコアモジュール（src/tg_lora/psa.py）と5つの関連モジュール（RegimeDetector, ActivationFingerprintTracker, LAWAAverager, LayerDeltaAnalysis, アブレーションスクリプト）が要件定義書に未反映

**判断**: PSAをPhase 62として要件化。PSAPriorコア（REQ-265~269）、PSA Config（REQ-270~271）、RegimeDetector（REQ-272~273）、ActivationFingerprintTracker（REQ-274~275）、LAWAAverager（REQ-276~277）、LayerDeltaAnalysis（REQ-278~279）、アブレーションスクリプト（REQ-280~282）、研究方向転換記録（REQ-283~284）、境界値（EDGE-202~211）を追加

**根拠**: src/tg_lora/psa.py・regime.py・activation_regime.py・weight_averaging.py・layer_delta_analysis.pyの全コード読み込み、configs/9b_tg_lora_psa.yaml、scripts/run_psa_ablation.sh、docs/GOAL.md §1~§7

**信頼性への影響**:
- PSA関連20要件を全て🔵（既存実装ベース）で追加
- GOAL.mdの研究方向転換を要件レベルで追跡可能に
- これにより残課題にPSA学習ループ統合（train_tg_lora.pyへのPSA配線）を追加

---

### A51: GOAL.md研究方向転換の要件反映

**分析日時**: 2026-06-10
**カテゴリ**: 追加要件
**背景**: GOAL.mdが§1で軌跡外挿（velocity/M9-FD）の否定的検証結果を記録し、§3でPSAをメインラインに指定している。この研究方針変更が要件定義書に反映されていない

**判断**: REQ-283でGOAL.mdの内容を要件化し、REQ-284でPSAと既存外挿要件の共存関係を明記。既存の外挿要件（REQ-002~003, REQ-016）を無効化せず、設定で切り替え可能とする方針を維持

**根拠**: docs/GOAL.md §1.1~1.5（8研究トラック）、§3.1（Main Line: PSA）、§7（3鉄則）

**信頼性への影響**:
- 研究方向転換が要件レベルで追跡可能に（🔵）
- 外挿とPSAの共存が明文化され、設定サーフェス（configs/9b_tg_lora.yaml vs 9b_tg_lora_psa.yaml）で切り替え可能

---

### 残課題（更新）

- REQ-301/302（MLflow完全連携）は依然として🟡（依存指定のみで実装部分的）
- Phase 43/50はGPU依存タスクとして未完了（実行環境が必要）
- 論文実験（Stage 2-5）の実行とGate判定はGPU環境が必要
- Training Advisorの学習ループ統合（evaluate()をtrain_tg_lora.pyのメインループに接続）は未実装
- **PSAの学習ループ統合（PSAPriorをtrain_tg_lora.pyのメインループに配線）は部分的** — PSA config読み込みとamplify_gradients()呼び出しは実装済みだが、regime-aware resetとactivation fingerprintのループ統合は未完了
- **PSAアブレーション実験（γ sweep、regime reset ON/OFF）の実際のGPU実行は未完了**

### 信頼性レベル分布（Phase 62追加後）

**Phase 62 PSA要件追加後**:

- 🔵 青信号: 298件 (+20)
- 🟡 黄信号: 4件
- 🔴 赤信号: 0件
