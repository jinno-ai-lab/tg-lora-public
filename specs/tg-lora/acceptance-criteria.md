# TG-LoRA 受け入れ基準


<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-05-21
**関連要件定義**: [requirements.md](requirements.md)
**関連ユーザストーリー**: [user-stories.md](user-stories.md)
**分析記録**: [interview-record.md](interview-record.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: PRD・既存要件定義書・設計文書・既存実装を参考にした確実な基準
- 🟡 **黄信号**: PRD・既存要件定義書・設計文書・既存実装から妥当な推測による基準
- 🔴 **赤信号**: 参照資料にない自動推定による基準

---

## REQ-001: Velocity追跡（EMA） 🔵

**信頼性**: 🔵 *velocity.py 既存実装・test_velocity.py より*

### Given（前提条件）

- Velocityインスタンスが初期化済み
- LoRAパラメータのdelta辞書が与えられている

### When（実行条件）

- `velocity.update(delta, beta)` を呼び出す

### Then（期待結果）

- 初回更新: velocity state = delta
- 2回目以降: velocity state = beta * previous_state + (1 - beta) * delta
- cosine_similarity が [0.0, 1.0] の範囲で計算される

### テストケース

#### 正常系

- [x] **TC-001-01**: 初回更新でdeltaがそのままstateになる 🔵
  - **入力**: delta = {"lora_A": tensor([1.0, 2.0])}, beta = 0.8
  - **期待結果**: state = {"lora_A": tensor([1.0, 2.0])}
  - **信頼性**: 🔵 *test_velocity.py `test_velocity_first_update` より*

- [x] **TC-001-02**: EMA更新が正しく計算される 🔵
  - **入力**: 2回連続更新、beta=0.8
  - **期待結果**: state = 0.8 * first_delta + 0.2 * second_delta
  - **信頼性**: 🔵 *test_velocity.py `test_velocity_ema` より*

- [x] **TC-001-03**: 同方向のcosine similarityが1.0 🔵
  - **入力**: state と同じ方向のdelta
  - **期待結果**: cosine_similarity = 1.0
  - **信頼性**: 🔵 *test_velocity.py `test_cosine_similarity` より*

#### 境界値

- [x] **TC-001-B01**: velocity未初期化時のcosine similarity 🔵
  - **入力**: state=None の状態で cosine_similarity 呼び出し
  - **期待結果**: 0.0
  - **信頼性**: 🔵 *test_velocity.py `test_cosine_similarity_no_state` より*

- [x] **TC-001-B02**: deltaにstateに存在しないキーが含まれる場合のcosine similarity 🔵
  - **入力**: delta = {"lora_A": tensor, "extra_key": tensor}, state = {"lora_A": tensor}
  - **期待結果**: KeyErrorなく、共通キーのみで類似度計算
  - **信頼性**: 🔵 *test_velocity.py・velocity.py KeyError修正(6717ee8)より*

- [x] **TC-001-B03**: update後にmagnitudeが正しく記録される 🔵
  - **入力**: delta = {"w": tensor([3.0, 4.0])}, beta = 0.8
  - **期待結果**: magnitudes[0] ≈ 5.0（L2 norm）
  - **信頼性**: 🔵 *test_velocity.py `test_magnitude_tracking_first_update` より*

- [x] **TC-001-B04**: reset()がstateとmagnitude_historyの両方をクリア 🔵
  - **入力**: 2回update後、reset()
  - **期待結果**: state=None, magnitudes=[]
  - **信頼性**: 🔵 *test_velocity.py `test_magnitude_reset` より*

- [x] **TC-001-B05**: max_history超過時に最古エントリが破棄される 🔵
  - **入力**: max_history=3, 4回update
  - **期待結果**: len(magnitudes)=3, 最古エントリが削除済み
  - **信頼性**: 🔵 *test_velocity.py `test_max_history_trims_magnitudes` より*

---

## REQ-049: Velocity Magnitude 異常検出 🔵

**信頼性**: 🔵 *velocity.py `is_magnitude_anomalous` 実装・test_velocity.py より*

### Given（前提条件）

- Velocityインスタンスが初期化済み
- magnitude_historyに複数件の記録がある

### When（実行条件）

- `is_magnitude_anomalous(threshold_sigma)` を呼び出す

### Then（期待結果）

- 履歴3件未満: False
- std < 1e-12: latest > mean * 2.0 で判定
- 通常: latest > mean + threshold_sigma * std で判定

### テストケース

#### 正常系

- [x] **TC-049-01**: 安定したmagnitudeの連続後にspikeを検出 🔵
  - **入力**: 5回小幅update → 大幅spike update
  - **期待結果**: is_magnitude_anomalous(threshold_sigma=2.0) == True
  - **信頼性**: 🔵 *test_velocity.py `test_is_magnitude_anomalous_detects_spike` より*

- [x] **TC-049-02**: 安定したmagnitudeでは非異常 🔵
  - **入力**: 5回同一規模update
  - **期待結果**: is_magnitude_anomalous() == False
  - **信頼性**: 🔵 *test_velocity.py `test_is_magnitude_anomalous_near_zero_std` より*

#### 境界値

- [x] **TC-049-B01**: 履歴2件ではFalse 🔵
  - **入力**: 2回update後
  - **期待結果**: False
  - **信頼性**: 🔵 *test_velocity.py `test_is_magnitude_anomalous_false_few_entries` より*

---

## REQ-050: Velocity Magnitude トレンド 🔵

**信頼性**: 🔵 *velocity.py `magnitude_trend` 実装・test_velocity.py より*

### Given（前提条件）

- Velocityインスタンスが初期化済み
- magnitude_historyに記録がある

### When（実行条件）

- `magnitude_trend(window)` を呼び出す

### Then（期待結果）

- データ2件未満: 0.0
- 減少系列: 負の値（収束傾向）
- 増加系列: 正の値（発散傾向）

### テストケース

#### 正常系

- [x] **TC-050-01**: 減少系列で負のトレンド 🔵
  - **入力**: magnitudes = [10, 8, 6, 4, 2]
  - **期待結果**: trend < 0
  - **信頼性**: 🔵 *test_velocity.py `test_magnitude_trend_negative` より*

#### 境界値

- [x] **TC-050-B01**: データ1件で0.0 🔵
  - **入力**: 1回update後
  - **期待結果**: 0.0
  - **信頼性**: 🔵 *test_velocity.py `test_magnitude_trend_insufficient_data` より*

---

## REQ-051: データスキーマ検証（DataRecord） 🔵

**信頼性**: 🔵 *schema.py DataRecord 実装・test_schema.py より*

### Given（前提条件）

- 生データレコード（dict）が与えられている

### When（実行条件）

- `DataRecord(**raw)` でインスタンス化

### Then（期待結果）

- textが空の場合: ValueError
- textにChatMLマーカーがない場合: ValueError
- token_count <= 0 の場合: ValueError
- 正常データ: インスタンス生成成功

### テストケース

#### 正常系

- [x] **TC-051-01**: 正常なChatMLレコードのインスタンス化 🔵
  - **入力**: text="<|im_start|>user\nHello<|im_start|>assistant\nHi"
  - **期待結果**: source=None, token_count=None
  - **信頼性**: 🔵 *test_schema.py `test_valid_record` より*

#### 異常系

- [x] **TC-051-E01**: 空textでValueError 🔵
  - **信頼性**: 🔵 *test_schema.py より*
- [x] **TC-051-E02**: ChatMLマーカーなしでValueError 🔵
  - **信頼性**: 🔵 *test_schema.py より*
- [x] **TC-051-E03**: 負のtoken_countでValueError 🔵
  - **信頼性**: 🔵 *test_schema.py より*

---

## REQ-052: バッチデータ検証（validate_records） 🔵

**信頼性**: 🔵 *schema.py ValidationSummary/validate_records 実装・test_schema.py より*

### Given（前提条件）

- 生データレコードのリストが与えられている

### When（実行条件）

- `validate_records(records)` を呼び出す

### Then（期待結果）

- 有効レコードのみ返却
- ValidationSummaryにtotal/valid/skipped/errorsが正しく集計される

### テストケース

#### 正常系

- [x] **TC-052-01**: 全正常レコードのバッチ検証 🔵
  - **信頼性**: 🔵 *test_schema.py より*
- [x] **TC-052-02**: 不正レコード混在時のスキップと集計 🔵
  - **信頼性**: 🔵 *test_schema.py より*

---

## REQ-002: 外挿の適用 🔵

**信頼性**: 🔵 *extrapolator.py 既存実装・test_extrapolator.py より*

### Given（前提条件）

- モデルにLoRAアダプタが適用済み
- Velocity stateが計算済み

### When（実行条件）

- `apply_extrapolation(model, velocity, active_names, alpha_by_name, default_alpha, n_steps, relative_update_cap)` を呼び出す

### Then（期待結果）

- 指定されたactive_namesのLoRAパラメータのみ更新される
- 更新量がrelative_update_capで制限される

### テストケース

#### 正常系

- [x] **TC-002-01**: 指定alpha, n_stepsで外挿が適用される 🔵
  - **入力**: alpha=0.3, n_steps=5, velocity方向に更新
  - **期待結果**: activeパラメータが velocity方向に alpha*n_steps 分移動
  - **信頼性**: 🔵 *test_extrapolator.py `test_apply_extrapolation` より*

- [x] **TC-002-02**: active_names外のパラメータは変更されない 🔵
  - **入力**: active_names で一部レイヤーのみ指定
  - **期待結果**: 指定外のパラメータは元の値を維持
  - **信頼性**: 🔵 *test_extrapolator.py `test_apply_extrapolation_partial_layers` より*

#### 境界値

- [x] **TC-002-B01**: 更新が上限を超える場合にcapされる 🔵
  - **入力**: 大きなvelocity、max_ratio=0.5
  - **期待結果**: 更新ノルムが参照ノルムの50%以下に制限
  - **信頼性**: 🔵 *test_extrapolator.py `test_cap_update` より*

- [x] **TC-002-B02**: 小さな更新はそのまま通過する 🔵
  - **入力**: 極小のvelocity
  - **期待結果**: 更新が変更されず通過
  - **信頼性**: 🔵 *test_extrapolator.py `test_cap_update_no_cap_needed` より*

---

## REQ-006: レイヤーサンプリング 🔵

**信頼性**: 🔵 *layer_sampler.py 既存実装・test_layer_sampler.py より*

### Given（前提条件）

- モデルにLoRAアダプタが適用済み
- レイヤー数が自動検出されている

### When（実行条件）

- `select_active_layers(model, strategy, ...)` を呼び出す

### Then（期待結果）

- 指定戦略に応じたレイヤーが選択される
- 非連続レイヤーインデックスでも正しく動作する

### テストケース

#### 正常系

- [x] **TC-006-01**: last_25_percent戦略で最後25%のレイヤーが選択される 🔵
  - **入力**: 12層モデル、strategy="last_25_percent"
  - **期待結果**: レイヤー9, 10, 11が選択
  - **信頼性**: 🔵 *test_layer_sampler.py `test_last_25_percent` より*

- [x] **TC-006-02**: last_25_percent_plus_random_2で最終25%+ランダム2層 🔵
  - **入力**: 12層モデル、strategy="last_25_percent_plus_random_2"
  - **期待結果**: レイヤー9,10,11 + 中間層から2つ
  - **信頼性**: 🔵 *test_layer_sampler.py `test_last_25_plus_random_2` より*

- [x] **TC-006-03**: レイヤー数が正しく検出される 🔵
  - **入力**: 12層モデル
  - **期待結果**: get_num_layers = 12
  - **信頼性**: 🔵 *test_layer_sampler.py `test_get_num_layers` より*

---

## REQ-009: ロールバック機構 🔵

**信頼性**: 🔵 *rollback_manager.py 既存実装・test_rollback_manager.py より*

### Given（前提条件）

- RollbackManagerが初期化済み
- モデルにLoRAアダプタが適用済み

### When（実行条件）

- save → パラメータ変更 → rollback を実行

### Then（期待結果）

- パラメータが保存時の状態に復元される

### テストケース

#### 正常系

- [x] **TC-009-01**: 保存→変更→ロールバックで元の状態に復元 🔵
  - **入力**: save → 値変更 → rollback(index=0)
  - **期待結果**: パラメータがsave時の値に復元
  - **信頼性**: 🔵 *test_rollback_manager.py `test_rollback_basic` より*

- [x] **TC-009-02**: デフォルト（最後の保存）にロールバック 🔵
  - **入力**: 複数回save → rollback()
  - **期待結果**: 最後のsaveに復元
  - **信頼性**: 🔵 *test_rollback_manager.py `test_rollback_last` より*

#### 境界値

- [x] **TC-009-B01**: 履歴空の状態でpop/clear 🔵
  - **入力**: 初期状態でpop()、clear()
  - **期待結果**: エラーなく処理
  - **信頼性**: 🔵 *test_rollback_manager.py `test_pop_and_clear` より*

---

## REQ-011: ランダムウォークコントローラ 🔵

**信頼性**: 🔵 *random_walk_controller.py 既存実装・test_random_walk_controller.py より*

### Given（前提条件）

- RandomWalkControllerが初期ハイパーパラメータで初期化済み

### When（実行条件）

- propose → accept/reject のサイクルを実行

### Then（期待結果）

- alphaが[alpha_min, alpha_max]の範囲内に収まる
- 受理/拒否に応じてハイパーパラメータが適応更新される

### テストケース

#### 正常系

- [x] **TC-011-01**: 初期状態のハイパーパラメータが正しい 🔵
  - **期待結果**: K, N, alpha, beta が初期値に設定
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_initial_state` より*

- [x] **TC-011-02**: alpha提案が範囲内に収まる 🔵
  - **入力**: 100回propose
  - **期待結果**: 全て alpha_min <= alpha <= alpha_max
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_propose_alpha_in_range` より*

- [x] **TC-011-03**: 受理判定が正しい（loss_after < loss_pilot * 1.005） 🔵
  - **入力**: loss_pilot=1.0, loss_after=0.99
  - **期待結果**: accepted=True, alphaが増加
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_accept_and_reward` より*

- [x] **TC-011-04**: 拒否時にalphaが減少 🔵
  - **入力**: loss_after > loss_pilot * (1 + tolerance)
  - **期待結果**: penalized, alphaが減少
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_penalize` より*

- [x] **TC-011-05**: 受理率が正しく計算される 🔵
  - **入力**: 3回中2回受理
  - **期待結果**: acceptance_rate ≈ 0.667
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_acceptance_rate` より*

---

## REQ-016: TG-LoRA学習サイクル 🔵

**信頼性**: 🔵 *train_tg_lora.py・test_smoke.py より*

### Given（前提条件）

- モデル、データローダー、オプティマイザが初期化済み
- TG-LoRA設定（K, N, alpha, beta等）が指定されている

### When（実行条件）

- 1サイクルの学習ループを実行

### Then（期待結果）

- pilot → snapshot → extrapolate → accept/rollback のフローが完了
- 損失値が有限（NaN/Infでない）

### テストケース

#### 正常系

- [x] **TC-016-01**: 1サイクルのE2E動作（pilot→外挿→受理/拒否） 🔵
  - **入力**: tiny GPT-2モデル、小データセット
  - **期待結果**: 1サイクル完了、損失が有限
  - **信頼性**: 🔵 *test_smoke.py `test_smoke_tg_lora_one_cycle` より*

- [x] **TC-016-02**: スナップショット→復元の往復テスト 🔵
  - **入力**: 実モデルでのsnapshot/restore
  - **期待結果**: 復元後のパラメータがsnapshotと完全一致
  - **信頼性**: 🔵 *test_smoke.py `test_smoke_snapshot_restore_roundtrip` より*

- [x] **TC-016-03**: ベースライン学習が正常に完了 🔵
  - **入力**: tiny GPT-2、5ステップ
  - **期待結果**: 全ステップで損失が有限
  - **信頼性**: 🔵 *test_smoke.py `test_smoke_baseline_training` より*

---

## REQ-021: モデル読み込み・LoRA適用 🔵

**信頼性**: 🔵 *load_model.py・lora_utils.py 既存実装より*

### Given（前提条件）

- HuggingFaceモデル名（Qwen/Qwen3.5-9B）が指定されている
- 設定ファイルで量子化・LoRAパラメータが指定されている

### When（実行条件）

- `load_base_model(cfg)` → `apply_lora(model, cfg)` を実行

### Then（期待結果）

- 4bit量子化モデルが読み込まれる
- LoRAアダプタが全Linear層に適用される
- LoRAパラメータが反復・カウント可能

### テストケース

#### 正常系

- [x] **TC-021-01**: iter_lora_paramsでLoRAパラメータを列挙 🔵
  - **期待結果**: lora_A/lora_Bを含むパラメータのみ抽出
  - **信頼性**: 🔵 *lora_utils.py 実装より*

- [x] **TC-021-02**: iter_lora_params_by_layerでレイヤー別グループ化 🔵
  - **期待結果**: レイヤーインデックス→パラメータリストの辞書
  - **信頼性**: 🔵 *lora_utils.py 実装より*

---

## REQ-025: データセットダウンロード 🔵

**信頼性**: 🔵 *download_data.py・docs/datasets.md より*

### Given（前提条件）

- HuggingFaceへのアクセスが可能

### When（実行条件）

- `make download-data` を実行

### Then（期待結果）

- data/raw/ にJSONL形式でデータセットが保存される
- Dolly 15k: instruction/context/response/categoryフィールド
- Capybara: マルチターン対話から最初のターンを抽出

### テストケース

#### 正常系

- [x] **TC-025-01**: Dolly 15kのダウンロードとJSONL変換 🔵
  - **期待結果**: data/raw/dolly_15k.jsonl が生成、15,011件
  - **信頼性**: 🔵 *download_data.py・docs/datasets.md より*
  - **テスト**: test_download_data.py `test_dolly_download_produces_jsonl`

- [x] **TC-025-02**: CapybaraのダウンロードとJSONL変換 🔵
  - **期待結果**: data/raw/capybara.jsonl が生成、~16,000件
  - **信頼性**: 🔵 *download_data.py・docs/datasets.md より*
  - **テスト**: test_download_data.py `test_capybara_download_produces_jsonl`

---

## REQ-026: データ前処理・ChatML変換 🔵

**信頼性**: 🔵 *prepare_data.py 既存実装より*

### Given（前提条件）

- 生JSONLデータがdata/raw/に存在

### When（実行条件）

- `make prepare-data` を実行

### Then（期待結果）

- ChatML形式のJSONLに変換される
- train.jsonl (3000件), valid_quick.jsonl (300件), valid_full.jsonl (300件), gold_test.jsonl (50件) が生成

### テストケース

#### 正常系

- [x] **TC-026-01**: ChatML形式への変換 🔵
  - **入力**: {instruction, response} レコード
  - **期待結果**: text = `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`
  - **信頼性**: 🔵 *prepare_data.py フォーマット実装より*

- [x] **TC-026-02**: context付きChatML変換 🔵
  - **入力**: {instruction, context, response} レコード
  - **期待結果**: userターンに `Context: {context}` が含まれる
  - **信頼性**: 🔵 *prepare_data.py context処理より*

---

## REQ-035: lm-evaluation-harness評価 🔵

**信頼性**: 🔵 *run_eval.sh・run_eval_lora.sh・docs/evaluation.md より*

### Given（前提条件）

- 学習済みモデルチェックポイントが存在
- lm-evalがインストール済み

### When（実行条件）

- `make eval MODEL_PATH=...` または `make eval-lora ADAPTER_PATH=...` を実行

### Then（期待結果）

- ARC-Easy, HellaSwag, GSM8K, TruthfulQA MC2 で評価される
- 結果がreports/eval/にJSONで保存される

### テストケース

#### 正常系

- [x] **TC-035-01**: マージ済みモデルの評価 🔵
  - **入力**: MODEL_PATH
  - **期待結果**: 4タスクのスコアがJSONで出力
  - **信頼性**: 🔵 *run_eval.sh より*
  - **テスト**: test_run_eval.py `test_run_eval_default_tasks`

- [x] **TC-035-02**: LoRAアダプタのマージ→評価→クリーンアップ 🔵
  - **入力**: ADAPTER_PATH
  - **期待結果**: マージ→評価→一時ファイル削除
  - **信頼性**: 🔵 *run_eval_lora.sh より*
  - **テスト**: test_run_eval.py `test_run_eval_lora_has_three_steps`

---

## REQ-036: 公正比較実験 🔵

**信頼性**: 🔵 *run_comparison.sh・compare_runs.py より*

### Given（前提条件）

- ベースライン設定とTG-LoRA設定が存在
- 同一の学習データが準備済み

### When（実行条件）

- `make compare BUDGET=1500` を実行

### Then（期待結果）

- ベースライン: 1500 backward pass
- TG-LoRA: 1500/K_initial サイクル
- 比較レポートが生成される

### テストケース

#### 正常系

- [x] **TC-036-01**: 同一予算での比較実験実行 🔵
  - **入力**: BUDGET=1500
  - **期待結果**: 両方のrun_metrics.jsonlが生成
  - **信頼性**: 🔵 *run_comparison.sh より*
  - **テスト**: test_compare_runs.py `test_budget_parity_formula`

- [x] **TC-036-02**: 比較レポートの生成 🔵
  - **入力**: 2つのrun_metrics.jsonl
  - **期待結果**: 損失曲線プロット、効率メトリクス、受理率を含むレポート
  - **信頼性**: 🔵 *compare_runs.py より*
  - **テスト**: test_compare_runs.py `test_generate_report_contains_key_sections`

---

## 非機能要件テスト

### NFR-001: 学習効率 🔵

**信頼性**: 🔵 *run_comparison.sh 公正比較ロジックより*

- [x] **TC-NFR-001-01**: TG-LoRAが同等backward pass予算でベースラインと比較可能
  - **測定項目**: backward pass数の一致
  - **目標値**: baseline_steps = tg_lora_cycles * K_initial
  - **信頼性**: 🔵 *run_comparison.sh より*
  - **テスト**: test_compare_runs.py `test_budget_parity_formula`

### NFR-103: eval_loss状態リーク防止 🔵

**信頼性**: 🔵 *eval_loss.py context manager実装より*

- [x] **TC-NFR-103-01**: eval前後でモデルのtraining modeが不変
  - **検証内容**: eval_loss呼び出し前後のmodel.training状態
  - **期待結果**: 状態が変更されない
  - **信頼性**: 🔵 *eval_loss.py コンテキストマネージャより*

### NFR-304: RunMetricsコンテキストマネージャ 🔵

**信頼性**: 🔵 *run_metrics.py `__enter__`/`__exit__` 実装(6717ee8)より*

- [x] **TC-NFR-304-01**: with文でRunMetricsを使用し、終了時にファイルがクローズされる 🔵
  - **入力**: `with RunMetrics(...) as m: m.write_header(...)` 
  - **期待結果**: ブロック終了後、m._file.closed == True
  - **信頼性**: 🔵 *test_run_metrics.py `test_context_manager` より*

### NFR-201: 再現性 🔵

**信頼性**: 🔵 *seed.py 実装より*

- [x] **TC-NFR-201-01**: 同一シードで同一結果が得られる
  - **検証内容**: 乱数シード設定の完全性
  - **期待結果**: random, numpy, torch, CUDA 全てでシードが設定される
  - **信頼性**: 🔵 *seed.py より*

---

## Phase 2: カバレッジ完成・コード品質テスト

### NFR-302: RunMetrics GPUパス 🔵

**信頼性**: 🔵 *run_metrics.py GPU依存行のモックテスト（TASK-0008）より*

- [x] **TC-NFR-302-01**: CUDA利用時に`reset_peak_memory_stats`が呼ばれる 🔵
  - **入力**: `torch.cuda.is_available()` = True でRunMetrics初期化
  - **期待結果**: `reset_peak_memory_stats()` が1回呼ばれる
  - **信頼性**: 🔵 *test_run_metrics.py `test_init_resets_peak_memory` より*

- [x] **TC-NFR-302-02**: write_headerでGPU名・総VRAMが取得される 🔵
  - **入力**: CUDA利用可能状態でwrite_header
  - **期待結果**: recordに gpu_name, gpu_total_memory_mb が含まれる
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_header_with_gpu` より*

- [x] **TC-NFR-302-03**: record_stepでGPU VRAM使用量が記録される 🔵
  - **入力**: モック化した`vram_usage_mb()` で allocated=512, reserved=1024
  - **期待結果**: recordの gpu_allocated_mb=512, gpu_reserved_mb=1024
  - **信頼性**: 🔵 *test_run_metrics.py `test_record_step_gpu_vram` より*

- [x] **TC-NFR-302-04**: ピークメモリ更新ロジックが動作する 🔵
  - **入力**: `max_memory_allocated` が前回より大きい値を返す
  - **期待結果**: gpu_peak_mb が更新される
  - **信頼性**: 🔵 *test_run_metrics.py `test_record_step_peak_memory_update` より*

### REQ-021拡張: load_model モックテスト 🔵

**信頼性**: 🔵 *load_model.py モックベースユニットテスト（TASK-0009）より*

- [x] **TC-021-03**: build_bnb_config が4bit有効/無効で正しいConfigを生成 🔵
  - **入力**: load_in_4bit=True/False, 各dtype文字列
  - **期待結果**: 4bit=True→BitsAndBytesConfig返却、False→None
  - **信頼性**: 🔵 *test_load_model.py `test_build_bnb_config_*` より*

- [x] **TC-021-04**: _resolve_dtype が全dtype文字列を正しく解決 🔵
  - **入力**: "fp16", "bf16", "fp32", 未知文字列
  - **期待結果**: 対応するtorch dtype、デフォルトはbfloat16
  - **信頼性**: 🔵 *test_load_model.py `test_resolve_dtype_*` より*

- [x] **TC-021-05**: get_device_map が設定値/デフォルトを正しく返す 🔵
  - **入力**: device_map設定あり/なし
  - **期待結果**: 設定値 または "cuda:0"（デフォルト）
  - **信頼性**: 🔵 *test_load_model.py `test_get_device_map_*` より*

- [x] **TC-021-06**: load_tokenizer がpad_tokenを設定 🔵
  - **入力**: pad_token=None のtokenizer
  - **期待結果**: eos_tokenに設定される
  - **信頼性**: 🔵 *test_load_model.py `test_load_tokenizer_sets_pad_token_when_none` より*

- [x] **TC-021-07**: apply_lora が正しいLoraConfigでget_peft_modelを呼ぶ 🔵
  - **入力**: モックモデル、LoRA設定
  - **期待結果**: LoraConfigパラメータが正しく、get_peft_modelが呼ばれる
  - **信頼性**: 🔵 *test_load_model.py `test_apply_lora_creates_config_and_calls_get_peft_model` より*

- [x] **TC-021-08**: load_base_model の基本フロー検証 🔵
  - **入力**: 4bit有効/無効、gradient_checkpointing有効/無効、VRAM閾値
  - **期待結果**: 各分岐が正しく動作
  - **信頼性**: 🔵 *test_load_model.py `test_load_base_model_*` より*

### TASK-0010回帰テスト: バグ修正検証 🔵

**信頼性**: 🔵 *TASK-0010 ソースコード精査で発見・修正された6件のバグより*

- [x] **TC-BUG-01**: extrapolator.py — active_namesにvelocity未存在キーでKeyError 🔵
  - **修正内容**: velocity[name]へのアクセスを.get()で安全化
  - **テスト**: test_extrapolator.py `test_active_names_key_missing_from_velocity`
  - **信頼性**: 🔵 *TASK-0010 Bug #1 より*

- [x] **TC-BUG-02**: lora_state.py — diff_loraのafter辞書キー欠落でKeyError 🔵
  - **修正内容**: after[k]へのアクセスを.get()で安全化
  - **テスト**: test_lora_state.py `test_diff_lora_missing_after_key`
  - **信頼性**: 🔵 *TASK-0010 Bug #2 より*

- [x] **TC-BUG-03**: lora_state.py — load_lora_snapshotのstateキー欠落でKeyError 🔵
  - **修正内容**: state[name]へのアクセスを.get()で安全化
  - **テスト**: test_lora_state.py `test_load_lora_snapshot_missing_key`
  - **信頼性**: 🔵 *TASK-0010 Bug #3 より*

- [x] **TC-BUG-04**: velocity.py — 新規キー含むdeltaでのKeyError 🔵
  - **修正内容**: update時の新規キーを安全に処理
  - **テスト**: test_velocity.py `test_update_with_new_keys`
  - **信頼性**: 🔵 *TASK-0010 Bug #4 より*

- [x] **TC-BUG-05**: eval_format.py — 例外時のmodel.train()未呼出 🔵
  - **修正内容**: try/finally で model.train() を保証
  - **テスト**: test_eval_modules.py `test_eval_format_restores_training_on_error`
  - **信頼性**: 🔵 *TASK-0010 Bug #5 より*

- [x] **TC-BUG-06**: eval_task.py — 例外時のmodel.train()未呼出 🔵
  - **修正内容**: try/finally で model.train() を保証
  - **テスト**: test_eval_modules.py `test_eval_task_restores_training_on_error`
  - **信頼性**: 🔵 *TASK-0010 Bug #6 より*

---

## Phase 3: CycleState・DeltaTracker・統合テスト

### REQ-038: CycleState サイクル状態追跡 🔵

**信頼性**: 🔵 *cycle_state.py・test_cycle_state.py より*

### Given（前提条件）

- CycleStateインスタンスが初期化済み

### When（実行条件）

- `record_cycle(K, N, grad_accum, train_loss, valid_loss, accepted)` を呼び出す

### Then（期待結果）

- cycle, full_backward_passes, extrapolation_steps が正しく累積される
- accepted/rejected カウントが更新される
- valid_loss が best_loss を改善した場合、best_loss と best_step が更新され stale_cycles がリセットされる

### テストケース

#### 正常系

- [x] **TC-038-01**: デフォルト初期値が全てゼロ/inf 🔵
  - **期待結果**: cycle=0, full_backward_passes=0, extrapolation_steps=0, best_loss=inf
  - **信頼性**: 🔵 *test_cycle_state.py `TestCycleStateInit` より*

- [x] **TC-038-02**: 1サイクル記録でカウンタが正しく更新 🔵
  - **入力**: K=5, N=10, grad_accum=2, train_loss=1.5
  - **期待結果**: cycle=1, full_backward_passes=10, extrapolation_steps=10
  - **信頼性**: 🔵 *test_cycle_state.py `TestRecordCycle.test_increments_counters` より*

- [x] **TC-038-03**: 拒否サイクルの rejected_count が増加 🔵
  - **入力**: accepted=False
  - **期待結果**: accepted_count=0, rejected_count=1
  - **信頼性**: 🔵 *test_cycle_state.py `TestRecordCycle.test_rejected_cycle` より*

- [x] **TC-038-04**: 複数サイクルで正しく累積 🔵
  - **入力**: 2サイクル (K=5,grad_accum=1), (K=3,grad_accum=2)
  - **期待結果**: full_backward_passes=5+6=11, extrapolation_steps=10+5=15
  - **信頼性**: 🔵 *test_cycle_state.py `TestRecordCycle.test_accumulates_across_multiple_cycles` より*

- [x] **TC-038-05**: best_loss改善時にstale_cyclesが0にリセット 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestRecordCycle.test_updates_best_loss_on_improvement` より*

- [x] **TC-038-06**: best_loss非改善時にstale_cyclesが増加 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestRecordCycle.test_increments_stale_on_no_improvement` より*

- [x] **TC-038-07**: valid_loss=Noneの場合はbest追跡をスキップ 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestRecordCycle.test_no_valid_loss_skips_best_tracking` より*

### REQ-039: CycleState 削減率・受理率 🔵

**信頼性**: 🔵 *cycle_state.py プロパティ・test_cycle_state.py より*

### テストケース

- [x] **TC-039-01**: ステップ0件時のreduction_rate=0.0 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestReductionRate.test_zero_when_no_steps` より*

- [x] **TC-039-02**: reduction_rate = 1 − backward/(backward+extrap) 🔵
  - **入力**: full_backward_passes=100, extrapolation_steps=300
  - **期待結果**: 0.75
  - **信頼性**: 🔵 *test_cycle_state.py `TestReductionRate.test_computes_correctly` より*

- [x] **TC-039-03**: 外挿なし時のreduction_rate=0.0 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestReductionRate.test_no_extrapolation` より*

- [x] **TC-039-04**: 全外挿時のreduction_rate=1.0 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestReductionRate.test_all_extrapolation` より*

- [x] **TC-039-05**: acceptance_rateが正しく計算される 🔵
  - **入力**: accepted=7, rejected=3
  - **期待結果**: 0.7
  - **信頼性**: 🔵 *test_cycle_state.py `TestAcceptanceRate.test_mixed` より*

### REQ-040: CycleState 早期終了判定 🔵

**信頼性**: 🔵 *cycle_state.py `should_stop`・test_cycle_state.py より*

### テストケース

- [x] **TC-040-01**: patience=Noneの場合は常にFalse 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestShouldStop.test_never_when_patience_none` より*

- [x] **TC-040-02**: stale_cycles >= patience かつ cycle >= min_cycles でTrue 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestShouldStop.test_stops_when_patience_exceeded` より*

- [x] **TC-040-03**: cycle < min_cycles の場合はFalse 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestShouldStop.test_no_stop_below_min_cycles` より*

- [x] **TC-040-04**: stale_cycles < patience の場合はFalse 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestShouldStop.test_no_stop_below_patience` より*

- [x] **TC-040-05**: 境界値（stale==patience, cycle==min_cycles）でTrue 🔵
  - **信頼性**: 🔵 *test_cycle_state.py `TestShouldStop.test_exact_boundary` より*

### REQ-040a: フル評価 stale_cycles 分離 🔵

**信頼性**: 🔵 *cycle_state.py `record_full_eval`・test_training_integration.py より*

- [x] **TC-040a-01**: record_full_eval が best_loss 改善時に stale_cycles=0 にリセット 🔵
  - **信頼性**: 🔵 *test_training_integration.py `TestRecordFullEval.test_improvement_resets_stale` より*

- [x] **TC-040a-02**: record_full_eval が非改善時に stale_cycles を増加 🔵
  - **信頼性**: 🔵 *test_training_integration.py `TestRecordFullEval.test_no_improvement_increments_stale` より*

- [x] **TC-040a-03**: best_loss と同値の場合は非改善として扱う 🔵
  - **信頼性**: 🔵 *test_training_integration.py `TestRecordFullEval.test_exact_best_loss_counts_as_no_improvement` より*

- [x] **TC-040a-04**: 改善時に best_step も更新される 🔵
  - **信頼性**: 🔵 *test_training_integration.py `TestRecordFullEval.test_best_step_updated_on_improvement` より*

- [x] **TC-040a-05**: フル評価サイクルで stale_cycles が二重計上されない 🔵
  - **信頼性**: 🔵 *test_training_integration.py `TestMockedTrainingLoop.test_full_eval_does_not_double_count_stale` より*

- [x] **TC-040a-06**: フル評価改善で stale_cycles がリセットされる 🔵
  - **信頼性**: 🔵 *test_training_integration.py `TestMockedTrainingLoop.test_full_eval_improvement_resets_stale` より*

### REQ-041: DeltaTracker 重み差分統計 🔵

**信頼性**: 🔵 *delta_tracker.py・test_delta_tracker.py より*

### テストケース

#### 正常系

- [x] **TC-041-01**: compute_mean_delta が (after-before)/K を返す 🔵
  - **入力**: before=[1,2,3], after=[6,7,8], K=5
  - **期待結果**: delta=[1,1,1]
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_mean_delta` より*

- [x] **TC-041-02**: per_layer_normがレイヤー別に計算される 🔵
  - **入力**: layers.0 と layers.3 のパラメータ
  - **期待結果**: "layer_0" と "layer_3" のキーが存在
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_stats_per_layer` より*

- [x] **TC-041-03**: レイヤーパターンにマッチしないキーは"other"に分類 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_stats_other_layer_key` より*

- [x] **TC-041-04**: max_component, mean_abs が正しく計算される 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_stats_single_tensor` より*

- [x] **TC-041-05**: compute_and_recordがnorm_historyに記録 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_tracker_compute_and_record` より*

### REQ-042: DeltaTracker 異常検出 🔵

**信頼性**: 🔵 *delta_tracker.py `is_anomalous`・test_delta_tracker.py より*

### テストケース

- [x] **TC-042-01**: 履歴3件未満は常にFalse 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_anomalous_insufficient_history` より*

- [x] **TC-042-02**: 正常範囲のdeltaは非異常 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_anomalous_not_anomalous` より*

- [x] **TC-042-03**: 大幅な外れ値は異常と判定 🔵
  - **入力**: [1.0, 1.0, 1.0, 1.0, 1.0] → 100.0
  - **期待結果**: True
  - **信頼性**: 🔵 *test_delta_tracker.py `test_anomalous_detected` より*

- [x] **TC-042-04**: カスタムsigma閾値で判定が変わる 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_anomalous_custom_threshold` より*

- [x] **TC-042-05**: ゼロ標準偏差時の特別ルール（mean*2.0を閾値） 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_anomalous_zero_std` より*

### REQ-043: DeltaTracker 収束トレンド 🔵

**信頼性**: 🔵 *delta_tracker.py `convergence_trend`・test_delta_tracker.py より*

### テストケース

- [x] **TC-043-01**: データ不足時は0.0 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_convergence_trend_insufficient_data` より*

- [x] **TC-043-02**: 減少系列で負のトレンド 🔵
  - **入力**: [10, 8, 6, 4, 2]
  - **期待結果**: < 0.0
  - **信頼性**: 🔵 *test_delta_tracker.py `test_convergence_trend_decreasing` より*

- [x] **TC-043-03**: 増加系列で正のトレンド 🔵
  - **入力**: [1, 2, 3, 4, 5]
  - **期待結果**: > 0.0
  - **信頼性**: 🔵 *test_delta_tracker.py `test_convergence_trend_increasing` より*

- [x] **TC-043-04**: カスタムwindowで直近のみ評価 🔵
  - **信頼性**: 🔵 *test_delta_tracker.py `test_convergence_trend_window` より*

### REQ-044: 学習ループ統合（CycleState + DeltaTracker） 🔵

**信頼性**: 🔵 *train_tg_lora.py・test_training_integration.py より*

### テストケース

- [x] **TC-044-01**: 1サイクルのrecord_cycleで全状態が整合 🔵
  - **期待結果**: cycle=1, full_backward_passes=K*grad_accum, acceptance_rate正しく
  - **信頼性**: 🔵 *test_training_integration.py `test_single_cycle_recording` より*

- [x] **TC-044-02**: 複数サイクルのreduction_rateが正確 🔵
  - **期待結果**: reduction_rate = 1 − backward/(backward+extrap)
  - **信頼性**: 🔵 *test_training_integration.py `test_multi_cycle_reduction_rate` より*

- [x] **TC-044-03**: should_stopがpatience超過後にTrue 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_early_stopping_via_should_stop` より*

- [x] **TC-044-04**: best_loss改善でstale_cyclesリセット 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_best_loss_tracking_resets_stale` より*

- [x] **TC-044-05**: DeltaTrackerのcompute_and_recordとanomaly/trendが連動 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_delta_tracker_records_and_tracks` より*

- [x] **TC-044-06**: CycleStateとDeltaTrackerのsummaryが結合可能 🔵
  - **期待結果**: merged dict に "cycles", "reduction_rate", "total_norm", "convergence_trend" が含まれる
  - **信頼性**: 🔵 *test_training_integration.py `test_combined_summary_as_in_training_loop` より*

### モック訓練ループ統合テスト 🔵

**信頼性**: 🔵 *test_training_integration.py `TestMockedTrainingLoop` より*

- [x] **TC-044-07**: 10サイクル（フル評価なし）で全状態が整合 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_ten_cycles_no_full_eval` より*

- [x] **TC-044-08**: 受理/拒否混在サイクルで受理率・delta追跡が正確 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_mixed_accept_reject_with_delta_tracking` より*

- [x] **TC-044-09**: フル評価＋early stoppingが patience/mode に基づき正しく動作 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_early_stopping_triggers_after_patience_of_full_evals` より*

- [x] **TC-044-10**: 収束するdelta normでconvergence_trendが負 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_convergence_trend_across_cycles` より*

- [x] **TC-044-11**: 急激なdelta spikeが異常検出でフラグされる 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_anomaly_detection_in_loop` より*

- [x] **TC-044-12**: build_training_summaryが3ソースの全キーを含む 🔵
  - **信頼性**: 🔵 *test_training_integration.py `test_build_training_summary_merges_all` より*

### should_run_full_eval 純粋関数テスト 🔵

**信頼性**: 🔵 *test_training_integration.py `TestShouldRunFullEval` より*

- [x] **TC-044-13**: cycle=0 は常にFalse 🔵
- [x] **TC-044-14**: 倍数サイクルでTrue 🔵
- [x] **TC-044-15**: 非倍数サイクルでFalse 🔵
- [x] **TC-044-16**: full_eval_every=0 で無効 🔵
- [x] **TC-044-17**: full_eval_every<0 で無効 🔵
- [x] **TC-044-18**: full_eval_every=1 で毎サイクルTrue 🔵

### REQ-044 純粋関数テスト（TASK-0012） 🔵

**信頼性**: 🔵 *test_training_pure_functions.py より*

- [x] **TC-044-19**: _compute_pilot_average 正常値の平均を返す 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `TestComputePilotAverage` より*

- [x] **TC-044-20**: _compute_pilot_average 空リストでNaN 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_empty_list_returns_nan` より*

- [x] **TC-044-21**: _compute_pilot_average NaN混入時にNaN 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_nan_in_losses` より*

- [x] **TC-044-22**: _decide_accept_rollback 改善時にaccept 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `TestDecideAcceptRollback` より*

- [x] **TC-044-23**: _decide_accept_rollback 許容範囲内でaccept 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_within_tolerance` より*

- [x] **TC-044-24**: _decide_accept_rollback 悪化時にreject 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_large_degradation` より*

- [x] **TC-044-25**: _decide_accept_rollback ゼロ許容率の挙動 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_zero_tolerance` より*

- [x] **TC-044-26**: _decide_accept_rollback ゼロ付近pilot lossの挙動 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_near_zero_pilot_loss_*` より*

- [x] **TC-044-27**: _format_cycle_progress 受理/拒否フォーマット 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `TestFormatCycleProgress` より*

- [x] **TC-044-28**: _evaluate_full_eval_outcome 改善時にbest更新・staleリセット 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `TestEvaluateFullEvalOutcome` より*

- [x] **TC-044-29**: _evaluate_full_eval_outcome 非改善時にstale増加 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_no_improvement_within_patience` より*

- [x] **TC-044-30**: _evaluate_full_eval_outcome patience超過でearly stop 🔵
  - **信頼性**: 🔵 *test_training_pure_functions.py `test_early_stop_triggered` より*

### REQ-015拡張: baseline学習ループ モックテスト（TASK-0014） 🔵

**信頼性**: 🔵 *test_baseline_training.py より*

- [x] **TC-015-01**: 初期化でseed/tokenizer/model/lora/dataset/metricsが呼ばれる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestInitialization` より*

- [x] **TC-015-02**: forward_backwardがgrad_accum回呼ばれる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestTrainingSteps` より*

- [x] **TC-015-03**: lossがgrad_accumで平均化される 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_loss_is_average_over_grad_accum` より*

- [x] **TC-015-04**: LR schedulerが正しいパラメータで作成される 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestLRScheduler` より*

- [x] **TC-015-05**: max_grad_normがoptimizer_stepに渡される 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestGradientClipping` より*

- [x] **TC-015-06**: eval_interval毎にeval_lossが呼ばれる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestEvalAndBestLoss` より*

- [x] **TC-015-07**: best_loss改善時にモデル保存 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_best_model_saved_on_improvement` より*

- [x] **TC-015-08**: checkpoint_interval毎にチェックポイント保存 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestCheckpointSave` より*

- [x] **TC-015-09**: write_footerとmetrics.closeが呼ばれる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestFinalization` より*

- [x] **TC-015-10**: 単一ステップ・高grad_accumのエッジケース 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `TestEdgeCases` より*

---

## テストケースサマリー

### カテゴリ別件数

| カテゴリ | 正常系 | 異常系 | 境界値 | 回帰 | 合計 |
|---------|--------|--------|--------|------|------|
| 機能要件 | 35 | 0 | 5 | 6 | 46 |
| 非機能要件 | 8 | 0 | 0 | 0 | 8 |
| CycleState | 7 | 0 | 5 | 0 | 12 |
| DeltaTracker | 9 | 0 | 5 | 0 | 14 |
| 統合テスト | 6 | 0 | 0 | 0 | 6 |
| フル評価分離 | 0 | 0 | 6 | 0 | 6 |
| モック訓練ループ | 6 | 0 | 0 | 0 | 6 |
| should_run_full_eval | 0 | 0 | 6 | 0 | 6 |
| 純粋関数（TASK-0012） | 12 | 0 | 0 | 0 | 12 |
| baselineモック（TASK-0014） | 10 | 0 | 0 | 0 | 10 |
| Velocity magnitude | 3 | 0 | 3 | 0 | 6 |
| Data schema | 3 | 3 | 0 | 0 | 6 |
| Velocity anomaly integration | 11 | 0 | 4 | 0 | 15 |
| 適応学習率境界 | 0 | 0 | 3 | 0 | 3 |
| 収束適応 | 2 | 0 | 0 | 0 | 2 |
| 適応LRスモーク | 3 | 0 | 0 | 0 | 3 |
| Phase 9: 10サイクルGPU学習 | 3 | 0 | 1 | 0 | 4 |
| Phase 9: Adaptive LR実験 | 2 | 0 | 2 | 0 | 4 |
| Phase 9: 比較実験妥当性 | 3 | 0 | 2 | 0 | 5 |
| 外挿安全性（REQ-056/057） | 1 | 1 | 1 | 0 | 3 |
| Config dtype検証（REQ-058） | 0 | 2 | 0 | 0 | 2 |
| 外挿安全性統合（REQ-059） | 1 | 0 | 3 | 0 | 4 |
| 回復フロー副作用（REQ-060） | 3 | 0 | 1 | 0 | 4 |
| 回復パスコンポーネント（REQ-059/060補完） | 6 | 0 | 0 | 0 | 6 |
| Phase 14: 設定extra='forbid'（REQ-061） | 0 | 4 | 0 | 0 | 4 |
| Phase 14: 非mapping YAML拒否（REQ-062） | 0 | 2 | 0 | 0 | 2 |
| Phase 14: cap_update安全性（REQ-063） | 1 | 1 | 0 | 0 | 2 |
| Phase 14: スナップショットサニタイズ（REQ-064） | 3 | 0 | 0 | 0 | 3 |
| Phase 14: ロールバックmax_history（REQ-065） | 0 | 0 | 2 | 0 | 2 |
| Phase 14: metrics キー不一致（REQ-066） | 0 | 0 | 3 | 0 | 3 |
| Phase 14: _compute_stats安全性（REQ-067） | 0 | 3 | 0 | 0 | 3 |
| Phase 14: norm_history ガード（REQ-068） | 0 | 2 | 0 | 0 | 2 |
| Phase 16: EvalLossResult統合（REQ-069） | 7 | 0 | 7 | 0 | 14 |
| Phase 16: Config完全性（REQ-070） | 2 | 0 | 0 | 0 | 2 |
| Phase 16: MLflow一貫性（REQ-071） | 6 | 0 | 0 | 0 | 6 |
| Phase 17: Temperature統合（REQ-072） | 5 | 0 | 6 | 0 | 11 |
| Phase 17: Perplexity E2E（REQ-073） | 3 | 0 | 6 | 0 | 9 |
| Phase 17: スタブ補完（REQ-074） | 4 | 0 | 0 | 0 | 4 |
| Phase 19: Perplexity E2E（REQ-069） | 3 | 0 | 2 | 0 | 5 |
| Phase 19: Trainerパリティ（REQ-070） | 1 | 0 | 0 | 0 | 1 |
| Phase 19: プロパティベース（REQ-071） | 0 | 0 | 4 | 0 | 4 |
| Phase 22: 公開API（REQ-073） | 2 | 0 | 0 | 0 | 2 |
| Phase 22: 入力検証（REQ-074） | 0 | 4 | 0 | 0 | 4 |
| Phase 22: 空ローダーNaN（REQ-075） | 0 | 0 | 2 | 0 | 2 |
| Phase 22: ロールバック安全性（REQ-076） | 0 | 2 | 0 | 0 | 2 |
| Phase 22: 非有限loss guard（REQ-077） | 0 | 0 | 2 | 0 | 2 |
| Phase 22: 共有ユーティリティ（REQ-078/079） | 3 | 0 | 0 | 0 | 3 |
| Phase 23: チェックポイント堅牢性（REQ-081~084） | 0 | 0 | 0 | 0 | 0 |
| Phase 24: メトリクスNaN/Infガード（REQ-089/090） | 0 | 0 | 5 | 0 | 5 |
| Phase 24: pilot average NaN/Inf（REQ-091） | 0 | 0 | 2 | 0 | 2 |
| Phase 24: 探索確率伝播（REQ-092/093） | 4 | 0 | 0 | 0 | 4 |
| Phase 24: 純粋関数拡充 | 20 | 0 | 8 | 0 | 28 |
| Phase 24: TG-LoRA特化MLflowメトリクス（REQ-096） | 5 | 0 | 0 | 0 | 5 |
| Phase 24: MLflowリトライロジック（REQ-097） | 4 | 0 | 0 | 0 | 4 |
| Phase 25: Query API（REQ-098） | 8 | 0 | 5 | 0 | 13 |
| Phase 25: ラン比較ダッシュボード（REQ-099） | 8 | 0 | 2 | 0 | 10 |
| Phase 25: 可視化プロット（REQ-100） | 5 | 0 | 6 | 0 | 11 |
| Phase 26: カバレッジ補強（TASK-0065） | 2 | 0 | 3 | 0 | 5 |
| Phase 27: ControllerState serialization（REQ-103） | 2 | 0 | 0 | 0 | 2 |
| Phase 27: CycleState from_dict（REQ-104） | 2 | 0 | 0 | 0 | 2 |
| Phase 27: TrainingState serialization（REQ-105） | 3 | 0 | 0 | 0 | 3 |
| Phase 27: Diagnose script（REQ-106） | 4 | 0 | 0 | 0 | 4 |
| Phase 27: Recovery script（REQ-107） | 4 | 0 | 0 | 0 | 4 |
| Phase 27: CI pipeline（REQ-108） | 1 | 0 | 0 | 0 | 1 |
| Phase 27: API reference（REQ-109） | 1 | 0 | 0 | 0 | 1 |
| **合計** | 271 | 24 | 125 | 6 | 446 |

### 信頼性レベル分布

- 🔵 青信号: 276件 (94%)
- 🟡 黄信号: 4件 (1%)
- 🔴 赤信号: 0件 (0%)
- 未実装: 0件 (0%)

**品質評価**: 高品質 — Phase 27のチェックポイントシリアライズ・運用診断・障害回復・CI パイプラインを含む全フェーズのテストが実装・検証済み

### 優先度別テストケース

- **Must Have**: 60件（コアアルゴリズム・学習ループ・データ・評価・バグ回帰・CycleState・DeltaTracker・統合・外挿安全性・設定安全性・数値信頼性）
- **Should Have**: 46件（安定性・品質管理・GPUパス・コンテキストマネージャ・モックテスト・境界値・Config検証）

---

## テスト実施計画

### Phase 1: 基本機能テスト

- REQ-001 ~ REQ-005（コアアルゴリズム）
- REQ-009, REQ-010（ロールバック）
- 優先度: Must Have
- 実施方法: `make test`

### Phase 1b: 学習ループテスト

- REQ-011 ~ REQ-014（ランダムウォーク）
- REQ-015, REQ-016（学習ループ）
- 優先度: Must Have
- 実施方法: `make test`（test_smoke.py含む）

### Phase 1c: データ・評価テスト

- REQ-025 ~ REQ-031（データパイプライン）
- REQ-032 ~ REQ-037（評価・比較）
- 優先度: Should Have
- 実施方法: `make test`

### Phase 2: カバレッジ完成・コード品質

- NFR-302（run_metrics GPUパス・4テスト）
- REQ-021拡張（load_model モックテスト・25テスト）
- TASK-0010回帰テスト（6バグ修正検証）
- 優先度: Must Have + Should Have
- 実施方法: `pytest tests/test_run_metrics.py tests/test_load_model.py tests/test_extrapolator.py tests/test_lora_state.py tests/test_velocity.py tests/test_eval_modules.py`

### Phase 3: 学習ループテスタビリティ向上

- REQ-044純粋関数（compute_pilot_average, decide_accept_rollback, format_cycle_progress, evaluate_full_eval_outcome・12テスト）
- REQ-015拡張（baseline学習ループモックテスト・35テスト）
- REQ-044統合（モック訓練ループ・フル評価・異常検出・45テスト）
- 優先度: Must Have + Should Have
- 実施方法: `pytest tests/test_training_pure_functions.py tests/test_training_integration.py tests/test_baseline_training.py`

### Phase 4: 設定検証と安全性向上

- REQ-045設定スキーマ検証（BaselineConfig, TGLoRAConfigのPydantic検証・15テスト）
- REQ-046学習開始前バリデーション（データパス存在確認・run_dir書き込み確認・preflight・8テスト）
- REQ-048 CLIエントリポイントモックテスト（main()・argparse・10テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_config_schema.py tests/test_preflight.py tests/test_cli_entry.py`

### Phase 5: Velocity異常検出パイプライン統合テスト

- Velocity anomaly detection pipeline（TestVelocityAnomalyPipelineEndToEnd・11テスト）
- Velocity + DeltaTracker combined anomaly（TestVelocityDeltaTrackerCombinedAnomaly・4テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_velocity_anomaly_integration.py`

### Phase 6: 適応学習率テスト

- REQ-054 lr境界クランプテスト（lr_min/lr_max連続受理拒否・3テスト）
- REQ-053収束適応テスト（adapt_to_convergence・2テスト）
- REQ-013a適応学習率スモークテスト（TestAdaptiveLrSmokeTest・3テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_random_walk_controller.py tests/test_training_integration.py`

### Phase 9: GPU学習検証テスト

- REQ-016拡張: 10サイクルGPU学習完了条件（4テスト）
- REQ-013a拡張: Adaptive LR実験検証条件（4テスト）
- REQ-036検証: 比較実験結果の妥当性条件（5テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_task_0028_ten_cycle_smoke.py tests/test_task_0030_comparison.py`

### Phase 10: 外挿安全性・Config文字列検証

- REQ-056外挿後パラメータ有限性検証（正常1・異常1・境界1 = 3テスト）
- REQ-057trainer間安全性一致（正常1テスト）
- REQ-058 dtype Literal enum検証（異常2テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_config_schema.py tests/test_training_integration.py tests/test_trainer_loop.py`

### Phase 11: 外挿安全性統合テスト

- REQ-059 非有限パラメータ回復フロー統合テスト（正常1・境界3 = 4テスト）
- REQ-060 回復フロー副作用検証（正常3・境界1 = 4テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_extrapolation_safety_integration.py`（新規ファイル）

### Phase 14: 設定安全性・数値信頼性テスト

- REQ-061 設定extra='forbid'（異常4 = 4テスト）
- REQ-062 非mapping YAML拒否（異常2 = 2テスト）
- REQ-063 cap_update安全性（正常1・異常1 = 2テスト）
- REQ-064 スナップショットサニタイズ（正常3 = 3テスト）
- REQ-065 ロールバックmax_history（境界2 = 2テスト）
- REQ-066 metrics キー不一致（境界3 = 3テスト）
- REQ-067 _compute_stats安全性（異常3 = 3テスト）
- REQ-068 norm_history ガード（異常2 = 2テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_config_schema.py tests/test_rollback_manager.py tests/test_delta_tracker.py tests/test_metrics.py tests/test_extrapolation_safety_direct.py`

### Phase 16: 評価メトリクス統一とConfig完全性テスト

- REQ-069 EvalLossResult統合（正常7・境界7 = 14テスト）
- REQ-070 Config完全性（正常2 = 2テスト）
- REQ-071 MLflow一貫性（正常6 = 6テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_eval_loss.py tests/test_config_schema.py tests/test_baseline_training.py`

### Phase 17: テスト品質とエッジケース補強テスト

- REQ-072 Temperature統合（正常5・境界6 = 11テスト）
- REQ-073 Perplexity E2E（正常3・境界6 = 9テスト）
- REQ-074 スタブ補完（正常4 = 4テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_layer_sampler.py tests/test_run_metrics.py tests/test_baseline_training.py`

### Phase 19: Perplexity E2E・Property-Based Testing

- REQ-069 Perplexity E2Eパイプライン検証（正常3・境界2 = 5テスト）
- REQ-070 Trainer間perplexity配管パリティ（正常1 = 1テスト）
- REQ-071 accept()プロパティベーステスト（境界4 = 4テスト、hypothesis依存追加必要）
- 優先度: Must Have（REQ-069/070）+ Should Have（REQ-071）
- 実施方法: `pytest tests/test_run_metrics_e2e.py tests/test_accept_property.py`（新規ファイル想定）

### Phase 24: メトリクスNaN/Infガード・探索確率伝播・純粋関数拡充

- REQ-089 metrics.total_norm() NaN/Infスキップ（境界5 = 5テスト）
- REQ-090 metrics.per_layer_norms() NaN/Infスキップ（境界2 = 2テスト）
- REQ-091 _compute_pilot_average NaN/Infフィルタリング（境界2 = 2テスト）
- REQ-092 探索確率設定→コントローラ伝播（正常4 = 4テスト）
- REQ-093 config-to-controller integration tests（正常8 = 12テスト）
- 純粋関数拡充（28テスト: should_run_full_eval, check_lora_params_finite, _compute_pilot_average, _decide_accept_rollback, _evaluate_full_eval_outcome, _format_cycle_progress, build_training_summary）
- 優先度: Must Have
- 実施方法: `pytest tests/test_metrics.py tests/test_train_tg_lora_pure.py tests/test_random_walk_controller.py`

### Phase 25: 実験分析ツール整備

- REQ-096 TG-LoRA特化MLflowメトリクス送信（正常5 = 5テスト）
- REQ-097 MLflowリトライロジック・指数バックオフ（正常4 = 4テスト）
- REQ-098 RunMetrics JSONLクエリAPI（正常8・境界5 = 13テスト）
- REQ-099 ラン比較ダッシュボード・JSON出力（正常8・境界2 = 10テスト）
- REQ-100 可視化プロット関数（正常5・境界6 = 11テスト）
- 優先度: Must Have
- 実施方法: `pytest tests/test_mlflow_logger.py tests/test_run_query.py tests/test_compare_runs.py`

---

## REQ-053: 収束適応（adapt_to_convergence） 🔵

**信頼性**: 🔵 *random_walk_controller.py `adapt_to_convergence` 実装・test_random_walk_controller.py より*

### Given（前提条件）

- RandomWalkControllerが初期化済み
- DeltaTrackerのconvergence_trend値が与えられている

### When（実行条件）

- `adapt_to_convergence(convergence_trend)` を呼び出す

### Then（期待結果）

- convergence_trend >= 0 かつ total_cycles > 2 の場合: lrを0.8倍に減少、Kを次候補に増加
- convergence_trend < 0（健全な収束）の場合: 変更なし
- total_cycles <= 2 の場合: 変更なし

### テストケース

#### 正常系

- [x] **TC-053-01**: 停滞トレンド（trend >= 0, cycles > 2）でlr減少・K増加 🔵
  - **入力**: convergence_trend=0.5, total_cycles=3
  - **期待結果**: lrが0.8倍に減少、Kが次候補に増加
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_adapt_to_convergence_stalling` より*

- [x] **TC-053-02**: 健全な収束（trend < 0）で変更なし 🔵
  - **入力**: convergence_trend=-0.5
  - **期待結果**: lr, K ともに変更なし
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_adapt_to_convergence_healthy` より*

---

## REQ-054: lr境界クランプ 🔵

**信頼性**: 🔵 *random_walk_controller.py lrクランプロジック・test_random_walk_controller.py より*

### Given（前提条件）

- RandomWalkControllerがlr_initial, lr_min, lr_max, lr_accept_boost, lr_reject_decayで初期化済み

### When（実行条件）

- 連続したreward/penalizeサイクルを実行

### Then（期待結果）

- lrが常に[lr_min, lr_max]の範囲内に留まる
- 連続拒否でlr==lr_minにクランプ
- 連続受理でlr==lr_maxにクランプ

### テストケース

#### 境界値

- [x] **TC-054-01**: 50回連続penalizeでlrがlr_minにクランプ 🔵
  - **入力**: lr_reject_decay=0.5, 50回penalize
  - **期待結果**: lr == lr_min
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_lr_clamps_at_lr_min_under_repeated_rejects` より*

- [x] **TC-054-02**: 50回連続rewardでlrがlr_maxにクランプ 🔵
  - **入力**: lr_accept_boost=1.2, 50回reward
  - **期待結果**: lr == lr_max
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_lr_clamps_at_lr_max_under_repeated_accepts` より*

- [x] **TC-054-03**: 100回交互accept/rejectでlrが常に[lr_min, lr_max]内 🔵
  - **入力**: lr_accept_boost=1.5, lr_reject_decay=0.5, 100回交互
  - **期待結果**: 各サイクルで lr_min <= lr <= lr_max
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_lr_alternating_accept_reject_stays_in_bounds` より*

---

## REQ-013a: 適応学習率スモークテスト 🔵

**信頼性**: 🔵 *test_training_integration.py `TestAdaptiveLrSmokeTest` より*

### Given（前提条件）

- train_tg_loraの設定にlr_initial, lr_min, lr_max, lr_accept_boost, lr_reject_decayが含まれている

### When（実行条件）

- モック化した学習ループを実行

### Then（期待結果）

- 設定値がRandomWalkControllerに正しくワイヤリングされる
- lrがaccept/rejectサイクルで変化する

### テストケース

#### 正常系

- [x] **TC-013a-01**: 設定のadaptive lrパラメータがcontrollerに反映される 🔵
  - **入力**: lr_initial=3e-4, lr_min=5e-6, lr_max=5e-3, lr_accept_boost=1.3, lr_reject_decay=0.4
  - **期待結果**: controllerのlr_min==5e-6, lr_max==5e-3, lr_accept_boost==1.3, lr_reject_decay==0.4
  - **信頼性**: 🔵 *test_training_integration.py `test_adaptive_lr_config_wired_to_controller` より*

- [x] **TC-013a-02**: 1サイクルの学習ループがadaptive lr付きで完了 🔵
  - **入力**: max_cycles=1, lr_initial=5e-4
  - **期待結果**: eval_lossが2回以上呼ばれ、正常完了
  - **信頼性**: 🔵 *test_training_integration.py `test_one_cycle_with_adaptive_lr_completes` より*

- [x] **TC-013a-03**: accept/reject後にlrが変化する 🔵
  - **入力**: 3サイクル（accept→reject→accept）, lr_accept_boost=2.0, lr_reject_decay=0.25
  - **期待結果**: 学習ループが正常完了、lrが適応変化
  - **信頼性**: 🔵 *test_training_integration.py `test_lr_changes_after_accept_reject_in_loop` より*

---

## Phase 9: GPU学習検証テスト

### REQ-016拡張: 10サイクルGPU学習 🔵

**信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py・TASK-0028完了条件より*

### Given（前提条件）

- GPU環境（RTX3060 12GB）が利用可能
- Qwen3.5-9B 4bit QLoRAモデルが読み込み済み
- 学習データがdata/train.jsonlに存在

### When（実行条件）

- TG-LoRA学習を10サイクル実行（max_cycles=10）

### Then（期待結果）

- 10サイクル全てが正常完了する
- run_metrics.jsonlに10サイクル分のメトリクスが記録される
- 損失値がNaN/Infにならない
- velocity cosine similarityが計算・記録されている

### テストケース

#### 正常系

- [x] **TC-016-P9-01**: 10サイクルの学習が正常完了する 🔵
  - **入力**: max_cycles=10, K_initial=3
  - **期待結果**: run_metrics.jsonlに10レコードが記録される
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_ten_cycle_run_metrics_recorded` より*

- [x] **TC-016-P9-02**: 受理/拒否の判定がrollback_tolerance=0.005に基づく 🔵
  - **入力**: rollback_tolerance=0.005, 各サイクルのpilot_lossとextrapolated_loss
  - **期待結果**: loss_after <= pilot_loss * (1 + 0.005) で受理判定
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_accept_reject_based_on_tolerance` より*

- [x] **TC-016-P9-03**: 全サイクルの損失が有限値 🔵
  - **入力**: 10サイクルのrun_metrics.jsonl
  - **期待結果**: 全loss値がNaN/Infでない
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_all_losses_are_finite` より*

#### 境界値

- [x] **TC-016-P9-B01**: velocity cosine similarityが全サイクルで記録される 🔵
  - **入力**: 10サイクルのrun_metrics.jsonl
  - **期待結果**: cosine_sim列が存在し、[0.0, 1.0]の範囲
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_cosine_similarity_recorded` より*

### REQ-013a拡張: Adaptive LR実験検証 🔵

**信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py・9b_tg_lora.yaml lr_reject_decay=0.5より*

### Given（前提条件）

- TG-LoRA学習設定にlr_reject_decay=0.5が設定されている
- 10サイクル学習のメトリクスが記録されている

### When（実行条件）

- run_metrics.jsonlからlr推移を分析

### Then（期待結果）

- 受理時にlrが増加（lr_accept_boost倍）
- 拒否時にlrが0.5倍に減少
- lrが[lr_min=1e-5, lr_max=1e-3]の範囲内に留まる

### テストケース

#### 正常系

- [x] **TC-013a-P9-01**: 受理サイクルでlrが増加する 🔵
  - **入力**: accepted=Trueのサイクルのlr値
  - **期待結果**: 次サイクルのlrが前回より増加（lr_accept_boost倍）
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_reward_increases_lr` より*

- [x] **TC-013a-P9-02**: 拒否サイクルでlrが0.5倍に減少する 🔵
  - **入力**: accepted=Falseのサイクルのlr値
  - **期待結果**: 次サイクルのlrが前回の0.5倍
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_penalize_halves_lr` より*

#### 境界値

- [x] **TC-013a-P9-B01**: 10サイクル全体でlrが[lr_min, lr_max]内に留まる 🔵
  - **入力**: 全10サイクルのlr値
  - **期待結果**: 1e-5 <= lr <= 1e-3
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_lr_stays_in_bounds_throughout_training` より*

- [x] **TC-013a-P9-B02**: 連続拒否でlrがlr_minに到達する 🔵
  - **入力**: 連続penalizeサイクル
  - **期待結果**: lrがlr_minまで減少し、それ以上下回らない
  - **信頼性**: 🔵 *test_task_0028_ten_cycle_smoke.py `test_repeated_penalize_clamps_lr_min` より*

### REQ-036検証: 比較実験結果の妥当性 🔵

**信頼性**: 🔵 *test_task_0030_comparison.py・compare_runs.py既存実装より*

### Given（前提条件）

- ベースラインとTG-LoRAのrun_metrics.jsonlが存在
- 同一backward pass予算での比較設定

### When（実行条件）

- compare_runs.pyで比較レポートを生成

### Then（期待結果）

- 損失曲線プロットが生成される
- 効率メトリクスが計算される
- 受理率・削減率が報告される

### テストケース

#### 正常系

- [x] **TC-036-P9-01**: 比較レポートに必須セクションが含まれる 🔵
  - **入力**: 2つのrun_metrics.jsonl
  - **期待結果**: レポートにSummary, Loss Comparison, Efficiency, Acceptance Rateセクションが含まれる
  - **信頼性**: 🔵 *test_task_0030_comparison.py `test_generate_report_contains_key_sections` より*

- [x] **TC-036-P9-02**: 予算パリティ公式が正しい 🔵
  - **入力**: tg_lora_cycles=10, K_initial=3
  - **期待結果**: baseline_steps = tg_lora_cycles * K_initial = 30
  - **信頼性**: 🔵 *test_task_0030_comparison.py `test_budget_parity_formula` より*

- [x] **TC-036-P9-03**: 効率メトリクス（loss/backward）が計算される 🔵
  - **入力**: 各runの最終loss値とbackward pass数
  - **期待結果**: loss/backward_pass比が正しく計算される
  - **信頼性**: 🔵 *test_task_0030_comparison.py `test_efficiency_metrics_computed` より*

#### 境界値

- [x] **TC-036-P9-B01**: 空メトリクスファイルでもエラーにならない 🔵
  - **入力**: 空のrun_metrics.jsonl
  - **期待結果**: 適切なエラーメッセージまたは空レポート
  - **信頼性**: 🔵 *test_task_0030_comparison.py `test_empty_metrics_handled_gracefully` より*

- [x] **TC-036-P9-B02**: 単一サイクルのメトリクスでも比較可能 🔵
  - **入力**: 1レコードのみのrun_metrics.jsonl
  - **期待結果**: レポートが正常生成される
  - **信頼性**: 🔵 *test_task_0030_comparison.py `test_single_record_comparison` より*

---

## Phase 10: 外挿安全性・Config文字列検証テスト

### REQ-056: 外挿後パラメータ有限性検証 🔵

**信頼性**: 🔵 *REQ-056要件定義・trainer_loop.py NumericalInstabilityErrorパターン・AI_HUB_MAKE_RUN_FEEDBACKより*

### Given（前提条件）

- TG-LoRA学習ループ内で外挿が適用される
- apply_extrapolationが実行される

### When（実行条件）

- apply_extrapolation後にLoRAパラメータがNaNまたはInfを含む

### Then（期待結果）

- 非有限パラメータが検出される
- 外挿が棄却として扱われる
- ロールバックが実行される

### テストケース

#### 正常系

- [x] **TC-056-01**: 外挿後に有限パラメータの場合は正常に受理される 🔵
  - **入力**: 正常なvelocityとalpha、n_steps
  - **期待結果**: 外挿が適用され、accept/rollback判定に進む
  - **信頼性**: 🔵 *test_trainer_loop.py `TestCheckLoraParamsFinite.test_finite_params_return_true` より*

#### 異常系

- [x] **TC-056-E01**: 外挿後にNaNパラメータが検出された場合の自動ロールバック 🔵
  - **入力**: 極端なvelocity値・alpha値の組み合わせでNaN発生を模擬
  - **期待結果**: NaN検出 → rollback_mgr.rollback() → penalize
  - **信頼性**: 🔵 *test_trainer_loop.py `TestCheckLoraParamsFinite.test_nan_params_return_false` より*

#### 境界値

- [x] **TC-056-B01**: 外挿後にInfパラメータが検出された場合の自動ロールバック 🔵
  - **入力**: 非常に大きなvelocity値でInf発生を模擬
  - **期待結果**: Inf検出 → rollback → penalize
  - **信頼性**: 🔵 *test_trainer_loop.py `TestCheckLoraParamsFinite.test_inf_params_return_false` より*

---

### REQ-057: Trainer間数値安全性カバレッジ一致 🔵

**信頼性**: 🔵 *train_baseline_qlora.py/train_tg_lora.py比較・trainer_loop.py共有安全性より*

### Given（前提条件）

- baseline trainerとTG-LoRA trainerの両方が利用可能
- 共通のtrainer_loop.pyが使用される

### When（実行条件）

- 両trainerで同一の数値安全性シナリオが発生

### Then（期待結果）

- 両trainerで同等の安全性動作（NaN/Inf検出、勾配クリッピング、バッチキー検証、学習不可能パラメータ検出）

### テストケース

#### 正常系

- [x] **TC-057-01**: forward_backwardのNumericalInstabilityErrorは両trainerで共有される 🔵
  - **入力**: NaN lossを返すforward_backward
  - **期待結果**: 両trainerでNumericalInstabilityErrorが送出される
  - **信頼性**: 🔵 *trainer_loop.py共有コード・test_trainer_loop.pyより*

---

### REQ-058: dtype/bnb_4bit_compute_dtype Literal enum検証 🔵

**信頼性**: 🔵 *config_schema.py ActiveLayerStrategy/BnbQuantTypeのLiteral enumパターン・REQ-058より*

### Given（前提条件）

- Pydanticスキーマ（config_schema.py）が利用可能
- ModelConfigのdtype, bnb_4bit_compute_dtypeフィールド

### When（実行条件）

- 不正な文字列値を指定してModelConfigを初期化

### Then（期待結果）

- 有効値（"bfloat16", "float16", "float32"）のみ受理される
- 無効値はValidationErrorで拒否される

### テストケース

#### 異常系

- [x] **TC-058-E01**: dtypeに"invalid_dtype"を指定するとValidationError 🔵
  - **入力**: dtype="invalid_dtype"
  - **期待結果**: Pydantic ValidationError
  - **信頼性**: 🔵 *test_config_schema.py `TestDtypeLiteral.test_invalid_dtype_rejected` より*

- [x] **TC-058-E02**: bnb_4bit_compute_dtypeに"test"を指定するとValidationError 🔵
  - **入力**: bnb_4bit_compute_dtype="test"
  - **期待結果**: Pydantic ValidationError
  - **信頼性**: 🔵 *test_config_schema.py `TestDtypeLiteral.test_invalid_bnb_compute_dtype_rejected` より*

---

## Phase 11: 外挿安全性統合テスト

### REQ-059: 非有限パラメータ回復フロー統合テスト 🔵

**信頼性**: 🔵 *train_tg_lora.py 非有限回復パス(332-355行)・AI_HUB_MAKE_RUN_FEEDBACKより*

### Given（前提条件）

- モック化した学習ループ環境
- apply_extrapolation後に非有限パラメータを注入するモックモデル

### When（実行条件）

- モックモデルで学習サイクルを実行し、外挿後にNaN/Infパラメータを発生させる

### Then（期待結果）

- rollback_mgr.rollback()が呼ばれる
- controller.penalize()が正しい引数で呼ばれる
- controller.update_layer_scores()がactive_indicesと-1.0で呼ばれる
- cycle_state.record_cycle()がaccepted=Falseで呼ばれる
- モデルパラメータが外挿前の状態に復元される
- continueにより通常のaccept/rollbackパスがスキップされる

### テストケース

#### 正常系

- [x] **TC-059-01**: 外挿→NaN検出→rollback→penalize→record_cycle の完全フロー 🔵
  - **入力**: モックモデル（apply_extrapolation後にNaN注入）、1サイクル
  - **期待結果**: rollback_mgr.rollback()が1回呼ばれる、controller.penalize()が1回呼ばれる、cycle_state.record_cycle(accepted=False)が1回呼ばれる
  - **信頼性**: 🔵 *train_tg_lora.py 332-355行の実装より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoveryFlow::test_nan_detection_triggers_rollback`

#### 境界値

- [x] **TC-059-B01**: rollback後にモデルパラメータが外挿前の状態に復元される 🔵
  - **入力**: モックモデル（NaN注入後rollback）、snapshot比較
  - **期待結果**: モデルパラメータがsnapshotと一致
  - **信頼性**: 🔵 *rollback_manager.py 復元動作より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoveryFlow::test_rollback_restores_model_params`

- [x] **TC-059-B02**: update_layer_scoresが正しいactive_indicesとスコアで呼ばれる 🔵
  - **入力**: モックモデル（NaN注入）、3層active
  - **期待結果**: update_layer_scores([0,1,2], -1.0)が呼ばれる
  - **信頼性**: 🔵 *train_tg_lora.py 340行より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoveryFlow::test_update_layer_scores_called_with_penalty`

- [x] **TC-059-B03**: 非有限検出後のcontinueで通常のaccept/rollbackパスがスキップされる 🔵
  - **入力**: モックモデル（NaN注入）、1サイクル
  - **期待結果**: eval_loss（外挿後）が呼ばれない、_decide_accept_rollbackが呼ばれない
  - **信頼性**: 🔵 *train_tg_lora.py 355行 continueより*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoveryFlow::test_non_finite_skips_normal_accept_rollback_path`

### REQ-060: 非有限回復フロー副作用検証 🔵

**信頼性**: 🔵 *train_tg_lora.py 非有限回復パス(338-354行)・AI_HUB_MAKE_RUN_FEEDBACKより*

### テストケース

#### 正常系

- [x] **TC-060-01**: penalizeがloss_pilotとinfで呼ばれる 🔵
  - **入力**: NaN注入、loss_pilot=2.5
  - **期待結果**: controller.penalize(2.5, float("inf"))が呼ばれる
  - **信頼性**: 🔵 *train_tg_lora.py 339行より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoverySideEffects::test_penalize_called_with_pilot_loss_and_inf`

- [x] **TC-060-02**: record_cycleの引数が正しい（K, N, grad_accum, accepted=False） 🔵
  - **入力**: NaN注入、K=3, N=5, grad_accum=2
  - **期待結果**: record_cycle(K=3, N=5, grad_accum=2, train_loss=..., valid_loss=None, accepted=False)が呼ばれる
  - **信頼性**: 🔵 *train_tg_lora.py 347-354行より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoverySideEffects::test_record_cycle_with_accepted_false`

- [x] **TC-060-03**: rollback_mgr.pop()がfinallyブロックで呼ばれsnapshotがクリーンアップされる 🔵
  - **入力**: NaN注入、1サイクル
  - **期待結果**: rollback_mgr.pop()が1回呼ばれる
  - **信頼性**: 🔵 *train_tg_lora.py 344-346行より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoverySideEffects::test_rollback_pop_in_finally_after_nan`

#### 境界値

- [x] **TC-060-B01**: 連続NaN発生（2サイクル連続非有限）で両サイクルとも正しく回復 🔵
  - **入力**: NaN注入、2サイクル連続
  - **期待結果**: 各サイクルでrollback→penalize→record_cycleが正しく実行される
  - **信頼性**: 🔵 *train_tg_lora.py 非有限パスのループ内動作より*
  - **テスト**: test_extrapolation_safety_integration.py `TestNonFiniteRecoverySideEffects::test_consecutive_nan_cycles_recover_correctly`

---

## Phase 14: 設定安全性・数値信頼性追加要件

### REQ-061: 設定スキーマ extra='forbid' 🔵

**信頼性**: 🔵 *config_schema.py 全11モデルのextra="forbid"設定・test_config_schema.py TestExtraFieldsRejected より*

### Given（前提条件）

- Pydantic設定スキーマ（config_schema.py）が利用可能
- YAML設定ファイルが存在

### When（実行条件）

- 未知フィールド（タイポ等）を含むYAMLを読み込む

### Then（期待結果）

- 全11モデルで未知フィールドがPydantic ValidationErrorで拒否される

### テストケース

#### 異常系

- [x] **TC-061-E01**: training セクションに "lerning_rate" タイポがある場合 ValidationError 🔵
  - **入力**: training: {"lerning_rate": 5e-4, ...}
  - **期待結果**: ValidationError（"lerning_rate"はTrainingConfigの未知フィールド）
  - **信頼性**: 🔵 *test_config_schema.py `test_typo_in_training_learning_rate` より*

- [x] **TC-061-E02**: model セクションに "name_or_pat" タイポがある場合 ValidationError 🔵
  - **入力**: model: {"name_or_pat": "...", ...}
  - **期待結果**: ValidationError
  - **信頼性**: 🔵 *test_config_schema.py `test_typo_in_model_name` より*

- [x] **TC-061-E03**: tg_lora セクションに "K_init" タイポがある場合 ValidationError 🔵
  - **入力**: tg_lora: {"K_init": 3, ...}
  - **期待結果**: ValidationError
  - **信頼性**: 🔵 *test_config_schema.py `test_tg_lora_extra_field_rejected` より*

- [x] **TC-061-E04**: トップレベルに未知セクション "typo_section" がある場合 ValidationError 🔵
  - **入力**: config に "typo_section" キーを追加
  - **期待結果**: ValidationError
  - **信頼性**: 🔵 *test_config_schema.py `test_unknown_top_level_key_rejected` より*

### REQ-062: 設定読込 非-mapping YAML 拒否 🔵

**信頼性**: 🔵 *config_schema.py load_and_validate_config・test_config_schema.py TestMalformedYAML より*

### テストケース

#### 異常系

- [x] **TC-062-E01**: 空のYAMLファイルを読み込んだ場合 ValueError 🔵
  - **入力**: 空のYAMLファイル
  - **期待結果**: ValueError（"did not resolve to a mapping"）
  - **信頼性**: 🔵 *test_config_schema.py `test_empty_yaml_rejected` より*

- [x] **TC-062-E02**: リスト形式のYAMLファイルを読み込んだ場合 ValueError 🔵
  - **入力**: YAML内容がリスト（例: "- item1"）
  - **期待結果**: ValueError（"did not resolve to a mapping"）
  - **信頼性**: 🔵 *test_config_schema.py `test_list_yaml_rejected` より*

### REQ-063: cap_update 非有限ゼロ返却 🔵

**信頼性**: 🔵 *extrapolator.py cap_update のtorch.isfiniteチェック・test_extrapolation_safety_direct.py より*

### Given（前提条件）

- cap_update()関数が利用可能
- 非有限（Inf）の更新テンソルが入力

### When（実行条件）

- 非有限テンソルでcap_update()を呼び出す

### Then（期待結果）

- ゼロテンソルが返却される（NaN伝播なし）

### テストケース

#### 異常系

- [x] **TC-063-E01**: Infテンソル入力時にゼロテンソルを返却 🔵
  - **入力**: update = torch.tensor([inf, inf])
  - **期待結果**: torch.zeros_like(update)
  - **信頼性**: 🔵 *test_extrapolation_safety_direct.py `test_cap_update_inf_returns_zeros_instead_of_nan` より*

#### 正常系

- [x] **TC-063-01**: 極端パラメータでもapply_extrapolation後に有限値を維持 🔵
  - **入力**: 極端なvelocity値・alpha値
  - **期待結果**: 全パラメータが有限値
  - **信頼性**: 🔵 *test_extrapolation_safety_direct.py `test_extreme_params_remain_finite_via_apply_extrapolation` より*

### REQ-064: ロールバックスナップショット NaN/Inf サニタイズ 🔵

**信頼性**: 🔵 *rollback_manager.py _sanitize_snapshot・test_rollback_manager.py より*

### Given（前提条件）

- RollbackManagerインスタンスが初期化済み
- LoRAパラメータにNaN/Inf値が含まれている

### When（実行条件）

- save()でスナップショットを作成

### Then（期待結果）

- NaN → 0.0、+Inf → 1e6、-Inf → -1e6 にサニタイズされる
- ロールバック時にサニタイズ済みの有限値が復元される

### テストケース

#### 正常系

- [x] **TC-064-01**: NaN値を含むスナップショットが0.0にサニタイズされる 🔵
  - **入力**: LoRAパラメータにNaNを含むモデル
  - **期待結果**: スナップショット内のNaNが0.0に置換
  - **信頼性**: 🔵 *test_rollback_manager.py `test_save_sanitize_nan` より*

- [x] **TC-064-02**: Inf値を含むスナップショットが±1e6にクランプされる 🔵
  - **入力**: LoRAパラメータにInfを含むモデル
  - **期待結果**: +Inf→1e6, -Inf→-1e6
  - **信頼性**: 🔵 *test_rollback_manager.py `test_save_sanitize_inf` より*

- [x] **TC-064-03**: ロールバック後にサニタイズ済みの有限値が復元される 🔵
  - **入力**: NaN→0.0サニタイズ済みスナップショット、ロールバック実行
  - **期待結果**: モデルパラメータが有限値（0.0）
  - **信頼性**: 🔵 *test_rollback_manager.py `test_rollback_restores_sanitized_state` より*

### REQ-065: ロールバック履歴 max_history 制限 🔵

**信頼性**: 🔵 *rollback_manager.py max_history・test_rollback_manager.py より*

### テストケース

#### 境界値

- [x] **TC-065-B01**: max_historyを超過しないことを確認 🔵
  - **入力**: max_history=3, save()を5回呼び出し
  - **期待結果**: len(history) == 3
  - **信頼性**: 🔵 *test_rollback_manager.py `test_max_history_bounds` より*

- [x] **TC-065-B02**: 超過時に最古エントリがFIFOで破棄される 🔵
  - **入力**: max_history=3, save()を4回呼び出し
  - **期待結果**: 最初のエントリが破棄され、最新3件が保持される
  - **信頼性**: 🔵 *test_rollback_manager.py `test_max_history_fifo_eviction` より*

### REQ-066: metrics.cosine_similarity キー不一致安全 🔵

**信頼性**: 🔵 *metrics.py cosine_similarity・test_metrics.py TestCosineSimilarityKeyMismatch より*

### テストケース

#### 境界値

- [x] **TC-066-B01**: bにaのキーが欠落している場合 KeyErrorなし 🔵
  - **入力**: a = {"k1": ..., "k2": ...}, b = {"k1": ...}
  - **期待結果**: k1のみで類似度計算、KeyErrorなし
  - **信頼性**: 🔵 *test_metrics.py `test_b_missing_key_from_a` より*

- [x] **TC-066-B02**: aにbのキーが欠落している場合 KeyErrorなし 🔵
  - **入力**: a = {"k1": ...}, b = {"k1": ..., "k2": ...}
  - **期待結果**: k1のみで類似度計算
  - **信頼性**: 🔵 *test_metrics.py `test_a_missing_key_from_b` より*

- [x] **TC-066-B03**: 完全に不一致のキーセットで0.0を返却 🔵
  - **入力**: a = {"k1": ...}, b = {"k2": ...}
  - **期待結果**: 0.0
  - **信頼性**: 🔵 *test_metrics.py `test_completely_disjoint_keys` より*

### REQ-067: DeltaTracker._compute_stats 非有限スキップ 🔵

**信頼性**: 🔵 *delta_tracker.py _compute_stats・test_delta_tracker.py より*

### テストケース

#### 異常系

- [x] **TC-067-E01**: NaNテンソルをスキップして統計を計算 🔵
  - **入力**: deltaにNaNテンソルを含む
  - **期待結果**: NaNテンソルを除外して統計計算
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_stats_skips_nan_tensor` より*

- [x] **TC-067-E02**: Infテンソルをスキップして統計を計算 🔵
  - **入力**: deltaにInfテンソルを含む
  - **期待結果**: Infテンソルを除外して統計計算
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_stats_skips_inf_tensor` より*

- [x] **TC-067-E03**: 全NaNテンソル入力でゼロstatsを返却 🔵
  - **入力**: 全テンソルがNaN
  - **期待結果**: total_norm=0.0, max_component=0.0, mean_abs=0.0
  - **信頼性**: 🔵 *test_delta_tracker.py `test_compute_stats_all_nan_returns_zeros` より*

### REQ-068: DeltaTracker norm_history 非有限ガード 🔵

**信頼性**: 🔵 *delta_tracker.py compute_and_record・test_delta_tracker.py より*

### テストケース

#### 異常系

- [x] **TC-068-E01**: NaN normをnorm_historyに追加しない 🔵
  - **入力**: compute_and_recordでNaN normを生成するdelta
  - **期待結果**: norm_historyに追加されない
  - **信頼性**: 🔵 *test_delta_tracker.py `test_tracker_nan_norm_not_appended_to_history` より*

- [x] **TC-068-E02**: Inf normをnorm_historyに追加しない 🔵
  - **入力**: compute_and_recordでInf normを生成するdelta
  - **期待結果**: norm_historyに追加されない
  - **信頼性**: 🔵 *test_delta_tracker.py `test_tracker_inf_norm_not_appended_to_history` より*

---

## Phase 16: 評価メトリクス統一とConfig完全性

### REQ-069: EvalLossResult統合 🔵

**信頼性**: 🔵 *TASK-0036: eval_loss.py eval_loss_detailed・test_eval_loss.py より*

### Given（前提条件）

- eval_loss.pyにeval_loss_detailed()関数が定義済み
- EvalLossResult dataclassが利用可能

### When（実行条件）

- eval_loss_detailed()を呼び出す

### Then（期待結果）

- EvalLossResult(avg_loss, perplexity, min_loss, max_loss)が返却される
- perplexity = exp(avg_loss)（有限値の場合）
- avg_loss >= threshold または NaN/Inf の場合、perplexity = inf

### テストケース

#### 正常系

- [x] **TC-069-01**: eval_loss_detailedが正しい平均損失を返す 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_returns_correct_avg` より*

- [x] **TC-069-02**: eval_loss_detailedがperplexityを計算 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_perplexity` より*

- [x] **TC-069-03**: eval_loss_detailedがmin/max損失を返す 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_min_max` より*

- [x] **TC-069-04**: eval_loss_detailedがmax_batchesを尊重 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_respects_max_batches` より*

- [x] **TC-069-05**: eval_loss_detailedが空dataloaderでNaNを返す 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_empty_dataloader` より*

- [x] **TC-069-06**: eval_loss_detailedがtraining modeを維持 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_preserves_training_mode` より*

- [x] **TC-069-07**: EvalLossResult repr表示が正しい 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_repr` より*

#### 境界値

- [x] **TC-069-B01**: perplexityが大loss値でinfになる 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_inf_for_large_loss` より*

- [x] **TC-069-B02**: perplexityがNaN lossでinfになる 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_inf_for_nan_loss` より*

- [x] **TC-069-B03**: perplexityが+Inf lossでinfになる 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_inf_for_pos_inf_loss` より*

- [x] **TC-069-B04**: perplexityが-Inf lossでinfになる 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_inf_for_neg_inf_loss` より*

- [x] **TC-069-B05**: perplexityが境界値100で正しく計算される 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_boundary_at_100` より*

- [x] **TC-069-B06**: perplexityが境界値未満で正しく計算される 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_just_below_threshold` より*

- [x] **TC-069-B07**: perplexityが負のlossで正しく計算される 🔵
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_result_perplexity_negative_loss` より*

### REQ-070: Config完全性・早期停止パラメータ露出 🔵

**信頼性**: 🔵 *TASK-0037: config_schema.py・test_config_schema.py より*

### テストケース

#### 正常系

- [x] **TC-070-01**: save_predictionsフィールドが削除済み 🔵
  - **信頼性**: 🔵 *test_config_schema.py `test_save_predictions_field_removed` より*

- [x] **TC-070-02**: 後方互換でsave_predictionsが無視される 🔵
  - **信頼性**: 🔵 *test_config_schema.py `test_backward_compat_ignores_save_predictions` より*

### REQ-071: MLflowロギング一貫性 🔵

**信頼性**: 🔵 *TASK-0038: test_baseline_training.py TestMLflowParamConsistency より*

### テストケース

#### 正常系

- [x] **TC-071-01**: schedule_typeがMLflowにロギングされる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_schedule_type_logged_to_mlflow` より*

- [x] **TC-071-02**: warmup_stepsがMLflowにロギングされる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_warmup_steps_logged_to_mlflow` より*

- [x] **TC-071-03**: デフォルトschedule_typeがlinear 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_default_schedule_type_is_linear` より*

- [x] **TC-071-04**: best_valid_perplexityがMLflowにロギングされる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_best_valid_perplexity_logged_to_mlflow` より*

- [x] **TC-071-05**: eval無し時のbest_valid_perplexityがinf 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_best_valid_perplexity_inf_when_no_eval` より*

- [x] **TC-071-06**: ベースラインとTG-LoRAで共通パラメータが存在 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_shared_params_present` より*

---

## Phase 17: テスト品質とエッジケース補強

### REQ-072: Layer Sampler temperatureパラメータ統合 🔵

**信頼性**: 🔵 *TASK-0039: test_layer_sampler.py TestTemperatureParameterFlow/DistributionVariation/BoundaryValues より*

### テストケース

#### 正常系

- [x] **TC-072-01**: Configからtemperatureがパースされる 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_config_parses_temperature` より*

- [x] **TC-072-02**: temperatureがselect_active_layersに伝播する 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_temperature_reaches_select_active_layers` より*

- [x] **TC-072-03**: 低temperatureが高スコアに集中させる 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_low_temp_concentrates_on_high_scores` より*

- [x] **TC-072-04**: 高temperatureが分布を拡散させる 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_high_temp_spreads_more` より*

- [x] **TC-072-05**: 中間temperatureが両極端の中間に位置する 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_mid_temp_between_extremes` より*

#### 境界値

- [x] **TC-072-B01**: 極小temperatureでもレイヤー選択が機能する 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_near_zero_temperature_still_selects` より*

- [x] **TC-072-B02**: 極大temperatureでもレイヤー選択が機能する 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_very_large_temperature_selects` より*

- [x] **TC-072-B03**: 極大temperatureがほぼ均一分布に近似する 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_large_temperature_approximates_uniform` より*

- [x] **TC-072-B04**: NaNスコアがハンドリングされる 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_nan_scores_handled` より*

- [x] **TC-072-B05**: Infスコアがハンドリングされる 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_inf_scores_handled` より*

- [x] **TC-072-B06**: -Infスコアがハンドリングされる 🔵
  - **信頼性**: 🔵 *test_layer_sampler.py `test_negative_inf_scores_handled` より*

### REQ-073: RunMetrics perplexity出力とE2E 🔵

**信頼性**: 🔵 *TASK-0040: test_run_metrics.py より*

### テストケース

#### 正常系

- [x] **TC-073-01**: write_footerに正常perplexityが出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_normal` より*

- [x] **TC-073-02**: write_footerにperplexity=Noneで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_none` より*

- [x] **TC-073-03**: EvalLossResult→RunMetricsのE2E伝播が確認される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_e2e_eval_loss_result_to_run_metrics` より*

#### 境界値

- [x] **TC-073-B01**: NaN perplexityで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_nan` より*

- [x] **TC-073-B02**: Inf perplexityで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_inf` より*

- [x] **TC-073-B03**: -Inf perplexityで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_neg_inf` より*

- [x] **TC-073-B04**: 負のperplexityで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_negative` より*

- [x] **TC-073-B05**: ゼロperplexityで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_zero` より*

- [x] **TC-073-B06**: 極大perplexityで"N/A"が出力される 🔵
  - **信頼性**: 🔵 *test_run_metrics.py `test_write_footer_perplexity_very_large` より*

### REQ-074: 空テストスタブ補完とエッジケース 🔵

**信頼性**: 🔵 *TASK-0041: test_baseline_training.py TestEdgeCases より*

### テストケース

#### 正常系

- [x] **TC-074-01**: 単一ステップの学習が正常完了する 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_single_step` より*

- [x] **TC-074-02**: 高grad_accumulationの学習が正常完了する 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_high_grad_accumulation` より*

- [x] **TC-074-03**: evalとcheckpointが同ステップで実行される 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_eval_and_checkpoint_on_same_step` より*

- [x] **TC-074-04**: create_optimizerが正しいパラメータで呼ばれる 🔵
  - **信頼性**: 🔵 *test_baseline_training.py `test_create_optimizer_called_with_correct_params` より*

---

## Phase 19: Perplexity E2E・Property-Based Testing

### REQ-069: Perplexity E2Eパイプライン検証 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「E2E integration test」指摘対応・run_metrics.py _sanitize_perplexity より*

### Given（前提条件）

- モック学習ループ環境（baseline/tg_lora両モード）
- RunMetricsインスタンスが初期化済み

### When（実行条件）

- モック学習ループを実行し、eval_loss_detailedが有限perplexityを返すシナリオと返さないシナリオをそれぞれ実行

### Then（期待結果）

- eval実行時: write_footer出力に有限floatのperplexityが含まれる
- eval未実行時: write_footer出力のperplexityがNone

### テストケース

#### 正常系

- [x] **TC-069-P19-01**: モックbaseline学習でwrite_footerに有限perplexityが含まれる 🔵
  - **入力**: eval_loss_detailedがperplexity=42.5を返すモック、baselineモード
  - **期待結果**: footer JSONLのperplexity == 42.5
  - **信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「asserts the RunMetrics footer contains a finite best_perplexity value」より*

- [x] **TC-069-P19-02**: モックtg_lora学習でwrite_footerに有限perplexityが含まれる 🔵
  - **入力**: eval_loss_detailedがperplexity=38.2を返すモック、tg_loraモード
  - **期待結果**: footer JSONLのperplexity == 38.2
  - **信頼性**: 🔵 *両trainerのperplexityパリティ確認*

- [x] **TC-069-P19-03**: eval未実行時はwrite_footerのperplexityがNone 🔵
  - **入力**: eval_loss_detailedが呼ばれないモック
  - **期待結果**: footer JSONLのperplexity == null
  - **信頼性**: 🔵 *run_metrics.py write_footer(perplexity=None)の既存動作より*

#### 境界値

- [x] **TC-069-P19-B01**: NaN perplexityは_sanitizeでNoneに変換される 🔵
  - **入力**: eval_loss_detailedがperplexity=NaNを返す
  - **期待結果**: footer JSONLのperplexity == null
  - **信頼性**: 🔵 *run_metrics.py _sanitize_perplexity 実装より*

- [x] **TC-069-P19-B02**: Inf perplexityは_sanitizeでNoneに変換される 🔵
  - **入力**: eval_loss_detailedがperplexity=Infを返す
  - **期待結果**: footer JSONLのperplexity == null
  - **信頼性**: 🔵 *run_metrics.py _sanitize_perplexity 実装より*

---

### REQ-070: Trainer間perplexity配管パリティ 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「verify parity」指摘対応・両trainer実装より*

### Given（前提条件）

- train_tg_lora.py と train_baseline_qlora.py の両方が利用可能

### When（実行条件）

- パラメータ化テストで両trainerのperplexityフローを比較

### Then（期待結果）

- 両trainerともeval_result.perplexityをbest_perplexityに保存
- 両trainerともbest_perplexityをwrite_footerに渡す
- 両trainerともMLflowにbest_valid_perplexityをロギング

### テストケース

#### 正常系

- [x] **TC-070-P19-01**: 両trainerでperplexity保存・write_footer渡しのパスが同一 🔵
  - **入力**: パラメータ化テスト（mode="baseline"|"tg_lora"）
  - **期待結果**: 両モードでbest_perplexityが正しく追跡・write_footerに渡される
  - **信頼性**: 🔵 *train_baseline_qlora.py:173, train_tg_lora.py:454 の実装より*

---

### REQ-071: accept()プロパティベーステスト 🟡

**信頼性**: 🟡 *AI_HUB_MAKE_RUN_FEEDBACK「property-based test」指摘対応・hypothesis仮依存*

### Given（前提条件）

- RandomWalkController.accept()が利用可能
- hypothesisがインストール済み

### When（実行条件）

- hypothesis で広範囲の（loss_pilot, loss_after）ペアを生成してaccept()を呼び出す

### Then（期待結果）

- loss_after <= loss_pilot の場合は常に True（大きさに無関係）
- loss_pilot == loss_after の場合は常に True
- NaN/Inf入力の場合は常に False
- relative tolerance 判定が loss_pilot の大きさ（1e-6〜1e6）に対して一貫

### テストケース

#### プロパティベース

- [x] **TC-071-P19-01**: loss_after <= loss_pilot は大きさに関わらず常にTrue 🟡
  - **入力**: hypothesis(st.floats(1e-6, 1e6), st.floats(0, 1.0)) — loss_pilotと改善率
  - **期待結果**: accept(improved_loss, pilot_loss) == True
  - **信頼性**: 🟡 *AI_HUB_MAKE_RUN_FEEDBACK「idempotence across loss-value magnitudes」より*

- [x] **TC-071-P19-02**: NaN/Inf入力は常にFalse 🟡
  - **入力**: hypothesis(st.one_of(st.just(float("nan")), st.just(float("inf")), ...))
  - **期待結果**: accept() == False
  - **信頼性**: 🟡 *accept()のmath.isfiniteガードより*

- [x] **TC-071-P19-03**: 相対許容誤差の大きさ一貫性 🟡
  - **入力**: hypothesis — 同じ改善率で異なる大きさのlossペア
  - **期待結果**: 改善率がrollback_tolerance以下なら常にTrue
  - **信頼性**: 🟡 *accept()相対許容誤差(02582db)の設計意図確認*

- [x] **TC-071-P19-04**: 1e-8 floor近傍での挙動確認 🟡
  - **入力**: hypothesis — loss_pilot ~ 1e-8の近傍値
  - **期待結果**: floorによる判定切替が一貫している
  - **信頼性**: 🟡 *max(abs(loss_pilot), 1e-8)の境界確認*

---

## Phase 22: 公開API・入力検証・エッジケース強化

### REQ-073: 公開APIエクスポート 🔵

**信頼性**: 🔵 *src/tg_lora/__init__.py __all__定義・cceccdeコミットより*

### Given（前提条件）

- tg_loraパッケージがインストール済み

### When（実行条件）

- `from src.tg_lora import Velocity, apply_extrapolation, ...` を実行

### Then（期待結果）

- 全16コンポーネントが正常にインポートされる
- __all__リストに16エントリが含まれる

### テストケース

#### 正常系

- [x] **TC-073-01**: __all__に16のエクスポートが含まれる 🔵
  - **信頼性**: 🔵 *src/tg_lora/__init__.py 確認済み*

- [x] **TC-073-02**: 各エクスポートが正しいモジュールからインポートされる 🔵
  - **信頼性**: 🔵 *src/tg_lora/__init__.py import文確認済み*

---

### REQ-074: RandomWalkController入力検証 🔵

**信頼性**: 🔵 *random_walk_controller.py __init__バリデーション・test_random_walk_controller.py test_reject_* より*

### Given（前提条件）

- RandomWalkControllerのコンストラクタが呼び出される

### When（実行条件）

- 不正なパラメータ値を渡す

### Then（期待結果）

- ValueErrorが発生し、学習開始前に設定エラーを検出

### テストケース

#### 異常系

- [x] **TC-074-E01**: 負のK_candidatesでValueError 🔵
  - **入力**: K_candidates=[-1, 2, 3]
  - **期待結果**: ValueError("All K_candidates must be positive")
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_reject_negative_K_candidates` より*

- [x] **TC-074-E02**: ゼロを含むN_candidatesでValueError 🔵
  - **入力**: N_candidates=[0, 1, 3]
  - **期待結果**: ValueError("All N_candidates must be positive")
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_reject_zero_N_candidates` より*

- [x] **TC-074-E03**: lr_min >= lr_maxでValueError 🔵
  - **入力**: lr_min=1e-3, lr_max=1e-5
  - **期待結果**: ValueError
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_reject_lr_min_ge_lr_max` より*

- [x] **TC-074-E04**: alpha_min >= alpha_maxでValueError 🔵
  - **入力**: alpha_min=0.1, alpha_max=0.01
  - **期待結果**: ValueError
  - **信頼性**: 🔵 *test_random_walk_controller.py `test_reject_alpha_min_ge_alpha_max` より*

---

### REQ-075: 空データローダーNaN返却 🔵

**信頼性**: 🔵 *eval_loss.py count==0/空batch_losses処理・test_eval_loss.py・e19da0fコミットより*

### Given（前提条件）

- 空のデータローダー（バッチ数0）が存在

### When（実行条件）

- eval_loss() または eval_loss_detailed() に空データローダーを渡す

### Then（期待結果）

- eval_loss(): float("nan")を返す（0.0ではない）
- eval_loss_detailed(): avg_loss=NaN, min_loss=NaN, max_loss=NaN, perplexity=inf

### テストケース

#### 境界値

- [x] **TC-075-B01**: eval_loss空データローダーでNaNを返す 🔵
  - **期待結果**: math.isnan(loss) == True
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_empty_dataloader` より*

- [x] **TC-075-B02**: eval_loss_detailed空データローダーでNaN/infを返す 🔵
  - **期待結果**: math.isnan(result.avg_loss), result.perplexity == inf, math.isnan(result.min_loss)
  - **信頼性**: 🔵 *test_eval_loss.py `test_eval_loss_detailed_empty_dataloader` より*

---

### REQ-076: ロールバックtry-catch安全性 🔵

**信頼性**: 🔵 *train_tg_lora.py rollback try-catch・e19da0fコミットより*

### Given（前提条件）

- train_tg_lora学習ループ内でrollback()が呼び出される
- rollback()がRuntimeErrorまたはIndexErrorを送出する可能性がある

### When（実行条件）

- rollback_manager.rollback()が例外を送出する

### Then（期待結果）

- 例外がキャッチされ、エラーログが出力される
- 学習がクラッシュしない

### テストケース

#### 異常系

- [x] **TC-076-E01**: rollback()がRuntimeErrorを送出しても学習継続 🔵
  - **信頼性**: 🔵 *train_tg_lora.py try-catch rollback・e19da0fコミットより*

- [x] **TC-076-E02**: rollback()がIndexErrorを送出しても学習継続 🔵
  - **信頼性**: 🔵 *train_tg_lora.py try-catch rollback・e19da0fコミットより*

---

### REQ-077: 非有限loss_afterガード 🔵

**信頼性**: 🔵 *train_tg_lora.py math.isfinite guard・e19da0fコミットより*

### Given（前提条件）

- 外挿後のeval_lossでloss_afterが計算される

### When（実行条件）

- loss_afterがNaNまたはInf

### Then（期待結果）

- loss_afterがfloat("inf")に設定される
- 受理判定が必ず拒否になる

### テストケース

#### 境界値

- [x] **TC-077-B01**: loss_after=NaNの場合、infに変換され拒否される 🔵
  - **信頼性**: 🔵 *train_tg_lora.py math.isfinite guard・e19da0fコミットより*

- [x] **TC-077-B02**: loss_after=Infの場合、infに変換され拒否される 🔵
  - **信頼性**: 🔵 *train_tg_lora.py math.isfinite guard・e19da0fコミットより*

---

### REQ-078: 共有InfiniteBatchIterator 🔵

**信頼性**: 🔵 *src/training/batch_iter.py・a020e5bコミットより*

### テストケース

- [x] **TC-078-01**: 空データセットでValueError 🔵
  - **信頼性**: 🔵 *test_infinite_batch_iterator.py より*

- [x] **TC-078-02**: 無限イテレーションが動作する 🔵
  - **信頼性**: 🔵 *test_infinite_batch_iterator.py より*

---

### REQ-079: 共有save_checkpoint 🔵

**信頼性**: 🔵 *src/utils/checkpoint.py・a020e5bコミットより*

### テストケース

- [x] **TC-079-01**: ディレクトリが存在しない場合に自動作成 🔵
  - **信頼性**: 🔵 *checkpoint.py mkdir(parents=True, exist_ok=True) より*

---

### TC-081〜084: 実装済みテスト 🔵

- [x] **TC-081-01**: save_checkpoint readback検証 🔵
  - **期待結果**: 保存後にディレクトリ内容を確認し、不完全なチェックポイントを検出
  - **テスト**: test_checkpoint.py `TestSaveCheckpointReadbackVerification`
  - **信頼性**: 🔵 *checkpoint.py に readback verification 実装済み、テストで警告ログを検証*

- [x] **TC-082-01**: InfiniteBatchIterator単一バッチエッジケース 🔵
  - **期待結果**: 1バッチのみのデータローダーで正しくループ
  - **テスト**: test_infinite_batch_iterator.py `TestSingleBatchDataloader`
  - **信頼性**: 🔵 *単一バッチの無限反復・StopIteration リセットを検証*

- [x] **TC-082-02**: InfiniteBatchIteratorデバイスキャスト 🔵
  - **期待結果**: テンソルが指定デバイスに移動される
  - **テスト**: test_infinite_batch_iterator.py `TestDeviceCastEdgeCases`
  - **信頼性**: 🔵 *str/device オブジェクト・マルチキー・dtype 保持を検証*

- [x] **TC-083-01**: 非有限lossガード発動時の警告ログ 🔵
  - **期待結果**: logger.warningで通知
  - **テスト**: test_training_integration.py `TestNonFiniteLossAfterWarning`
  - **信頼性**: 🔵 *NaN/Inf loss_after の警告ログ・有限値の非警告を検証*

- [x] **TC-084-01**: ロールバック失敗E2Eテスト 🔵
  - **期待結果**: モックでrollback()をraiseさせ、学習継続または安全な失敗を検証
  - **テスト**: test_training_integration.py `TestRollbackFailureResilience`
  - **信頼性**: 🔵 *RuntimeError/IndexError 発生時のエラーログと学習継続を検証*

---

### TC-085〜088: 探索確率パラメータテスト 🔵

- [x] **TC-085-01**: 探索確率パラメータのデフォルト値 🔵
  - **期待結果**: None指定時にクラス定数デフォルト値が使用される
  - **テスト**: test_random_walk_controller.py `test_explore_prob_defaults`
  - **信頼性**: 🔵 *random_walk_controller.py _DEFAULT_*_PROB クラス定数との一致を検証*

- [x] **TC-085-02**: カスタム探索確率値 🔵
  - **期待結果**: コンストラクタで指定した値が設定される
  - **テスト**: test_random_walk_controller.py `test_explore_prob_custom_values`
  - **信頼性**: 🔵 *4パラメータのカスタム値設定を検証*

- [x] **TC-086-01**: k_explore_prob=0.0でK不変 🔵
  - **期待結果**: 200回propose()してもKが変化しない
  - **テスト**: test_random_walk_controller.py `test_zero_k_explore_prob_never_changes_K`
  - **信頼性**: 🔵 *確率0でrandom.random() < 0.0が常にFalseとなることを検証*

- [x] **TC-086-02**: k_explore_prob=1.0でK常変化 🔵
  - **期待結果**: 100回propose()でKが常に初期値と異なる
  - **テスト**: test_random_walk_controller.py `test_full_k_explore_prob_always_changes_K`
  - **信頼性**: 🔵 *非端インデックスでの隣接候補移動を検証*

- [x] **TC-087-01**: N/beta/strategyの極端値テスト 🔵
  - **期待結果**: prob=0.0で変化なし、prob=1.0で常変化
  - **テスト**: test_random_walk_controller.py `test_zero/full_n/beta/strategy_*`
  - **信頼性**: 🔵 *N（8テスト）、beta（2テスト）、strategy（2テスト）の極端値動作を検証*

- [x] **TC-088-01**: TGLoRAParams探索確率スキーマ検証 🔵
  - **期待結果**: デフォルト値が正しい。0.0と1.0がValueErrorで拒否される
  - **テスト**: test_random_walk_controller.py `test_explore_prob_config_schema_validation`
  - **信頼性**: 🔵 *Pydantic Field(gt=0.0, lt=1.0)制約の有効性を検証*

---

## Phase 27: チェックポイントシリアライズ・運用診断・CI パイプライン

### REQ-103: ControllerState summary()/from_dict() 🔵

**信頼性**: 🔵 *random_walk_controller.py ControllerState・4435fdeコミットより*

### Given（前提条件）

- RandomWalkControllerインスタンスが複数サイクル実行済み
- ControllerStateの内部状態（K, N, alpha, beta, lr, strategy, layer_scores等）が設定されている

### When（実行条件）

- ControllerState.summary()でシリアライズし、from_dict()で復元する

### Then（期待結果）

- summary()が全状態を含むdictを返す
- from_dict()で完全に復元可能な往復ラウンドトリップが成功する

### テストケース

#### 正常系

- [x] **TC-103-01**: summary()/from_dict()の完全往復が一致する 🔵
  - **信頼性**: 🔵 *test_random_walk_controller.py TestControllerStateSerialization より*

- [x] **TC-103-02**: デフォルト値からのsummary()が全キーを含む 🔵
  - **信頼性**: 🔵 *test_random_walk_controller.py より*

### REQ-104: CycleState from_dict() 🔵

**信頼性**: 🔵 *cycle_state.py from_dict()・04a7581コミットより*

### Given（前提条件）

- CycleState.summary()の出力dictが利用可能

### When（実行条件）

- CycleState.from_dict()でdictからCycleStateを再構築する

### Then（期待結果）

- 全プロパティ（cycles, backward_passes, extrapolation_steps等）が正しく復元される
- 空辞書・部分データの場合はデフォルト値で初期化される

### テストケース

#### 正常系

- [x] **TC-104-01**: summary()→from_dict()の往復で全プロパティが一致する 🔵
  - **信頼性**: 🔵 *test_cycle_state.py TestCycleStateFromDict より*

- [x] **TC-104-02**: 空辞書からfrom_dict()でデフォルト値が設定される 🔵
  - **信頼性**: 🔵 *test_cycle_state.py より*

### REQ-105: TrainingState シリアライズ・デシリアライズ 🔵

**信頼性**: 🔵 *checkpoint.py TrainingState・57739faコミットより*

### Given（前提条件）

- CycleState, ControllerState, Velocity, DeltaTrackerの状態が利用可能
- TrainingState dataclassが初期化済み

### When（実行条件）

- save_training_state()でディスクに保存し、load_training_state()で復元する

### Then（期待結果）

- PyTorch tensorがCPU変換されて保存される
- 全コンポーネントがfrom_dict()または直接代入で正しく再構築される
- Velocity/DeltaTrackerの履歴が正しく復元される

### テストケース

#### 正常系

- [x] **TC-105-01**: save/load_training_state()の完全往復が一致する 🔵
  - **信頼性**: 🔵 *test_fault_recovery.py TestTrainingStateSaveLoad より*

- [x] **TC-105-02**: 個別コンポーネントからTrainingStateを構築できる 🔵
  - **信頼性**: 🔵 *test_fault_recovery.py TestTrainingStateFromComponents より*

- [x] **TC-105-03**: Velocity/DeltaTrackerの空状態エッジケース 🔵
  - **信頼性**: 🔵 *test_fault_recovery.py TestTrainingStateVelocityEdgeCases より*

### REQ-106: 運用診断スクリプト (diagnose.py) 🔵

**信頼性**: 🔵 *scripts/diagnose.py・b942b4bコミット・test_diagnose.py より*

### Given（前提条件）

- diagnose.pyスクリプトが利用可能
- チェック対象（GPU, チェックポイント, 設定, ログ）が存在

### When（実行条件）

- check_gpu(), check_checkpoint(), check_config(), check_logs()を実行する

### Then（期待結果）

- 各チェックがCheckResult(status: ok/warn/error)を返す
- run_all_checks()で全結果が集約される
- --json出力とpretty-print出力の両方が可能

### テストケース

#### 正常系

- [x] **TC-106-01**: check_gpu()がCheckResultを返す 🔵
  - **信頼性**: 🔵 *test_diagnose.py より*

- [x] **TC-106-02**: check_config()が値域チェックを実行する 🔵
  - **信頼性**: 🔵 *test_diagnose.py より*

- [x] **TC-106-03**: check_logs()がエラーパターンを検出する 🔵
  - **信頼性**: 🔵 *test_diagnose.py より*

- [x] **TC-106-04**: run_all_checks()が全チェック結果を集約する 🔵
  - **信頼性**: 🔵 *test_diagnose.py より*

### REQ-107: 障害回復スクリプト (recover.py) 🔵

**信頼性**: 🔵 *scripts/recover.py・66987e4コミット・test_recover.py より*

### Given（前提条件）

- recover.pyスクリプトが利用可能
- 障害ログ・チェックポイント・設定ファイルが存在

### When（実行条件）

- analyze_fault(), sanitize_checkpoint(), generate_recovery_config(), apply_remediation()を実行する

### Then（期待結果）

- OOM/CUDA/NaN/Instability障害タイプが正しく分析される
- チェックポイント内のNaN/Infがサニタイズされる
- 復旧設定が推奨範囲内に調整される

### テストケース

#### 正常系

- [x] **TC-107-01**: analyze_fault()がOOM障害を検出する 🔵
  - **信頼性**: 🔵 *test_recover.py より*

- [x] **TC-107-02**: sanitize_checkpoint()がNaN/Infを修正する 🔵
  - **信頼性**: 🔵 *test_recover.py より*

- [x] **TC-107-03**: generate_recovery_config()が安全な値に調整する 🔵
  - **信頼性**: 🔵 *test_recover.py より*

- [x] **TC-107-04**: apply_remediation()が完全自動回復を実行する 🔵
  - **信頼性**: 🔵 *test_recover.py より*

### REQ-108: CI パイプライン (make ci) 🔵

**信頼性**: 🔵 *Makefile ci target・test_fault_recovery.py より*

### Given（前提条件）

- Makefileにciターゲットが定義されている
- ruff, pytestがインストール済み

### When（実行条件）

- `make ci` を実行する

### Then（期待結果）

- ruff check + format check + pytest + スクリプトインポート健全性チェックが順次実行される
- いずれかのステップが失敗した場合、CIがFAILになる

### テストケース

#### 正常系

- [x] **TC-108-01**: diagnose.py/recover.pyのインポートが正常に完了する 🔵
  - **信頼性**: 🔵 *test_fault_recovery.py test_diagnose_recover_imports より*

### REQ-109: APIリファレンス完全性 🔵

**信頼性**: 🔵 *docs/api_reference.md・b942b4bコミットより*

### Given（前提条件）

- docs/api_reference.mdが存在する

### When（実行条件）

- APIリファレンスの内容を確認する

### Then（期待結果）

- 全34エクスポート関数・クラスのシグネチャ・パラメータ・使用例が文書化されている

### テストケース

#### 正常系

- [x] **TC-109-01**: APIリファレンスに全エクスポートが記載されている 🔵
  - **信頼性**: 🔵 *docs/api_reference.md 既存610行確認済み*

---

### Phase 27 テストケースサマリー追加

| カテゴリ | 正常系 | 異常系 | 境界値 | 回帰 | 合計 |
|---------|--------|--------|--------|------|------|
| Phase 27: ControllerState serialization（REQ-103） | 2 | 0 | 0 | 0 | 2 |
| Phase 27: CycleState from_dict（REQ-104） | 2 | 0 | 0 | 0 | 2 |
| Phase 27: TrainingState serialization（REQ-105） | 3 | 0 | 0 | 0 | 3 |
| Phase 27: Diagnose script（REQ-106） | 4 | 0 | 0 | 0 | 4 |
| Phase 27: Recovery script（REQ-107） | 4 | 0 | 0 | 0 | 4 |
| Phase 27: CI pipeline（REQ-108） | 1 | 0 | 0 | 0 | 1 |
| Phase 27: API reference（REQ-109） | 1 | 0 | 0 | 0 | 1 |

---

## Phase 29: OptimizerLifecycleManager・キャッシュメトリクス追跡

### REQ-119: OptimizerLifecycleManager ライフサイクル管理 🔵

**信頼性**: 🔵 *src/training/optimizer_lifecycle.py・3fdf57a/d2e2a51コミット・test_optimizer_lifecycle.py より*

### Given（前提条件）

- OptimizerLifecycleManagerが初期化済み
- モデルがロード済み

### When（実行条件）

- prepare_for_cycle()を異なるlrで複数回呼び出す

### Then（期待結果）

- recreate_per_cycle: 呼び出しごとに新しいoptimizerインスタンスが返され、stateが空
- reuse_state_reset_experimental: 同一optimizerインスタンスが返され、state tensorがin-place zero-resetされる

### テストケース

#### 正常系

- [x] **TC-119-01**: recreate_per_cycleで毎回新しいoptimizerが返される 🔵
  - **入力**: policy="recreate_per_cycle", lr=1e-3→2e-3
  - **期待結果**: opt1 is not opt2, opt2.stateが空, opt2.param_groups[0]["lr"]==2e-3
  - **信頼性**: 🔵 *test_optimizer_lifecycle.py `test_recreate_policy_returns_new_optimizer` より*

- [x] **TC-119-02**: reuse_state_reset_experimentalでstateがin-place zero-resetされる 🔵
  - **入力**: policy="reuse_state_reset_experimental", materialize state→prepare_for_cycle(2e-3)
  - **期待結果**: opt1 is opt2, state tensor data_ptr不変, 全要素がゼロ
  - **信頼性**: 🔵 *test_optimizer_lifecycle.py `test_reuse_policy_zeros_state_in_place` より*

- [x] **TC-119-03**: reuse_policyが統合テストでoptimizerを1回のみ生成する 🔵
  - **入力**: TestOptimizerLifecycleIntegration, 2サイクル
  - **期待結果**: create_optimizerが1回のみ呼ばれる
  - **信頼性**: 🔵 *test_training_integration.py TestOptimizerLifecycleIntegration より*

### REQ-120: TrainingConfig optimizer_lifecycle設定 🔵

**信頼性**: 🔵 *src/training/config_schema.py TrainingConfig.optimizer_lifecycle・d69a57dコミットより*

### テストケース

#### 正常系

- [x] **TC-120-01**: デフォルト値が"recreate_per_cycle" 🔵
  - **期待結果**: TrainingConfig().optimizer_lifecycle == "recreate_per_cycle"
  - **信頼性**: 🔵 *config_schema.py TrainingConfig定義より*

### REQ-121: RunMetrics header optimizer_lifecycle出力 🔵

**信頼性**: 🔵 *src/utils/run_metrics.py write_header・2ac68d1コミット・test_run_metrics.py より*

### テストケース

#### 正常系

- [x] **TC-121-01**: write_headerにoptimizer_lifecycleが含まれる 🔵
  - **期待結果**: header JSONLにoptimizer_lifecycleキーが存在
  - **信頼性**: 🔵 *test_run_metrics.py TestRunMetricsHeader・2ac68d1コミットより*

- [x] **TC-121-02**: optimizer_lifecycleフィールド不在時にNoneが出力される 🔵
  - **期待結果**: getattr(cfg.training, "optimizer_lifecycle", None) == None
  - **信頼性**: 🔵 *run_metrics.py getattr使用・2ac68d1コミットより*

### REQ-122: ActivationCache hit/miss追跡メトリクス 🔵

**信頼性**: 🔵 *src/training/train_tg_lora.py activation_cache_*_count・src/utils/run_metrics.py record_step・f45a269コミットより*

### テストケース

#### 正常系

- [x] **TC-122-01**: RunMetrics.record_stepにcache fieldsが含まれる 🔵
  - **期待結果**: record JSONLにtg_lora_cache_built, tg_lora_cache_eligible, tg_lora_cache_hitが存在
  - **信頼性**: 🔵 *run_metrics.py record_step定義・f45a269コミットより*

#### 境界値

- [x] **TC-122-B01**: eligible_count=0でhit_rate=0.0（ゼロ除算回避） 🔵
  - **期待結果**: activation_cache_hit_rate == 0.0
  - **信頼性**: 🔵 *train_tg_lora.py if activation_cache_eligible_count > 0 else 0.0 より*

### REQ-123: ベンチマークスクリプト 🔵

**信頼性**: 🔵 *scripts/benchmark_optimizer_lifecycle.py・d69a57dコミットより*

### テストケース

#### 正常系

- [x] **TC-123-01**: ベンチマークスクリプトが両policyの比較JSONを出力する 🔵
  - **期待結果**: JSONにrecreate_per_cycle, reuse_state_reset_experimental, deltaセクションが含まれる
  - **信頼性**: 🔵 *benchmark_optimizer_lifecycle.py _round_record(comparison) より*

### REQ-124: 実験用optimizer再利用設定サーフェス 🔵

**信頼性**: 🔵 *configs/9b_tg_lora_optimizer_reuse_experimental.yaml・d69a57dコミットより*

### テストケース

#### 正常系

- [x] **TC-124-01**: 実験設定にoptimizer_lifecycle=reuse_state_reset_experimentalが含まれる 🔵
  - **期待結果**: cfg.training.optimizer_lifecycle == "reuse_state_reset_experimental"
  - **信頼性**: 🔵 *9b_tg_lora_optimizer_reuse_experimental.yaml line 34 より*

- [x] **TC-124-02**: enable_random_walk=falseで決定論的設定 🔵
  - **期待結果**: cfg.tg_lora.enable_random_walk == False
  - **信頼性**: 🔵 *9b_tg_lora_optimizer_reuse_experimental.yaml line 72 より*

---

### Phase 29 テストケースサマリー追加

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 29: OptimizerLifecycleManager（REQ-119） | 3 | 0 | 0 | 3 |
| Phase 29: TrainingConfig optimizer_lifecycle（REQ-120） | 1 | 0 | 0 | 1 |
| Phase 29: RunMetrics header（REQ-121） | 2 | 0 | 0 | 2 |
| Phase 29: Cache hit/miss metrics（REQ-122） | 1 | 0 | 1 | 2 |
| Phase 29: Benchmark script（REQ-123） | 1 | 0 | 0 | 1 |
| Phase 29: Experimental config（REQ-124） | 2 | 0 | 0 | 2 |

---

## Phase 32: prefix_feature_cache堅牢性テスト・compare-prefix smoke test

### REQ-128: 破損キャッシュファイルハンドリング 🔵

**信頼性**: 🔵 *prefix_feature_cache.py load_prefix_feature_dataset・design-interview A27「corrupted cache file handling」推奨より*

### Given（前提条件）

- ディスク上にキャッシュファイル（.pt）が存在する
- キャッシュファイルが部分的に書き込まれている、または不正フォーマット

### When（実行条件）

- load_prefix_feature_dataset()で破損キャッシュファイルを読み込む

### Then（期待結果）

- ValueErrorまたはRuntimeErrorが送出される
- エラーメッセージにキャッシュパスが含まれる

### テストケース

#### 異常系

- [x] **TC-128-E01**: 部分書き込みファイル（1バイトのみ）の読み込みでエラー 🔵
  - **入力**: 1バイトのファイル
  - **期待結果**: 例外送出（torch.load失敗）
  - **信頼性**: 🔵 *torch.loadの不正ファイル動作より*

- [x] **TC-128-E02**: 不正フォーマット（非dict）の読み込みでエラー 🔵
  - **入力**: torch.saveでテンソル単体を保存したファイル
  - **期待結果**: KeyError/TypeErrorで失敗
  - **信頼性**: 🔵 *load_prefix_feature_datasetがblob["hidden_states"]等のキーアクセスを前提とする*

- [x] **TC-128-E03**: 欠落キー（hidden_statesなし）の読み込みでエラー 🔵
  - **入力**: {"format_version": 1, "metadata": {}} のみを含む.ptファイル
  - **期待結果**: KeyError
  - **信頼性**: 🔵 *load_prefix_feature_datasetのキーアクセスより*

---

### REQ-129: force_rebuildフラグ動作 🔵

**信頼性**: 🔵 *train_tg_lora.py _maybe_cache_dataset force_rebuild分岐・config_schema.py TrainingConfig.prefix_feature_cache_force_rebuildより*

### Given（前提条件）

- ディスク上に有効なキャッシュファイルが存在する
- prefix_feature_cache_force_rebuild設定が指定されている

### When（実行条件）

- _maybe_cache_dataset()をforce_rebuild=trueとfalseでそれぞれ呼び出す

### Then（期待結果）

- force_rebuild=false: 既存ディスクキャッシュをロード（source="disk"）
- force_rebuild=true: ディスクキャッシュをスキップして再ビルド（source="built"）

### テストケース

#### 正常系

- [x] **TC-129-01**: force_rebuild=falseで既存ディスクキャッシュが再利用される 🔵
  - **入力**: 有効なキャッシュファイルが存在、force_rebuild=false
  - **期待結果**: load_prefix_feature_datasetが呼ばれ、buildがスキップされる
  - **信頼性**: 🔵 *_maybe_cache_dataset `cache_path.exists() and not prefix_feature_cache_force_rebuild` 分岐より*

- [x] **TC-129-02**: force_rebuild=trueでディスクキャッシュがスキップされる 🔵
  - **入力**: 有効なキャッシュファイルが存在、force_rebuild=true
  - **期待結果**: build_prefix_feature_datasetが呼ばれ、既存ファイルが無視される
  - **信頼性**: 🔵 *_maybe_cache_dataset force_rebuild分岐より*

---

### REQ-130: position_idsビルドパス 🔵

**信頼性**: 🔵 *prefix_feature_cache.py build_prefix_feature_dataset batch.get("position_ids")・design-interview A27推奨より*

### Given（前提条件）

- データセットバッチにposition_idsが含まれている

### When（実行条件）

- build_prefix_feature_dataset()をposition_ids付きデータセットで呼び出す

### Then（期待結果）

- 各PrefixFeatureExampleのposition_idsが正しく保存される
- collate_prefix_feature_batch()でposition_idsがバッチ化される

### テストケース

#### 正常系

- [x] **TC-130-01**: position_ids付きデータセットでビルドが正しく動作する 🔵
  - **入力**: position_idsを含む_TokenDataset、split_layer_idx=2
  - **期待結果**: examples[i].position_idsが非None、期待値と一致
  - **信頼性**: 🔵 *build_prefix_feature_dataset `position_batch = batch.get("position_ids")` 分岐より*

- [x] **TC-130-02**: position_ids付きキャッシュのsave/load往復が正しい 🔵
  - **入力**: position_ids付きPrefixFeatureDataset、save→load
  - **期待結果**: 再読込後のposition_idsが元と一致
  - **信頼性**: 🔵 *save_prefix_feature_dataset `has_position_ids` 分岐・load_prefix_feature_dataset `position_ids[idx].clone()` より*

---

### REQ-131: model.training状態復元 🔵

**信頼性**: 🔵 *prefix_feature_cache.py build_prefix_feature_dataset try/finally・design-interview A27推奨より*

### Given（前提条件）

- model.training = Trueの状態でbuildを開始

### When（実行条件）

- ビルド中に例外が発生（モデルフォワードでエラー）

### Then（期待結果）

- finallyブロックでhook.remove()が実行される
- model.trainingがTrueに復元される

### テストケース

#### 異常系

- [x] **TC-131-E01**: ビルド中の例外発生時にmodel.trainingが復元される 🔵
  - **入力**: forward()でRuntimeErrorを送出するモックモデル、model.train()で初期化
  - **期待結果**: 例外送出後もmodel.training == True
  - **信頼性**: 🔵 *build_prefix_feature_dataset try/finally `if was_training: model.train()` より*

- [x] **TC-131-E02**: 正常終了時にmodel.trainingが復元される 🔵
  - **入力**: model.train()で初期化、正常ビルド
  - **期待結果**: ビルド完了後 model.training == True
  - **信頼性**: 🔵 *build_prefix_feature_dataset `model.eval()` → finally `model.train()` より*

---

### REQ-132: SHA-256キャッシュ無効化 🔵

**信頼性**: 🔵 *prefix_feature_cache.py get_prefix_feature_cache_path・build_prefix_feature_cache_metadata より*

### Given（前提条件）

- 2つの異なるメタデータセットが存在

### When（実行条件）

- 異なるハイパーパラメータでget_prefix_feature_cache_path()を呼び出す

### Then（期待結果）

- 異なるパラメータで異なるキャッシュパスが生成される
- 同一パラメータで同じパスが生成される

### テストケース

#### 正常系

- [x] **TC-132-01**: 異なるハイパーパラメータで異なるキャッシュパスが生成される 🔵
  - **入力**: metadata(seed=42) vs metadata(seed=43)
  - **期待結果**: キャッシュパスが異なる
  - **信頼性**: 🔵 *get_prefix_feature_cache_path SHA-256 digestより*

- [x] **TC-132-02**: 同一パラメータで同じキャッシュパスが生成される 🔵
  - **入力**: 同一metadataで2回呼び出し
  - **期待結果**: パスが完全一致
  - **信頼性**: 🔵 *SHA-256の決定性より*

- [x] **TC-132-03**: lora_r変更でパスが変わる 🔵
  - **入力**: lora_r=16 vs lora_r=32
  - **期待結果**: キャッシュパスが異なる
  - **信頼性**: 🔵 *build_prefix_feature_cache_metadataにlora_rが含まれる*

---

### REQ-133: format_version不一致 🔵

**信頼性**: 🔵 *prefix_feature_cache.py load_prefix_feature_dataset format_versionチェックより*

### Given（前提条件）

- キャッシュファイルのformat_versionが現在のバージョンと異なる

### When（実行条件）

- load_prefix_feature_dataset()で旧フォーマットキャッシュを読み込む

### Then（期待結果）

- ValueErrorが送出される
- エラーメッセージに検出されたバージョン番号が含まれる

### テストケース

#### 異常系

- [x] **TC-133-E01**: format_version=0のキャッシュでValueError 🔵
  - **入力**: format_version=0を含む.ptファイル
  - **期待結果**: ValueError("Unsupported prefix feature cache format version: 0")
  - **信頼性**: 🔵 *load_prefix_feature_dataset `if blob.get("format_version") != _PREFIX_FEATURE_CACHE_FORMAT_VERSION` より*

- [x] **TC-133-E02**: format_versionキーなしでValueError 🔵
  - **入力**: format_versionキーなしの.ptファイル
  - **期待結果**: ValueError("Unsupported prefix feature cache format version: None")
  - **信頼性**: 🔵 *blob.get("format_version") が None を返す場合の挙動より*

---

### REQ-134: 空データセット拒否 🔵

**信頼性**: 🔵 *prefix_feature_cache.py save_prefix_feature_dataset 空チェックより*

### Given（前提条件）

- 空のPrefixFeatureDataset（examples=[]）

### When（実行条件）

- save_prefix_feature_dataset()を空データセットで呼び出す

### Then（期待結果）

- ValueError("Cannot persist an empty PrefixFeatureDataset")が送出される
- ディスクにファイルが作成されない

### テストケース

#### 異常系

- [x] **TC-134-E01**: 空データセットでValueError 🔵
  - **入力**: PrefixFeatureDataset(examples=[])
  - **期待結果**: ValueError、ファイル未作成
  - **信頼性**: 🔵 *save_prefix_feature_dataset `if not examples: raise ValueError(...)` より*

---

### REQ-135: compare-prefix-coldwarm smoke test 🔵

**信頼性**: 🔵 *Makefile compare-prefix-coldwarm target・AI_HUB_MAKE_RUN_FEEDBACK「compare-prefix-coldwarm targetのsmoke実行CI step」推奨より*

### Given（前提条件）

- GPT-2 tinyモデル等の軽量モデルが利用可能
- compare-prefix関連のMakefile targetsが定義されている
- 軽量テスト用YAML設定が存在

### When（実行条件）

- compare-prefix-coldを実行し、次いでcompare-prefix-warmを実行する

### Then（期待結果）

- cold run: exit code 0、キャッシュがディスクに作成される、source="built"
- warm run: exit code 0、既存キャッシュが再利用される、source="disk"

### テストケース

#### 正常系

- [x] **TC-135-01**: compare-prefix-coldが正常完了しキャッシュが作成される 🔵
  - **入力**: BUDGET=6, MAX_SEQ_LEN=64, QUICK_EVAL_EXAMPLES=4
  - **期待結果**: exit code 0, .cache/prefix_feature_cache_compare/に.ptファイルが存在
  - **信頼性**: 🔵 *Makefile compare-prefix-cold target・run_comparison.shより*

- [x] **TC-135-02**: compare-prefix-warmがキャッシュを再利用して正常完了する 🔵
  - **入力**: cold run完了後の既存キャッシュ
  - **期待結果**: exit code 0, prefix_feature_cache_*_sourceに"disk"が含まれる
  - **信頼性**: 🔵 *_maybe_cache_dataset disk hit パスより*

#### 境界値

- [x] **TC-135-B01**: cold→warmの実行で2回の完了と異なるsourceが確認される 🔵
  - **入力**: compare-prefix-coldwarmを1回実行
  - **期待結果**: cold runはsource="built"、warm runはsource="disk"
  - **信頼性**: 🔵 *compare-prefix-coldwarm targetのcold→warm順次実行仕様より*

---

### Phase 32 テストケースサマリー追加

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 32: Corrupted cache handling（REQ-128） | 0 | 3 | 0 | 3 |
| Phase 32: force_rebuild flag（REQ-129） | 2 | 0 | 0 | 2 |
| Phase 32: position_ids build path（REQ-130） | 2 | 0 | 0 | 2 |
| Phase 32: model.training restoration（REQ-131） | 1 | 2 | 0 | 3 |
| Phase 32: SHA-256 invalidation（REQ-132） | 3 | 0 | 0 | 3 |
| Phase 32: format version mismatch（REQ-133） | 0 | 2 | 0 | 2 |
| Phase 32: Empty dataset rejection（REQ-134） | 0 | 1 | 0 | 1 |
| Phase 32: compare-prefix smoke test（REQ-135） | 2 | 0 | 1 | 3 |
| **Phase 32 合計** | **10** | **8** | **1** | **19** |

---

## Phase 32a: AsyncCacheBuilder・非同期キャッシュビルド設定

### REQ-136: AsyncCacheBuilder バックグラウンドビルド 🔵

**信頼性**: 🔵 *async_cache_builder.py AsyncCacheBuilder・train_tg_lora.py 統合実装・eceddf3/a316624コミットより*

### Given（前提条件）

- AsyncCacheBuilderが有効なcfg、raw_datasets、cache_dirで初期化されている
- バックグラウンドデバイス（CPUまたはcuda:1等）が利用可能

### When（実行条件）

- start()でdaemon threadを開始し、バックグラウンドでモデルロード→キャッシュビルドを実行する

### Then（期待結果）

- ビルド完了後、get_result(label)がAsyncCacheBuildResultを返す
- result.datasetが有効なPrefixFeatureDatasetである
- result.errorがNoneである
- ビルド失敗時はfailed=True、errorに例外が格納される
- 学習ループはビルド中もブロックされない

### テストケース

#### 正常系

- [x] **TC-136-01**: バックグラウンドビルドが正常完了しキャッシュ済みデータセットが取得できる 🔵
  - **入力**: valid_quick + valid_fullの2データセット、CPU、mockモデル
  - **期待結果**: get_result("valid_quick").dataset が非None、len==4、error==None
  - **信頼性**: 🔵 *test_async_cache_builder.py `test_async_build_produces_cached_dataset` より*

- [x] **TC-136-02**: ディスクに既存キャッシュがある場合、ビルドをスキップしてロードする 🔵
  - **入力**: 事前保存済みキャッシュファイル、force_rebuild=False
  - **期待結果**: result.source=="disk"、datasetがロードされたものと一致
  - **信頼性**: 🔵 *test_async_cache_builder.py `test_async_build_disk_hit_skips_build` より*

#### 異常系

- [x] **TC-136-E01**: モデルロード失敗時はfailed=Trueとなりerrorに例外が格納される 🔵
  - **入力**: load_base_modelがRuntimeError("no GPU")を送出するmock
  - **期待結果**: builder.failed==True、"no GPU" in str(builder.error)
  - **信頼性**: 🔵 *test_async_cache_builder.py `test_async_build_error_sets_failed` より*

---

### REQ-137: AsyncCacheBuilder スレッドAPI 🔵

**信頼性**: 🔵 *async_cache_builder.py threading.Lock・start()/poll()/get_result()/join() 実装より*

### Given（前提条件）

- AsyncCacheBuilderインスタンスが初期化されている

### When（実行条件）

- start()でdaemon threadを開始し、poll()で非ブロッキングに完了確認する

### Then（期待結果）

- start()後、daemon threadが起動する（thread.name=="async-cache-builder"）
- poll()は非ブロッキング（100ms以内に返却）でcompleted状態を返す
- join(timeout)でスレッド終了を待機する
- thread-safeなlock機構でresults/completed/failed状態を管理する

### テストケース

#### 正常系

- [x] **TC-137-01**: poll()が非ブロッキングである（100ms以内に返却） 🔵
  - **入力**: start()直後にpoll()を呼び出す
  - **期待結果**: elapsed < 0.1秒
  - **信頼性**: 🔵 *test_async_cache_builder.py `test_async_poll_is_nonblocking` より*

#### 境界値

- [x] **TC-137-B01**: 存在しないlabelでget_result()を呼ぶとNoneを返す 🔵
  - **入力**: ビルド完了後にget_result("nonexistent")を呼ぶ
  - **期待結果**: None
  - **信頼性**: 🔵 *get_resultのdict.get(label)実装より*

---

### REQ-138: AsyncCacheBuilder 設定バリデーション 🔵

**信頼性**: 🔵 *config_schema.py TrainingConfig prefix_feature_cache_async/async_device validators・43a329aコミットより*

### Given（前提条件）

- TrainingConfigのPydanticスキーマが定義されている

### When（実行条件）

- prefix_feature_cache_async=TrueでTrainingConfigを構築する

### Then（期待結果）

- async=Trueでasync_device=Noneの場合、ValidationErrorで拒否される
- async=Trueでexperimental=Falseの場合、ValidationErrorで拒否される
- async=Falseの場合はバリデーションエラーなし

### テストケース

#### 異常系

- [x] **TC-138-E01**: prefix_feature_cache_async=True で async_device=None の場合ValidationError 🔵
  - **入力**: prefix_feature_cache_async=True, prefix_feature_cache_async_device=None
  - **期待結果**: ValidationError("prefix_feature_cache_async requires prefix_feature_cache_async_device")
  - **信頼性**: 🔵 *test_config_schema.py `test_async_cache_requires_device` より*

- [x] **TC-138-E02**: prefix_feature_cache_async=True で experimental=False の場合ValidationError 🔵
  - **入力**: prefix_feature_cache_async=True, prefix_feature_cache_async_device="cuda:1", experimental未指定
  - **期待結果**: ValidationError("prefix_feature_cache_async requires prefix_feature_cache_experimental")
  - **信頼性**: 🔵 *test_config_schema.py `test_async_cache_requires_experimental` より*

#### 正常系

- [x] **TC-138-01**: 9b_tg_lora_prefix_feature_cache_async.yamlがPydantic検証を通過する 🔵
  - **入力**: configs/9b_tg_lora_prefix_feature_cache_async.yaml
  - **期待結果**: 検証成功、例外なし
  - **信頼性**: 🔵 *test_config_schema.py `test_suffix_only_configs_load` より*

---

### Phase 32a テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 32a: AsyncCacheBuilder build（REQ-136） | 2 | 1 | 0 | 3 |
| Phase 32a: Thread API（REQ-137） | 1 | 0 | 1 | 2 |
| Phase 32a: Config validation（REQ-138） | 1 | 2 | 0 | 3 |
| **Phase 32a 合計** | **4** | **3** | **1** | **8** |

---

## Phase 33: 堅牢化・高速化・統合テストギャップ解消（REQ-139~143）

### REQ-139: AsyncCacheBuilder フルライフサイクル統合テスト 🔵

**信頼性**: 🔵 *AI_HUB_MAKE_RUN_FEEDBACK指摘・async_cache_builder.py 実装・test_async_cache_builder_integration.py より*

### Given（前提条件）

- AsyncCacheBuilderが実装済み
- モックベースのユニットテスト8件が存在するが、DataLoader差し替え・ディスク永続化・poll-and-swapパターンのE2E検証が欠落している

### When（実行条件）

- CPU上の軽量モデル（_TinyModel）でフルライフサイクル統合テストを実行する

### Then（期待結果）

- ビルド完了後にDataLoaderを作成し、正しいバッチ形状のデータを取得できる
- poll-and-swapパターンがトレーニングループと同じ手順で動作する
- ビルド失敗時にraw datasetで学習継続可能
- ディスクキャッシュが永続化され、2回目の実行で再利用される
- 並行するpoll()/get_result()呼び出しがスレッドセーフに動作する

### テストケース

#### 正常系

- [x] **TC-139-01**: CPU上でキャッシュビルド → DataLoader作成 → バッチ形状検証 🔵
  - **入力**: _TinyModel(vocab=32, hidden=16, layers=4), _TokenDataset(n=6), split_layer=2
  - **期待結果**: dataset長=6, DataLoader バッチ形状=(2,8,16), cache_path に .pt ファイルが存在
  - **信頼性**: 🔵 *test_async_cache_builder_integration.py `test_full_lifecycle_build_wait_load_on_cpu` より*

- [x] **TC-139-02**: poll-and-swapパターンでraw→cached DataLoaders切り替え 🔵
  - **入力**: builder.start() → raw DataLoader消費 → poll() → cached DataLoader切り替え
  - **期待結果**: raw_batches_consumed >= 1, cached_batches = 2, hidden_states.isfinite().all()
  - **信頼性**: 🔵 *test_async_cache_builder_integration.py `test_poll_and_swap_pattern_simulates_training` より*

- [x] **TC-139-03**: 1回目build → 2回目disk load (source='disk') 🔵
  - **入力**: 同一cfg/cache_dirでAsyncCacheBuilderを2回実行
  - **期待結果**: run1 source='built', run2 source='disk', len(run2.dataset)==len(run1.dataset)
  - **信頼性**: 🔵 *test_async_cache_builder_integration.py `test_disk_cache_reuse_skips_rebuild` より*

#### 境界値

- [x] **TC-139-B01**: ビルド失敗時のgraceful degradation（raw dataset継続） 🔵
  - **入力**: load_base_model side_effect=RuntimeError("simulated OOM")
  - **期待結果**: builder.failed=True, raw DataLoader から正常にバッチ取得可能
  - **信頼性**: 🔵 *test_async_cache_builder_integration.py `test_build_failure_continues_with_raw_dataset` より*

- [x] **TC-139-B02**: 4スレッドから並行poll()/get_result()呼び出し（各50回） 🔵
  - **入力**: barrier同期後に4スレッドで同時poll/get_result
  - **期待結果**: エラーなし, builder.poll()=True, builder.failed=False
  - **信頼性**: 🔵 *test_async_cache_builder_integration.py `test_concurrent_poll_and_get_result_are_threadsafe` より*

---

### REQ-140: diff_lora 高速化 🔵

**信頼性**: 🔵 *lora_state.py diff_lora fast paths・7a643a9コミットより*

### Given（前提条件）

- diff_lora(before, after, scale) 関数が実装済み

### When（実行条件）

- scale==0.0 または scale==1.0 で diff_lora を呼び出す

### Then（期待結果）

- scale==0.0: 全テンソルがゼロの辞書を返却（乗算回避）
- scale==1.0: after[k] - before[k] のみ実行（乗算回避）

### テストケース

#### 境界値

- [x] **TC-140-B01**: scale==0.0で返却テンソルが全てゼロ 🔵
  - **入力**: before/after 辞書、scale=0.0
  - **期待結果**: 全値が0.0
  - **信頼性**: 🔵 *test_lora_state.py diff_lora scale=0 テストより*

- [x] **TC-140-B02**: scale==1.0で返却テンソルがafter-beforeに一致 🔵
  - **入力**: before/after 辞書、scale=1.0
  - **期待結果**: result[k] == after[k] - before[k]（要素レベル）
  - **信頼性**: 🔵 *test_lora_state.py diff_lora scale=1 テストより*

---

### REQ-141: cosine_similarity 直交ベクトル警告 🔵

**信頼性**: 🔵 *metrics.py cosine_similarity warnings・7a643a9コミットより*

### Given（前提条件）

- cosine_similarity(a, b) 関数が実装済み

### When（実行条件）

- 完全に直交するベクトル（内積=0、ノルム>0）を入力する

### Then（期待結果）

- warnings.warn() で警告が発せられる
- 戻り値は 0.0
- stacklevel=2 で呼び出し元の行番号が表示される

### テストケース

#### 境界値

- [x] **TC-141-B01**: 直交ベクトル入力時に warnings.warn が発せられる 🔵
  - **入力**: a = {"k": tensor([1.0, 0.0])}, b = {"k": tensor([0.0, 1.0])}
  - **期待結果**: UserWarning が発せられ、戻り値=0.0
  - **信頼性**: 🔵 *test_metrics.py cosine_similarity orthogonal warning テストより*

---

### REQ-142: _get_decoder_layers ロギング 🔵

**信頼性**: 🔵 *activation_cache.py _get_decoder_layers debug log/enhanced error・7a643a9コミットより*

### Given（前提条件）

- _get_decoder_layers がモデルからdecoder層を探索する

### When（実行条件）

- decoder層を発見した場合、または全候補パスの探索に失敗した場合

### Then（期待結果）

- 発見時: logger.debug でパスをログ出力
- 失敗時: エラーメッセージに候補パス数を含む

### テストケース

#### 正常系

- [x] **TC-142-01**: decoder層発見時にdebugログが出力される 🔵
  - **入力**: layers属性を持つモデル
  - **期待結果**: logger.debug がパス付きで呼び出される
  - **信頼性**: 🔵 *activation_cache.py _get_decoder_levels debug log 実装より*

#### 異常系

- [x] **TC-142-E01**: 全候補パス失敗時に候補数を含むエラー 🔵
  - **入力**: decoder層を持たないモデル
  - **期待結果**: AttributeError メッセージに "Tried {N} paths" が含まれる
  - **信頼性**: 🔵 *activation_cache.py _get_decoder_layers enhanced error 実装より*

---

### REQ-143: smoke_async_prefix.yaml 設定サーフェス 🔵

**信頼性**: 🔵 *configs/smoke_async_prefix.yaml・182de29コミットより*

### Given（前提条件）

- 非同期キャッシュビルド検証用のYAML設定ファイルが存在する

### When（実行条件）

- smoke_async_prefix.yaml をPydanticスキーマで検証する

### Then（期待結果）

- prefix_feature_cache_async=true
- prefix_feature_cache_async_device="cuda:1"
- force_top_layers_only=true
- enable_random_walk=false

### テストケース

#### 正常系

- [x] **TC-143-01**: smoke_async_prefix.yamlがPydantic検証を通過する 🔵
  - **入力**: configs/smoke_async_prefix.yaml
  - **期待結果**: 検証成功、async=true, async_device="cuda:1"
  - **信頼性**: 🔵 *test_config_schema.py 設定読み込みテストより*

---

### Phase 33 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| AsyncCacheBuilder統合（REQ-139） | 3 | 0 | 2 | 5 |
| diff_lora高速化（REQ-140） | 0 | 0 | 2 | 2 |
| cosine_similarity警告（REQ-141） | 0 | 0 | 1 | 1 |
| decoder層ロギング（REQ-142） | 1 | 1 | 0 | 2 |
| 設定サーフェス（REQ-143） | 1 | 0 | 0 | 1 |
| **Phase 33 合計** | **5** | **1** | **5** | **11** |

---

## Phase 34: In-place tensor ops・data_ptr保存検証・velocity opsベンチマーク（REQ-144~148）

### REQ-144: In-place EMA update data_ptr保存 🔵

**信頼性**: 🔵 *velocity.py mul_(beta).add_(delta[k], alpha=(1.0-beta))・851041eコミット・test_velocity.py TestVelocityDataPtrPreservation 5テストより*

### Given（前提条件）

- Velocity.updateがEMA更新を実行する

### When（実行条件）

- 既存キーに対してin-place mul_/add_でEMA更新を実行する
- 新規キーに対してclone()で新しいテンソルを割り当てる

### Then（期待結果）

- 既存キーのテンソルdata_ptrが更新前と同一
- 新規キーのテンソルは既存テンソルと異なるdata_ptrを持つ
- メモリアロケーションオーバーヘッドが削減される

### テストケース

#### 正常系

- [x] **TC-144-01**: EMA更新後の既存キーdata_ptr不変 🔵
  - **入力**: velocity.update(delta, beta=0.9) を2回呼び出し
  - **期待結果**: 1回目更新後の既存キーdata_ptr == 2回目更新後のdata_ptr
  - **信頼性**: 🔵 *test_velocity.py test_ema_update_preserves_data_ptr より*

- [x] **TC-144-02**: 新規キーが既存テンソルと異なるdata_ptrを持つ 🔵
  - **入力**: 既存2キーに新規1キーを含むdeltaでupdate
  - **期待結果**: 新規キーのdata_ptrが既存2キーのいずれとも異なる
  - **信頼性**: 🔵 *test_velocity.py test_new_key_gets_different_data_ptr より*

#### 境界値

- [x] **TC-144-B01**: 混在時の既存キーdata_ptr保存 🔵
  - **入力**: 既存3キー + 新規1キーの混在deltaでupdate
  - **期待結果**: 既存3キーのdata_ptrが全て保存される
  - **信頼性**: 🔵 *test_velocity.py test_mixed_update_preserves_existing_data_ptr より*

- [x] **TC-144-B02**: 複数キー同時更新時のdata_ptr保存 🔵
  - **入力**: 5キー全て既存のdeltaでupdate
  - **期待結果**: 全5キーのdata_ptrが更新前と同一
  - **信頼性**: 🔵 *test_velocity.py test_multiple_keys_preserved より*

- [x] **TC-144-B03**: 新規キーによる既存テンソルの置き換え検出 🔵
  - **入力**: 同名の新規キーを含むdeltaでupdate（cloneパス）
  - **期待結果**: 置き換えられたキーのdata_ptrが前回と異なる
  - **信頼性**: 🔵 *test_velocity.py test_new_key_replaces_tensor より*

---

### REQ-145: In-place cap_update data_ptr保存 🔵

**信頼性**: 🔵 *extrapolator.py update.mul_(max_norm/update_norm)・851041eコミット・test_extrapolator.py TestCapUpdateDataPtrPreservation 4テストより*

### Given（前提条件）

- cap_updateが更新テンソルのcappingを実行する

### When（実行条件）

- update_norm > max_norm の場合にin-place mul_でスケーリングする
- update_norm <= max_norm の場合はテンソルを変更しない

### Then（期待結果）

- capping適用時: 返却テンソルのdata_ptrが入力と同一
- capping不要時: テンソルが変更されずdata_ptrが保存される
- 非有限入力時: 新しいゼロテンソルを返却（REQ-063参照）

### テストケース

#### 正常系

- [x] **TC-145-01**: capping適用時のdata_ptr保存 🔵
  - **入力**: 大きなupdate、max_ratio=0.01
  - **期待結果**: cap_update後のdata_ptr == 入力のdata_ptr
  - **信頼性**: 🔵 *test_extrapolator.py test_capping_preserves_data_ptr より*

- [x] **TC-145-02**: capping不要時のdata_ptr保存 🔵
  - **入力**: 小さなupdate（1e-8スケール）、max_ratio=0.01
  - **期待結果**: cap_update後のdata_ptr == 入力のdata_ptr
  - **信頼性**: 🔵 *test_extrapolator.py test_no_capping_preserves_data_ptr より*

#### 境界値

- [x] **TC-145-B01**: 非有限入力時の新規ゼロテンソル返却 🔵
  - **入力**: NaN/Infを含むupdateテンソル
  - **期待結果**: 返却テンソルのdata_ptrが入力と異なる、全要素が0.0
  - **信頼性**: 🔵 *test_extrapolator.py test_non_finite_returns_new_tensor より*

- [x] **TC-145-B02**: 非有限入力でもREQ-063ゼロ返却が維持される 🔵
  - **入力**: 全要素Infのupdateテンソル
  - **期待結果**: 返却テンソルが全要素0.0の新規テンソル
  - **信頼性**: 🔵 *test_extrapolator.py test_non_finite_returns_zeros（REQ-063テスト）より*

---

### REQ-146: data_ptr保存検証テスト 🔵

**信頼性**: 🔵 *test_velocity.py 5テスト・test_extrapolator.py 4テスト・c9928b6コミット(TASK-0079)より*

### Given（前提条件）

- velocity.updateとcap_updateのin-place操作が実装されている

### When（実行条件）

- 各in-place操作のdata_ptr保存を検証するテストを実行する

### Then（期待結果）

- 9テスト全てがパスする（velocity 5 + extrapolator 4）

### テストケース

#### 総合テスト

- [x] **TC-146-01**: TestVelocityDataPtrPreservation 5テスト全パス 🔵
  - **入力**: pytest tests/test_velocity.py::TestVelocityDataPtrPreservation -v
  - **期待結果**: 5 passed
  - **信頼性**: 🔵 *c9928b6コミット(TASK-0079)完了確認より*

- [x] **TC-146-02**: TestCapUpdateDataPtrPreservation 4テスト全パス 🔵
  - **入力**: pytest tests/test_extrapolator.py::TestCapUpdateDataPtrPreservation -v
  - **期待結果**: 4 passed
  - **信頼性**: 🔵 *c9928b6コミット(TASK-0079)完了確認より*

---

### REQ-147: benchmark_velocity_ops.pyマイクロベンチマーク 🔵

**信頼性**: 🔵 *scripts/benchmark_velocity_ops.py 236行・c51fd5bコミット(TASK-0080)・test_benchmark_velocity_ops.py より*

### Given（前提条件）

- velocity.pyとextrapolator.pyのin-place操作が実装されている

### When（実行条件）

- benchmark_velocity_ops.pyを--quick（10反復）または--iterations Nで実行する

### Then（期待結果）

- JSON出力にvelocity_emaセクション（time_ms, per_iter_ms, mem_delta_kb, iterations）が含まれる
- JSON出力にcap_updateセクション（time_ms, per_iter_ms, mem_delta_kb, nocap_time_ms, nocap_per_iter_ms, iterations, tensor_shape）が含まれる

### テストケース

#### 正常系

- [x] **TC-147-01**: --quickで有効なJSON出力 🔵
  - **入力**: python scripts/benchmark_velocity_ops.py --quick
  - **期待結果**: JSON出力、iterations=10
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_quick_json_output より*

- [x] **TC-147-02**: JSON出力に必須フィールドが含まれる 🔵
  - **入力**: --quick出力をJSON parse
  - **期待結果**: velocity_ema, cap_update キーが存在
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_json_output_has_required_fields より*

---

### REQ-148: Makefile bench-velocity-opsターゲット 🔵

**信頼性**: 🔵 *Makefile line 215-217・AI_HUB_MAKE_RUN_FEEDBACK指摘より*

### Given（前提条件）

- Makefileにbench-optimizer, bench-prefix-cacheパターンが存在する

### When（実行条件）

- `make bench-velocity-ops`を実行する

### Then（期待結果）

- scripts/benchmark_velocity_ops.pyが実行される
- JSON結果が標準出力に出力される

### テストケース

#### 正常系

- [x] **TC-148-01**: make bench-velocity-opsがexit code 0で完了する 🔵
  - **入力**: make bench-velocity-ops ITERATIONS=10
  - **期待結果**: exit code 0、JSON出力にvelocity_ema/cap_updateセクション
  - **信頼性**: 🔵 *Makefile target定義・benchmark_velocity_ops.py --iterations 10 動作より*

---

### Phase 34 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| In-place EMA data_ptr（REQ-144） | 2 | 0 | 3 | 5 |
| In-place cap_update data_ptr（REQ-145） | 2 | 0 | 2 | 4 |
| data_ptr検証テスト（REQ-146） | 2 | 0 | 0 | 2 |
| Velocity opsベンチマーク（REQ-147） | 2 | 0 | 0 | 2 |
| Makefileベンチマーク統合（REQ-148） | 1 | 0 | 0 | 1 |
| **Phase 34 合計** | **9** | **0** | **5** | **14** |

---

## Phase 35: bench-velocity-ops CI gate・回帰自動検出（REQ-149）

### REQ-149: bench-velocity-ops-ci CI gate 🔵

**信頼性**: 🔵 *benchmark_velocity_ops.py --baseline/--threshold実装済み・design-interview A31 🔴指摘解消・AI_HUB_MAKE_RUN_FEEDBACK「Wire bench-velocity-ops --baseline into CI」より*

### Given（前提条件）

- benchmark_velocity_ops.pyに--baseline/--save-baseline/--thresholdが実装済み
- baselines/velocity_ops.jsonがリポジトリにチェックインされている
- TestBaselineRegressionDetection 7テストで回帰検出が検証済み

### When（実行条件）

- `make bench-velocity-ops-ci`を実行する
- または`python scripts/benchmark_velocity_ops.py --quick --baseline baselines/velocity_ops.json --threshold 20`を実行する

### Then（期待結果）

- 全メトリクスが閾値内の場合: exit code 0、JSON出力にbaseline_comparison.regressed=false
- 1つでも閾値超過の場合: exit code 1、stderrに回帰詳細、JSON出力にbaseline_comparison.regressed=true
- ベースラインファイル不存在時: exit code 2

### テストケース

#### 正常系

- [x] **TC-149-01**: 回帰なし時にexit code 0で成功 🔵
  - **入力**: 同一ベースラインファイルでbaseline comparison実行
  - **期待結果**: exit code 0、regressions=[]
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_baseline_no_regression_exits_zero より*

- [x] **TC-149-02**: --save-baselineでベースラインJSONが作成される 🔵
  - **入力**: python benchmark_velocity_ops.py --quick --save-baseline /tmp/baseline.json
  - **期待結果**: ファイル作成、有効なJSON
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_save_baseline_creates_file より*

#### 異常系

- [x] **TC-149-E01**: 回帰検出時にexit code 1で失敗 🔵
  - **入力**: 意図的に大きいベースライン値でbaseline comparison実行
  - **期待結果**: exit code 1、stderrに回帰詳細出力
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_baseline_regression_exits_nonzero より*

- [x] **TC-149-E02**: ベースラインファイル不存在時にexit code 2で失敗 🔵
  - **入力**: --baseline /nonexistent/path.json
  - **期待結果**: exit code 2
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_baseline_missing_file_exits_nonzero より*

#### 境界値

- [x] **TC-149-B01**: threshold フラグが感度を制御する 🔵
  - **入力**: --threshold 100（100%閾値）で回帰が検出されない
  - **期待結果**: exit code 0
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_threshold_flag_controls_sensitivity より*

- [x] **TC-149-B02**: _compare_with_baseline単体テストで回帰検出ロジック検証 🔵
  - **入力**: 大きいcurrent値 vs 小さいbaseline値
  - **期待結果**: regressions listが非空
  - **信頼性**: 🔵 *test_benchmark_velocity_ops.py test_compare_with_baseline_detects_regression より*

---

### Phase 35 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| CI gate（REQ-149） | 2 | 2 | 2 | 6 |
| **Phase 35 合計** | **2** | **2** | **2** | **6** |

---

## Phase 36-37: LR探索統合・propose→training loop配線（REQ-150~152）

### REQ-150: lr_explore_prob/lr_log_sigma パラメータ定義 🔵

**信頼性**: 🔵 *config_schema.py lr_explore_prob/lr_log_sigma フィールド・random_walk_controller.py propose() log-normal walk・test_training_integration.py TestLrExplorationIntegration より*

### Given（前提条件）

- TGLoRAParamsスキーマにlr_explore_prob/lr_log_sigmaフィールドが定義されている
- RandomWalkControllerがlr_explore_prob/lr_log_sigmaをコンストラクタで受け取る

### When（実行条件）

- lr_explore_prob=0.8, lr_log_sigma=0.25を設定してtraining loopを実行

### Then（期待結果）

- controller.lr_explore_prob == 0.8
- controller.lr_log_sigma == 0.25

### テストケース

#### 正常系

- [x] **TC-150-01**: lr_explore_prob/lr_log_sigmaがconfigからcontrollerに伝播される 🔵
  - **入力**: lr_explore_prob=0.8, lr_log_sigma=0.25
  - **期待結果**: controller.lr_explore_prob == 0.8, controller.lr_log_sigma == 0.25
  - **信頼性**: 🔵 *test_training_integration.py TestLrExplorationIntegration.test_lr_explore_prob_wired_from_config より*

---

### REQ-151: proposal.lrのcontroller.state.lr反映 🔵

**信頼性**: 🔵 *train_tg_lora.py controller.state.lr = proposal.lr・test_training_integration.py TestLrExplorationIntegration より*

### Given（前提条件）

- lr_explore_prob=1.0（探索確率100%）でlog-normal lr探索が有効
- propose()が探索lrを生成

### When（実行条件）

- 1サイクルのpropose→acceptを実行

### Then（期待結果）

- controller.state.lrが探索済みlrに更新される
- 探索済みlrが決定論的boost/decayパス（5e-4 * 1.2 = 6e-4）と一致しない

### テストケース

#### 正常系

- [x] **TC-151-01**: propose()で生成された探索lrがcontroller.state.lrに反映される 🔵
  - **入力**: lr_explore_prob=1.0, lr_log_sigma=0.3, 1サイクルaccept
  - **期待結果**: state.lrが決定論的パスと異なる値で[lr_min, lr_max]範囲内
  - **信頼性**: 🔵 *test_training_integration.py TestLrExplorationIntegration.test_proposed_lr_applied_to_state より*

---

### REQ-152: 複数サイクルLR変動検証 🔵

**信頼性**: 🔵 *test_training_integration.py TestLrExplorationIntegration より*

### Given（前提条件）

- lr_explore_prob=1.0でlog-normal lr探索が有効
- lr_accept_boost=1.5, lr_reject_decay=0.5

### When（実行条件）

- 5サイクルのpropose→accept/reject交互パターン（accept, reject, accept, reject, accept）を実行

### Then（期待結果）

- lrが[lr_min, lr_max]範囲内に収まる
- lrが初期値(5e-4)から変化する
- lrが決定論的boost/decay計算値と一致しない

### テストケース

#### 正常系

- [x] **TC-152-01**: 複数サイクル後のlrが探索による変動を示す 🔵
  - **入力**: 5サイクル、accept/reject交互、lr_explore_prob=1.0
  - **期待結果**: lrが[lr_min, lr_max]内、初期値から変化、決定論的計算値と不一致
  - **信頼性**: 🔵 *test_training_integration.py TestLrExplorationIntegration.test_full_propose_accept_reject_cycle_with_lr_walk より*

---

### Phase 36 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| LR探索パラメータ配線（REQ-150） | 1 | 0 | 0 | 1 |
| 探索lr反映（REQ-151） | 1 | 0 | 0 | 1 |
| 複数サイクルLR変動（REQ-152） | 1 | 0 | 0 | 1 |
| **Phase 36 合計** | **3** | **0** | **0** | **3** |

---

### Phase 37: magnitude_acceleration・入力検証強化（REQ-153~159）

#### REQ-153: magnitude_acceleration 🔵

**信頼性**: 🔵 *velocity.py magnitude_acceleration() 実装・TASK-0090より*

##### テストケース

- [x] **TC-153-01**: magnitude_acceleration正常計算 🔵
  - **入力**: 5件以上のmagnitude history（[1.0, 2.0, 4.0, 7.0, 11.0] — 加速的増大）
  - **期待結果**: 正の加速度（slopes=[1,2,3,4], acceleration > 0）
  - **信頼性**: 🔵 *test_velocity.py より*

- [x] **TC-153-02**: 3件未満で0.0返却 🔵
  - **入力**: 2件のmagnitude history
  - **期待結果**: 0.0
  - **信頼性**: 🔵 *velocity.py n<3 guard より*

#### REQ-154: cap_update非有限値ロギング 🔵

**信頼性**: 🔵 *extrapolator.py cap_update() warning logging・TASK-0090より*

##### テストケース

- [x] **TC-154-01**: NaN/Inf検出時の警告ログ検証 🔵
  - **入力**: NaN/Infを含むupdateテンソル
  - **期待結果**: 警告ログにNaN数とInf数が含まれる
  - **信頼性**: 🔵 *test_extrapolator.py より*

#### REQ-155: DeltaTracker key-mismatch検証 🔵

**信頼性**: 🔵 *delta_tracker.py compute_and_record() key validation・TASK-0091より*

##### テストケース

- [x] **TC-155-01**: beforeに欠落キーがある場合 🔵
  - **入力**: afterに"key_a"が含まれbeforeに含まれない
  - **期待結果**: ValueErrorに"missing in before: ['key_a']"が含まれる
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

- [x] **TC-155-02**: afterに欠落キーがある場合 🔵
  - **入力**: beforeに"key_b"が含まれafterに含まれない
  - **期待結果**: ValueErrorに"missing in after: ['key_b']"が含まれる
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

- [x] **TC-155-03**: 双方に欠落キーがある場合 🔵
  - **入力**: 双方に異なるキーが欠落
  - **期待結果**: ValueErrorに両方の欠落が含まれる
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

#### REQ-156: RollbackManager max_historyガード 🔵

**信頼性**: 🔵 *rollback_manager.py __init__・TASK-0091より*

##### テストケース

- [x] **TC-156-01**: max_history=0でValueError 🔵
  - **入力**: max_history=0
  - **期待結果**: ValueError
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

- [x] **TC-156-02**: max_history=-1でValueError 🔵
  - **入力**: max_history=-1
  - **期待結果**: ValueError
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

- [x] **TC-156-03**: max_history=1で正常初期化 🔵
  - **入力**: max_history=1
  - **期待結果**: 正常初期化
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

#### REQ-157: snapshot_lora_delta空ベース検証 🔵

**信頼性**: 🔵 *lora_state.py snapshot_lora_delta()・df57154コミットより*

##### テストケース

- [x] **TC-157-01**: 空base辞書でValueError 🔵
  - **入力**: base={}
  - **期待結果**: ValueError("base snapshot must not be empty")
  - **信頼性**: 🔵 *test_validation_hardening_0091.py より*

#### REQ-158: propose() OverflowError防止 🔵

**信頼性**: 🔵 *random_walk_controller.py propose()・580680dコミットより*

##### テストケース

- [x] **TC-158-01**: 極大alpha_log_sigmaでOverflowErrorなし 🔵
  - **入力**: 極大alpha_log_sigmaでpropose()呼び出し
  - **期待結果**: OverflowErrorなし、alphaがalpha_max以下にクランプ
  - **信頼性**: 🔵 *test_random_walk_controller.py より*

#### REQ-159: _compute_stats autograd漏れ防止 🔵

**信頼性**: 🔵 *delta_tracker.py _compute_stats() @torch.no_grad()・580680dコミットより*

##### テストケース

- [x] **TC-159-01**: _compute_stats実行後にautogradグラフが構築されない 🔵
  - **入力**: requires_grad=Trueのテンソルを含むdelta
  - **期待結果**: 統計計算後にグラフノードが増加しない
  - **信頼性**: 🔵 *test_delta_tracker.py より*

### Phase 37 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| magnitude_acceleration（REQ-153） | 1 | 0 | 1 | 2 |
| cap_updateロギング（REQ-154） | 0 | 1 | 0 | 1 |
| key-mismatch検証（REQ-155） | 0 | 3 | 0 | 3 |
| max_historyガード（REQ-156） | 1 | 2 | 0 | 3 |
| 空ベース検証（REQ-157） | 0 | 1 | 0 | 1 |
| OverflowError防止（REQ-158） | 1 | 0 | 0 | 1 |
| autograd漏れ防止（REQ-159） | 1 | 0 | 0 | 1 |
| **Phase 37 合計** | **4** | **7** | **1** | **12** |

---

### Phase 38: 加速度適応パラメータ設定サーフェス（REQ-160~161）

#### REQ-160: accel param config schema 🔵

**信頼性**: 🔵 *config_schema.py accel_instability_lr_decay/accel_convergence_lr_boost・1bc6345コミットより*

##### テストケース

- [x] **TC-160-01**: デフォルト値の検証 🔵
  - **入力**: YAMLにaccel params未指定
  - **期待結果**: accel_instability_lr_decay=0.7, accel_convergence_lr_boost=1.1
  - **信頼性**: 🔵 *test_config_schema.py TestAccelParamConfig より*

- [x] **TC-160-02**: 範囲外値の拒否 🔵
  - **入力**: accel_instability_lr_decay=0.0, 1.0; accel_convergence_lr_boost=1.0
  - **期待結果**: Pydantic ValidationError
  - **信頼性**: 🔵 *test_config_schema.py TestAccelParamConfig より*

- [x] **TC-160-03**: カスタム値の受入 🔵
  - **入力**: accel_instability_lr_decay=0.5, accel_convergence_lr_boost=1.5
  - **期待結果**: 正常にロード
  - **信頼性**: 🔵 *test_config_schema.py TestAccelParamConfig より*

#### REQ-161: accel param controller配線 🔵

**信頼性**: 🔵 *random_walk_controller.py コンストラクタ・train_tg_lora.py 配線・1bc6345コミットより*

##### テストケース

- [x] **TC-161-01**: カスタムinstability decayの適用 🔵
  - **入力**: accel_instability_lr_decay=0.5、正の加速度
  - **期待結果**: lr×0.5の減衰が適用
  - **信頼性**: 🔵 *test_random_walk_controller.py test_custom_instability_decay_applied より*

- [x] **TC-161-02**: カスタムconvergence boostの適用 🔵
  - **入力**: accel_convergence_lr_boost=1.5、負の加速度
  - **期待結果**: lr×1.5の増加が適用
  - **信頼性**: 🔵 *test_random_walk_controller.py test_custom_convergence_boost_applied より*

- [x] **TC-161-03**: None渡し時のデフォルト使用 🔵
  - **入力**: accel params=None
  - **期待結果**: デフォルト値(0.7/1.1)が使用される
  - **信頼性**: 🔵 *test_random_walk_controller.py test_accel_params_none_uses_defaults より*

### Phase 38 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| config schema（REQ-160） | 1 | 2 | 0 | 3 |
| controller配線（REQ-161） | 3 | 0 | 0 | 3 |
| **Phase 38 合計** | **4** | **2** | **0** | **6** |

---

### Phase 39: --resume障害回復再開・加速度適応観測性（REQ-162~166）

#### REQ-162: restore_state() 🔵

**信頼性**: 🔵 *random_walk_controller.py restore_state()・9f195f0コミット・test_fault_recovery.py より*

##### テストケース

- [x] **TC-162-01**: 保存済みControllerStateの正常復元 🔵
  - **入力**: save_training_state→load_training_state→restore_state
  - **期待結果**: K/alpha/total_cycles/accepted_countが保存値と一致、last_accel_action=0
  - **信頼性**: 🔵 *test_fault_recovery.py TestRestoreStateIntegration.test_restore_state_from_saved_checkpoint より*

- [x] **TC-162-02**: config保持確認 🔵
  - **入力**: 異なるK_candidatesで作成したcontrollerにrestore_state
  - **期待結果**: candidates/boundsはコンストラクタ時の値を保持
  - **信頼性**: 🔵 *random_walk_controller.py restore_state() "keeps its config" コメント・9f195f0コミットより*

#### REQ-163: resume_pathによる学習再開 🔵

**信頼性**: 🔵 *train_tg_lora.py resume_path引数・9f195f0コミット・test_fault_recovery.py より*

##### テストケース

- [x] **TC-163-01**: resume_path指定時のTrainingState復元 🔵
  - **入力**: resume_path有効パス、mock load_training_state
  - **期待結果**: controller.restore_state/velocity/delta_tracker/cycle_state/cycle_offsetが復元
  - **信頼性**: 🔵 *test_fault_recovery.py TestRestoreStateIntegration.test_resume_path_loads_training_state より*

- [x] **TC-163-02**: cycle_offset未満サイクルのスキップ 🔵
  - **入力**: cycle_offset=3、total_cycles=10
  - **期待結果**: cycle 0-2がスキップ、cycle 3から学習開始
  - **信頼性**: 🔵 *train_tg_lora.py `if cycle < cycle_offset: continue`・9f195f0コミットより*

#### REQ-164: --resume CLI引数 🔵

**信頼性**: 🔵 *train_tg_lora.py main() --resume・9f195f0コミット・test_fault_recovery.py より*

##### テストケース

- [x] **TC-164-01**: --resume引数のパースと配線 🔵
  - **入力**: `--resume path/to/training_state.pt`
  - **期待結果**: train_tg_lora(cfg, resume_path="path/to/training_state.pt")が呼ばれる
  - **信頼性**: 🔵 *test_fault_recovery.py test_resume_path_loads_training_state より*

#### REQ-165: last_accel_action属性 🔵

**信頼性**: 🔵 *random_walk_controller.py last_accel_action・b2eb409コミット・test_random_walk_controller.py 6テストより*

##### テストケース

- [x] **TC-165-01**: デフォルト値が0 🔵
  - **入力**: 新規作成controller
  - **期待結果**: last_accel_action == 0
  - **信頼性**: 🔵 *test_random_walk_controller.py test_last_accel_action_default_is_zero より*

- [x] **TC-165-02**: 正の加速度で1 🔵
  - **入力**: adapt_to_acceleration(acceleration=1.0)
  - **期待結果**: last_accel_action == 1
  - **信頼性**: 🔵 *test_random_walk_controller.py test_last_accel_action_positive_accel より*

- [x] **TC-165-03**: 負の加速度で-1 🔵
  - **入力**: adapt_to_acceleration(acceleration=-0.5)
  - **期待結果**: last_accel_action == -1
  - **信頼性**: 🔵 *test_random_walk_controller.py test_last_accel_action_negative_accel より*

- [x] **TC-165-04**: ゼロ加速度で0 🔵
  - **入力**: adapt_to_acceleration(1.0)→adapt_to_acceleration(0.0)
  - **期待結果**: 1→0に遷移
  - **信頼性**: 🔵 *test_random_walk_controller.py test_last_accel_action_zero_accel より*

- [x] **TC-165-05**: random_walk無効時は常に0 🔵
  - **入力**: enable_random_walk=False、adapt_to_acceleration(1.0)
  - **期待結果**: last_accel_action == 0
  - **信頼性**: 🔵 *test_random_walk_controller.py test_last_accel_action_disabled_random_walk より*

- [x] **TC-165-06**: summary()にlast_accel_action含まれる 🔵
  - **入力**: adapt_to_acceleration(1.0)→summary()
  - **期待結果**: "last_accel_action"キーが存在、値==1
  - **信頼性**: 🔵 *test_random_walk_controller.py test_last_accel_action_in_summary より*

#### REQ-166: MLflow cycle metrics 🔵

**信頼性**: 🔵 *train_tg_lora.py MLflow magnitude_acceleration/accel_action・b2eb409コミットより*

##### テストケース

- [x] **TC-166-01**: magnitude_accelerationがMLflow metricsに含まれる 🔵
  - **入力**: velocity with magnitude history
  - **期待結果**: cycle metrics辞書に"magnitude_acceleration"キーが存在
  - **信頼性**: 🔵 *train_tg_lora.py MLflow metrics wiring・b2eb409コミットより*

- [x] **TC-166-02**: accel_actionがMLflow metricsに含まれる 🔵
  - **入力**: controller with last_accel_action
  - **期待結果**: cycle metrics辞書に"accel_action"キーが存在
  - **信頼性**: 🔵 *train_tg_lora.py MLflow metrics wiring・b2eb409コミットより*

### Phase 39 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| restore_state（REQ-162） | 2 | 0 | 0 | 2 |
| resume_path（REQ-163） | 2 | 0 | 0 | 2 |
| --resume CLI（REQ-164） | 1 | 0 | 0 | 1 |
| last_accel_action（REQ-165） | 6 | 0 | 0 | 6 |
| MLflow metrics（REQ-166） | 2 | 0 | 0 | 2 |
| **Phase 39 合計** | **13** | **0** | **0** | **13** |

---

### Phase 40: --resume E2E統合テスト（REQ-167）

#### REQ-167: E2E resume flow 🔵

**信頼性**: 🔵 *test_resume_e2e.py実装・TASK-0090完了より*

##### テストケース

- [x] **TC-167-01**: E2E save→interrupt→resume→loss継続 🔵
  - **入力**: モック学習ループで2サイクル実行→TrainingState保存→中断→resume_pathで再開
  - **期待結果**: cycle 0-1がスキップ、cycle 2から継続、lossが復元状態から連続
  - **信頼性**: 🔵 *test_resume_e2e.py TestResumeE2E.test_full_resume_flow_loss_continuity より*

- [x] **TC-167-02**: cycle_offset未満サイクルのスキップ 🔵
  - **入力**: cycle_offset=3で保存→resume
  - **期待結果**: cycle 0-2がスキップされることをアサート
  - **信頼性**: 🔵 *test_resume_e2e.py TestResumeE2E.test_cycle_skipping_on_resume より*

- [x] **TC-167-03**: resume後velocity方向の保存 🔵
  - **入力**: velocity構築後に保存→ロード→velocity state比較
  - **期待結果**: velocity stateの方向が保存時と一致
  - **信頼性**: 🔵 *test_resume_e2e.py TestResumeE2E.test_resume_preserves_velocity_direction より*

### Phase 41: TruthfulQA分析・accel param実験（REQ-168~170）

#### REQ-168: Benchmark analysis script 🔵

**信頼性**: 🔵 *TASK-0091 spec・scripts/analyze_benchmark.py設計より*

##### テストケース

- [x] **TC-168-01**: baseline/TG-LoRAメトリクス差分計算 🔵
  - **入力**: モックベンチマーク結果JSON（baseline + TG-LoRA）
  - **期待結果**: 各メトリクスの差分が正しく計算される
  - **信頼性**: 🔵 *scripts/analyze_benchmark.py設計より*

- [x] **TC-168-02**: 欠損メトリクス時のエラーハンドリング 🔵
  - **入力**: 一部メトリクスが欠損したJSON
  - **期待結果**: 欠損メトリクスをスキップし、利用可能なメトリクスのみで差分計算
  - **信頼性**: 🔵 *scripts/analyze_benchmark.py設計より*

#### REQ-169: Accel param sensitivity 🔵

**信頼性**: 🔵 *TASK-0091 spec・random_walk_controller.py adapt_to_acceleration()より*

##### テストケース

- [x] **TC-169-01**: accel_instability_lr_decay値によるlr減衰率変化 🟡
  - **入力**: accel_instability_lr_decay=0.3, 0.5, 0.7, 0.9
  - **期待結果**: 減衰率がパラメータ値に応じて変化
  - **信頼性**: 🟡 *random_walk_controller.py adapt_to_acceleration()実装から妥当な推測*

- [x] **TC-169-02**: accel_convergence_lr_boost値によるlr回復率変化 🟡
  - **入力**: accel_convergence_lr_boost=1.1, 1.3, 1.5, 2.0
  - **期待結果**: 回復率がパラメータ値に応じて変化
  - **信頼性**: 🟡 *random_walk_controller.py adapt_to_acceleration()実装から妥当な推測*

#### REQ-170: Experiment config validation 🔵

**信頼性**: 🔵 *TASK-0092 spec・config_schema.py Pydantic検証より*

##### テストケース

- [x] **TC-170-01**: 全実験configのPydantic検証 🔵
  - **入力**: 4つのaccel実験config YAML
  - **期待結果**: 全てTGLoRAConfigで検証成功、accel paramsが期待値
  - **信頼性**: 🔵 *config_schema.py Pydantic検証・test_config_schema.pyパターンより*

#### REQ-171: Sweep execution script 🔵

**信頼性**: 🔵 *scripts/run_accel_sweep.sh実装・TASK-0092 specより*

##### テストケース

- [x] **TC-171-01**: sweepスクリプトが4configを順次実行 🔵
  - **入力**: run_accel_sweep.sh実行
  - **期待結果**: conservative, aggressive, balanced, no_accelの4configが順次実行され、結果がreports/accel_sweep/に集約される
  - **信頼性**: 🔵 *scripts/run_accel_sweep.sh実装・537c0a9コミットより*

- [x] **TC-171-02**: 個別実験失敗時のエラー耐性 🔵
  - **入力**: 一部のconfigで学習エラーが発生
  - **期待結果**: エラーを記録しつつ残りのconfig実験を継続、最終レポートに失敗を含む
  - **信頼性**: 🔵 *scripts/run_accel_sweep.sh error handling・537c0a9コミットより*

#### REQ-172: Sweep results summarization 🔵

**信頼性**: 🔵 *scripts/summarize_sweep.py実装・TASK-0092運用インフラより*

##### テストケース

- [x] **TC-172-01**: sweep結果の集約とvalidation lossソート 🔵
  - **入力**: 複数run_metrics.jsonlを含むsweepディレクトリ
  - **期待結果**: 各実験の受理率・学習統計を計算、validation loss順でソートしたサマリーを出力
  - **信頼性**: 🔵 *scripts/summarize_sweep.py実装・537c0a9コミットより*

### Phase 40~41 テストケースサマリー

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| E2E resume（REQ-167） | 3 | 0 | 0 | 3 |
| benchmark analysis（REQ-168） | 2 | 0 | 0 | 2 |
| accel sensitivity（REQ-169） | 2 | 0 | 0 | 2 |
| config validation（REQ-170） | 1 | 0 | 0 | 1 |
| sweep execution（REQ-171） | 2 | 0 | 0 | 2 |
| sweep summarization（REQ-172） | 1 | 0 | 0 | 1 |
| **Phase 40~41 合計** | **11** | **0** | **0** | **11** |

---

## テストケース総合サマリー

### フェーズ別テストケース数

| フェーズ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 32 | 10 | 8 | 1 | 19 |
| Phase 32a | 4 | 3 | 1 | 8 |
| Phase 33 | 5 | 1 | 5 | 11 |
| Phase 34 | 9 | 0 | 5 | 14 |
| Phase 35 | 2 | 2 | 2 | 6 |
| Phase 36 | 3 | 0 | 0 | 3 |
| Phase 37 | 4 | 7 | 1 | 12 |
| Phase 38 | 4 | 2 | 0 | 6 |
| Phase 39 | 13 | 0 | 0 | 13 |
| Phase 40 | 3 | 0 | 0 | 3 |
| Phase 41 | 8 | 0 | 0 | 8 |
| **Phase 32~41 合計** | **65** | **23** | **15** | **103** |

---

## Phase 42: コンストラクタ入力検証・テスト非決定性排除

### REQ-173: DeltaTracker max_history検証 🔵

**信頼性**: 🔵 *delta_tracker.py 既存実装・RollbackManager検証パターン（REQ-156）より*

#### Given（前提条件）

- DeltaTrackerコンストラクタがmax_historyパラメータを受け取る

#### When（実行条件）

- max_historyに0または負の値を渡す

#### Then（期待結果）

- ValueErrorが送出される

#### テストケース

##### 境界値

- [x] **TC-173-01**: DeltaTracker(max_history=0) → ValueError 🔵
- [x] **TC-173-02**: DeltaTracker(max_history=-1) → ValueError 🔵

---

### REQ-174: Velocity max_history検証 🔵

**信頼性**: 🔵 *velocity.py 既存実装・RollbackManager検証パターン（REQ-156）より*

#### Given（前提条件）

- Velocityコンストラクタがmax_historyパラメータを受け取る

#### When（実行条件）

- max_historyに0または負の値を渡す

#### Then（期待結果）

- ValueErrorが送出される

#### テストケース

##### 境界値

- [x] **TC-174-01**: Velocity(max_history=0) → ValueError 🔵
- [x] **TC-174-02**: Velocity(max_history=-1) → ValueError 🔵

---

### REQ-175: OptimizerLifecycleManager lr/weight_decay検証 🔵

**信頼性**: 🔵 *optimizer_lifecycle.py 既存実装・config_schema.py Field(gt=0.0)パターンより*

#### Given（前提条件）

- OptimizerLifecycleManagerコンストラクタがlr, weight_decayパラメータを受け取る

#### When（実行条件）

- lr <= 0 または weight_decay < 0 を渡す

#### Then（期待結果）

- ValueErrorが送出される

#### テストケース

##### 境界値

- [x] **TC-175-01**: OptimizerLifecycleManager(lr=0.0) → ValueError 🔵
- [x] **TC-175-02**: OptimizerLifecycleManager(lr=-0.001) → ValueError 🔵
- [x] **TC-175-03**: OptimizerLifecycleManager(weight_decay=-0.01) → ValueError 🔵
- [x] **TC-175-04**: OptimizerLifecycleManager(weight_decay=0.0) → 成功（0.0は有効） 🔵

---

### REQ-176: PrefixFeatureDataset/MappedPrefixFeatureDataset検証 🔵

**信頼性**: 🔵 *prefix_feature_cache.py 既存実装・InfiniteBatchIterator空データセット検証（REQ-078）パターンより*

#### Given（前提条件）

- PrefixFeatureDataset/MappedPrefixFeatureDatasetコンストラクタが呼び出される

#### When（実行条件）

- 空のexamplesリスト、不正なsplit_layer_idx、または互換性のないテンソル形状を渡す

#### Then（期待結果）

- ValueErrorが送出される

#### テストケース

##### 境界値

- [x] **TC-176-01**: PrefixFeatureDataset(examples=[]) → ValueError 🔵
- [x] **TC-176-02**: MappedPrefixFeatureDataset(split_layer_idx=-1) → ValueError 🔵

---

### REQ-177: AsyncCacheBuilder検証 🔵

**信頼性**: 🔵 *async_cache_builder.py 既存実装・RandomWalkController検証パターン（REQ-074）より*

#### Given（前提条件）

- AsyncCacheBuilderコンストラクタが呼び出される

#### When（実行条件）

- 不正なdevice文字列、範囲外のsplit_layer、またはNone configを渡す

#### Then（期待結果）

- ValueErrorが送出される

#### テストケース

##### 境界値

- [x] **TC-177-01**: AsyncCacheBuilder(device="invalid") → ValueError 🔵
- [x] **TC-177-02**: AsyncCacheBuilder(split_layer=-1) → ValueError 🔵

---

### REQ-178: テスト非決定性排除 🔵

**信頼性**: 🔵 *test_restore_state flaky fix（f5fe40fコミット）・AI_HUB_MAKE_RUN_FEEDBACK「scan the full test suite for other tests using default lr_explore_prob」より*

#### Given（前提条件）

- RandomWalkControllerを使用するテストが存在する

#### When（実行条件）

- テストがexplore_probパラメータを明示的に指定せずにコントローラをインスタンス化する

#### Then（期待結果）

- テスト対象以外の探索確率は0.0に設定され、テストの非決定性が排除される

#### テストケース

##### 正常系

- [x] **TC-178-01**: 全テストファイルのRandomWalkController呼び出しで、テスト対象のexplore_prob以外が0.0に設定されていることをlint/regexで検証 🔵

##### 境界値

- [x] **TC-178-02**: k_explore_prob=0.0設定時にpropose()のK変更回数が0であることを検証 🔵

---

### Phase 42テストケース数

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| DeltaTracker検証（REQ-173） | 0 | 0 | 2 | 2 |
| Velocity検証（REQ-174） | 0 | 0 | 2 | 2 |
| OptimizerLifecycleManager検証（REQ-175） | 0 | 0 | 4 | 4 |
| PrefixFeatureDataset検証（REQ-176） | 0 | 0 | 2 | 2 |
| AsyncCacheBuilder検証（REQ-177） | 0 | 0 | 2 | 2 |
| テスト非決定性排除（REQ-178） | 1 | 0 | 1 | 2 |
| **Phase 42 合計** | **1** | **0** | **13** | **14** |

---

## テストケース総合サマリー（更新）

| フェーズ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 32 | 10 | 8 | 1 | 19 |
| Phase 32a | 4 | 3 | 1 | 8 |
| Phase 33 | 5 | 1 | 5 | 11 |
| Phase 34 | 9 | 0 | 5 | 14 |
| Phase 35 | 2 | 2 | 2 | 6 |
| Phase 36 | 3 | 0 | 0 | 3 |
| Phase 37 | 4 | 7 | 1 | 12 |
| Phase 38 | 4 | 2 | 0 | 6 |
| Phase 39 | 13 | 0 | 0 | 13 |
| Phase 40 | 3 | 0 | 0 | 3 |
| Phase 41 | 8 | 0 | 0 | 8 |
| Phase 42 | 1 | 0 | 13 | 14 |
| **Phase 32~42 合計** | **66** | **23** | **28** | **117** |

---

## Phase 54: Frontier Sweep パイプライン強化テストケース（REQ-198~204）

### REQ-198: G2.3 Frontier Separation 自動評価 🔵

**信頼性**: 🔵 *evaluate_paper_gates.py --frontier-report実装より*

#### 正常系

- [x] **TC-198-01**: --frontier-report指定時にfrontier_separation_detected=trueでG2.3 pass 🔵
- [x] **TC-198-02**: --frontier-report指定時にfrontier_separation_detected=falseでG2.3 fail 🔵

#### 異常系

- [x] **TC-198-E01**: --frontier-report指定なしてG2.3がinformational/skip 🔵

---

### REQ-199: 構造化メタデータパイプライン 🔵

**信頼性**: 🔵 *run_frontier_sweep.sh run_metadata.json書き出しより*

#### 正常系

- [x] **TC-199-01**: run_metadata.jsonがmake_exit/summary_exists/oom_in_logの3フィールドを含む 🔵

#### 境界値

- [x] **TC-199-B01**: run_metadata.json不在時のレガシーフォールバック 🔵

---

### REQ-200: メモリデルタメトリクス 🔵

**信頼性**: 🔵 *frontier_report.py memory_delta_mb/memory_savings_pct計算より*

#### 正常系

- [x] **TC-200-01**: memory_delta_mb = baseline_peak_mb - tg_peak_mbが正しく計算される 🔵
- [x] **TC-200-02**: avg_memory_savings_pctが完了runの加重平均で計算される 🔵

#### 境界値

- [x] **TC-200-B01**: 完了runなし（deltaリスト空）時avg_savings_pct=None 🔵

---

### REQ-201: OOM検知・ステータス分類 🔵

**信頼性**: 🔵 *frontier_report.py detect_oom_from_log()/determine_status()より*

#### 正常系

- [x] **TC-201-01**: exit_code=0 + summary_exists → completed 🔵
- [x] **TC-201-02**: exit_code=137 → oom 🔵
- [x] **TC-201-03**: CUDA out of memoryパターン検知 → oom 🔵
- [x] **TC-201-04**: Killedパターン検知 → oom 🔵

#### 境界値

- [x] **TC-201-B01**: 全OOMパターンの個別検知（4パターン） 🔵

---

### REQ-204: Frontier Sweep テストカバレッジ 🔵

**信頼性**: 🔵 *test_frontier_report.py 675行・14+テストクラスより*

- [x] **TC-204-01**: test_frontier_report.pyがOOM検知・ステータス分類・frontier boundary・memory delta・metadataパイプラインをカバー 🔵

---

### Phase 54テストケース数

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| G2.3自動評価（REQ-198） | 2 | 1 | 0 | 3 |
| メタデータパイプライン（REQ-199） | 1 | 0 | 1 | 2 |
| メモリデルタ（REQ-200） | 2 | 0 | 1 | 3 |
| OOM検知（REQ-201） | 4 | 0 | 1 | 5 |
| テストカバレッジ（REQ-204） | 1 | 0 | 0 | 1 |
| **Phase 54 合計** | **10** | **1** | **3** | **14** |

---

## テストケース総合サマリー（更新）

| フェーズ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 32 | 10 | 8 | 1 | 19 |
| Phase 32a | 4 | 3 | 1 | 8 |
| Phase 33 | 5 | 1 | 5 | 11 |
| Phase 34 | 9 | 0 | 5 | 14 |
| Phase 35 | 2 | 2 | 2 | 6 |
| Phase 36 | 3 | 0 | 0 | 3 |
| Phase 37 | 4 | 7 | 1 | 12 |
| Phase 38 | 4 | 2 | 0 | 6 |
| Phase 39 | 13 | 0 | 0 | 13 |
| Phase 40 | 3 | 0 | 0 | 3 |
| Phase 41 | 8 | 0 | 0 | 8 |
| Phase 42 | 1 | 0 | 13 | 14 |
| Phase 54 | 10 | 1 | 3 | 14 |
| **Phase 32~54 合計** | **76** | **24** | **31** | **131** |

---

## Phase 55: 運用スクリプト・ユーティリティモジュール（REQ-205~217）

### REQ-205: io.py JSON/JSONL I/O 🔵

**信頼性**: 🔵 *src/utils/io.py 既存実装・pyproject.toml orjson>=3.9依存より*

#### Given（前提条件）

- orjson パッケージがインストール済み

#### When（実行条件）

- save_json / load_json / save_jsonl / load_jsonl を呼び出す

#### Then（期待結果）

- JSON/JSONL ファイルが正しく読み書きされる

#### テストケース

##### 正常系

- [x] **TC-205-01**: save_json→load_json の往復テスト 🔵
  - **入力**: 辞書データ → JSON ファイル → 読込
  - **期待結果**: 元の辞書と完全一致
  - **信頼性**: 🔵 *既存テストパターンより*

##### 異常系

- [x] **TC-205-E01**: 存在しないファイルの load_json 🔵
  - **入力**: 存在しないパス
  - **期待結果**: FileNotFoundError または同等のエラー
  - **信頼性**: 🔵 *io.py 実装より*

---

### REQ-206: memory.py VRAM/パラメータユーティリティ 🔵

**信頼性**: 🔵 *src/utils/memory.py 既存実装より*

#### Given（前提条件）

- CUDA 利用可能な GPU 環境

#### When（実行条件）

- vram_usage_mb() / count_parameters() を呼び出す

#### Then（期待結果）

- GPU メモリ使用量（MB）とパラメータ数が正しく返される

#### テストケース

##### 正常系

- [x] **TC-206-01**: vram_usage_mb が正の値を返す 🔵
  - **期待結果**: デバイス別の MB 値が float で返される
  - **信頼性**: 🔵 *既存実装より*

---

### REQ-207: run_query.py JSONL クエリ API 🔵

**信頼性**: 🔵 *src/utils/run_query.py 既存実装・TASK-0060 より*

#### テストケース

##### 正常系

- [x] **TC-207-01**: parse_jsonl が JSONL を辞書リストに変換 🔵
- [x] **TC-207-02**: get_footer が run_footer レコードを取得 🔵
- [x] **TC-207-03**: get_cycle_history がサイクル別ステップを返す 🔵

---

### REQ-208: logging.py ロギング設定 🔵

**信頼性**: 🔵 *src/utils/logging.py 既存実装より*

#### テストケース

##### 正常系

- [x] **TC-208-01**: get_logger が "tg-lora" ロガーを返す 🔵
- [x] **TC-208-02**: ensure_dir がディレクトリを作成 🔵

---

### REQ-209: checkpoint.py チェックポイント管理 🔵

**信頼性**: 🔵 *src/utils/checkpoint.py 既存実装・REQ-163 resume_path 連携より*

#### テストケース

##### 正常系

- [x] **TC-209-01**: save_training_state→load_training_state の往復テスト 🔵
- [x] **TC-209-02**: _sanitize_tensors が NaN/Inf をサニタイズ 🔵

---

### REQ-210: run_sweep.sh HP スイープ 🔵

**信頼性**: 🔵 *scripts/run_sweep.sh 既存実装67行より*

#### テストケース

##### 正常系

- [x] **TC-210-01**: 9設定のスイープが各設定の run_metrics.jsonl を生成 🔵

---

### REQ-211: run_ablation_suite.sh アブレーション 🔵

**信頼性**: 🔵 *scripts/run_ablation_suite.sh 既存実装137行より*

#### テストケース

##### 正常系

- [x] **TC-211-01**: baseline→paper POC→adaptive K5→no-convergence の全変種が実行される 🔵

---

### REQ-212: run_high_lr_comparison.sh 高LR安定性 🔵

**信頼性**: 🔵 *scripts/run_high_lr_comparison.sh 既存実装141行より*

#### テストケース

##### 正常系

- [x] **TC-212-01**: 高LRでベースラインが発散しTG-LoRAがロールバックで安定すること 🔵

---

### REQ-213: run_kstep_rollback_test.sh K-step検証 🔵

**信頼性**: 🔵 *scripts/run_kstep_rollback_test.sh 既存実装118行・REQ-118 より*

#### テストケース

##### 正常系

- [x] **TC-213-01**: 高LR+大Kで中間ロールバックが正常動作すること 🔵

---

### REQ-214~217: accel sweep並列・自動化・ダッシュボード 🔵

#### テストケース

##### 正常系

- [x] **TC-214-01**: 2-GPU並列で2設定が同時実行されること 🔵
- [x] **TC-216-01**: generate_sweep_dashboard.py が自己完結型HTMLを生成 🔵
- [x] **TC-217-01**: compare_paper_memory_modes.py がメモリメトリクス相対デルタを報告 🔵

---

### Phase 55テストケース数

| カテゴリ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| ユーティリティモジュール（REQ-205~209） | 8 | 1 | 0 | 9 |
| 運用スクリプト（REQ-210~217） | 6 | 0 | 0 | 6 |
| **Phase 55 合計** | **14** | **1** | **0** | **15** |

---

## テストケース総合サマリー（最終更新）

| フェーズ | 正常系 | 異常系 | 境界値 | 合計 |
|---------|--------|--------|--------|------|
| Phase 32 | 10 | 8 | 1 | 19 |
| Phase 32a | 4 | 3 | 1 | 8 |
| Phase 33 | 5 | 1 | 5 | 11 |
| Phase 34 | 9 | 0 | 5 | 14 |
| Phase 35 | 2 | 2 | 2 | 6 |
| Phase 36 | 3 | 0 | 0 | 3 |
| Phase 37 | 4 | 7 | 1 | 12 |
| Phase 38 | 4 | 2 | 0 | 6 |
| Phase 39 | 13 | 0 | 0 | 13 |
| Phase 40 | 3 | 0 | 0 | 3 |
| Phase 41 | 8 | 0 | 0 | 8 |
| Phase 42 | 1 | 0 | 13 | 14 |
| Phase 54 | 10 | 1 | 3 | 14 |
| Phase 55 | 14 | 1 | 0 | 15 |
| Phase 56 | 10 | 0 | 0 | 10 |
| Phase 59 | 5 | 0 | 0 | 5 |
| Phase 57 | 8 | 2 | 0 | 10 |
| Phase 58 | 10 | 0 | 1 | 11 |
| Phase 61 | 14 | 3 | 0 | 17 |
| **Phase 32~61 合計** | **137** | **30** | **32** | **199** |

---

## Phase 56: モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析（REQ-218~231）

### REQ-218: モデル構造検査ツール 🔵

**信頼性**: 🔵 *scripts/inspect_model.py 既存実装・README.md記載より*

### Given（前提条件）

- HuggingFaceモデル名またはYAML設定ファイルパスが指定されている

### When（実行条件）

- `scripts/inspect_model.py --model <model_name>` または `--config <yaml_path>` を実行する

### Then（期待結果）

- 全Linear層が名前パターンで列挙される
- target_modulesとして推奨されるモジュール一覧が出力される
- 重みなしモード（デフォルト）ではconfig.jsonのみで動作する

### テストケース

#### 正常系

- [x] **TC-218-01**: inspect_model.pyがQwen/Qwen3.5-9Bのモデル構造を正常に出力する 🔵
- [x] **TC-218-02**: --config引数でYAML設定経由のモデル検査が正常動作する 🔵

---

### REQ-219: モデル検査Makefileターゲット 🔵

**信頼性**: 🔵 *Makefile inspect/inspect-configターゲット既存実装より*

### テストケース

#### 正常系

- [x] **TC-219-01**: make inspectが正常終了する 🔵
- [x] **TC-219-02**: make inspect-configが正常終了する 🔵

---

### REQ-220: 比較ダッシュボード（マルチラン） 🔵

**信頼性**: 🔵 *scripts/compare_runs.py dashboardサブコマンド既存実装より*

### テストケース

#### 正常系

- [x] **TC-220-01**: dashboardサブコマンドが複数ランの比較テーブルを生成する 🔵
- [x] **TC-220-02**: --format jsonでJSON形式で出力される 🔵

---

### REQ-221: 比較可視化プロット関数 🔵

**信頼性**: 🔵 *scripts/compare_runs.py 5つのplot_*関数既存実装より*

### テストケース

#### 正常系

- [x] **TC-221-01**: plot_acceptance_rate, plot_reduction_rate, plot_velocity_magnitude, plot_layer_scores, plot_hyperparamsが正常にPNGを生成する 🔵

---

### REQ-224: ワンショットPrefix Feature Cache 🔵

**信頼性**: 🔵 *src/tg_lora/prefix_feature_cache.py MappedPrefixFeatureDataset disk-backed mode既存実装より*

### テストケース

#### 正常系

- [x] **TC-224-01**: prefix_feature_cache_mode="one_shot"でPrefixFeatureDatasetがdisk-backedとして構築される 🔵
- [x] **TC-224-02**: configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yamlがPydantic検証を通過する 🔵

---

### REQ-226: Prefix Cache損益分岐点分析 🔵

**信頼性**: 🔵 *scripts/analyze_prefix_cache_break_even.py 既存実装148行より*

### テストケース

#### 正常系

- [x] **TC-226-01**: analyze_prefix_cache_break_even.pyがbreak_even_cyclesを計算して出力する 🔵

---

## Phase 59: 学習軌跡分析・収束予測・早期停止推奨（REQ-232~236）

### REQ-232: TrajectoryAnalyzer コアモジュール 🔵

**信頼性**: 🔵 *新規実装 src/tg_lora/trajectory.py より*

### Given（前提条件）

- 学習サイクルのloss履歴（train_loss, valid_loss）が利用可能

### When（実行条件）

- TrajectoryAnalyzerにloss履歴を追加し、full_report()を実行する

### Then（期待結果）

- 収束推定（ConvergenceEstimate）が生成される
- 早期停止推奨（EarlyStopAdvice）が生成される
- 異常検知結果（anomalies）が生成される

### テストケース

#### 正常系

- [x] **TC-227-01**: TrajectoryAnalyzerが減少loss系列に対し負のloss_trend、正のconvergence_rateを返す 🔵
- [x] **TC-227-02**: 収束済みloss系列に対しconverged=Trueを返す 🔵
- [x] **TC-227-03**: 停滞loss系列に対しearly_stop.should_stop=Trueを返す 🔵
- [x] **TC-227-04**: 異常スパイク・反転・velocity発散を検出する 🔵

---

### REQ-233: analyze_trajectory.py CLIツール 🔵

**信頼性**: 🔵 *新規実装 scripts/analyze_trajectory.py より*

### テストケース

#### 正常系

- [x] **TC-228-01**: --from-losses引数でloss系列を直接指定して解析結果をJSON出力する 🔵

---

## Phase 60: Trajectory-Informed Adaptive Control（REQ-237~240）

### REQ-237: TrajectoryController コアモジュール 🔵

**信頼性**: 🔵 *新規実装 src/tg_lora/trajectory_controller.py より*

### Given（前提条件）

- RandomWalkControllerとTrajectoryAnalyzerが初期化済み
- 学習サイクルのloss履歴が利用可能

### When（実行条件）

- `trajectory_controller.record_cycle()` で各サイクルのlossを記録する

### Then（期待結果）

- CycleDecisionが返却される（提案・停止信号・異常検知・適応調整）
- 収束検知時にalpha_maxが減衰する
- 異常検知時にlr_reject_decayが減衰する
- 停滞検知時にalpha_maxが増加する

### テストケース

#### 正常系

- [x] **TC-229-01**: record_cycleがCycleDecisionを返す 🔵
- [x] **TC-229-02**: 収束検知時にパラメータ適応が発生する 🔵

---

### REQ-238: 異常検知パラメータ調整 🔵

**信頼性**: 🔵 *trajectory_controller.py _apply_trajectory_insights() より*

### テストケース

#### 正常系

- [x] **TC-230-01**: loss spikeでanomaly_detected=Trueが返る 🔵
- [x] **TC-230-02**: 異常検知時にlr_reject_decayが減衰する 🔵

---

### REQ-239: 早期停止信号伝播 🔵

**信頼性**: 🔵 *trajectory_controller.py early_stop連携より*

### テストケース

#### 正常系

- [x] **TC-231-01**: 停滞lossでshould_stop=Trueが返る 🔵
- [x] **TC-231-02**: 改善lossでshould_stop=Falseが維持される 🔵

---

### REQ-240: 状態エクスポート・復元 🔵

**信頼性**: 🔵 *trajectory_controller.py export_state()/restore_state() より*

### テストケース

#### 正常系

- [x] **TC-232-01**: export/restoreで軌跡と適応履歴が保存・復元される 🔵
- [x] **TC-232-02**: summary()が現在状態を正しく反映する 🔵

---

## Phase 57: 論文結果エクスポート・ハイパーパラメータ感度分析（REQ-241~244）

### REQ-241: 論文結果エクスポートツール 🔵

**信頼性**: 🔵 *scripts/export_paper_results.py 既存実装・paper_experiment_plan.md Stage 2-5出力要件より*

### Given（前提条件）

- aggregate_summary.jsonにper_seedまたはaggregateキーが含まれる

### When（実行条件）

- `python scripts/export_paper_results.py aggregate_summary.json --format all --output-dir paper_tables/` を実行する

### Then（期待結果）

- LaTeX形式のテーブルが生成される
- Markdown形式のテーブルが生成される
- CSV形式のファイルが生成される

### テストケース

#### 正常系

- [x] **TC-233-01**: load_aggregate()がper_seedキーを含むJSONを正常読み込みする 🔵
- [x] **TC-233-02**: generate_latex_table()が有効なLaTeXテーブル環境を出力する 🔵
- [x] **TC-233-03**: generate_markdown_table()がパイプ区切りMarkdownテーブルを出力する 🔵
- [x] **TC-233-04**: export_csv()がヘッダ付きCSVファイルを出力する 🔵

#### 異常系

- [x] **TC-233-E01**: 不正な構造のJSON入力時にValueErrorが送出される 🔵
- [x] **TC-233-E02**: 存在しないファイルパス指定時にFileNotFoundErrorが送出される 🔵

---

### REQ-243: ハイパーパラメータ感度分析 🔵

**信頼性**: 🔵 *scripts/analyze_sensitivity.py 既存実装・src.utils.run_query依存より*

### テストケース

#### 正常系

- [x] **TC-234-01**: load_sweep_results()がスイープ実験ディレクトリから結果を読み込む 🔵
- [x] **TC-234-02**: compute_correlation_matrix()がパラメータとメトリクス間のPearson相関を計算する 🔵
- [x] **TC-234-03**: rank_sensitivity()が平均絶対相関でパラメータをランク付けする 🔵
- [x] **TC-234-04**: generate_sensitivity_report()がJSONレポートをファイル出力する 🔵

---

## Phase 58: 学習サイクルヘルスモニタ・実験構成マトリクス比較（REQ-245~250）

### REQ-245: CycleMonitor コアモジュール 🔵

**信頼性**: 🔵 *src/tg_lora/cycle_monitor.py 既存実装・TrainingAdvisor基盤より*

### Given（前提条件）

- CycleMonitorが初期化済み（patience=5, spike_threshold=2.0）

### When（実行条件）

- `monitor.update(cycle_data)` で各サイクルのデータを渡す

### Then（期待結果）

- HealthReportが返却される（status, divergence, stagnation, recommendations）
- NaN/Inf値がcritical severityで検出される
- loss比率がspike_thresholdを超える場合にhigh severityで検出される
- patienceサイクル以上の改善がない場合に停滞が検出される

### テストケース

#### 正常系

- [x] **TC-235-01**: NaN値のlossでDivergenceReport.detected=True、severity="critical"が返る 🔵
- [x] **TC-235-02**: loss比率がspike_thresholdを超える場合にDivergenceReport.severity="high"が返る 🔵
- [x] **TC-235-03**: patienceサイクル以上改善がない場合にStagnationReport.detected=Trueが返る 🔵
- [x] **TC-235-04**: recommend_intervention()が発散・停滞状態に応じた介入推奨を返す 🔵
- [x] **TC-235-05**: health_summary()が現在状態の完全な辞書を返す 🔵

#### 境界値

- [x] **TC-235-B01**: patience < 1またはspike_threshold ≤ 0でValueErrorが送出される 🔵

---

### REQ-249: 実験構成マトリクス比較 🔵

**信頼性**: 🔵 *scripts/compare_experiment_configs.py 既存実装・src.utils.run_query依存より*

### テストケース

#### 正常系

- [x] **TC-236-01**: discover_experiments()がrunsディレクトリ配下の実験を自動検出する 🔵
- [x] **TC-236-02**: build_comparison_matrix()がComparisonMatrixを構築する 🔵
- [x] **TC-236-03**: rank_experiments()がbest_valid_lossで実験をランク付けする 🔵
- [x] **TC-236-04**: format_as_markdown()がMarkdownテーブルを出力する 🔵
- [x] **TC-236-05**: format_as_json()がJSON構造を出力する 🔵

---

## Phase 61: Training Advisor モジュール・CLI（REQ-251~258）

### REQ-251: TrainingAdvisor コアモジュール 🔵

**信頼性**: 🔵 *src/tg_lora/training_advisor.py 既存実装・CycleMonitor + TrajectoryAnalyzer統合より*

### Given（前提条件）

- TrainingAdvisorがAdvisorConfigで初期化済み
- 学習サイクルのメトリクス（train_loss等）が利用可能

### When（実行条件）

- `advisor.evaluate(cycle, train_loss=..., ...)` を実行する

### Then（期待結果）

- AdvisoryReportが返却される（overall_health, actions, summary, cycle_health, trajectory_summary）
- NaN/Inf検出時はrollback + reduce_lrのcriticalアクションが生成される
- 停滞検知時はincrease_kのhighアクションが生成される
- アクションはpriority（critical > high > medium > low）で順序付けされる

### テストケース

#### 正常系

- [x] **TC-237-01**: evaluate()がAdvisoryReportを返す 🔵
- [x] **TC-237-02**: NaN lossでrollback + reduce_lrのcriticalアクションが生成される 🔵
- [x] **TC-237-03**: 停滞検知時（stagnation_patience超過）にincrease_kアクションが生成される 🔵
- [x] **TC-237-04**: loss spike検知時にreduce_lrのhighアクションが生成される 🔵
- [x] **TC-237-05**: 降下トレンド検知時にincrease_lrのlowアクションが生成される 🔵
- [x] **TC-237-06**: top_action()が最高優先度アクションを返す 🔵

#### 異常系

- [x] **TC-237-E01**: confidenceが[0,1]範囲外の場合にValueErrorが送出される 🔵

---

### REQ-253: AdvisoryReport 🔵

**信頼性**: 🔵 *training_advisor.py AdvisoryReport dataclassより*

### テストケース

#### 正常系

- [x] **TC-238-01**: AdvisoryReportにoverall_health, actions, summary, timestampが含まれる 🔵
- [x] **TC-238-02**: overall_healthがhealthy/warning/criticalのいずれかになる 🔵

---

### REQ-254: AdvisorConfig 🔵

**信頼性**: 🔵 *training_advisor.py AdvisorConfig dataclassより*

### テストケース

#### 正常系

- [x] **TC-239-01**: AdvisorConfigのデフォルト値が正常に設定される（stagnation_patience=5等） 🔵

---

### REQ-257: advise_training.py CLI 🔵

**信頼性**: 🔵 *scripts/advise_training.py 既存実装・training_advisor.py統合より*

### テストケース

#### 正常系

- [x] **TC-240-01**: run_metrics.jsonl入力からAdvisoryReportがJSON出力される 🔵
- [x] **TC-240-02**: --jsonフラグなしで人間可読テキストが出力される 🔵
- [x] **TC-240-03**: -o引数でファイル出力される 🔵
- [x] **TC-240-04**: exit code 2がcritical training stateで返る 🔵

#### 異常系

- [x] **TC-240-E01**: 存在しないファイルパス指定時にexit code 1で終了する 🔵
- [x] **TC-240-E02**: cycle_stepレコードが含まれないJSONLでexit code 1で終了する 🔵

---

## Phase 62: PSA受入れ基準

### REQ-265: PSAPriorコアモジュール 🔵

**信頼性**: 🔵 *src/tg_lora/psa.py PSAPrior実装より*

### Given（前提条件）

- PSAPriorが初期化済み（history_length=6, gain=0.5）
- テンソルdeltaが利用可能

### When（実行条件）

- record_delta()でdeltaを記録
- extract_priors()でPC1方向を抽出
- amplify_gradients()で勾配を増幅

### Then（期待結果）

- 増幅後の勾配がG + gamma * <G, v_PSA> * v_PSA公式に従う
- PC1方向がpower iterationで正しく抽出される

### テストケース

#### 正常系

- [x] **TC-265-01**: record_delta()後にリングバッファにdeltaが記録される 🔵
- [x] **TC-265-02**: extract_priors()がPC1方向を返す 🔵
- [x] **TC-265-03**: amplify_gradients()が正しい増幅公式で勾配を変更する 🔵
- [x] **TC-265-04**: compute_gain_map()がlayer-type別gainを返す 🔵

#### 境界値

- [x] **TC-265-B01**: history_length未満のdelta記録ではextract_priors()が空priorを返す 🔵
- [x] **TC-265-B02**: warmup_steps未満ではshould_update()がFalseを返す 🔵
- [x] **TC-265-B03**: reset_priors()後にpriorと履歴がクリアされる 🔵

---

### REQ-266: PSA L2正則化 🔵

**信頼性**: 🔵 *psa.py extract_priors() l2_reg実装より*

### テストケース

- [x] **TC-266-01**: l2_reg > 0で前回priorからの乖離がペナルティされる 🔵
- [x] **TC-266-02**: l2_reg = 0でL2正則化が無効になる 🔵

---

### REQ-267: Layer-type-specific gain 🔵

**信頼性**: 🔵 *psa.py compute_gain_map()より*

### テストケース

- [x] **TC-267-01**: out_projテンソルにgain×1.2が適用される 🔵
- [x] **TC-267-02**: v_projテンソルにgain×1.1が適用される 🔵
- [x] **TC-267-03**: MLPテンソルにgain×0.7が適用される 🔵
- [x] **TC-267-04**: 未知テンソル名にデフォルトgain（1.0）が適用される 🔵

---

### REQ-272: RegimeDetector 🔵

**信頼性**: 🔵 *src/tg_lora/regime.py実装より*

### テストケース

- [x] **TC-272-01**: 減少loss系列でSTABLEが返る 🔵
- [x] **TC-272-02**: 横ばいloss系列でPLATEAUが返る 🔵
- [x] **TC-272-03**: 急激なloss変動でTRANSITIONが返る 🔵
- [x] **TC-272-04**: consume_reset_signal()がワンショットでTrueを返し、次回はFalseを返す 🔵
- [x] **TC-272-05**: min_history未満のデータでSTABLEが返る 🔵

---

### REQ-274: ActivationFingerprintTracker 🔵

**信頼性**: 🔵 *src/tg_lora/activation_regime.py実装より*

### テストケース

- [x] **TC-274-01**: 高cosine similarity（>0.95）でSTABLEが分類される 🔵
- [x] **TC-274-02**: 低cosine similarity（<0.5）でCHAOTICが分類される 🔵
- [x] **TC-274-03**: regime_inventoryが各レジームの割合を返す 🔵
- [x] **TC-274-04**: compute_regime_null_baseline()が時系列シャッフル結果を返す 🔵

---

### REQ-276: LAWAAverager 🔵

**信頼性**: 🔵 *src/tg_lora/weight_averaging.py実装より*

### テストケース

- [x] **TC-276-01**: record()後にスナップショットがバッファに追加される 🔵
- [x] **TC-276-02**: average_snapshot()がスライディングウィンドウの算術平均を返す 🔵
- [x] **TC-276-03**: window_size超過時に最古のエントリが破棄される 🔵
- [x] **TC-276-04**: evaluate_with_lawa()が評価後に元の重みを復元する 🔵
- [x] **TC-276-05**: start_cycle未満ではis_ready=Falseである 🔵

---

### REQ-278: LayerDeltaAnalysis 🔵

**信頼性**: 🔵 *src/tg_lora/layer_delta_analysis.py実装より*

### テストケース

- [x] **TC-278-01**: compute_rank1_dominance()が正しいPC1分散比率を返す 🔵
- [x] **TC-278-02**: compute_direction_stability()がPC1方向安定性を返す 🔵
- [x] **TC-278-03**: marchenko_pastur_expected_rank1()がランダムヌル期待値を返す 🔵
- [x] **TC-278-04**: classify_layer_type()が正しいレイヤータイプを返す 🔵
- [x] **TC-278-05**: group_by_layer_type()がタイプ別に集約する 🔵
- [x] **TC-278-06**: ゼロ行列のrank1_dominanceが0.0を返す 🔵

---

### REQ-280~282: PSAアブレーションスクリプト 🔵

**信頼性**: 🔵 *scripts/run_psa_ablation.sh/run_psa_gamma_sweep.sh/summarize_psa_sweep.pyより*

### テストケース

- [x] **TC-280-01**: run_psa_ablation.shの3条件が順次実行される 🔵
- [x] **TC-281-01**: run_psa_gamma_sweep.shのγスイープが実行される 🔵
- [x] **TC-282-01**: summarize_psa_sweep.pyが結果を集約する 🔵

---

### Phase 62 境界値テスト

- [x] **TC-EDGE-202**: PSAPrior(history_length=0)がValueErrorを送出する 🔵
- [x] **TC-EDGE-203**: PSAPrior(gain=-0.1)がValueErrorを送出する 🔵
- [x] **TC-EDGE-204**: RegimeDetector min_history未満でSTABLEが返る 🔵
- [x] **TC-EDGE-205**: hook未登録でstep()がエラーなくスキップする 🔵
- [x] **TC-EDGE-206**: LAWAAverager(window_size=0)がValueErrorを送出する 🔵
- [x] **TC-EDGE-207**: 空バッファでaverage_snapshot()が None を返す 🔵
- [x] **TC-EDGE-208**: ゼロ行列でrank1_dominanceが0.0を返す 🔵
- [x] **TC-EDGE-209**: 未知テンソル名でLayerType.UNKNOWNが返る 🔵
- [x] **TC-EDGE-210**: PSAConfig gain<0 がPydantic ValidationErrorで拒否される（gain=0.0はγスイープ基準として許可） 🔵
- [x] **TC-EDGE-211**: 9b_tg_lora_psa.yamlがPydantic検証を通過する 🔵


<!-- spine:references:begin -->
## Spine: external references

- [TASK-0007: 受け入れ基準の全テストケース検証](tasks/TASK-0007.md)
- [TASK-0011: Phase 2 受け入れ基準・ドキュメント更新](tasks/TASK-0011.md)
- [TASK-0015: Phase 3 受け入れ基準・ドキュメント更新](tasks/TASK-0015.md)
- [TASK-0019: Phase 4 受け入れ基準・ドキュメント更新](tasks/TASK-0019.md)
- [TASK-0068: OptimizerLifecycleManager E2E スモークテスト](tasks/TASK-0068.md)
- [TASK-0070: Phase 30 ドキュメント更新](tasks/TASK-0070.md)
- [TASK-0076: REQ-136~138 acceptance criteria追加](tasks/TASK-0076.md)
- [TASK-0089: Phase 38 ドキュメント同期と受け入れ基準更新](tasks/TASK-0089.md)
- [TASK-0090: --resume E2E統合テスト（save→interrupt→resume→verify loss）](tasks/TASK-0090.md)
- [TASK-0102: NaN/Inf バリデーションとランタイムガード完全化](tasks/TASK-0102.md)
- [TASK-0112: モデル検査・比較ダッシュボード・ワンショットキャッシュのacceptance criteria追加](tasks/TASK-0112.md)
- [TASK-0113: コスト分析・データ細粒度・クリーンアップターゲットのテスト追加](tasks/TASK-0113.md)
- [TASK-0119: 学習軌跡分析・収束予測・早期停止推奨](tasks/TASK-0119.md)
- [TASK-0120: 軌跡連動適応制御モジュール・テスト](tasks/TASK-0120.md)
- [TASK-0121: Training Advisor モジュール・CLI](tasks/TASK-0121.md)

<!-- spine:references:end -->
