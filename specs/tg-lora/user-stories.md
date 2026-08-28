# TG-LoRA ユーザストーリー


<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-05-21
**関連要件定義**: [requirements.md](requirements.md)
**分析記録**: [interview-record.md](interview-record.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: PRD・既存要件定義書・設計文書・既存実装を参考にした確実なストーリー
- 🟡 **黄信号**: PRD・既存要件定義書・設計文書・既存実装から妥当な推測によるストーリー
- 🔴 **赤信号**: 参照資料にない自動推定によるストーリー

---

## エピック1: TG-LoRA学習の実行

### ストーリー 1.1: TG-LoRA学習で外挿ベースの効率化を実行する 🔵

**信頼性**: 🔵 *train_tg_lora.py・AGENTS.mdより*

**私は** ML研究者 **として**
**TG-LoRA学習を実行し、速度ベクトル外挿で学習効率を向上させたい**
**そうすることで** 同等の計算予算でより低い損失を達成できる

**関連要件**: REQ-001, REQ-002, REQ-016

**詳細シナリオ**:

1. configs/9b_tg_lora.yaml で学習設定を指定
2. `make train-tg-lora` で学習を開始
3. 各サイクルでK歩のpilot学習 → 速度更新 → 外挿適用 → 受理/拒否
4. 学習終了後、チェックポイントをruns/に保存

**前提条件**:

- ベースモデル（Qwen3.5-9B）がHuggingFaceからダウンロード可能
- 学習データ（data/train.jsonl）が準備済み
- GPU環境（RTX3060 12GB以上）が利用可能

**制約事項**:

- 1サイクル = K歩のbackward pass + N歩の外挿
- 外挿更新はrelative_update_capで制限

**優先度**: Must Have

---

### ストーリー 1.2: ベースラインQLoRA学習を実行する 🔵

**信頼性**: 🔵 *train_baseline_qlora.py・Makefileより*

**私は** ML研究者 **として**
**標準QLoRA学習を実行し、TG-LoRAとの比較基準を作りたい**
**そうすることで** TG-LoRAの効率化効果を定量的に評価できる

**関連要件**: REQ-015

**詳細シナリオ**:

1. configs/9b_baseline.yaml でベースライン設定を指定
2. `make train-baseline` で学習を開始
3. max_steps歩の学習を実行し、定期評価・チェックポイント保存
4. 学習終了後、runs/qlora_9b_baseline/に最良モデルを保存

**前提条件**:

- ベースモデルがHuggingFaceからダウンロード可能
- 学習データが準備済み

**優先度**: Must Have

---

### ストーリー 1.3: ハイパーパラメータを自動探索する 🔵

**信頼性**: 🔵 *random_walk_controller.py 既存実装より*

**私は** ML研究者 **として**
**TG-LoRAのハイパーパラメータ（K, N, alpha, beta）を自動探索したい**
**そうすることで** 手動チューニングの手間を減らし、最適な学習設定を見つけられる

**関連要件**: REQ-011, REQ-012, REQ-013, REQ-013a, REQ-014, REQ-053, REQ-054

**詳細シナリオ**:

1. 初期ハイパーパラメータをconfigで指定（K, N, alpha, beta, lr）
2. 各サイクルでRandomWalkControllerが新しい提案を生成
3. pilot損失と外挿後損失を比較し、受理/拒否を判定
4. 受理時にalphaを増加、lrを増加（lr_accept_boost倍）
5. 拒否時にalphaを減少、lrを減少（lr_reject_decay倍）
6. 探索停滞時（convergence_trend >= 0）にproactiveにlr減少・K増加
7. lrは常に[lr_min, lr_max]の範囲内にクランプ
8. 受理率を追跡し、学習終了時にサマリーを出力

**前提条件**:

- RandomWalkControllerが初期化済み
- 許容率（rollback_tolerance）が設定済み

**優先度**: Must Have

---

### ストーリー 1.4: velocity magnitudeの異常と傾向を監視する 🔵

**信頼性**: 🔵 *velocity.py magnitude history・anomaly detection・trend実装(bbcb7e7)より*

**私は** ML研究者 **として**
**TG-LoRA学習中のvelocity magnitudeの異常と収束/発散傾向をリアルタイムに監視したい**
**そうすることで** 学習の安定性を早期に評価し、発散リスクを検知できる

**関連要件**: REQ-001, REQ-049, REQ-050

**詳細シナリオ**:

1. 各velocity update後にL2 normをmagnitude historyに記録
2. is_magnitude_anomalousで外れ値magnitudeを検出（σ閾値ベース）
3. magnitude_trendで直近window件の線形回帰傾きを計算
4. 異常検出・傾向情報を学習サマリーに含めて出力

**前提条件**:

- Velocityインスタンスがmax_history付きで初期化済み
- 学習ループが各サイクルでvelocity.updateを呼び出している

**優先度**: Must Have

---

### ストーリー 1.5: 学習データの品質をPydanticスキーマで検証する 🔵

**信頼性**: 🔵 *schema.py DataRecord/ValidationSummary実装・test_schema.pyより*

**私は** ML研究者 **として**
**ChatML形式学習データの品質をPydanticスキーマで自動検証したい**
**そうすることで** 不正なデータが学習パイプラインに混入するのを防げる

**関連要件**: REQ-051, REQ-052

**詳細シナリオ**:

1. DataRecordでtext必須・非空・ChatMLマーカー含有を検証
2. token_count正値制約を適用
3. validate_recordsでバッチ検証し、ValidationSummaryに結果を集計
4. 無効レコードをスキップし、エラー理由をログ出力

**前提条件**:

- Pydantic>=2.5がインストール済み
- 生データがdictリストとして読み込み可能

**優先度**: Must Have

---

## エピック2: 学習安定性の確保

### ストーリー 2.1: 外挿失敗時に自動ロールバックする 🔵

**信頼性**: 🔵 *rollback_manager.py・train_tg_lora.pyより*

**私は** ML研究者 **として**
**外挿によって損失が悪化した場合、または外挿後に非有限パラメータが発生した場合、自動的にロールバックしたい**
**そうすることで** 学習が発散するリスクを防ぎ、安定した学習を維持できる

**関連要件**: REQ-009, REQ-010, REQ-201, REQ-056, REQ-059

**詳細シナリオ**:

1. 外挿適用前にRollbackManagerでLoRA状態を保存
2. 外挿後の損失を評価
3. 損失がpilot損失×許容率を超過した場合、保存した状態にロールバック
4. 外挿後にLoRAパラメータがNaN/Infの場合、ロールバック→penalize→record_cycleで回復
5. try/finallyで例外発生時も確実にロールバック
6. 統合テストで完全な回復フローを検証（REQ-059）

**前提条件**:

- RollbackManagerが初期化済み
- スナップショット保存用メモリが確保されている

**優先度**: Must Have

---

### ストーリー 2.2: レイヤーサンプリングで計算を効率化する 🔵

**信頼性**: 🔵 *layer_sampler.py 既存実装より*

**私は** ML研究者 **として**
**外挿対象レイヤーを選別し、重要なレイヤーに集中して更新したい**
**そうすることで** 外挿の計算コストを削減しつつ、効果的な学習ができる

**関連要件**: REQ-006, REQ-007, REQ-008

**詳細シナリオ**:

1. 設定でレイヤー戦略（last_25_percent_plus_random_2等）を指定
2. 各サイクルで選択されたレイヤーのみ外挿を適用
3. 受理/拒否に応じてレイヤー重要度スコアを更新
4. lisa_like_weighted戦略ではスコアベースの重み付きサンプリングを実行

**前提条件**:

- モデルにLoRAアダプタが適用済み
- レイヤー数が自動検出されている

**優先度**: Should Have

---

### ストーリー 2.3: 学習が改善しない場合に早期終了する 🔵

**信頼性**: 🔵 *train_tg_lora.py 早期終了実装より*

**私は** ML研究者 **として**
**損失が改善しない場合に学習を自動終了したい**
**そうすることで** 不要な計算コストを削減できる

**関連要件**: REQ-020, REQ-202

**詳細シナリオ**:

1. 一定サイクル数で最良損失を追跡
2. 指定サイクル数以上改善がない場合、学習を終了
3. 最良モデルのチェックポイントを保持

**前提条件**:

- 評価用データが準備されている
- 早期終了のpatienceが設定されている

**優先度**: Should Have

---

### ストーリー 2.4: 学習サイクルの集計状態を追跡する 🔵

**信頼性**: 🔵 *cycle_state.py・test_cycle_state.py より*

**私は** ML研究者 **として**
**TG-LoRA学習の各サイクルでbackward pass数・外挿数・受理率・削減率を追跡したい**
**そうすることで** 学習の効率性をリアルタイムで評価し、早期終了の判断ができる

**関連要件**: REQ-038, REQ-039, REQ-040

**詳細シナリオ**:

1. CycleStateが各サイクルでK, N, grad_accum, train_loss, valid_loss, acceptedを記録
2. 削減率（1 − backward/(backward+extrap)）と受理率が自動計算される
3. patience-based早期終了がstale_cyclesで判定される
4. summary()で全統計を一括取得

**前提条件**:

- 学習ループがCycleStateを使用するよう統合済み（train_tg_lora.py）

**優先度**: Must Have

---

### ストーリー 2.5: 重み差分の統計を記録・監視する 🔵

**信頼性**: 🔵 *delta_tracker.py・test_delta_tracker.py より*

**私は** ML研究者 **として**
**各サイクルのLoRA重み差分の統計（norm、per-layer分析、異常検出、収束傾向）を追跡したい**
**そうすることで** 学習の健全性を監視し、発散や異常を早期に検出できる

**関連要件**: REQ-041, REQ-042, REQ-043

**詳細シナリオ**:

1. DeltaTrackerが各サイクルの重み差分をcompute_and_recordで記録
2. total_norm, per_layer_norm, max_component, mean_absを自動計算
3. is_anomalous()で外れ値deltaを検出（σ閾値ベース）
4. convergence_trend()で収束・発散の傾向を定量化
5. summary()で全統計を一括取得

**前提条件**:

- LoRAスナップショット（W0, WK）が取得可能
- DeltaTrackerが学習ループに統合済み

**優先度**: Must Have

---

## エピック3: データ準備と評価

### ストーリー 3.1: 公開データセットをダウンロード・前処理する 🔵

**信頼性**: 🔵 *download_data.py・prepare_data.py・docs/datasets.mdより*

**私は** ML研究者 **として**
**公開データセット（Dolly 15k, Capybara）をダウンロードして学習可能な形式に変換したい**
**そうすることで** 初期検証フェーズでアルゴリズムの動作を確認できる

**関連要件**: REQ-025, REQ-026

**詳細シナリオ**:

1. `make download-data` でHuggingFaceからデータセットをダウンロード
2. `make prepare-data` でChatML形式JSONLに変換・分割
3. data/ 配下にtrain.jsonl, valid_quick.jsonl, valid_full.jsonl, gold_test.jsonlを生成

**前提条件**:

- HuggingFaceへのアクセスが可能
- ディスク容量が十分

**優先度**: Must Have

---

### ストーリー 3.2: ベンチマークで学習結果を定量評価する 🔵

**信頼性**: 🔵 *run_eval.sh・docs/evaluation.mdより*

**私は** ML研究者 **として**
**標準ベンチマーク（ARC, HellaSwag, GSM8K, TruthfulQA）でモデル性能を評価したい**
**そうすることで** TG-LoRAの効果を客観的に測定できる

**関連要件**: REQ-035

**詳細シナリオ**:

1. `make eval MODEL_PATH=runs/...` でlm-evaluation-harnessを実行
2. 4タスク（ARC-Easy, HellaSwag, GSM8K, TruthfulQA MC2）で評価
3. 結果をreports/eval/に保存

**前提条件**:

- lm-evalがインストール済み
- 評価対象のモデルチェックポイントが存在

**優先度**: Must Have

---

### ストーリー 3.3: ベースラインとTG-LoRAを公正に比較する 🔵

**信頼性**: 🔵 *run_comparison.sh・compare_runs.pyより*

**私は** ML研究者 **として**
**同一の計算予算でベースラインとTG-LoRAを比較し、効率性を定量化したい**
**そうすることで** TG-LoRAの優位性をデータで示せる

**関連要件**: REQ-036, REQ-037

**詳細シナリオ**:

1. `make compare BUDGET=1500` で比較実験を実行
2. ベースライン: 1500 backward pass の標準QLoRA
3. TG-LoRA: 1500/K_initial サイクル（各サイクルK歩のpilot + N歩の外挿）
4. 比較レポート（損失曲線、効率メトリクス、受理率）を生成

**前提条件**:

- 両方の実験設定がconfigs/に存在
- 十分なGPU計算時間が確保できる

**優先度**: Must Have

---

## エピック4: データ品質管理

### ストーリー 4.1: 合成データの品質をフィルタリングする 🔵

**信頼性**: 🔵 *filter_dataset.py・dedup.py既存実装より*

**私は** ML研究者 **として**
**生成した合成データから低品質・重複を除去したい**
**そうすることで** 学習データの品質を担保できる

**関連要件**: REQ-029, REQ-030

**詳細シナリオ**:

1. テキスト長によるフィルタリング（min_length, max_length）
2. 品質スコアによるフィルタリング
3. 完全一致による重複排除
4. 埋め込みベースの意味的重複排除（FAISS/numpy）

**前提条件**:

- sentence-transformersがインストール済み
- FAISSが利用可能（オプション）

**優先度**: Should Have

---

### ストーリー 4.2: データ来歴を追跡する 🔵

**信頼性**: 🔵 *provenance.py既存実装より*

**私は** ML研究者 **として**
**各データの生成元・品質スコア・レビュー状況を記録したい**
**そうすることで** データの信頼性を追跡・管理できる

**関連要件**: REQ-031

**詳細シナリオ**:

1. データ生成時に来歴メタデータを自動付与
2. 生成モデル名、ソース種別、品質スコアを記録
3. レビュー状況（auto_passed等）を管理

**優先度**: Should Have

---

## エピック5: 数値信頼性と設定安全性

### ストーリー 5.1: 設定タイポを学習開始前に検出する 🔵

**信頼性**: 🔵 *config_schema.py extra='forbid'・test_config_schema.py TestExtraFieldsRejected 8テストより*

**私は** ML研究者 **として**
**YAML設定ファイルのタイポや未知フィールドを学習開始前に検出したい**
**そうすることで** 数時間のGPU学習が無駄になるのを防げる

**関連要件**: REQ-061, REQ-062, EDGE-126, EDGE-127

**詳細シナリオ**:

1. configs/*.yamlにタイポ（例: "lerning_rate"）を記述
2. load_and_validate_config()がPydantic ValidationErrorで拒否
3. 空ファイルやリスト形式のYAMLもValueErrorで拒否
4. 全11モデルでextra='forbid'により未知フィールドを一括検出

**前提条件**:

- Pydantic>=2.5がインストール済み
- config_schema.pyが全モデルにextra='forbid'を設定済み

**優先度**: Must Have

---

### ストーリー 5.2: NaN/Infの伝播をシステム全体で防止する 🔵

**信頼性**: 🔵 *extrapolator.py cap_update・rollback_manager.py _sanitize_snapshot・delta_tracker.py _compute_stats・metrics.py total_norm/per_layer_norms・train_tg_lora.py _compute_pilot_average・test_* テストより*

**私は** ML研究者 **として**
**学習中にNaN/Infが発生してもシステムが自動的にサニタイズ・回復してほしい**
**そうすることで** 長時間の学習ジョブが数値エラーでクラッシュしない

**関連要件**: REQ-063, REQ-064, REQ-065, REQ-066, REQ-067, REQ-068, REQ-089, REQ-090, REQ-091, EDGE-128~134, EDGE-146~149

**詳細シナリオ**:

1. cap_update()が非有限更新をゼロに置換（NaN伝播防止）
2. RollbackManagerがスナップショット保存時にNaN/Infをサニタイズ
3. RollbackManagerが履歴サイズをmax_history=100で制限（メモリ肥大化防止）
4. metrics.cosine_similarity()がキー不一致を安全にスキップ
5. DeltaTrackerが非有限ノルムをスキップしnorm_historyを保護
6. 全てのNaN/Inf経路でテストカバレッジあり（27テスト + metrics/pilot average 7テスト = 34テスト）

**前提条件**:

- 各モジュールに数値安全性ガードが実装済み
- テストスイートに非有限値テストが含まれている

**優先度**: Must Have

---

## エピック6: Phase 28 評価最適化・決定論的モード・高度Accept/Rollback

### ストーリー 6.1: ActivationCacheで評価計算を最適化する 🔵

**信頼性**: 🔵 *src/tg_lora/activation_cache.py実装・0bc7236コミットより*

**私は** ML研究者 **として**
**外挿後の評価でキャッシュされた隠れ状態を再利用し、計算コストを削減したい**
**そうすることで** 32層モデルで約75%の評価FLOPsを削減できる

**関連要件**: REQ-110, REQ-111, REQ-112

**詳細シナリオ**:

1. ActivationCacheがスプリットレイヤーの隠れ状態をキャッシュ
2. 外挿後評価でキャッシュから部分フォワードのみ実行
3. 予測戦略が実際と一致する場合はキャッシュを再利用
4. decoderレイヤーが検出できない場合は非キャッシュ評価にフォールバック

**前提条件**:

- モデルにdecoderレイヤーが存在する
- force_top_layers_onlyでスプリットレイヤーが一貫している

**優先度**: Must Have

---

### ストーリー 6.2: 決定論的モードで再現性のある学習を実行する 🔵

**信頼性**: 🔵 *random_walk_controller.py enable_random_walk・train_tg_lora.py force_top_layers_only・0782acd/555287dコミットより*

**私は** ML研究者 **として**
**ハイパーパラメータ探索を無効化し、決定論的な学習を実行したい**
**そうすることで** 同一設定で常に同じ結果が得られ、実験の再現性が保証される

**関連要件**: REQ-113, REQ-114

**詳細シナリオ**:

1. enable_random_walk=falseでK, N, alpha, beta, lrが固定
2. force_top_layers_only=trueで常に"last_25_percent"戦略を使用
3. ActivationCacheのスプリットレイヤーが一貫
4. 探索確率パラメータが0.0に設定される

**前提条件**:

- 初期ハイパーパラメータが適切に設定されている
- seed.pyで乱数シードが固定されている

**優先度**: Should Have

---

### ストーリー 6.3: 移動平均ベースラインとsoft acceptでノイズに強い判定を行う 🔵

**信頼性**: 🔵 *train_tg_lora.py _decide_accept_rollback()・555287d/d3f834bコミットより*

**私は** ML研究者 **として**
**評価ノイズによる誤ったrejectを減らし、境界ケースを適切に処理したい**
**そうすることで** 学習効率が向上し、局所最適解から脱出しやすくなる

**関連要件**: REQ-115, REQ-116

**詳細シナリオ**:

1. accepted_valid_historyの移動平均をベースラインとして使用
2. loss_afterがベースライン以下なら受理
3. soft_accept_temperature > 0の場合、Metropolis-Hastings確率で境界受理
4. temperature = 0の場合は確率的受理が無効

**前提条件**:

- moving_avg_window >= 1が設定されている
- accepted_valid_historyに記録が蓄積されている

**優先度**: Should Have

---

### ストーリー 6.4: K-step中間ロールバックでdivergenceから回復する 🔵

**信頼性**: 🔵 *train_tg_lora.py intermediate_deltas・64bd8a8/a1ffe6dコミットより*

**私は** ML研究者 **として**
**pilotフェーズで学習が発散した場合、最良の中間点にロールバックしたい**
**そうすることで** 発散時の損失を最小限に抑え、学習の無駄を削減できる

**関連要件**: REQ-118

**詳細シナリオ**:

1. pilot Kステップの各ステップでdelta snapshotを記録
2. pilot損失が直近valid損失+toleranceを超過した場合に中間ロールバック発動
3. 全中間点をeval_lossで評価し最良点を選択
4. 最良点にロールバックし、dWとvelocityを再計算
5. 全中間点が悪化している場合はW0にフルロールバック

**前提条件**:

- intermediate_deltasが各pilot stepで記録されている
- 評価用データローダーが利用可能

**優先度**: Should Have

---

### ストーリー 6.5: velocityが安定している時に評価を省略する 🔵

**信頼性**: 🔵 *train_tg_lora.py confident_skip・555287dコミットより*

**私は** ML研究者 **として**
**velocity方向が安定している場合、不要な評価を省略したい**
**そうすることで** 安定期の計算コストを削減しつつ、不安定時には確実に評価できる

**関連要件**: REQ-117

**詳細シナリオ**:

1. confident_skip_cos > 0が設定されている
2. cos_sim >= threshold かつ acceptance_rate >= 0.8 かつ total_cycles >= min_cycles
3. velocity.magnitudeが非異常であることを確認
4. 全条件満た時: loss_after = loss_pilotとして自動受理
5. 一つでも条件を満たさない場合は通常評価を実行

**前提条件**:

- confident_skip_cosが適切な閾値（推奨: 0.92~0.96）に設定されている
- 十分なサイクル数が経過している

**優先度**: Could Have

---

## エピック7: Prefix Feature Cache堅牢性

### ストーリー 7.1: Prefix Feature Cacheの堅牢性を包括的にテストする 🔵

**信頼性**: 🔵 *prefix_feature_cache.py既存実装・design-interview A27改善推奨・AI_HUB_MAKE_RUN_FEEDBACK指摘より*

**私は** ML研究者 **として**
**破損キャッシュ・force_rebuild・position_ids等のエッジケースでもprefix feature cacheが安全に動作することを確認したい**
**そうすることで** キャッシュの永続化に頼る長時間学習で予期しないクラッシュを防げる

**関連要件**: REQ-128, REQ-129, REQ-130, REQ-131, REQ-132, REQ-133, REQ-134, REQ-135

**詳細シナリオ**:

1. 破損キャッシュファイル（部分書き込み、不正フォーマット、欠落キー）でエラー検出を確認
2. force_rebuild=trueで既存キャッシュをスキップして再ビルドすることを確認
3. position_ids付きデータセットでビルド・保存・読込が正しく動作することを確認
4. ビルド中の例外発生時にmodel.training状態が復元されることを確認
5. SHA-256ハッシュ変更でキャッシュパスが変わることを確認
6. format_version不一致でValueErrorが送出されることを確認
7. 空データセットでsaveが拒否されることを確認
8. compare-prefix-coldwarmが正常にexit code 0で完了し、cache hit/missが期待通りであることを確認

**前提条件**:

- prefix_feature_cache.pyの全関数が利用可能
- Makefile compare-prefix targetsが定義済み
- GPT-2 tinyモデル等の軽量モデルがテストで利用可能

**優先度**: Must Have

---

## エピック8: In-place最適化とパフォーマンス監視

### ストーリー 8.1: In-place演算でメモリアロケーションを最適化する 🔵

**信頼性**: 🔵 *velocity.py mul_/add_ EMA・extrapolator.py mul_ capping・851041e/c9928b6コミットより*

**私は** ML研究者 **として**
**velocity EMA更新とcap_updateでin-place演算を使用し、メモリアロケーションを削減したい**
**そうすることで** 長時間学習でのGCプレッシャーを軽減し、学習スループットを向上できる

**関連要件**: REQ-144, REQ-145, REQ-146

**詳細シナリオ**:

1. velocity.update()が既存キーにmul_(beta).add_(delta, alpha)でin-place EMA更新
2. cap_update()がcapping時にmul_でin-placeスケーリング
3. data_ptrが更新前後で同一であることをテストで検証
4. 新規キーはclone()で別テンソル、非有限入力は新規ゼロテンソル返却

**前提条件**:

- velocity.py/extrapolator.pyのin-place演算が実装済み
- data_ptr保存テストが実装済み（9テスト）

**優先度**: Must Have

---

### ストーリー 8.2: Velocity opsマイクロベンチマークで性能回帰を監視する 🔵

**信頼性**: 🔵 *benchmark_velocity_ops.py・c51fd5bコミット(TASK-0080)・AI_HUB_MAKE_RUN_FEEDBACK指摘より*

**私は** ML研究者 **として**
**velocity EMAとcap_updateのマイクロベンチマークを実行し、性能回帰を継続的に監視したい**
**そうすることで** コード変更による性能劣化を早期に検出できる

**関連要件**: REQ-147, REQ-148

**詳細シナリオ**:

1. benchmark_velocity_ops.py --quickで10反復のスモークベンチマークを実行
2. JSON形式でper-iter timeとメモリ使用量を出力
3. Makefile bench-velocity-opsターゲットから実行可能
4. CI gateで性能回帰閾値を自動判定（将来拡張）

**前提条件**:

- benchmark_velocity_ops.pyが実装済み
- Makefile bench-velocity-opsターゲットが定義済み（REQ-148）

**優先度**: Must Have

---

### ストーリー 1.6: velocity magnitudeの加速的変化を早期検出する 🔵

**信頼性**: 🔵 *velocity.py magnitude_acceleration() 実装・TASK-0090より*

**私は** ML研究者 **として**
**velocity magnitudeの二階微分（加速度）を監視し、加速的な不安定化を早期に検出したい**
**そうすることで** 発散が始まる前に予防的な対応（lr減少等）ができる

**関連要件**: REQ-153

**詳細シナリオ**:

1. magnitude_acceleration()で直近window件のmagnitudeの二階微分を計算
2. 正の加速度 → magnitudeが加速的に増大 → 潜在的不安定性の兆候
3. 負の加速度 → magnitude増大が減速 → 収束傾向
4. 3件未満のデータでは0.0を返す（判定不可）

**前提条件**:

- Velocityインスタンスがmax_history付きで初期化済み
- 各サイクルでvelocity.updateが呼び出されている

**優先度**: Should Have

---

## ストーリー 5.3: 入力検証の強化でランタイムエラーを防止する 🔵

**信頼性**: 🔵 *delta_tracker.py/rollback_manager.py/lora_state.py/random_walk_controller.py 入力検証・TASK-0090/0091より*

**私は** ML研究者 **として**
**不正な入力（空ベース、キー不一致、不正max_history）を学習開始前または初期段階に検出したい**
**そうすることで** 数時間のGPU学習がランタイムエラーで無駄になるのを防げる

**関連要件**: REQ-155, REQ-156, REQ-157, REQ-158

**詳細シナリオ**:

1. DeltaTracker.compute_and_record()がafter/beforeキー不一致をValueErrorで検出
2. RollbackManager(max_history=0)がValueErrorで拒否
3. snapshot_lora_delta()が空baseをValueErrorで拒否
4. propose()がlog-normal探索のOverflowErrorをクランプで防止

**前提条件**:

- 各モジュールの入力検証が実装済み
- テストスイートに検証テストが含まれている

**優先度**: Must Have

---

## ストーリーマップ

```
エピック1: TG-LoRA学習の実行
├── ストーリー 1.1 (🔵 Must Have) - TG-LoRA学習実行
├── ストーリー 1.2 (🔵 Must Have) - ベースライン学習実行
├── ストーリー 1.3 (🔵 Must Have) - ハイパーパラメータ自動探索
├── ストーリー 1.4 (🔵 Must Have) - velocity magnitude異常・傾向監視
├── ストーリー 1.5 (🔵 Must Have) - 学習データPydanticスキーマ検証
└── ストーリー 1.6 (🔵 Should Have) - velocity magnitude加速的変化の早期検出

エピック2: 学習安定性の確保
├── ストーリー 2.1 (🔵 Must Have)    - 自動ロールバック
├── ストーリー 2.2 (🔵 Should Have)   - レイヤーサンプリング
├── ストーリー 2.3 (🔵 Should Have)   - 早期終了
├── ストーリー 2.4 (🔵 Must Have)     - サイクル状態追跡（CycleState）
└── ストーリー 2.5 (🔵 Must Have)     - 重み差分統計・異常検出（DeltaTracker）

エピック3: データ準備と評価
├── ストーリー 3.1 (🔵 Must Have) - データダウンロード・前処理
├── ストーリー 3.2 (🔵 Must Have) - ベンチマーク評価
└── ストーリー 3.3 (🔵 Must Have) - 公正比較実験

エピック4: データ品質管理
├── ストーリー 4.1 (🔵 Should Have) - 品質フィルタリング・重複排除
└── ストーリー 4.2 (🔵 Should Have) - データ来歴追跡

エピック5: 数値信頼性と設定安全性
├── ストーリー 5.1 (🔵 Must Have) - 設定タイポ検出（extra='forbid'）
├── ストーリー 5.2 (🔵 Must Have) - NaN/Inf伝播防止（システム全体）
└── ストーリー 5.3 (🔵 Must Have) - 入力検証強化によるランタイムエラー防止

エピック6: Phase 28 評価最適化・決定論的モード・高度Accept/Rollback
├── ストーリー 6.1 (🔵 Must Have)     - ActivationCache評価最適化
├── ストーリー 6.2 (🔵 Should Have)   - 決定論的モード（enable_random_walk）
├── ストーリー 6.3 (🔵 Should Have)   - 移動平均ベースライン・soft accept
├── ストーリー 6.4 (🔵 Should Have)   - K-step中間ロールバック
└── ストーリー 6.5 (🔵 Could Have)    - confident-skip評価省略

エピック7: Prefix Feature Cache堅牢性
└── ストーリー 7.1 (🔵 Must Have)     - キャッシュ堅牢性・比較実験smoke test

エピック8: In-place最適化とパフォーマンス監視
├── ストーリー 8.1 (🔵 Must Have)     - In-place EMA/cap_update最適化
└── ストーリー 8.2 (🔵 Must Have)     - Velocity opsマイクロベンチマーク

エピック9: 障害回復と観測性
├── ストーリー 9.1 (🔵 Must Have)     - --resume障害チェックポイント再開
├── ストーリー 9.2 (🔵 Must Have)     - 加速度適応MLflow監視
├── ストーリー 9.3 (🔵 Must Have)     - E2E resumeフロー検証
├── ストーリー 9.4 (🔵 Must Have)     - ベンチマーク品質ギャップ分析
└── ストーリー 9.5 (🔵 Must Have)     - accel paramスイープ実験
```

## 信頼性レベルサマリー

- 🔵 青信号: 31件 (100%)
- 🟡 黄信号: 0件 (0%)
- 🔴 赤信号: 0件 (0%)

**品質評価**: 高品質 — 全ストーリーが既存実装に基づいており、推測に依存する項目なし

---

## エピック9: 障害回復と観測性

### ストーリー 9.1: 障害チェックポイントから学習を再開する 🔵

**信頼性**: 🔵 *train_tg_lora.py --resume・random_walk_controller.py restore_state()・9f195f0コミットより*

**私は** ML研究者 **として**
**OOM/CUDAエラーで中断された学習を--resumeで再開し、失われたサイクルを再実行せずに続きから学習したい**
**そうすることで** 長時間の学習ジョブがハードウェアエラーで無駄にならず、復旧後も継続的に改善を追跡できる

**関連要件**: REQ-162, REQ-163, REQ-164, EDGE-183, EDGE-184

**詳細シナリオ**:

1. 学習中にOOM/CUDAエラーで中断される（training_state.ptが自動保存される）
2. `--resume runs/<exp>/training_state.pt` で再開
3. load_training_state()でcontroller/velocity/delta_tracker/cycle_stateを復元
4. cycle < cycle_offsetのサイクルをスキップし、中断箇所から継続
5. 復元ログにパス・サイクル番号・受理率が出力される

**前提条件**:

- training_state.ptが存在し有効な形式
- 復元先の設定が元の学習と互換性がある

**優先度**: Must Have

---

### ストーリー 9.2: 加速度適応の実行状態をMLflowで監視する 🔵

**信頼性**: 🔵 *random_walk_controller.py last_accel_action・train_tg_lora.py MLflow magnitude_acceleration/accel_action・b2eb409コミットより*

**私は** ML研究者 **として**
**加速度適応の実行状態（不安定/収束/無行動）とmagnitude加速度をMLflowダッシュボードでリアルタイムに監視したい**
**そうすることで** 学習中の加速度適応が効果的に動作しているかを定量的に評価できる

**関連要件**: REQ-165, REQ-166, EDGE-185, EDGE-186

**詳細シナリオ**:

1. 各サイクルでadapt_to_acceleration()が実行される
2. last_accel_actionが1（不安定）/-1（収束）/0（無行動）に更新される
3. MLflowサイクルメトリクスにmagnitude_accelerationとaccel_actionが記録される
4. MLflowダッシュボードで加速度適応の推移を可視化可能

**前提条件**:

- MLflowが有効化されている
- velocity.magnitude_acceleration()が計算可能

**優先度**: Must Have

---

### ストーリー 9.3: E2E resumeフローを検証する 🔵

**信頼性**: 🔵 *test_resume_e2e.py TestResumeE2E・TASK-0090完了・d3d77b9コミットより*

**私は** ML研究者 **として**
**学習中断からの再開が正しく動作することをE2Eテストで検証したい**
**そうすることで** 長時間学習中の障害発生時でも安全に学習を継続できることを保証できる

**関連要件**: REQ-167, EDGE-187, EDGE-188, EDGE-189

**詳細シナリオ**:

1. TrainingStateを2サイクル実行後に保存する
2. resume_pathで再開し、cycle 0-1がスキップされることを確認する
3. 再開後のlossが保存時から連続的に推移することを確認する
4. resume後のvelocity state方向が保存時と一致することを確認する

**前提条件**:

- --resume機能が実装済み（REQ-162~164）
- TrainingState保存・復元が動作可能

**優先度**: Must Have

---

### ストーリー 9.4: ベンチマーク結果の品質ギャップを分析する 🔵

**信頼性**: 🔵 *scripts/analyze_benchmark.py・TASK-0091 spec・537c0a9コミットより*

**私は** ML研究者 **として**
**baseline/TG-LoRAのベンチマーク評価結果の差分を自動分析したい**
**そうすることで** TG-LoRAの品質改善効果を定量的に評価し、改善領域を特定できる

**関連要件**: REQ-168

**詳細シナリオ**:

1. baseline/TG-LoRAのベンチマーク結果JSONを読み込む
2. 各メトリクス（accuracy, perplexity等）の差分を計算する
3. 欠損メトリクスは安全にスキップし、利用可能なメトリクスのみで報告する
4. フォーマットされた分析レポートを出力する

**前提条件**:

- lm-evaluation-harnessが実行済み
- ベンチマーク結果JSONが生成されている

**優先度**: Must Have

---

### ストーリー 9.5: accel paramスイープで体系的に実験する 🔵

**信頼性**: 🔵 *scripts/run_accel_sweep.sh・configs/9b_tg_lora_accel_*.yaml・TASK-0092 spec・537c0a9コミットより*

**私は** ML研究者 **として**
**加速度適応パラメータのスイープ実験を自動化したい**
**そうすることで** TruthfulQA品質ギャップを埋める最適なaccel paramsを体系的に探索できる

**関連要件**: REQ-169, REQ-170, REQ-171, REQ-172, EDGE-190, EDGE-191

**詳細シナリオ**:

1. 4つの実験config（conservative/aggressive/balanced/no_accel）を準備する
2. run_accel_sweep.shで4configを順次実行する
3. 個別実験の失敗は記録しつつ全体は継続する
4. summarize_sweep.pyで結果をvalidation loss順に集約する
5. 最適なaccel params設定を特定する

**前提条件**:

- accel paramsがYAML設定からチューニング可能（REQ-160, REQ-161）
- GPU環境が利用可能
- ベースライン学習が完了している

**優先度**: Must Have

---

## エピック10: 運用スクリプトとユーティリティの活用

### ストーリー 10.1: ユーティリティモジュールで開発効率を向上させる 🔵

**信頼性**: 🔵 *src/utils/ 既存実装・pyproject.toml依存より*

**私は** TG-LoRA開発者 **として**
**共通ユーティリティ（I/O、メモリ、ロギング、チェックポイント）を利用したい**
**そうすることで** 各スクリプト・モジュールでボイラープレートを削減し、一貫した動作を確保できる

**関連要件**: REQ-205, REQ-206, REQ-207, REQ-208, REQ-209

**詳細シナリオ**:

1. run_query.pyでRunMetrics JSONLから任意のrunのサイクル履歴を取得する
2. io.pyの高速JSON/JSONL I/Oで評価結果を保存・読込する
3. memory.pyでGPU VRAM使用量を監視し、OOM対策の判断材料にする
4. checkpoint.pyでTrainingStateを保存・復元し、障害回復を可能にする
5. logging.pyで一貫したRichHandler ロギング設定を適用する

**前提条件**:

- orjson, richがインストール済み
- RunMetrics JSONLログが存在する（run_query.py利用時）

**優先度**: Must Have

---

### ストーリー 10.2: ハイパーパラメータスイープで最適設定を探索する 🔵

**信頼性**: 🔵 *scripts/run_sweep.sh 既存実装67行・Makefile sweepターゲットより*

**私は** ML研究者 **として**
**ハイパーパラメータ（lr, rollback_tolerance, K/N）の9設定スイープを自動実行したい**
**そうすることで** 手動比較では困難な多様な設定の結果を系統的に比較できる

**関連要件**: REQ-210

**詳細シナリオ**:

1. SWEEP_GRIDで9設定のlr×tolerance×K/N組み合わせを定義する
2. run_sweep.shで各設定を順次実行し、結果をreports/に集約する
3. 各設定のrun_metrics.jsonlを比較し、最適設定を特定する

**前提条件**:

- GPU環境が利用可能
- ベースライン学習データが準備済み

**優先度**: Should Have

---

### ストーリー 10.3: アブレーションスタディでTG-LoRAの貢献を分離する 🔵

**信頼性**: 🔵 *scripts/run_ablation_suite.sh 既存実装137行・Makefile ablationターゲットより*

**私は** ML研究者 **として**
**ベースライン vs TG-LoRA変種（paper POC / adaptive K5 / no-convergence）のアブレーションを実行したい**
**そうすることで** 各機能の個別貢献を定量的に評価できる

**関連要件**: REQ-211

**詳細シナリオ**:

1. run_ablation_suite.shでbaseline→paper POC→adaptive K5→no-convergenceの順に実行する
2. 各変種の損失曲線・受理率・backward pass削減率を比較する
3. 収束適応機能の有無による差を分析する

**前提条件**:

- GPU環境が利用可能
- 4つのconfigファイルが存在する

**優先度**: Must Have

---

### ストーリー 10.4: 高LR安定性テストでロールバック優位性を実証する 🔵

**信頼性**: 🔵 *scripts/run_high_lr_comparison.sh 既存実装141行・REQ-009 rollback_manager.pyより*

**私は** ML研究者 **として**
**通常の10-25倍の学習率でTG-LoRAとベースラインを比較したい**
**そうすることで** TG-LoRAのロールバック機構が不安定学習から自動回復する能力を実証できる

**関連要件**: REQ-212, REQ-213

**詳細シナリオ**:

1. 通常LRの10倍、25倍でベースラインとTG-LoRAをそれぞれ実行する
2. ベースラインの学習発散を確認する
3. TG-LoRAがロールバック機構で発散を回避し安定学習を継持することを確認する

**前提条件**:

- GPU環境が利用可能
- 高LR用configが準備済み

**優先度**: Should Have

---

### ストーリー 10.5: accel sweep並列実行とダッシュボード可視化 🔵

**信頼性**: 🔵 *scripts/run_accel_sweep_parallel.sh 既存実装129行・scripts/generate_sweep_dashboard.py 既存実装より*

**私は** ML研究者 **として**
**2 GPU並列でaccel paramスイープを実行し、HTMLダッシュボードで結果を可視化したい**
**そうすることで** 4設定のスイープを効率的に完了し、比較結果を一望できる

**関連要件**: REQ-214, REQ-215, REQ-216, REQ-217

**詳細シナリオ**:

1. run_accel_sweep_parallel.shで2 GPUに各configを割り当て並列実行する
2. generate_sweep_dashboard.pyでranking.jsonから自己完結型HTMLダッシュボードを生成する
3. サマリテーブル、ペアワイズ比較、次アクション推奨を確認する
4. compare_paper_memory_modes.pyでreuse vs one-shotメモリモードを比較する

**前提条件**:

- 2-GPU環境が利用可能（並列実行時）
- analyze_accel_sweep.pyのranking.jsonが生成済み

**優先度**: Should Have

---

### ストーリー 11.1: 論文結果の出版形式エクスポート 🔵

**信頼性**: 🔵 *scripts/export_paper_results.py 既存実装・paper_experiment_plan.mdより*

**私は** ML研究者 **として**
**実験結果をLaTeX・Markdown・CSV形式で出版可能なテーブルとしてエクスポートしたい**
**そうすることで** 論文執筆時に統計的結果（信頼区間付き）を直接利用できる

**関連要件**: REQ-241, REQ-242

**詳細シナリオ**:

1. aggregate_summary.jsonをexport_paper_results.pyで読み込む
2. --format allでLaTeX・Markdown・CSV全形式をpaper_tables/に出力する
3. 統計サマリ（mean, std, 95% CI）を含む出版可能テーブルを得る

**優先度**: Must Have

---

### ストーリー 11.2: ハイパーパラメータ感度分析 🔵

**信頼性**: 🔵 *scripts/analyze_sensitivity.py 既存実装より*

**私は** ML研究者 **として**
**ハイパーパラメータが結果メトリクスに与える影響を定量的に分析したい**
**そうすることで** 重要なハイパーパラメータを特定し、探索範囲を絞り込める

**関連要件**: REQ-243, REQ-244

**詳細シナリオ**:

1. スイープ実験結果をanalyze_sensitivity.pyで読み込む
2. パラメータとメトリクス間のPearson相関行列を計算する
3. 感度ランキングで最も影響の大きいパラメータを特定する

**優先度**: Should Have

---

### ストーリー 11.3: 学習サイクルのヘルスモニタリング 🔵

**信頼性**: 🔵 *src/tg_lora/cycle_monitor.py 既存実装・TrainingAdvisor基盤より*

**私は** ML研究者 **として**
**学習中の発散や停滞を自動検知し、適切な介入推奨を受けたい**
**そうすることで** 学習の安定性を維持し、リソースの無駄を防げる

**関連要件**: REQ-245, REQ-246, REQ-247, REQ-248

**詳細シナリオ**:

1. CycleMonitor.update()に各サイクルのlossデータを渡す
2. NaN/Inf値やloss spikeを発散としてcritical/high severityで検知する
3. patienceサイクル以上の改善がない停滞を検知する
4. 状態に応じた介入推奨（reduce_lr, rollback, increase_K）を受け取る

**優先度**: Must Have

---

### ストーリー 11.4: 実験構成の横断比較 🔵

**信頼性**: 🔵 *scripts/compare_experiment_configs.py 既存実装より*

**私は** ML研究者 **として**
**複数の実験構成を横断比較し、最適な構成を特定したい**
**そうすることで** スイープ結果からベスト構成を効率的に選択できる

**関連要件**: REQ-249, REQ-250

**詳細シナリオ**:

1. runsディレクトリ配下の実験をcompare_experiment_configs.pyで自動検出する
2. 構成パラメータと結果メトリクスの比較マトリクスを構築する
3. best_valid_loss等のメトリクスでランク付けしMarkdown/JSON出力する

**優先度**: Should Have

---

### ストーリー 11.5: 統合学習アドバイザ 🔵

**信頼性**: 🔵 *src/tg_lora/training_advisor.py 既存実装・CycleMonitor + TrajectoryAnalyzer統合より*

**私は** ML研究者 **として**
**学習の軌跡・健全性を統合的に分析し、次に取るべきアクションを推奨されたい**
**そうすることで** 学習の品質を自律的に管理し、手動監視の負担を減らせる

**関連要件**: REQ-251, REQ-252, REQ-253, REQ-254, REQ-255, REQ-256

**詳細シナリオ**:

1. TrainingAdvisor.evaluate()にサイクルメトリクスを渡す
2. CycleMonitorとTrajectoryAnalyzerの統合分析結果を受け取る
3. 優先順位付きアクション（critical > high > medium > low）を確認する
4. NaN/Inf・loss spike・停滞・異常・降下トレンドに応じた適切なアクションを実行する

**優先度**: Must Have

---

### ストーリー 11.6: CLI学習アドバイスレポート 🔵

**信頼性**: 🔵 *scripts/advise_training.py 既存実装・training_advisor.py統合より*

**私は** ML研究者 **として**
**run_metrics.jsonlから学習アドバイスレポートをCLIで生成したい**
**そうすることで** 学習完了後に hindsight で学習品質を評価できる

**関連要件**: REQ-257, REQ-258

**詳細シナリオ**:

1. `python scripts/advise_training.py runs/my_run/run_metrics.jsonl` を実行する
2. cycle_step/stepレコードを抽出し正規化する
3. 人間可読テキストまたはJSON形式でAdvisoryReportを出力する
4. exit code 0/1/2でヘルス状態を通知する

**優先度**: Must Have

---

## ストーリーマップ（更新版）

```
エピック1: TG-LoRA学習の実行
├── ストーリー 1.1 (🔵 Must Have)
├── ストーリー 1.2 (🔵 Must Have)
└── ストーリー 1.3 (🔵 Must Have)

エピック2: 学習の安全性・安定性
├── ストーリー 2.1 (🔵 Must Have)
└── ストーリー 2.2 (🔵 Must Have)

エピック3: データ・評価パイプライン
├── ストーリー 3.1 (🔵 Must Have)
└── ストーリー 3.2 (🔵 Must Have)

エピック4: 設定・運用品質
├── ストーリー 4.1 (🔵 Must Have)
└── ストーリー 4.2 (🔵 Must Have)

エピック5: Paper Gate評価・論文パイプライン
├── ストーリー 5.1 (🔵 Must Have)
├── ストーリー 5.2 (🔵 Must Have)
└── ストーリー 5.3 (🔵 Must Have)

エピック6-9: (既存)
├── ...

エピック10: 運用スクリプトとユーティリティ
├── ストーリー 10.1 (🔵 Must Have) - ユーティリティモジュール
├── ストーリー 10.2 (🔵 Should Have) - HP スイープ
├── ストーリー 10.3 (🔵 Must Have) - アブレーション
├── ストーリー 10.4 (🔵 Should Have) - 高LR安定性
└── ストーリー 10.5 (🔵 Should Have) - 並列sweep/ダッシュボード

エピック11: 学習分析・アドバイザパイプライン
├── ストーリー 11.1 (🔵 Must Have) - 論文結果エクスポート
├── ストーリー 11.2 (🔵 Should Have) - 感度分析
├── ストーリー 11.3 (🔵 Must Have) - ヘルスモニタリング
├── ストーリー 11.4 (🔵 Should Have) - 実験横断比較
├── ストーリー 11.5 (🔵 Must Have) - 統合アドバイザ
└── ストーリー 11.6 (🔵 Must Have) - CLIアドバイスレポート
```

## 信頼性レベルサマリー（更新版）

- 🔵 青信号: 43件 (100%)
- 🟡 黄信号: 0件 (0%)
- 🔴 赤信号: 0件 (0%)

**品質評価**: 高品質

---

## エピック12: Prior-based Subspace Amplification（PSA）

### ストーリー 12.1: PSA勾配増幅で学習効率を向上させる 🔵

**信頼性**: 🔵 *src/tg_lora/psa.py PSAPrior実装・docs/GOAL.md §3 Track01より*

**私は** ML研究者 **として**
**安定したper-tensor PC1方向に沿って勾配を増幅し、学習効率を向上させたい**
**そうすることで** 外挿の不確実性なく、安定したsubspace方向で効率的な学習ができる

**関連要件**: REQ-265, REQ-266, REQ-267, REQ-268, REQ-269

**詳細シナリオ**:

1. PSAPriorを初期化（history_length=6, gain=0.5, update_interval=3）
2. 各サイクルでrecord_delta()でdeltaをリングバッファに記録
3. update_interval間隔でextract_priors()がpower iterationでPC1方向を抽出
4. amplify_gradients()が勾配をin-placeに増幅（G + gamma * <G, v_PSA> * v_PSA）
5. compute_gain_map()がlayer-type-specific gain（out_proj ×1.2等）を適用
6. warmup_steps未満では増幅をスキップ

**前提条件**:

- LoRAパラメータがモデルに適用済み
- DeltaTrackerが各サイクルのdeltaを提供

**優先度**: Must Have

---

### ストーリー 12.2: レジーム検知で学習フェーズ遷移を自動検出する 🔵

**信頼性**: 🔵 *src/tg_lora/regime.py RegimeDetector実装・docs/GOAL.md §4 Track03より*

**私は** ML研究者 **として**
**学習中のフェーズ遷移（STABLE→PLATEAU→TRANSITION）を自動検知し、PSA priorのリセットタイミングを知りたい**
**そうすることで** 学習フェーズが変わる時にpriorが古い方向を保持せず、新しいsubspaceに適応できる

**関連要件**: REQ-272, REQ-273

**詳細シナリオ**:

1. RegimeDetectorを初期化（window=8, plateau_eps=1e-4, transition_z=2.0）
2. 各サイクルでupdate(loss)を呼び出し、velocity z-scoreを計算
3. STABLE（負のvelocity）、PLATEAU（ほぼゼロ）、TRANSITION（外れ値）を分類
4. TRANSITION検出時にconsume_reset_signal()がPSA prior resetをトリガー
5. ワンショット消費パターンで重複リセットを防止

**前提条件**:

- loss履歴が3件以上蓄積されている

**優先度**: Must Have

---

### ストーリー 12.3: 活性化フィンガープリントでレジーム別効率上限を評価する 🔵

**信頼性**: 🔵 *src/tg_lora/activation_regime.py ActivationFingerprintTracker実装・docs/GOAL.md §4 Track02より*

**私は** ML研究者 **として**
**活性化のcosine similarityに基づくレジーム分類で、各レジームの時間割合を測定したい**
**そうすることで** 安定/遷移/カオス各レジームの割合からPSA効率改善の理論的上限を推定できる

**関連要件**: REQ-274, REQ-275

**詳細シナリオ**:

1. ActivationFingerprintTrackerを初期化（window=10, stable_threshold=0.95）
2. register_hook()でモデルにforward hookを登録
3. step()で連続ステップ間cosine similarityを計算（forward-only、追加backwardなし）
4. STABLE（cos>0.95）/TRANSITION/CHAOTIC（cos<0.5）のレジームを分類
5. regime_inventoryで各レジーム割合を取得
6. compute_regime_null_baseline()で時系列シャッフルによるヌルベースラインを計算

**前提条件**:

- モデルにforward hookが登録可能
- 学習ループでstep()が呼び出される

**優先度**: Should Have

---

### ストーリー 12.4: LAWAベースラインでPSA優位性を検証する 🔵

**信頼性**: 🔵 *src/tg_lora/weight_averaging.py LAWAAverager実装・docs/GOAL.md §3.3より*

**私は** ML研究者 **として**
**LAWA（Latest-Window Weight Averaging）ベースラインを構築し、PSAと比較したい**
**そうすることで** PSAの優位性を適切なベースライン（plain LoRA + 重み平均）に対して実証できる

**関連要件**: REQ-276, REQ-277

**詳細シナリオ**:

1. LAWAAveragerを初期化（window_size=5, start_cycle=0）
2. 各サイクルでrecord(model, cycle)でLoRA重みスナップショットを記録
3. average_snapshot()でスライディングウィンドウの算術平均を計算
4. evaluate_with_lawa()で一時的に平均重みに差し替えて評価
5. PSA vs LAWA vs plain LoRAの3者比較でPSA優位性を確認

**前提条件**:

- 学習ループでLAWAAveragerが使用されている
- 評価用データローダーが利用可能

**優先度**: Must Have

---

### ストーリー 12.5: レイヤー別ΔW分析でPSA理論基盤を検証する 🔵

**信頼性**: 🔵 *src/tg_lora/layer_delta_analysis.py実装・docs/GOAL.md §4 Track08より*

**私は** ML研究者 **として**
**per-tensor ΔWのrank-1 dominanceと方向安定性をレイヤータイプ別に分析したい**
**そうすることで** 「out_proj最安定仮説」を検証し、PSA gain mapの理論的正当性を確立できる

**関連要件**: REQ-278, REQ-279

**詳細シナリオ**:

1. DeltaTrackerの履歴からper-tensor delta行列を取得
2. compute_rank1_dominance()でPC1分散比率を計算
3. compute_direction_stability()で前半/後半PC1 cosineを計算
4. marchenko_pastur_expected_rank1()でランダムヌル期待値を計算
5. classify_layer_type()でATTENTION_OUT/ATTENTION_V/MLP等に分類
6. group_by_layer_type()でレイヤータイプ別に集約
7. z-score vs Marchenko-Pasturヌルで統計的有意性を評価

**前提条件**:

- 十分なサイクル数のdelta履歴が蓄積されている
- DeltaTrackerが有効化されている

**優先度**: Should Have

---

### ストーリー 12.6: PSAアブレーションでγ感度とregime reset効果を評価する 🔵

**信頼性**: 🔵 *scripts/run_psa_ablation.sh/run_psa_gamma_sweep.sh/summarize_psa_sweep.py既存実装より*

**私は** ML研究者 **として**
**PSA gain（γ）の複数値をスイープし、regime reset ON/OFFのアブレーションを自動実行したい**
**そうすることで** 最適なγとregime reset効果を定量的に評価し、最良のPSA設定を特定できる

**関連要件**: REQ-280, REQ-281, REQ-282

**詳細シナリオ**:

1. run_psa_ablation.shでPSA vs plain LoRA vs LAWAの3条件を実行
2. run_psa_gamma_sweep.shで複数γ値（0.1, 0.3, 0.5, 0.7, 1.0）のスイープ
3. regime reset ON/OFFのアブレーションを含める
4. summarize_psa_sweep.pyで結果を集約し最適設定を特定

**前提条件**:

- GPU環境が利用可能
- 3つのconfig（PSA/plain LoRA/LAWA）が準備済み

**優先度**: Must Have

---

## ストーリーマップ（Phase 62追加版）

```
エピック12: Prior-based Subspace Amplification
├── ストーリー 12.1 (🔵 Must Have)   - PSA勾配増幅コア
├── ストーリー 12.2 (🔵 Must Have)   - レジーム検知
├── ストーリー 12.3 (🔵 Should Have) - 活性化フィンガープリント
├── ストーリー 12.4 (🔵 Must Have)   - LAWAベースライン
├── ストーリー 12.5 (🔵 Should Have) - レイヤー別ΔW分析
└── ストーリー 12.6 (🔵 Must Have)   - PSAアブレーション
```

## 信頼性レベルサマリー（Phase 62追加版）

- 🔵 青信号: 49件 (100%)
- 🟡 黄信号: 0件 (0%)
- 🔴 赤信号: 0件 (0%)

**品質評価**: 高品質
