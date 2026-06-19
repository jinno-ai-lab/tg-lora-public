# TG-LoRA 要件定義書


<!-- spine:anchor:begin -->
> **Spine anchor**: [TG-LoRA アーキテクチャ設計](architecture.md)
>
> - parent: `tg-lora/architecture.md`
> - role: `detailed`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**最終更新**: 2026-06-10（Phase 62追加: PSA Prior-based Subspace Amplification・RegimeDetector・ActivationFingerprint・LAWA・LayerDeltaAnalysis要件）

## 概要

TG-LoRA (Tangent-Gradient LoRA) は、LoRA学習における勾配速度ベクトルの外挿を用いて学習効率を向上させる手法。学習中のLoRA重み変化の速度（velocity）を追跡し、外挿（extrapolation）で次ステップの重みを予測することで、backward pass回数を削減しつつ同等以上の性能を目指す。

## 関連文書

- **分析記録**: [interview-record.md](interview-record.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)

## 機能要件（EARS記法）

**【信頼性レベル凡例】**:

- 🔵 **青信号**: PRD・既存要件定義書・設計文書・既存実装を参考にした確実な要件
- 🟡 **黄信号**: PRD・既存要件定義書・設計文書・既存実装から妥当な推測による要件
- 🔴 **赤信号**: 参照資料にない自動推定による要件

### 通常要件

#### コアアルゴリズム

- REQ-001: システムはLoRA重み更新の速度ベクトルを指数移動平均（EMA）で追跡しなければならない。cosine_similarity計算時、deltaのキーとstateのキーが完全一致しない場合でもKeyErrorを発生させず、共通キーのみで計算しなければならない。update時、deltaにstateに存在しない新規キーが含まれる場合もKeyErrorを発生させず安全に処理しなければならない。各更新後にvelocity stateのL2 norm（magnitude）を計算し、リングバッファ（max_history）に記録しなければならない。reset()はstateとmagnitude_historyの両方をクリアしなければならない 🔵 *velocity.py 既存実装・KeyError修正(6717ee8, TASK-0010 Bug #4)・magnitude history実装(bbcb7e7)より*
- REQ-002: システムは速度ベクトルに基づいてLoRA重みを外挿予測し、モデルに適用しなければならない。active_namesにvelocityに存在しないキーが含まれる場合、KeyErrorを発生させず安全にスキップしなければならない 🔵 *extrapolator.py 既存実装・TASK-0010 Bug #1 より*
- REQ-003: システムは外挿更新の大きさをパラメータノルムに対する比率で制限（cap）しなければならない 🔵 *extrapolator.py `cap_update` 実装より*
- REQ-004: システムはK歩のpilot学習前後のLoRA重み差分を計算し、その平均デルタを求めなければならない 🔵 *delta_tracker.py 既存実装より*
- REQ-005: システムはLoRAパラメータのスナップショット作成・復元・差分計算を提供しなければならない。diff_loraのafter辞書にbeforeのキーが欠落している場合、およびload_lora_snapshotのstateにモデルパラメータが欠落している場合でもKeyErrorを発生させず安全に処理しなければならない 🔵 *lora_state.py 既存実装・TASK-0010 Bug #2, #3 より*

#### レイヤーサンプリング

- REQ-006: システムは複数のレイヤー選択戦略（last_25_percent, last_25_percent_plus_random_2, middle_random, lisa_like_weighted）をサポートしなければならない 🔵 *layer_sampler.py 既存実装より*
- REQ-007: システムはレイヤーの重要度スコアに基づく重み付きサンプリングをサポートしなければならない 🔵 *layer_sampler.py `_lisa_weighted` 実装より*
- REQ-008: システムはモデル内のLoRA対象レイヤー数を自動検出しなければならない 🔵 *layer_sampler.py `get_num_layers` 実装より*

#### ロールバック機構

- REQ-009: システムは外挿適用前のLoRA状態を保存し、損失悪化時に元の状態へロールバックしなければならない 🔵 *rollback_manager.py 既存実装より*
- REQ-010: システムは複数段階の状態履歴を管理し、任意の時点へ復元可能でなければならない 🔵 *rollback_manager.py 履歴管理実装より*

#### ランダムウォークコントローラ

- REQ-011: システムはK（pilot歩数）、N（外挿歩数）、alpha（ステップサイズ）、beta（EMA係数）、lr（学習率）のハイパーパラメータをランダムウォークで探索しなければならない 🔵 *random_walk_controller.py 既存実装・9b_tg_lora.yaml lr_initial/lr_min/lr_max/lr_accept_boost/lr_reject_decay設定より*
- REQ-012: システムはpilot損失と外挿後損失を比較し、許容範囲内（rollback_tolerance）であれば外挿を受理しなければならない 🔵 *random_walk_controller.py `accept` 実装より*
- REQ-013: システムは受理時にalphaを増加させ、拒否時にalphaを減少させる適応制御を行わなければならない 🔵 *random_walk_controller.py `reward`/`penalize` 実装より*
- REQ-013a: システムは受理時にlrを増加（lr_accept_boost倍）させ、拒否時にlrを減少（lr_reject_decay倍）させる適応学習率制御を行わなければならない。lrは常に[lr_min, lr_max]の範囲内にクランプされなければならない 🔵 *random_walk_controller.py `reward`/`penalize` lr適応・9b_tg_lora.yaml lr_reject_decay=0.5より*

  **lr_reject_decay=0.5の設計根拠（Phase 9実験検証）**:
  - **旧値からの変更**: lr_reject_decayを0.7から0.5に変更。拒否時にlrをより速く保守的な値に戻すため。
  - **設計意図**: 外挿が悪化した（拒否された）サイクルでは、学習が不安定方向に進んでいる可能性が高い。0.5倍減衰により、1回の拒否でlrを半減させ、急激に保守的な学習率へ移行する。これにより不安定な外挿が連鎖するリスクを低減する。
  - **実験による検証**: TASK-0028の10サイクルGPU学習で、lr_reject_decay=0.5の動作を検証。連続拒否サイクルでlrがlr_min（1e-5）まで適切に減少し、受理後の回復もlr_accept_boost=1.2で安定して動作することを確認。
  - **回復速度の比較**: lr_reject_decay=0.7では、3回連続拒否後のlrは初期値の34.3%（0.7^3）に留まる。lr_reject_decay=0.5では12.5%（0.5^3）まで急速に低下し、不安定領域からの脱出速度が向上。受理後の回復はlr_accept_boost=1.2で段階的に行われ、安定性と効率のバランスを確保。
- REQ-014: システムはハイパーパラメータ提案の受理率を追跡し、サマリーとして出力しなければならない 🔵 *random_walk_controller.py `acceptance_rate`/`summary` 実装より*
- REQ-053: システムは探索が停滞した場合（DeltaTrackerのconvergence_trendが0以上）に、proactivelyにlrを減少（0.8倍）させKを増加させる収束適応（adapt_to_convergence）を行わなければならない。total_cyclesが2以下の場合は適応を実行してはならない 🔵 *random_walk_controller.py `adapt_to_convergence` 実装・test_random_walk_controller.py より*
- REQ-054: システムは連続する受理/拒否サイクルにおいて、lrが常に[lr_min, lr_max]の範囲内に留まることを保証しなければならない。交互の受理/拒否サイクルでもlrが境界を逸脱してはならない 🔵 *random_walk_controller.py lrクランプロジック・test_random_walk_controller.py `test_lr_clamps_at_*`/`test_lr_alternating_*` より*
- REQ-055: システムはアクティブレイヤーの受理/拒否フィードバックに基づいてレイヤー重要度スコア（layer_scores）を更新しなければならない 🔵 *random_walk_controller.py `update_layer_scores` 実装より*

#### サイクル状態追跡（CycleState）

- REQ-038: システムはサイクルレベルの集計状態（サイクル数、累積backward pass数、累積外挿ステップ数、最良損失、staleサイクル数、受理/拒否カウント）を追跡しなければならない 🔵 *cycle_state.py `CycleState` dataclass・train_tg_lora.py より*
- REQ-039: システムは削減率（1 − backward_passes / (backward_passes + extrapolation_steps)）と受理率（accepted / total）を計算・提供しなければならない 🔵 *cycle_state.py `reduction_rate`/`acceptance_rate` プロパティより*
- REQ-040: システムはpatience-based早期終了を判定し、指定サイクル数以上損失改善がない場合に学習停止を通知しなければならない。patience=Noneの場合は早期終了を無効にしなければならない 🔵 *cycle_state.py `should_stop` メソッドより*
- REQ-040a: システムはフル評価（full eval）とクイック評価（quick eval）のstale_cycles追跡を分離しなければならない。フル評価サイクルではrecord_cycleのvalid_loss追跡をスキップし、record_full_evalでbest_loss/stale_cyclesを更新しなければならない 🔵 *cycle_state.py `record_full_eval` メソッド・train_tg_lora.py stale_cycles二重計上修正より*

#### 重み差分追跡（DeltaTracker）

- REQ-041: システムは各サイクルの重み差分（delta）の統計（total norm, per-layer norm, max component, mean absolute value）を計算・記録しなければならない 🔵 *delta_tracker.py `DeltaStats` dataclass・`_compute_stats` 関数より*
- REQ-042: システムは最新delta normが履歴のmean+σ*thresholdを超過した場合に異常（anomalous）と判定しなければならない。履歴が3件未満の場合は異常判定を行わない。標準偏差が極小（< 1e-12）の場合はmean×2.0を閾値としなければならない 🔵 *delta_tracker.py `is_anomalous` メソッドより*
- REQ-043: システムは直近window件のdelta norm履歴に対する線形回帰の傾き（convergence trend）を計算し、負の値を収束傾向、正の値を発散傾向として報告しなければならない 🔵 *delta_tracker.py `convergence_trend` メソッドより*

#### Velocity異常検出・トレンド追跡

- REQ-049: システムはvelocityのmagnitude履歴に基づく異常検出（is_magnitude_anomalous）を提供しなければならない。履歴が3件未満の場合はFalseを返さなければならない。標準偏差が極小（< 1e-12）の場合はmean×2.0を閾値としなければならない。正常時はmean+threshold_sigma*stdを閾値としなければならない 🔵 *velocity.py `is_magnitude_anomalous` 実装・test_velocity.py より*
- REQ-050: システムはvelocityのmagnitude履歴の直近window件に対する線形回帰の傾き（magnitude_trend）を計算しなければならない。データが2件未満の場合は0.0を返さなければならない。負の値を収束傾向、正の値を発散傾向として報告しなければならない 🔵 *velocity.py `magnitude_trend` 実装・test_velocity.py より*

#### データスキーマ検証

- REQ-051: システムはPydanticベースのデータスキーマ（DataRecord）により、ChatML形式学習データのtext必須・非空・ChatMLマーカー含有を検証し、token_countの正値制約を適用しなければならない 🔵 *schema.py `DataRecord` 実装・test_schema.py より*
- REQ-052: システムはValidationSummaryによりバッチ検証結果（total/valid/skipped/errors）を集計し、validate_records関数でレコード一括検証を行わなければならない 🔵 *schema.py `ValidationSummary`/`validate_records` 実装・test_schema.py より*

#### 学習ループ

- REQ-015: システムは標準QLoRA学習のベースラインループを提供しなければならない 🔵 *train_baseline_qlora.py 既存実装より*
- REQ-016: システムはサイクルベースのTG-LoRA学習ループを提供し、各サイクルでpilot → 外挿 → 受理/拒否のサイクルを実行しなければならない 🔵 *train_tg_lora.py 既存実装より*
- REQ-017: システムは勾配蓄積（gradient accumulation）をサポートしなければならない 🔵 *trainer_loop.py `forward_backward` 実装より*
- REQ-018: システムは学習率ウォームアップ付きの線形スケジューラをサポートしなければならない 🔵 *trainer_loop.py `create_scheduler` 実装より*
- REQ-019: システムは勾配クリッピング（max_grad_norm）を適用しなければならない 🔵 *trainer_loop.py `optimizer_step` 実装より*
- REQ-020: システムは早期終了（early stopping）をサポートし、損失改善がない場合に学習を停止しなければならない 🔵 *train_tg_lora.py try/finally rollback 実装より*
- REQ-044: 学習ループは各サイクルでCycleStateにサイクル情報（K, N, grad_accum, train_loss, valid_loss, accepted）を記録し、DeltaTrackerに重み差分統計を記録し、CycleState.should_stopで早期終了を判定し、学習終了時に両サマリーを結合して出力しなければならない。フル評価サイクルではクイック評価ではなくフル評価の損失でstale追跡を行わなければならない。サマリー結合はbuild_training_summary関数で行わなければならない 🔵 *train_tg_lora.py 統合実装・test_training_integration.py より*

#### モデル管理

- REQ-021: システムは4bit量子化（NF4）でベースモデルを読み込み、LoRAアダプタを適用しなければならない 🔵 *load_model.py 既存実装より*
- REQ-022: システムはVRAM残量に基づいてEmbedding/LMヘッドのfp32キャストを自動判定しなければならない 🔵 *load_model.py fp32 cast実装より*
- REQ-023: システムは勾配チェックポイントを有効にし、VRAM使用量を最適化しなければならない 🔵 *load_model.py 実装より*
- REQ-024: システムはLoRAパラメータの反復・レイヤー別グループ化・カウント機能を提供しなければならない 🔵 *lora_utils.py 既存実装より*

#### データパイプライン

- REQ-025: システムはDolly 15kおよびCapybaraデータセットをダウンロード可能でなければならない 🔵 *download_data.py 既存実装より*
- REQ-026: システムは生データをChatML形式のJSONLに変換し、train/valid/gold_testに分割しなければならない 🔵 *prepare_data.py 既存実装より*
- REQ-027: システムはJSONL形式のデータセットを読み込み、トークナイズしてPyTorch Datasetとして提供しなければならない 🔵 *build_seed_dataset.py 既存実装より*
- REQ-028: システムはオープンソースモデルによる合成データ生成をサポートしなければならない 🔵 *generate_open_data.py 既存実装より*
- REQ-029: システムはテキスト長・品質スコア・必須フィールドによるデータフィルタリングを提供しなければならない 🔵 *filter_dataset.py 既存実装より*
- REQ-030: システムは完全一致および埋め込みベースの意味的重複排除をサポートしなければならない 🔵 *dedup.py 既存実装より*
- REQ-031: システムはデータ来歴（provenance）メタデータを記録しなければならない 🔵 *provenance.py 既存実装より*

#### 評価

- REQ-032: システムは指定データローダー上での平均損失評価を提供しなければならない 🔵 *eval_loss.py 既存実装より*
- REQ-033: システムはタスク評価（生成 → メトリクス計算）をサポートしなければならない 🔵 *eval_task.py 既存実装より*
- REQ-034: システムはJSONフォーマット準拠性の自動評価を提供しなければならない 🔵 *eval_format.py 既存実装より*
- REQ-035: システムはlm-evaluation-harnessによる標準ベンチマーク評価（ARC, HellaSwag, GSM8K, TruthfulQA）をサポートしなければならない 🔵 *run_eval.sh / run_eval_lora.sh 実装より*

#### 比較システム

- REQ-036: システムはベースラインとTG-LoRAの同一計算予算での公正な比較実験を実行しなければならない 🔵 *run_comparison.sh / compare_runs.py 実装より*
- REQ-037: システムは損失曲線、効率メトリクス、受理率を含む比較レポートを生成しなければならない 🔵 *compare_runs.py レポート生成実装より*
- REQ-037a: compare_runs.pyのgather_runs()はJSONLパース失敗時にstderrへの警告出力に加え、構造化parse_warningsリストを各run辞書に収集しなければならない。format_json()はrunsキーとparse_warningsキー（警告ありの場合のみ）を含むオブジェクトを出力し、render_dashboard()はRich Panelで警告を表示しなければならない。compare_experiment_configs.pyのparse_warningsパターンと一貫しなければならない 🔵 *compare_runs.py parse_warnings収集・compare_experiment_configs.py ExperimentSummary.parse_warningsとのパリティより*

### 条件付き要件

- REQ-101: LoRAアダプタを評価する場合、システムはアダプタをベースモデルにマージしてから評価し、評価後にクリーンアップしなければならない 🔵 *run_eval_lora.sh 3-phase実装より*
- REQ-102: FAISSが利用可能な場合、システムはFAISSを使用した高効率な意味的類似度検索を行い、利用不可の場合はnumpy行列演算にフォールバックしなければならない 🔵 *dedup.py フォールバック実装より*
- REQ-103: VRAM使用量が50%未満の場合、システムはEmbedding/LMヘッドのfp32キャストを実行しなければならない 🔵 *load_model.py VRAM判定実装より*
- REQ-104: データに"context"フィールドが含まれる場合、システムはChatMLテンプレートにコンテキスト情報を含めなければならない 🔵 *prepare_data.py context処理実装より*

### 状態要件

- REQ-201: 学習が不安定（外挿後損失がpilot損失×許容率を超過）な状態にある場合、システムはロールバックを実行し、ハイパーパラメータにペナルティを適用しなければならない 🔵 *train_tg_lora.py accept/rollback分岐より*
- REQ-202: early stopping条件（損失が指定ステップ数改善なし）に達した状態にある場合、システムは学習を終了し、最良モデルを保存しなければならない 🔵 *train_tg_lora.py 早期終了実装より*
- REQ-203: velocity状態が初期化されていない（初回更新）状態にある場合、システムは入力デルタをそのままvelocityとして設定しなければならない 🔵 *velocity.py 初回更新実装より*

### オプション要件

- REQ-301: システムはMLflowバックエンドによる実験ログ記録をサポートしてもよい 🔵 *src/utils/mlflow_logger.py MLflowLogger クラス（log_params, log_metrics, set_tag実装）・lazy import mlflow パターン・TASK-0057より*
- REQ-302: システムはMLflowによるチェックポイントの保存・管理をしてもよい 🔵 *src/utils/mlflow_logger.py log_artifact() 実装・train_tg_lora.py save_checkpoint後の呼び出し・TASK-0056より*
- REQ-303: システムはカスタムメトリクス関数によるタスク評価をしてもよい 🔵 *eval_task.py metric_fn引数より*
- REQ-304: システムは追加のベンチマークタスク（MMLU等）を設定で指定してもよい 🔵 *run_eval.sh --tasks引数より*

### 制約要件

- REQ-401: システムはRTX3060 12GB VRAMで9Bモデルの4bit QLoRA学習が可能でなければならない 🔵 *AGENTS.md VRAM仕様より*
- REQ-402: システムはQwen3.5-9Bのハイブリッドアーキテクチャ（24 DeltaNet + 8 Attention層）に対応しなければならない 🔵 *AGENTS.md モデル仕様より*
- REQ-403: システムはLoRA対象モジュールとして`all-linear`を使用し、DeltaNet層を漏れなく対象に含めなければならない 🔵 *AGENTS.md target_modules指定より*
- REQ-404: システムは初期検証フェーズでは公開データセットのみを使用し、自社データは使用しないことを前提とする 🔵 *AGENTS.md データ戦略より*
- REQ-405: data/, runs/, reports/ ディレクトリはgit管理外とし、実験設定はconfigs/にYAMLで管理しなければならない 🔵 *AGENTS.md・.gitignoreより*
- REQ-406: モデルチェックポイントは`runs/<experiment_name>/`に保存しなければならない 🔵 *AGENTS.md・configsより*

## 非機能要件

### パフォーマンス

- NFR-001: TG-LoRAの1サイクルはK歩のbackward pass + N歩の外挿で構成され、同等のbackward pass予算でベースラインと比較可能でなければならない 🔵 *run_comparison.sh 公正比較ロジックより*
- NFR-002: レイヤーサンプリングにより、外挿対象パラメータ数を削減し、計算コストを低減しなければならない 🔵 *layer_sampler.py 複数戦略実装（last_25_percent, weighted sampling）・test_layer_sampler.py カバレッジ完備*
- NFR-003: 4bit量子化（NF4）+ LoRA (r=16) により、9Bモデルの学習をシングルGPU 12GBで実行可能にしなければならない 🔵 *AGENTS.md VRAM仕様より*

### 信頼性

- NFR-101: ロールバック機構により、外挿による学習不安定を自動回復しなければならない 🔵 *rollback_manager.py 実装より*
- NFR-102: try/finallyによるロールバック安全性を保証し、例外発生時にも学習状態を保全しなければならない 🔵 *train_tg_lora.py try/finally実装より*
- NFR-103: 学習中のeval_loss、eval_format、eval_taskの状態リーク（dropout等）を防止しなければならない。評価関数内で例外が発生した場合でも、model.train()を確実に呼び出し元のtraining modeを復元しなければならない 🔵 *eval_loss.py context manager実装・TASK-0010 Bug #5, #6 より*

### 再現性

- NFR-201: 乱数シードを設定することで、全ライブラリ（random, numpy, torch, CUDA）にわたり再現性を確保しなければならない 🔵 *seed.py 実装より*
- NFR-202: 実験設定はYAMLファイルで完全に記述可能でなければならない 🔵 *configs/ 構成より*

### 運用性

- NFR-301: Makefileにより主要操作を単一コマンドで実行可能にしなければならない 🔵 *Makefile 既存実装より*
- NFR-302: 学習メトリクスはJSONL形式で構造化ログとして記録しなければならない 🔵 *run_metrics.py 実装より*
- NFR-304: RunMetricsはコンテキストマネージャ（with文）をサポートし、ブロック終了時にファイルハンドルを確実にクローズしなければならない 🔵 *run_metrics.py `__enter__`/`__exit__` 実装(6717ee8)より*
- NFR-303: 比較レポートには損失曲線のプロット（PNG）を含めなければならない 🔵 *compare_runs.py `plot_loss_curves` 実装より*

### テストカバレッジ

- NFR-401: 全ソースモジュール（src/tg_lora, src/training, src/model, src/data, src/eval, src/utils）に対応するユニットテストが存在しなければならない 🔵 *92テストファイル、1857テストケース全パス（Phase 42: コンストラクタ検証テスト + flaky test fix含む）*
- NFR-402: 統合テスト（スモークテスト）でTG-LoRA学習のE2E動作を検証しなければならない 🔵 *test_smoke.py 4テストケースより*

#### 外挿安全性

- REQ-056: システムは外挿適用後にLoRAパラメータが有限値（NaN/Infでない）であることを検証しなければならない。非有限パラメータが検出された場合、外挿を棄却として扱いロールバックしなければならない 🔵 *train_tg_lora.py外挿パスに安全性ギャップ（AI_HUB_MAKE_RUN_FEEDBACK指摘）・trainer_loop.py NumericalInstabilityErrorパターンより*
- REQ-057: TG-LoRA学習ループはベースライン学習ループと同等の数値安全性カバレッジ（NaN/Inf検出、学習不可能パラメータ検出、勾配クリッピング、バッチキー検証）を備えなければならない。forward_backwardのNumericalInstabilityErrorは学習ループの最上位でキャッチされ、安全な終了処理が行われなければならない 🔵 *train_baseline_qlora.py/train_tg_lora.py比較・trainer_loop.py NumericalInstabilityError定義より*

#### 外挿安全性統合テスト

- REQ-059: システムは外挿後の非有限パラメータ検出→ロールバック→ペナルティ→サイクル状態記録の完全な回復フローを統合テストで検証しなければならない。モックモデルが非有限パラメータを生成するシナリオで、rollback_manager.rollback()、controller.penalize()、controller.update_layer_scores()、cycle_state.record_cycle()の全てが正しく呼び出され、モデルパラメータがロールバック後に復元されることを確認しなければならない 🔵 *train_tg_lora.py 非有限回復パス(332-355行)・AI_HUB_MAKE_RUN_FEEDBACK指摘「integration test exercising the full extrapolation→NaN detection→rollback→cycle_state.record_cycle() path」より*
- REQ-060: 統合テストは非有限パラメータ検出後の副作用（controller.penalizeの呼出回数・引数、controller.update_layer_scoresのactive_indicesとスコア-1.0、cycle_state.record_cycleのK/N/grad_accum/accepted=False）を個別に検証しなければならない 🔵 *train_tg_lora.py 非有限回復パス(338-354行)・AI_HUB_MAKE_RUN_FEEDBACK指摘「verify the penalize/score-update side effects」より*

#### 設定検証

- REQ-045: システムは学習設定をPydanticスキーマで検証し、無効な設定値（型違い、欠落フィールド、値域逸脱）を学習開始前に拒否しなければならない 🔵 *configs/*.yaml構造・Pydantic>=2.5依存より*
- REQ-046: システムは学習開始前にデータファイルの存在・アクセス可能性を確認し、欠落している場合はエラーを報告して学習を中止しなければならない 🔵 *既存train_*.pyの前提条件より*
- REQ-047: システムは学習設定の値域制約（K > 0, N > 0, alpha_min < alpha_max, max_seq_len >= 32等）を検証し、不正な値を学習開始前に検出しなければならない 🔵 *9b_tg_lora.yaml設定値の解析より*
- REQ-048: CLIエントリポイント（main()）はargparseで設定パスを受け取り、スキーマ検証→preflight検証の順序で初期化しなければならない 🔵 *既存main()関数の構造より*
- REQ-058: システムは設定スキーマの列挙可能文字列フィールド（dtype, bnb_4bit_compute_dtype）をLiteral型で検証し、無効な文字列を学習開始前に拒否しなければならない 🔵 *config_schema.py ActiveLayerStrategy/BnbQuantTypeのLiteral enumパターン・dtypeフィールド未検証ギャップより*
- REQ-061: 全てのPydantic設定モデル（ExperimentConfig, ModelConfig, LoRAConfig, DataConfig, TrainingConfig, EvalConfig, MLflowConfig, LoggingConfig, TGLoRAParams, BaselineConfig, TGLoRAConfig）はextra='forbid'を設定し、YAML内の未知フィールド（タイポ等）を学習開始前に拒否しなければならない（RISK-0015/0016） 🔵 *config_schema.py 全11モデルのextra="forbid"設定・test_config_schema.py TestExtraFieldsRejected 8テストより*
- REQ-062: 設定読込時、YAMLファイルがdict/mappingに解決されない場合（空ファイル、リスト等）はValueErrorで拒否しなければならない 🔵 *config_schema.py load_and_validate_config のisinstance(data, dict)チェック・test_config_schema.py TestMalformedYAML より*

#### 外挿安全性の追加要件

- REQ-063: cap_update()は非有限（NaN/Inf）の更新テンソルを検出した場合、NaNを伝播させる代わりにゼロテンソルを返さなければならない。これによりinf * 0 = NaNの伝播を防止する 🔵 *extrapolator.py cap_updateのtorch.isfiniteチェック・test_extrapolation_safety_direct.py テスト更新より*

#### ロールバック安全性の追加要件

- REQ-064: RollbackManagerはスナップショット保存時にNaN/Inf値をサニタイズ（NaN→0.0, +Inf→1e6, -Inf→-1e6）し、ロールバック時に破損状態を復元しないことを保証しなければならない（RISK-0074） 🔵 *rollback_manager.py _sanitize_snapshot関数・test_rollback_manager.py test_save_sanitize_nan/inf より*
- REQ-065: RollbackManagerは履歴サイズをmax_history（デフォルト100）で制限し、超過時に最古のエントリをFIFOで破棄しなければならない（RISK-0074） 🔵 *rollback_manager.py max_history パラメータ・test_rollback_manager.py test_max_history_bounds/fifo_eviction より*

#### メトリクス・差分追跡の安全性要件

- REQ-066: metrics.cosine_similarity()は辞書間でキーの不一致がある場合、欠落キーを安全にスキップし、完全に不一致の場合は0.0を返さなければならない 🔵 *metrics.py cosine_similarity のk not in b チェック・test_metrics.py TestCosineSimilarityKeyMismatch 3テストより*
- REQ-067: DeltaTracker._compute_stats()は非有限ノルムのテンソルをスキップし、全テンソルが非有限の場合はゼロstatsを返さなければならない 🔵 *delta_tracker.py _compute_stats のmath.isfinite(norm_val)チェック・test_delta_tracker.py test_compute_stats_skips_nan/inf_tensor より*
- REQ-068: DeltaTrackerはnorm_historyに非有限値を追加してはならず、math.isfinite(norm)ガードで異常検出・収束トレンドの腐敗を防止しなければならない 🔵 *delta_tracker.py compute_and_record のmath.isfinite(norm)ガード・test_delta_tracker.py test_tracker_nan/inf_norm_not_appended_to_history より*

## Edgeケース

### エラー処理

- EDGE-001: eval中にeval_lossがモデルのdropout/training mode状態を変更する場合、評価後に元の状態を復元しなければならない 🔵 *eval_loss.py context managerより*
- EDGE-002: レイヤーインデックスが非連続（例: DeltaNet + Attention混在）の場合でもレイヤーサンプリングが正しく動作しなければならない 🔵 *layer_sampler.py 非連続対応コミット(fcae9fb)より*
- EDGE-003: LoRAアダプタ評価後のマージモデル一時ファイルを確実にクリーンアップしなければならない 🔵 *run_eval_lora.sh trap handler実装済み（TASK-0127）*

### 境界値

- EDGE-101: velocity初期状態（初回更新）でのcosine similarityは0.0を返さなければならない 🔵 *velocity.py 初期状態テストより*
- EDGE-102: 外挿更新が0に近い場合、cap_updateは元の更新を変更せず通過させなければならない 🔵 *extrapolator.py テストより*
- EDGE-103: ロールバック履歴が空の状態でpop/clear操作が呼ばれた場合、エラーなく処理しなければならない 🔵 *rollback_manager.py テストより*
- EDGE-104: alpha提案が[alpha_min, alpha_max]の範囲外にならないことを保証しなければならない 🔵 *random_walk_controller.py テストより*
- EDGE-105: cosine_similarity計算時、deltaのキーにstateに存在しないキーが含まれていてもエラーなく処理しなければならない 🔵 *velocity.py KeyError修正(6717ee8)より*
- EDGE-106: 外挿適用時、active_namesにvelocity辞書に存在しないキーが含まれていてもKeyErrorを発生させず安全にスキップしなければならない 🔵 *extrapolator.py KeyError修正(TASK-0010 Bug #1)より*
- EDGE-107: diff_lora計算時、after辞書にbefore辞書の全キーが含まれていない場合でもKeyErrorを発生させず安全に処理しなければならない 🔵 *lora_state.py KeyError修正(TASK-0010 Bug #2)より*
- EDGE-108: load_lora_snapshot時、state辞書にモデルパラメータの全キーが含まれていない場合でもKeyErrorを発生させず安全に処理しなければならない 🔵 *lora_state.py KeyError修正(TASK-0010 Bug #3)より*
- EDGE-109: eval_format/eval_taskの評価中に例外が発生した場合、model.train()をfinallyブロックで確実に呼び出し元の状態を復元しなければならない 🔵 *eval_format.py/eval_task.py try/finally修正(TASK-0010 Bug #5, #6)より*
- EDGE-110: CycleStateの削減率はbackward_passes + extrapolation_steps = 0の場合に0.0を返さなければならない 🔵 *cycle_state.py `reduction_rate` 初期状態テストより*
- EDGE-111: CycleStateの受理率は受理・拒否ともに0件の場合に0.0を返さなければならない 🔵 *cycle_state.py `acceptance_rate` 初期状態テストより*
- EDGE-112: DeltaTrackerの異常検出は履歴が3件未満の場合Falseを返さなければならない 🔵 *delta_tracker.py `is_anomalous` テストより*
- EDGE-113: DeltaTrackerの収束トレンドはデータが2件未満の場合0.0を返さなければならない 🔵 *delta_tracker.py `convergence_trend` テストより*
- EDGE-114: DeltaTrackerのnorm履歴はmax_historyを超過した場合、最古のエントリを破棄しなければならない 🔵 *delta_tracker.py `max_history` テストより*
- EDGE-115: Velocityのmagnitude_historyがmax_historyを超過した場合、最古のエントリを破棄しなければならない 🔵 *velocity.py `max_history` テスト・bbcb7e7より*
- EDGE-116: Velocityのis_magnitude_anomalousはmagnitude履歴が3件未満の場合Falseを返さなければならない 🔵 *velocity.py `is_magnitude_anomalous` テスト・bbcb7e7より*
- EDGE-117: Velocityのmagnitude_trendはデータが2件未満の場合0.0を返さなければならない 🔵 *velocity.py `magnitude_trend` テスト・bbcb7e7より*
- EDGE-118: 連続する拒否（penalize）サイクルでlrがlr_minを下回ってはならない。lr_reject_decay=0.5の場合、50回連続拒否後もlr==lr_minでなければならない 🔵 *random_walk_controller.py lrクランプ・test_random_walk_controller.py `test_lr_clamps_at_lr_min_under_repeated_rejects` より*
- EDGE-119: 連続する受理（reward）サイクルでlrがlr_maxを上回ってはならない。50回連続受理後もlr==lr_maxでなければならない 🔵 *random_walk_controller.py lrクランプ・test_random_walk_controller.py `test_lr_clamps_at_lr_max_under_repeated_accepts` より*
- EDGE-120: 交互の受理/拒否サイクル（100回）でlrが常に[lr_min, lr_max]内に留まらなければならない 🔵 *random_walk_controller.py lrクランプ・test_random_walk_controller.py `test_lr_alternating_accept_reject_stays_in_bounds` より*
- EDGE-121: 外挿適用後にLoRAパラメータがNaNまたはInfになった場合、外挿を棄却として扱いロールバックしなければならない 🔵 *REQ-056対応・trainer_loop.py NumericalInstabilityErrorパターンより*
- EDGE-122: config_schema.pyのdtypeフィールドに"bfloat16", "float16", "float32"以外の文字列を指定した場合、Pydantic検証で拒否されなければならない 🔵 *REQ-058対応・ActiveLayerStrategy Literal enumパターンより*
- EDGE-123: 統合テストで非有限パラメータ検出後にcycle_state.record_cycle()がaccepted=Falseで正しく呼び出されることを検証しなければならない 🔵 *train_tg_lora.py 非有限回復パス(347-354行)・AI_HUB_MAKE_RUN_FEEDBACKより*
- EDGE-124: 統合テストで非有限パラメータ検出後にcontroller.update_layer_scores()がactive_indicesと-1.0で呼び出されることを検証しなければならない 🔵 *train_tg_lora.py 非有限回復パス(340行)・AI_HUB_MAKE_RUN_FEEDBACKより*
- EDGE-125: 統合テストで非有限パラメータ検出後のcontinueで、通常のaccept/rollback評価パス（eval_loss外挿後・_decide_accept_rollback）がスキップされることを検証しなければならない 🔵 *train_tg_lora.py 355行 continue・AI_HUB_MAKE_RUN_FEEDBACKより*
- EDGE-126: 設定YAMLにタイポフィールド（例: "lerning_rate"）が含まれる場合、Pydanticのextra='forbid'により学習開始前に拒否されなければならない 🔵 *config_schema.py extra="forbid"・test_config_schema.py test_typo_in_training_learning_rate等8テストより*
- EDGE-127: 空のYAMLファイルまたはリストに解決されるYAMLファイルは、ValueErrorで拒否されなければならない 🔵 *config_schema.py load_and_validate_config・test_config_schema.py TestMalformedYAML より*
- EDGE-128: cap_update()にInfテンソルが入力された場合、NaNを返すのではなくゼロテンソルを返さなければならない 🔵 *extrapolator.py cap_updateのtorch.isfiniteチェック・test_extrapolation_safety_direct.py test_cap_update_inf_returns_zeros_instead_of_nan より*
- EDGE-129: ロールバックスナップショットにNaN値が含まれる場合、0.0にサニタイズされて保存されなければならない 🔵 *rollback_manager.py _sanitize_snapshot・test_rollback_manager.py test_save_sanitize_nan より*
- EDGE-130: ロールバックスナップショットにInf値が含まれる場合、±1e6にクランプされて保存されなければならない 🔵 *rollback_manager.py _sanitize_snapshot・test_rollback_manager.py test_save_sanitize_inf より*
- EDGE-131: ロールバック履歴がmax_historyを超過した場合、最古のエントリがFIFOで破棄されなければならない 🔵 *rollback_manager.py max_history・test_rollback_manager.py test_max_history_fifo_eviction より*
- EDGE-132: metrics.cosine_similarity()に完全に不一致のキーセット（共通キーなし）が入力された場合、0.0を返さなければならない 🔵 *metrics.py cosine_similarity・test_metrics.py test_completely_disjoint_keys より*
- EDGE-133: DeltaTracker._compute_stats()に全てNaNのテンソルが入力された場合、ゼロstats（total_norm=0.0, max_component=0.0, mean_abs=0.0）を返さなければならない 🔵 *delta_tracker.py _compute_stats・test_delta_tracker.py test_compute_stats_all_nan_returns_zeros より*
- EDGE-134: DeltaTrackerにNaNまたはInfのnormが入力された場合、norm_historyに追加してはならない 🔵 *delta_tracker.py compute_and_record・test_delta_tracker.py test_tracker_nan/inf_norm_not_appended_to_history より*
- EDGE-135: RunMetrics.write_footerの出力レコードは、モック学習ループ実行後にperplexityフィールドが有限値（float）またはNone（未計算時）であることを保証しなければならない。NaN/Infのperplexityが入力された場合はNoneとして出力されなければならない 🔵 *run_metrics.py _sanitize_perplexity・AI_HUB_MAKE_RUN_FEEDBACK「integration test exercising the full perplexity pipeline」より*
- EDGE-136: RandomWalkController.accept()の相対許容誤差判定は、loss_pilotの大きさ（1e-3〜1e3の範囲）に依存せず一貫した受理/拒否判定を行わなければならない。絶対許容誤差から相対許容誤差への移行(02582db)により、大きなloss値で受理が緩くなる挙動が導入されているため、プロパティベーステストで検証しなければならない 🔵 *tests/test_accept_property.py test_accept_magnitude_consistency・REQ-071プロパティベーステスト実装済み*
- EDGE-137: RandomWalkControllerのコンストラクタに負のK_candidatesが渡された場合、ValueError("All K_candidates must be positive")で拒否されなければならない 🔵 *random_walk_controller.py バリデーション・test_random_walk_controller.py test_reject_negative_K_candidates・cceccdeコミットより*
- EDGE-138: RandomWalkControllerのコンストラクタに0を含むN_candidatesが渡された場合、ValueError("All N_candidates must be positive")で拒否されなければならない 🔵 *random_walk_controller.py バリデーション・test_random_walk_controller.py test_reject_zero_N_candidates・cceccdeコミットより*
- EDGE-139: RandomWalkControllerのコンストラクタにlr_min >= lr_maxが渡された場合、ValueErrorで拒否されなければならない 🔵 *random_walk_controller.py バリデーション・test_random_walk_controller.py test_reject_lr_min_ge_lr_max・cceccdeコミットより*
- EDGE-140: RandomWalkControllerのコンストラクタにalpha_min >= alpha_maxが渡された場合、ValueErrorで拒否されなければならない 🔵 *random_walk_controller.py バリデーション・test_random_walk_controller.py test_reject_alpha_min_ge_alpha_max・cceccdeコミットより*
- EDGE-141: eval_loss()に空のデータローダー（バッチ数0）が渡された場合、0.0ではなくNaNを返さなければならない 🔵 *eval_loss.py count==0 時NaN返却・test_eval_loss.py test_eval_loss_empty_dataloader・e19da0fコミットより*
- EDGE-142: eval_loss_detailed()に空のデータローダーが渡された場合、avg_loss=NaN, perplexity=infのEvalLossResultを返さなければならない 🔵 *eval_loss.py batch_losses空判定・test_eval_loss.py test_eval_loss_detailed_empty_dataloader・e19da0fコミットより*
- EDGE-143: rollback_manager.rollback()がRuntimeErrorまたはIndexErrorを送出した場合、学習ループはクラッシュせずエラーをログ出力して継続しなければならない 🔵 *train_tg_lora.py try-catch rollback・e19da0fコミットより*
- EDGE-144: 外挿後のeval_lossでloss_afterがNaN/Infの場合、loss_afterをfloat("inf")に設定し受理判定が必ず拒否になることを保証しなければならない 🔵 *train_tg_lora.py math.isfinite guard・e19da0fコミットより*
- EDGE-145: TGLoRAParamsにk_explore_prob=0.0またはk_explore_prob=1.0を渡した場合、ValueErrorで拒否されなければならない。探索確率は開区間(0.0, 1.0)でなければならない 🔵 *config_schema.py Field(gt=0.0, lt=1.0)・test_random_walk_controller.py test_explore_prob_config_schema_validation・本コミットより*
- EDGE-146: metrics.total_norm()に全て非有限（NaN/Inf）のテンソルのみが含まれる場合、0.0を返さなければならない 🔵 *test_metrics.py test_all_nonfinite_returns_zero・1bb591dコミットより*
- EDGE-147: metrics.per_layer_norms()で一部レイヤーのテンソルが非有限の場合、非有限テンソルをスキップし有限テンソルのみで正しいノルムを計算しなければならない 🔵 *test_metrics.py test_nan/inf_tensor_skipped・1bb591dコミットより*
- EDGE-148: _compute_pilot_averageに全て非有限のstep_lossesが入力された場合、NaNとfinite_count=0を返さなければならない 🔵 *train_tg_lora.py _compute_pilot_average・test_training_pure_functions.py・1bb591dコミットより*
- EDGE-149: _compute_pilot_averageに有限と非有限が混在するstep_lossesが入力された場合、有限値のみで平均・min・maxを計算しなければならない 🔵 *train_tg_lora.py _compute_pilot_average・test_training_pure_functions.py・1bb591dコミットより*

#### Phase 28: ActivationCache・決定論的モード・中間ロールバック境界値

- EDGE-150: ActivationCacheのdetermine_split_layer()は、モデルにdecoderレイヤーが存在しない場合、またはsplit_layer_idx >= num_layersの場合はNoneを返し、非キャッシュ評価にフォールバックしなければならない 🔵 *src/tg_lora/activation_cache.py determine_split_layer()・0bc7236コミットより*
- EDGE-151: ActivationCacheのeval_from_cache()は、キャッシュが空の場合またはsplit_layer_idxがNoneの場合にエラーを発生させず、非キャッシュ評価にフォールバックしなければならない 🔵 *src/tg_lora/activation_cache.py eval_from_cache()・0bc7236コミットより*
- EDGE-152: enable_random_walk=falseの場合、propose()の連続呼び出しは常に同じ（K, N, alpha, beta, lr, strategy）を返し、内部状態を変更してはならない 🔵 *src/tg_lora/random_walk_controller.py enable_random_walk・0782acdコミットより*
- EDGE-153: force_top_layers_only=trueの場合、active_layer_strategyがlisa_like_weightedやmiddle_randomに設定されていても、"last_25_percent"戦略を強制しなければならない 🔵 *src/training/train_tg_lora.py force_top_layers_only・555287dコミットより*
- EDGE-154: _decide_accept_rollback()のmoving_avg_windowがaccepted_valid_historyのサイズより大きい場合、利用可能な履歴のみで平均を計算しなければならない。履歴が空の場合はloss_pilotをベースラインとしなければならない 🔵 *src/training/train_tg_lora.py _decide_accept_rollback()・555287dコミットより*
- EDGE-155: soft_accept_temperature=0.0の場合、Metropolis-Hastings確率的受理は完全に無効化され、従来の閾値判定のみを行わなければならない 🔵 *src/training/train_tg_lora.py soft accept・d3f834bコミットより*
- EDGE-156: confident_skip_cos=0.0の場合、confident_skipメカニズムは発動せず、全サイクルで通常の評価を実行しなければならない 🔵 *src/training/train_tg_lora.py confident_skip・555287dコミットより*
- EDGE-157: K-step中間ロールバックで全中間点が直近valid損失より悪化している場合、W0へのフルロールバックを実行し、velocityをdelta=0で更新しなければならない 🔵 *src/training/train_tg_lora.py pilot_full_rollback・64bd8a8コミットより*
- EDGE-158: moving_avg_windowが1に設定された場合、直近1件のaccepted valid lossがベースラインとなり、従来のloss_pilot比較と等価でなければならない 🔵 *src/training/train_tg_lora.py _decide_accept_rollback()・555287dコミットより*

#### Phase 29: OptimizerLifecycleManager・キャッシュメトリクス境界値

- EDGE-159: reuse_state_reset_experimentalポリシーでprepare_for_cycleを複数回呼び出した場合、state tensorのdata_ptrが不変（メモリ再確保なし）でなければならない。zero-reset後に全state tensorの非ゼロ要素数が0でなければならない 🔵 *test_optimizer_lifecycle.py test_reuse_policy_zeros_state_in_place・3fdf57aコミットより*
- EDGE-160: recreate_per_cycleポリシーでprepare_for_cycleを呼び出すたびに、前回と異なるoptimizerインスタンスが返され、新規optimizerのstateが空でなければならない 🔵 *test_optimizer_lifecycle.py test_recreate_policy_returns_new_optimizer・3fdf57aコミットより*
- EDGE-161: RunMetrics.write_headerはcfg.trainingにoptimizer_lifecycleフィールドが存在しない場合でもKeyErrorを発生させず、Noneを出力しなければならない 🔵 *src/utils/run_metrics.py getattr(cfg.training, "optimizer_lifecycle", None)・2ac68d1コミットより*
- EDGE-162: activation_cache_eligible_countが0の場合、hit_rate計算でZeroDivisionErrorを発生させず、hit_rateを0.0としなければならない 🔵 *src/training/train_tg_lora.py activation_cache_hit_rate計算・f45a269コミットより*
- EDGE-163: benchmark_optimizer_lifecycle.pyはreuse_state_reset_experimentalの前後でstate tensorのdata_ptrが一致することを検証し、結果JSONのreuse_state_ptrs_preservedフィールドで報告しなければならない 🔵 *scripts/benchmark_optimizer_lifecycle.py pointers_before/pointers_after比較・d69a57dコミットより*

#### Phase 33: 高速化・警告・統合テスト境界値

- EDGE-164: diff_loraのscale==0.0の場合、返却される全テンソルがゼロであることを検証しなければならない。scale==1.0の場合、返却テンソルがafter[k] - before[k]と完全に一致することを検証しなければならない 🔵 *lora_state.py diff_lora fast paths・7a643a9コミットより*
- EDGE-165: cosine_similarityに完全に直交するベクトル（内積=0、ノルム>0）を入力した場合、警告が発せられ、0.0が返されなければならない 🔵 *metrics.py cosine_similarity warnings・7a643a9コミットより*
- EDGE-166: _get_decoder_layersが全候補パスの探索に失敗した場合、エラーメッセージに候補パス数が含まれなければならない 🔵 *activation_cache.py _get_decoder_layers enhanced error・7a643a9コミットより*
- EDGE-167: AsyncCacheBuilder統合テストはCPU上の軽量モデル（GPT-2 tiny等）でビルド→完了確認→DataLoader差し替え→学習継続のフルライフサイクルを検証しなければならない 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「integration test with mock model on CPU to validate the full lifecycle (build → wait → load) end-to-end」より*

#### Perplexity E2Eパイプライン検証

- REQ-069: システムはモック学習ループ（baseline/tg_lora両モード）を実行し、RunMetrics.write_footerの出力にbest_perplexityとして有限値が含まれることを検証するE2Eテストを備えなければならない。テストは（a）eval_loss_detailedが有限perplexityを返す場合、footerに有限floatが含まれること、（b）eval未実行の場合、footerのperplexityがNoneであることの両方を検証しなければならない 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「E2E integration test that runs a short mock training loop and asserts the RunMetrics footer contains a finite best_perplexity value」より*
- REQ-070: train_tg_lora.pyとtrain_baseline_qlora.pyはperplexity取り扱い（eval_result.perplexityの保存、best_perplexityの追跡、write_footerへのperplexity引数渡し）においてパリティを保たなければならない。パリティ検証は比較テストまたはパラメータ化テストで確認しなければならない 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「verify train_tg_lora.py has parity with train_baseline_qlora.py's perplexity plumbing」・両trainerのwrite_footer呼び出し実装より*

#### プロパティベーステスト

- REQ-071: RandomWalkController.accept()はhypothesis等のプロパティベーステストフレームワークで、以下のプロパティを検証しなければならない: (a) loss_after <= loss_pilotの場合は常にTrueを返す（大きさに無関係）、(b) loss_pilotとloss_afterが等しい場合は常にTrueを返す、(c) 非有限値入力（NaN/Inf）の場合は常にFalseを返す、(d) relative tolerance判定がloss_pilotの大きさ（1e-6〜1e6）に対して一貫している 🔵 *tests/test_accept_property.py: TC-071-P19-01~04全実装・hypothesis @given によるプロパティベーステスト完了*

#### プロセス要件

- REQ-072: docs/llm-wiki/ 配下のブートストラップ（自動生成メタデータ：バイト数・行数・モジュール一覧等）は、正本（src/, tests/, configs/）に実質的変更がない場合は独立したコミットとせず、既存の実質的コミットに含めるかスキップしなければならない。コミット数の水増しを防ぐためである 🟡 *AI_HUB_MAKE_RUN_FEEDBACK「The wiki bootstrap commit is pure metadata churn — skip wiki bootstraps when no canonical source content changed」・ba286a2コミットより*

#### 公開API・入力検証

- REQ-073: システムはsrc/tg_lora/__init__.pyを通じて全コアアルゴリズムコンポーネント（Velocity, apply_extrapolation, cap_update, DeltaTracker, compute_mean_delta, CycleState, select_active_layers, get_num_layers, StrategyName, RollbackManager, RandomWalkController, snapshot_lora, load_lora_snapshot, diff_lora, cosine_similarity, total_norm, per_layer_norms）を公開APIとしてエクスポートしなければならない。__all__リストで明示的にエクスポートを管理しなければならない 🔵 *src/tg_lora/__init__.py cceccde コミット・pyproject.toml packages設定より*
- REQ-074: RandomWalkControllerはコンストラクタで以下の入力検証を行い、不正な値に対してValueErrorを発生させなければならない: (a) 全K_candidates > 0、(b) 全N_candidates > 0、(c) lr_min < lr_max、(d) alpha_min < alpha_max 🔵 *random_walk_controller.py __init__ バリデーション・test_random_walk_controller.py test_reject_* 4テスト・cceccdeコミットより*

#### 評価エッジケース強化

- REQ-075: eval_loss()は空データローダー（バッチ数0）の場合にfloat("nan")を返さなければならない。eval_loss_detailed()は空データローダーの場合にavg_loss=NaN, min_loss=NaN, max_loss=NaN, perplexity=infのEvalLossResultを返さなければならない。誤解を招く0.0損失を返してはならない 🔵 *eval_loss.py 空データローダーNaN返却・test_eval_loss.py test_eval_loss_empty_dataloader・e19da0fコミットより*

#### ロールバック安全性強化

- REQ-076: train_tg_lora.pyのrollback_manager.rollback()呼び出しはtry-catchで囲み、RuntimeError・IndexErrorを捕捉してログ出力しなければならない。ロールバック失敗が学習全体をクラッシュさせてはならない 🔵 *train_tg_lora.py rollback try-catch・e19da0fコミットより*
- REQ-077: train_tg_lora.pyのaccept/rollback判定前にloss_afterが有限値でない場合、loss_afterをfloat("inf")に設定し、非有限評価損失が受理判定を通過しないことを保証しなければならない 🔵 *train_tg_lora.py math.isfinite guard・e19da0fコミットより*

#### 共有ユーティリティ

- REQ-078: システムはInfiniteBatchIterator（src/training/batch_iter.py）を共有ユーティリティとして提供し、TG-LoRA・ベースライン両trainerで使用しなければならない。空データセットでの初期化時にはValueErrorを発生させなければならない 🔵 *src/training/batch_iter.py InfiniteBatchIterator・a020e5bコミットより*
- REQ-079: システムはsave_checkpoint（src/utils/checkpoint.py）を共有ヘルパーとして提供し、TG-LoRA・ベースライン両trainerのチェックポイント保存で使用しなければならない 🔵 *src/utils/checkpoint.py save_checkpoint・a020e5bコミットより*
- REQ-080: RandomWalkControllerの戦略一覧はtyping.get_args(StrategyName)から自動生成し（_ALL_STRATEGIES）、ハードコードされた戦略リストの重複を排除しなければならない 🔵 *random_walk_controller.py _ALL_STRATEGIES・a020e5bコミットより*

#### チェックポイント・バッチイテレータ堅牢性

- REQ-081: save_checkpoint()は保存後にreadback検証（保存先ディレクトリの存在確認・ファイル数確認）を行い、不完全なチェックポイントを検出した場合に警告をログ出力しなければならない 🔵 *src/utils/checkpoint.py readback検証実装・test_checkpoint.py 7テスト・TASK-0051完了より*
- REQ-082: InfiniteBatchIteratorは単一バッチのデータローダー・デバイスキャスト・dtype維持のエッジケースを専用テストで検証しなければならない 🔵 *test_infinite_batch_iterator.py TestSingleBatchDataloader/TestDeviceCastEdgeCases/TestDtypePreservation・TASK-0052完了より*

#### ロールバック・非有限損失の運用観測性

- REQ-083: train_tg_lora.pyの非有限loss_afterガード発動時にlogger.warningでログ出力しなければならない。デバッグ可能性を確保するためである 🔵 *train_tg_lora.py logger.warning追加・test_training_integration.py TestNonFiniteLossAfterWarning 3テスト・TASK-0053完了より*
- REQ-084: RollbackManager.rollback()が例外を送出するシナリオのエンドツーエンドテスト（モックでrollbackをraiseさせ、学習継続または安全な失敗を検証）を備えなければならない 🔵 *test_training_integration.py TestRollbackFailureResilience 2テスト・test_training_integration.py TestNonFiniteParamsRollbackException E2Eテスト・TASK-0054完了より*

#### 探索確率パラメータの設定可能性と検証

- REQ-085: RandomWalkControllerはk_explore_prob、n_explore_prob、beta_explore_prob、strategy_explore_probの4つの探索確率パラメータをコンストラクタで受け取り、propose()メソッドの探索頻度を制御しなければならない 🔵 *random_walk_controller.py 探索確率フィールド・propose()分岐・28f709bコミットより*
- REQ-086: 探索確率パラメータが省略された場合（None）、クラス定数のデフォルト値（k=0.4, n=0.4, beta=0.15, strategy=0.08）を使用しなければならない 🔵 *random_walk_controller.py __init__ None fallback・28f709bコミットより*
- REQ-087: k_explore_prob=0.0の場合はKが変化せず、k_explore_prob=1.0の場合はKが常に変化（候補非端インデックス時）することを検証するテストを備えなければならない。N, beta, strategyも同様に極端値での動作を検証しなければならない 🔵 *test_random_walk_controller.py test_zero/full_*_explore_prob 11テスト・本コミットより*
- REQ-088: TGLoRAParamsスキーマは4つの探索確率パラメータをfloat型で受け付け、gt=0.0, lt=1.0の範囲制約で検証しなければならない。0.0以下と1.0以上の値はValueErrorで拒否されなければならない 🔵 *config_schema.py TGLoRAParams exploration probability fields・test_random_walk_controller.py test_explore_prob_config_schema_validation・本コミットより*

#### メトリクスNaN/Inf安全性ガード

- REQ-089: metrics.total_norm()は非有限（NaN/Inf）のテンソルノルムをスキップし、有限テンソルのみの合計を返さなければならない。全テンソルが非有限の場合は0.0を返さなければならない。delta_tracker._compute_stats（Phase 14）やextrapolator.cap_updateと一貫した安全性パターンを適用しなければならない 🔵 *metrics.py total_norm math.isfiniteチェック・test_metrics.py TestTotalNorm test_nan/inf_tensor_skipped/test_all_nonfinite_returns_zero・1bb591dコミットより*
- REQ-090: metrics.per_layer_norms()は非有限（NaN/Inf）のテンソルノルムをスキップし、有限テンソルのみでレイヤー別ノルムを集計しなければならない。一部テンソルが非有限でも他の有限テンソルのノルムは正しく計算されなければならない 🔵 *metrics.py per_layer_norms math.isfiniteチェック・test_metrics.py TestPerLayerNorms test_nan/inf_tensor_skipped・1bb591dコミットより*
- REQ-091: _compute_pilot_averageはstep_lossesリストの非有限値（NaN/Inf）をフィルタリングしてから平均を計算しなければならない。全てのlossが非有限の場合はNaNとfinite_count=0を返さなければならない。min_loss/max_lossも有限値のみから計算しなければならない 🔵 *train_tg_lora.py _compute_pilot_average finite_losses フィルタリング・test_training_pure_functions.py・1bb591dコミットより*
- REQ-092: train_tg_lora.pyは探索確率パラメータ（k_explore_prob, n_explore_prob, beta_explore_prob, strategy_explore_prob）を設定からRandomWalkControllerのコンストラクタに渡さなければならない。設定省略時はNoneを渡し、RandomWalkControllerのデフォルト値が使用されなければならない 🔵 *train_tg_lora.py RandomWalkController初期化・test_random_walk_controller.py config-to-controller integration tests・1bb591dコミットより*
- REQ-093: config-to-controller integration testsはYAML設定の探索確率値がOmegaConf経由でRandomWalkControllerに正しく伝播されることを検証しなければならない。明示的値、省略時デフォルト、実際の9b_tg_lora.yaml値、propose()の行動への影響の全シナリオをカバーしなければならない 🔵 *test_random_walk_controller.py TestConfigToControllerIntegration・03d1e46コミットより*

#### MLflow実験管理高度化

- REQ-094: システムはsave_checkpoint()実行後にMLflowLogger.log_artifact()でチェックポイントディレクトリをMLflowアーティファクトとしてロギングしなければならない。mlflow.enabled=Falseの場合はロギングをスキップしなければならない 🔵 *mlflow_logger.py log_artifact()既存メソッド・train_tg_lora.py save_checkpoint呼び出し・TASK-0056より*
- REQ-095: MLflowLoggerはコンテキストマネージャ进入時に設定情報から「{実験名}_{タイムスタンプ}_{K}-{N}」形式のラン名を自動生成し、主要ハイパーパラメータ（K, N, alpha, beta, lr）をMLflowタグとして記録しなければならない 🔵 *mlflow_logger.py __enter__既存パターン・set_tag()既存メソッド・TASK-0057より*
- REQ-096: train_tg_lora.pyは各サイクルでvelocity magnitude、delta total_norm、convergence_trend、acceptance_rate、reduction_rateをMLflowメトリクスとして記録しなければならない。現在RunMetrics JSONLにのみ記録されているTG-LoRA特化メトリクスをMLflowにも送信しなければならない 🔵 *train_tg_lora.py既存メトリクス追跡・MLflowLogger.log_metrics()・TASK-0058より*
- REQ-097: MLflowLoggerは一時的なネットワーク障害（ConnectionError, Timeout）に対して指数バックオフリトライ（最大3回）を行わなければならない。リトライ対象外の例外は即座に失敗させなければならない 🔵 *src/utils/mlflow_logger.py _retry_mlflow() 実装・ConnectionError/Timeout 指数バックオフ・TASK-0059より*

#### 実験分析ツール

- REQ-098: システムはRunMetrics JSONLログを読み込み、ベストロス・ベストパープレキシティ・サイクル履歴をクエリするAPIを提供しなければならない。get_best_loss(), get_best_perplexity(), get_cycle_history(), list_runs()等の関数を提供する 🔵 *run_metrics.py既存JSONL形式・orjson依存・TASK-0060より*
- REQ-099: システムは複数実験ランの結果を横断比較するCLIサブコマンドを提供し、ベストパフォーマンスを自動選出し、richライブラリでダッシュボード表示しなければならない 🔵 *compare_runs.py既存比較ロジック・rich>=13.7依存・TASK-0061より*
- REQ-100: システムはacceptance rate推移、velocity magnitude推移、layer score分布、ハイパーパラメータ探索軌跡の可視化プロット関数を提供しなければならない 🔵 *scripts/compare_runs.py plot_acceptance_rate/plot_velocity_magnitude/plot_layer_scores/plot_hyperparams/plot_reduction_rate 実装・TASK-0062より*

#### 本番運用品質

- REQ-101a: システムはチェックポイントから学習状態（CycleState, RandomWalkController, Velocity, DeltaTracker）を復元し、中断箇所から学習を再開する機能を提供しなければならない 🔵 *train_tg_lora.py既存save_checkpoint・CycleState dataclass・TASK-0063より*
- REQ-102a: システムはOOM時・CUDA エラー時にチェックポイントを保存してgracefulに終了し、再開可能な状態を確保しなければならない 🔵 *train_tg_lora.py既存try/finally・TASK-0063より*

### Phase 27: チェックポイントシリアライズ・運用スクリプト（REQ-103~109）

#### チェックポイントシリアライズ・デシリアライズ

- REQ-103: RandomWalkControllerはサイクル状態をsummary()でシリアライズ可能な辞書形式（K, N, alpha, beta, lr, strategy, layer_scores, cycle/accept/rollback counts, boost/decay params）で出力し、from_dict()クラスメソッドで完全に復元可能でなければならない 🔵 *random_walk_controller.py ControllerState summary()/from_dict()・4435fdeコミットより*
- REQ-104: CycleStateはfrom_dict()クラスメソッドでsummary()出力から完全に再構築可能でなければならない。空辞書・部分データの場合はデフォルト値で初期化しなければならない 🔵 *cycle_state.py from_dict()・04a7581コミットより*
- REQ-105: システムはTrainingState dataclass（CycleState + ControllerState + Velocity + DeltaTracker状態を統合）を提供し、save_training_state() / load_training_state()でディスクへのシリアライズ・デシリアライズを行わなければならない。PyTorch tensorはCPU変換・historyのリスト化を行い、JSON形式で保存しなければならない 🔵 *src/utils/checkpoint.py TrainingState・save_training_state()/load_training_state()・57739faコミットより*

#### 運用診断スクリプト

- REQ-106: システムは診断スクリプト（scripts/diagnose.py）によるGPU状態・チェックポイント完全性・設定バリデーション・ログ分析の自動ヘルスチェックを提供しなければならない。各チェック結果はCheckResult（status: ok/warn/error）として統一されなければならない 🔵 *scripts/diagnose.py CheckResult・check_gpu()/check_checkpoint()/check_config()/check_logs()・b942b4bコミットより*
- REQ-107: システムは障害回復スクリプト（scripts/recover.py）によるOOM/CUDA/NaNエラーの自動分析（analyze_fault）・チェックポイントサニタイズ（sanitize_checkpoint）・復旧設定生成（generate_recovery_config）・完全自動回復（apply_remediation）を提供しなければならない 🔵 *scripts/recover.py・66987e4コミットより*

#### CI パイプライン

- REQ-108: Makefileはciターゲットを提供し、lint（ruff check + format check）+ テスト（pytest）+ スクリプトインポート健全性チェック（diagnose.py/recover.pyの正常import確認）を単一コマンドで実行しなければならない。インポートパスエラーを早期に検出しなければならない 🔵 *Makefile ci target・AI_HUB_MAKE_RUN_FEEDBACK「add a Makefile ci target that runs both with --dry-run or a mock flag to catch import/path regressions early」より*

#### ドキュメント完全性

- REQ-109: システムはAPIリファレンス（docs/api_reference.md）で全34エクスポート関数・クラスのシグネチャ・パラメータ・使用例を文書化しなければならない 🔵 *docs/api_reference.md 既存610行・b942b4bコミットより*

### Phase 28: ActivationCache・決定論的モード・中間ロールバック・高度Accept/Rollback（REQ-110~118）

#### ActivationCache（レイヤースキップ評価最適化）

- REQ-110: システムはActivationCacheクラスにより、評価時のフォワードパス計算を最適化するレイヤースキップ評価を提供しなければならない。事前フォワードで指定スプリットレイヤーの隠れ状態をキャッシュし、後続評価ではキャッシュからの部分フォワードのみを実行しなければならない 🔵 *src/tg_lora/activation_cache.py ActivationCache・eval_and_cache()/eval_from_cache()・0bc7236コミットより*
- REQ-111: ActivationCacheはCachedBatch dataclass（hidden_states, attention_mask, position_ids, labels）でバッチ単位のキャッシュを管理し、forward pre-hookでスプリットレイヤー入力の隠れ状態を捕捉しなければならない。decoderレイヤーが検出できない場合、またはsplit_layer_idxが範囲外の場合は非キャッシュ評価にフォールバックしなければならない 🔵 *src/tg_lora/activation_cache.py CachedBatch・determine_split_layer()・0bc7236コミットより*
- REQ-112: train_tg_lora.pyは外挿後評価でActivationCacheを活用し、予測戦略が実際の戦略と一致する場合はキャッシュを再利用しなければならない。予測が不一致の場合はキャッシュを無効化し、通常評価にフォールバックしなければならない 🔵 *src/training/train_tg_lora.py ActivationCache統合・0bc7236/64bd8a8コミットより*

#### 決定論的モード

- REQ-113: RandomWalkControllerはenable_random_walkフラグをサポートし、falseの場合はpropose()が静的提案（現在値の変更なし）を返し、reward()/penalize()が適応調整をスキップし、adapt_to_convergence()を無効化しなければならない 🔵 *src/tg_lora/random_walk_controller.py enable_random_walk・0782acdコミットより*
- REQ-114: TGLoRAParamsスキーマはforce_top_layers_onlyフラグ（bool、デフォルトfalse）を提供しなければならない。trueの場合、train_tg_lora.pyはactive_layer_strategy設定を問わず"last_25_percent"戦略を使用し、ActivationCacheのスプリットレイヤー一貫性を保証しなければならない 🔵 *src/training/config_schema.py force_top_layers_only・src/training/train_tg_lora.py 555287dコミットより*

#### 高度Accept/Rollback判定

- REQ-115: システムは_decide_accept_rollback()で移動平均ベースライン判定をサポートしなければならない。accepted_valid_historyの直近moving_avg_window件の平均をベースラインとし、loss_afterがベースライン以下であれば受理しなければならない。historyがwindow件未満の場合はloss_pilotをベースラインとしなければならない 🔵 *src/training/train_tg_lora.py _decide_accept_rollback() moving average・555287d/d3f834bコミットより*
- REQ-116: システムはsoft_accept_temperatureパラメータによるMetropolis-Hastings確率的受理をサポートしなければならない。temperature > 0かつloss_after > baselineの場合、exp(-(loss_after - baseline) / temperature)の確率で受理を許可しなければならない。temperature = 0の場合は確率的受理を無効にしなければならない 🔵 *src/training/train_tg_lora.py Metropolis-Hastings soft accept・d3f834bコミットより*
- REQ-117: システムはconfident_skipメカニズムにより、velocity方向が高安定時に評価を省略し自動受理する機能を提供しなければならない。confident_skip_cos > 0かつcos_sim >= confident_skip_cos かつacceptance_rate >= 0.8 かつtotal_cycles >= confident_skip_min_cycles かつvelocityが非異常の場合に発動し、loss_after = loss_pilotとして自動受理しなければならない 🔵 *src/training/train_tg_lora.py confident_skip・555287dコミットより*

#### K-step中間ロールバック

- REQ-118: システムはpilotフェーズ中の各Kステップでdelta snapshotを記録し、pilot損失が直近valid損失+toleranceを超過した場合に中間ロールバックを実行しなければならない。全中間点（step 0 ~ K-1）をeval_lossで評価し、最良の中間点にロールバックしなければならない。全中間点が悪化している場合はW0へのフルロールバックを実行し、dWとvelocityをロールバック先状態で再計算しなければならない 🔵 *src/training/train_tg_lora.py intermediate_deltas/pilot_rollback・64bd8a8/a1ffe6dコミットより*

### Phase 29: OptimizerLifecycleManager・キャッシュメトリクス追跡（REQ-119~124）

#### Optimizerライフサイクル管理

- REQ-119: システムはOptimizerLifecycleManagerにより、サイクル間のAdamW optimizerライフサイクルを管理しなければならない。デフォルトポリシー（recreate_per_cycle）はサイクル毎にoptimizerを再生成し、実験的ポリシー（reuse_state_reset_experimental）は同一optimizerインスタンスを保持しstate tensorをin-place zero-resetして再利用しなければならない。reuse_policyではstate tensorのdata_ptrがサイクル間で不変であることを保証しなければならない 🔵 *src/training/optimizer_lifecycle.py OptimizerLifecycleManager・3fdf57a/d2e2a51コミット・test_optimizer_lifecycle.py より*
- REQ-120: TrainingConfigスキーマはoptimizer_lifecycleフィールド（OptimizerLifecyclePolicy型、デフォルト"recreate_per_cycle"）をサポートしなければならない 🔵 *src/training/config_schema.py TrainingConfig.optimizer_lifecycle・d69a57dコミットより*

#### メトリクス追跡

- REQ-121: RunMetrics.write_headerはoptimizer_lifecycle設定値を出力レコードに含めなければならない。設定にoptimizer_lifecycleフィールドが存在しない場合はNoneを出力しなければならない 🔵 *src/utils/run_metrics.py write_header・2ac68d1コミット・test_run_metrics.py より*
- REQ-122: train_tg_lora.pyはActivationCacheの利用状況（eligible count, hit count, hit rate）をサイクルサマリーと最終training summaryに記録しなければならない。RunMetrics.record_stepは各ステップでcache_built, cache_eligible, cache_hitを追跡しなければならない 🔵 *src/training/train_tg_lora.py activation_cache_*_count・src/utils/run_metrics.py record_step cache fields・f45a269コミットより*

#### ベンチマーク・実験サーフェス

- REQ-123: システムはscripts/benchmark_optimizer_lifecycle.pyにより、recreate_per_cycleとreuse_state_reset_experimentalの定常状態オーバーヘッド（prepare時間、step時間、メモリ増分）を比較するベンチマークを提供しなければならない 🔵 *scripts/benchmark_optimizer_lifecycle.py・d69a57dコミットより*
- REQ-124: configs/9b_tg_lora_optimizer_reuse_experimental.yamlは実験用optimizer再利用設定サーフェスとして提供され、enable_random_walk=false, force_top_layers_only=true, optimizer_lifecycle=reuse_state_reset_experimentalで決定論的比較を可能にしなければならない 🔵 *configs/9b_tg_lora_optimizer_reuse_experimental.yaml・d69a57dコミットより*

### Phase 31: trainable_lora_scope・prefix_feature_cache・Makefile検証（REQ-125~127）

#### LoRA学習範囲制御

- REQ-125: システムはTrainingConfigのtrainable_lora_scopeフィールドによりLoRA学習範囲を制御しなければならない。"all"の場合は全LoRAパラメータの勾配を有効にし、"last_25_percent"の場合は末尾25%のデコーダレイヤーのみ勾配を有効にしなければならない。configure_trainable_lora_scope関数がスコープ設定に基づきrequires_gradを制御し、有効なパラメータ名の集合とレイヤーインデックスの集合を返さなければならない 🔵 *src/model/lora_utils.py configure_trainable_lora_scope/set_all_lora_trainable/set_trainable_lora_layers/get_last_fraction_lora_layer_indices・src/training/config_schema.py TrainingConfig.trainable_lora_scope・test_lora_utils.py より*

#### Prefix Feature Cache

- REQ-126: システムはprefix_feature_cache_experimentalフラグによる評価時隠れ状態事前計算をサポートしなければならない。build_prefix_feature_dataset関数が指定スプリットレイヤーの前段隠れ状態を事前計算し、PrefixFeatureDatasetとして提供する。PrefixFeatureExampleはhidden_states, attention_mask, labels, split_layer_idx, position_idsを保持し、collate_prefix_feature_batchでバッチ化される。TrainingConfigはprefix_feature_cache_train/valid_quick/valid_fullでデータセット別の有効/無効、num_workers/pin_memory/persistent_workers/prefetch_factorでDataLoader設定を制御しなければならない 🔵 *src/tg_lora/prefix_feature_cache.py PrefixFeatureDataset/build_prefix_feature_dataset・src/training/config_schema.py TrainingConfig prefix_feature_cache_* fields・configs/9b_tg_lora_prefix_feature_cache_experimental.yaml・test_prefix_feature_cache.py より*

#### Makefile実験ターゲットの検証可能性

- REQ-127: Makefileはsmoke, ablation, bench-optimizer, train-tg-lora-optreuse, train-tg-lora-prefix, compare-prefix等の実験ターゲットを提供し、各ターゲットが対応するYAML設定ファイルを正しく参照してPydantic検証を通過しなければならない。テストによりsmokeターゲットの設定存在確認、ablation設定のスキーマ検証、bench-optimizerスクリプトのインポート健全性、実験configターゲットのYAML→Pydantic検証が自動確認されなければならない 🔵 *Makefile smoke/ablation/bench-optimizer/compare-prefix targets・test_makefile_targets.py・TASK-0071/0074 より*

### Phase 32: prefix_feature_cache堅牢性テスト・compare-prefix smoke test（REQ-128~135）

#### Prefix Feature Cache堅牢性

- REQ-128: load_prefix_feature_datasetは破損したキャッシュファイル（不完全書き込み、不正フォーマット、欠落キー）を検出し、明確なエラーメッセージと共にValueErrorまたはRuntimeErrorを送出しなければならない。学習ループ（_maybe_cache_dataset）は破損キャッシュを検出時に自動的に再ビルドにフォールバックしなければならない 🔵 *prefix_feature_cache.py load_prefix_feature_dataset・train_tg_lora.py _maybe_cache_dataset try/except パターン・design-interview A27「corrupted cache file handling」推奨より*
- REQ-129: prefix_feature_cache_force_rebuild=trueの場合、_maybe_cache_datasetはメモリキャッシュとディスクキャッシュの両方をスキップし、必ずbuild_prefix_feature_datasetを再実行しなければならない。force_rebuild=falseの場合は既存のメモリ/ディスクキャッシュを再利用しなければならない 🔵 *train_tg_lora.py _maybe_cache_dataset force_rebuild分岐・config_schema.py TrainingConfig.prefix_feature_cache_force_rebuild・design-interview A27「force_rebuild flag behavior」推奨より*
- REQ-130: build_prefix_feature_datasetはposition_idsを含むデータセットバッチを正しく処理し、各PrefixFeatureExampleにposition_idsを保存しなければならない。position_idsなしのデータセットとの混合は禁止され、save_prefix_feature_datasetが検証時にValueErrorを送出しなければならない 🔵 *prefix_feature_cache.py build_prefix_feature_dataset position_ids処理・design-interview A27「position_ids build path」推奨より*
- REQ-131: build_prefix_feature_datasetは例外発生時でもmodel.training状態を確実に復元しなければならない。finallyブロックでhook.remove()とmodel.train()（必要な場合）を確実に実行し、モデルがevalモードのまま残らないことを保証しなければならない 🔵 *prefix_feature_cache.py build_prefix_feature_dataset try/finally・design-interview A27「model.training state restoration」推奨より*
- REQ-132: get_prefix_feature_cache_pathはメタデータのSHA-256ハッシュに基づいてキャッシュパスを生成し、ハイパーパラメータ（model_name, seed, max_seq_len, split_layer_idx, lora_r, lora_alpha, lora_dropout, lora_target_modules, trainable_lora_scope, dataset_path/size/mtime）のいずれかが変更された場合に異なるパスを生成しなければならない。これにより古いキャッシュの誤用を防止する 🔵 *prefix_feature_cache.py get_prefix_feature_cache_path・build_prefix_feature_cache_metadata より*
- REQ-133: load_prefix_feature_datasetはキャッシュファイルのformat_versionを検証し、現在の_PREFIX_FEATURE_CACHE_FORMAT_VERSIONと一致しない場合はValueErrorを送出しなければならない。将来のフォーマット変更との互換性を確保する 🔵 *prefix_feature_cache.py load_prefix_feature_dataset format_versionチェックより*
- REQ-134: save_prefix_feature_datasetは空のPrefixFeatureDatasetに対してValueError("Cannot persist an empty PrefixFeatureDataset")を送出し、不完全または空のキャッシュファイルがディスクに書き込まれるのを防止しなければならない 🔵 *prefix_feature_cache.py save_prefix_feature_dataset 空チェックより*

#### compare-prefix smoke test

- REQ-135: Makefileのcompare-prefix-coldwarmターゲットに対するsmoke testを提供し、（a）cold runがexit code 0で完了しキャッシュがディスクに作成されること、（b）warm runがexit code 0で完了し既存キャッシュを再利用（cache_hit）することを自動検証しなければならない。テストはGPT-2 tinyモデル等の軽量モデルで実行し、CI環境で再現可能でなければならない 🔵 *Makefile compare-prefix-coldwarm target・design-interview A27「compare-prefix-coldwarm targetのsmoke実行CI step」推奨・AI_HUB_MAKE_RUN_FEEDBACK「compare-prefix-cold/warm/coldwarm targetsが追加されたが、実際にこれらをsmoke実行してexit code 0とcache hit/miss logを確認するテストかCI stepを追加する」より*

### Phase 32a: AsyncCacheBuilder・非同期キャッシュビルド設定（REQ-136~138）

#### AsyncCacheBuilder（2-GPU非同期キャッシュビルド）

- REQ-136: システムはAsyncCacheBuilderクラスにより、バックグラウンドGPU（cuda:1等）上で別スレッドのdaemon threadを使用してPrefix Feature Cacheを非同期にビルドしなければならない。ビルド中も学習ループをブロックせず、raw datasetで学習を即座に開始しなければならない。ビルド完了後はDataLoaderをキャッシュ版に差し替えなければならない。ビルド失敗時は警告ログを出力し、raw datasetで学習を継続しなければならない 🔵 *src/training/async_cache_builder.py AsyncCacheBuilder・src/training/train_tg_lora.py 統合実装・eceddf3/a316624コミットより*
- REQ-137: AsyncCacheBuilderはthread-safeなlock機構でビルド結果（AsyncCacheBuildResult）と完了状態（completed/failed）を管理し、start()でdaemon threadを開始し、poll()で非ブロッキングに完了確認し、get_result(label)でラベル別のビルド結果を取得し、join(timeout)でスレッド終了を待機しなければならない 🔵 *async_cache_builder.py threading.Lock・start()/poll()/get_result()/join() 実装より*
- REQ-138: TrainingConfigのprefix_feature_cache_async（bool）とprefix_feature_cache_async_device（str）により非同期キャッシュビルドを制御しなければならない。async=trueの場合はexperimental=trueとasync_deviceの指定が必須であり、欠落時はPydantic ValidationErrorで拒否しなければならない 🔵 *config_schema.py TrainingConfig prefix_feature_cache_async/async_device validators・43a329aコミットより*

### Phase 33: 堅牢化・高速化・統合テストギャップ解消（REQ-139~143）

#### AsyncCacheBuilder統合テスト

- REQ-139: システムはAsyncCacheBuilderのフルライフサイクル（build → wait → load into trainer）をCPU上のモックモデルで検証する統合テストを備えなければならない。テストは（a）CPU上でのキャッシュビルド完了確認、（b）ビルド結果のDataLoaderへの反映確認、（c）ビルド失敗時のraw dataset継続確認、（d）thread-safeなpoll/get_result APIの動作確認をカバーしなければならない。現在のテストはモックベースのユニットテストのみであり、実際のモデル読み込み・キャッシュビルド・DataLoader差し替えのE2E検証が欠落している 🔵 *AI_HUB_MAKE_RUN_FEEDBACK「Add an integration test exercising the full async cache lifecycle (build on mock GPU → wait → load into trainer)」・async_cache_builder.py 実装・test_async_cache_builder.py モックテスト8件より*

#### diff_lora高速化

- REQ-140: diff_loraはscale==0.0の場合にゼロテンソルを直接返し（乗算を回避）、scale==1.0の場合に単純減算のみを実行し（乗算を回避）しなければならない。これにより一般的なスケール値での不要な乗算演算を最適化する 🔵 *lora_state.py diff_lora fast paths・7a643a9コミットより*

#### cosine_similarity直交ベクトル警告

- REQ-141: cosine_similarityは分母が1e-12以下かつ分子ノルムが非ゼロの場合（直交ベクトル）、warnings.warnで警告を発しなければならない。stacklevel=2で呼び出し元の行番号を正しく表示しなければならない 🔵 *metrics.py cosine_similarity warnings・7a643a9コミットより*

#### ActivationCache decoder層ロギング

- REQ-142: _get_decoder_layersはdecoder層を発見した際にlogger.debugでパスをログ出力し、全候補パスの探索に失敗した場合は候補数を含むエラーメッセージを送出しなければならない 🔵 *activation_cache.py _get_decoder_layers debug log/enhanced error・7a643a9コミットより*

#### 非同期キャッシュビルド設定サーフェス

- REQ-143: configs/smoke_async_prefix.yamlは非同期キャッシュビルド検証用の設定サーフェスとして提供され、prefix_feature_cache_async=true、prefix_feature_cache_async_device="cuda:1"、force_top_layers_only=true、enable_random_walk=falseで決定論的テストを可能にしなければならない 🔵 *configs/smoke_async_prefix.yaml・182de29コミットより*

### Phase 34: In-place tensor ops・data_ptr保存検証・velocity opsベンチマーク（REQ-144~148）

#### In-place EMA更新

- REQ-144: Velocity.updateのEMA更新は既存キーに対してin-place演算（mul_/add_）を使用し、テンソルのdata_ptrを保存しなければならない。新規キーに対してはclone()で新しいテンソルを割り当てなければならない。これによりメモリアロケーションオーバーヘッドを削減する 🔵 *velocity.py mul_(beta).add_(delta[k], alpha=(1.0-beta))・851041eコミット・test_velocity.py TestVelocityDataPtrPreservation 5テストより*

#### In-place cap_update

- REQ-145: cap_updateはupdate_norm > max_normの場合にin-place mul_でスケーリングし、data_ptrを保存しなければならない。update_norm <= max_normの場合はテンソルを変更せずそのまま返さなければならない。非有限入力の場合は新しいゼロテンソルを返さなければならない（REQ-063参照） 🔵 *extrapolator.py update.mul_(max_norm/update_norm)・851041eコミット・test_extrapolator.py TestCapUpdateDataPtrPreservation 4テストより*

#### data_ptr保存検証テスト

- REQ-146: システムはvelocity.updateとcap_updateのin-place操作がテンソルのdata_ptrを保存することを検証するテストを備えなければならない。テストは（a）EMA更新後の既存キーのdata_ptr不変、（b）新規キーのdata_ptrが既存と異なる、（c）混在時の既存キーdata_ptr保存、（d）複数キー同時更新時のdata_ptr保存、（e）新規キーによる既存テンソルの置き換え検出、（f）cap_update capping時のdata_ptr保存、（g）cap_update非有限時の新規テンソル返却をカバーしなければならない 🔵 *test_velocity.py TestVelocityDataPtrPreservation 5テスト・test_extrapolator.py TestCapUpdateDataPtrPreservation 4テスト・c9928b6コミット(TASK-0079)より*

#### Velocity opsマイクロベンチマーク

- REQ-147: システムはscripts/benchmark_velocity_ops.pyにより、velocity EMA updateとcap_updateのマイクロベンチマークを提供しなければならない。1000反復（--quick時は10反復）で実行時間・メモリ使用量をJSON出力しなければならない。出力はvelocity_ema（time_ms, per_iter_ms, mem_delta_kb, iterations）とcap_update（time_ms, per_iter_ms, mem_delta_kb, nocap_time_ms, nocap_per_iter_ms, iterations, tensor_shape）の両セクションを含まなければならない 🔵 *scripts/benchmark_velocity_ops.py・c51fd5bコミット(TASK-0080)・test_benchmark_velocity_ops.py 9テストより*

#### Makefileベンチマーク統合

- REQ-148: Makefileはbench-velocity-opsターゲットを提供し、scripts/benchmark_velocity_ops.pyを--quickモードで実行してJSON結果を標準出力に出力しなければならない。bench-optimizer、bench-prefix-cacheと同じパターンで定義されなければならない 🔵 *Makefile既存bench-*パターン・AI_HUB_MAKE_RUN_FEEDBACK「Add a Makefile target (bench-velocity-ops) wiring the new script」より*

### Phase 35: bench-velocity-ops CI gate・回帰自動検出（REQ-149）

#### CI回帰検出ゲート

- REQ-149: Makefileはbench-velocity-ops-ciターゲットを提供し、scripts/benchmark_velocity_ops.pyを--quick --baseline baselines/velocity_ops.json --threshold 20で実行し、性能回帰を自動検出しなければならない。回帰検出時はexit code 1でCI失敗としなければならない。baselines/velocity_ops.jsonはリポジトリにチェックインされ、make bench-velocity-ops --quick --save-baseline baselines/velocity_ops.jsonで更新されなければならない。design-interview.md A31で🔴（未設計）として指摘されていたCI回帰閾値の設計ギャップを閉じる 🔵 *benchmark_velocity_ops.py --baseline/--threshold実装済み・design-interview.md A31「CI回帰閾値未実装」指摘・AI_HUB_MAKE_RUN_FEEDBACK「Wire bench-velocity-ops --baseline into CI」より*

#### Phase 34境界値

- EDGE-168: Velocity.updateのEMA更新後、既存キーのテンソルdata_ptrが更新前と同一でなければならない。新規キーのテンソルは既存のどのテンソルとも異なるdata_ptrを持たなければならない 🔵 *test_velocity.py test_ema_update_preserves_data_ptr/test_new_key_gets_different_data_ptr・c9928b6コミットより*
- EDGE-169: cap_updateでcappingが適用された場合（update_norm > max_norm）、返却テンソルのdata_ptrが入力テンソルと同一でなければならない。cappingが不要な場合もdata_ptrが保存されなければならない 🔵 *test_extrapolator.py test_capping_preserves_data_ptr/test_no_capping_preserves_data_ptr・c9928b6コミットより*
- EDGE-170: cap_updateに非有限テンソルが入力された場合、返却テンソルは入力とは異なるdata_ptrを持つ新しいゼロテンソルでなければならない 🔵 *test_extrapolator.py test_non_finite_returns_new_tensor・c9928b6コミットより*
- EDGE-171: benchmark_velocity_ops.pyの--quick出力は有効なJSONであり、トップレベルに"velocity_ema"と"cap_update"キーを含み、iterations=10でなければならない 🔵 *test_benchmark_velocity_ops.py test_quick_json_output/test_json_output_has_required_fields・c51fd5bコミットより*

#### Phase 35: CI gate境界値

- EDGE-172: bench-velocity-ops-ciは（a）ベースラインファイルが存在しない場合にexit code 2で失敗し、（b）全メトリクスが閾値内の場合にexit code 0で成功し、（c）1つでもメトリクスが閾値超過の場合にexit code 1でCI失敗しなければならない。threshold=20の場合、current > baseline * 1.2 で回帰判定される 🔵 *benchmark_velocity_ops.py exit code 0/1/2実装・test_benchmark_velocity_ops.py TestBaselineRegressionDetection より*

#### Phase 37: Velocity加速度・入力検証・数値安定性境界値

- EDGE-173: Velocity.magnitude_acceleration()はmagnitude_historyが3件未満の場合は0.0を返さなければならない 🔵 *velocity.py magnitude_acceleration() n<3 guard・TASK-0090・test_velocity.py より*
- EDGE-174: cap_update()の非有限値警告ログはNaN要素数とInf要素数を正確に報告しなければならない 🔵 *extrapolator.py cap_update() n_nan/n_inf logging・TASK-0090・test_extrapolator.py より*
- EDGE-175: DeltaTracker.compute_and_record()にbeforeにのみ存在するキーがある場合、ValueErrorに"missing in before"が含まれなければならない。afterにのみ存在するキーがある場合、"missing in after"が含まれなければならない 🔵 *delta_tracker.py compute_and_record() error message・TASK-0091・test_validation_hardening_0091.py より*
- EDGE-176: RollbackManager(max_history=0)はValueErrorを送出しなければならない。RollbackManager(max_history=-1)もValueErrorを送出しなければならない 🔵 *rollback_manager.py max_history validation・TASK-0091・test_validation_hardening_0091.py より*
- EDGE-177: snapshot_lora_delta()に空のbase辞書を渡した場合、ValueError("base snapshot must not be empty")を送出しなければならない 🔵 *lora_state.py snapshot_lora_delta() empty base check・df57154コミット・test_validation_hardening_0091.py より*
- EDGE-178: RandomWalkController.propose()でalpha_log_sigmaが極端に大きい値の場合でもOverflowErrorが発生してはならない。math.expの引数が700を超えないことを保証しなければならない 🔵 *random_walk_controller.py propose() min(log+noise, 700) clamping・580680dコミット・test_random_walk_controller.py より*

#### Phase 38: 加速度適応パラメータ境界値

- EDGE-179: accel_instability_lr_decay=0.0はPydantic ValidationErrorで拒否されなければならない。accel_instability_lr_decay=1.0も拒否されなければならない。有効範囲は開区間(0.0, 1.0)である 🔵 *config_schema.py Field(gt=0.0, lt=1.0)・test_config_schema.py TestAccelParamConfig・1bc6345コミットより*
- EDGE-180: accel_convergence_lr_boost=1.0はPydantic ValidationErrorで拒否されなければならない。有効範囲はx>1.0である 🔵 *config_schema.py Field(gt=1.0)・test_config_schema.py TestAccelParamConfig・1bc6345コミットより*
- EDGE-181: RandomWalkControllerにaccel_instability_lr_decay=0.5を渡した場合、正の加速度検出時にlr×0.5の減衰が適用されなければならない 🔵 *test_random_walk_controller.py test_custom_instability_decay_applied・1bc6345コミットより*
- EDGE-182: RandomWalkControllerにaccel_convergence_lr_boost=1.5を渡した場合、負の加速度検出時にlr×1.5の増加が適用されなければならない 🔵 *test_random_walk_controller.py test_custom_convergence_boost_applied・1bc6345コミットより*

### Phase 36: LR探索統合・propose→training loop配線（REQ-150~152）

#### LR探索パラメータとpropose()統合

- REQ-150: TGLoRAParamsスキーマはlr_explore_prob（float、デフォルト0.3、0.0≤x<1.0）とlr_log_sigma（float、デフォルト0.1、>0.0）をLR探索パラメータとして受け付けなければならない。lr_explore_probはpropose()でlog-normalランダムウォークによるlr探索をトリガーする確率であり、lr_log_sigmaは対数正規分布の標準偏差である 🔵 *config_schema.py lr_explore_prob/lr_log_sigma フィールド・random_walk_controller.py propose() log-normal walk・b712540コミットより*

- REQ-151: train_tg_lora.pyはcontroller.propose()の戻り値Proposalのlrフィールドをcontroller.state.lrに反映しなければならない。これによりlog-normal探索で生成されたlrが次サイクルのpilot trainingで使用される。反映後、reward/penalizeが探索済みlrをさらに調整する。lr_explore_prob=0の場合はpropose()がstate.lrをそのまま返すため、反映は冪等になる 🔵 *train_tg_lora.py controller.state.lr = proposal.lr・本コミットより*

- REQ-152: lr_explore_prob > 0の場合、train_tg_lora.pyは複数サイクルのpropose→accept/rejectを通じてlrが決定論的boost/decay単独では説明できない変動を示すことを統合テストで検証しなければならない。具体的には（a）lr_explore_probとlr_log_sigmaがconfigからcontrollerに伝播されること、（b）propose()で生成されたlrがstateに反映されること、（c）複数サイクル後のlrが純粋なboost/decay計算値と一致しないことを確認しなければならない 🔵 *test_training_integration.py TestLrExplorationIntegration 3テスト・本コミットより*

### Phase 37: Velocity加速度・入力検証強化・数値安定性（REQ-153~159）

#### Velocity magnitude acceleration

- REQ-153: システムはVelocityのmagnitude履歴に対する二階微分（magnitude_acceleration）を計算しなければならない。正の加速度は速度が加速的に増大している（潜在的な不安定性）ことを示し、負の加速度は増大が減速している（収束傾向）ことを示す。データが3件未満の場合は0.0を返さなければならない 🔵 *velocity.py magnitude_acceleration() 実装・TASK-0090・test_velocity.py より*

#### cap_update非有限値ロギング強化

- REQ-154: cap_update()は非有限更新テンソルを検出した際、NaN要素数とInf要素数を含む警告ログを出力しなければならない。ゼロ化された更新の腐敗原因をデバッグ可能にするためである 🔵 *extrapolator.py cap_update() warning logging・TASK-0090・test_extrapolator.py より*

#### DeltaTracker key-mismatch検証

- REQ-155: DeltaTracker.compute_and_record()はafter/before辞書のキーが一致しない場合にValueErrorを送出しなければならない。エラーメッセージは欠落キー（beforeに欠落、afterに欠落）を明示的に列挙しなければならない 🔵 *delta_tracker.py compute_and_record() key validation・TASK-0091・test_validation_hardening_0091.py より*

#### RollbackManager max_historyガード

- REQ-156: RollbackManagerのコンストラクタはmax_history <= 0の場合にValueErrorを送出しなければならない。不正な履歴サイズ設定によるランタイムエラーを防止するためである 🔵 *rollback_manager.py __init__ max_history validation・TASK-0091・test_validation_hardening_0091.py より*

#### snapshot_lora_delta空ベース検証

- REQ-157: snapshot_lora_delta()はbase辞書が空の場合にValueErrorを送出しなければならない。空の参照状態に対する差分計算を防止するためである 🔵 *lora_state.py snapshot_lora_delta() empty base validation・df57154コミット・test_validation_hardening_0091.py より*

#### propose() OverflowError防止

- REQ-158: RandomWalkController.propose()のlog-normal探索でmath.exp()の引数を700にクランプし、OverflowErrorを防止しなければならない。alpha_log_sigmaとlr_log_sigmaのガウスノイズが極端な値を生成した場合でも安全に動作しなければならない 🔵 *random_walk_controller.py propose() exp clamping・580680dコミット・test_random_walk_controller.py より*

#### _compute_stats autograd漏れ防止

- REQ-159: DeltaTracker._compute_stats()は@torch.no_grad()デコレータを適用し、サイクル統計計算時のautogradグラフ構築を防止しなければならない。不要なautogradノード生成によるメモリリークと性能劣化を防止するためである 🔵 *delta_tracker.py _compute_stats() @torch.no_grad()・580680dコミット・test_delta_tracker.py より*

### Phase 38: 加速度適応パラメータの設定サーフェス（REQ-160~161）

#### 加速度適応パラメータのコンフィグ露出

- REQ-160: TGLoRAParamsスキーマはaccel_instability_lr_decay（float、デフォルト0.7、0.0<x<1.0）とaccel_convergence_lr_boost（float、デフォルト1.1、x>1.0）を加速度適応パラメータとして受け付けなければならない。accel_instability_lr_decayはadapt_to_acceleration()で正の加速度検出時のlr減衰率を制御し、accel_convergence_lr_boostは負の加速度検出時のlr増加率を制御する。これらはYAML設定からチューニング可能でなければならない 🔵 *config_schema.py accel_instability_lr_decay/accel_convergence_lr_boost フィールド・random_walk_controller.py インスタンス属性化・1bc6345コミットより*
- REQ-161: RandomWalkControllerはコンストラクタでaccel_instability_lr_decayとaccel_convergence_lr_boostを受け取り、adapt_to_acceleration()でハードコードされたクラス定数の代わりにインスタンス属性を使用しなければならない。None渡し時はデフォルト値（0.7/1.1）を使用しなければならない。train_tg_lora.pyはYAML設定値をcontrollerに配線しなければならない 🔵 *random_walk_controller.py コンストラクタ・adapt_to_acceleration()・train_tg_lora.py 配線・1bc6345コミットより*

### Phase 39: --resume障害回復再開・加速度適応観測性（REQ-162~166）

#### 障害回復からの再開（--resume）

- REQ-162: RandomWalkControllerはrestore_state(state: ControllerState)メソッドを提供し、保存済みControllerStateを採用しなければならない。この際、コントローラのconfig（candidates, bounds, tolerances, exploration probs）は保持し、state値（K, N, alpha, beta, lr, counts）のみを置き換えなければならない。restore_state()実行後、last_accel_actionは0にリセットしなければならない 🔵 *random_walk_controller.py restore_state()・9f195f0コミット・test_fault_recovery.py TestRestoreStateIntegration より*

- REQ-163: train_tg_lora()はresume_path引数を受け付け、指定されたtraining_state.ptファイルから学習状態を復元しなければならない。復元対象はcontroller.restore_state()によるコントローラ状態、velocity、delta_tracker、cycle_state、およびcycle_offsetである。復元ログにはパス・サイクル番号・受理率を含めなければならない 🔵 *train_tg_lora.py resume_path引数・load_training_state復元・9f195f0コミット・test_fault_recovery.py より*

- REQ-164: CLIエントリポイントは--resume引数を提供し、training_state.ptへのパスを受け付けなければならない。パスが指定された場合、train_tg_lora(cfg, resume_path=path)として呼び出さなければならない。未指定の場合は通常の新規学習を開始しなければならない 🔵 *train_tg_lora.py main() --resume引数・9f195f0コミット・test_fault_recovery.py より*

#### 加速度適応の観測性

- REQ-165: RandomWalkControllerはlast_accel_action属性を提供し、adapt_to_acceleration()の実行結果を追跡しなければならない。正の加速度検出時は1（不安定）、負の加速度検出時は-1（収束）、閾値内の場合は0（無行動）を設定しなければならない。enable_random_walk=falseの場合は常に0を維持しなければならない。summary()出力にlast_accel_actionを含めなければならない 🔵 *random_walk_controller.py last_accel_action属性・b2eb409コミット・test_random_walk_controller.py 6テストより*

- REQ-166: train_tg_loraのMLflowサイクルメトリクスはmagnitude_acceleration（velocity.magnitude_acceleration()の値）とaccel_action（controller.last_accel_actionの値）を含まなければならない。これにより加速度適応の実行状態がMLflowダッシュボードで観測可能になる 🔵 *train_tg_lora.py MLflow cycle metrics magnitude_acceleration/accel_action・b2eb409コミットより*

#### Phase 39境界値

- EDGE-183: restore_state()実行後、controller.state.K/N/alpha/beta/lrが保存済みControllerStateの値と完全に一致しなければならない。controllerのcandidates/bounds/tolerancesはコンストラクタ時の値を保持しなければならない 🔵 *test_fault_recovery.py test_restore_state_from_saved_checkpoint・9f195f0コミットより*
- EDGE-184: resume_pathが指定されcycle < cycle_offsetのサイクルは実行をスキップし、cycle_offset以降のサイクルから学習を再開しなければならない 🔵 *train_tg_lora.py cycle < cycle_offset skip・9f195f0コミットより*
- EDGE-185: last_accel_actionはadapt_to_acceleration()の呼び出しごとに更新され、直前の呼び出し結果を反映しなければならない。連続呼び出し時は最新の値のみが保持されなければならない 🔵 *test_random_walk_controller.py test_last_accel_action_zero_accel（1→0遷移）・b2eb409コミットより*
- EDGE-186: adapt_to_acceleration()がacceleration=0.0で呼ばれた場合、last_accel_actionは0（無行動）に設定されなければならない 🔵 *test_random_walk_controller.py test_last_accel_action_zero_accel・b2eb409コミットより*

### Phase 40: --resume E2E統合テスト（REQ-167）

#### E2E resume フロー検証

- REQ-167: システムは--resumeフローのE2E検証を提供しなければならない。TrainingState保存→中断→resume_pathによる再開→loss継続性の完全なサイクルを検証する。cycle_offset未満のサイクルは実行をスキップし、cycle_offset以降から学習を再開しなければならない。resume後のvelocity state方向は保存時と一致しなければならない 🔵 *test_resume_e2e.py TestResumeE2E 3テスト・TASK-0090完了・d3d77b9コミットより*

#### Phase 40境界値

- EDGE-187: resume後のlossは復元された状態から連続的に推移し、保存直前のloss値と矛盾しない値で再開しなければならない 🔵 *test_resume_e2e.py test_full_resume_flow_loss_continuity より*
- EDGE-188: cycle_offset=3で保存されたTrainingStateをresumeした場合、cycle 0~2は完全にスキップされ、cycle 3から実行が開始されなければならない 🔵 *test_resume_e2e.py test_cycle_skipping_on_resume より*
- EDGE-189: resume後のvelocity.state辞書の各テンソルは保存時と方向（sign）が一致しなければならない 🔵 *test_resume_e2e.py test_resume_preserves_velocity_direction より*

### Phase 41: TruthfulQA分析・accel param実験（REQ-168~172）

#### ベンチマーク分析スクリプト

- REQ-168: analyze_benchmark.pyはbaseline/TG-LoRAのベンチマーク評価結果（JSON）を読み込み、各メトリクス（accuracy, perplexity等）の差分を計算・報告しなければならない。欠損メトリクスが存在する場合はエラーを発生させず、利用可能なメトリクスのみで差分計算を行わなければならない 🔵 *scripts/analyze_benchmark.py・TASK-0091 spec・537c0a9コミットより*

#### accel param 感度検証

- REQ-169: accel param sensitivityはaccel_instability_lr_decay（0.3, 0.5, 0.7, 0.9）とaccel_convergence_lr_boost（1.1, 1.5, 2.0）の値域に対する学習率変化の感度を検証しなければならない。adapt_to_acceleration()の呼び出しにおいて、accel_instability_lr_decayの値に応じてlr減衰率が線形に変化し、accel_convergence_lr_boostの値に応じてlr回復率が線形に変化することを確認する 🔵 *random_walk_controller.py adapt_to_acceleration()・TASK-0091 spec・537c0a9コミットより*

#### 実験config検証

- REQ-170: 実験config群（accel_conservative, accel_aggressive, accel_balanced, accel_no_accel）は全てTGLoRAConfig Pydantic検証を通過し、モデル・データ・LoRA設定が全config間で一致（公正比較の前提）し、experiment_nameが一意でなければならない。accel_no_accel configのaccel paramsは実質的に無効化（near-identity: decay≈0.99, boost≈1.01）されていなければならない 🔵 *config_schema.py Pydantic検証・test_accel_experiment_configs.py・TASK-0092 spec・537c0a9コミットより*

#### パラメータスイープ実行

- REQ-171: run_accel_sweep.shは4つのaccel実験config（conservative, aggressive, balanced, no_accel）を順次実行し、各実験の結果をreports/accel_sweep/に集約しなければならない。実行前にconfig存在確認を行い、全実験完了後に比較レポートを生成しなければならない。個別実験の失敗は全体を中断せず、エラーを記録して継続しなければならない 🔵 *scripts/run_accel_sweep.sh・TASK-0092 spec・537c0a9コミットより*

- REQ-172: summarize_sweep.pyはスイープ実行結果のrun_metrics.jsonlを読み込み、各実験の受理率・学習統計を計算し、validation loss順でソートしたサマリーを出力しなければならない。controller state情報を含む詳細レポートを生成しなければならない 🔵 *scripts/summarize_sweep.py・TASK-0092完了後の運用インフラ・537c0a9コミットより*

#### Phase 41境界値

- EDGE-190: accel_no_accel configのaccel_instability_lr_decay=0.99とaccel_convergence_lr_boost=1.01は、adapt_to_acceleration()呼び出し時に実質的にlrを変更しない（±1%以内）ことを保証しなければならない 🔵 *configs/9b_tg_lora_accel_no_accel.yaml near-identity設定・test_accel_experiment_configs.py より*
- EDGE-191: 4つのaccel実験config間でmodel_name, dataset, lora_rank, lora_alpha, K_initial, N_initial, alpha_initial, beta_initialが完全に一致し、accel paramsのみが異なることを保証しなければならない 🔵 *test_accel_experiment_configs.py 公正比較テスト・537c0a9コミットより*

### Phase 42: コンストラクタ入力検証ギャップ・テスト非決定性排除（REQ-173~178）

#### 未検証コンストラクタの入力検証

- REQ-173: DeltaTrackerのコンストラクタはmax_history > 0の場合のみを受け付け、0または負の値に対してValueErrorを送出しなければならない。RollbackManager（REQ-156）と同等の検証パターンを適用しなければならない 🔵 *delta_tracker.py __init__ パラメータ検証不在・random_walk_controller.py 検証パターン（REQ-074）より*
- REQ-174: Velocityのコンストラクタはmax_history > 0の場合のみを受け付け、0または負の値に対してValueErrorを送出しなければならない。RollbackManager（REQ-156）と同等の検証パターンを適用しなければならない 🔵 *velocity.py __init__ パラメータ検証不在・random_walk_controller.py 検証パターン（REQ-074）より*
- REQ-175: OptimizerLifecycleManagerのコンストラクタは（a）lr > 0、（b）weight_decay >= 0の検証を行い、不正な値に対してValueErrorを送出しなければならない。学習率0以下や負のweight_decayはPyTorch optimizerの不正動作を引き起こすため、学習開始前に検出しなければならない 🔵 *optimizer_lifecycle.py __init__ パラメータ検証不在・config_schema.py TrainingConfig.learning_rate Field(gt=0.0)パターンより*
- REQ-176: PrefixFeatureDatasetのコンストラクタは空のexamplesリストに対してValueErrorを送出しなければならない。MappedPrefixFeatureDatasetのコンストラクタは（a）split_layer_idx >= 0、（b）テンソル形状の互換性（all_hidden_states, all_attention_mask等のバッチ次元一致）を検証し、不正な入力に対してValueErrorを送出しなければならない 🔵 *prefix_feature_cache.py PrefixFeatureDataset/MappedPrefixFeatureDataset __init__ 検証不在・InfiniteBatchIterator空データセット検証（REQ-078）パターンより*
- REQ-177: AsyncCacheBuilderのコンストラクタは（a）configがNoneでないこと、（b）device文字列が有効（"cuda:N"または"cpu"）であること、（c）split_layerが0以上の整数であることを検証し、不正な値に対してValueErrorを送出しなければならない 🔵 *async_cache_builder.py __init__ パラメータ検証不在・RandomWalkController検証パターン（REQ-074）より*

#### テスト非決定性の排除

- REQ-178: RandomWalkControllerを使用する全テストは、探索確率パラメータ（k_explore_prob, n_explore_prob, beta_explore_prob, strategy_explore_prob, lr_explore_prob）を明示的に指定しなければならない。テスト対象の探索確率以外は0.0に設定し、テストの非決定性を排除しなければならない。test_restore_state_propose_uses_restored_lrのflaky fix（f5fe40fコミット）と同じパターンを全テストに適用しなければならない 🔵 *test_random_walk_controller.py, test_training_integration.py, test_fault_recovery.py, test_resume_e2e.py, test_extrapolation_safety_integration.py におけるデフォルトexplore_prob使用・AI_HUB_MAKE_RUN_FEEDBACK「scan the full test suite for other tests using default lr_explore_prob or other randomized defaults」より*

#### Phase 42境界値

- EDGE-192: DeltaTracker(max_history=0)はValueErrorを送出しなければならない。DeltaTracker(max_history=-1)もValueErrorを送出しなければならない 🔵 *delta_tracker.py 入力検証追加後の境界値テスト・RollbackManagerパターン（EDGE-176）と同等*
- EDGE-193: Velocity(max_history=0)はValueErrorを送出しなければならない。Velocity(max_history=-1)もValueErrorを送出しなければならない 🔵 *velocity.py 入力検証追加後の境界値テスト・RollbackManagerパターン（EDGE-176）と同等*
- EDGE-194: OptimizerLifecycleManager(lr=0.0)はValueErrorを送出しなければならない。OptimizerLifecycleManager(lr=-0.001)もValueErrorを送出しなければならない 🔵 *optimizer_lifecycle.py 入力検証追加後の境界値テストより*
- EDGE-195: OptimizerLifecycleManager(weight_decay=-0.01)はValueErrorを送出しなければならない。weight_decay=0.0は有効として受け付けなければならない 🔵 *optimizer_lifecycle.py 入力検証追加後の境界値テストより*
- EDGE-196: PrefixFeatureDataset(examples=[])はValueErrorを送出しなければならない 🔵 *prefix_feature_cache.py 入力検証追加後の境界値テストより*
- EDGE-197: k_explore_prob=0.0を明示的に渡したテストでは、propose()のK変更回数が0であることを検証しなければならない。n_explore_prob, beta_explore_prob, strategy_explore_probも同様に、0.0設定時に対応パラメータが変更されないことを検証しなければならない 🔵 *test_random_walk_controller.py test_zero/full_*_explore_prob パターン・AI_HUB_MAKE_RUN_FEEDBACK「The flaky-test-fix pattern is high-leverage」より*

#### Phase 53: Runtime Prefix Offload境界値

- EDGE-198: offload_prefix_runtime_to_cpuのsplit_layer_idx=0はValueErrorを送出しなければならない 🔵 *prefix_runtime_offload.py split_layer_idx < 1 check・test_prefix_runtime_offload.py より*
- EDGE-199: offload_prefix_runtime_to_cpuのsplit_layer_idxがdecoder層数を超過する場合、ValueErrorを送出しなければならない 🔵 *prefix_runtime_offload.py split_layer_idx > len(decoder_layers) check・test_prefix_runtime_offload.py より*
- EDGE-200: offload_prefix_runtime_to_cpuは同じモジュールが重複登録される場合（embeddingがdecoder層と同一参照）、重複を除外してオフロードしなければならない 🔵 *prefix_runtime_offload.py seen_modules setによる重複排除より*
- EDGE-201: prefix_feature_cache_offload_prefix_to_cpu=trueでprefix_feature_cache_experimental=falseの設定はPydantic ValidationErrorで拒否されなければならない 🔵 *config_schema.py prefix_runtime_offload_valid validator・test_config_schema.py より*

### Phase 50: Stage 2 マルチシード複製・Paper Gate評価自動化（REQ-179~184）

#### Paper Gate評価スクリプト

- REQ-179: システムはscripts/evaluate_paper_gates.pyにより、run_paper_memory_suite.shが生成するaggregate_summary.jsonを読み込み、paper_experiment_plan.mdで定義されたGate G0–G4のpass/failを自動判定しなければならない。各Gateの判定結果（pass/fail、詳細、根拠数値）を含むレポートを標準出力に出力し、JSON形式のレポートファイル（-o）も出力可能でなければならない。exit codeは0（全Gate pass）、1（少なくとも1つのGateがfail）、2（入力エラー）でなければならない 🔵 *paper_experiment_plan.md Gate G0–G4定義・run_paper_memory_suite.sh aggregate_summary.json構造・TASK-0105完了より*

- REQ-180: Gate G0 (Hygiene) は（a）seedsリストが空でないこと、（b）per_seedエントリ数がseeds数と一致すること、（c）aggregateにwarm_tg_loss_red_per_wall_minute/warm_baseline_loss_red_per_wall_minute/warm_tg_best_valid_loss/warm_baseline_best_valid_lossの全てが存在しmeanがNoneでないことを検証しなければならない 🔵 *paper_experiment_plan.md G0定義・run_paper_memory_suite.sh出力構造より*

- REQ-181: Gate G1 (Replicated Internal Efficiency) は（a）全seedでTGのloss_red_per_wall_minuteがbaselineを上回ること、（b）全seedでTGのbackward_passesがbaselineより少ないこと、（c）aggregate meanでTG/BL ratioが指定閾値（デフォルト2.0x）以上であること、（d）TGのbest_valid_lossの相対悪化が指定閾値（デフォルト1%）未満であることの4条件を全て満たさなければpassとならない。各条件は個別にチェック結果を出力しなければならない 🔵 *paper_experiment_plan.md G1定義より*

- REQ-182: Gate G2 (Memory Frontier Separation) は（a）aggregate meanでTG peak memory削減率が指定閾値（デフォルト20%）以上であること、（b）runtime offload freed MB > 0であることを検証し、frontier separation（baseline OOM vs TG completion）はinformationalとして別スイープが必要である旨を出力しなければならない 🔵 *paper_experiment_plan.md G2定義より*

- REQ-183: Gate G3 (External Quality Retention) とGate G4 (Causal Attribution) は、それぞれ外部評価結果とablation実験結果が必要であるため、aggregate_summary.jsonからは評価不可である旨をinformationalとして出力しなければならない 🔵 *paper_experiment_plan.md G3/G4定義より*

#### Makefile Gate評価ターゲット

- REQ-184: Makefileはpaper-memory-evaluate-gatesターゲットを提供し、scripts/evaluate_paper_gates.pyをGATE_SUMMARY引数で指定されたaggregate_summary.jsonに対して実行しなければならない。GATE_OUTPUT、GATE_SKIP、G1_LOSS_RED_RATIO、G1_QUALITY_TOLERANCE、G2_MEMORY_IMPROVEMENTの各パラメータをカスタマイズ可能でなければならない 🔵 *Makefile既存paper-memory-*パターン・TASK-0105完了より*

### Phase 51: Paper Pipeline Stage 3-5自動化（REQ-185~191）

#### Stage 3 Frontier Sweep自動化

- REQ-185: システムはscripts/run_frontier_sweep.shにより、複数のMAX_SEQ_LEN値（デフォルト1536,2048,3072）を受け取り、各値でmake paper-memoryを逐次実行しfrontier separationを自動検出しなければならない。baseline OOM/CUDA failureとTG completionのペアをfrontier separationとして報告しなければならない 🔵 *paper_experiment_plan.md Stage 3定義・run_paper_memory_suite.sh既存パターンより*
- REQ-186: Makefileはpaper-memory-frontier-sweepターゲットを提供し、scripts/run_frontier_sweep.shを実行しなければならない。SEQSパラメータでMAX_SEQ_LENリストをカスタマイズ可能でなければならない 🔵 *Makefile既存paper-memory-*パターンより*

#### 外部品質評価パイプライン

- REQ-187: システムはscripts/run_paper_external_eval.pyにより、paper-memory suite出力からbest modelパスを特定し、lm-evaluation-harnessで外部評価（TruthfulQA MC2, ARC Easy, HellaSwag、オプションGSM8K）を実行しなければならない。TG vs baselineの外部評価結果を比較し、external_eval_results.jsonに集約しなければならない 🔵 *paper_experiment_plan.md G3定義・既存run_eval.shパターンより*
- REQ-188: Gate G3判定は（a）aggregate mean relative drop < 1%、（b）単一task relative drop < 3%の両条件を満たさなければpassとならない。frontier separationがある場合は同一frontierとnearest feasible baselineの両方を評価しなければならない 🔵 *paper_experiment_plan.md G3 pass条件より*

#### 因果分析評価ロジック拡張

- REQ-189: evaluate_paper_gates.pyの_check_g4()はinformationalから実際の判定ロジックに拡張し、cold vs warm summary比較（warm speedup全seed正）、train-cache on vs off比較（onのmemory/frontier効果がoffより強い）を自動判定しなければならない 🔵 *paper_experiment_plan.md G4定義・evaluate_paper_gates.py既存実装より*

#### Stage 2実行前Smoke検証

- REQ-190: run_paper_memory_suite.shはDRY_RUN環境変数によるdry-run検証モードを提供しなければならない。dry-run時はconfig存在確認、seed展開、出力パス生成のみを実行し、実際のtrainingは実行してはならない 🔵 *run_paper_memory_suite.sh既存構造・Makefile dry-runパターンより*
- REQ-191: run_paper_memory_suite.shはdry-run時にconfig内のprefix_feature_cache_train設定がtrueであることを検証しなければならない 🔵 *paper_experiment_plan.md Main Blocking Reason（prefix_feature_cache_train: true必須）より*

### Phase 52: 論文結果統合（REQ-192）

- REQ-192: システムはscripts/consolidate_paper_results.pyにより、全gate評価結果（aggregate_summary.json, frontier_report.json, external_eval_results.json, gate_report.json）を読み込み、Claim Ladder判定（C0: G1 pass, C1: G1+G3 pass, C2: G1+G2+G3 pass with frontier separation）を自動出力し、LaTeX/Markdown形式のメインテーブルを生成しなければならない 🔵 *paper_experiment_plan.md Claim Ladder定義・既存JSON構造より*

### Phase 53: Runtime Prefix Offload・補助スクリプト要件（REQ-193~197）

#### Runtime Prefix Offload（GPU→CPU オフロード）

- REQ-193: システムはoffload_prefix_runtime_to_cpu関数により、指定スプリットレイヤー以前の全decoder層（およびオプションでembedding層）のパラメータをCPUにオフロードし、GPU VRAMを解放しなければならない。オフロード後にtorch.cuda.empty_cache()を呼び出し、CUDAメモリプールを解放しなければならない。オフロード結果としてオフロード済みモジュール数、パラメータ数、embedding含むフラグ、split_layer_idxを辞書で返さなければならない 🔵 *src/tg_lora/prefix_runtime_offload.py offload_prefix_runtime_to_cpu()・train_tg_lora.py 統合実装より*
- REQ-194: offload_prefix_runtime_to_cpuはsplit_layer_idx < 1またはsplit_layer_idx > len(decoder_layers)の場合にValueErrorを送出し、不正なレイヤーインデックスによる実行時エラーを防止しなければならない 🔵 *prefix_runtime_offload.py split_layer_idx検証より*
- REQ-195: TrainingConfigはprefix_feature_cache_offload_prefix_to_cpuフラグを提供し、prefix_feature_cache_experimental=trueの場合のみ有効にできなければならない。offload_prefix_to_cpu=trueでexperimental=falseの場合はPydantic ValidationErrorで拒否しなければならない 🔵 *config_schema.py TrainingConfig.prefix_runtime_offload_valid validator・train_tg_lora.py offload呼び出しガードより*

#### 補助スクリプト要件

- REQ-196: システムはscripts/precompute_prefix_cache_parallel.pyにより、複数GPU（複数のCUDA device）でprefix feature cacheをオフライン事前計算する並列バッチジョブを提供しなければならない 🔵 *scripts/precompute_prefix_cache_parallel.py 実装済み435行・paper_experiment_plan.md Stage 3のfrontierスイープでの使用見込みより*
- REQ-197: システムはscripts/benchmark_prefix_cache.pyにより、cold/warm両パスでのprefix feature cache性能ベンチマークを提供し、GPU peak memory・wall-clock時間・loss reduction per wall-minuteをJSON出力しなければならない 🔵 *scripts/benchmark_prefix_cache.py 既存実装231行・既存compare_runs.pyパターンより*

### Phase 54: Frontier Sweep パイプライン強化・G2.3 自動評価（REQ-198~204）

#### G2.3 Frontier Separation 自動評価

- REQ-198: evaluate_paper_gates.pyの_check_g2()は--frontier-report引数でfrontier_report.jsonパスを受け取り、G2.3 frontier separationをinformational判定から実際のpass/fail判定に切り替えなければならない。frontier_separation_detected=trueの場合にpassとし、frontier_boundaryのseq_lenを報告しなければならない。frontier_report.jsonが指定されない場合、またはJSON decodeに失敗した場合はfailとしなければならない 🔵 *evaluate_paper_gates.py _check_g2() --frontier-report・84dcf4eコミットより*

#### 構造化メタデータパイプライン

- REQ-199: run_frontier_sweep.shは各MAX_SEQ_LENの実行後にrun_metadata.jsonを各runディレクトリに書き出さなければならない。JSONには（a）make_exit: makeコマンドの終了コード（int）、（b）summary_exists: aggregate_summary.jsonの存在（bool）、（c）oom_in_log: ログ内OOM検知（bool）の3フィールドを含めなければならない。この構造化JSONがfrontier_report.pyの一次データソースとなり、暗黙的なファイルシステムヒューリスティックを置き換える 🔵 *run_frontier_sweep.sh run_metadata.json書き出し・1e53759コミットより*

#### メモリデルタメトリクス

- REQ-200: frontier_report.pyは各runエントリにmemory_delta_mb（baseline_peak_mb - tg_peak_mb）とmemory_savings_pct（delta / baseline * 100）を含めなければならない。aggregateレベルでavg_memory_savings_pct（全完了runの加重平均）を計算し、frontier_report.jsonに含めなければならない。generated_atタイムスタンプ（UTC ISO形式）も含め、再現性を確保しなければならない 🔵 *frontier_report.py build_frontier_report() memory_delta_mb/memory_savings_pct/avg_memory_savings_pct/generated_at・7da731aコミットより*

#### OOM検知・ステータス分類

- REQ-201: frontier_report.pyは複数ソースからのOOM検知を統合しなければならない。（a）終了コード137はOOMと分類、（b）CUDA out of memory / CUDA error / out of memory / Killedパターンを正規表現でログから検知、（c）終了コードとsummary存在判定の組み合わせでcompleted/oom/failedを決定しなければならない。終了コード0かつsummary存在でcompleted、137でoom、OOMパターン検知でoom、終了コード非ゼロかつsummary不在でfailedとしなければならない 🔵 *frontier_report.py detect_oom_from_log()/determine_status()・4530f8cコミットより*

#### メタデータ読み込みの堅牢性

- REQ-202: frontier_report.pyは_read_run_meta()でrun_metadata.jsonを一次ソースとして読み込まなければならない。JSON decode失敗時は_read_legacy_files()でmake_exit_codeファイルとaggregate_summary.json存在確認にフォールバックしなければならない。run_metadata.json不在時も同様にレガシーフォールバックを実行し、後方互換性を維持しなければならない 🔵 *frontier_report.py _read_run_meta()/_read_legacy_files()・1e53759コミットより*

#### ログ分割によるコンポーネント別OOM帰属

- REQ-203: frontier_report.pyは_split_oom_log()でログを行ごとに分割し、OOMパターンを含む行をbaseline/TGに帰属させなければならない。"baseline"を含む行はbaselineに、"tg"を含む行はTGに、どちらも含まない行は両方に帰属（保守的判定）しなければならない。これによりmake paper-memoryの単一終了コードからbaselineとTGの個別ステータスを導出する 🔵 *frontier_report.py _split_oom_log()・4530f8cコミットより*

#### Frontier Sweep テストカバレッジ

- REQ-204: テストスイートはfrontier_report.pyの全主要機能（OOM検知、ステータス分類、frontier boundary検出、memory delta計算、メタデータ読み込み、run_metadata.jsonパイプライン統合）をカバーしなければならない。テストは（a）各OOMパターンの検知、（b）全ステータス分類パターン（completed/oom/failed）、（c）frontier検出と非検出、（d）memory savings計算（ゼロ除算含む）、（e）run_metadata.json読み込みとレガシーフォールバックを含まなければならない 🔵 *test_frontier_report.py 675行・14+テストクラス・84dcf4e/1e53759/7da731a/4530f8cコミットより*

### Phase 55: 運用スクリプト・ユーティリティモジュール要件（REQ-205~217）

#### ユーティリティモジュール

- REQ-205: システムはsrc/utils/io.pyによりorjsonベースの高速JSON/JSONL I/O（save_json, load_json, save_jsonl, load_jsonl）を提供しなければならない 🔵 *src/utils/io.py 既存実装・pyproject.toml orjson>=3.9依存より*
- REQ-206: システムはsrc/utils/memory.pyによりVRAM使用量（vram_usage_mb）とパラメータ数（count_parameters）のユーティリティを提供しなければならない 🔵 *src/utils/memory.py 既存実装・train_tg_lora.pyでの使用より*
- REQ-207: システムはsrc/utils/run_query.pyによりRunMetrics JSONLログのクエリAPI（parse_jsonl, get_footer, get_cycle_history, get_best_loss, list_runs）を提供しなければならない 🔵 *src/utils/run_query.py 既存実装・TASK-0060で追加より*
- REQ-208: システムはsrc/utils/logging.pyによりRichHandler ベースのロギング設定（setup_logging, get_logger）とディレクトリ確保（ensure_dir）を提供しなければならない 🔵 *src/utils/logging.py 既存実装より*
- REQ-209: システムはsrc/utils/checkpoint.pyによりモデルチェックポイント保存（save_checkpoint）とTrainingState直列化/復元（save_training_state, load_training_state）を提供しなければならない。NaN/Inf値のサニタイズ（_sanitize_tensors）を含まなければならない 🔵 *src/utils/checkpoint.py 既存実装・train_tg_lora.py resume_path連携（REQ-163）より*

#### 運用スクリプト（Orchestration）

- REQ-210: システムはscripts/run_sweep.shによりハイパーパラメータスイープ（lr, rollback_tolerance, K/Nの9設定組み合わせ）を順次実行し、各設定の結果をreports/に集約しなければならない 🔵 *scripts/run_sweep.sh 既存実装67行・Makefile sweepターゲットより*
- REQ-211: システムはscripts/run_ablation_suite.shによりベースライン vs TG-LoRA変種（paper POC / adaptive K5 / no-convergence）のアブレーションスタディを自動実行しなければならない 🔵 *scripts/run_ablation_suite.sh 既存実装137行・Makefile ablationターゲットより*
- REQ-212: システムはscripts/run_high_lr_comparison.shにより高学習率（通常の10-25倍）でのTG-LoRAロールバック優位性を検証する安定性比較を提供しなければならない 🔵 *scripts/run_high_lr_comparison.sh 既存実装141行・rollback_manager.py検証より*
- REQ-213: システムはscripts/run_kstep_rollback_test.shによりK-step中間ロールバック機構（REQ-118）の高LR+大Kでの動作検証を提供しなければならない 🔵 *scripts/run_kstep_rollback_test.sh 既存実装118行・REQ-118中間ロールバック検証より*

#### 運用スクリプト（Accel Sweep Orchestration）

- REQ-214: システムはscripts/run_accel_sweep_parallel.shにより2 GPU並列でのaccel paramスイープ実行とペアワイズ比較を提供しなければならない 🔵 *scripts/run_accel_sweep_parallel.sh 既存実装129行・2-GPU環境前提より*
- REQ-215: システムはscripts/run_accel_sweep_auto.shによりGPU空き監視→自動スイープ開始のラッパーを提供しなければならない 🔵 *scripts/run_accel_sweep_auto.sh 既存実装58行・GPU polling logicより*
- REQ-216: システムはscripts/generate_sweep_dashboard.pyによりaccel paramスイープ結果の自己完結型HTMLダッシュボード（サマリテーブル、ペアワイズ比較、次アクション推奨）を生成しなければならない 🔵 *scripts/generate_sweep_dashboard.py 既存実装・analyze_accel_sweep.py ranking.json消費より*
- REQ-217: システムはscripts/compare_paper_memory_modes.pyによりreuse vs one-shot paper memory aggregateの比較レポート（メモリメトリクス相対デルタ）を提供しなければならない 🔵 *scripts/compare_paper_memory_modes.py 既存実装300行・Makefile compare-prefixターゲットより*

### Phase 56: モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析（REQ-218~231）

#### モデル構造検査ツール

- REQ-218: システムはscripts/inspect_model.pyによりHuggingFaceモデルのLoRA互換ターゲットモジュールを自動発見する機能を提供しなければならない。config.jsonのみ（重みなし）または重み付きでモデル構造を読み込み、全Linear層を名前パターンで列挙し、target_modulesとして推奨するモジュール一覧を出力しなければならない。--model引数による直接指定と--config引数によるYAML設定経由の指定をサポートしなければならない 🔵 *scripts/inspect_model.py 既存実装・README.md Quick Start inspect記載・Makefile inspect/inspect-configターゲットより*

- REQ-219: Makefileはinspectターゲット（モデル名直接指定）とinspect-configターゲット（YAML設定経由）を提供し、それぞれscripts/inspect_model.pyを対応モードで呼び出さなければならない 🔵 *Makefile inspect/inspect-configターゲット既存実装より*

#### 比較ダッシュボード・可視化

- REQ-220: システムはscripts/compare_runs.pyのdashboardサブコマンドにより、指定ディレクトリ内の全実験ランを横断比較するマルチランダッシュボードを提供しなければならない。gather_runs()で全run_metrics.jsonlを自動発見し、find_best_run()でベストパフォーマンスを自動選出し、build_comparison_table()でソート済み比較テーブルを生成し、render_dashboard()でrichライブラリによるコンソールダッシュボード表示を行わなければならない。--format jsonオプションでJSON出力もサポートしなければならない 🔵 *scripts/compare_runs.py dashboardサブコマンド・gather_runs/find_best_run/build_comparison_table/render_dashboard実装より*

- REQ-221: compare_runs.pyはacceptance rate推移（plot_acceptance_rate）、reduction rate推移（plot_reduction_rate）、velocity magnitude推移（plot_velocity_magnitude）、layer score分布（plot_layer_scores）、ハイパーパラメータ探索軌跡（plot_hyperparams）の5種類の可視化プロット関数を提供し、比較レポートに含めなければならない 🔵 *scripts/compare_runs.py 5つのplot_*関数既存実装・REQ-037拡張としてより*

- REQ-222: compare_runs.pyはgenerate_markdown_report()により、ベースライン/TG-LoRA比較のMarkdown形式レポート（サイクル別サマリー、効率メトリクス、受理率推移、velocity分析）を生成しなければならない 🔵 *scripts/compare_runs.py generate_markdown_report()既存実装より*

- REQ-223: システムはscripts/compare_runs.pyのMLflow連携により、比較レポートとプロット画像をMLflowアーティファクトとして自動ロギングする機能を提供しなければならない。log_reports_to_mlflow()で実行しなければならない 🔵 *scripts/compare_runs.py log_reports_to_mlflow()既存実装より*

#### ワンショットPrefix Feature Cache

- REQ-224: システムはprefix_feature_cache_mode="one_shot"によるSSDバッキングのワンショットキャッシュモードをサポートしなければならない。ワンショットモードはPrefixFeatureDatasetをdisk-backedとして構築し、オンデマンドで個別サンプルをディスクから読み込む。reuseモード（デフォルト）はキャッシュ構築後に全データをメモリ保持する。TrainingConfigのprefix_feature_cache_modeフィールド（PrefixFeatureCacheMode型、Literal["reuse", "one_shot"]、デフォルト"reuse"）で制御しなければならない 🔵 *src/tg_lora/prefix_feature_cache.py MappedPrefixFeatureDataset disk-backed mode・config_schema.py PrefixFeatureCacheMode定義・configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yamlより*

- REQ-225: configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yamlはワンショットキャッシュ検証用の設定サーフェスとして提供され、prefix_feature_cache_mode: one_shot、prefix_feature_cache_dir: .cache/prefix_feature_cache_one_shot_poc、enable_random_walk=false、force_top_layers_only=trueで決定論的テストを可能にしなければならない 🔵 *configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml既存実装より*

#### コスト・効果分析

- REQ-226: システムはscripts/analyze_prefix_cache_break_even.pyにより、prefix feature cacheのコールドビルドコストがウォーム実行の節約でいつ回収できるかの損益分岐点分析を提供しなければならない。単一ランサマリーとaggregateサマリーの両形式をサポートし、分析結果（break_even_cycles, amortization_metrics）を出力しなければならない 🔵 *scripts/analyze_prefix_cache_break_even.py 既存実装148行・Makefile analyze-prefix-break-evenターゲットより*

- REQ-227: Makefileはanalyze-prefix-break-evenターゲットを提供し、scripts/analyze_prefix_cache_break_even.pyを実行して損益分岐点分析を行わなければならない 🔵 *Makefile analyze-prefix-break-evenターゲット既存実装より*

#### データパイプライン細粒度ターゲット

- REQ-228: Makefileはdownload-dolly（Dolly 15kのみダウンロード）、download-capybara（Capybaraのみダウンロード）、prepare-data-small（1k train用の小規模データセット準備）、prepare-capybara（Capybaraデータセット準備）の細粒度ターゲットを提供し、個別データセットの独立した取得・準備を可能にしなければならない 🔵 *Makefile download-dolly/download-capybara/prepare-data-small/prepare-capybaraターゲット既存実装より*

#### クリーンアップ・運用ターゲット

- REQ-229: Makefileはclean（生成ファイル削除）、clean-data（ダウンロード・生成データ削除）、clean-runs（全実験ラン削除）のクリーンアップターゲットを提供し、それぞれ対象ディレクトリの確認付きで実行しなければならない 🔵 *Makefile clean/clean-data/clean-runsターゲット既存実装より*

#### 比較実験キャッシュモード

- REQ-230: Makefileはcompare-prefix-cold（キャッシュクリア後の比較実行）、compare-prefix-warm（既存キャッシュ利用の比較実行）、compare-prefix-coldwarm（cold→warm順次実行）のキャッシュモード別比較ターゲットを提供しなければならない。cold実行ではprefix_feature_cache_dirをクリアしてから実行し、warm実行では既存キャッシュを再利用し、coldwarm実行では両方を順次実行してキャッシュヒット効果を検証しなければならない 🔵 *Makefile compare-prefix-cold/warm/coldwarmターゲット既存実装より*

- REQ-231: configs/9b_baseline_suffix_only_last25.yamlはsuffix-onlyモードのベースライン設定サーフェスとして提供され、trainable_lora_scope: last_25_percent、prefix_feature_cache_experimentalなしで標準的なsuffix-only学習を定義しなければならない 🔵 *configs/9b_baseline_suffix_only_last25.yaml既存実装・Makefile compare-prefixターゲットのベースラインとしてより*

### Phase 59: 学習軌跡分析・収束予測・早期停止推奨（REQ-232~236）

#### 軌跡分析コアモジュール

- REQ-232: システムはsrc/tg_lora/trajectory.pyのTrajectoryAnalyzerクラスにより、loss履歴から収束予測・早期停止推奨・異常検知を提供しなければならない。TrajectoryPointで学習サイクルデータを入力し、compute_loss_trend()で線形トレンド、compute_convergence_rate()で収束率、estimate_convergence()で収束推定（ConvergenceEstimate）、early_stop_advice()で早期停止推奨（EarlyStopAdvice）、detect_anomalies()で異常検知をそれぞれ提供しなければならない。from_loss_history()とfrom_dicts()のファクトリメソッドも提供しなければならない 🔵 *新規実装 src/tg_lora/trajectory.py より*

- REQ-233: システムはscripts/analyze_trajectory.pyにより、run_metrics.json・JSONL・直接loss値指定から軌跡分析レポートを生成するCLIツールを提供しなければならない。--from-losses引数でカンマ区切りloss値、--target-lossで目標loss、--patienceで早期停止忍耐値、--windowで解析窓幅、--outputでJSON出力をサポートしなければならない 🔵 *新規実装 scripts/analyze_trajectory.py より*

- REQ-234: TrajectoryAnalyzerは減少loss系列に対し負のloss_trend・正のconvergence_rateを返し、収束済み系列に対しconverged=Trueを返し、停滞系列に対しshould_stop=Trueを返さなければならない 🔵 *test_trajectory.py テストケース TC-227-01~04 より*

- REQ-235: analyze_trajectory.pyはJSON・JSONLファイル入力と--from-losses入力の両方をサポートし、CONINUE/STOP推奨を含むレポートを標準出力し、--outputでJSON形式の構造化レポートをファイル出力しなければならない 🔵 *test_trajectory.py テストケース TC-228-01 より*

- REQ-236: TrajectoryAnalyzerクラスはsrc/tg_lora/__init__.pyの公開API（__all__）に含まれ、パッケージレベルでインポート可能でなければならない 🔵 *src/tg_lora/__init__.py __all__更新より*

### Phase 60: Trajectory-Informed Adaptive Control（REQ-237~240）

#### 軌跡連動適応制御コアモジュール

- REQ-237: システムはsrc/tg_lora/trajectory_controller.pyのTrajectoryControllerクラスにより、TrajectoryAnalyzerの軌跡分析結果に基づいてRandomWalkControllerのパラメータをリアルタイムに適応させなければならない。record_cycle()で学習サイクルデータを入力し、収束検知時はalpha_maxを減衰し、停滞検知時はalpha_maxを増加し、異常検知時はlr_reject_decayを減衰させなければならない。CycleDecisionデータクラスで提案・停止信号・異常検知・適応調整を返さなければならない 🔵 *新規実装 src/tg_lora/trajectory_controller.py より*

- REQ-238: TrajectoryControllerはloss spike時（異常検知時）にanomaly_detected=Trueを返し、lr_reject_decayをanomaly_lr_decay係数で減衰させ、alpha_maxをanomaly_alpha_decay係数で減衰させなければならない。enable_adaptive_lrとenable_adaptive_alphaで各適応を個別に無効化できなければならない 🔵 *trajectory_controller.py _apply_trajectory_insights() より*

- REQ-239: TrajectoryControllerはTrajectoryAnalyzerの早期停止推奨をCycleDecision.should_stopに伝播させなければならない。停滞loss系列ではshould_stop=Trueとstop_reasonを返し、改善loss系列ではshould_stop=Falseを維持しなければならない 🔵 *trajectory_controller.py early_stop連携より*

- REQ-240: TrajectoryControllerはexport_state()/restore_state()で軌跡点・適応履歴・サイクルカウントを直列化・復元できなければならない。summary()で現在状態のコントローラサマリ・累積適応・異常検知・収束状態を返さなければならない。TrajectoryController、CycleDecision、TrajectoryControllerConfigはsrc/tg_lora/__init__.pyの公開API（__all__）に含まれなければならない 🔵 *trajectory_controller.py export_state()/restore_state()/summary()、__init__.py更新より*

### Phase 57: 論文結果エクスポート・ハイパーパラメータ感度分析（REQ-241~244）

#### 論文結果エクスポートツール

- REQ-241: システムはscripts/export_paper_results.pyにより、aggregate_summary.jsonからLaTeX・Markdown・CSV形式の出版可能テーブルを生成しなければならない。load_aggregate()で集約JSONを読み込み、generate_latex_table()でLaTeXテーブル、generate_markdown_table()でMarkdownテーブル、export_csv()でCSV出力をそれぞれ提供しなければならない。--format引数で単一形式（latex/markdown/csv）または全形式（all）を指定でき、allの場合は--output-dir（デフォルトpaper_tables）に全形式を出力しなければならない 🔵 *scripts/export_paper_results.py 既存実装・paper_experiment_plan.md Stage 2-5出力要件より*
- REQ-242: export_paper_results.pyは集約JSONにper_seedまたはaggregateキーが含まれない場合にValueErrorを拒否し、ファイルが存在しない場合にFileNotFoundErrorを送出しなければならない 🔵 *export_paper_results.py load_aggregate() バリデーションより*

#### ハイパーパラメータ感度分析

- REQ-243: システムはscripts/analyze_sensitivity.pyにより、ハイパーパラメータスイープ結果からパラメータと結果メトリクス間のPearson相関行列を計算し感度ランキングを生成しなければならない。load_sweep_results()でスイープ実験を読み込み、compute_correlation_matrix()で相関行列を計算し、rank_sensitivity()でパラメータを平均絶対相関でランク付けし、generate_sensitivity_report()でJSONレポートを出力しなければならない。デフォルト分析パラメータはtg_lora_K, tg_lora_N, tg_lora_alpha, tg_lora_beta, learning_rate, batch_size, lora_r, lora_alphaとし、デフォルトメトリクスはbest_valid_loss, final_train_loss, total_wall_secondsとしなければならない 🔵 *scripts/analyze_sensitivity.py 既存実装・src.utils.run_query依存より*
- REQ-244: analyze_sensitivity.pyはNone値ペアをフィルタリングし、指定がない場合は利用可能なパラメータ・メトリクスを自動検出しなければならない 🔵 *analyze_sensitivity.py Noneハンドリング・自動検出ロジックより*

### Phase 58: 学習サイクルヘルスモニタ・実験構成マトリクス比較（REQ-245~250）

#### 学習サイクルヘルスモニタ

- REQ-245: システムはsrc/tg_lora/cycle_monitor.pyのCycleMonitorクラスにより、学習サイクルの健全性監視を提供しなければならない。update()でサイクルデータを受け取りHealthReportを返し、detect_divergence()で発散検知（NaN/Inf検出、loss spike比率検知）、detect_stagnation()で停滞検知（patienceサイクル以上の改善なし）、recommend_intervention()で介入推奨（reduce_lr, rollback, increase_K）をそれぞれ提供しなければならない 🔵 *src/tg_lora/cycle_monitor.py 既存実装・TrajectoryAnalyzerとTrainingAdvisorの基盤より*
- REQ-246: CycleMonitorはconstructor引数のpatience（≥1）とspike_threshold（>0）を検証し、不正値の場合ValueErrorを送出しなければならない 🔵 *cycle_monitor.py コンストラクタ検証より*
- REQ-247: DivergenceReportはNaN/Inf値をcritical severity、loss比率がspike_thresholdを超える場合をhigh severityとして報告しなければならない。loss選択ロジックはvalid_lossを優先し、無い場合はtrain_lossを使用しなければならない 🔵 *cycle_monitor.py detect_divergence()・loss選択ロジックより*
- REQ-248: CycleMonitorのbest_loss追跡はNaN/Inf値を除外し、health_summary()で現在状態の完全な辞書を返さなければならない 🔵 *cycle_monitor.py best_loss管理・health_summary()より*

#### 実験構成マトリクス比較

- REQ-249: システムはscripts/compare_experiment_configs.pyにより、runsディレクトリ配下の実験を自動発見し構成パラメータと結果メトリクスの比較マトリクスを生成しなければならない。discover_experiments()で実験を検出しExperimentSummaryを作成し、build_comparison_matrix()でComparisonMatrixを構築し、rank_experiments()でメトリクス別にランク付けし、format_as_markdown()とformat_as_json()で出力フォーマットを提供しなければならない 🔵 *scripts/compare_experiment_configs.py 既存実装・src.utils.run_query依存より*
- REQ-250: compare_experiment_configs.pyは損失・時間メトリクスの最適化方向（lower_is_better）を自動判定し、欠損JSONLファイルやパースエラーを安全に処理し、空カラムを出力テーブルから除外しなければならない 🔵 *compare_experiment_configs.py 自動判定・エラーハンドリング・カラムフィルタリングより*

### Phase 61: Training Advisor モジュール・CLI（REQ-251~258）

#### Training Advisor コアモジュール

- REQ-251: システムはsrc/tg_lora/training_advisor.pyのTrainingAdvisorクラスにより、CycleMonitorとTrajectoryAnalyzerの統合監視結果から優先順位付きアクションを生成する学習アドバイザを提供しなければならない。evaluate()でサイクルデータを受け取りAdvisoryReportを返し、NaN/Inf検出時はrollback + reduce_lrのcriticalアクションを、loss spike検知時はreduce_lrのhighアクションを、停滞検知時はincrease_kのhighアクションを、異常検知時はreduce_lr + adjust_alphaのmediumアクションを、降下トレンド検知時はincrease_lrのlowアクションをそれぞれ生成しなければならない 🔵 *src/tg_lora/training_advisor.py 既存実装・CycleMonitor + TrajectoryAnalyzer統合より*
- REQ-252: AdvisoryActionはaction_type（reduce_lr/increase_lr/stop_training/save_checkpoint/increase_k/decrease_k/adjust_alpha/rollback/resume/no_action）、priority（critical/high/medium/low）、reason、suggested_value、confidence（0.0~1.0）を含み、confidenceが[0,1]範囲外の場合ValueErrorを送出しなければならない 🔵 *training_advisor.py AdvisoryAction dataclass・バリデーションより*
- REQ-253: AdvisoryReportはoverall_health（healthy/warning/critical）、優先順位付きactionsリスト、summary、cycle_health（HealthReport）、trajectory_summary、UTC ISO timestampを含み、top_action()で最高優先度アクションを返さなければならない 🔵 *training_advisor.py AdvisoryReport dataclass・top_action()より*
- REQ-254: AdvisorConfigはstagnation_patience（デフォルト5）、spike_threshold（デフォルト2.0）、trajectory_window（デフォルト5）、convergence_threshold（デフォルト1e-4）、plateau_lr_factor、anomaly_lr_factor、convergence_alpha_factor、plateau_alpha_factor、early_stop_min_cycles（デフォルト10）、save_checkpoint_on_bestを設定パラメータとして提供しなければならない 🔵 *training_advisor.py AdvisorConfig dataclassより*
- REQ-255: TrainingAdvisor.evaluate()はtrain_lossを必須とし、valid_loss/grad_norm/velocity_magnitude/loss_pilot/loss_after/acceptance_rateをオプションとして受け取らなければならない。best_loss追跡はvalid_lossを優先し、NaN/Inf値はgracefulに処理しなければならない 🔵 *training_advisor.py evaluate()・NaN/Infハンドリング・best_loss管理より*
- REQ-256: generate_advice_from_history()関数はサイクル履歴レコードリストを受け取り、最終サイクルのAdvisoryReportを返さなければならない。非有限lossを安全に処理しなければならない 🔵 *training_advisor.py generate_advice_from_history()より*

#### Training Advisor CLI

- REQ-257: システムはscripts/advise_training.pyにより、run_metrics.jsonlファイルから学習アドバイスレポートを生成するCLIツールを提供しなければならない。--jsonフラグでJSON出力、-oでファイル出力、--patienceで停滞忍耐値、--spike-thresholdでspike閾値、--trajectory-windowで軌跡分析窓幅を指定できなければならない。JSONL読み込みで不正行を警告付きスキップし、type=="cycle_step"または"step"のレコードを抽出し、正規化してgenerate_advice_from_history()に渡さなければならない 🔵 *scripts/advise_training.py 既存実装・training_advisor.py統合より*
- REQ-258: advise_training.pyはファイル未検出・レコードなし・サイクルレコードなしの場合にexit code 1で終了し、training stateがcriticalの場合にexit code 2で終了し、正常の場合にexit code 0で終了しなければならない。--json出力はAdvisoryReportをJSON直列化し、text出力はhealth・summary・cycle health・trajectory・actionsセクションを人間可読形式で表示しなければならない 🔵 *advise_training.py exit code・出力フォーマット・エラーハンドリングより*

### Phase 57 ギャップ補完: マルチシード統計分析・論文実験運用Makefile（REQ-259~264）

#### 統計分析モジュール（src/analysis/stats.py）

- REQ-259: システムはsrc/analysis/stats.pyのconfidence_interval()により、標本データの信頼区間（mean, lower_bound, upper_bound）を計算しなければならない。n=1の場合は(mean, mean, mean)を返し、空入力の場合はValueErrorを送出しなければならない。scipy依存なしでt分布に基づく区間推定を提供しなければならない 🔵 *src/analysis/stats.py confidence_interval() 既存実装・export_paper_results.pyで使用より*
- REQ-260: システムはpaired_t_test()によりベースライン群と処置群のペア付きt検定を実行し、(t_statistic, p_value)を返さなければならない。サンプルサイズが一致しない場合・2組未満の場合はValueErrorを送出し、分散ゼロの場合は適切な特殊値を返さなければならない 🔵 *src/analysis/stats.py paired_t_test() 既存実装・evaluate_paper_gates.py Gate統計検定で使用より*
- REQ-261: システムはcohens_d()により二群間のCohen's d効果量を計算しなければならない。正のdはtreatment > baselineを示し、プール標準偏差ゼロの場合は0.0を返し、空入力の場合はValueErrorを送出しなければならない 🔵 *src/analysis/stats.py cohens_d() 既存実装より*
- REQ-262: システムはanalyze_multi_seed()によりマルチシード実験の集約サマリーからメトリクス別統計（n, mean, std, CI）を計算しなければならない。per_seed内の数値メトリクスを自動検出し、非数値・None値を除外し、2件以上でCI・標準偏差を提供しなければならない 🔵 *src/analysis/stats.py analyze_multi_seed() 既存実装・export_paper_results.py/evaluate_paper_gates.pyで使用より*

#### 論文実験運用Makefileターゲット

- REQ-263: Makefileはpaper-memory-dry-run（GPUなし設定バリデーション）、paper-memory-one-shot（ワンショットSSDバッキングモードのマルチシードスイート）、paper-memory-compare-modes（reuse/one-shot集約サマリー比較）、paper-memory-all-modes（reuse→one-shot→比較の全自動実行）、paper-memory-external-eval（Gate G3外部品質評価）の論文実験運用ターゲットを提供しなければならない 🔵 *Makefile paper-memory-* ターゲット既存実装・paper_experiment_plan.md Stage 2-5運用フローより*
- REQ-264: Makefileはprecompute-prefix-cache（オフラインマルチGPU prefix-cache事前計算）とbench-velocity-ops-save-baseline（CI比較用velocity opsベースラインJSON再生成）の運用ターゲットを提供しなければならない 🔵 *Makefile precompute-prefix-cache/bench-velocity-ops-save-baseline ターゲット既存実装より*

### Phase 62: Prior-based Subspace Amplification・レジーム検知・LAWAベースライン・レイヤー分析（REQ-265~284）

#### PSAPrior コアモジュール

- REQ-265: システムはsrc/tg_lora/psa.pyのPSAPriorクラスにより、安定したper-tensor PC1方向に沿った勾配増幅（Prior-based Subspace Amplification）を提供しなければならない。record_delta()でdeltaをリングバッファに記録し、extract_priors()でpower iterationによるPC1方向を抽出し、amplify_gradients()で勾配をin-placeに増幅しなければならない。増幅公式はG_amplified = G + gamma * <G, v_PSA> * v_PSAでなければならない 🔵 *src/tg_lora/psa.py PSAPrior実装・docs/GOAL.md §3 Track01より*
- REQ-266: PSAPriorのextract_priors()はL2正則化（RNA理論、arXiv:1805.09639）を適用し、前回priorからの乖離をl2_reg係数でペナルティしなければならない。これによりpriorの急激な方向転換を防止し、安定したsubspace追跡を保証しなければならない 🔵 *src/tg_lora/psa.py extract_priors() l2_reg実装・docs/GOAL.md §2.4より*
- REQ-267: PSAPriorのcompute_gain_map()はテンソル名に基づくlayer-type-specific gain scalingを提供しなければならない。out_projは×1.2、v_projは×1.1、MLPは×0.7、その他はデフォルトgain（1.0）を使用しなければならない。GOAL §4 step 2の「out_proj最安定仮説」に基づく設計である 🔵 *src/tg_lora/psa.py compute_gain_map()・docs/GOAL.md §4 Track08より*
- REQ-268: PSAPriorはwarmup_stepsパラメータ（デフォルト4）を提供し、指定ステップ数までは増幅を無効化しなければならない。should_update(step)でupdate_interval間隔でのprior更新タイミングを制御しなければならない 🔵 *src/tg_lora/psa.py warmup_steps/update_interval実装より*
- REQ-269: PSAPriorはreset_priors()でpriorと履歴をクリアできなければならない。レジーム遷移時（RegimeDetectorのconsume_reset_signal()発動時）に呼び出され、新しい学習フェーズでpriorを再構築しなければならない 🔵 *src/tg_lora/psa.py reset_priors()・src/tg_lora/regime.py統合より*

#### PSA 設定・Config

- REQ-270: config_schema.pyはPSAConfigモデル（history_length: int=6, gain: float=0.5, update_interval: int=3, warmup_steps: int=4, l2_reg: float=0.01, regime_reset_enabled: bool=true）を提供し、extra='forbid'で未知フィールドを拒否しなければならない 🔵 *config_schema.py PSAConfig・configs/9b_tg_lora_psa.yamlより*
- REQ-271: configs/9b_tg_lora_psa.yamlはPSA実験用の設定サーフェスとして提供され、enable_random_walk=false、force_top_layers_only=trueで決定論的比較を可能にしなければならない。psa設定ブロック（gain=0.5, history_length=6, update_interval=3, warmup_steps=4, l2_reg=0.01, regime_reset_enabled=true）を含まなければならない 🔵 *configs/9b_tg_lora_psa.yaml既存実装より*

#### RegimeDetector

- REQ-272: システムはsrc/tg_lora/regime.pyのRegimeDetectorクラスにより、学習フェーズ遷移（STABLE→PLATEAU→TRANSITION）の自動検知を提供しなければならない。update(loss)でlossを記録し、velocity z-score統計に基づいて現在のRegime（STABLE/PLATEAU/TRANSITION）を返さなければならない 🔵 *src/tg_lora/regime.py RegimeDetector実装・docs/GOAL.md §4 Track03より*
- REQ-273: RegimeDetectorはconsume_reset_signal()でワンショット消費パターンのリセット信号を提供し、PSAのregime-aware prior resetで使用しなければならない。should_reset_priorsプロパティでピーク（非消費）をサポートしなければならない 🔵 *src/tg_lora/regime.py consume_reset_signal()・src/tg_lora/psa.py統合より*

#### ActivationFingerprintTracker

- REQ-274: システムはsrc/tg_lora/activation_regime.pyのActivationFingerprintTrackerクラスにより、連続ステップ間のcosine similarityに基づく活性化フィンガープリントレジーム分類を提供しなければならない。register_hook()でforward hookを登録し、step()でcosine similarityを計算し、STABLE（cos>0.95）/TRANSITION/CHAOTIC（cos<0.5）のレジームを分類しなければならない。forward-only診断（追加backwardなし）でなければならない 🔵 *src/tg_lora/activation_regime.py実装・docs/GOAL.md §4 Track02より*
- REQ-275: ActivationFingerprintTrackerはregime_inventoryプロパティで各レジームの割合を返し、compute_regime_null_baseline()で時系列シャッフルによるヌルベースラインを計算しなければならない。GOAL §7の「全メトリクスにヌルベースライン必須」鉄則に準拠しなければならない 🔵 *src/tg_lora/activation_regime.py regime_inventory/compute_regime_null_baseline()・docs/GOAL.md §7より*

#### LAWA Averager

- REQ-276: システムはsrc/tg_lora/weight_averaging.pyのLAWAAveragerクラスにより、LAtest-Window Weight Averagingベースラインを提供しなければならない。record()でLoRA重みスナップショットを記録し、average_snapshot()でスライディングウィンドウの算術平均を計算しなければならない。GOAL §3.3でmandatoryとされるベースラインである 🔵 *src/tg_lora/weight_averaging.py LAWAAverager実装・docs/GOAL.md §3.3より*
- REQ-277: LAWAAveragerはevaluate_with_lawa()コンテキストで一時的に平均重みに差し替えて評価し、評価後に元の重みを復元しなければならない。start_cycleパラメータで平均開始サイクルを制御しなければならない 🔵 *src/tg_lora/weight_averaging.py evaluate_with_lawa()実装より*

#### LayerDeltaAnalysis

- REQ-278: システムはsrc/tg_lora/layer_delta_analysis.pyによりper-tensor ΔW分析を提供しなければならない。compute_rank1_dominance()でPC1分散比率、compute_direction_stability()でPC1方向安定性、marchenko_pastur_expected_rank1()でMarchenko-Pasturランダムヌル期待値を計算し、classify_layer_type()でATTENTION_OUT/ATTENTION_V/ATTENTION_OTHER/DELTANET/MLP/UNKNOWNに分類しなければならない 🔵 *src/tg_lora/layer_delta_analysis.py実装・docs/GOAL.md §4 Track08より*
- REQ-279: analyze_tensor_deltas()はper-tensor分析結果を返し、group_by_layer_type()はレイヤータイプ別に集約し、Marchenko-Pasturヌルベースラインに対するz-scoreを含めなければならない。GOAL §7の「ヌルベースライン必須」鉄則に準拠しなければならない 🔵 *src/tg_lora/layer_delta_analysis.py analyze_tensor_deltas/group_by_layer_type()・docs/GOAL.md §7より*

#### PSAアブレーション・スイープスクリプト

- REQ-280: システムはscripts/run_psa_ablation.shにより、PSA vs plain LoRA vs LAWAの同一backward予算比較アブレーションを自動実行しなければならない。3条件（PSA有効、plain LoRA、LAWA only）を順次実行し、結果をreports/psa_ablation/に集約しなければならない 🔵 *scripts/run_psa_ablation.sh既存実装・docs/GOAL.md §4.3より*
- REQ-281: システムはscripts/run_psa_gamma_sweep.shにより、PSA gain（gamma）のスイープ実験（複数gamma値でのPSA性能比較）を自動実行しなければならない。regime reset ON/OFFのアブレーションも含まなければならない 🔵 *scripts/run_psa_gamma_sweep.sh既存実装・docs/GOAL.md §4 Track01感度分析より*
- REQ-282: システムはscripts/summarize_psa_sweep.pyにより、PSAスイープ実験結果を集約し、最適gammaとregime reset効果をレポートしなければならない 🔵 *scripts/summarize_psa_sweep.py既存実装より*

#### 研究方向転換（GOAL.md反映）

- REQ-283: docs/GOAL.mdは、軌跡外挿（velocity/M9-FD）の否定的検証結果と、PSAへの研究方向転換を記録しなければならない。8つの研究トラック（Track01~08）と、「全メトリクスにヌルベースライン必須」「per-tensor分析必須」「レジーム認識制御」の鉄則を含まなければならない 🔵 *docs/GOAL.md §1~§7既存文書・4cdc230/b4d4447コミットより*
- REQ-284: TG-LoRAのメインラインはPSA（Prior-based Subspace Amplification）であり、外挿ベースのアプローチはサーフェスとして残すがメインではないことを要件定義書に反映しなければならない。既存の外挿要件（REQ-002~003, REQ-016）はPSAと共存し、設定で切り替え可能でなければならない 🔵 *docs/GOAL.md §3.1 Main Line vs Auxiliary Line・configs/9b_tg_lora.yaml vs 9b_tg_lora_psa.yamlより*

#### Phase 62 境界値

- EDGE-202: PSAPriorのhistory_lengthが0以下の場合、ValueErrorを送出しなければならない 🔵 *psa.py コンストラクタ検証パターンより*
- EDGE-203: PSAPriorのgainが負の場合、ValueErrorを送出しなければならない 🔵 *psa.py コンストラクタ検証パターンより*
- EDGE-204: RegimeDetectorのmin_history未満のデータ点数ではSTABLEを返さなければならない 🔵 *regime.py min_history=3 guardより*
- EDGE-205: ActivationFingerprintTrackerのregister_hook()が呼ばれていない状態でstep()を呼び出した場合、エラーを発生させずにスキップしなければならない 🔵 *activation_regime.py None checkより*
- EDGE-206: LAWAAveragerのwindow_sizeが0以下の場合、ValueErrorを送出しなければならない 🔵 *weight_averaging.py コンストラクタ検証パターンより*
- EDGE-207: LAWAAveragerはrecord()呼び出し前にaverage_snapshot()を呼び出した場合、空バッファとして空辞書を返さなければならない 🔵 *weight_averaging.py is_ready checkより*
- EDGE-208: compute_rank1_dominance()にゼロ行列が入力された場合、0.0を返さなければならない 🔵 *layer_delta_analysis.py ゼロ除算ガードより*
- EDGE-209: classify_layer_type()は未知のテンソル名パターンに対してLayerType.UNKNOWNを返さなければならない 🔵 *layer_delta_analysis.py UNKNOWN fallbackより*
- EDGE-210: PSAConfigのgain=0.0はPydantic ValidationErrorで拒否されなければならない。gainは正値でなければならない 🔵 *config_schema.py Field(gt=0.0)パターンより*
- EDGE-211: configs/9b_tg_lora_psa.yamlはPydantic extra='forbid'検証を通過しなければならない 🔵 *config_schema.py 全モデル extra='forbid'より*


<!-- spine:references:begin -->
## Spine: external references

- [TASK-0016: Pydantic 設定スキーマによる設定検証](tasks/TASK-0016.md)
- [TASK-0017: CLI エントリポイント GPUモックテスト](tasks/TASK-0017.md)
- [TASK-0018: 学習開始前バリデーションと設定スキーマ統合](tasks/TASK-0018.md)
- [TASK-0081: Phase 34 overview.md更新とテスト数同期](tasks/TASK-0081.md)
- [TASK-0088: テストスイート警告解消と品質向上](tasks/TASK-0088.md)
- [TASK-0126: requirements.md 黄信号要件ステータスの実態反映](tasks/TASK-0126.md)
- [TASK-0127: run_eval_lora.sh 終了時trap handler追加](tasks/TASK-0127.md)
- [TASK-0128: overview.md フェーズ進捗とテスト状況の最新化](tasks/TASK-0128.md)
- [TASK-0129: parse_warnings corrupt-JSONL end-to-end integration tests](tasks/TASK-0129.md)

<!-- spine:references:end -->
