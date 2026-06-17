# TG-LoRA アーキテクチャ設計

**作成日**: 2026-05-21
**関連要件定義**: [requirements.md](requirements.md)
**分析記録**: [design-interview.md](design-interview.md)

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 要件定義書・既存設計文書・既存実装を参考にした確実な設計
- 🟡 **黄信号**: 要件定義書・既存設計文書・既存実装から妥当な推測による設計
- 🔴 **赤信号**: 参照資料にない自動推定による設計

---

## システム概要 🔵

**信頼性**: 🔵 *要件定義書・AGENTS.md・既存実装より*

TG-LoRA (Tangent-Gradient LoRA) は、LoRA学習における勾配速度ベクトルの外挿を用いて学習効率を向上させるPython ML研究プロジェクト。Qwen3.5-9B (4bit QLoRA) をRTX3060 12GBで学習可能にし、velocity追跡→外挿→受理/拒否のサイクルベース学習でbackward pass効率を改善する。

## アーキテクチャパターン 🔵

**信頼性**: 🔵 *既存実装のディレクトリ構成・モジュール設計より*

- **パターン**: モジュラーパイプラインアーキテクチャ（ML研究向け）
- **選択理由**: 各アルゴリズムコンポーネント（velocity, extrapolator, layer_sampler, rollback, random_walk）が独立したモジュールとして実装され、学習ループがこれらをオーケストレーションする構造。研究実験での迅速な反復とモジュールの差し替えが可能。

## コンポーネント構成

### コアアルゴリズム層 (`src/tg_lora/`) 🔵

**信頼性**: 🔵 *既存実装8ファイルの完全読み込みより*

| モジュール | 責務 | 主要クラス/関数 |
|-----------|------|----------------|
| `velocity.py` | 勾配速度のEMA追跡・magnitude history・異常検出・トレンド・加速度 | `Velocity.update()`, `Velocity.cosine_similarity()`, `Velocity.is_magnitude_anomalous()`, `Velocity.magnitude_trend()`, `Velocity.magnitude_acceleration()`（REQ-153: 二階微分で不安定/収束を検出） |
| `activation_cache.py` | レイヤースキップ評価最適化・スプリットレイヤー隠れ状態キャッシュ・Qwen3.5 rotary embedding対応・decoder層検出ロギング | `ActivationCache.eval_and_cache()`, `eval_from_cache()`, `determine_split_layer()`（REQ-110~112, REQ-142: decoder層検出ロギング）, `_get_rotary_emb()`, `_get_layer_types()`（Qwen3.5 hybrid attention: linear→2D mask, full→None） |
| `extrapolator.py` | 速度ベースの重み外挿・更新制限・非有限更新ガード | `apply_extrapolation()`, `cap_update()`（NaN/Inf入力時ゼロ返却: REQ-063） |
| `delta_tracker.py` | pilot前後のLoRA重み差分計算・非有限normガード | `compute_mean_delta()`, `DeltaTracker.compute_and_record()`（NaN/Inf norm履歴除外: REQ-067/068） |
| `cycle_state.py` | サイクルレベルの集計状態追跡・last_valid_loss管理 | `CycleState.record_cycle()`, `record_full_eval()`, `should_stop()`, `from_dict()`（last_valid_loss復元含む: REQ-104） |
| `layer_sampler.py` | レイヤー選択戦略 | `select_active_layers()`, `get_num_layers()` |
| `rollback_manager.py` | LoRA状態の保存・復元・NaN/Infサニタイズ・履歴上限管理 | `RollbackManager.save()`, `rollback()`（REQ-064/065: スナップショットサニタイズ, max_history FIFO） |
| `random_walk_controller.py` | ハイパーパラメータ適応探索（K, N, alpha, beta, lr） | `RandomWalkController.propose()`, `accept()`, `reward()`, `penalize()`, `adapt_to_convergence()`, `adapt_to_acceleration()`, `update_layer_scores()`, `restore_state()`（REQ-162: 障害回復再開用）, `last_accel_action`（REQ-165: 加速度適応観測）（enable_random_walkフラグ: REQ-113, enable_convergence_adaptation: 収束適応の独立制御） |
| `lora_state.py` | LoRAパラメータのスナップショット管理・デルタスナップショット | `snapshot_lora()`, `load_lora_snapshot()`, `snapshot_lora_delta()`, `apply_delta_snapshot()`（REQ-118: 中間ロールバック用増分スナップショット）, `diff_lora()`（REQ-140: scale==0/1 fast paths付き高速化） |
| `metrics.py` | cosine similarity等の計算ユーティリティ・キー不一致安全処理・直交ベクトル警告 | `cosine_sim()`（REQ-066: キー不一致時安全スキップ、REQ-141: 直交ベクトル警告） |
| `prefix_feature_cache.py` | 評価時隠れ状態事前計算・suffix-only trainable mode（REQ-126）・ワンショットSSDバッキングモード（REQ-224） | `PrefixFeatureDataset`, `MappedPrefixFeatureDataset`, `PrefixFeatureExample`, `build_prefix_feature_dataset()`, `collate_prefix_feature_batch()`, `load_prefix_feature_dataset(lazy=)`（one_shot: disk-backed MappedPrefixFeatureDataset, reuse: in-memory PrefixFeatureDataset） |
| `prefix_runtime_offload.py` | 学習開始時のprefix層GPU→CPUオフロード・VRAM解放（REQ-193~195） | `offload_prefix_runtime_to_cpu()`, `_find_optional_module()`, `_get_input_embeddings()` |
| `cycle_monitor.py` | 学習サイクル健全性監視・発散・停滞検知・介入推奨（REQ-245~248） | `CycleMonitor.update()`, `detect_divergence()`, `detect_stagnation()`, `recommend_intervention()`, `health_summary()` |
| `trajectory.py` | 学習軌跡分析・収束予測・早期停止推奨・異常検知（REQ-232~236） | `TrajectoryAnalyzer.compute_loss_trend()`, `compute_convergence_rate()`, `estimate_convergence()`, `early_stop_advice()`, `detect_anomalies()`, `from_loss_history()`, `from_dicts()` |
| `trajectory_controller.py` | 軌跡分析に基づくRandomWalkController適応制御・状態直列化（REQ-237~240） | `TrajectoryController.record_cycle()`, `export_state()`, `restore_state()`, `summary()`, `CycleDecision` dataclass |
| `training_advisor.py` | 統合監視シグナルから優先順位付きアクションを生成する学習アドバイザ（REQ-251~258） | `TrainingAdvisor.evaluate()`, `summary()`, `generate_advice_from_history()`, `AdvisoryAction`/`AdvisoryReport`/`AdvisorConfig` dataclass |
| `psa.py` | Prior-based Subspace Amplification: 安定したper-tensor PC1方向に沿った勾配増幅（REQ-265~269） | `PSAPrior.record_delta()`, `extract_priors()`, `amplify_gradients()`, `compute_gain_map()`, `reset_priors()`, `amplify_gradients_psa()` |
| `regime.py` | 学習フェーズ遷移（STABLE→PLATEAU→TRANSITION）の自動検知（REQ-272~273） | `RegimeDetector.update()`, `consume_reset_signal()`, `Regime` enum |
| `activation_regime.py` | 活性化フィンガープリントによるレジーム分類・ヌルベースライン（REQ-274~275） | `ActivationFingerprintTracker.register_hook()`, `step()`, `regime_inventory`, `compute_regime_null_baseline()` |
| `weight_averaging.py` | LAWA（LAtest-Window Weight Averaging）ベースライン（REQ-276~277） | `LAWAAverager.record()`, `average_snapshot()`, `evaluate_with_lawa()` context manager |
| `layer_delta_analysis.py` | per-tensor ΔW分析・Marchenko-Pasturヌルベースライン（REQ-278~279） | `analyze_tensor_deltas()`, `group_by_layer_type()`, `compute_rank1_dominance()`, `compute_direction_stability()`, `marchenko_pastur_expected_rank1()`, `classify_layer_type()` |

### 状態シリアライズ層 (`src/tg_lora/` + `src/utils/`) 🔵

**信頼性**: 🔵 *cycle_state.py・random_walk_controller.py・checkpoint.py 実装・Phase 27より*

| モジュール | 責務 | 主要クラス/関数 |
|-----------|------|----------------|
| `cycle_state.py` | サイクル状態の集計・早期終了・シリアライズ | `CycleState.summary()`, `CycleState.from_dict()`（REQ-104） |
| `random_walk_controller.py` | ハイパーパラメータ状態のシリアライズ | `ControllerState.summary()`, `ControllerState.from_dict()`（REQ-103） |
| `checkpoint.py` | 学習状態全体の保存・復元 | `TrainingState` dataclass, `save_training_state()`, `load_training_state()`（REQ-105） |

### 学習ループ層 (`src/training/`) 🔵

**信頼性**: 🔵 *train_tg_lora.py・train_baseline_qlora.py・trainer_loop.py・Phase 21 DRYリファクタリングより*

| モジュール | 責務 |
|-----------|------|
| `train_tg_lora.py` | TG-LoRAサイクルベース学習（pilot→外挿→受理/拒否）。純粋関数 `should_run_full_eval()`, `build_training_summary()` を含む |
| `train_baseline_qlora.py` | 標準QLoRA学習（比較基準用） |
| `trainer_loop.py` | 共通学習プリミティブ（forward_backward, optimizer_step, scheduler） |
| `optimizer_lifecycle.py` | サイクル間optimizerライフサイクル管理（`recreate_per_cycle` 再生成 / `reuse_state_reset_experimental` in-place state zero-reset）。`OptimizerLifecycleManager` がpolicyに応じてAdamWの再確保またはstate tensor再利用を切り替える |
| `config_schema.py` | Pydantic設定スキーマ検証（BaselineConfig, TGLoRAConfig） |
| `preflight.py` | 学習開始前バリデーション（データパス存在確認、run_dir書き込み確認） |
| `async_cache_builder.py` | バックグラウンドGPUでのPrefix Feature Cache非同期ビルド（2-GPU構成向け）。daemon thread上で2つ目のモデルコピーをロードし、キャッシュを並列構築。学習開始をブロックせず、キャッシュ完成時にDataLoaderを差し替え | `AsyncCacheBuilder.start()/poll()/get_result()/join()`, `AsyncCacheBuildResult` dataclass |
| `batch_iter.py` | 無限バッチイテレータ（DataLoaderラッパ、デバイスキャスト、空データセット検出）（REQ-078） |
| `loss.py` | 共通損失計算（バッチキー検証付き） |

### モデル管理層 (`src/model/`) 🔵

**信頼性**: 🔵 *load_model.py・lora_utils.py既存実装より*

| モジュール | 責務 |
|-----------|------|
| `load_model.py` | 4bit量子化モデル読込・VRAM判定fp32キャスト |
| `lora_utils.py` | LoRAパラメータ反復・レイヤー別グループ化・trainable_lora_scopeによる学習範囲制御（configure_trainable_lora_scope, set_trainable_lora_layers, get_last_fraction_lora_layer_indices）（REQ-125） |

### データパイプライン層 (`src/data/`) 🔵

**信頼性**: 🔵 *6ファイルの既存実装より*

| モジュール | 責務 |
|-----------|------|
| `build_seed_dataset.py` | JSONL読込→トークナイズ→PyTorch Dataset |
| `schema.py` | ChatML学習データのPydanticスキーマ検証（DataRecord, ValidationSummary） |
| `generate_open_data.py` | オープンソースモデルによる合成データ生成 |
| `filter_dataset.py` | テキスト長・品質・必須フィールドによるフィルタリング |
| `dedup.py` | 完全一致 + 埋め込みベース重複排除（FAISS/numpy） |
| `provenance.py` | データ来歴メタデータ管理 |

### 評価層 (`src/eval/`) 🔵

**信頼性**: 🔵 *3ファイルの既存実装より*

| モジュール | 責務 |
|-----------|------|
| `eval_loss.py` | 平均損失評価（context managerで状態リーク防止） |
| `eval_task.py` | 生成→メトリクス計算のタスク評価 |
| `eval_format.py` | JSONフォーマット準拠性評価 |

### 統計分析層 (`src/analysis/`) 🔵

**信頼性**: 🔵 *src/analysis/stats.py 既存実装・export_paper_results.py/evaluate_paper_gates.py使用より*

| モジュール | 責務 | 主要クラス/関数 |
|-----------|------|----------------|
| `stats.py` | マルチシード実験の統計分析（信頼区間・t検定・効果量） | `confidence_interval()`, `paired_t_test()`, `cohens_d()`, `analyze_multi_seed()`（REQ-259~262, scipy非依存） |

### ユーティリティ層 (`src/utils/`) 🔵

**信頼性**: 🔵 *既存実装・Phase 21 DRYリファクタリング・Phase 27 checkpoint拡張より*

| モジュール | 責務 |
|-----------|------|
| `run_metrics.py` | JSONL形式の構造化メトリクスログ（footer/best perplexity出力） |
| `logging.py` | ログバックエンド設定（JSONL/MLflow） |
| `seed.py` | 乱数シード設定（random, numpy, torch, CUDA） |
| `checkpoint.py` | 共有チェックポイント保存ヘルパー（model + tokenizer save_pretrained）（REQ-079）+ `TrainingState` dataclass（CycleState + ControllerState + Velocity + DeltaTracker統合）による学習状態シリアライズ・デシリアライズ（REQ-105） |
| `io.py` | JSON/JSONLファイルI/Oユーティリティ（orjson使用） |
| `memory.py` | VRAM使用量・パラメータ数ユーティリティ |
| `mlflow_logger.py` | MLflow実験ロガー（graceful degradation、インストール不要時はno-op） |
| `run_query.py` | RunMetrics JSONL履歴クエリAPI（TASK-0060） | `parse_jsonl()`, `query_runs()`, `filter_by_metric()`, `aggregate_runs()` |

### 運用層 (`scripts/`) 🔵

**信頼性**: 🔵 *scripts/diagnose.py・scripts/recover.py実装・Phase 27（TASK-0064）より*

| モジュール | 責務 | 主要クラス/関数 |
|-----------|------|----------------|
| `diagnose.py` | GPU・チェックポイント・設定・ログの自動ヘルスチェック（369行） | `CheckResult` dataclass, `check_gpu()`, `check_checkpoint()`, `check_config()`, `check_logs()`, `run_all_checks()`（REQ-106） |
| `recover.py` | 障害回復自動化（OOM/CUDA/NaN分析・チェックポイントサニタイズ・復旧設定生成）（437行） | `RecoveryResult`, `analyze_fault()`, `sanitize_checkpoint()`, `generate_recovery_config()`, `apply_remediation()`（REQ-107） |
| `benchmark_optimizer_lifecycle.py` | recreate_per_cycle vs reuse_state_reset_experimental 定常状態オーバーヘッド比較（200行） | `main()`, `_measure_cycle()`, `_state_summary()`, `state_tensor_pointers()`（REQ-123） |
| `benchmark_velocity_ops.py` | velocity EMA更新・cap_update in-place操作のマイクロベンチマーク（143行） | `benchmark_velocity_ema()`, `benchmark_cap_update()`, `main()`（REQ-147, 1000回反復・JSON出力） |
| `analyze_benchmark.py` | TruthfulQA等ベンチマーク結果のbaseline/TG-LoRA間差分分析・JSON出力（REQ-168） | `main()`, `load_results()`, `compute_deltas()`（Phase 41） |
| `analyze_accel_sweep.py` | accel paramスイープ結果分析・収束軌跡・受理率トラッキング・スイープ検証（Phase 43-44） | `analyze_sweep()`, `compute_loss_trajectory()`, `_detect_plateau()`, `validate_sweep_results()`, `validate_sweep_configs()`, `generate_summary()`（JSONL二重解析排除: parse済みrecords再利用） |
| `run_accel_sweep.sh` | accel paramグリッドサーチ実行・results集約スクリプト（Phase 41） | bash entry point（REQ-170） |
| `compare_paper_memory_modes.py` | reuse vs one-shot paper-memory aggregate比較・relative delta・Markdownレポート出力（300行） | `_load_summary()`, `build_mode_comparison()`, `_render_markdown()`, `main()`（Phase 47+: aggregate + legacy benchmark 2形式対応） |
| `run_paper_memory_suite.sh` | Paper実験マルチシードスイート実行・aggregate summary生成（175行） | bash entry point, seed毎のcold/warm benchmark, aggregate_summary.json/md生成（Phase 50） |
| `evaluate_paper_gates.py` | paper_experiment_plan.md Gate G0–G4の自動pass/fail判定・JSON+stdoutレポート（Phase 50） | `_load_summary()`, `_check_g0()`~`_check_g4()`, `_mean()`, `main()`（REQ-179~183, exit code 0/1/2） |
| `precompute_prefix_cache_parallel.py` | 複数GPUでのprefix feature cache並列事前計算・ランクシャードデータ・キャッシュマージ（Phase 53） | `main()`, shard/mergeロジック, `--devices auto`（REQ-196, 435行） |
| `benchmark_prefix_cache.py` | cold/warmパスでのprefix cache性能ベンチマーク・GPU peak memory・wall-clock測定・JSON summary（Phase 53） | `main()`, `_run_pass()`, `run_comparison.sh`統合（REQ-197, 231行） |
| `frontier_report.py` | Stage 3 frontier sweep結果のfrontier_report.json生成・OOM検知・ステータス分類・memory delta計算（Phase 54） | `detect_oom_from_log()`, `determine_status()`, `find_frontier_boundary()`, `build_frontier_report()`, `_read_run_meta()`, `_split_oom_log()`（REQ-198~204, 279行） |
| `run_frontier_sweep.sh` | Stage 3 frontier sweep実行・複数MAX_SEQ_LENでpaper-memory逐次実行・run_metadata.json生成（Phase 54） | bash entry point, per-run structured metadata, frontier_report.py呼び出し（REQ-199） |
| `summarize_sweep.py` | TG-LoRAハイパーパラメータスイープ結果の要約・効率メトリクス計算・ランキング出力 | `load_run()`, `_compute_efficiency()`, `main()`（200行） |
| `analyze_prefix_cache_break_even.py` | prefix cache のコールドビルドコストがウォーム実行の節約でいつ回収できるかの損益分岐点分析 | `_extract_from_single_run()`, `_extract_from_aggregate()`, `analyze_break_even()`, `main()`（148行） |
| `generate_sweep_dashboard.py` | ハイパーパラメータスイープ結果の自己完結型HTMLダッシュボード生成 | `load_ranking()`, `generate_html()`, `main()`（219行） |
| `run_sweep.sh` | 9構成のハイパーパラメータグリッドスイープ実行・summarize_sweep.pyで分析 | bash entry point, `SWEEP_GRID` 配列ループ（67行） |
| `run_ablation_suite.sh` | baseline vs TG-LoRA バリアント（paper POC / adaptive K5 / no-convergence）のアブレーションスタディ実行 | bash entry point, `_run_baseline()`, `_run_tg()`（137行） |
| `run_high_lr_comparison.sh` | 高学習率（10-25x）でのbaseline vs TG-LoRA安定性比較・ロールバック優位性検証 | bash entry point, `EXPERIMENTS` 配列ループ（141行） |
| `run_kstep_rollback_test.sh` | K-step中間ロールバック機構の検証・高LRでのpilot発散時回復確認 | bash entry point, `_run_tg()`, `_run_baseline()`（118行） |
| `run_best_config_eval.sh` | スイープ最良構成でのlm-evaluation-harness評価 | bash entry point, ranking.jsonから最良構成抽出（141行） |
| `run_accel_sweep_parallel.sh` | 4 accel構成を2GPUで並列実行・pairwise比較とHTMLダッシュボード生成 | bash entry point, `run_config()`, 2ラウンド×2並列（129行） |
| `run_accel_sweep_auto.sh` | GPU空き監視後にaccel paramスイープ自動起動 | bash entry point, GPU memory ポーリング（58行） |
| `run_remaining_accel_configs.sh` | baseline完了後に残り3 accel構成を逐次実行・フル分析パイプライン | bash entry point, main configループ（123行） |
| `inspect_model.py` | HuggingFaceモデルのLoRA互換ターゲットモジュール自動発見（REQ-218） | `inspect_from_config()`, `inspect_from_yaml()`, `_analyze_model()`（257行） |
| `compare_runs.py` | ベースライン/TG-LoRA比較レポート・マルチランダッシュボード・5種可視化プロット・MLflow連携（REQ-220~223）・構造化parse_warnings収集（REQ-037a） | `gather_runs()`, `find_best_run()`, `build_comparison_table()`, `render_dashboard()`, `plot_acceptance_rate()`, `plot_reduction_rate()`, `plot_velocity_magnitude()`, `plot_layer_scores()`, `plot_hyperparams()`, `generate_markdown_report()`, `log_reports_to_mlflow()`, `format_json()`（860行） |
| `export_paper_results.py` | aggregate_summary.jsonからLaTeX/Markdown/CSV出版テーブル生成（REQ-241~242） | `load_aggregate()`, `generate_latex_table()`, `generate_markdown_table()`, `export_csv()` |
| `analyze_sensitivity.py` | ハイパーパラメータ感度分析・Pearson相関・ランキング（REQ-243~244） | `load_sweep_results()`, `compute_correlation_matrix()`, `rank_sensitivity()`, `generate_sensitivity_report()` |
| `compare_experiment_configs.py` | 実験構成マトリクス自動比較・ランク付け（REQ-249~250） | `discover_experiments()`, `build_comparison_matrix()`, `rank_experiments()`, `format_as_markdown()`, `format_as_json()` |
| `advise_training.py` | run_metrics.jsonlから学習アドバイスレポート生成CLI（REQ-257~258） | `_load_jsonl()`, `_extract_cycle_records()`, `_compute_acceptance_rate()`, `_format_report_text()`, `_report_to_dict()`, `main()` |
| `analyze_trajectory.py` | run_metrics.json/JSONLから軌跡分析レポート生成CLI（REQ-233~235） | `main()`（--from-losses/--target-loss/--patience/--window/--output対応） |
| `run_psa_ablation.sh` | PSA vs plain LoRA vs LAWA同一backward予算比較アブレーション（REQ-280） | bash entry point, 3条件（PSA/plain/LAWA）+ γ/regime_reset/intervalスイープ（304行） |
| `run_psa_gamma_sweep.sh` | PSA gain（γ）スイープ実行・regime reset ON/OFFアブレーション（REQ-281） | bash entry point, 4×2=8構成グリッド（125行） |
| `summarize_psa_sweep.py` | PSAスイープ結果集約・最適γ・regime reset効果レポート（REQ-282） | `main()`, pairwise比較, γ effect分析, next-action推奨（270行） |

## システム構成図

```mermaid
graph TB
    Config[configs/*.yaml<br/>Hydra/OmegaConf]
    Data[data/*.jsonl<br/>ChatML形式]

    Config --> Trainer
    Data --> Trainer

    subgraph Training Layer
        Trainer[Training Loop<br/>train_*.py]
        TLoop[trainer_loop.py<br/>共通プリミティブ]
        Trainer --> TLoop
    end

    subgraph Core Algorithm
        Vel[velocity.py<br/>EMA速度追跡]
        Ext[extrapolator.py<br/>重み外挿]
        DT[delta_tracker.py<br/>差分計算]
        LS[layer_sampler.py<br/>レイヤー選択]
        RM[rollback_manager.py<br/>状態保存・復元]
        RW[random_walk_controller.py<br/>適応探索]
        LS_t[lora_state.py<br/>スナップショット]
        AC[activation_cache.py<br/>レイヤースキップ評価]
        PFC[prefix_feature_cache.py<br/>隠れ状態事前計算]
        ACB[async_cache_builder.py<br/>2-GPU非同期キャッシュビルド]
    end

    subgraph Model
        Model[Qwen3.5-9B<br/>4bit NF4 + LoRA r=16]
        LoRA[LoRA Adapter<br/>all-linear対象]
    end

    subgraph Evaluation
        EvalLoss[eval_loss.py<br/>平均損失]
        EvalTask[eval_task.py<br/>タスク評価]
        EvalFmt[eval_format.py<br/>フォーマット評価]
        LmEval[lm-evaluation-harness<br/>標準ベンチマーク]
    end

    subgraph Output
        Metrics[run_metrics.jsonl<br/>構造化ログ]
        Checkpoints[runs/<exp>/<br/>チェックポイント]
        Reports[reports/<br/>比較レポート・プロット]
    end

    subgraph Shared Utils
        BatchIter[batch_iter.py<br/>無限バッチイテレータ]
        CheckpointHelper[checkpoint.py<br/>チェックポイント保存<br/>+ TrainingState]
        LossFn[loss.py<br/>共通損失計算]
        MLflowLog[mlflow_logger.py<br/>MLflowロガー]
        IO[io.py<br/>JSON/JSONL I/O]
        Mem[memory.py<br/>VRAM/パラメータ計測]
    end

    subgraph Operations
        Diagnose[diagnose.py<br/>GPU/ckpt/config/log<br/>ヘルスチェック]
        Recover[recover.py<br/>OOM/CUDA/NaN<br/>障害回復]
    end

    subgraph Sweep & Analysis
        Summarize[summarize_sweep.py<br/>スイープ結果要約]
        Dashboard[generate_sweep_dashboard.py<br/>HTMLダッシュボード]
        BreakEven[analyze_prefix_cache<br/>_break_even.py<br/>cache損益分岐点]
        RunSweep[run_sweep.sh<br/>グリッドスイープ]
        Ablation[run_ablation_suite.sh<br/>アブレーション]
    end

    subgraph Paper Experiment Pipeline
        RunSuite[run_paper_memory_suite.sh<br/>マルチシードスイート実行]
        CompareModes[compare_paper_memory_modes.py<br/>モード比較レポート]
        EvalGates[evaluate_paper_gates.py<br/>Gate G0-G4評価]
        SuiteOut[aggregate_summary.json<br/>スイート結果]
        FrontierSweep[run_frontier_sweep.sh<br/>Stage 3 frontier sweep]
        FrontierReport[frontier_report.py<br/>OOM検知・メモリ分析]
        FrontierOut[frontier_report.json<br/>フロンティアレポート]
    end

    subgraph Training Advisory Pipeline
        CycleMon[cycle_monitor.py<br/>発散・停滞検知]
        Trajectory[trajectory.py<br/>軌跡分析・収束予測]
        TrajCtrl[trajectory_controller.py<br/>適応制御]
        Advisor[training_advisor.py<br/>統合アドバイザ]
        AdviseCLI[advise_training.py<br/>CLIレポート]
    end

    subgraph Analysis Tools
        Sensitivity[analyze_sensitivity.py<br/>感度分析]
        ExportPaper[export_paper_results.py<br/>出版テーブル生成]
        CmpExpCfg[compare_experiment_configs.py<br/>実験構成比較]
    end

    subgraph PSA Pipeline
        PSA[psa.py<br/>PSAPrior<br/>勾配増幅]
        REGIME[regime.py<br/>RegimeDetector<br/>フェーズ検知]
        ACTREG[activation_regime.py<br/>ActivationFingerprint<br/>活性化レジーム]
        LAWA[weight_averaging.py<br/>LAWAAverager<br/>重み平均]
        LDA[layer_delta_analysis.py<br/>ΔW分析<br/>Marchenko-Pastur]
        PSAConfig[9b_tg_lora_psa.yaml<br/>PSA実験設定]
    end

    subgraph PSA Experiment
        PSAablation[run_psa_ablation.sh<br/>PSAアブレーション]
        PSAgamma[run_psa_gamma_sweep.sh<br/>γスイープ]
        PSAsummary[summarize_psa_sweep.py<br/>スイープ集約]
    end

    Trainer --> Vel & Ext & DT & LS & RM & RW & LS_t & AC & ACB
    Trainer --> BatchIter & CheckpointHelper & LossFn
    TLoop --> Model
    Model --> LoRA
    Trainer --> EvalLoss
    Trainer --> Metrics & Checkpoints
    LmEval --> Reports
    EvalLoss & EvalTask & EvalFmt --> Metrics

    Checkpoints --> Diagnose
    Config --> Diagnose
    Metrics --> Diagnose
    Checkpoints --> Recover
    Config --> Recover
    Recover --> Checkpoints

    Metrics --> Summarize
    Metrics --> Dashboard
    Metrics --> BreakEven
    Config --> RunSweep
    Config --> Ablation

    Config --> RunSuite
    Checkpoints --> RunSuite
    RunSuite --> SuiteOut
    SuiteOut --> EvalGates
    SuiteOut --> CompareModes
    FrontierSweep --> SuiteOut
    FrontierSweep --> FrontierReport
    FrontierReport --> FrontierOut
    FrontierOut --> EvalGates

    Metrics --> CycleMon
    Metrics --> Trajectory
    CycleMon --> Advisor
    Trajectory --> TrajCtrl
    TrajCtrl --> Trainer
    Trajectory --> Advisor
    Metrics --> AdviseCLI
    Advisor --> AdviseCLI

    Metrics --> Sensitivity
    Metrics --> CmpExpCfg
    SuiteOut --> ExportPaper

    PSAConfig --> Trainer
    Trainer --> PSA
    Trainer --> REGIME
    Trainer --> ACTREG
    Trainer --> LAWA
    PSA --> REGIME
    LDA --> PSA
    Metrics --> PSAablation
    Metrics --> PSAgamma
    PSAgamma --> PSAsummary
    PSAablation --> PSAsummary
```

**信頼性**: 🔵 *既存実装の全モジュール関係より*

## ディレクトリ構造 🔵

**信頼性**: 🔵 *既存プロジェクト構造より*

```
./
├── configs/
│   ├── 9b_baseline.yaml      # QLoRA ベースライン設定
│   ├── 9b_tg_lora.yaml       # TG-LoRA 実験設定
│   ├── 9b_tg_lora_adaptive_k5.yaml        # adaptive branch
│   ├── 9b_tg_lora_adaptive_k5_no_conv.yaml # ablation（convergence adaptation off）
│   ├── 9b_tg_lora_optimizer_reuse_experimental.yaml  # optimizer再利用実験
│   ├── 9b_tg_lora_prefix_feature_cache_experimental.yaml  # prefix feature cache実験
│   ├── 9b_tg_lora_accel_conservative.yaml  # accel保守シフト実験
│   ├── 9b_tg_lora_accel_aggressive.yaml    # accel攻撃シフト実験
│   ├── 9b_tg_lora_accel_balanced.yaml      # accelバランス実験
│   ├── 9b_tg_lora_accel_no_accel.yaml      # accel ablation基準線
│   ├── 9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml  # ワンショットキャッシュ検証用（REQ-225）
│   ├── 9b_tg_lora_psa.yaml  # PSA実験設定・決定論的比較用（REQ-271）
│   ├── 9b_baseline_suffix_only_last25.yaml  # suffix-onlyベースライン（REQ-231）
│   └── smoke_async_prefix.yaml  # 非同期キャッシュビルドsmoke test用（REQ-143）
├── data/                      # git管理外
│   ├── raw/                   # ダウンロード済み生データ
│   ├── train.jsonl            # ChatML形式学習データ
│   ├── valid_quick.jsonl      # 高速評価用
│   ├── valid_full.jsonl       # 全件評価用
│   └── gold_test.jsonl        # 最終テスト用
├── scripts/
│   ├── download_data.py       # データセットダウンロード
│   ├── prepare_data.py        # データ前処理・分割
│   ├── diagnose.py            # 運用ヘルスチェック（GPU/ckpt/config/log）
│   ├── recover.py             # 障害回復自動化（OOM/CUDA/NaN）
│   ├── run_eval.sh            # lm-evaluation-harness実行
│   ├── run_eval_lora.sh       # LoRAアダプタ評価（3-phase）
│   ├── run_comparison.sh      # 公正比較実験
│   ├── compare_runs.py        # 比較レポート生成
│   ├── inspect_model.py       # モデル構造検査
│   ├── analyze_benchmark.py   # ベンチマーク結果分析（Phase 41）
│   ├── run_accel_sweep.sh     # accel paramスイープ実行（Phase 41）
│   ├── run_paper_memory_suite.sh  # Paper マルチシードスイート実行（Phase 50）
│   ├── evaluate_paper_gates.py    # Paper Gate G0-G4自動評価（Phase 50）
│   ├── summarize_sweep.py         # ハイパーパラメータスイープ結果要約
│   ├── analyze_prefix_cache_break_even.py  # prefix cache損益分岐点分析
│   ├── generate_sweep_dashboard.py  # スイープ結果HTMLダッシュボード生成
│   ├── run_sweep.sh               # グリッドスイープ実行
│   ├── run_ablation_suite.sh      # アブレーションスタディ実行
│   ├── run_high_lr_comparison.sh  # 高LR安定性比較
│   ├── run_kstep_rollback_test.sh # K-step rollback検証
│   ├── run_best_config_eval.sh    # 最良構成lm-eval評価
│   ├── run_accel_sweep_parallel.sh # 2GPU並列accelスイープ
│   ├── run_accel_sweep_auto.sh    # GPU空き監視accelスイープ
│   ├── run_remaining_accel_configs.sh # 残りaccel構成逐次実行
│   ├── frontier_report.py         # Frontier sweep結果分析・OOM検知・メモリ分析（Phase 54）
│   ├── run_frontier_sweep.sh      # Stage 3 frontier sweep実行（Phase 54）
│   ├── run_psa_ablation.sh        # PSA vs plain vs LAWAアブレーション（Phase 62）
│   ├── run_psa_gamma_sweep.sh     # PSA γスイープ実行（Phase 62）
│   ├── summarize_psa_sweep.py     # PSAスイープ結果集約（Phase 62）
│   └── setup_env.sh           # 環境セットアップ
├── src/
│   ├── tg_lora/               # コアアルゴリズム（16モジュール：+psa.py, regime.py, activation_regime.py, weight_averaging.py, layer_delta_analysis.py）
│   ├── training/              # 学習ループ（8モジュール + __init__）
│   ├── model/                 # モデル管理（2モジュール）
│   ├── data/                  # データパイプライン（6モジュール）
│   ├── eval/                  # 評価（3モジュール）
│   ├── analysis/              # 統計分析（1モジュール）
│   └── utils/                 # ユーティリティ（8モジュール + __init__）
├── tests/                     # pytest（129テストファイル、2634テストケース）
├── specs/                     # 要件・設計文書
├── docs/                      # ドキュメント
├── runs/                      # 実験出力（git管理外）
└── reports/                   # 評価レポート（git管理外）
```

## TG-LoRA学習サイクルアーキテクチャ 🔵

**信頼性**: 🔵 *train_tg_lora.py の完全な学習フロー実装より*

```
┌─────────────────────────────────────────────────────────────┐
│                     1 Cycle = K + N steps                    │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐ │
│  │ Pilot Phase  │───▶│  Extrapolate │───▶│  Accept/      │ │
│  │  K steps     │    │  N steps     │    │  Rollback     │ │
│  │  (backward)  │    │  (zero-cost) │    │               │ │
│  └──────────────┘    └──────────────┘    └───────────────┘ │
│         │                   │                    │           │
│    snapshot W0        compute dW            loss_after ≤    │
│    K backward         update velocity       loss_pilot ×    │
│    snapshot WK        select layers         (1+tol)?        │
│    mean delta =       apply W += Nαv        YES: accept     │
│    (WK-W0)/K                               NO: rollback    │
└─────────────────────────────────────────────────────────────┘
```

### コンポーネント相互作用 🔵

**信頼性**: 🔵 *train_tg_lora.pyのサイクルループ実装より*

1. **DeltaTracker**: pilot前後のスナップショット差分からmean deltaを計算
2. **Velocity**: mean deltaをEMAで平滑化し、安定した速度ベクトルを維持
3. **LayerSampler**: 設定された戦略に基づき外挿対象レイヤーを選択
4. **Extrapolator**: 選択レイヤーにのみvelocity方向の重み更新を適用（cap付き）
5. **RollbackManager**: 外挿前状態を保存し、損失悪化時に復元
6. **RandomWalkController**: 受理/拒否フィードバックからK, N, α, lrを適応調整。探索停滞時はadapt_to_convergenceでproactiveにlr減少・K増加。`enable_convergence_adaptation`フラグでrandom_walkとは独立して収束適応を制御可能
7. **CycleState**: サイクル数・backward/extrapolation比・受理率を集計。フル評価サイクルでは `record_full_eval()` でstale_cycles/best_lossを更新（クイック評価と分離して二重計上を防止）
8. **should_run_full_eval**: フル評価実行タイミングを判定する純粋関数
9. **NaN/Inf検証**（REQ-056）: 外挿後にLoRAパラメータの有限性を検証。非有限値検出時は外挿を棄却としロールバック実行。`NumericalInstabilityError`（forward_backward経由）は両trainerの最上位でキャッチされ安全な終了処理を実行（REQ-057）
10. **ActivationCache**（REQ-110~112）: 外挿後評価でスプリットレイヤーの隠れ状態をキャッシュ。予測戦略が実際と一致する場合はキャッシュを再利用し、評価FLOPsを削減。force_top_layers_onlyでスプリットレイヤーの一貫性を保証。Qwen3.5対応: `_get_rotary_emb()` でrotary embedding（cos/sin）を抽出しposition_embeddingsとしてdecoder層に供給、`_get_layer_types()` でhybrid attention（linear_attention→2D full-ones mask, full_attention→None）を自動判定
11. **移動平均ベースライン**（REQ-115）: _decide_accept_rollbackがaccepted_valid_historyの移動平均をベースラインとして使用し、loss_pilotのみの比較よりノイズ耐性が高い評価を実現
12. **Soft accept**（REQ-116）: soft_accept_temperature > 0の場合、Metropolis-Hastings確率で境界ケースを受理し、局所最適解からの脱出を支援
13. **Confident-skip**（REQ-117）: velocity方向が高安定時に評価を省略し自動受理。cos_sim、acceptance_rate、magnitude異常検出を組み合わせた条件判定
14. **K-step中間ロールバック**（REQ-118）: pilotフェーズでdivergence検出時、delta snapshotから中間点を評価し最良点にロールバック。フルロールバック時はvelocityをdelta=0で更新
15. **diff_lora高速化**（REQ-140）: scale==0.0の場合はゼロテンソル直接返却、scale==1.0の場合は単純減算のみ実行。不要な乗算を回避
16. **cosine_similarity直交ベクトル警告**（REQ-141）: 内積=0・ノルム>0の直交ベクトル入力時にwarnings.warn()で警告を発し、デバッグを支援
17. **AsyncCacheBuilder統合テストギャップ**（REQ-139）: 現在モックベースのユニットテストのみ。CPU上の軽量モデルでbuild→wait→loadのフルライフサイクル統合テストが必要
18. **--resume障害回復再開**（REQ-162~164）: `--resume path/to/training_state.pt`で保存済みTrainingStateからcontroller/velocity/delta_tracker/cycle_stateを復元。cycle < cycle_offsetのサイクルをスキップし中断箇所から継続。controller.restore_state()はconfig保持しつつstate値のみ置換
19. **加速度適応観測性**（REQ-165~166）: `last_accel_action`（1=不安定, -1=収束, 0=無行動）でadapt_to_acceleration()の実行結果を追跡。MLflow cycle metricsに`magnitude_acceleration`と`accel_action`を追加し、加速度適応の効果をダッシュボードで可視化可能
20. **Velocity加速度検出**（REQ-153~155）: `magnitude_acceleration()`がmagnitude履歴の二階微分を計算し、正値（不安定）時にlr減衰+K増加、負値（収束）時にlr増加。adapt_to_convergence()とは独立した適応軸（Phase 37）
21. **加速度適応パラメータの設定露出**（REQ-160~161）: `accel_instability_lr_decay`（開区間(0,1)）と`accel_convergence_lr_boost`（>1.0）をYAML設定から制御可能。Pydantic値域検証で不正値を拒否（Phase 38）

### PSA (Prior-based Subspace Amplification) パイプライン 🔵

**信頼性**: 🔵 *Phase 62（REQ-265~284）psa.py・regime.py・activation_regime.py・weight_averaging.py・layer_delta_analysis.py実装・docs/GOAL.mdより*

PSAはTG-LoRAのメインライン研究方向（REQ-284）。外挿ベースのアプローチは設定で切り替え可能なサーフェスとして残存するが、PSAが主開発対象。

1. **PSAPrior勾配増幅**（REQ-265）: backward→optimizer.step()間で勾配をin-place増幅。G_amplified = G + gamma * <G, v_PSA> * v_PSA。power iterationで抽出した安定したper-tensor PC1方向に沿った増幅
2. **L2正則化prior安定化**（REQ-266）: extract_priors()がRNA理論のL2正則化を適用し、前回priorからの急激な方向転換を防止
3. **Layer-type gain適応**（REQ-267）: compute_gain_map()がテンソル名に基づくlayer-type-specific gain（out_proj×1.2, v_proj×1.1, MLP×0.7）を提供
4. **Regime-aware prior reset**（REQ-269/272/273）: RegimeDetectorがSTABLE→PLATEAU→TRANSITION遷移を検知し、consume_reset_signal()でPSAPriorにリセットを通知。新フェーズでpriorを再構築
5. **Warmup制御**（REQ-268）: warmup_steps（デフォルト4）までは増幅無効。update_interval間隔でprior更新
6. **ActivationFingerprintTracker**（REQ-274/275）: forward hookで活性化cosine similarityを計算し、STABLE/TRANSITION/CHAOTICを分類。compute_regime_null_baseline()で時系列シャッフルヌルベースライン
7. **LAWAベースライン**（REQ-276/277）: スライディングウィンドウ重み平均をmandatoryベースラインとして提供。evaluate_with_lawa()で一時的に平均重みに差し替えて評価
8. **Layer Delta Analysis**（REQ-278/279）: per-tensor ΔWのPC1 dominance, direction stability, Marchenko-Pasturヌルベースラインz-scoreを計算し、ATTENTION_OUT/V/OTHER/DELTANET/MLP/UNKNOWNに分類

## 設定管理 🔵

**信頼性**: 🔵 *configs/*.yamlの完全な構造より*

Hydra/OmegaConf形式のYAMLで実験設定を完全に記述。ベースライン（`9b_baseline.yaml`）とTG-LoRA（`9b_tg_lora.yaml`）で共通セクション（model, lora, data, training, eval, logging）を共有し、TG-LoRAに`tg_lora`セクションが追加される構成。

### 設定スキーマ検証 🔵

**信頼性**: 🔵 *config_schema.py extra='forbid'実装・REQ-061/062要件定義より*

Pydanticスキーマによる多層検証（REQ-045）:

- **extra='forbid'**（REQ-061）: 全11モデル（ExperimentConfig, ModelConfig, LoRAConfig, DataConfig, TrainingConfig, EvalConfig, MLflowConfig, LoggingConfig, TGLoRAParams, BaselineConfig, TGLoRAConfig）で未知フィールドを拒否。YAML内のタイポ（例: "lerning_rate"）を学習開始前に検出（EDGE-126）
- **不正YAML検出**（REQ-062）: 空ファイルやリストに解決されるYAMLをValueErrorで拒否（EDGE-127）
- **列挙型検証**（REQ-058）: `dtype`, `bnb_4bit_compute_dtype` をLiteral型で検証（EDGE-122）。`active_layer_strategy`, `bnb_4bit_quant_type` も既にLiteral enumで検証済み
- **値域制約**（REQ-047）: K > 0, N > 0, alpha_min < alpha_max, max_seq_len >= 32等を検証

### TG-LoRA固有パラメータ 🔵

**信頼性**: 🔵 *9b_tg_lora.yaml設定より*

| パラメータ | 型 | 説明 | 設定値 |
|-----------|-----|------|--------|
| `K_initial` | int | pilot歩数 | 3 |
| `K_candidates` | list[int] | ランダムウォーク候補 | [2, 3, 5, 8] |
| `N_initial` | int | 外挿歩数 | 5 |
| `N_candidates` | list[int] | ランダムウォーク候補 | [1, 3, 5, 10, 20] |
| `alpha_initial` | float | 外挿強度 | 0.3 |
| `alpha_min/max` | float | α範囲制約 | 0.03 / 1.5 |
| `alpha_log_sigma` | float | αランダムウォーク対数ステップ幅 | 0.15 |
| `beta_initial` | float | velocity EMA係数 | 0.8 |
| `beta_candidates` | list[float] | βランダムウォーク候補 | [0.5, 0.8, 0.9, 0.95] |
| `lr_initial` | float | 初期学習率 | 5e-4 |
| `lr_min/max` | float | lr範囲制約 | 1e-5 / 1e-3 |
| `lr_accept_boost` | float | 受理時lr増加倍率 | 1.2 |
| `lr_reject_decay` | float | 拒否時lr減衰率 | 0.5 |
| `relative_update_cap` | float | 更新上限比率 | 0.005 |
| `active_layer_strategy` | str | レイヤー選択戦略 | last_25_percent_plus_random_2 |
| `random_middle_layers` | int | ランダム中間層選択数 | 2 |
| `force_top_layers_only` | bool | 決定論的レイヤー選択強制（REQ-114） | false |
| `enable_random_walk` | bool | ランダムウォーク有効/無効（REQ-113） | true |
| `lr_explore_prob` | float | propose()でlog-normal lr探索をトリガーする確率（REQ-150） | 0.3 |
| `lr_log_sigma` | float | log-normal lr探索の標準偏差（REQ-150） | 0.1 |
| `confident_skip_cos` | float | confident-skip cos類似度閾値（0.0=無効）（REQ-117） | 0.0 |
| `confident_skip_min_cycles` | int | confident-skip最低サイクル数（REQ-117） | 10 |
| `layer_sample_temperature` | float | レイヤーサンプリング温度 | 1.0 |

### 評価設定追加パラメータ 🔵

**信頼性**: 🔵 *config_schema.py EvalConfig・9b_tg_lora.yaml設定より*

| パラメータ | 型 | 説明 | 設定値 |
|-----------|-----|------|--------|
| `moving_avg_window` | int | 受理判定移動平均ウィンドウ（REQ-115） | 3 |
| `soft_accept_temperature` | float | Metropolis-Hastings温度（0.0=無効）（REQ-116） | 0.0 |

### PSA設定パラメータ 🔵

**信頼性**: 🔵 *config_schema.py PSAConfig・configs/9b_tg_lora_psa.yaml・Phase 62（REQ-270~271）より*

| パラメータ | 型 | 説明 | 設定値 |
|-----------|-----|------|--------|
| `enable_psa` | bool | PSA有効/無効マスタースイッチ | true |
| `psa_history_length` | int | deltaリングバッファ長（REQ-265） | 6 |
| `psa_gain` | float | 増幅係数γ（REQ-265） | 0.5 |
| `psa_update_interval` | int | prior更新間隔（REQ-268） | 3 |
| `psa_warmup_steps` | int | 増幅無効ステップ数（REQ-268） | 4 |
| `psa_l2_reg` | float | prior L2正則化係数（REQ-266） | 0.01 |
| `psa_regime_reset_enabled` | bool | regime-aware prior reset（REQ-269） | true |
| `psa_regime_window` | int | RegimeDetector loss窓幅 | 8 |
| `psa_regime_plateau_eps` | float | plateau判定閾値 | 1e-4 |
| `psa_regime_transition_z` | float | transition判定z-score閾値 | 2.0 |
| `enable_lawa` | bool | LAWAベースライン有効/無効（REQ-276） | false |
| `lawa_window_size` | int | LAWAスライディングウィンドウ | 5 |
| `lawa_start_cycle` | int | LAWA平均開始サイクル | 10 |
| `activation_regime_enabled` | bool | 活性化レジーム追跡有効/無効（REQ-274） | false |

## 非機能要件の実現方法

### パフォーマンス 🔵

**信頼性**: 🔵 *AGENTS.md VRAM仕様・load_model.py実装より*

- **VRAM最適化**: 4bit NF4量子化 + LoRA r=16 + gradient checkpointing で9Bモデルを12GB VRAMに収容
- **計算効率**: レイヤーサンプリングで外挿対象パラメータを削減
- **外挿のzero-cost性**: 外挿はbackward passなしのパラメータ直接操作

### 信頼性 🔵

**信頼性**: 🔵 *rollback_manager.py・train_tg_lora.py実装・Phase 14修正より*

- **ロールバック**: 外挿前状態を必ず保存し、損失悪化時（loss_after > loss_pilot × (1+tolerance)）に自動復元
- **スナップショットサニタイズ**（REQ-064）: `save()` 時にNaN→0.0, +Inf→+1e6, -Inf→-1e6にサニタイズし、破損状態の復元を防止（RISK-0074）
- **履歴上限管理**（REQ-065）: `max_history`（デフォルト100）でFIFO破棄し、長時間学習でのメモリリークを防止（RISK-0074）
- **try/finally安全性**: 例外発生時もロールバックが実行されることを保証
- **eval状態リーク防止**: eval_loss.py のcontext manager でdropout/training modeの変更を確実に復元

### 数値安全性 🔵

**信頼性**: 🔵 *trainer_loop.py NumericalInstabilityError定義・Phase 14修正・要件定義より*

- **forward_backward NaN/Inf検出**: `forward_backward()` で非有限lossを検出し `NumericalInstabilityError` を送出。両trainerで共有される安全機構（REQ-057）
- **外挿後パラメータ有限性検証**（REQ-056）: `apply_extrapolation()` 後にLoRAパラメータが有限値であることを検証。NaN/Inf検出時は外挿を棄却として扱いロールバックを実行（EDGE-121）
- **cap_update非有限ガード**（REQ-063）: `cap_update()` が非有限（NaN/Inf）の更新テンソルを検出した場合、NaN伝播（inf * 0 = NaN）を防止しゼロテンソルを返す（EDGE-128）
- **メトリクス安全性**（REQ-066）: `cosine_similarity()` が辞書間キー不一致を安全にスキップし、完全不一致時は0.0を返す（EDGE-132）
- **差分追跡安全性**（REQ-067/068）: `DeltaTracker._compute_stats()` が非有限normテンソルをスキップし、`compute_and_record()` がNaN/Inf normを履歴に追加しない（EDGE-133/134）
- **勾配クリッピング**: `optimizer_step()` で `max_grad_norm` を適用。両trainerで共有
- **設定dtype列挙検証**（REQ-058）: `dtype`, `bnb_4bit_compute_dtype` を `Literal` 型で検証し、無効な文字列を学習開始前に拒否（EDGE-122）。既に `bnb_4bit_quant_type` と `active_layer_strategy` は `Literal` enumで検証済み

### 再現性 🔵

**信頼性**: 🔵 *seed.py実装・configs構造より*

- **乱数シード**: random, numpy, torch, CUDA の全レイヤーでシード設定
- **設定の完全記述**: YAMLで全ハイパーパラメータを記録し、実験の再現を保証
- **構造化ログ**: JSONL形式でステップごとのメトリクスを記録

### 運用性 🔵

**信頼性**: 🔵 *Makefile・scripts/構成・Phase 27運用スクリプトより*

- **Makefileターゲット**: `make install`, `download-data`, `prepare-data`, `train-baseline`, `train-tg-lora`, `eval`, `compare`, `ci`, `diagnose`, `recover`
- **CI パイプライン**（REQ-108）: `make ci` で lint（ruff check + format check）+ テスト（pytest）+ スクリプトインポート健全性チェック（diagnose.py/recover.pyの正常import確認）を単一コマンドで実行
- **E2E統合テスト**: `make test-integration` でadvise_training・analyze_trajectoryのE2Eテスト（18テスト）を分離実行
- **軌跡・アドバイザテスト**: `make test-trajectory` でPhase 59-61関連テスト（157テスト）を一括実行
- **CLI健全性テスト**: `make test-cli-help` で全25スクリプトの--help応答・import健全性を検証
- **運用診断**（REQ-106）: `make diagnose` でGPU状態・チェックポイント完全性・設定バリデーション・ログ分析の自動ヘルスチェック
- **障害回復**（REQ-107）: `make recover` でOOM/CUDA/NaNエラーの自動分析・チェックポイントサニタイズ・復旧設定生成
- **公正比較**: `make compare BUDGET=1500` で同一backward pass予算の比較実験
- **レポート自動生成**: 損失曲線プロット、効率メトリクス、受理率を含む比較レポート
- **Paper実験スイート**（REQ-184）: `make paper-memory` でマルチシードcold/warm benchmark実行、`make paper-memory-evaluate-gates` でGate G0–G4自動評価
- **モデル検査**（REQ-218~219）: `make inspect MODEL=Qwen/Qwen3.5-9B` でLoRA互換ターゲットモジュール自動発見、`make inspect-config` でYAML設定経由検査
- **損益分岐点分析**（REQ-226~227）: `make analyze-prefix-break-even` でprefix feature cache投資回収分析
- **データ細粒度ターゲット**（REQ-228）: `download-dolly`, `download-capybara`, `prepare-data-small`, `prepare-capybara` で個別データセットの独立操作
- **クリーンアップ**（REQ-229）: `clean`, `clean-data`, `clean-runs` で生成物削除
- **キャッシュモード比較**（REQ-230）: `compare-prefix-cold`, `compare-prefix-warm`, `compare-prefix-coldwarm` でcold/warm比較実験
- **PSAアブレーション**（REQ-280）: `bash scripts/run_psa_ablation.sh`でPSA vs plain LoRA vs LAWAの同一backward予算比較
- **PSA γスイープ**（REQ-281）: `bash scripts/run_psa_gamma_sweep.sh`でγ ∈ {0.0, 0.5, 1.0, 2.0} × regime_reset {on, off}のグリッドサーチ

**信頼性**: 🔵 *checkpoint.py TrainingState実装・cycle_state.py from_dict()・random_walk_controller.py ControllerState実装・Phase 27より*

- **TrainingState dataclass**（REQ-105）: CycleState + ControllerState + Velocity + DeltaTracker + cycle_offset を統合した学習状態コンテナ
- **保存フロー**: PyTorch tensorはCPU変換、historyはlist化、JSON形式で `torch.save()` によりディスクに保存
- **復元フロー**: `torch.load()` でblobを読み込み、各コンポーネントを `from_dict()` または直接代入で再構築。CycleStateは`summary()`と旧checkpoint形式の両方に対応（後方互換）
- **ControllerState**（REQ-103）: K, N, alpha, beta, lr, strategy, layer_scores, boost/decay params を `summary()` → `from_dict()` で完全往復可能
- **CycleState**（REQ-104）: `summary()` → `from_dict()` で完全往復可能。空辞書・部分データはデフォルト値で初期化

## Paper実験パイプライン 🔵

**信頼性**: 🔵 *docs/paper_experiment_plan.md・scripts/run_paper_memory_suite.sh・scripts/evaluate_paper_gates.py・Phase 50（REQ-179~184）より*

Paper実験パイプラインは、[paper_experiment_plan.md](../../docs/paper_experiment_plan.md)で定義されたClaim Ladder（C0~C2）に従い、Stage 0~5の段階的実験で不確実性を潰すためのインフラストラクチャ。

### Gate評価アーキテクチャ 🔵

**信頼性**: 🔵 *evaluate_paper_gates.py実装・REQ-179~183より*

| Gate | 責務 | 判定基準 |
|------|------|----------|
| G0: Hygiene | アーティファクト完全性 | seeds非空・per_seed数一致・aggregate必須キー存在（REQ-180） |
| G1: Replicated Internal Efficiency | マルチシード内部効率再現 | 全seed TG>BL・backward減少・TG/BL ratio≥2.0x・quality悪化<1%（REQ-181） |
| G2: Memory Frontier Separation | メモリフロンティア分離 | TG peak memory削減≥20%・runtime offload freed>0（REQ-182） |
| G3: External Quality Retention | 外部品質保持 | aggregate relative drop<1%・単一task drop<3%（informational from suite） |
| G4: Causal Attribution | 因果帰属 | warm speedup正・cold/warm差明確・train cache on>off（informational from suite） |

### 実験サーフェス 🔵

**信頼性**: 🔵 *paper_experiment_plan.md・configs/より*

| サーフェス | config | 役割 |
|-----------|--------|------|
| Main baseline | `9b_baseline_suffix_only_last25.yaml` | 比較基準 |
| Main treatment | `9b_tg_lora_prefix_feature_cache_paper_poc.yaml` | TG-LoRA + prefix feature cache（paper本命） |
| Secondary support | `9b_tg_lora_paper_poc.yaml` | 歴史的比較用 |
| Mode comparison | one-shot vs reuse | cache永続化戦略の比較（one-shotが本命） |

### Stage 2実行パラメータ 🔵

**信頼性**: 🔵 *paper_experiment_plan.md Stage 2定義・run_paper_memory_suite.shより*

```
make paper-memory SEEDS='42 43 44' TARGET_BP=240 MAX_SEQ_LEN=1024 \
    OUTPUT_BASE=runs/paper_memory_suite_ms3_s1024
```

- 3 seed（42, 43, 44）でcold/warmペア実行
- 各seedでbaseline + TG-LoRA cold/warm benchmark
- aggregate_summary.jsonで全seed統計（mean, stdev, per_seed）
- evaluate_paper_gates.pyでGate自動評価

## 技術的制約

### ハードウェア制約 🔵

**信頼性**: 🔵 *AGENTS.md・要件定義REQ-401より*

- RTX3060 12GB VRAMでのみ検証済み
- 4bit量子化（NF4）必須。fp16/bf16ではVRAM超過の可能性
- gradient checkpointing必須（無効化するとOOM）

### モデル制約 🔵

**信頼性**: 🔵 *AGENTS.md・要件定義REQ-402より*

- Qwen3.5-9Bのハイブリッドアーキテクチャ（24 DeltaNet + 8 Attention層）専用設計
- LoRA対象: `all-linear`（DeltaNet層のLinearも含む）
- 4096 hidden, 248K vocab

### データ制約 🔵

**信頼性**: 🔵 *要件定義REQ-404・docs/datasets.mdより*

- 初期検証は公開データセットのみ（Dolly 15k, Capybara）
- ChatML形式（Qwen系モデルのテンプレート）
- max_seq_len = 2048

## 関連文書

- **データフロー**: [dataflow.md](dataflow.md)
- **設計分析記録**: [design-interview.md](design-interview.md)
- **要件定義**: [requirements.md](requirements.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **実装分析記録**: [interview-record.md](interview-record.md)
- **APIリファレンス**: [docs/api_reference.md](../../docs/api_reference.md)（REQ-109）

## 信頼性レベルサマリー

- 🔵 青信号: 92件 (99%)
- 🟡 黄信号: 1件 (1%)
- 🔴 赤信号: 0件 (0%)🟡は設定値の将来的な変更可能性のみ。Phase 62 PSAパイプライン（psa.py PSAPrior・regime.py RegimeDetector・activation_regime.py ActivationFingerprintTracker・weight_averaging.py LAWAAverager・layer_delta_analysis.py、REQ-265~284）を反映。Phase 57 ギャップ補完（統計分析モジュール: src/analysis/stats.py、REQ-259~262）。Phase 59（軌跡分析: trajectory.py TrajectoryAnalyzer・analyze_trajectory.py CLI、REQ-232~236）。Phase 60（軌跡連動適応制御: trajectory_controller.py TrajectoryController・CycleDecision、REQ-237~240）。Phase 57（論文結果エクスポート: export_paper_results.py LaTeX/Markdown/CSV、感度分析: analyze_sensitivity.py Pearson相関・ランキング、REQ-241~244）。Phase 58（サイクルヘルスモニタ: cycle_monitor.py CycleMonitor・実験構成比較: compare_experiment_configs.py、REQ-245~250）。Phase 61（学習アドバイザ: training_advisor.py TrainingAdvisor・AdvisoryReport・advise_training.py CLI、REQ-251~258）。Phase 21 DRYリファクタリング（batch_iter, checkpoint, loss, io, memory, mlflow_logger抽出）とPhase 22（公開API・入力検証）を反映済み。Phase 27~62の全内容を反映済み。dataflow.mdにPhase 62のFlow 24~28を追加（PSAパイプライン・レジーム検知・活性化フィンガープリント・LAWAベースライン・レイヤー分析）。


<!-- spine:children:begin -->
## Spine: child documents

- [TG-LoRA 受け入れ基準](acceptance-criteria.md)
- [TG-LoRA データフロー図](dataflow.md)
- [TG-LoRA 自動分析記録](interview-record.md)
- [TG-LoRA 要件定義書](requirements.md)
- [TASK-0001: build_seed_dataset ユニットテスト追加](tasks/TASK-0001.md)
- [TASK-0002: filter_dataset, dedup, provenance ユニットテスト追加](tasks/TASK-0002.md)
- [TASK-0003: load_model, lora_utils ユニットテスト追加](tasks/TASK-0003.md)
- [TASK-0004: eval_loss, eval_task, eval_format ユニットテスト追加](tasks/TASK-0004.md)
- [TASK-0005: trainer_loop ユニットテスト追加](tasks/TASK-0005.md)
- [TASK-0006: AGENTS.md と要件定義の同期](tasks/TASK-0006.md)
- [TASK-0007: 受け入れ基準の全テストケース検証](tasks/TASK-0007.md)
- [TASK-0008: run_metrics.py GPUパス カバレッジ完成](tasks/TASK-0008.md)
- [TASK-0009: load_model.py モックベースユニットテスト追加](tasks/TASK-0009.md)
- [TASK-0010: コアモジュール ソースコード精査とバグ修正](tasks/TASK-0010.md)
- [TASK-0011: Phase 2 受け入れ基準・ドキュメント更新](tasks/TASK-0011.md)
- [TASK-0012: train_tg_lora.py 純粋関数抽出](tasks/TASK-0012.md)
- [TASK-0013: モックベース学習ループ統合テスト](tasks/TASK-0013.md)
- [TASK-0014: train_baseline_qlora.py モックテストカバレッジ](tasks/TASK-0014.md)
- [TASK-0015: Phase 3 受け入れ基準・ドキュメント更新](tasks/TASK-0015.md)
- [TASK-0016: Pydantic 設定スキーマによる設定検証](tasks/TASK-0016.md)
- [TASK-0017: CLI エントリポイント GPUモックテスト](tasks/TASK-0017.md)
- [TASK-0018: 学習開始前バリデーションと設定スキーマ統合](tasks/TASK-0018.md)
- [TASK-0019: Phase 4 受け入れ基準・ドキュメント更新](tasks/TASK-0019.md)
- [TASK-0020: docs/llm-wiki git tracking 解除とクリーンアップ](tasks/TASK-0020.md)
- [TASK-0021: _InfiniteBatchIterator StopIteration テスト強化](tasks/TASK-0021.md)
- [TASK-0022: MLflow 実験ロギング統合](tasks/TASK-0022.md)
- [TASK-0023: GitHub Actions CI/CD パイプライン構築](tasks/TASK-0023.md)
- [TASK-0024: Docker 開発環境構築](tasks/TASK-0024.md)
- [TASK-0025: データスキーマバリデーション追加](tasks/TASK-0025.md)
- [TASK-0026: 比較レポート可視化強化](tasks/TASK-0026.md)
- [TASK-0027: GPU学習環境準備とモデル読み込み検証](tasks/TASK-0027.md)
- [TASK-0028: TG-LoRA 10サイクル学習スモークテスト](tasks/TASK-0028.md)
- [TASK-0029: ベースラインQLoRA学習実行](tasks/TASK-0029.md)
- [TASK-0030: 公正比較実験と結果分析](tasks/TASK-0030.md)
- [TASK-0031: Phase 9 受け入れ基準・ドキュメント更新](tasks/TASK-0031.md)
- [TASK-0032: 実外挿コードによるNaN検出統合テスト](tasks/TASK-0032.md)
- [TASK-0033: 多様化障害サイクル統合テスト](tasks/TASK-0033.md)
- [TASK-0034: Phase 14 信頼性修正の全テストスイート検証とリグレッション確認](tasks/TASK-0034.md)
- [TASK-0035: Phase 15 ドキュメント・受け入れ基準更新](tasks/TASK-0035.md)
- [TASK-0036: ベースライン学習にEvalLossResult統合](tasks/TASK-0036.md)
- [TASK-0037: 未使用Config項目の整理と早期停止パラメータ露出](tasks/TASK-0037.md)
- [TASK-0038: MLflow ロギングのベースライン/TG-LoRA間一貫性確保](tasks/TASK-0038.md)
- [TASK-0039: Layer Sampler temperatureパラメータ統合テスト](tasks/TASK-0039.md)
- [TASK-0040: RunMetrics perplexity出力とEvalLossResult E2Eテスト](tasks/TASK-0040.md)
- [TASK-0041: 空テストスタブ補完とエッジケースカバレッジ向上](tasks/TASK-0041.md)
- [TASK-0042: Phase 16-17 ドキュメント更新](tasks/TASK-0042.md)
- [TASK-0043: Perplexity E2Eパイプライン統合テスト](tasks/TASK-0043.md)
- [TASK-0044: Trainer間perplexity配管パリティテスト](tasks/TASK-0044.md)
- [TASK-0045: accept()プロパティベーステスト（hypothesis）](tasks/TASK-0045.md)
- [TASK-0046: Phase 19 完了ドキュメント更新](tasks/TASK-0046.md)
- [TASK-0047: GPUテストOOM保護とリソース競合対策](tasks/TASK-0047.md)
- [TASK-0048: テストスイート全通過確認とリグレッションテスト](tasks/TASK-0048.md)
- [TASK-0049: DRYリファクタリング — InfiniteBatchIterator・StrategyList・CheckpointHelper](tasks/TASK-0049.md)
- [TASK-0050: Phase 21 全テストスイート検証とoverview更新](tasks/TASK-0050.md)
- [TASK-0051: save_checkpoint readback検証テスト](tasks/TASK-0051.md)
- [TASK-0052: InfiniteBatchIterator エッジケーステスト](tasks/TASK-0052.md)
- [TASK-0053: 非有限loss_after warning log追加](tasks/TASK-0053.md)
- [TASK-0054: RollbackManager rollback例外E2Eテスト](tasks/TASK-0054.md)
- [TASK-0055: Phase 23 全テストスイート検証とoverview更新](tasks/TASK-0055.md)
- [TASK-0056: MLflowアーティファクトロギング統合](tasks/TASK-0056.md)
- [TASK-0057: MLflowランメタデータ自動生成](tasks/TASK-0057.md)
- [TASK-0058: TG-LoRA特化メトリクスMLflow統合](tasks/TASK-0058.md)
- [TASK-0059: MLflowリトライロジックとエラー強化](tasks/TASK-0059.md)
- [TASK-0060: RunMetrics履歴クエリAPI](tasks/TASK-0060.md)
- [TASK-0061: ラン比較CLI・ダッシュボード](tasks/TASK-0061.md)
- [TASK-0062: 学習曲線可視化ユーティリティ強化](tasks/TASK-0062.md)
- [TASK-0063: 学習ジョブ障害回復・自動リスタート](tasks/TASK-0063.md)
- [TASK-0064: 運用ランブック・APIリファレンス整備](tasks/TASK-0064.md)
- [TASK-0065: Phase 24-26 全テスト検証・ドキュメント更新](tasks/TASK-0065.md)
- [TASK-0066: Ruff lint・format クリーンアップ](tasks/TASK-0066.md)
- [TASK-0067: テストスイート全通過確認とoverview整合性更新](tasks/TASK-0067.md)
- [TASK-0068: OptimizerLifecycleManager E2E スモークテスト](tasks/TASK-0068.md)
- [TASK-0069: ベンチマークスクリプトスモークテスト](tasks/TASK-0069.md)
- [TASK-0070: Phase 30 ドキュメント更新](tasks/TASK-0070.md)
- [TASK-0071: Makefile smoke・ablation・bench-optimizer ターゲット検証](tasks/TASK-0071.md)
- [TASK-0072: trainable_lora_scope 統合テスト](tasks/TASK-0072.md)
- [TASK-0073: prefix_feature_cache 拡張テスト](tasks/TASK-0073.md)
- [TASK-0074: Makefile 実験configターゲット配線検証](tasks/TASK-0074.md)
- [TASK-0075: Phase 31 ドキュメント更新](tasks/TASK-0075.md)
- [TASK-0076: REQ-136~138 acceptance criteria追加](tasks/TASK-0076.md)
- [TASK-0077: AsyncCacheBuilder境界値テスト追加](tasks/TASK-0077.md)
- [TASK-0078: Phase 33 overview.md更新とテスト数修正](tasks/TASK-0078.md)
- [TASK-0079: In-place tensor ops data_ptr保存検証テスト](tasks/TASK-0079.md)
- [TASK-0080: Velocity EMA・cap_update マイクロベンチマーク](tasks/TASK-0080.md)
- [TASK-0081: Phase 34 overview.md更新とテスト数同期](tasks/TASK-0081.md)
- [TASK-0082: bench-velocity-ops-ci Makefile target + baseline file](tasks/TASK-0082.md)
- [TASK-0083: scripts/inspect_model.py・summarize_sweep.py インポート健全性テスト](tasks/TASK-0083.md)
- [TASK-0084: スクリプトConfig YAML Schema検証統合](tasks/TASK-0084.md)
- [TASK-0085: Phase 35-36 overview.md更新とテスト数同期](tasks/TASK-0085.md)
- [TASK-0086: LR探索統合・propose→training loop配線](tasks/TASK-0086.md)
- [TASK-0087: LR探索Config明示化と全Config検証](tasks/TASK-0087.md)
- [TASK-0088: テストスイート警告解消と品質向上](tasks/TASK-0088.md)
- [TASK-0089: Phase 38 ドキュメント同期と受け入れ基準更新](tasks/TASK-0089.md)
- [TASK-0090: --resume E2E統合テスト（save→interrupt→resume→verify loss）](tasks/TASK-0090.md)
- [TASK-0091: TruthfulQAベンチマーク結果分析とaccel param効果調査](tasks/TASK-0091.md)
- [TASK-0092: Accel adaptation param実験config作成とスイープスクリプト](tasks/TASK-0092.md)
- [TASK-0093: Phase 40-41 ドキュメント更新](tasks/TASK-0093.md)
- [TASK-0094: Accel param sweep実行と結果分析](tasks/TASK-0094.md)
- [TASK-0095: OptimizerLifecycleManager + InfiniteBatchIterator コンストラクタ検証](tasks/TASK-0095.md)
- [TASK-0096: LoraDataset + PrefixFeatureDataset + MappedPrefixFeatureDataset コンストラクタ検証](tasks/TASK-0096.md)
- [TASK-0097: MLflowLogger + RunMetrics + EvalLossResult コンストラクタ検証](tasks/TASK-0097.md)
- [TASK-0098: AsyncCacheBuilder コンストラクタ検証](tasks/TASK-0098.md)
- [TASK-0099: RandomWalkController lr_explore_prob 非決定論性修正](tasks/TASK-0099.md)
- [TASK-0100: 統計アサーション堅牢化](tasks/TASK-0100.md)
- [TASK-0101: Phase 44-46 テストスイート検証とoverview更新](tasks/TASK-0101.md)
- [TASK-0102: NaN/Inf バリデーションとランタイムガード完全化](tasks/TASK-0102.md)
- [TASK-0103: Phase 49 テスト数同期・ドキュメント整合性更新](tasks/TASK-0103.md)
- [TASK-0104: CI gate baselineテスト安定性調査と修正](tasks/TASK-0104.md)
- [TASK-0105: Paper Gate評価自動化スクリプト](tasks/TASK-0105.md)
- [TASK-0106: Stage 2 マルチシード実験実行とGate評価](tasks/TASK-0106.md)
- [TASK-0107: Stage 3 メモリフロンティアスイープ自動化スクリプト](tasks/TASK-0107.md)
- [TASK-0108: 外部品質評価パイプライン（G3 Gate）](tasks/TASK-0108.md)
- [TASK-0109: 因果分析評価ロジック拡張（G4 Gate）](tasks/TASK-0109.md)
- [TASK-0110: 論文結果統合・テーブル自動生成スクリプト](tasks/TASK-0110.md)
- [TASK-0111: Stage 2 実行前Smoke検証強化](tasks/TASK-0111.md)
- [TASK-0112: モデル検査・比較ダッシュボード・ワンショットキャッシュのacceptance criteria追加](tasks/TASK-0112.md)
- [TASK-0113: コスト分析・データ細粒度・クリーンアップターゲットのテスト追加](tasks/TASK-0113.md)
- [TASK-0114: マルチシード統計分析モジュール](tasks/TASK-0114.md)
- [TASK-0115: 論文結果エクスポートツール](tasks/TASK-0115.md)
- [TASK-0116: ハイパーパラメータ感度分析ツール](tasks/TASK-0116.md)
- [TASK-0117: 学習サイクル健全性モニター](tasks/TASK-0117.md)
- [TASK-0118: クロス構成実験マトリクスコンパレータ](tasks/TASK-0118.md)
- [TASK-0119: 学習軌跡分析・収束予測・早期停止推奨](tasks/TASK-0119.md)
- [TASK-0120: 軌跡連動適応制御モジュール・テスト](tasks/TASK-0120.md)
- [TASK-0121: Training Advisor モジュール・CLI](tasks/TASK-0121.md)
- [TASK-0122: advise_training.py E2E統合テスト](tasks/TASK-0122.md)
- [TASK-0123: analyze_trajectory.py E2E統合テスト](tasks/TASK-0123.md)
- [TASK-0124: CLIスモークテスト Makefile ターゲット](tasks/TASK-0124.md)
- [TASK-0125: CI回帰テスト Makefile ターゲット](tasks/TASK-0125.md)
- [TASK-0126: requirements.md 黄信号要件ステータスの実態反映](tasks/TASK-0126.md)
- [TASK-0127: run_eval_lora.sh 終了時trap handler追加](tasks/TASK-0127.md)
- [TASK-0128: overview.md フェーズ進捗とテスト状況の最新化](tasks/TASK-0128.md)
- [TASK-0129: parse_warnings corrupt-JSONL end-to-end integration tests](tasks/TASK-0129.md)
- [TG-LoRA タスク概要](tasks/overview.md)
- [TG-LoRA ユーザストーリー](user-stories.md)

<!-- spine:children:end -->
