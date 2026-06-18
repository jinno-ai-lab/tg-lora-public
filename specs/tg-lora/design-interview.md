# TG-LoRA 設計自動分析記録


<!-- spine:anchor:begin -->
> **Spine anchor**: [Tangent-Gradient LoRA (TG-LoRA) — 全体設計・研究指針 (GOAL)](../../docs/GOAL.md)
>
> - parent: `docs/GOAL.md`
> - status: `canonical_child`
<!-- spine:anchor:end -->

**作成日**: 2026-05-21
**分析実施**: step4 既存情報ベースの差分分析と自動統合

## 分析目的

TG-LoRAの要件定義・既存設計・実装を確認し、技術設計文書の作成にあたる設計判断・差分・統合内容を記録する。既存の [interview-record.md](interview-record.md)（A1-A6: 実装完全性検証）を補完する形で、設計レベルの分析を実施。

## 分析項目と判断

### A7: アーキテクチャパターンの妥当性

**分析日時**: 2026-05-21
**カテゴリ**: アーキテクチャ
**背景**: モジュラー構成がML研究プロジェクトとして適切か確認

**判断**: 既存のモジュラーパイプラインアーキテクチャは妥当。各コンポーネント（velocity, extrapolator等）が独立モジュールとして実装され、学習ループがオーケストレーションする構造は、研究実験での差し替え・比較を容易にする。

**根拠**: `src/tg_lora/` 配下8モジュールが単一責務で分離、`src/training/` がループ制御のみを担当。テストも各モジュール単位で存在し、結合度が低い。

**信頼性への影響**:
- アーキテクチャ設計項目は全て 🔵（既存実装ベース）
- 新規推定なし

---

### A8: 設計文書の対象範囲決定

**分析日時**: 2026-05-21
**カテゴリ**: 技術選択
**背景**: テンプレートに含まれる interfaces.ts, database-schema.sql, api-endpoints.md がTG-LoRAプロジェクトに適用可能か判定

**判断**: 3ファイルとも不要と判定:
- **interfaces**: Python MLプロジェクトであり、外部API契約がない。内部モジュールの型定義は各Pythonファイルの型ヒントで表現済み
- **database-schema**: データベースを使用せず、JSONLファイルベースのデータ管理
- **api-endpoints**: REST APIを持たず、CLI/Makefileが操作インターフェース

**根拠**: プロジェクトの全ソース構造（src/, scripts/, configs/）にDB/API/UI層が存在しない

**信頼性への影響**:
- スキップした3ファイルに該当する設計項目なし
- 生成する3ファイル（architecture, dataflow, design-interview）に集中

---

### A9: テストカバレッジのギャップ分析

**分析日時**: 2026-05-21
**カテゴリ**: 品質
**背景**: コアアルゴリズムはテスト済みだが、データ・評価・モデル層のテストが不足

**判断**: 以下のテストギャップを特定:

| 対象 | テスト状況 | 影響度 |
|------|-----------|--------|
| velocity, extrapolator, layer_sampler, rollback, random_walk, lora_state, metrics | 包括的ユニットテストあり | 🔵 |
| run_metrics (JSONLログ) | テストあり | 🔵 |
| smoke test (GPT-2 E2E) | テストあり | 🔵 |
| data/build_seed_dataset | テストなし | 🟡 |
| data/filter_dataset, dedup, provenance | テストなし | 🟡 |
| eval/eval_loss, eval_task, eval_format | テストなし | 🟡 |
| model/load_model, lora_utils | テストなし | 🟡 |
| training/trainer_loop | smoke test経由のみ | 🟡 |

**根拠**: tests/ 配下9ファイルの内容確認。コアtg_loraモジュールにユニットテストが集中し、周辺モジュールが未カバー。

**信頼性への影響**:
- コアアルゴリズム設計: 🔵（テストで検証済み）
- データ・評価・モデル層の設計: 🟡（実装確認済み、テスト未検証）

---

### A10: 既存ドキュメントとの統合判断

**分析日時**: 2026-05-21
**カテゴリ**: 統合
**背景**: docs/datasets.md, docs/evaluation.md, AGENTS.md, README.mdが既に存在

**判断**: 既存ドキュメントは移行元・参照元として扱い、新規設計文書とは重複させない:

| 既存文書 | 位置づけ | 統合方針 |
|---------|---------|---------|
| docs/datasets.md | データ形式の詳細仕様 | 参照元としてリンク、重複記載なし |
| docs/evaluation.md | 評価手法の詳細仕様 | 参照元としてリンク、重複記載なし |
| AGENTS.md | 開発者向け運用指示 | 参照元としてリンク |
| README.md | プロジェクト概要 | 参照元 |
| specs/tg-lora/requirements.md | EARS要件定義 | 正本として参照 |
| specs/tg-lora/interview-record.md | 実装完全性分析 | 補完関係として共存 |

**根拠**: 既存ドキュメントが各領域をカバーしており、設計文書は要件→実装の橋渡しに集中すべき

**信頼性への影響**:
- 全項目 🔵（既存文書ベース）
- 統合による重複排除で保守性向上

---

### A11: 学習ループ設計上のトレードオフ

**分析日時**: 2026-05-21
**カテゴリ**: アーキテクチャ
**背景**: TG-LoRAのサイクルベース学習に内在する設計上のトレードオフを記録

**判断**: 以下のトレードオフを特定・記録:

1. **per-cycle optimizer再作成**: 各サイクルで新規AdamWを作成する設計。optimizer state（モメンタム等）がリセットされるが、pilot→外挿の明確な分離を保証する。
2. **infinite batch iterator**: epoch境界なしの無限イテレータ。データ順序の偏りリスクがあるが、固定ステップ数での実験制御を優先。
3. **quick eval (64 examples)**: 高速評価は代表性格に疑問ありうるが、サイクルごとの迅速な受理/拒否判定を優先。
4. **baselineにearly stoppingなし**: baselineはmax_steps固定、TG-LoRAのみearly stoppingあり。公正比較の観点で 要確認。

**根拠**: train_tg_lora.py・train_baseline_qlora.pyの実装詳細確認

**信頼性への影響**:
- トレードオフ事項の記録: 🟡（設計判断の根拠は実装から推測）
- 実装動作: 🔵（コードベースで確認済み）

---

### A12: 設定管理の完全性

**分析日時**: 2026-05-21
**カテゴリ**: データモデル
**背景**: configs/*.yamlが実験の完全な再現性を担保するか確認

**判断**: 設定ファイルは実験再現に必要な全パラメータを含む:
- model: 名前、量子化、dtype、device
- lora: r, alpha, dropout, target_modules
- data: パス、max_seq_len
- training: batch_size, grad_accum, lr, weight_decay, max_steps/cycles
- eval: quick_eval_examples, eval間隔, rollback_tolerance
- tg_lora: K, N, alpha, beta, 候補リスト, layer strategy
- logging: バックエンド、ログ間隔、save間隔、run_dir

不足なし。NFR-202（YAML完全記述要件）を満たす。

**根拠**: 9b_baseline.yaml・9b_tg_lora.yamlの全フィールド確認

**信頼性への影響**:
- 設定管理設計: 🔵

---

## 分析結果サマリー

### 確認できた事項

- モジュラーパイプラインアーキテクチャがML研究に適合
- 既存テストはコアアルゴリズムを包括的にカバー
- 設定ファイルが実験の完全な再現性を担保
- 既存ドキュメント（docs/, AGENTS.md）と新規設計文書は補完関係

### 設計方針の決定事項

- interfaces / database-schema / api-endpoints は対象外（Python MLプロジェクト）
- 既存 interview-record.md と共存、本ファイルは設計レベルの分析に特化
- 設計文書は3ファイル（architecture.md, dataflow.md, design-interview.md）に絞る

### 残課題

- データ・評価・モデル層のテストカバレッジ不足（Phase 3対応） — 一部解消済み（test_training_integration.py 24テスト）
- per-cycle optimizer再作成の性能影響の定量的評価
- baseline/TG-LoRA間のearly stopping非対称の公平性確認
- MLflow統合の実際の状況（RunMetrics JSONLで代替されている可能性）

### Phase 4分析: stale_cycles二重計上バグ修正と純粋関数抽出

**分析日時**: 2026-05-21
**カテゴリ**: バグ修正・リファクタリング
**背景**: train_tg_lora.pyのfull evalセクションがrecord_cycleとは別にstale_cyclesを更新しており、full evalサイクルでstaleが最大2増加するバグを発見

**判断**:
1. `CycleState.record_full_eval(full_loss)` を追加 — full eval時のbest_loss/stale_cycles更新をカプセル化
2. `should_run_full_eval(cycle, full_eval_every)` と `build_training_summary(controller, cycle_state, delta_tracker)` をtrain_tg_lora.pyから抽出
3. フル評価サイクルでは record_cycle の valid_loss=None とし、record_full_eval にstale追跡を委譲

**根拠**: train_tg_lora.py lines 196-258 のコード分析。モック訓練ループテストで二重計上を再現・修正確認

**信頼性への影響**:
- REQ-040a（フル評価 stale 分離）を新規追加 🔵
- train_tg_lora.py の学習ループ品質向上
- テストケース 18件追加（321→339）

---

## 分析結果サマリー

### 信頼性レベル分布

**分析前** (既存 interview-record.md):
- 🔵 青信号: 34
- 🟡 黄信号: 6
- 🔴 赤信号: 0

**設計分析後** (A7-A14 + Phase 7):
- 🔵 青信号: 51 (+8: A13安全性分析+4, A14列挙型分析+4)
- 🟡 黄信号: 8 (+0)
- 🔴 赤信号: 0 (+0)

**Phase 6分析後** (パラメータ表修正 + 適応LRフロー追加):
- 🔵 青信号: 47 (+4)
- 🟡 黄信号: 7 (-1: パラメータ表不正確→修正)
- 🔴 赤信号: 0 (+0)

**Phase 15設計更新後** (Phase 14信頼性修正反映):
- 🔵 青信号: 58 (+7: architecture.md +4, dataflow.md +3)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**Phase 23設計更新後** (Phase 21 DRYリファクタリング反映・Phase 23計画追加):
- 🔵 青信号: 63 (+5: utils +4, training +2, -1重複)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 1 (+1: Phase 23計画のREQ-081~084は🔴だがdocs-onlyの設計記録として扱う)

**Phase 23完了後** (REQ-081~084全テスト実装・検証済み):
- 🔵 青信号: 65 (+2: REQ-081 readback完了, REQ-084 rollback E2E完了)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (-1: Phase 23計画項目全て実装完了)

**Phase 29設計整合性回復後** (A23~A26):
- 🔵 青信号: 77 (+4: activation_cache統合 +1, run_query追加 +1, テスト数正確性 +1, CycleState/ActivationCache詳細 +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

### Phase 7分析: Trainer間安全性ギャップとConfig文字列列挙

**A13: Trainer間数値安全性カバレッジギャップ**

**分析日時**: 2026-05-21
**カテゴリ**: 品質・安全性
**背景**: AI_HUB_MAKE_RUN_FEEDBACK指摘。train_baseline_qlora.pyに追加された数値安全性カバレッジがtrain_tg_lora.pyに伝播されていない

**判断**:
1. **forward_backward NaN/Inf検出**: 両trainer共に `trainer_loop.py` の `forward_backward()` で `NumericalInstabilityError` を送出する共有機構を持つ。既に実装済み
2. **外挿後パラメータ有限性検証**: `apply_extrapolation()` 後にNaN/Infチェックが存在しない。設計上のギャップ（REQ-056）
3. **NumericalInstabilityError最上位キャッチ**: 両trainerとも明示的catchがないが、`forward_backward`からのエラーは伝播してプロセス終了させるため、try/finally rollbackで学習状態の保全は担保される
4. **勾配クリッピング**: `optimizer_step()` で共通適用済み
5. **バッチキー検証**: `compute_loss()` で共通適用済み

**根拠**: trainer_loop.py, train_baseline_qlora.py, train_tg_lora.py の完全読み込み比較

**信頼性への影響**:
- 数値安全性設計: 新規追加 🔵（trainer_loop.py共有コードベース）
- 外挿安全性フロー: 新規追加 🔵（REQ-056要件定義ベース）

---

**A14: Config文字列フィールドの列挙型ギャップ**

**分析日時**: 2026-05-21
**カテゴリ**: 設定検証
**背景**: AI_HUB_MAKE_RUN_FEEDBACK指摘。active_layer_strategy='test'が旧スキーマで受理されていた問題と同パターンの文字列フィールドが残存

**判断**:
1. **既にLiteral enum化済み**: `active_layer_strategy` (ActiveLayerStrategy), `bnb_4bit_quant_type` (BnbQuantType)
2. **未対応の文字列フィールド**:
   - `dtype`: 現在 `str` 型 → `Literal["bfloat16", "float16", "float32"]` 化が必要（REQ-058, EDGE-122）
   - `bnb_4bit_compute_dtype`: 現在 `str` 型 → dtypeと同じLiteral型化が必要
   - `backend`, `device_map`, `target_modules`: 値域が広く現状維持で妥当
3. **設計上の決定**: `dtype` と `bnb_4bit_compute_dtype` を `DtypeLiteral = Literal["bfloat16", "float16", "float32"]` に変更

**根拠**: config_schema.py読み込み、test_config_schema.py旧テスト問題、AI_HUB_MAKE_RUN_FEEDBACK指摘

**信頼性への影響**:
- dtype検証設計: 新規追加 🔵（ActiveLayerStrategy/BnbQuantTypeパターンと同一）

### Phase 5分析: Velocity異常検出パイプラインの統合テスト

**分析日時**: 2026-05-21
**カテゴリ**: 品質・テスト
**背景**: Velocity単体テスト（test_velocity.py）は各メソッドを個別に検証しているが、update→magnitude記録→anomaly検出→trend追跡のパイプライン全体をrealisticな時系列データでE2E検証するテストが未整備

**判断**:
1. `test_velocity_anomaly_integration.py` を新規追加（15テストケース）
2. `TestVelocityAnomalyPipelineEndToEnd`: 収束→スパイク→回復のフルライフサイクル、安定訓練の偽陽性防止、単一レイヤー/マルチスケールパラメータ、リングバッファオーバーフロー、リセット状態クリア、cosine similarityのanomaly context一貫性、beta変動対応
3. `TestVelocityDeltaTrackerCombinedAnomaly`: Velocity + DeltaTracker双方のanomaly検出が明確なスパイクで一致、収束トレンドの方向一致、フル訓練シミュレーション（20ステップ収束→スパイク→15ステップ回復）

**根拠**: AI Hubフィードバック「Add an integration test for the anomaly detection pipeline that feeds realistic time-series data through velocity.py and asserts detection/trend outputs end-to-end」

**信頼性への影響**:
- REQ-049/REQ-050のテストカバレッジがunit→integrationに拡張
- テスト総数 547→562 (+15)

### Phase 6分析: 適応学習率設計検証とパラメータ表正確性

**分析日時**: 2026-05-21
**カテゴリ**: 品質・設計検証
**背景**: architecture.mdのTG-LoRA固有パラメータ表の値が実際のconfig (9b_tg_lora.yaml) と一致していなかった。またdataflow.mdに適応LRフローが未記載だったため、実装との整合性確認と補完が必要

**判断**:

1. **パラメータ表の不正確性**: 以下の値がconfigと不一致だった（既定値→実際の設定値）:
   - K_initial: 5 → 3
   - K_candidates: [3,5,7] → [2,3,5,8]
   - N_initial: 10 → 5
   - N_candidates: [5,10,15] → [1,3,5,10,20]
   - alpha_min: 0.01 → 0.03, alpha_max: 2.0 → 1.5
   - beta_initial: 0.9 → 0.8
   - relative_update_cap: 0.5 → 0.005

2. **欠落パラメータ**: lr_initial (5e-4), lr_min (1e-5), lr_max (1e-3), lr_accept_boost (1.2), lr_reject_decay (0.5), alpha_log_sigma (0.15), beta_candidates ([0.5,0.8,0.9,0.95]), random_middle_layers (2) が表から漏れていた

3. **適応LRフロー**: dataflow.mdに以下を追加:
   - RandomWalk状態図にlr boost/decay/clamp処理とadapt_to_convergenceを統合
   - 適応学習率フロー図（受理→boost, 拒否→decay, 境界クランプ, 収束適応判定）
   - lr_reject_decay=0.5の設計意図の明記

4. **テスト検証**: 575テスト全パス確認（test_random_walk_controller.py: LR境界テスト3件 + 収束適応テスト, test_training_integration.py: 適応LRスモークテスト3件含む）

**根拠**: configs/9b_tg_lora.yaml直接読み込み、src/tg_lora/random_walk_controller.py実装確認、pytest 575 passed

**信頼性への影響**:
- パラメータ表: 🟡→🔵（実装と完全一致に修正）
- 適応LRフロー: 新規追加 🔵（実装ベース）
- 全体の信頼性向上: dataflow 🔵件数 +3

---

### Phase 11分析: 外挿安全性統合テスト実装

**A15: REQ-059/060統合テストによる回復フローE2E検証**

**分析日時**: 2026-05-21
**カテゴリ**: 品質・テスト
**背景**: AI_HUB_MAKE_RUN_FEEDBACK指摘。外挿後NaN/Inf検出→rollback→penalize→cycle_state.record_cycle()の完全な回復フローを統合テストでE2E検証する必要がある。REQ-059/060として仕様定義済みだがテスト未実装だった

**判断**:
1. `test_extrapolation_safety_integration.py` を新規追加（14テストケース）
2. `TestNonFiniteRecoveryFlow`: apply_extrapolationをモック化してNaN注入、完全フロー検証（4テスト）
   - TC-059-01: NaN注入→rollback→penalize→record_cycleのE2Eフロー
   - TC-059-B01: rollback後のモデルパラメータ復元検証
   - TC-059-B02: update_layer_scores呼び出し検証
   - TC-059-B03: 非有限検出後のcontinueで通常パスがスキップされる検証
3. `TestNonFiniteRecoverySideEffects`: 副作用の個別検証（4テスト）
   - TC-060-01: penalize(loss_pilot, inf)呼び出し検証
   - TC-060-02: record_cycle(K, N, grad_accum, accepted=False)引数検証
   - TC-060-03: rollback_mgr.pop()のfinally呼び出し検証
   - TC-060-B01: 2サイクル連続NaNの両方回復検証
4. `TestNonFinitePathComponents`: 個別コンポーネントの直接検証（6テスト）
   - check_lora_params_finite: NaN/Inf検出
   - RollbackManager: NaN後のsave/restore
   - RandomWalkController: penalize→alpha変化、update_layer_scores→スコア減少
   - CycleState: rejected_cycle正確記録

**根拠**: train_tg_lora.py 331-355行の非有限回復パス実装・test_training_integration.pyのモックパターン

**信頼性への影響**:
- REQ-059/060テストケースが全て実装済み（チェック状態 `[x]`）
- テスト総数 720→734 (+14)
- 全734テストパス確認

---

### Phase 15分析: Phase 14信頼性修正の設計文書反映

**A16: 設定スキーマ検証の設計文書反映**

**分析日時**: 2026-05-21
**カテゴリ**: 設定検証・アーキテクチャ
**背景**: Phase 14でconfig_schema.pyの全11モデルにextra='forbid'が追加され（REQ-061）、不正YAML検出も追加された（REQ-062）が、architecture.mdとdataflow.mdに未反映

**判断**:
1. architecture.mdに「設定スキーマ検証」セクションを新規追加。extra='forbid'（REQ-061）、不正YAML検出（REQ-062）、列挙型検証（REQ-058）、値域制約（REQ-047）を記載
2. dataflow.mdに「設定検証フロー」を新規追加。YAML読込→dict判定→Pydantic検証→値域制約→列挙型→preflightの完全な検証パイプラインをmermaid図で記述

**根拠**: config_schema.py extra='forbid'実装・load_and_validate_configの完全読み込み

**信頼性への影響**:
- 設定検証設計: 🔵（実装ベース、テスト27件で検証済み）

---

**A17: ロールバック・数値安全性の設計文書更新**

**分析日時**: 2026-05-21
**カテゴリ**: 信頼性・数値安全性
**背景**: Phase 14でrollback_manager.pyにNaN/Infサニタイズ（REQ-064）とmax_history（REQ-065）、extrapolator.pyにcap_update非有限ガード（REQ-063）、metrics.pyにキー不一致安全処理（REQ-066）、delta_tracker.pyに非有限normガード（REQ-067/068）が追加されたが、設計文書に未反映

**判断**:
1. architecture.md:
   - コンポーネント表: extrapolator/delta_tracker/rollback_manager/metricsの責務列にPhase 14機能を追記
   - 信頼性セクション: スナップショットサニタイズ（REQ-064）、履歴上限管理（REQ-065）を追加
   - 数値安全性セクション: cap_update非有限ガード（REQ-063）、メトリクス安全性（REQ-066）、差分追跡安全性（REQ-067/068）を追加
2. dataflow.md:
   - 学習サイクルシーケンス図: save()に「NaN/Infサニタイズ: REQ-064」注記追加（3箇所）
   - 詳細ステップ: スナップショット保存・delta計算・外挿適用にPhase 14安全性を追記
   - エラーハンドリングフロー: サニタイズ注記・履歴上限・設定検証フローを追加
   - データ整合性: スナップショット整合性・履歴サイズ制約・差分統計/履歴整合性を追加

**根拠**: Phase 14コミット5件（f03932a, 7720c98, 81ee464, 3fe19ae, cc8e19f）の完全読み込み

**信頼性への影響**:
- ロールバック設計: 🔵（実装ベース、テスト5件で検証済み）
- 数値安全性設計: 🔵（実装ベース、テスト12件で検証済み）
- architecture.md 🔵件数: 23→31 (+8)
- dataflow.md 🔵件数: 23→29 (+6)

**YAML設定検証**: configs/9b_baseline.yaml, configs/9b_tg_lora.yamlともにextra='forbid'で正常に検証通過を確認。runs/配下にYAMLファイルは存在しない。

---

### Phase 23分析: 共有ユーティリティテストカバレッジと運用堅牢性

**A18: Phase 21 DRYリファクタリング後のコンポーネントギャップ**

**分析日時**: 2026-05-22
**カテゴリ**: 品質・テストカバレッジ
**背景**: Phase 21でInfiniteBatchIterator・save_checkpoint・loss.py等が共有ユーティリティとして抽出されたが、architecture.md/dataflow.mdに未反映のモジュールがあった

**判断**:
1. **src/utils/ ギャップ**: checkpoint.py（6行）、io.py（34行）、memory.py（17行）、mlflow_logger.py（113行）がarchitecture.mdのコンポーネント表に未記載。Phase 21 DRYリファクタリングとPhase 7-8 MLflow/CI/CD追加で生じた未反映
2. **src/training/ ギャップ**: batch_iter.py（22行）、loss.py（23行）が同様に未記載
3. **設計文書更新**: 上記5モジュールをarchitecture.mdコンポーネント表とdataflow.mdに反映。システム構成図にもShared Utilsサブグラフを追加
4. **テストカバレッジギャップ**: checkpoint.pyとbatch_iter.pyはPhase 21で新規作成されたが、save_checkpoint readback検証（REQ-081）とInfiniteBatchIteratorエッジケーステスト（REQ-082）が未実装

**根拠**: `ls src/utils/ src/training/` で8ファイル/8ファイルを確認。architecture.mdには3/5モジュールのみ記載

**信頼性への影響**:
- コンポーネント表: 🔵件数 31→36 (+5: utils +4, training +2, -1重複)
- Phase 23計画セクション (+4 🔴: REQ-081~084) → 全て完了し🔵に昇格

---

**A19: Phase 23対象要件の技術分析**

**分析日時**: 2026-05-22
**カテゴリ**: 設計判断
**背景**: AI_HUB_MAKE_RUN_FEEDBACKがREQ-081~084の実装を推奨。技術的実現可能性と優先度を分析

**判断**:

| 要件 | 技術的実現性 | 優先度 | 影響範囲 | 推定工数 |
|------|-------------|--------|---------|---------|
| REQ-081 (checkpoint readback) | 高 — `Path.exists()` + `len(list(dir.glob("*")))` で実装可能。save_checkpointは6行だが、ディスク書き込みの失敗モードを実運用で検出する価値あり | 高 | checkpoint.py | 1h |
| REQ-082 (InfiniteBatchIterator edge cases) | 高 — 単一バッチDataLoaderの無限反復、`torch.device("cuda")` と `"cuda"` の両方受け入れをテスト。既存テスト（test_training_integration.py）で基本的な使用パターンは検証済み | 中 | batch_iter.py | 0.5h |
| REQ-083 (non-finite loss warning) | 高 — `math.isfinite(loss_after)` ガード内に `logger.warning()` を1行追加。デバッグ可能性とobs serving性の両方に寄与 | 高 | train_tg_lora.py | 0.5h |
| REQ-084 (rollback resilience E2E) | 中 — モックでRollbackManager.rollbackを例外送出に設定。train_tg_lora.pyのtry-catch rollback（REQ-076）が既にあるため、E2Eテストのみ追加 | 高 | tests/ | 1h |

**Phase 23完了実績**: TASK-0051~0055全て完了。891テスト→900テスト（+9テスト追加）。推定総工数 8h → 実績 8h

**根拠**: checkpoint.py/batch_iter.py/train_tg_lora.pyの完全読み込み。AI_HUB_MAKE_RUN_FEEDBACK指摘事項の分析。TASK-0051~0055完了コミットで実装検証済み

**信頼性への影響**:
- Phase 23完了後の信頼性: 🔵 65 (+26: 全REQ実装・テスト検証済み), 🟡 7 (不变), 🔴 0 (REQ-081~084全て実装完了)

---

### Phase 27分析: 運用層・チェックポイントシリアライズ・CI パイプライン

**A20: 運用層のアーキテクチャ的位置づけ**

**分析日時**: 2026-05-22
**カテゴリ**: アーキテクチャ
**背景**: TASK-0064で scripts/diagnose.py（369行）と scripts/recover.py（437行）が追加された。これらは学習ループ外で動作する運用スクリプトであり、既存アーキテクチャのどの層にも属さない新規レイヤー

**判断**:
1. **新規「運用層（Operations Layer）」** を `scripts/` 配下に定義。学習・評価・データパイプラインの各層とは独立し、学習ジョブの安定稼働を支援する外部ツール群
2. **diagnose.py**: GPU・チェックポイント・設定・ログの4軸ヘルスチェック。`CheckResult` dataclass で統一された結果形式（ok/warn/error）。CLIフラグで個別/全体チェックを選択可能
3. **recover.py**: 障害パターン（OOM, CUDA error, NaN loss, instability）の自動検出と回復。`RecoveryResult` で構造化された回復結果。sanitize→fix-config→remediateの段階的回復パス
4. **設計原則**: ランブック手順をそのまま実行可能スクリプトに変換（"runbook as code"）。各関数が独立して実行可能（--analyze, --sanitize, --fix-config, --remediate）
5. **依存関係**: 既存コアモジュール（cycle_state, delta_tracker等）には依存せず、torch, yaml, pathlib等の標準的なライブラリのみ使用。モデル学習コードとの結合度を低く保つ設計

**根拠**: scripts/diagnose.py・scripts/recover.pyの完全読み込み。test_diagnose.py（24テスト）・test_recover.py（27テスト）で検証済み

**信頼性への影響**:
- 運用層設計: 新規追加 🔵（実装ベース、テスト51件で検証済み）
- architecture.md: 運用層セクション追加（+2コンポーネント）
- dataflow.md: 診断・障害回復フロー追加（+2フロー）

---

**A21: チェックポイントシリアライズの設計判断**

**分析日時**: 2026-05-22
**カテゴリ**: データモデル
**背景**: TASK-0063/0064で学習ジョブの中断→再開を可能にするため、CycleState・ControllerState・Velocity・DeltaTrackerの完全な状態をシリアライズ・デシリアライズする機能が追加された（REQ-103~105）

**判断**:

1. **TrainingState dataclass**: 4コンポーネント（CycleState + ControllerState + Velocity + DeltaTracker）を統合する単一コンテナ。cycle_offset で中断位置を追跡
2. **シリアライズ方式**: `torch.save()` でPyTorch形式のblobとして保存。JSON形式ではなくPyTorch形式を選択した理由は、Velocity/DeltaTrackerの内部状態にPyTorch tensorが含まれるため
3. **tensor CPU変換**: GPU tensorをシリアライズ前にCPUに移動。`{k: v.cpu() for k, v in state.velocity._state.items()}` で明示的に変換
4. **後方互換性**: `CycleState.from_dict()` は `summary()` キーと旧checkpoint形式キーの両方を受け入れる二重マッピングを実装。`data.get("cycles", data.get("cycle", 0))` のパターン
5. **DeltaTracker再計算**: ロード時に `_compute_stats` で最後の履歴エントリから `_last_stats` を再計算。履歴全体ではなく最終エントリのみで十分（現在のサイクルの統計のみ使用）
6. **保存先**: `runs/<experiment>/` 配下に学習モデルチェックポイントとは別ファイルとして保存。モデルは `save_pretrained()`、状態は `torch.save()` と分離

**根拠**: checkpoint.py TrainingState実装（115行）・cycle_state.py from_dict()（19行）・random_walk_controller.py ControllerState from_dict()（19行）の完全読み込み。test_checkpoint.py（7テスト）・test_fault_recovery.py（19テスト）で検証済み

**信頼性への影響**:
- シリアライズ設計: 新規追加 🔵（実装ベース、テスト26件で検証済み）
- architecture.md: 状態シリアライズ層セクション・チェックポイントシリアライズセクション追加
- dataflow.md: 学習状態シリアライズ・デシリアライズフロー追加

---

**A22: CI パイプラインの設計判断**

**分析日時**: 2026-05-22
**カテゴリ**: 運用性
**背景**: AI_HUB_MAKE_RUN_FEEDBACKが「add a Makefile 'ci' target that runs both with --dry-run or a mock flag to catch import/path regressions early」を推奨。これを受けて Makefile ci ターゲットが追加された（REQ-108）

**判断**:
1. **CI パイプライン構成**: lint（ruff check + format check）→ test（pytest）→ script import check（diagnose.py/recover.py）
2. **スクリプトインポートチェック**: diagnose.py/recover.pyは `scripts/` 配下のstandaloneスクリプトだが、Pythonモジュールとしてもimport可能。`import scripts.diagnose; import scripts.recover` を試行し、失敗時は `importlib.util` でフォールバック。これによりパス解決エラーをCI段階で早期検出
3. **Makefile ターゲット追加**: `ci`（フルパイプライン）, `diagnose`（ヘルスチェック）, `recover`（障害回復）の3ターゲットを追加
4. **設計原則**: CIパイプラインはGPUを必要としない。テストはCPUで実行可能（モックベース）。運用スクリプトもimportチェックのみでGPU不要

**根拠**: Makefile ci target実装・test_diagnose.py・test_recover.pyの完全読み込み

**信頼性への影響**:
- CI設計: 新規追加 🔵（Makefile実装ベース）
- architecture.md: 運用性セクションにCI パイプライン・診断・障害回復を追加
- dataflow.md: CI パイプラインフロー追加

---

### 分析結果サマリー更新

**Phase 27設計更新後**:
- 🔵 青信号: 73 (+8: 運用層 +2, シリアライズ +3, CI +2, テスト数更新 +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- 運用層は既存コアモジュールと低結合で設計されている
- TrainingState シリアライズはPyTorch tensorの特性に適した方式を採用
- CI パイプラインはGPU不要でCPU環境で完全実行可能
- 60テストファイル、1146テストケースが全てパス

**設計方針の決定事項**:
- 運用層の各スクリプトはCLI + Python APIの両方で利用可能
- チェックポイントシリアライズは段階的拡張可能（追加状態フィールドの追加が容易）
- CI は `make ci` の単一コマンドで完結

**残課題**:
- 実際のGPU学習でのTrainingState保存・復元のE2E検証（TASK-0063はモックベースで検証済み）
- 長時間学習でのDeltaTracker履歴サイズ管理（max_historyは実装済みだが実運用での検証が必要）

---

### Phase 29分析: 設計文書とコードベースの整合性回復

**A23: activation_cache.py 重複エントリの解消**

**分析日時**: 2026-05-23
**カテゴリ**: 品質・整合性
**背景**: architecture.md のコアアルゴリズム表で activation_cache.py が2行に重複記載されていた。一方は基本機能（REQ-110~112）、他方はレイヤースキップ詳細（同じREQ-110~112）を記載

**判断**: 2行を1行に統合し、Qwen3.5互換性情報（_get_rotary_emb, _get_layer_types, hybrid attention対応）を併記。評価FLOPs削減効果（約75%）も保持

**根拠**: architecture.md lines 36, 46 の重複確認・activation_cache.py実装の直接読み込み

**信頼性への影響**:
- コンポーネント表の一貫性向上（重複解消）
- Qwen3.5互換性設計情報が 🔵 で正確に記録

---

**A24: run_query.py のユーティリティ層への追加**

**分析日時**: 2026-05-23
**カテゴリ**: アーキテクチャ
**背景**: TASK-0060 で追加された src/utils/run_query.py（108行）が architecture.md のコンポーネント表に未記載。RunMetrics JSONLの履歴クエリAPIを提供

**判断**: ユーティリティ層に run_query.py を追加。モジュール数 7→8 に更新

**根拠**: src/utils/run_query.py の完全読み込み・TASK-0060 定義

**信頼性への影響**:
- ユーティリティ層の完全性が 🔵 で復元

---

**A25: テスト数とファイル数の実態への一致**

**分析日時**: 2026-05-23
**カテゴリ**: 品質・正確性
**背景**: 複数の設計文書でテスト数が実態と乖離:
- architecture.md: "60テストファイル、1145テストケース"
- dataflow.md: "59ファイル, 1128テスト"
- design-interview.md Phase 27: "59テストファイル、1128テストケース"
- 実際: 60テストファイル、1146テストケース（1139 passed + 7 skipped）

**判断**: 全文書のテスト数を実測値に更新:
- architecture.md: "60テストファイル、1146テストケース"
- dataflow.md CI section: "60ファイル, 1146テスト"
- design-interview.md Phase 27: "60テストファイル、1146テストケース"

**根拠**: `pytest --co -q` で1146 collected, `ls tests/test_*.py | wc -l` で60ファイル

**信頼性への影響**:
- テスト数記述の正確性が 🔵 に向上（実測ベース）

---

**A26: CycleState last_valid_loss 追記とQwen3.5互換性**

**分析日時**: 2026-05-23
**カテゴリ**: アーキテクチャ
**背景**: CycleState.from_dict() が last_valid_loss を復元する機能（commit 85589cd）が設計文書に未記載。またactivation_cache.py のQwen3.5 hybrid attention対応がコンポーネント表に未反映

**判断**:
1. CycleState 記述に last_valid_loss 復元を追記
2. activation_cache.py エントリに Qwen3.5 互換性情報（rotary_emb, hybrid attention mask handling）を追記

**根拠**: src/tg_lora/cycle_state.py from_dict() 実装・src/tg_lora/activation_cache.py eval_from_cache() 実装

**信頼性への影響**:
- CycleState 設計: 🔵（実装ベース）
- ActivationCache 設計: 🔵（実装ベース）

---

**A27: Prefix Feature Cache アーキテクチャ分析と設計ギャップ**

**分析日時**: 2026-05-23
**カテゴリ**: アーキテクチャ
**背景**: Phase 31で追加された `prefix_feature_cache`（REQ-126）が dataflow.md に未記載。architecture.md にはモジュール表・構成図に記載済みだが、dataflow観点の分析とテストカバレッジの評価が必要。

**判断**:
1. **dataflow.md に4サブフローを追加**: ビルドフェーズ（forward hook事前計算）、キャッシュ管理（メモリ→ディスク→ビルドの3段階ルックアップ）、推論フェーズ（サフィックス層のみforward）、学習ループ統合（ガード・アクティブ層固定）の全フローを文書化
2. **テストカバレッジ評価**:
   - **十分カバー**: split_layer_idx境界値（0, num_layers拒否）、max_batches制限、position_ids有無、ディスク往復、損失等価性、設定ガード（trainable_lora_scope/dropout）、configスキーマ検証、ディスクヒット統合テスト
   - **改善推奨**: corrupted cacheファイル処理、position_ids経由のbuild_path、force_rebuildフラグの動作確認、ビルド失敗時のmodel.training状態復元、キャッシュ無効化（ハイパーパラメータ変更時のSHA-256差分）の明示的テスト
3. **アーキテクチャ妥当性**: 3段階キャッシュ（メモリ→ディスク→ビルド）は ML研究の反復実験に適した設計。SHA-256ベースのキャッシュキーがハイパーパラメータ変更を自動検知する点は合理的

**根拠**: `src/tg_lora/prefix_feature_cache.py`（272行）、`src/training/train_tg_lora.py`（_maybe_cache_dataset等）、`tests/test_prefix_feature_cache.py`（4テスト）、`tests/test_prefix_feature_cache_extended.py`（8テスト）、`tests/test_training_integration.py`（2テスト）の完全読み込み

**信頼性への影響**:
- dataflow.md: 🔵 +4件（ビルド/キャッシュ管理/推論/学習ループ統合、全て実装ベース）
- design-interview.md: 🔵 +1件（A27分析自体）
- architecture.md: 既存記載で更新不要

---

### 分析結果サマリー更新

**Phase 31設計更新後**:
- 🔵 青信号: 82 (+5: prefix_feature_cache dataflow +4, A27分析 +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**Phase 32設計更新後** (A28 AsyncCacheBuilder):
- 🔵 青信号: 86 (+4: AsyncCacheBuilder dataflow +3, architecture更新 +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- 設計文書のコンポーネント表がソースコード実態と完全に一致
- テスト数カウントが実測値に更新され、将来の乖離が追跡可能
- Qwen3.5互換性要件が明示的に記録
- Prefix Feature Cache の全データフローが実装ベースで文書化完了

**残課題**:
- 実際のGPU学習でのTrainingState保存・復元のE2E検証
- 長時間学習でのDeltaTracker履歴サイズ管理
- Prefix Feature Cache: corrupted cache・force_rebuild・position_ids build path のテスト追加（改善推奨）

---

**A28: AsyncCacheBuilder のアーキテクチャ分析**

**分析日時**: 2026-05-23
**カテゴリ**: アーキテクチャ
**背景**: Phase 32で追加された AsyncCacheBuilder（src/training/async_cache_builder.py, 245行）が2-GPU構成でのバックグラウンドPrefix Feature Cacheビルドを提供。train_tg_lora.pyに統合され、学習をブロックせずにキャッシュを構築可能。architecture.md/dataflow.mdに未反映だった。

**判断**:
1. **architecture.md更新**: 学習ループ層に async_cache_builder.py を追加（8モジュールに増加）。システム構成図にACB（AsyncCacheBuilder）ノードを追加。テスト数を1289に更新
2. **dataflow.md更新**: 主要フロー7a（AsyncCacheBuilder）を新規追加。非同期ビルドフロー図・設定クロスバリデーションフロー図・AsyncCacheBuildResultデータ構造表を記載。全て 🔵（実装ベース）
3. **設計判断**: daemon thread上で2つ目のモデルコピーをロードし、各データセットのキャッシュをシーケンシャルにビルド。thread-safeなlock機構でresults/completed/failed状態を管理。学習ループはpoll()で非ブロッキングに完了確認し、get_result()でキャッシュ済みデータセットを取得してDataLoaderを差し替え
4. **正当性の根拠**: PEFTのLoRA B行列ゼロ初期化により、初期化直後のモデルコピーはベースモデルと同一の出力を生成するため、キャッシュの正確性が保証される

**根拠**: `src/training/async_cache_builder.py`（245行）・`src/training/train_tg_lora.py` 統合部分・`src/training/config_schema.py` バリデータ・`tests/test_async_cache_builder.py`（4テスト）の完全読み込み。コミット eceddf3/43a329a/a316624/75b8032

**信頼性への影響**:
- AsyncCacheBuilder設計: 新規追加 🔵（実装ベース、テスト6件で検証済み: 4 unit + 2 config validation）
- architecture.md: 学習ループ層モジュール数 7→8
- dataflow.md: フロー 🔵件数 +4

---

**A30: REQ-139 AsyncCacheBuilder統合テストギャップ解消**

**分析日時**: 2026-05-23
**カテゴリ**: テスト品質
**背景**: Phase 33のAI_HUB_MAKE_RUN_FEEDBACKで指摘された、AsyncCacheBuilderのモックベースユニットテスト（8件）と実際のトレーニングランタイム動作のギャップ。DataLoader差し替え、ディスク永続化、poll-and-swapパターンのE2E検証が欠落していた。

**判断**:
1. **統合テスト追加**: `tests/test_async_cache_builder_integration.py`（5テスト）を新規作成。CPU上のTinyModelでフルライフサイクルを検証
2. **テストカバレッジ**:
   - `test_full_lifecycle_build_wait_load_on_cpu`: ビルド → DataLoader作成 → バッチ形状検証 → ディスク永続化確認
   - `test_poll_and_swap_pattern_simulates_training`: トレーニングループのpoll-and-swapパターンをシミュレート
   - `test_build_failure_continues_with_raw_dataset`: ビルド失敗時のgraceful degradation検証
   - `test_disk_cache_reuse_skips_rebuild`: 2回目実行時のdisk cache再利用（source='disk'）検証
   - `test_concurrent_poll_and_get_result_are_threadsafe`: 4スレッドからの並行アクセス安全性検証
3. **acceptance-criteria.md更新**: REQ-139~143のPhase 33受け入れ基準（11テストケース）を追加
4. **設計ギャップの解消**: DataLoader差し替え、ディスクキャッシュ永続化、失敗時のフォールバックが全て統合テストで検証済み

**根拠**: AI_HUB_MAKE_RUN_FEEDBACK「Add an integration test exercising the full async cache lifecycle (build on mock GPU → wait → load into trainer)」・test_async_cache_builder.py モックテスト8件・async_cache_builder.py 244行の完全読み込み

**信頼性への影響**:
- AsyncCacheBuilderテスト: モック8件 + 統合5件 = 13件に拡充
- REQ-139: 🔴（ギャップ指摘）→ 🔵（統合テストで検証完了）
- acceptance-criteria.md: Phase 33テストケース11件追加

**分析結果サマリー更新**

**Phase 33設計更新後** (A30 AsyncCacheBuilder統合テスト):
- 🔵 青信号: 91 (+5: REQ-139~143 acceptance criteria +5, A30分析)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- AsyncCacheBuilderのフルライフサイクルがCPU統合テストで検証完了
- DataLoader差し替えパターンがトレーニングループと同一手順で検証済み
- ディスクキャッシュ永続化と再利用がE2Eで確認済み
- 並行アクセス安全性が4スレッド×50回で検証済み

**残課題**:
- 実際のGPU学習でのTrainingState保存・復元のE2E検証
- 長時間学習でのDeltaTracker履歴サイズ管理
- GPT-2 tiny以外のモデルでの統合テスト（優先度低）

---

**A31: Phase 34 in-place ops検証・velocity benchmarks・Makefile統合**

**分析日時**: 2026-05-23
**カテゴリ**: パフォーマンス検証
**背景**: Phase 34（REQ-144~148）で追加されたin-place tensor操作のdata_ptr保存検証（TASK-0079）とvelocity EMA/cap_updateマイクロベンチマーク（TASK-0080）の設計反映。ベンチマークスクリプトは実装済みだが、Makefileターゲット（bench-velocity-ops）とCI回帰閾値が未統合。

**判断**:
1. **data_ptr保存検証完了**（TASK-0079）: velocity.pyのEMA更新（mul_/add_）とextrapolator.pyのcap_update（mul_）がin-place操作後もdata_ptrを維持することを確認。メモリ再確保を排除しGCプレッシャーを低減。
2. **マイクロベンチマーク実装完了**（TASK-0080）: `scripts/benchmark_velocity_ops.py`が1000回反復でEMA更新・cap_update（capping有/無）の実行時間とメモリ使用量を測定。JSON出力対応。
3. **Makefile統合ギャップ**: `bench-velocity-ops`ターゲット（REQ-148）が未追加。既存の`bench-optimizer`/`bench-prefix-cache`パターンに従って追加が必要。
4. **CI回帰閾値未実装**: ベンチマークは観測的（JSON出力のみ）で、閾値比較やCI FAIL機能がない。`--baseline`フラグによる回帰検出を次フェーズで設計。
5. **テスト数同期**: 実測値1344テスト（76ファイル）。設計文書内の1289/1299の記載を実測値に更新。

**根拠**: TASK-0079/TASK-0080完了チェックリスト・scripts/benchmark_velocity_ops.py実装（143行）・Makefile既存bench-*ターゲットパターン・pytest --collect-only実行結果1344テスト

**信頼性への影響**:
- REQ-144~145: 🔵（実装・テスト完了）
- REQ-146: 🔵（test_velocity.py data_ptrテスト5件で検証）
- REQ-147: 🔵（benchmark_velocity_ops.py実装完了）
- REQ-148: 🟡→🔵（Makefileターゲット追加を設計に反映、実装は本更新に含む）
- ベンチマークCI閾値: 🔴→🔵（REQ-149でbench-velocity-ops-ci CI gate要件を明示的に設計。design-interview A31 🔴指摘を解消）

**分析結果サマリー更新**

**Phase 34設計更新後** (A31 in-place ops検証・velocity benchmarks):
- 🔵 青信号: 96 (+5: REQ-144~148設計反映 + A31分析)
- 🟡 黄信号: 8 (+1: REQ-148 Makefile統合)
- 🔴 赤信号: 1 (+1: CI回帰閾値は未設計)

**Phase 35設計更新後** (A32 CI gate):
- 🔵 青信号: 97 (+1: REQ-149 CI gate設計)
- 🟡 黄信号: 7 (-1: REQ-148 🟡→🔵昇格)
- 🔴 赤信号: 0 (-1: CI回帰閾値🔴→REQ-149で解消)

**確認できた事項**:
- in-place opsがdata_ptrを維持し、GCプレッシャーを低減することを検証完了
- velocity EMA・cap_updateのマイクロベンチマークがJSON出力で定量的測定可能
- ベンチマークスクリプトのスモークテスト（--quick JSON出力検証）が緑

**残課題**:
- bench-velocity-ops Makefileターゲットの追加（本更新で実施） → 完了
- CI回帰閾値の設計（--baselineフラグ + 閾値比較 + CI FAIL） → REQ-149で要件化
- 長時間学習でのin-place ops安定性の継続監視

**A32: bench-velocity-ops CI gate設計（REQ-149）**

**分析日時**: 2026-05-23
**カテゴリ**: CI・運用
**背景**: A31で🔴（未設計）として指摘されたCI回帰閾値。benchmark_velocity_ops.pyには--baseline/--save-baseline/--thresholdが完全実装され、TestBaselineRegressionDetection 7テストで検証済み。しかしMakefileにCI gateターゲットが存在せず、デザインギャップが残存していた。AI_HUB_MAKE_RUN_FEEDBACKが「Wire bench-velocity-ops --baseline into CI」を明示的に推奨。

**判断**: 以下の設計判断を行う:
1. **bench-velocity-ops-ci Makefileターゲット**: `--quick --baseline baselines/velocity_ops.json --threshold 20`で実行。CI環境で10反復による高速回帰チェックを実現
2. **チェックインベースライン**: `baselines/velocity_ops.json`をリポジトリにチェックイン。更新は`make bench-velocity-ops --quick --save-baseline baselines/velocity_ops.json`で明示的に行う
3. **CI統合位置**: 既存の`ci`ターゲットには含めない（性能ベンチマークはハードウェア依存のため）。独立した`bench-velocity-ops-ci`ターゲットとして提供し、CI workflowで必要に応じて呼び出す
4. **閾値設計**: デフォルト20%は--quick（10反復）のノイズを許容するゆるい閾値。CI環境での変動を吸収しつつ、大きな性能劣化を検出する

**根拠**:
- benchmark_velocity_ops.py: _compare_with_baseline実装（3メトリクス比較）、exit code 0/1/2
- test_benchmark_velocity_ops.py: TestBaselineRegressionDetection 7テスト（save/load/regression/no-regression/missing-file/threshold/unit）
- Makefile既存パターン: bench-optimizer, bench-prefix-cache
- design-interview A31: 🔴指摘「CI回帰閾値未実装」

**信頼性への影響**:
- REQ-149: 🔵（実装済みの--baseline/--thresholdを活用、Makefileパターンは確立済み）
- A31 CI閾値🔴→🔵解消

---

**A33: Phase 36 LR探索統合・propose→training loop配線の設計検証**

**分析日時**: 2026-05-24
**カテゴリ**: アーキテクチャ・テスト品質
**背景**: Phase 36で追加されたLR探索統合（REQ-150~152）の完全性を検証。lr_explore_prob/lr_log_sigmaパラメータのconfig_schema→controller→training loop配線と、propose()→controller.state.lr反映のE2E検証がAI_HUB_MAKE_RUN_FEEDBACKで推奨されていた。

**判断**:
1. **パラメータ配線の完全性**: config_schema.pyのTGLoRAParams（lr_explore_prob: 0.3, lr_log_sigma: 0.1）→train_tg_lora.py lines 539-540でcontrollerに渡す→random_walk_controller.py propose()でlog-normal walk実行→train_tg_lora.py line 891でproposal.lrをstate.lrに反映。全段階でパラメータが欠落なく伝播される
2. **統合テストの完全性**: TestLrExplorationIntegration 3テストが全てのE2E検証をカバー: (a) configからcontrollerへのパラメータ伝播、(b) 探索lrのstate反映とdeterministic pathからの逸脱、(c) 5サイクルのpropose→accept/rejectフルサイクルでlrが[lr_min, lr_max]内に留まること
3. **設計上の安全性**: lr_explore_prob=0の場合はpropose()がstate.lrをそのまま返すため、探索を無効化可能。lr=0の境界ではlog-normal walkがスキップされ安全。config_schemaのgt=0.0, lt=1.0制約で0.0以下と1.0以上を拒否
4. **設計文書の更新**: architecture.mdにlr_explore_prob/lr_log_sigmaをパラメータ表に追加。dataflow.mdにLR探索log-normal walkフロー図を追加。design-interview.mdにA33分析を追加

**根拠**: config_schema.py TGLoRAParams lines 189-190・random_walk_controller.py propose() lines 265-270・train_tg_lora.py lines 539-540, 891・tests/test_training_integration.py TestLrExplorationIntegration 3テスト（1442 passed + 9 skipped = 1451 collected）の完全読み込み

**信頼性への影響**:
- REQ-150~152: 🔵（パラメータ配線・propose→state反映・統合テスト全て完了）
- architecture.md: パラメータ表 +2件、テスト数更新（77ファイル1451テスト）
- dataflow.md: LR探索フロー図 +1件
- design-interview.md: A33分析 +1件

**分析結果サマリー更新**

**Phase 36設計更新後** (A33 LR探索統合):
- 🔵 青信号: 100 (+3: architecture params +2, A33分析 +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- LR探索パラメータがconfigからtraining loopまで完全に配線されている
- 統合テストがフルサイクル（propose→accept/reject）でLRの非自明な変動を検証
- 全1451テストがパス（77ファイル）
- AI_HUB_MAKE_RUN_FEEDBACKの指摘事項は全て解消済み

---

**A34: Phase 37 Velocity加速度・入力検証・数値安定性の設計分析**

**分析日時**: 2026-05-24
**カテゴリ**: アーキテクチャ・品質
**背景**: Phase 37（REQ-153~159）で追加されたvelocity magnitude acceleration、入力検証強化、数値安定性境界値の設計分析。magnitude_acceleration()が二階微分で不安定/収束を検出し、adapt_to_acceleration()がlr/K調整を行う新規適応ループがtraining loopに統合された。

**判断**:
1. **magnitude_acceleration()の設計妥当性**: magnitude_historyの最近window件から一次差分（slopes）を計算し、その二次差分の平均をaccelerationとする手法は、velocityの増加加速度を定量化する合理的なアプローチ。n<3で0.0を返す設計は統計的有意性の下限を確保
2. **adapt_to_acceleration()の設計**: 正のacceleration（不安定）時にlr decay + K増加、負のacceleration（収束）時にlr boost、ゼロ付近は不変。adapt_to_convergence()とは独立した適応軸として機能し、二重の安定化機構を構成
3. **入力検証強化（EDGE-173~178）**: RollbackManager(max_history=0)のValueError、snapshot_lora_delta()空baseチェック、propose()のOverflowError防止（math.exp引数700クランプ）、cap_update()のNaN/Inf要素数正確報告など、境界値防御が包括的に実装
4. **_compute_stats() @torch.no_grad()**: autogradグラフ構築を防止し、サイクル統計計算時の不要なメモリ消費と性能劣化を排除（REQ-159）

**根拠**: velocity.py magnitude_acceleration() lines 81-98・random_walk_controller.py adapt_to_acceleration() lines 427-462・delta_tracker.py _compute_stats() @torch.no_grad()・rollback_manager.py max_history validation・lora_state.py snapshot_lora_delta() empty check・tests/test_velocity.py TestMagnitudeAcceleration・tests/test_random_walk_controller.py adapt_to_acceleration tests

**信頼性への影響**:
- magnitude_acceleration設計: 🔵（実装・テスト完了、6テストで検証）
- adapt_to_acceleration設計: 🔵（実装・テスト完了、7テストで検証）
- 入力検証強化: 🔵（EDGE-173~178全てテスト検証済み）

---

**A35: Phase 38 加速度適応パラメータの設定サーフェス設計分析**

**分析日時**: 2026-05-24
**カテゴリ**: 設定管理・API設計
**背景**: Phase 38（REQ-160~161）で加速度適応パラメータ（accel_instability_lr_decay, accel_convergence_lr_boost）がYAML設定ファイルから制御可能になった。Pydantic検証による値域制約と、デフォルト値からのカスタマイズが可能な設計。

**判断**:
1. **Pydantic値域制約の妥当性**: accel_instability_lr_decay は (0.0, 1.0) の開区間（decay率として0と1は無意味）、accel_convergence_lr_boost は >1.0（boostとして1.0以下は無意味）の制約。値域の設計根拠が明確
2. **デフォルト値の選択**: decay=0.7（30%減衰、保守的だが極端ではない）、boost=1.1（10%増加、段階的探索）。いずれも既存のlr_accept_boost/lr_reject_decayパターンと整合
3. **Controller側のOptional受け渡し**: コンストラクタ引数をOptional[float] = Noneとし、Noneの場合は_DEFAULT定数を使用。Pydantic検証済みの値が渡されるか、未設定時は安全なデフォルト値が適用される二重防御
4. **設定伝播パス**: YAML → Hydra → config_schema.py（Pydantic検証）→ train_tg_lora.py → RandomWalkController。full pathでの型安全性を確保

**根拠**: config_schema.py TGLoRAParams lines 192-194（Field gt/lt制約）・random_walk_controller.py lines 134-135, 191-200, 424-425（Optional引数とデフォルト定数）・train_tg_lora.py lines 541-542（tg_cfg.get渡し）・tests/test_config_schema.py TestAccelParamConfig（6テスト）・tests/test_random_walk_controller.py test_custom_instability_decay_applied/test_custom_convergence_boost_applied

**信頼性への影響**:
- accel params設計: 🔵（Pydantic検証 + Controller統合 + 8テストで検証）
- config_schema拡張: 🔵（既存パターンに準拠、extra='forbid'で未知フィールド拒否）

---

**A36: Phase 39 --resume障害回復再開・加速度適応観測性の設計分析**

**分析日時**: 2026-05-24
**カテゴリ**: 運用・観測性
**背景**: Phase 39（REQ-162~166）で--resumeフラグによる障害回復再開と、MLflow経由の加速度適応観測性が追加された。直前のPhase 37-38で追加されたmagnitude accelerationとconfigurable accel paramsが、training loopで統合的に動作する設計の完全性を検証。

**判断**:
1. **--resume設計の妥当性**: TrainingState dataclassがcycle_state/controller_state/velocity/delta_tracker/cycle_offsetの5要素を一括保存・復元。cycle_offsetに基づくスキップ機構は、completed cyclesの再実行を防ぎ、損失の二重計上を回避。restore_state()がconfig（candidates/bounds）を保持しつつstate値のみ置換する設計は、再開後に異なるハイパーパラメータ候補で探索を継続することを可能にする
2. **TrainingState永続化の安全性**: velocity/delta_trackerのtensorはcpu()でCPUに移動してから保存。_sanitize_tensors()でNaN/Infを除去し、破損状態の復元を防止。DeltaTrackerのlast_statsは復元時に最新履歴から再計算（_compute_stats）し、統計の整合性を保証
3. **MLflow観測性の設計**: magnitude_acceleration（float）とaccel_action（int: 1/0/-1）をcycle metricsに追加。accel_actionの3値符号化は不安定/安定/収束を一目で判別可能にし、ダッシュボードでのフィルタリングを容易にする。last_accel_actionはadapt_to_acceleration()呼び出しごとに更新され、直前の結果のみを保持（EDGE-185）。acceleration=0.0の場合は0に設定（EDGE-186）
4. **dataflow.mdへの反映**: 新規フロー8（--resume recovery）と加速度適応フローを追加。TrainingState永続化テーブルとacceleration設定フロー図を含め、Phase 37-39のデータフローを完全に文書化

**根拠**: train_tg_lora.py --resume lines 562-576・cycle_offset skip lines 646-648・MLflow metrics lines 1135-1136・checkpoint.py TrainingState dataclass lines 48-56・save/load lines 59-131・random_walk_controller.py restore_state() lines 216-224・last_accel_action line 202・tests/test_fault_recovery.py TestTrainingStateSaveLoad + TestRestoreStateIntegration・tests/test_random_walk_controller.py test_last_accel_action_* 4テスト

**信頼性への影響**:
- --resume設計: 🔵（実装・テスト完了、TestTrainingStateSaveLoad/TestRestoreStateIntegrationでE2E検証）
- MLflow観測性: 🔵（実装完了、4テストでlast_accel_action遷移を検証）
- dataflow.md: 🔵件数 +6（--resume flow, acceleration flow, config flow, テーブル等）

**分析結果サマリー更新**

**Phase 39設計更新後** (A34~A36):
- 🔵 青信号: 108 (+8: A34分析+2, A35分析+2, A36分析+2, dataflow +6から+2重複除外)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- Phase 37-39の全要件が設計文書に反映完了
- --resume recoveryのフルデータフローが文書化完了
- 加速度適応の設定→実行→観測のE2Eパスが検証済み
- MLflowダッシュボードで加速度適応の効果が可視化可能

**残課題**:
- --resumeのE2E統合テスト（save → interrupt → resume → verify loss decreasing）
- 実際のGPU学習でのTrainingState保存・復元のE2E検証
- TruthfulQA外部ベンチマークでのTG-LoRA品質改善検証（delta -0.00045 acc）

**A37: Phase 42 Accel param sweep infrastructure検証と実験分離設計**

**分析日時**: 2026-05-24
**カテゴリ**: 実験設計・運用品質
**背景**: TASK-0092で作成されたaccel param実験config群（4ファイル）とsweepスクリプト（run_accel_sweep.sh）の健全性を検証。AI_HUB_MAKE_RUN_FEEDBACKが「verify scripts/compare_runs.py exists and aligns with the sweep script's invocation signature」を推奨。

**判断**:
1. **sweep↔compare_runs.py インテグレーションバグ（修正済み）**: run_accel_sweep.shが複数`--tg-lora`引数を`compare_runs.py`に渡していたが、compare_runs.pyのargparseは単一値のみ受け入れる（`nargs='+'`や`action='append'`なし）。最後の`--tg-lora`のみが使用され、中間configが黙ってドロップされる状態だった。修正: no-accel baselineに対するpairwise比較に変更し、各treatment configを個別に比較。dashboard overviewも追加。
2. **実験分離設計の妥当性**: 全4configで以下を統一し、accel paramsのみを変動:
   - `enable_random_walk: false`（random walk探索を無効化、確定的な比較を保証）
   - `enable_convergence_adaptation` フィールドなし（旧adaptive branchから完全分離）
   - `lr_explore_prob: 0.0`（LR探索を無効化）
   - 同一のK/N/alpha/beta/lr初期値・候補リスト
   - 同一のlayer strategy（last_25_percent, force_top_layers_only: true）
   - 同一のseed: 42、max_cycles: 500
   - 唯一の変動軸: `accel_instability_lr_decay` と `accel_convergence_lr_boost`
3. **no-accel baseline設計**: `accel_no_accel.yaml`は完全無効化ではなく、near-identity値（decay=0.99, boost=1.01）を採用。コードパスを完全に同一に保ちつつaccel effectを実質的にゼロにする設計。これにより、if分岐の差異による副作用を排除
4. **実験configパラメータマトリクス**:

   | Config | decay | boost | 効果 |
   |--------|-------|-------|------|
   | conservative | 0.3 | 1.1 | 強い保守シフト + 控えめ回復 |
   | balanced | 0.5 | 1.5 | バランス型 |
   | aggressive | 0.9 | 2.0 | 弱い保守シフト + 強い回復 |
   | no_accel | 0.99 | 1.01 | ablation基準線（near-identity） |

**根拠**: run_accel_sweep.sh元実装・compare_runs.py argparse lines 761-762・4config YAML全フィールド比較・random_walk_controller.py adapt_to_acceleration() lines 427-462

**信頼性への影響**:
- sweep infrastructure: 🟡→🔵（インテグレーションバグ修正、pairwise比較で正確な結果を保証）
- 実験分離設計: 🔵（単一変動軸、全confounder統一）
- 次フェーズ: TASK-0094でsweep実行→結果分析→config反復の設計を追加

---

**分析結果サマリー更新**

**Phase 42設計更新後** (A37 sweep infrastructure検証):
- 🔵 青信号: 112 (+4: sweep修正 +1, 実験分離設計 +2, A37分析 +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- sweep↔compare_runs.pyのインテグレーションバグを修正（pairwise比較 + dashboard overview）
- accel param実験の分離設計が単一変動軸で正しく実装されていることを確認
- no-accel baselineのnear-identity設計がコードパス同一性を保証することを確認
- 次ステップ: TASK-0094で実際のsweep実行と結果分析を実施

**残課題**:
- 実際のGPUでのsweep実行と結果収集
- 結果に基づくaccel param最適値の特定
- 最適configでのTruthfulQA品質ギャップの再検証

---

**A38: Phase 43 収束適応のrandom walkフラグからの分離**

**分析日時**: 2026-05-24
**カテゴリ**: アーキテクチャ・設計判断
**背景**: adapt_to_convergence()がenable_random_walkフラグに依存していたため、random walk探索を無効化した確定的実験（enable_random_walk=false）で収束適応も同時に無効化されていた。収束適応は探索とは独立した安定化機構として機能すべきであり、両者の結合は設計上の誤りだった。

**判断**:
1. **分離の妥当性**: adapt_to_convergence()はproactiveなlr減衰+K増加の適応であり、random walkのK/N/alpha/beta探索とは独立した関心事。enable_random_walk=falseの確実実験でも収束停滞を検出してlrを下げることは有効な安定化戦略
2. **後方互換性**: enable_convergence_adaptation=Trueがデフォルトのため、既存の動作は変更なし。明示的にfalseを指定した場合のみ収束適応が無効化される
3. **テストカバレッジ**: test_adapt_to_convergence_can_be_disabled_independently（独立無効化）、test_adapt_to_convergence_active_when_random_walk_disabled（random walk無効時の動作）で検証

**根拠**: random_walk_controller.py lines 440-463 adapt_to_convergence()・tests/test_random_walk_controller.py lines 316-361・commit 5a2ecb3

**信頼性への影響**:
- 収束適応設計: 🔵（実装・テスト完了、独立フラグで制御可能）
- architecture.md・dataflow.mdに反映

---

**A39: Phase 43-44 analyze_accel_sweep拡張の設計分析**

**分析日時**: 2026-05-24
**カテゴリ**: 運用・解析パイプライン
**背景**: analyze_accel_sweep.pyが3つの新機能（収束軌跡分析・受理率トラッキング・JSONL二重解析排除）と2つの検証機能（sweep結果検証・sweep config検証）で大幅に拡張された。スクリプトが約390行に成長し、単純なanalysis scriptからsweep検証ツールに進化。

**判断**:
1. **JSONL二重解析排除**: augmentation loopでparse_jsonl()を1回呼び出し、結果を`_parsed_records`に格納して_accept_rate()・augmentation・trajectory分析で再利用。以前は_accept_rate()が別途parse_jsonl()を呼び出していたため、同一JSONLファイルの二重I/Oが発生していた
2. **収束軌跡分析（compute_loss_trajectory）**: 線形回帰slope・plateau検出（5-step window）・convergence speed（前25%の改善割合）・half-reduction cycleの4メトリクスを計算。実験間の収束特性を定量的に比較可能
3. **受理率トラッキング（_accept_rate）**: _parsed_recordsからtg_lora_accepted=Trueのstepを計数。baseline vs treatment間のaccept rate差分も計算し、実験パラメータが受理率に与える影響を可視化
4. **スイープ結果検証（validate_sweep_results）**: header/footer欠損・NaN/Inf loss・loss爆発（>10x初期loss）を自動検出。不完全なrunの混入を防止
5. **スイープconfig検証（validate_sweep_configs）**: Pydantic validation + 制御変数同一性確認。decay/boost以外のパラメータが実験間で一致することを保証（実験分離の検証）

**根拠**: scripts/analyze_accel_sweep.py 572行・tests/test_analyze_accel_sweep.py・commits 976bf51/abd87c7/25e6a7e

**信頼性への影響**:
- sweep分析パイプライン: 🔵（実装・テスト完了、二重解析排除でI/O効率化）
- 収束軌跡分析: 🔵（線形回帰+plateau検出で定量的比較実現）
- スイープ検証: 🔵（実験分離の自動検証で信頼性担保）
- architecture.mdにanalyze_accel_sweep.py追加、dataflow.mdにsweep分析フロー追加

---

**A40: Phase 43-44 accel bounds境界値テストの包括性**

**分析日時**: 2026-05-24
**カテゴリ**: テスト品質
**背景**: accel param bounds validation（REQ-160~161）の境界値テストが不足していた。正常値のテストは存在したが、min/max境界値・負値・None/NaN・ゼロ幅range等のエッジケースが未検証。

**判断**:
1. **境界値テスト**: accel_instability_lr_decay（0拒否、0.99受理、1.0拒否）、accel_convergence_lr_boost（1.0拒否、1.01受理）の境界値を明示的にテスト
2. **負値テスト**: alpha_initial/lr_initial/K_initial等の主要パラメータで負値を拒否
3. **min/max境界**: alpha_initial=alpha_min受理、alpha_initial=alpha_max受理、lr_initial=lr_min/lr_max受理を検証
4. **ゼロ幅・反転range**: alpha_min==alpha_max（ゼロ幅）受理、alpha_min>alpha_max（反転）拒否
5. **None値処理**: Optional accel paramsでNone渡し時にデフォルト値が適用されることを確認

**根拠**: tests/test_random_walk_controller.py lines 1843-2165・commit 25e6a7e

**信頼性への影響**:
- accel bounds検証: 🔵（境界値・負値・None・ゼロ幅・反転rangeの全ケースをテスト）
- テスト件数: +186件のエッジケーステスト追加（1979テストケース）。Phase 48: +18テスト（2025テストケース）

---

### 分析結果サマリー更新

**Phase 43-47設計更新後** (A38~A40):
- 🔵 青信号: 119 (+7: 収束適応分離 +2, sweep分析パイプライン +3, accel boundsテスト +2)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- 収束適応がrandom walkフラグから独立し、確実実験でも有効化可能
- sweep分析パイプラインが収束軌跡・受理率・スイープ検証を統合
- JSONL二重解析排除でI/O効率化
- accel bounds境界値の全エッジケースがテスト検証済み
- 1979テストケースが全てパス

---

**A41: Phase 48 compare_paper_memory_modes.py テストカバレッジ拡張**

**分析日時**: 2026-05-25
**カテゴリ**: テスト品質
**背景**: AI_HUB_MAKE_RUN_FEEDBACKが「scripts/compare_paper_memory_modes.py (300 lines) is a new utility script but has no corresponding tests」を指摘。前回実行で3件のテストが追加されていたが、_load_summary（2形式対応）、_relative_delta（エッジケース）、_series_mean（型変換・欠損）、_render_markdown（テーブル出力）、main()（統合）のテストが未カバーだった

**判断**:
1. **18テスト追加**: 4テストクラス + 1統合テストを新規追加
   - `TestRelativeDelta` (5テスト): 正/負のdelta、ゼロ基準、None入力、両None
   - `TestSeriesMean` (5テスト): float/int抽出、キー不在、None/string値
   - `TestLoadSummary` (3テスト): aggregate形式、legacy benchmark形式、不正形式ValueError
   - `TestRenderMarkdown` (4テスト): 集計テーブル、per-seedセクション、空データ、None値ダッシュ
   - `TestMainIntegration` (1テスト): main()でJSON+Markdown書き出しのE2E検証
2. **テストカバレッジ**: 全5関数（_load_summary, _relative_delta, _series_mean, _render_markdown, main）をカバー
3. **テスト件数**: 3→21（+18）。テストスイート全体: 2007→2025 passed (+18), 7 skipped

**根拠**: AI_HUB_MAKE_RUN_FEEDBACK指摘・compare_paper_memory_modes.py 300行の完全読み込み

**信頼性への影響**:
- compare_paper_memory_modes.py テスト: 🔵（全パス確認済み）
- テストスイート全通過確認: 2025 passed, 7 skipped, 1 warning

---

### 分析結果サマリー更新

**Phase 48設計更新後** (A41 テスト拡張):
- 🔵 青信号: 122 (+3: compare_paper_memory_modes.py script entry, テスト拡張, A41分析)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- compare_paper_memory_modes.py の全関数がテストカバーされた
- テストスイート2025テスト全通過を確認
- parametrizeリファクタリング（test_config_schema.py, test_random_walk_controller.py）がカバレッジを維持

**残課題**:
- 実際のGPU学習でのTrainingState保存・復元のE2E検証
- 長時間学習でのDeltaTracker履歴サイズ管理
- llm-wiki生成ノイズの除外（.gitignoreまたはmake-runコミット制御）

---

**A42: Phase 50 Paper Gate評価自動化設計**

**分析日時**: 2026-05-25
**カテゴリ**: アーキテクチャ / Paper実験パイプライン
**背景**: docs/paper_experiment_plan.mdで定義されたGate G0–G4の判定を自動化し、Stage 2マルチシード複製の実行→評価→意思決定を一環して行う必要がある。手動判定では実験結果の解釈に一貫性がなく、Claim Ladderの格下げ判断が遅れるリスクがあった。

**判断**:
1. **evaluate_paper_gates.py**: aggregate_summary.jsonを読み込み、Gate G0–G4を自動判定。各Gateのpass/fail条件をpaper_experiment_plan.mdの定義通りにコード化（REQ-179~183）
2. **Gate判定の分離**: G0（Hygiene）→G1（Internal Efficiency）→G2（Memory Frontier）は順次評価。G3/G4は外部評価・ablationが必要なためinformational出力に留める
3. **CLI統合**: exit code 0/1/2（全pass/1つ以上fail/入力エラー）でCIパイプラインへの統合を可能にする
4. **Makefile統合**: `paper-memory-evaluate-gates`ターゲットでGATE_SUMMARY等のパラメータをカスタマイズ可能（REQ-184）
5. **Stage 2実行コマンド固定**: `make paper-memory SEEDS='42 43 44' TARGET_BP=240 MAX_SEQ_LEN=1024`をcanonical Stage 2コマンドとして文書化

**根拠**: paper_experiment_plan.md Gate定義・evaluate_paper_gates.py実装・run_paper_memory_suite.sh出力構造・REQ-179~184要件定義

**信頼性への影響**:
- Paper Gate評価: 🔵（実装済み・テスト済み）
- Stage 2実行パイプライン: 🔵（run_paper_memory_suite.sh実装済み・smoke検証済み）
- architecture.mdにPaper実験パイプラインセクション・Gate評価アーキテクチャ・実験サーフェス・Stage 2パラメータを追加
- dataflow.mdにFlow 10（マルチシードスイート実行）・Flow 11（Gate評価パイプライン）・Stage 2決定木を追加

---

**A43: Stage 2マルチシード複製の実行準備完了判定**

**分析日時**: 2026-05-25
**カテゴリ**: 実験設計 / Stage 2
**背景**: AI_HUB_MAKE_RUN_FEEDBACKが「Run Stage 2 multi-seed replication using the consolidated smoke references as baselines」を最高優先度として指示。Stage 2実行に必要な全コンポーネントが揃っているかを検証する。

**判断**: Stage 2実行に必要な全コンポーネントが実装・テスト済み:

| コンポーネント | 状態 | 根拠 |
|---------------|------|------|
| Canonical baseline config | ✅ | `configs/9b_baseline_suffix_only_last25.yaml` |
| Canonical treatment config | ✅ | `configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml` (prefix_feature_cache_train=true) |
| Multi-seed suite runner | ✅ | `scripts/run_paper_memory_suite.sh` (175行) |
| Aggregate summary generator | ✅ | suite内Python集約スクリプト |
| Gate evaluator | ✅ | `scripts/evaluate_paper_gates.py` |
| Smoke evidence (one-shot) | ✅ | `runs/paper_memory_one_shot_offload_smoke_v2/aggregate_summary.json` |
| Smoke evidence (reuse) | ✅ | `runs/paper_memory_reuse_offload_smoke_v1/aggregate_summary.json` |
| Mode comparison | ✅ | `runs/paper_memory_offload_mode_compare_smoke_v2.json` |
| Makefile targets | ✅ | `paper-memory`, `paper-memory-evaluate-gates` |
| CI baseline test | ✅ | TASK-0104で安定化済み |

**根拠**: docs/paper_experiment_plan.md Stage 2定義・run_paper_memory_suite.sh全175行・evaluate_paper_gates.py全200行以上・Makefile paper-memoryターゲット・paper_results_snapshot.md

**信頼性への影響**:
- Stage 2実行準備: 🔵（全コンポーネント実装・テスト済み）
- 残作業: GPU計算リソースの確保と実際のmulti-seed実行のみ。コード変更は不要

---

### 分析結果サマリー更新

**Phase 50設計更新後** (A42~A43):
- 🔵 青信号: 128 (+6: Paper Gate評価自動化 +3, Stage 2実行準備判定 +3)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- Gate G0–G4自動評価パイプラインが完全実装
- Stage 2 multi-seed実行に必要な全コンポーネントが揃っている
- Smoke evidenceがone-shot/reuse両方でconsolidated
- CI gate baselineテストが安定化（TASK-0104）
- 2044テストケースが収集される

**残課題**:
- Stage 2の実際のGPU実行（コード変更不要、計算リソースの確保のみ）
- Stage 3以降のfrontier sweep実行
- paper_results_snapshot.mdはwrite-once per experiment roundとして扱う

---

**A44: Phase 53 Runtime Prefix Offload要件ギャップと補助スクリプトの設計分析**

**分析日時**: 2026-05-25
**カテゴリ**: アーキテクチャ / 要件整合性
**背景**: Phase 53の要件整合性確認で、`prefix_runtime_offload.py`が本番で使用されているが要件が未定義、補助スクリプト（precompute_prefix_cache_parallel.py、benchmark_prefix_cache.py）も要件に未記載だった。interview-record.md A40で指摘・修正済みだが、design-interview.mdに未反映だった。

**判断**:
1. **Runtime Prefix Offload**: `prefix_runtime_offload.py`（75行）はtrain_tg_lora.py:577で呼び出され、学習開始時にprefix層をCPUにオフロードしてVRAMを解放。config_schema.pyに`prefix_runtime_offload_valid`バリデータが存在し、offload=true + experimental=falseを拒否。テストも実装済み（test_prefix_runtime_offload.py）
2. **Parallel Cache Precomputation**: `precompute_prefix_cache_parallel.py`（435行）はrank-sharded並列事前計算。DDP/NCCLを使わない軽量設計で、各プロセスが独立してデータシャードを処理
3. **Benchmark Prefix Cache**: `benchmark_prefix_cache.py`（231行）はcold/warm両パスの性能ベンチマーク。既存compare_runs.py/run_comparison.shパターンを再利用
4. **Phase 51-52タスク**: TASK-0107~0111がStage 3-5自動化のために定義済み。 frontier sweep（TASK-0107）、外部品質評価（TASK-0108）、因果分析（TASK-0109）、結果統合（TASK-0110）、Smoke検証（TASK-0111）

**根拠**: prefix_runtime_offload.py完全読み込み・precompute_prefix_cache_parallel.py・benchmark_prefix_cache.py・interview-record.md A40・requirements.md REQ-193~197

**信頼性への影響**:
- architecture.md: 重複エントリ解消 + 補助スクリプト2件追加
- dataflow.md: Flow 12~14（Runtime Prefix Offload + Parallel Cache + Benchmark）追加
- design-interview.md: A44分析追加

### 分析結果サマリー更新

**Phase 53設計更新後** (A44 Runtime Prefix Offload + 補助スクリプト):
- 🔵 青信号: 134 (+6: A44分析 +1, architecture補助スクリプト +2, dataflow Flow 12~14 +3)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- prefix_runtime_offload.pyの要件・実装・テストが完全に整合
- 補助スクリプトの要件（REQ-196/197）が定義済み
- Phase 51-52タスクがStage 3-5自動化のために適切に定義されている
- dataflow.mdに3つの新フローが追加され、Phase 53の全データフローが文書化完了

**残課題**:
- TASK-0107~0111の実装（Stage 3-5自動化コード）
- Stage 2の実際のGPU実行とGate評価

## 関連文書

- **アーキテクチャ設計**: [architecture.md](architecture.md)
- **データフロー**: [dataflow.md](dataflow.md)
- **要件定義**: [requirements.md](requirements.md)
- **ユーザストーリー**: [user-stories.md](user-stories.md)
- **受け入れ基準**: [acceptance-criteria.md](acceptance-criteria.md)
- **実装分析記録**: [interview-record.md](interview-record.md)
- **APIリファレンス**: [docs/api_reference.md](../../docs/api_reference.md)（REQ-109）

### Phase 55分析: 未文書化スクリプト統合とテスト数更新

**A45: 運用層スクリプトの文書化ギャップ**

**分析日時**: 2026-05-25
**カテゴリ**: 品質・文書整合性
**背景**: architecture.mdの運用層テーブルがscripts/の全Pythonスクリプト・Shellスクリプトを網羅していなかった。またテスト数が94ファイル2032テストケース（Phase 43-47時点）から古くなっていた。

**判断**:
1. **未文書化Pythonスクリプト**: 3スクリプトがarchitecture.mdに未記載だった
   - `summarize_sweep.py`（200行）: スイープ結果要約・効率メトリクス・ランキング
   - `analyze_prefix_cache_break_even.py`（148行）: prefix cache損益分岐点分析
   - `generate_sweep_dashboard.py`（219行）: HTMLダッシュボード生成
2. **未文書化Shellスクリプト**: 8スクリプトが未記載だった
   - `run_sweep.sh`, `run_ablation_suite.sh`, `run_high_lr_comparison.sh`, `run_kstep_rollback_test.sh`, `run_best_config_eval.sh`, `run_accel_sweep_parallel.sh`, `run_accel_sweep_auto.sh`, `run_remaining_accel_configs.sh`
3. **テスト数更新**: 94ファイル2032テスト → 98ファイル2098テスト（+4ファイル、+66テスト）
4. **dataflow.md追加**: Flow 15（Frontier Sweep Pipeline）とFlow 16（Sweep結果分析パイプライン）を追加

**根拠**: `ls scripts/` と `pytest --co -q` による現状確認

**信頼性への影響**:
- 運用層スクリプトの文書化: 🔵（実装ベース）
- テスト数正確性: 🔵（CI出力ベース）
- dataflow Flow 15/16: 🔵（既存スクリプト実装ベース）

---

### 分析結果サマリー更新

**Phase 55設計更新後** (A45 未文書化スクリプト統合):
- 🔵 青信号: 140 (+6: A45分析 +1, architecture 運用層スクリプト +3, dataflow Flow 15/16 +2)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- scripts/の全PythonスクリプトとShellスクリプトがarchitecture.mdに網羅記載完了
- テスト数が最新（98ファイル、2098テスト）に更新
- dataflow.mdにfrontier sweepパイプラインとsweep分析パイプラインのデータフローが追加
- 全設計文書がコードベースの現状と整合

**残課題**:
- Stage 2の実際のGPU実行とGate G1評価（seed 44完了待ち）
- G1 pass後のseq-len frontier sweep（Stage 3）の実行
- test_frontier_report.py（450行）のパラメータ化テスト/共有フィクスチャによる保守性改善検討

---

### Phase 56分析: モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析

**A46: Phase 56要件の設計文書統合ギャップ**

**分析日時**: 2026-05-25
**カテゴリ**: 品質・文書整合性・要件網羅性
**背景**: Phase 56（REQ-218~231）で14件の要件が追加されたが、architecture.md・dataflow.md・design-interview.mdに未反映だった。inspect_model.pyとcompare_runs.pyは運用層テーブルに未記載、4つのデータフロー（モデル検査・ダッシュボード・ワンショットキャッシュ・損益分岐点分析）が欠落していた。

**判断**:
1. **運用層テーブル追記**（architecture.md）:
   - `inspect_model.py`（257行）: モデル構造検査・LoRA互換ターゲット自動発見（REQ-218）
   - `compare_runs.py`（860行）: マルチランダッシュボード・5種可視化プロット・Markdownレポート・MLflow連携（REQ-220~223）
2. **設定サーフェス追記**（architecture.md）:
   - `configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml`（REQ-225）
   - `configs/9b_baseline_suffix_only_last25.yaml`（REQ-231）
3. **Makefileターゲット追記**（architecture.md）:
   - モデル検査（REQ-218~219）、損益分岐点（REQ-226~227）、データ細粒度（REQ-228）、クリーンアップ（REQ-229）、キャッシュモード比較（REQ-230）
4. **コアモジュール更新**（architecture.md）:
   - `prefix_feature_cache.py`にMappedPrefixFeatureDataset（one_shot/disk-backed）の記載を追加
5. **データフロー4件追加**（dataflow.md）:
   - Flow 17: モデル構造検査パイプライン（inspect_model.py）
   - Flow 18: マルチラン比較ダッシュボード（compare_runs.py dashboard）
   - Flow 19: ワンショットPrefix Feature Cache（MappedPrefixFeatureDataset）
   - Flow 20: Prefix Cache損益分岐点分析（analyze_prefix_cache_break_even.py）
6. **信頼性サマリー更新**:
   - architecture.md: 65→69件（+4）、99%青信号
   - dataflow.md: 77→81件（+4）、100%青信号

**根拠**:
- `specs/tg-lora/requirements.md` REQ-218~231（Phase 56要件定義）
- `scripts/inspect_model.py`（257行）・`scripts/compare_runs.py`（860行）・`src/tg_lora/prefix_feature_cache.py`（466行）・`scripts/analyze_prefix_cache_break_even.py`（149行）の既存実装
- `configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml`・`configs/9b_baseline_suffix_only_last25.yaml`の既存設定
- Makefile inspect/inspect-config/analyze-prefix-break-even/download-dolly/download-capybara/prepare-data-small/prepare-capybara/clean/clean-data/clean-runs/compare-prefix-cold/warm/coldwarmターゲット

**信頼性への影響**:
- Phase 56要件は全て既存実装に基づくため、設計更新の信頼性は🔵（青信号）
- inspect_model.pyとcompare_runs.pyの運用層追記: 🔵（実装ベース）
- 4データフロー追加: 🔵（既存スクリプト実装ベース）
- ワンショットキャッシュモード: 🔵（MappedPrefixFeatureDataset実装ベース）
- 新規推定なし（全て既存コードから導出）

---

### 分析結果サマリー更新

**Phase 56設計更新後** (A46 モデル検査・比較ダッシュボード・ワンショットキャッシュ・コスト分析):
- 🔵 青信号: 150 (+10: A46分析 +1, architecture 運用層 +2, architecture 設定サーフェス +2, architecture Makefile +5, dataflow Flow 17~20 +4, but -4 counted in architecture sub-items)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- Phase 56要件（REQ-218~231）がarchitecture.md・dataflow.mdに完全統合
- inspect_model.py・compare_runs.pyが運用層テーブルに記載完了
- ワンショットキャッシュモード（MappedPrefixFeatureDataset）がコアモジュールテーブルに反映
- Makefileターゲット（inspect/break-even/データ細粒度/クリーンアップ/キャッシュ比較）が記載完了
- dataflow.mdに4つの新フロー（Flow 17~20）が追加され、Phase 56の全データフローを網羅
- 全設計文書がコードベースの現状と整合（REQ-218~231全要件対応）

**残課題**:
- TASK-0112/0113のテスト実装（acceptance criteria TC-218~231）

---

### Phase 57-58分析: 論文実験統計分析強化・学習品質モニタリング

**A47: Phase 57-58要件の設計文書統合ギャップ**

**分析日時**: 2026-05-25
**カテゴリ**: 品質・文書整合性
**背景**: Phase 57（REQ-241~244: 論文結果エクスポート・感度分析）とPhase 58（REQ-245~250: サイクルモニタ・実験比較）で10件の要件が追加され、4モジュールが実装されたが、dataflow.mdにPhase 59-61のデータフローが未反映だった。

**判断**:
1. **Phase 57モジュール**: export_paper_results.py（LaTeX/Markdown/CSV出版テーブル生成）とanalyze_sensitivity.py（Pearson相関・ランキング感度分析）はarchitecture.mdの運用層テーブルに既に記載済み
2. **Phase 58モジュール**: cycle_monitor.py（CycleMonitor・発散・停滞検知）とcompare_experiment_configs.py（実験構成マトリクス比較）もarchitecture.mdに記載済み
3. **dataflow.md ギャップ**: Phase 59-61のデータフロー図（Flow 21-23）が未追加だったため追加:
   - Flow 21: 学習軌跡分析パイプライン（trajectory.py + analyze_trajectory.py CLI）
   - Flow 22: 軌跡連動適応制御パイプライン（trajectory_controller.py）
   - Flow 23: Training Advisor パイプライン（training_advisor.py + advise_training.py CLI）
4. **信頼性サマリー更新**:
   - dataflow.md: 81→84件（+3 Flow 21-23）、100%青信号

**根拠**:
- `specs/tg-lora/requirements.md` REQ-241~258（Phase 57-61要件定義）
- `src/tg_lora/cycle_monitor.py`（DivergenceReport, StagnationReport, HealthReport）
- `src/tg_lora/trajectory.py`（TrajectoryAnalyzer, ConvergenceEstimate, EarlyStopAdvice, TrajectoryReport）
- `src/tg_lora/trajectory_controller.py`（TrajectoryController, CycleDecision）
- `src/tg_lora/training_advisor.py`（TrainingAdvisor, AdvisoryAction, AdvisoryReport, AdvisorConfig）
- `scripts/analyze_trajectory.py`・`scripts/advise_training.py` CLI実装
- `specs/tg-lora/tasks/overview.md` Phase 59-61完了確認

**信頼性への影響**:
- Phase 57-61要件は全て既存実装に基づくため、設計更新の信頼性は🔵
- 3データフロー追加: 🔵（既存モジュール・CLI実装ベース）
- 新規推定なし（全て既存コードから導出）

---

### 分析結果サマリー更新

**Phase 57-61設計更新後** (A47 軌跡分析・適応制御・Training Advisorのデータフロー統合):
- 🔵 青信号: 154 (+4: A47分析 +1, dataflow Flow 21~23 +3)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- Phase 57-61要件（REQ-241~258）の主要モジュールがarchitecture.mdに記載済みであることを確認
- Phase 59-61のデータフロー図（Flow 21~23）を追加し、軌跡分析→適応制御→Training Advisorの完全なデータフローを網羅
- 全設計文書がPhase 57-61の実装と整合

**残課題**:
- TASK-0119.mdとTASK-0121.mdのタスクファイルがoverview.mdで参照されているが実ファイルが存在しない（実装は完了しているため、タスク仕様ファイルの補完が必要）
- Training Advisory Pipelineの学習ループへのリアルタイム統合（現在は事後分析CLIのみ）
- 統合テストの追加: 合成的 run_metrics.jsonl を生成し、advise_training.py → AdvisoryReport のE2Eパスを検証するテストが未整備
- Stage 2の実際のGPU実行とGate G1評価（seed 44完了待ち）

---

### A48: compare_runs.py 構造化parse_warnings収集の設計反映

**分析日時**: 2026-05-25
**カテゴリ**: アーキテクチャ・データフロー
**背景**: REQ-037a（compare_runs.pyのparse_warnings構造化収集・compare_experiment_configs.pyとのパターン統一）がrequirements.mdに追加されたが、architecture.md・dataflow.md・design-interview.mdの3ファイルに未反映だった。

**判断**: REQ-037aを3ファイルに反映:
- **architecture.md**: compare_runs.pyのモジュール説明に「構造化parse_warnings収集（REQ-037a）」を追加
- **dataflow.md**: Flow 18（マルチラン比較ダッシュボード）にparse_warnings収集・表示フローを追加。gather_runs()→parse_warningsリスト収集→render_dashboard()/format_json()での条件付き表示。compare_experiment_configs.pyのExperimentSummary.parse_warningsと同一パターンであることを明記
- **design-interview.md**: 本分析記録（A48）を追加

**根拠**:
- `specs/tg-lora/requirements.md` REQ-037a定義
- `scripts/compare_runs.py` gather_runs() L598-618（parse_warnings収集）、render_dashboard() L734-740（Rich Panel表示）、format_json() L746-749（条件付きJSON出力）
- `scripts/compare_experiment_configs.py` ExperimentSummary.parse_warnings dataclass（L33）、同一のtry/except収集パターン（L90）

**信頼性への影響**:
- 3ファイルの更新は全て既存実装ベースのため 🔵
- parse_warningsパターンの統一性確認により、compare_runs.pyとcompare_experiment_configs.pyの設計一貫性が 🔵 に向上

---

### 分析結果サマリー更新

**REQ-037a設計反映後** (A48 compare_runs.py parse_warnings設計ギャップ補完):
- 🔵 青信号: 157 (+3: A48分析 +1, architecture +1, dataflow +1)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- REQ-037aが要件定義に追加済みであることを確認
- compare_runs.pyとcompare_experiment_configs.pyのparse_warnings収集パターンが統一されていることを実装コードから確認
- 設計上の意図的分割を確認: dashboardモード（gather_runs, parse_warnings収集・graceful handling）とlegacy compareモード（load_run, 厳密パース・エラー伝播）は別コードパス。generate_markdown_report()はlegacy compareパス専用であり、parse_warnings非対応は意図的設計（2ファイルの明示的指定時はmalformedでエラー終了が適切）

**残課題**:
- TASK-0119.mdとTASK-0121.mdのタスクファイル補完
- Training Advisory Pipelineの学習ループへのリアルタイム統合
- Stage 2の実際のGPU実行とGate G1評価

---

### A49: Phase 62 PSA (Prior-based Subspace Amplification) 設計反映

**分析日時**: 2026-06-10
**カテゴリ**: アーキテクチャ・データモデル
**背景**: Phase 62でPSAパイプライン（psa.py, regime.py, activation_regime.py, weight_averaging.py, layer_delta_analysis.py）が追加され、TG-LoRAのメインライン研究方向が外挿ベースからPSAに転換した。アーキテクチャ・データフロー設計文書への反映が必要。

**判断**:
1. **PSAパイプラインはコアアルゴリズム層に追加**: PSAPrior勾配増幅が学習ループのbackward→optimizer.step()間に挿入される新たなメインライン処理。既存の外挿（extrapolator.py）は設定で切り替え可能なサーフェスとして残存（REQ-284）
2. **RegimeDetectorはPSAと強結合**: consume_reset_signal()のワンショット消費パターンがPSAのregime-aware prior resetと直接連携。独立した診断モジュールではなくPSAの制御ループの一部
3. **ActivationFingerprintTrackerは独立診断**: forward-onlyで追加backwardなし。レジーム統計の収集とヌルベースライン計算を提供し、PSAの効果検証に使用
4. **LAWAはmandatoryベースライン**: GOAL §3.3で必須とされる重み平均ベースライン。evaluate_with_lawa() context managerで評価時のみ一時的に平均重みに差し替え
5. **LayerDeltaAnalysisはper-tensor検証**: Marchenko-PasturヌルベースラインでPC1 dominanceの統計的有意性を判定。GOAL §7の鉄則に準拠

**根拠**:
- `src/tg_lora/psa.py`（228行）: PSAPriorクラス + amplify_gradients_psa()関数
- `src/tg_lora/regime.py`（149行）: RegimeDetector + Regime enum
- `src/tg_lora/activation_regime.py`（325行）: ActivationFingerprintTracker + compute_regime_null_baseline()
- `src/tg_lora/weight_averaging.py`（138行）: LAWAAverager + evaluate_with_lawa()
- `src/tg_lora/layer_delta_analysis.py`（243行）: per-tensor分析 + Marchenko-Pastur
- `configs/9b_tg_lora_psa.yaml`（177行）: PSA実験用設定
- `scripts/run_psa_ablation.sh`（304行）: PSA vs plain vs LAWAアブレーション
- `scripts/run_psa_gamma_sweep.sh`（125行）: γスイープ
- `scripts/summarize_psa_sweep.py`（270行）: スイープ結果集約
- `docs/GOAL.md`（134行）: PSA研究方向の設計意図・鉄則

**更新ファイル**:
- **architecture.md**: コアアルゴリズム層に6モジュール追加（psa, regime, activation_regime, weight_averaging, layer_delta_analysis）。運用層に3スクリプト追加。PSAパイプラインサブグラフ・PSA Experimentサブグラフ追加。PSA設定パラメータテーブル追加。運用性セクションにPSAコマンド追加。信頼性サマリー更新（82→92件）
- **dataflow.md**: Flow 24（PSAパイプライン）、Flow 25（RegimeDetector）、Flow 26（Activation Fingerprint）、Flow 27（LAWAベースライン）、Flow 28（Layer Delta Analysis）を追加
- **design-interview.md**: 本分析記録（A49）を追加

**信頼性への影響**:
- Phase 62の全要件（REQ-265~284）は既存実装ベースのため 🔵
- PSAメインラインへの方向転換（REQ-284）はdocs/GOAL.md §3.1に基づく 🔵
- テスト数更新: 101→129ファイル、2130→2634テストケース

---

### 分析結果サマリー更新

**Phase 62 PSA設計反映後** (A49):
- 🔵 青信号: 168 (+11: A49分析 +1, architecture +5, dataflow +5)
- 🟡 黄信号: 7 (+0)
- 🔴 赤信号: 0 (+0)

**確認できた事項**:
- PSAPriorのpower iteration実装がper-tensor独立処理（cos≈0の前提）で正しく動作することを確認
- RegimeDetectorのconsume_reset_signal()ワンショット消費パターンがPSAのregime-aware prior resetと正しく連携することを実装コードから確認
- ActivationFingerprintTrackerのcompute_regime_null_baseline()がGOAL §7のヌルベースライン鉄則に準拠していることを確認
- LAWA evaluate_with_lawa()のcontext managerが例外安全に重み復元を保証することを確認
- layer_delta_analysis.pyのMarchenko-Pastur計算がランダム行列理論に基づく統計的有意性判定であることを確認
- config_schema.pyのPSAConfigがextra='forbid'で未知フィールドを拒否し、PSA/M9相互排他検証があることを確認

**残課題**:
- PSA統合テストの追加（PSAPrior + RegimeDetector + ActivationFingerprintの相互作用）
- PSA γスイープの実際のGPU実行と結果分析
- GOAL §4の8トラック段階的検証の進捗
