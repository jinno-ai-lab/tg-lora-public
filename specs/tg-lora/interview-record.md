# TG-LoRA 設計自動分析記録

**作成日**: 2026-06-10
**分析実施**: step4 既存情報ベースの差分分析と自動統合
**関連アーキテクチャ**: [architecture.md](architecture.md)
**関連データフロー**: [dataflow.md](dataflow.md)

## 分析目的

TG-LoRA リポジトリの README・ソースコード（11ファイル）・テストコード（12ファイル・350+テスト）・評価スクリプト（4ファイル）・docs/llm-wiki を確認し、既存実装から機能要件を体系的に抽出した。

## 分析項目と判断

### A1: コアアルゴリズム要件の抽出

**分析日時**: 2026-06-10
**カテゴリ**: 既存設計確認
**背景**: README.md にアルゴリズム概要が記載されているが、詳細はソースコードにのみ存在するため、実装との整合性確認が必要

**判断**: README のアルゴリズム記述と実装は完全に一致。サイクル（propose → snapshot → K steps → delta → velocity update → extrapolate → eval → accept/rollback）は random_walk_controller, velocity, extrapolator, rollback_manager, cycle_state にまたがるモジュール構成で実現されている。

**根拠**: README.md Algorithm セクション、velocity.py, extrapolator.py, random_walk_controller.py, cycle_state.py, rollback_manager.py

**信頼性への影響**:
- REQ-001 ~ REQ-005 の信頼性レベルは 🔵（既存実装と完全一致）

---

### A2: モジュール境界と API サーフェスの確認

**分析日時**: 2026-06-10
**カテゴリ**: 既存設計確認
**背景**: __init__.py から公開 API を特定し、各モジュールの責務を明確化するため

**判断**: __init__.py は 10 のシンボルをエクスポート: Velocity, RandomWalkController, RollbackManager, DeltaTracker, CycleState, apply_extrapolation, cap_update, select_active_layers, get_num_layers, snapshot_lora, load_lora_snapshot, diff_lora, TrajectoryAnalyzer, TrajectoryPoint。各モジュールは単一責務で疎結合。

**根拠**: tg_lora/__init__.py、全モジュールの import 関係

**信頼性への影響**:
- REQ-006 ~ REQ-010 の信頼性レベルは 🔵（公開 API から直接確認）

---

### A3: 層選択戦略の要件整理

**分析日時**: 2026-06-10
**カテゴリ**: 詳細化
**背景**: layer_sampler.py に 4 つの戦略が実装されているが、要件として明文化が必要

**判断**: 4 つの戦略（last_25_percent, last_25_percent_plus_random_2, middle_random, lisa_like_weighted）はいずれも完全に実装済みでテストカバレッジあり。デフォルトは last_25_percent_plus_random_2。

**根拠**: layer_sampler.py、test_layer_sampler.py

**信頼性への影響**:
- REQ-009 の信頼性レベルは 🔵

---

### A4: 評価スクリプトの位置づけ

**分析日時**: 2026-06-10
**カテゴリ**: 影響範囲
**背景**: 4 つの eval スクリプトが存在するが、これらは tg_lora パッケージの外部（scripts/）にあり要件スコープの判断が必要

**判断**: 評価スクリプトはライブラリ本体ではなく利用例・ベンチマークとして位置づける。パッケージ要件（REQ-001~010）の対象外とするが、非機能要件（品質確認手段）として参照する。

**根拠**: scripts/ ディレクトリ構成、Makefile ターゲット、pyproject.toml の packages.find 設定

**信頼性への影響**:
- 評価スクリプト関連は要件から除外し、補足情報として扱う

---

### A5: 既存要件・重複確認

**分析日時**: 2026-06-10
**カテゴリ**: 重複確認
**背景**: specs/, docs/spec/, docs/design/, docs/tasks/ に既存要件文書がないか確認

**判断**: 既存の要件定義書・設計書・タスク管理ファイルは一切存在しない。docs/llm-wiki/ は自動生成されたリポジトリ分析ドキュメントであり、要件文書ではない。統合・マージ対象なし。

**根拠**: ディレクトリ構造調査、rg --files 検索結果

**信頼性への影響**:
- 統合対象なし。全要件を新規作成。

---

## 分析結果サマリー

### 確認できた事項

- アルゴリズムは README 記述と完全一致する実装済みコードとして存在する
- 10 個のソースモジュールが単一責務・疎結合で構成されている
- 350+ のテストが全モジュールをカバーしている
- 公開 API は __init__.py で明示的に管理されている
- 既存要件文書は一切存在しない（新規作成）

### 追加/変更要件

- なし（既存実装から要件を抽出するのみ）

### 残課題

- 評価スクリプトの要件化要否（現在はライブラリ外部と位置づけ）
- MLX 版評価スクリプトの正式サポート要否
- 運用要件（CI/CD、デプロイ）の未定義

### 信頼性レベル分布

**分析前**:

- 🔵 青信号: 0
- 🟡 黄信号: 0
- 🔴 赤信号: 13（全要件が未確認）

**分析後**:

- 🔵 青信号: 11 (+11)
- 🟡 黄信号: 2 (+2)
- 🔴 赤信号: 0 (-13→0)

## 関連文書

- **アーキテクチャ設計**: [architecture.md](architecture.md)
- **データフロー**: [dataflow.md](dataflow.md)
- **要件定義書**: [requirements.md](requirements.md)
