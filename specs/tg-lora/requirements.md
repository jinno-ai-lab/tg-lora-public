# TG-LoRA 要件定義書（軽量版）

## 概要

TG-LoRA は、LoRA ファインチューニングにおいて velocity（重み差分の指数移動平均）ベースの外挿により後向きパスコストを削減する Python ライブラリ。K ステップの実勾配計算（pilot phase）後に N ステップの外挿を実施し、検証損失に基づいて accept/rollback を判定するサイクルを繰り返す。

## 関連文書

- **分析記録**: [interview-record.md](interview-record.md)

## 主要機能要件

**【信頼性レベル凡例】**:

- 🔵 **青信号**: 既存実装・テストコードを参考にした確実な要件
- 🟡 **黄信号**: 既存実装から妥当な推測による要件
- 🔴 **赤信号**: 参照資料にない自動推定による要件

### 必須機能（Must Have）

- REQ-001: システムは、LoRA パラメータの重み差分の指数移動平均（EMA）を velocity として追跡しなければならない 🔵 *velocity.py 実装より*
- REQ-002: システムは、各層の LoRA パラメータ差分のノルム・統計量を記録し、異常検知・収束トレンド分析を提供しなければならない 🔵 *delta_tracker.py 実装より*
- REQ-003: システムは、外挿失敗時に復元可能な LoRA 重みスナップショットを保存・管理しなければならない 🔵 *rollback_manager.py 実装より*
- REQ-004: システムは、サイクルごとに reduction_rate・acceptance_rate を追跡し、early stopping 判定を提供しなければならない 🔵 *cycle_state.py 実装より*
- REQ-005: システムは、velocity に基づく重み外挿を適用し、相対更新上限（relative_update_cap）による安全な制約を課さなければならない 🔵 *extrapolator.py 実装より*
- REQ-006: システムは、random walk によるハイパーパラメータ適応制御（K, N, alpha, beta, lr）を提供し、accept 時に reward・reject 時に penalize を実行しなければならない 🔵 *random_walk_controller.py 実装より*
- REQ-007: システムは、LoRA パラメータの反復・層グループ化・学習対象スコープ設定（all / last_25_percent）のユーティリティを提供しなければならない 🔵 *lora_utils.py 実装より*
- REQ-008: システムは、LoRA 重みのスナップショット・差分計算・復元・メモリ効率的な差分保存を提供しなければならない 🔵 *lora_state.py 実装より*
- REQ-009: システムは、複数の層選択戦略（last_25_percent, last_25_percent_plus_random_2, middle_random, lisa_like_weighted）を提供しなければならない 🔵 *layer_sampler.py 実装より*
- REQ-010: システムは、学習軌跡の分析（損失トレンド・ボラティリティ・収束予測・early stop アドバイス）を提供しなければならない 🔵 *trajectory.py 実装より*

### 基本的な制約

- REQ-401: Python 3.11 以上、PyTorch 2.1 以上、CUDA GPU が必要である 🔵 *pyproject.toml・README.md より*
- REQ-402: 依存パッケージ: transformers>=4.36, peft>=0.7, bitsandbytes>=0.41, accelerate>=0.25, datasets>=2.16, safetensors>=0.4, tqdm>=4.66 🔵 *pyproject.toml より*
- REQ-403: パッケージ名は `tg_lora` とし、pip install -e . でインストール可能であること 🔵 *pyproject.toml より*

## 簡易ユーザーストーリー

### ストーリー1: LoRA ファインチューニングの高速化

**私は** ML エンジニア **として**
**velocity ベース外挿により後向きパスを削減しファインチューニングを高速化したい**
**そうすることで** 同等品質を保ちながら GPU 計算コストを削減できる

**関連要件**: REQ-001, REQ-005, REQ-006

### ストーリー2: ハイパーパラメータの自動適応

**私は** ML エンジニア **として**
**random walk controller に K, N, alpha, beta, lr を自動調整させたい**
**そうすることで** 手動チューニングの手間を省き、安定した学習を維持できる

**関連要件**: REQ-006, REQ-010

### ストーリー3: 安全な外挿とロールバック

**私は** ML エンジニア **として**
**外挿が失敗した場合に自動的にロールバックさせたい**
**そうすることで** 学習が発散するリスクなく外挿を利用できる

**関連要件**: REQ-003, REQ-005

## 基本的な受け入れ基準

### REQ-001: Velocity EMA 追跡

**Given（前提条件）**: Velocity インスタンスが初期化されている
**When（実行条件）**: delta と beta を指定して update() を呼び出す
**Then（期待結果）**: EMA が更新され、cosine_similarity / magnitude_trend / is_magnitude_anomalous が利用可能になる

**テストケース**:

- [ ] 正常系: EMA 更新・方向一貫性・異常検知が正しく動作する 🔵
- [ ] 主要な異常系: NaN/Inf 入力に対して安全に処理される 🔵

### REQ-005: 重み外挿

**Given（前提条件）**: モデルに velocity とアクティブ層が設定されている
**When（実行条件）**: apply_extrapolation() を呼び出す
**Then（期待結果）**: 指定層の LoRA 重みが velocity 方向に N×alpha×v だけ更新され、relative_update_cap を超えない

**テストケース**:

- [ ] 正常系: 外挿が cap 制約内で適用される 🔵
- [ ] 境界値: ゼロ velocity に対して安全に処理される 🔵

### REQ-006: Random Walk Controller

**Given（前提条件）**: Controller が初期パラメータで設定されている
**When（実行条件）**: propose() → 学習 → accept/reject → reward/penalize
**Then（期待結果）**: ハイパーパラメータが adapt され、convergence_adaptation と acceleration_adaptation が適用される

**テストケース**:

- [ ] 正常系: reward 時に lr/alpha が増加、penalize 時に減少する 🔵
- [ ] 異常系: パラメータが min/max 境界を超えない 🔵

## 最小限の非機能要件

- **パフォーマンス**: 外挿ステップは勾配計算なしで O(params) の加算のみで完結する 🟡 *アルゴリズム定義から推測*
- **セキュリティ**: 評価スクリプトにおける Hugging Face Hub 通信は HF_TOKEN 環境変数を使用する 🔵 *eval スクリプト実装より*
- **品質**: pytest によるユニットテスト・統合テストが全件通ること 🔵 *tests/ 実装より*
