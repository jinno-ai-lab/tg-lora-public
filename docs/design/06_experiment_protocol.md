# 本番実験プロトコル (Experiment Protocol) 設計書

## 1. 事実 (Facts)

### 1.1 実験環境とターゲット設定
- **対象タスクとモデル (Track A)**:
  - ベースモデル: `Qwen/Qwen3.5-9B` ([run_phase2_m9_suite.py:L121](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L121))
  - 量子化 / 手法: bitsandbytes 4bit 量子化 (NF4) + HuggingFace PEFT QLoRA
  - 対象レイヤー: PEFTによる LoRA アダプタの適用対象は `all-linear`（全線形レイヤー、43,278,336 パラメータ）だが、実際の最適化対象（requires_grad=True）は `trainable_lora_scope: last_25_percent`（上部 8層のみ、10,819,584 パラメータ）に制限されて動作している ([train_tg_lora.py:L725-729](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L725-L729))。
- **ハードウェア**: NVIDIA GeForce RTX 3060 12GB (VRAM約8-10GB)
- **乱数シード**: `42`, `43`, `44` の3シード ([run_phase2_m9_suite.py:L28](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L28))

### 1.2 データセットとシーケンス長
- **データセット**: Dolly 15k
  - 3者分割 (Seed=42): train=5000, valid=500, test=500
  - 分割の独立性: シャッフルされたデータからスライス `[:valid_size]` (valid)、`[valid_size : valid_size+test_size]` (test)、`[valid_size+test_size : valid_size+test_size+train_size]` (train) を用いて切り出しているため、各分割間にデータの重複はない ([prepare_data.py:L172-174](file:///home/jinno/tg-lora/scripts/prepare_data.py#L172-174))。
- **最大シーケンス長**: `max_seq_len = 1024`
  - `run_phase2_m9_suite.py` の実行時コード ([run_phase2_m9_suite.py:L78](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L78) および [L103](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L103)) により、Baseline / TG-LoRA 双方の yaml 設定ファイルの値（Baseline: 2048, TG-LoRA: 1024）にかかわらず、実行時に `1024` に上書きして統一されているため、比較の公平性は保たれている。

### 1.3 比較条件とアライメント (等価ステップ)
Baseline と TG-LoRA の比較は、データ消化量 (Epochs / 等価ステップ数) が揃うように設計されている。
- **等価ステップの定義**:
  - `current_equiv_steps = (cycle_state.full_backward_passes + cycle_state.speculative_equivalent_backward_passes) // grad_accum` ([train_tg_lora.py:L588](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L588))
- **アライメントチェックポイント**:
  - ターゲットとなる等価ステップ: `[250, 500, 750, 1000, 1250, 1500]` ([run_phase2_m9_suite.py:L50](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L50))
  - TG-LoRAの学習ループ中、等価ステップ数が上記ターゲット値に到達した時点で、強制的にチェックポイントを保存し、詳細なフル検証（`valid_full`）を実行する ([train_tg_lora.py:L591-602](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L591-602))。
  - Baseline は標準 QLoRA で `1500 steps` まで回し、同様のステップターゲットでチェックポイントを保存する。

### 1.4 下流タスク評価 (Downstream Evaluation)
等価ステップチェックポイントおよび最良モデル（`best_model`）に対して、`lm-evaluation-harness` を用いた評価を行う ([run_phase2_m9_suite.py:L136-189](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L136-189))。
- **評価タスク**: `arc_easy`, `hellaswag`, `truthfulqa_mc2` ([run_phase2_m9_suite.py:L120](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L120))
- **バッチサイズ**: `eval_batch_size = 1` ([run_phase2_m9_suite.py:L104](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L104))

### 1.5 早期終了 (Early Stopping)
検証ロスの改善がストップした場合に備え、early stopping を行う。
- **patience設定**: 最良検証ロスが更新されなくなってから 5サイクル（Baselineは 5段階評価）で学習を停止する ([train_tg_lora.py:L521](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L521))。
- **最良チェックポイントの選択**: `best_model` フォルダに保存された重みを事後選択して最終評価に使用する。

### 1.6 効率の計上方法 (実コストの定義)
- TG-LoRAで消費された実 backward パス（分子）には、`pilot` ステップ、`reject` (拒絶) されたサイクル、`rollback` (ロールバック) が発生したサイクルなど、すべての無駄を含めた実際の計算負荷を計上する。
- 削減率 (`reduction_rate`) の計算式は、同一データを消化するのに必要な Baseline の実 backward パス（分母）と TG-LoRA の実 backward パス（分子）の比率に基づく ([run_phase2_m9_suite.py:L244-247](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L244-247))。
- 時間（wall-clock time）に関する推定や測定値は、システム間のオーバーヘッド等の要因に左右されるため、削減効果の主張対象からは除外され、実 backward 数の削減率のみを指標とする。

---

## 2. 設計意図 (Rationale)

- **なぜ乱数シードを 42, 43, 44 の3点に定めたのか**:
  - TBD（人間確定待ち）
- **なぜ下流タスクとして arc_easy, hellaswag, truthfulqa_mc2 の3つを選択したのか**:
  - TBD（人間確定待ち）
- **なぜ patience=5 に設定したのか**:
  - TBD（人間確定待ち）
- **なぜ等価ステップ数を (実 backward + 削減 backward) // 8 として Baseline の step 数 (grad_accum=8) と揃えるアライメントにしたのか**:
  - TBD（人間確定待ち）
- **なぜ max_seq_len を 2048 から 1024 に下げたのか**:
  - RTX 3060 12GB のメモリ制約により Baseline 2048 を維持できず TG-LoRA 側を1024へ統一、公平性のため両系を同一系列長に揃えた（設計者の選好ではなくハードウェア制約）。
- **なぜ train_size を 5000（dolly_15k からの 5K subset 分割）にしたのか**:
  - TBD（人間確定待ち）
