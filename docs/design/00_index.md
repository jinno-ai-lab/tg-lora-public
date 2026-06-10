# 00_index.md — ドキュメント地図と用語集

## 1. ドキュメント地図 (Document Map)

本ドキュメント群は、TG-LoRA (Tangent-Gradient LoRA) の設計、実装、会計、および本番実験のプロトコルを定義する正本である。

1.  [00_index.md](file:///home/jinno/tg-lora/docs/design/00_index.md): ドキュメント地図と用語集（本ファイル）
2.  [01_algorithm_dataflow.md](file:///home/jinno/tg-lora/docs/design/01_algorithm_dataflow.md): 1サイクルの完全データフローと処理シーケンス
3.  [02_cost_accounting.md](file:///home/jinno/tg-lora/docs/design/02_cost_accounting.md): 実 backward パス数に基づく効率会計
4.  [03_gating_and_N_selection.md](file:///home/jinno/tg-lora/docs/design/03_gating_and_N_selection.md): コサイン類似度制限と投機ステップ $N$ の決定
5.  [04_acceptance_and_validation.md](file:///home/jinno/tg-lora/docs/design/04_acceptance_and_validation.md): 答え合わせ（二階層検証）と承認/拒否の判定論理
6.  [05_subspace_m9.md](file:///home/jinno/tg-lora/docs/design/05_subspace_m9.md): Prior-based Subspace Learning (M9) の詳細
7.  [linearity_guard.md](file:///home/jinno/tg-lora/docs/design/linearity_guard.md): Linearity Guard (アブレーション用フォールバック安全装置)
8.  [06_experiment_protocol.md](file:///home/jinno/tg-lora/docs/design/06_experiment_protocol.md): 本番実験 (Seed 42,43,44) の公平性比較プロトコル
9.  [07_design_decisions_log.md](file:///home/jinno/tg-lora/docs/design/07_design_decisions_log.md): アーキテクチャの発展史と過去の設計決定の履歴
10. [08_claims_and_evidence_map.md](file:///home/jinno/tg-lora/docs/design/08_claims_and_evidence_map.md): 論文の主張とそれを支える証拠の対応マッピング
11. [10_progressive_freezing.md](10_progressive_freezing.md): Progressive Freezing + Activation Matching の設計と原理説明

---

## 2. 用語集 (Glossary)

アルゴリズムの認識ずれを防ぐため、用語とそのコード上の対応物を一意に定義する。

### 2.1 pilot step
*   **定義**: 通常のオプティマイザによる、実 backward (誤差逆伝播) を伴うパラメータ更新ステップ。
*   **コード上の対応物**: [train_tg_lora.py:L1482](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1482) 内の `for _ in range(pilot_K):` の実更新ループ。
*   **1サイクルでの登場回数**: `K` ステップ（ステップごとに `grad_accumulation = 8` 回の `forward_backward` が走るため、実 backward 回数としては `K * 8` 回）。

### 2.2 cycle
*   **定義**: 1回の `pilot step` の実行から、それに続く `投機判定`、`外挿`（またはスキップ）、および `答え合わせ` までの一連の交代プロセスの最小単位。
*   **コード上の対応物**: `train_tg_lora.py` のメインループ `for cycle in pbar:` における 1回のループ。
*   **1サイクルでの登場回数**: 1回。

### 2.3 N
*   **定義**: 投機（外挿）によってスキップしようとするオプティマイザの更新ステップ数。
*   **コード上の対応物**: `proposal.N` または `tg_lora_N`。
*   **1サイクルでの登場回数**: 1回（コサイン類似度閾値を満たし、承認された場合はオプティマイザ `N` ステップ分を一挙に数式でジャンプする）。

### 2.4 等価ステップ (Equivalent Step)
*   **定義**: TG-LoRA が実際に行った更新ステップ（実 backward 由来）と、外挿成功によってスキップした更新ステップ（投機由来）を合算した、**データ消化量基準での進捗ステップ数**。
*   **コード上の対応物**: `current_equiv_steps = (cycle_state.full_backward_passes + cycle_state.speculative_equivalent_backward_passes) // grad_accum`
*   **1サイクルでの登場回数**: 1回（毎サイクルの最後で計算される）。

### 2.5 投機 (Speculation / Extrapolation)
*   **定義**: パラメータの支配方向（Prior）に沿って、数式的な外挿によって重みを `N` ステップ先へと一気にジャンプさせる処理。
*   **コード上の対応物**: [src/tg_lora/extrapolator.py](file:///home/jinno/tg-lora/src/tg_lora/extrapolator.py) の `apply_extrapolation`。
*   **1サイクルでの登場回数**: 最大 1回（類似度閾値を満たし、外挿が行われる場合のみ）。

### 2.6 答え合わせ (Accept-probe evaluation)
*   **定義**: 外挿が成功した（＝モデル性能が悪化していない）かを判定するため、独立した validation セット（32サンプル）を用いて Forward 損失を計算し、accept/reject を下す処理。M9設計への移行に伴い、データリーク防止のためM9でもこの独立検証による判定が行われるようになった。
*   **コード上の対応物**: [train_tg_lora.py:L2750-2790](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L2750-L2790) の `eval_loss(model, valid_quick_loader, ...)`。
*   **1サイクルでの登場回数**: 最大 1回（外挿が適用された場合のみ）。
