# 主張と証拠の対応表 (Claims and Evidence Map) 設計書

## 1. 事実 (Facts)

### 1.1 主張とそれを支える証拠の定義
実験コードおよび評価スクリプトに基づいて、以下の対応関係が定義されている。

| 主張 (Claims) | 証拠となるデータ / 実装 (Evidence) | 証拠の評価と詳細 |
| :--- | :--- | :--- |
| **1. 支配的な勾配方向 ($v_0$) の実在** | - オフライン勾配収集スクリプト: `collect_true_gradients.py` の分析結果<br>- オンライン学習時のコサイン類似度ログ: `tg_lora_cosine_sim` ([train_tg_lora.py:L1861](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1861)) | 連続する学習ステップ間において、勾配ベクトルがランダムではなく特定の方向（主成分）に強いコサイン類似度を示すことを確認。`collect_true_gradients.py` とオンライン cosine ログで支持され、比較的堅牢である。 |
| **2. Prior-based 外挿 (M9) の有効性 (品質維持)** | - 3シード本番ランの等価ステップにおける `loss_valid` 履歴 ([run_phase2_m9_suite.py:L303-338](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L303-338))<br>- テストセット・下流タスク評価: `report_aligned.md` 内の downstream 結果 ([run_phase2_m9_suite.py:L376-410](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L376-410)) | Baselineと同等以下の検証ロスおよび下流タスク（ARC, HellaSwag, TruthfulQA）スコアを達成していることを確認。オフライン CV の hold-out 評価（Fold1/Fold2 で Subspace Fit(4) > Tuned LR(2)、leakage 非依存）で支持される。 |
| **3. 計算コスト (実 backward) の削減** | - `total_backward_passes` および `tg_lora_reduction_rate` の記録 ([train_tg_lora.py:L1861-1865](file:///home/jinno/tg-lora/src/training/train_tg_lora.py#L1861-L1865))<br>- 累積実 backward 数のアライメント表 ([run_phase2_m9_suite.py:L342-374](file:///home/jinno/tg-lora/scripts/run_phase2_m9_suite.py#L342-374)) | 同等データ消化量に達するまでに必要な累積実 backward パス数が Baseline より削減されていることを確認。N バグ・リーク修正後の120サイクル run による Baseline 同一 budget・493件フル評価の比較完走を待つ「検証中」とする（撤回はしない。効率主張は本研究のゴールであるドメイン特化モデル量産のための超効率 LoRA に直結するため、検証中として保持し、結果に応じて確定する）。 |

### 1.2 主張しないこと (Non-claims)
- **処理時間 (Wall-clock Speed)**: 本プロジェクトでは、検証フォワードのオーバーヘッドや量子化モデルの有限差分探索にかかる実測時間（秒）の短縮は主張しない。指標は「実 backward 数の削減率」のみに限定される（全主張を通じて wall-clock は非主張）。
- **生 Prior の直接適用**: 外部から得た Prior をそのまま重みに足し込むのではなく、あくまで局所的な subspace ($v_0, u_1, u_2$) 内で有限差分によりフィットさせた係数（$\alpha, \beta$）を用いて外挿を行う。
- **直線探索 (Component 1) の新規性/優位性**: 探索コストが大きく頭打ちになった直線探索（$\alpha$-line など）ではなく、Priorベースの部分空間学習 (M9) を主要なアプローチとする。

### 1.3 評価の分岐ルール
下流タスク (downstream task) 評価結果の判定基準：
- **良好 (Good)**: TG-LoRA の平均スコアが Baseline の平均スコアに対して一定のマージン（TBD）以内、またはそれを上回る場合。
  - **結果**: 提案手法が下流タスク性能を損なわずに効率化可能であると結論付ける。
- **不十分 (Insufficient)**: 下流タスク性能が Baseline と比較して有意に劣化している場合。
  - **結果**: SFT損失の削減と下流タスク一般化性能の解離（TBD）を事実として報告する。

---

## 2. 設計意図 (Rationale)

- **なぜ処理時間 (Wall-clock time) の主張を完全に排除したのか**:
  - wall-clock は GPU・実装・バッチサイズに強く依存し再現性と公平性を担保しにくいため主要指標から除外する（[06_experiment_protocol.md](file:///home/jinno/tg-lora/docs/design/06_experiment_protocol.md) の wall-clock 非主張方針と整合）。
- **なぜ下流タスク評価の判定マージン（良好/不十分の境界）を現時点で数値固定していないのか**:
  - 主張3（計算コスト削減）の120サイクル本番runによる完走結果を待つ検証中ステータスであり、効率向上の度合いに応じて下流タスク性能とのバランスを確定するため（人間確定待ち）。
- **なぜ直線探索（Component 1）の優位性・新規性の主張を放棄し、M9に一本化したのか**:
  - 探索コストが大きく頭打ちになった直線探索ではなく、Priorベースの部分空間学習 (M9) が本質的な効率と品質の維持（過適合の防止）を両立する主要なアプローチであるため。
