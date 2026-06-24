# SYSTEM_CONSTITUTION.md — TG-LoRA 自律開発ループの核心的使命

> **位置づけ**: 本ファイルは自律開発ループ（`25_purpose_driven_executor` 等）が
> Phase 1 で読む「核心的使命」の**ルート固定名エントリポイント**である。
> **正本（canonical source）は [docs/GOAL.md](docs/GOAL.md)**（§0 目的・§2 現在の理解・
> §4 実行計画・§5 効率会計・§7 鉄則）。本ファイルは GOAL.md を**蒸留**した不変の
> 使命宣言であり、GOAL.md と矛盾した場合は **GOAL.md を正**とする。本ファイル自体を
> 編集する場合は GOAL.md の対応節を先に確認すること。

---

## 核心的使命（1 行）

TG-LoRA は、少データ・多エポックの LoRA ファインチューニング（同一データを何度も周回する
前提）において、**学習の進行に応じて後段から順にレイヤをフリーズする Progressive Freezing**
により、full backprop と同等の品質を保ちながら**総学習コスト（backward FLOPs・VRAM・時間）
を有意に削減する**研究である。

> 現行路線 = **第6期 Progressive Freezing + Activation Matching**（GOAL §1.6）。
> 第1期〜第5期（velocity 外挿 / 漸進ランク ZO / B-filter / PSA）は GOAL §1.1–§1.5 で
> 帰無基準により**棄却または保留**済み。PSA 本体は `src/tg_lora/psa.py` に実装保留置き。

---

## 成功条件（定量）— GOAL §4「成功の定義」

ある（順序・深度・タイミング・損失）の組で、以下を**同時に**満たすこと:

- **品質保持**: valid_loss の劣化が許容閾（full backprop 比で **+数% 以内**、閾値は
  ベースライン分散から決定）に収まる。
- **コスト削減**: backward FLOPs が **ランダム順フリーズ対照（サロゲート）を有意に超えて**
  削減できる（GOAL §4 / §7「対照を超えて初めて主張」）。

両方を満たさなければ「層間独立が足りず成立しない」と診断し記録して閉じる（GOAL §4 末尾）。

---

## 品質ゲート（P0–P3）

| Gate | 内容 | 根拠 |
|------|------|------|
| **P0** | **科学的誠実性（最重要）** — 全ての指標にランダム帰無基準（Marchenko-Pastur 期待値・ノルム保存サロゲート・項を共有しないホールドアウト）を併記。未測定のまま結論しない。「通ったように見えるが実は不活性」を排除する。評価リーク禁止（fit バッチと答え合わせバッチを分離）。 | GOAL §7 鉄則 |
| **P1** | **コア deliverable** — 上記の成功条件（品質保持 ∧ コスト削減）を満たす最適フリーズスケジュールを特定する。 | GOAL §4 |
| **P2** | **公平比較** — 素 LoRA + 調整済み LR、および **LAWA（重み平均）** を必須ベースラインとし、いかなる手法もこれらに勝って初めて価値がある。評価条件は `max_seq_len=1024`・`valid_full.jsonl`・同一 eval 関数で統一。 | GOAL §3.3 |
| **P3** | **精密なコスト会計** — 実 backward 数 = K × grad_accumulation。外挿/PSA/フリーズは backward を消費しない。削減率 = 1 − progressive/full、VRAM 削減 = フリーズ層の optimizer 状態 (+ Level 2 の activation 勾配)。 | GOAL §5 |

---

## 絶対原則

1. **検証可能性** — 中間指標（cos・R²・σ）だけで「効いている」と結論しない。必ず loss 着地と
   full backprop の直接比較、および帰無基準を併記する（GOAL §7）。過去に cos=−0.5・R²=0.71・
   2.6σ を「信号」と誤判断した教訓。
2. **層・時間・モードを安易に平均しない** — 信号は層内に集中・層間は独立（cos≈0）。全ての
   解析・ギミックは per-tensor / 層タイプ別（DeltaNet/Attention/FFN/expert）。**グローバル平均は禁止**（GOAL §2, §7）。
3. **小差分主義** — 1 タスク = 1 コミット、diff 200 行以下、テスト駆動（Red→Green）。各タスクに
   検証コマンドを明記。
4. **差し替え可能性** — 受口（pluggable receiver）は正しく、データが届けば即活性化する構造
   （§6.x 誠実性パターン）。producer と受口を分離し、片側だけが欠けても silent にならない。
5. **教育的説明責任** — 設計意図（Design Intent）と実装実態（Implementation Fact）を区別し、
   未確認は **[UNVERIFIED]** と明示する（GOAL.md 表紙）。

---

## 対象

- **Track A**: Qwen3.5-9B（dense hybrid, 32 層 = 24 GDN + 8 Attention）
- **Track B**: Qwen3.6-35B-A3B（MoE, 40 層 = 30 GDN + 10 Attention, 256 experts/8+1 active）
- **ターゲット環境**: 個人・小規模（RTX 3060 12GB 級）でドメイン特化の小データを反復ファインチューニング。

---

> 次のステップの選定は [PURPOSE.md](PURPOSE.md) の未達 deliverable から行う（`25_purpose_driven_executor` Phase 2）。
