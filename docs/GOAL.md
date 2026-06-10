# Tangent-Gradient LoRA (TG-LoRA) — 全体設計・研究指針 (GOAL)

> **本書の位置づけ**: TG-LoRA 自律開発ループの最上位ガイドライン。設計意図（Design Intent）と実装実態（Implementation Fact）を区別し、未確認は **[UNVERIFIED]** と明示する。
> **最終更新**: 2026-06-09（先行研究レビュー Tracks 01-08 を統合し全面改訂。旧 §8 の逐次実験ログ・棄却済み路線の詳細は除去）。
> **先行研究の正本**: [docs/research/README.md](file:///home/jinno/tg-lora/docs/research/README.md)（全8トラック）。

---

## 0. 目的（Purpose）

TG-LoRA は QLoRA 学習を「重み軌跡の構造」を利用して効率化する研究である。当初の狙いは velocity 外挿による実 backward スキップだったが、検証の結果その素朴な路線は**原理的に棄却**された。現在の中核は **安定な優位部分空間に沿った増幅 (PSA) ＋ regime-aware 制御** である。

**最終ゴール**: 同一品質を baseline より少ない backward で達成する、または同一 backward で品質を上げる。対象は **Track A: Qwen3.5-9B（dense hybrid）** と **Track B: Qwen3.6-35B-A3B（MoE）**。

---

## 1. 研究の経緯（What we tried and what we learned）

確定した事実のみを時系列で要約する。各結論は帰無基準（サロゲート/理論値/ホールドアウト）で検証済み。

### 1.1 第1期 — velocity / M9-FD 外挿 → **棄却**
- **設計**: 軌跡平均方向 `v0` + 低次元係数 (α, β1, β2) を有限差分(FD)でフィットして外挿し、backward を投機スキップ。
- **修正**: ウォームアップ中の lr 崩落バグ（非対称減衰）を解消し `w_traj` を回復。
- **結果**: FD フィットは bf16 + batch=1 でノイズ支配（α std ≈ 4.46）。accept 率 3%、reduction 7.7%。
- **決定的診断**: グローバル ΔW のステップ間 cos ≈ 0.016、過去10歩の最適線形予測でも R² ≈ 0.0008。加速度の "構造" は MA(1) アーティファクト（cos ≈ −0.5 は数学的恒等式、ラグ回帰 R² は共通項汚染）。
- → **グローバルな運動学的外挿は予測可能率 ≈ 0% で成立しない**。

### 1.2 第2期 — 漸進ランク ZO → **棄却**
- **仮説**: 低ランクから段階開放すれば低次元 ZO 窓を維持できる（命題A: 低次元集中 → B: ZO有効 → C: 飽和検出）。
- **結果**: P1 で r=2 でも全体 ΔW の top-2 累積寄与率 24%（基準50%未満）。命題A不成立で中止。

### 1.3 第3期 — 層別再解析 → **重要な反転**
- 「全層連結 → グローバル SVD」という**解析手法自体が人工物**だった（方向の違う124本の矢印を1点に集めて平均していた）。
- **確定構造: 層内集中・層間非整列**。各テンソルは rank-1 支配 (0.78)、`lora_B` rank-1 方向は安定 (z=6.8σ)、per-tensor 速度自己相関 (z=5.9σ)。層間は独立 (cos ≈ 0)。
- ただし**層別の velocity 予測外挿も loss 着地で baseline/FO に全敗**（rank-1 外挿 Δloss=+0.004 vs FO pilot −0.026、9回中0勝）。→ 予測ベース外挿は層別でも死亡。
- **生き残った資産**: 予測ではなく、**静的・準静的な優位部分空間（軸）の安定性**を使う道。

### 1.4 第4期 — B-filter 棄却 → **PSA へ転換**
- B透過後 A_signal 安定性 0.91 は B 時系列シャッフルでも 0.91（z=0.31σ）→ 「Bが ΔA と動的協調してノイズ濾過する」仮説を**棄却**。Bの安定性は受動的慣性。
- A側の時間実効モード数 PR=5.78（Marchenko-Pastur 期待 10.98、z=61.8σ で有意に低次元）。
- **cycle 6 同期**: 全層一斉に ΔA ノルムが前サイクルの約 45〜51% へ急減。非同期な波動伝播ではなく**グローバルなフェーズ遷移**。
- → 動的協調をあきらめ、安定優位部分空間を増幅する **PSA (Prior-based Subspace Amplification)** へ。

### 1.5 第5期 — 先行研究レビュー（Tracks 01-08, 2026-06-09）
全8トラックを `docs/research/` に整備。TG-LoRA に直結する要点:
- **cycle 6 = グローバル Catapult / Edge-of-Stability 相転移**（[Track02 §20](file:///home/jinno/tg-lora/docs/research/02_training_dynamics_analysis.md)）。バグでも障害でもなく、設計が尊重すべき地形イベント。
- **Leap+Verify（[Track05](file:///home/jinno/tg-lora/docs/research/05_speculative_extrapolation_zeroth_order.md)）**: 投機的重み予測は TG-LoRA と同型。(1) **モメンタム/optimizer-state 外挿は全スケールで破滅、有限差分のみ機能** → FD 採用は正しい。(2) ボトルネックは予測精度でなく **regime availability** → 効率頭打ちは構造的限界の疑い。(3) **activation cosine による相検出が forward only で実装可能**。
- **RNA / Anderson Acceleration（[Track06](file:///home/jinno/tg-lora/docs/research/06_sequence_acceleration_forward_gradient.md)）**: 外挿の係数フィットは原理的に不安定 → **L2 正則化は後回しにせず最初から必須**。
- **LAWA（Track06）**: 重み平均は単純かつ強力。**外挿はこれに勝てなければ存在意義がない** → 必須ベースライン。
- **Forward gradient（Track06）**: 方向微分推定は次元比例で高分散 → 低次元射影・多方向平均は正攻法。
- **アーキ/低精度（[Track07](file:///home/jinno/tg-lora/docs/research/07_architecture_lowprecision_specifics.md)）**: config 確定 `mamba_ssm_dtype: float32` により DeltaNet 状態は fp32。FD の bf16 丸め問題は **Attention/FFN の bf16 経路で深刻** → Stochastic Rounding はそこへ優先適用。
- **対象モデル構造（[Track08](file:///home/jinno/tg-lora/docs/research/08_target_model_structure_prestudy.md)）**: Track A=32層（24 GDN + 8 Attention, dense FFN 12288）、Track B=40層（30 GDN + 10 Attention, 256 experts/8+1 active）。**out_proj を持つ少数 Attention 層が最安定**。MoE は hot expert のみ外挿可。

---

## 2. 現在の理解（Operating model — 設計はこれに従う）

- **予測ベースの軌跡外挿は死亡**（グローバルも層別も loss 着地で負ける）。今後は「予測」ではなく **構造の安定性**（優位部分空間・regime）を使う。
- **信号は層内に集中**。すべての解析・ギミックは **per-tensor / 層タイプ別**（DeltaNet / Attention / FFN / expert）で行う。**グローバル平均は禁止**。
- **cycle 6 はグローバル相転移**。相をまたぐ外挿・増幅は崩壊する。相を検出してゲート/リセットする。
- **効率の上限は regime availability で律速される可能性**。実装を弄る前に、まず相の在庫を測る。
- **FD は正しいが低精度数値条件に弱い**。L2 正則化・Stochastic Rounding・低次元射影・多方向平均で守る。

---

## 3. これからの設計（Forward design）

2路線を並行検討。いずれも**追加 forward を増やさない、増やすなら厳密に会計する**。

### 3.1 主路線 A: PSA（Prior-based Subspace Amplification）— **実装済み**
通常 backward に I/O のみのオーバーヘッドで統合済み（`src/tg_lora/psa.py`、テスト `tests/test_psa.py`）。投機・答え合わせの 38 forward/cycle を全廃。`enable_psa` と M9 は排他。

- **Prior 抽出（実装済）**: 各 LoRA テンソルの跨サイクル履歴 ΔW（永続 ring buffer）から **power iteration** で優位方向 `v_PSA`(PC1) を算出（全 SVD より安価）。**per-tensor**（層間 cos≈0 のため独立）。
- **方向的ゲイン（実装済）**: backward と optimizer.step の間で `G' = G + γ·⟨G, v_PSA⟩·v_PSA`。ノルム 2x 上限クランプで暴発防止。
- **L2 正則化（実装済）**: prior 更新を直前方向へブレンド（RNA 理論 arXiv:1805.09639 準拠、`psa_l2_reg`）。Track06 の「最初から L2」を反映済み。
- **層タイプ別ゲイン（実装済）**: `out_proj`×1.2 / `v_proj`×1.1 / `mlp`×0.7。Track08 の out_proj 最安定仮説を反映。
- **config（実装値）**: `enable_psa`, `psa_history_length=6`, `psa_gain=0.5`, `psa_update_interval=3`, `psa_warmup_steps=4`, `psa_l2_reg=0.01`, `psa_regime_reset_enabled=True`, `psa_regime_window=8`, `psa_regime_plateau_eps=1e-4`, `psa_regime_transition_z=2.0`。
- **相転移リセット（実装済）**: `RegimeDetector` が loss velocity の z-score で STABLE / PLATEAU / TRANSITION を分類。TRANSITION 検出時に `consume_reset_signal()` で one-shot 消費し、`PSAPrior.reset_priors()` を発火。`psa_regime_reset_enabled` で ON/OFF 切替可能。

### 3.2 副路線 B: Regime-gated Leap+Verify（外挿を残す場合の保険）
- forward-only の **activation-fingerprint cosine** で相（chaotic / transition / stable）を分類。
- **stable 相でのみ** FD 外挿、transition 近傍は quadratic、chaotic は通常学習に倒す。
- 外挿係数に **L2 正則化を最初から**。**Stochastic Rounding** を bf16 経路（Attention/FFN）に適用。
- MoE（Track B）は **routing 統計で hot expert のみ**外挿（cold expert は破綻）。
- 単独では LAWA に勝てない公算が高く、PSA が頭打ちのときのみ着手。

### 3.3 必須ベースライン（公平比較）
- **素 LoRA + 調整済み LR**（[Track04](file:///home/jinno/tg-lora/docs/research/04_snr_adaptive_amplification.md) の反証「適切な LR なら素の LoRA で十分」への対応）。
- **LAWA（重み平均）**。PSA/外挿はこれに勝てて初めて価値がある。
- **評価条件統一**: `max_seq_len=1024`・`valid_full.jsonl`(493件)・同一 eval 関数（`src/eval/eval_loss.py`）。quick(32件)は accept 判定のみ、最終品質比較は full(493件)。評価リーク（fit バッチと答え合わせの混同）は禁止。

---

## 4. 自律実行の優先順位（Autonomous execution plan）

依存順に実行。**各ステップは観測値を単独解釈せず、必ず帰無基準を併記して判定する**（§7 の鉄則）。

1. **相在庫の計測**（最優先・安価）: activation-fingerprint cosine 時系列を取得し 3相分類。**stable 相の割合＝効率の理論上限**を確定し、1.24x 頭打ちの真因（実装 vs 地形律速）を判定する。
2. **層タイプ別 ΔW 解析**: DeltaNet / Attention / FFN で rank-1 支配度・方向安定性を分離測定（out_proj 最安定仮説の検証、`mamba` fp32 経路の数値優位の確認）。
3. **PSA の ablation・評価**（PSA 本体は実装済み）: γ・history・update_interval スイープ。**素 LoRA・LAWA と同一データ消化軸**で比較し、勝てるかを判定。
4. **PSA に cycle 6 相転移リセット** — **実装済**: `RegimeDetector` + `consume_reset_signal()` + `PSAPrior.reset_priors()`。`psa_regime_reset_enabled` でアブレーション可能。
5. **（条件付き）Regime-gated 外挿**: PSA が不十分な場合のみ。L2 + Stochastic Rounding + stable 相ゲートを同時投入。
6. **Track B 拡張**: hot-expert 限定・層タイプ別戦略を MoE で検証。

---

## 5. 効率会計（Exact Cost Accounting — 保持）

- **実 backward 数 = K × grad_accumulation**。外挿・答え合わせ・PSA 増幅は backward を消費しない。
- **reduction_rate = 1 − full_bp / (full_bp + speculative_equivalent_bp)**。外挿承認サイクルでのみ分母に `N × accum` が上乗せされる。
- 外挿の forward オーバーヘッド（FD 6 + accept eval 32 = **38 forward/cycle**）は無駄も含め厳密計上する。**PSA はこのオーバーヘッドを全廃する**のが利点。

---

## 6. 実装構成（現状）

1. `src/tg_lora/velocity.py` — 勾配速度・コサイン類似度（`predicted_consistency`）・`choose_N`。
2. `src/tg_lora/psa.py` — **PSA 本体（実装済）**: per-tensor PC1（power iteration）・L2 正則化・層タイプ別ゲイン・勾配増幅フック `amplify_gradients_psa`。
3. `src/tg_lora/extrapolator.py` — M9 部分空間再構築・FD フィット（M9 路線は棄却済み・残置）。
4. `src/tg_lora/cycle_state.py` — 実 backward / 等価ステップ会計・`reduction_rate`。
5. `src/training/train_tg_lora.py` — メインループ。PSA 統合済み（履歴記録→prior 抽出→gain map→勾配増幅→相転移リセット）。
6. `src/tg_lora/regime.py` — **相検出（実装済）**: `RegimeDetector` — loss velocity z-score による 3相分類 + `consume_reset_signal()` one-shot 消費。

---

## 7. 鉄則（過去の罠の回避 — 最重要規律）

過去に cos=−0.5・R²=0.71・2.6σ などを誤って「信号」と判断した教訓に基づく安全装置。

- **層・時間・モードを安易に平均しない**。平均は補助、判断は分布と地図で行う。
- **すべての指標にランダム帰無基準を併記**（Marchenko-Pastur 期待値・ノルム保存サロゲート・項を共有しないホールドアウト）。
- **統計的有意性と実用的集中度を区別**（2.6σでも ZO に使えなかった前例）。
- **予測力 cos は loss 着地で検証するまで信用しない**（cos=0.065 が loss に変換できなかった前例）。
- **評価条件を必ず統一し、評価リークを禁止**（fit バッチと答え合わせバッチを分離）。
