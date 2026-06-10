# Tangent-Gradient LoRA (TG-LoRA) — 全体設計・研究指針 (GOAL)

> **本書の位置づけ**: TG-LoRA 自律開発ループの最上位ガイドライン。設計意図（Design Intent）と実装実態（Implementation Fact）を区別し、未確認は **[UNVERIFIED]** と明示する。
> **最終更新**: 2026-06-10（§1.6 新設: Progressive Freezing + Activation Matching の全体計画を統合。§3・§4 を新路線中心に再構成）。
> **先行研究の正本**: [docs/research/README.md](docs/research/README.md)（全8トラック）。

---

## 0. 目的（Purpose）

TG-LoRA は QLoRA 学習を「重み軌跡の構造」を利用して効率化する研究である。

- **第1期〜第5期**（完了）: velocity 外挿 → PSA (Prior-based Subspace Amplification) に至る検証履歴。PSA 本体は実装済みで baseline と比較済み。
- **第6期**（現行）: **Progressive Freezing + Activation Matching**。少データ・多エポックの LoRA 学習において、学習の進行に応じて後段から順にレイヤをフリーズし、backward 計算を削減する。

**最終ゴール**: 少データ・多エポックの LoRA ファインチューニング（画像 LoRA 20枚×100エポック等）で、最適なフリーズスケジュールにより、full backprop と同等の品質を保ちながら総学習コスト（backward FLOPs・VRAM・時間）を有意に削減する。対象は **Track A: Qwen3.5-9B（dense hybrid）** と **Track B: Qwen3.6-35B-A3B（MoE）**。

**ターゲットユースケース**: 個人・小規模環境（RTX 3060 12GB級）で、ドメイン特化の小データを反復ファインチューニングする状況。同一データを多エポックで周回する前提。

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
- **cycle 6 = グローバル Catapult / Edge-of-Stability 相転移**（[Track02 §20](docs/research/02_training_dynamics_analysis.md)）。バグでも障害でもなく、設計が尊重すべき地形イベント。
- **Leap+Verify（[Track05](docs/research/05_speculative_extrapolation_zeroth_order.md)）**: 投機的重み予測は TG-LoRA と同型。(1) **モメンタム/optimizer-state 外挿は全スケールで破滅、有限差分のみ機能** → FD 採用は正しい。(2) ボトルネックは予測精度でなく **regime availability** → 効率頭打ちは構造的限界の疑い。(3) **activation cosine による相検出が forward only で実装可能**。
- **RNA / Anderson Acceleration（[Track06](docs/research/06_sequence_acceleration_forward_gradient.md)）**: 外挿の係数フィットは原理的に不安定 → **L2 正則化は後回しにせず最初から必須**。
- **LAWA（Track06）**: 重み平均は単純かつ強力。**外挿はこれに勝てなければ存在意義がない** → 必須ベースライン。
- **Forward gradient（Track06）**: 方向微分推定は次元比例で高分散 → 低次元射影・多方向平均は正攻法。
- **アーキ/低精度（[Track07](docs/research/07_architecture_lowprecision_specifics.md)）**: config 確定 `mamba_ssm_dtype: float32` により DeltaNet 状態は fp32。FD の bf16 丸め問題は **Attention/FFN の bf16 経路で深刻** → Stochastic Rounding はそこへ優先適用。
- **対象モデル構造（[Track08](docs/research/08_target_model_structure_prestudy.md)）**: Track A=32層（24 GDN + 8 Attention, dense FFN 12288）、Track B=40層（30 GDN + 10 Attention, 256 experts/8+1 active）。**out_proj を持つ少数 Attention 層が最安定**。MoE は hot expert のみ外挿可。

### 1.6 第6期 — Progressive Freezing + Activation Matching（2026-06-10）

PSA は勾配増幅により「各ステップを少しだけ良くする」路線だが、backward グラフ自体は変わらない。計算の根本的な削減には、**backward グラフの一部を物理的に切断する**必要がある。この認識から、Progressive Freezing 路線に移行した。

#### 1.6.1 中核の洞察

**前提**: レイヤ X 以降がフリーズ済みであるとする。データ x に対して、フリーズした時点でレイヤ X の入力 `xin` を記録しておく。この `xin` は「レイヤ X が正しく動くために受け取るべき入力」である。なぜなら、レイヤ X 以降はもう固定で、その固定された後段が良い最終出力を出すことは、フリーズ時点での学習結果として保証されている。その後段が前提としている入力が `xin`。

**帰結**: レイヤ X-1 は「データ x を受け取ったら `xin` を出す」ように学習すればいい。`xin` を target にした local loss でレイヤ X-1 を学習する。後段 X は一切通さない。backward はレイヤ X-1 とその M のぶんだけ。

**なぜ learning signal が生まれるか**: `xin` は「フリーズ時点（＝full 学習が十分進んで後段が良くなった時点）の値」であり、現在の未熟なレイヤ X-1 の出力とは異なる。この差が learning signal になる。「現状維持」にはならない。

**可逆性の回避**: `xin` は逆算した値ではなく、実際に forward で観測された実測値。`f⁻¹` を解く必要がない。

これは feature/activation matching による layer-wise training の一種であり、既知の系統である。

#### 1.6.2 単一 run 内での漸進フリーズ

これは「full run で target を作ってから別 run で使い回す」二段構えではない。**1回の学習プロセスの内部で**、エポックが進むにつれて後段から順にフリーズしていく。

具体例（画像 LoRA: 20枚×100エポック相当）:
- 最初の 20エポック: 全層 full 学習（荒く全体を動かす）
- 次の 20エポック: 最終10層をフリーズ、残りだけ学習
- その次: さらにフリーズ範囲を拡大

同じデータを何度も周回する（エポックを重ねる）前提だからこそ、「後段はもう固まった」という段階が自然に訪れる。この段階で後段を固めれば、その周回以降の backward が軽くなる。

**償却問題は発生しない**: target を別に作る二重学習ではなく、学習の進行そのものが target を生成するから。N* や移植性の論点は、単一 run 内のスケジュール設計には直接関係しない。

#### 1.6.3 2段階の実装レベル

**Level 1（基本）: Progressive Freezing**
- 後段をフリーズ（weight gradient を停止）
- activation gradient は貫通する（最終 loss は伝播）
- それでも backward は軽くなる（フリーズ層の weight gradient 計算と optimizer 更新が消える）
- VRAM も減る（フリーズ層の optimizer 状態が不要）
- 実装が枯れていて確実。少データ多エポックで即効く。

**Level 2（発展）: Activation Matching**
- 後段の activation gradient の貫通も省く
- フロント層を `xin` の再現（local loss）で閉じる
- 計算は最大限減るが、代理ロスの整合性リスクあり
- Level 1 が成立した後の発展実験として位置づける

#### 1.6.4 既存の観測との関係

これまでの TG-LoRA の観測が、フリーズスケジュール設計に直接効く:

- **層間非整列 (cos ≈ 0)**: 層間の依存が弱い = フリーズした後段の target が、前段の学習進行に対して安定。Progressive Freezing の前提に有利。
- **cycle 6 相転移**: 全層が一斉に挙動を変える。フリーズのタイミングを相転移の前後で変えることで、target の品質が変わる。設計の重要な自由度。
- **初期安定（attention の rank-1 方向）**: attention 層は早い段階で方向が固まる。これらを先にフリーズする候補になる。
- **rank-1 支配**: 各テンソルの動きが1方向に集中している = フリーズ後もその方向は安定しやすい。

#### 1.6.5 制約

1. **フリーズのタイミング**: 各レイヤをフリーズする時点で、後段がすでに十分成熟している必要がある。早すぎると未熟な後段の `xin` を target にしてしまい、未熟さを下流へ伝播させる。cycle 6 相転移のような観測が「いつフリーズしてよいか」の判断材料になる。
2. **代理ロスの限界 (Level 2)**: `xin` を target にした MSE 等の local loss は最終タスク loss の代理 (proxy)。点ごとの一致 (MSE) では捉えきれない統計的構造のずれが残りうる。ただし層間非整列が強い（層間依存が弱い）ため、分布ずれの伝播は小さい可能性がある。

---

## 2. 現在の理解（Operating model — 設計はこれに従う）

- **予測ベースの軌跡外挿は死亡**（グローバルも層別も loss 着地で負ける）。今後は「予測」ではなく **構造の安定性**（優位部分空間・regime）を使う。
- **信号は層内に集中**。すべての解析・ギミックは **per-tensor / 層タイプ別**（DeltaNet / Attention / FFN / expert）で行う。**グローバル平均は禁止**。
- **層間は独立 (cos ≈ 0)**。この独立性が Progressive Freezing の前提を支える。フリーズした後段の target は前段の学習に対して安定。
- **cycle 6 はグローバル相転移**。相をまたぐ操作は崩壊する。相を検出してゲート/リセットする。
- **少データ多エポック**: フリーズが効く土俵。同データを何度も見るから、後段は早い段階で役割を固め終える。

---

## 3. これからの設計（Forward design）

### 3.1 主路線: Progressive Freezing + Activation Matching [UNVERIFIED]

**研究の問い**: 少データ・多エポックの LoRA 学習で、学習の進行に応じて後段から順にフリーズしていくと、最終品質を保ちながら総学習コスト（backward FLOPs・VRAM・時間）をどれだけ削れるか。最適なフリーズスケジュール（いつ・どの層を・どの順で固めるか）は何か。

#### Phase 0: ベースライン確定
- 同一データ・同一サイクル数で **full backprop**（全 LoRA 層を最終 loss で学習）の valid_loss・総 backward コスト・収束カーブを記録。
- この run 自体が Progressive Freezing の出発点になる（最初のNエポックは全層 full 学習としてそのまま使う）。
- baseline_plain best_valid=1.0565、accum16=1.0704 と整合確認。
- 比較対照として、同一条件で「フリーズなし full 最後まで」を別シードでも回しておく。

#### Phase 1: 最小ゲート（単層フリーズ）
- Phase 0 と同じ run の中で、学習が十分進んだ段階（例: 全エポックの 20%地点）で **最終段1層だけをフリーズ**。
- フリーズ直前の forward で観測されたその層の入力 `xin` をキャッシュ（別 run から取るのではない。その run 自身の forward で自然に得られる値）。
- フリーズ後のエポックでは、直前の1層だけを `xin` を target にした local loss（まず MSE）で学習。
- **判定基準**:
  - フリーズ後も valid_loss が full backprop の収束カーブに追従するか（代理ロスの整合性）
  - フリーズ層の backward が省かれたことによる FLOPs 削減の実測
- **このゲートを通らなければ、順次フリーズに進む意味がない**。

#### Phase 2: フリーズスケジュールの設計
Phase 1 を通過後、3つの自由度をスイープして最適スケジュールを決定。

**フリーズ順序（3候補）**:
1. **出力側→入力側に素直に1層ずつ**: 計算削減は最も確実。suffix が確実に伸びる。
2. **収束順**: 各層の方向安定性が閾値を超えた順にフリーズ。xin 品質は高いが、出力側に連続しない場合は suffix が切れず計算が減らない。
3. **折衷**: 出力側から順だが、各層は方向安定性が閾値に達するまでフリーズを保留。

**事前予測**: 層間非整列 (cos ≈ 0) が強ければ、候補1（後方フリーズ）で性能が崩れにくいはず。

**フリーズ深度**: 後段から1層、2層、…と段階的に深度を増やし、各深度での valid_loss 劣化と FLOPs 削減をプロット。効率と性能のフロンティア曲線を描く。

**フリーズタイミング**: 各層を、相転移 (cycle 6) の前・後・大幅後の3点でフリーズ。xin 品質が valid_loss に与える影響を測る。事前予測は「相転移後の方が xin が成熟して良い」。

#### Phase 3: Activation Matching 版（Level 2 発展実験）
最適スケジュールが固まった後、「後段貫通も省けばさらに削れるか」を試す。
- 損失関数のアブレーション: MSE 単独 vs MSE+cos vs 分布も合わせる版
- Level 1 (progressive freeze + 最終loss伝播) と Level 2 (activation matching) の定量比較

#### Phase 4: 跨条件検証（スケジュールの汎用性）[副次]
主実験は Phase 2 までで完結する。Phase 4 は「見つけた最適スケジュールが、別の条件でも通用するか」を確認する付加検証:
- 学習率を変える
- データを一部入れ替える/追加する
- LoRA rank を変える
- シードを変える
- 有効半径（スケジュールが通用する条件の範囲）を地図化

※ ここで検証するのは「target xin の使い回し」ではなく、「スケジュール（いつ何層固めるかという手順）の使い回し」。手順は具体的な活性値より条件変動に強いと予測される。

### 3.2 副路線: PSA（Prior-based Subspace Amplification）— **実装済み・保留**
通常 backward に I/O のみのオーバーヘッドで統合済み（`src/tg_lora/psa.py`）。PSA は backward グラフ自体は変えないが、各ステップを増幅して品質を上げる路線。Progressive Freezing と直交するため、将来的な組み合わせは可能。

- **Prior 抽出**: 各 LoRA テンソルの跨サイクル履歴 ΔW から power iteration で優位方向 `v_PSA`(PC1) を算出。per-tensor（層間 cos≈0 のため独立）。
- **方向的ゲイン**: `G' = G + γ·⟨G, v_PSA⟩·v_PSA`。ノルム 2x 上限クランプ。
- **層タイプ別ゲイン**: `out_proj`×1.2 / `v_proj`×1.1 / `mlp`×0.7。
- **相転移リセット**: `RegimeDetector` が loss velocity z-score で STABLE / PLATEAU / TRANSITION を分類。TRANSITION 検出時に prior リセット。
- **config**: `enable_psa`, `psa_history_length=6`, `psa_gain=0.5`, `psa_update_interval=3`, `psa_warmup_steps=4`, `psa_l2_reg=0.01`, `psa_regime_reset_enabled=True`。

### 3.3 必須ベースライン（公平比較）
- **素 LoRA + 調整済み LR**。
- **LAWA（重み平均）**。いかなる手法もこれに勝てて初めて価値がある。
- **評価条件統一**: `max_seq_len=1024`・`valid_full.jsonl`(493件)・同一 eval 関数（`src/eval/eval_loss.py`）。評価リークは禁止。

---

## 4. 実行計画（Execution plan）

依存順に実行。**各ステップは観測値を単独解釈せず、必ず帰無基準を併記して判定する**（§7 の鉄則）。

### 直近（Phase 0 + Phase 1）

1. **Phase 0: full backprop run**（ベースライン + Progressive Freezing の開始点）
   - 全 LoRA 層を最終 loss で学習。valid_loss・総 backward コストを記録。
   - この run の前半（例: 全エポックの 20%）がそのまま Progressive Freezing の「全層 full 学習期間」になる。
   - baseline_plain / accum16 と整合確認。

2. **Phase 1: 単層ゲート**
   - 同じ run の中で学習が十分進んだ段階で最終段1層をフリーズ。フリーズ直前の forward で `xin` をキャッシュ。
   - 直前の1層を MSE activation matching で学習。
   - **判定**: フリーズ後の valid_loss が full の収束カーブに追従するか。backward 削減の実測。
   - **不通過なら**: Progressive Freezing 路線全体を「層間独立が足りず不成立」と診断し記録して閉じる。

### Phase 1 通過後

3. **Phase 2: フリーズスケジュール設計**
   - 順序アブレーション（後方素直 / 収束順 / 折衷）
   - 深度スイープ（1層〜N層、フロンティア曲線）
   - タイミングスイープ（相転移前・後・大幅後）
   - 対照: (i) full backprop, (ii) ランダム順フリーズ（サロゲート）, (iii) 同層数を最終 loss で学習

4. **Phase 3: Activation Matching 版**（Level 2）
   - 損失関数アブレーション（MSE / MSE+cos / 分布一致）
   - Level 1 との定量比較

5. **Phase 4: 跨条件検証**
   - 最適スケジュールを異なる条件（LR / データ / r / シード）に適用
   - 有効半径の地図化
   - 損益分岐分析

### データ量の使い分け

| 段階 | データ量 | 目的 |
|------|----------|------|
| Phase 1（ゲート） | 最小（1サイクル分） | 安価に原理判定 |
| Phase 2（スケジュール） | 中規模（数サイクル分） | フロンティア曲線の安定化 |
| Phase 3-4（最終確認） | Full run 同等 | 本番条件で1回だけ検証 |

### 統計の歯止め

- 各条件は複数シードで回す
- valid_loss 差はブートストラップ CI で評価
- ランダム順フリーズ（サロゲート）を超えた削減・性能だけを有効と認定
- 「計算が減った」「性能が保てた」も対照を超えて初めて主張

### 成功の定義

ある（順序・深度・タイミング・損失）の組で、以下を同時に満たすこと:
- valid_loss 劣化が許容閾（full 比 +数%以内、閾値はベースライン分散から決定）に収まる
- backward FLOPs がランダム順フリーズ対照を有意に超えて削減できる

劣化が大きいか削減が小さければ、「層間独立が足りず成立しない」と診断し記録して閉じる。

---

## 5. 効率会計（Exact Cost Accounting）

- **実 backward 数 = K × grad_accumulation**。外挿・答え合わせ・PSA 増幅は backward を消費しない。
- **Progressive Freezing での計測**:
  - 総 backward FLOPs = Σ(各エポックでの活性層の backward FLOPs)
  - 削減率 = 1 − (progressive freezing の総 FLOPs) / (full backprop の総 FLOPs)
  - VRAM 削減 = フリーズ層の optimizer 状態 + activation 勾配の削減分
- **Activation Matching (Level 2) の追加削減**: backward グラフの suffix 切断による FLOPs 削減

---

## 6. 実装構成（現状）

### 既存（PSA 路線）
1. `src/tg_lora/velocity.py` — 勾配速度・コサイン類似度・`choose_N`。
2. `src/tg_lora/psa.py` — PSA 本体: per-tensor PC1・L2 正則化・層タイプ別ゲイン。
3. `src/tg_lora/extrapolator.py` — M9 部分空間再構築（棄却済み・残置）。
4. `src/tg_lora/cycle_state.py` — 実 backward / 等価ステップ会計。
5. `src/training/train_tg_lora.py` — メインループ。PSA 統合済み。
6. `src/tg_lora/regime.py` — 相検出: `RegimeDetector`。
7. `src/tg_lora/activation_cache.py` — 活性キャッシュ（xin 記録の基盤として再利用可能）。

### 新規必要（Progressive Freezing 路線） [UNVERIFIED]
- フリーズスケジュール制御（いつ・どの層をフリーズするか）
- `xin` キャッシュ機構（フリーズ時点の入力活性を記録）
- Activation matching loss（MSE / cos / 分布一致）
- 層別フリーズ制御（weight gradient 停止 + Level 2 では activation gradient も切断）

---

## 7. 鉄則（過去の罠の回避 — 最重要規律）

過去に cos=−0.5・R²=0.71・2.6σ などを誤って「信号」と判断した教訓に基づく安全装置。

- **層・時間・モードを安易に平均しない**。平均は補助、判断は分布と地図で行う。
- **すべての指標にランダム帰無基準を併記**（Marchenko-Pastur 期待値・ノルム保存サロゲート・項を共有しないホールドアウト）。
- **統計的有意性と実用的集中度を区別**（2.6σでも ZO に使えなかった前例）。
- **予測力 cos は loss 着地で検証するまで信用しない**（cos=0.065 が loss に変換できなかった前例）。
- **評価条件を必ず統一し、評価リークを禁止**（fit バッチと答え合わせバッチを分離）。
- **フリーズの効果は full backprop と直接比較する**（中間指標だけで「効いている」と結論しない）。
