# Master Plan: TG-LoRA Paper Completion

この文書は TG-LoRA 論文の全実験を統括する唯一の正本です。
意思決定の根拠・実行手順・完了条件・現在の状況を一元的に管理します。

---

## 1. 現状サマリ

### 1.1 環境

| 項目 | 値 |
|---|---|
| ローカルマシン | Apple M2 Ultra, 64GB RAM, MPS (CUDA なし) |
| CUDA マシン | RTX 3060 12GB (paper-memory suite 実行環境) |
| Track A モデル | Qwen3.5-9B (CUDA 12GB で TG-LoRA 有効性証明) |
| Track B モデル | **Qwen3.6-35B-A3B** (MoE, 35B total / 3.3B active, MLX 64GB でスケール証明) |
| 学習データ | Dolly 15k (15,011件) から seed=42 でランダム抽出した 5,000件 (旧 1,000件) |
| 評価バックエンド | lm-evaluation-harness 0.4.12 + MLX カスタムバックエンド |

### 1.2 2つの実験トラックの最適化された棲み分け

無駄な重複とハードウェアの特性（RTX 3060 12GB の小 VRAM、M2 Ultra 64GB の UMA）を考慮し、実験トラックを以下のように棲み分けます。

**Track A: CUDA PyTorch Core (RTX 3060 12GB)**
- **役割**: TG-LoRA のコア性能・効率・品質・安全策の主要検証（G1, G3, G4）およびメモリフロンティア測定（G2）
- **実行内容**:
  - 5Kデータ 3-seed (42, 43, 44) one-shot 1024 比較 (PyTorch) は完了済み
  - メモリフロンティアスイープ (MAX_SEQ_LEN=1024, 1536, 2048, 3072) での Baseline OOM vs TG-LoRA Pass の境界特定 (G2)
  - 3-seed 外部品質評価 (G3)
  - Component 2 の cosine-driven `N` 選択と validation-cost removal ablation
- **理由**: 12GB VRAM はコンシューマ環境の標準であり、Baseline が OOM して TG-LoRA が動く限界境界（フロンティア）を劇的に証明するのに最適。また、PyTorch+CUDA の最適化を活かして高速に core 実験を完遂する。

**Track B: Mac MLX Platform (M2 Ultra 64GB)**
- **役割**: Qwen3.6-35B-A3B (MoE) でのスケーラビリティ証明、および Unified Memory Architecture (UMA) による PCIe 転送ボトルネックのない環境（ゼロコピーキャッシュ上限値）におけるシステム性能・メモリオーバーヘッドの評価。
- **実行内容**:
  - Qwen3.6-35B-A3B の MLX 4-bit 変換と QLoRA Baseline 学習 (5Kデータ)
  - 複数チェックポイントでの外部評価 (ARC-Easy, HellaSwag, TruthfulQA)
  - 統一された `run_metrics.jsonl` と `summary.json` 形式での実行ログの生成 (`mlx_coordination_rules.md` に準拠)
  - UMA 環境下での Prefix Caching マイクロベンチマーク実行と、Linux/PCIe-bound キャッシュ結果との転送遅延比較
  - 生成された `runs/mlx_qlora_*` ディレクトリを CUDA 側の環境へ rsync で転送・統合
- **理由**: Qwen3.6-35B-A3B は MoE で active 3.3B/token のため学習コストは 9B と同程度。35B total は CUDA 12GB では不可能だが、MLX 64GB では余裕で動作。論文のストーリーが「9B で有効性証明 → 35B でスケール証明」と強力になる。また、PCIe 物理境界のない UMA 上でキャッシュを測定することで、転送オーバーヘッドを除いた純粋なシステム上の恩恵の上限値を分離評価できる。
- **論文での位置づけ**: Track A が TG-LoRA のアルゴリズム有効性や物理境界でのフロンティア拡張を示すなら、Track B は「より大きなモデルでも品質を維持できること（スケール特性）」、および「PCIe 転送のない UMA でのゼロコピーによる究極的なメモリ転送効果」を示す補強証拠。


### 1.3 廃止する無駄な実験
- **Mac (MPS/PyTorch) 上での TG-LoRA 3-seed 学習の廃止**: MPS 上で PyTorch TG-LoRA を何十時間も回すのは非効率極まりないため、3-seed スイープは Linux/CUDA (Track A) に完全に一本化します。これにより、マシンの計算時間と電力を大幅に節約します。

### 1.4 判定済みの問題点

| 問題 | 深刻度 | 状態 | 影響範囲 |
|---|---|---|---|
| データ 1K件は 9Bモデルに不足 (4エポックで過学習) | 高 | **M0 解決** (5Kに拡大) | Track B |
| valid_quick = valid_full (同一ファイル) | 中 | **M0 解決** (独立化) | 両 Track |
| gold_test が valid の部分集合 | 中 | test.jsonl で代替 | 両 Track |
| MLX 学習でロス記録なし | 中 | **M0 解決** (metrics 追記実装済み) | Track B |
| チェックポイント評価が 2 点のみ | 低 | M1 で対応予定 | Track B |
| M2 Ultra 64GB では OOM フロンティア観測困難 | 情報 | Track A に一元化 | Track B |

---

### 1.5 最新の Component 2 状況 (2026-06-03)

Component 2 は、当初の「一般的な speculative trajectory prediction」ではなく、
`lr` 正規化 EMA が LoRA 更新軌道の支配的直進方向を抽出し、その方向へ複数
optimizer-step 相当を外挿する機構として再整理する。

完了済み:

- offline predictability controls により、未来更新方向と EMA 方向の cosine は
  ランダム対照を大きく上回ることを確認。
- shuffle 対照が true future cosine に近いため、予測力の主因は時間順序の局所的先読みではなく、
  30-step window 全体に存在する低周波の支配的直進成分であると解釈する。
- runtime cosine-N ablation は、固定Nに対して `reduction_rate` を `0.625`
  から `0.752066` に引き上げ、3-seedで rollback 0、best valid loss 実質同等を確認。
- ただし wall-clock は固定N比 `0.9929x` に留まり、残る支配コストは post-extrapolation eval と診断。
- validation-skip diagnostic は完了。skip 条件下でも cosine-driven `N` は固定Nに対して
  `reduction_rate` を `0.54945` から `0.71407` に引き上げ、best valid loss は
  fixed-N mean `1.13131`、cosine-N mean `1.12976` と同等以上。
- ただし wall-clock は fixed-N 比 `1.00006x` に留まった。skip された post-extrapolation
  eval cycle では rollback 0 だったため skip 方針は安全側に動いたが、pilot validation
  forwards (`20`/seed) と scheduled full eval が支配コストとして残った。
- seed 42 の final-eval-only smoke (`EVAL_POINTS=1`) では、scheduled full eval を抑えると
  fixed-N wall-clock `805.6s`、cosine-N wall-clock `806.2s`、baseline `833.7s` まで短縮。
  cosine-N は `reduction_rate=0.71698`、rollback `0.0`、best valid loss `1.13229`。
  これは scheduled full eval が主要固定コストだったことを示すが、単一 seed のため
  manuscript-level claim にはしない。

完了 artifact:

- `runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json`
- preliminary smoke: `runs/cosine_n_skip_final_eval_only_20260603_132236/cosine_n_ablation_summary.json`

次の判定:

- 次は final-eval-only 設定を3-seedへ拡張し、pilot validation の頻度または方式を削る
  固定コスト分解 ablation を行う。
- それでも wall-clock が動かなければ、checkpoint I/O、model reload、ログ出力、cache-equivalence
  check を次の固定コスト候補として分解する。

---

### 1.6 Component 2 の新設計決定 (2026-06-05)

- **課題**: TG-LoRAの効率が1.24倍（理論上限1.5倍）に頭打ちしていた原因が「実装の退化」と確定。これは、方向 $v$ を固定し、スケールを毎ステップその場の少サンプルlossで手探りで調整していたことによる。
- **是正案**: 軌跡から方向 $v$ とスケール $w_{\text{traj}}$ の両方を prior として推定し、その prior のまわりの低次元係数 $\{\alpha, \beta_j\}$ のみをデータで緩やかに学習する設計（Prior-based Subspace Learning）に移行する。
- **数値的安定化**: 係数の勾配 $dv, dw$ は方向微分として取得するが、Qwen 4bit/bitsandbytes の制限で JVP (Jacobian-Vector Product) が使えないため有限差分にフォールバックする。数値条件を改善するため、以下の正規化を適用：
  - 方向の単位化（Unit Normalization）
  - $w_{\text{traj}}$ による無次元化（Dimensionless Scaling）
  - 補助方向の直交化（Auxiliary Orthogonalization）
- **方針**: 実装前に、この設計が成立するかをオフライン検証で確認する。


---

## 2. Claim Ladder と Gates

### 2.1 Claim Ladder

| レベル | 主張 | 必要証拠 | 現状 |
|---|---|---|---|
| **C0 (Safe)** | TG-LoRA は同一 backward-pass budget で internal efficiency を改善する | loss/wall-minute の一貫した改善 | **達成** (1.48x loss reduction per minute aggregate, 1.14x wall-clock speedup on 1K subset, seed42以外でTG>BL) |
| **C1 (Strong)** | TG-LoRA + Layer-Prefix Caching は品質を維持しつつ memory/convergence 効率を改善する | multi-seed 効率 + 外部品質保持 + メモリ指標 | G3 PASS (3-seed summary)、G2 PASS、G1 部分達成 (G1.2/G1.4 PASS [-0.0495 valid loss改善]、G1.1/G1.3 FAIL [PCIeオーバーヘッドによる 0.98x 速度]) |
| **C2 (Revolutionary)** | TG-LoRA + Layer-Prefix Caching は baseline が扱えない訓練領域を可能にする | matched baseline OOM / TG 成功の frontier separation | **達成** (G2 PASS: frontier at 1536/2048 under matched configurations on RTX 3060 12GB) |

「革命的」という表現を使えるのは C2 達成時のみ。

### 2.2 Decision Gates

| Gate | 内容 | 合格条件 | 現状 |
|---|---|---|---|
| G0 | 衛生条件 | 成果物の完全性 | **PASS** |
| G1 | 内部効率再現性 | 全seedでTG効率 > baseline、平均 ≥ 1.25x | **FAIL** (5K Dolly移行により G1.2/G1.4 PASS [-0.0495 valid loss改善]、G1.1/G1.3 FAIL [PCIeオーバーヘッドによる 0.98x 速度]) |
| G2 | メモリフロンティア | baseline OOM で TG 成功、または TG gpu_peak ≥ 20%低い | **PASS** (1536/2048でのOOM分離に成功、3.32 GB [30.8%] VRAM削減、4.6 GB parameters offloaded) |
| G3 | 外部品質保持 | 平均相対低下 < 1%、単一タスク < 3% | **PASS** (3-seed summary: aggregate drop ≈ 0.00%; ARC-EasyはTG優位、HellaSwag 0.52%、TruthfulQA MC2 0.55%。単一 best-checkpoint比較の 1.26% drop は補助 artifact として扱う) |
| G4 | 因果アトリビューション | warm/cold/cache on/off の明確な差 | **FAIL** (G4.2 PASS [VRAM削減効果], G4.1 FAIL) → **オプティマイザ・コンファウンドを解消したキャッシュ分離アブレーション実験・再計測中** |

---

## 3. データ更新手順

### 3.1 問題の整理

現行データ (1K train / 100 valid / 50 gold_test) の問題:

1. **train 1,000件は 9B モデルに不足**: 500 steps × grad_accum=8 ÷ 1000 = 4エポック → 過学習
2. **valid_quick と valid_full が同一**: `prepare_data.py` が同一内容を両方に書き出す
3. **gold_test が valid の部分集合**: テストとして独立でない
4. **test ファイルが存在しない**: valid と独立した最終評価用データがない

### 3.2 新データ仕様

| スプリット | サイズ | 役割 | 備考 |
|---|---|---|---|
| train | 5,000 | 学習用 | Dolly 15k から seed=42 でランダム抽出 |
| valid | 500 | 学習中のバリデーション・ハイパラ調整 | train と独立 |
| test | 500 | 最終品質評価 (論文掲載値) | valid と独立 |
| gold_test | 50 | 軽量スモークチェック | valid の先頭 50件 (従来互換) |

500 steps × grad_accum=8 ÷ 5000 = **0.8 エポック** に改善。

### 3.3 データ生成コマンド (M0 完了済み)

```bash
# prepare_data.py は 3-way split 対応済み
python scripts/prepare_data.py \
  --source dolly \
  --train-size 5000 \
  --valid-size 500 \
  --test-size 500 \
  --seed 42 \
  --output-dir data

# MLX 用シンボリックリンク更新
cd data_mlx && ln -sf ../data/train.jsonl train.jsonl && ln -sf ../data/valid_quick.jsonl valid.jsonl
```

---

## 4. Milestones & 実行計画

### Milestone 0: インフラ修正 (完了)

**タスク**:
- [x] M0-1: `prepare_data.py` を 3-way split 対応に修正
- [x] M0-2: 新データ生成 (train=5000, valid=500, test=500)
- [x] M0-3: データ品質チェックリストの全項目をパス
- [x] M0-4: `data_mlx/` シンボリックリンク更新
- [x] M0-5: MLX 学習でのロス記録手段を確認 (自動 metrics 追記実装済み)

---

### Milestone 1: Qwen3.6-35B-A3B MLX Baseline 学習 & スケール証明 (Track B - Mac 専任) — 進行中

**目的**: 35B MoE モデルについて、9B と対称な3 point (base, @100, @500) で品質推移を比較してスケーラビリティを証明する。また、`mlx_coordination_rules.md` に準拠した structured metrics 出力を確立し、Mac (UMA) でのメモリ評価結果を CUDA ワークスペースにマージする。

**論文テーブル構造 (9B/35B 対称)**:
| | 9B base | 9B @100 | 9B @500 | 35B base | 35B @100 | 35B @500 |
|---|---|---|---|---|---|---|
| ARC-Easy | ✓ | ✓ | ✓ | 82.03% | **82.24%** | **84.26%** |
| HellaSwag | ✓ | ✓ | ✓ | 63.07% | **61.58%** | **61.76%** |
| TruthfulQA | ✓ | ✓ | ✓ | 53.90% | **50.27%** | **48.08%** |

**間引きの理由**:
- 元計画: 10 チェックポイント (@50-@500) × 9h = 93h
- @50 は base と同一スコア (Avg 0.6602 vs 0.6633) → 学習初期の未変動を確認済み、論文には不要
- 中間点 (@150-@400) は過学習曲線の連続性を示すが、学習 loss 曲線で代替可能
- **レビューで「なぜ3点しかない？」→ 「computational budget constraint」+ 学習 loss 曲線で連続性を補完**

**タスク**:
- [x] M1-0: Qwen3.6-35B-A3B を HuggingFace からダウンロードし MLX 4-bit に変換
- [x] M1-1: Baseline 学習を実行し、`run_metrics.jsonl` および `summary.json` を生成 (500 steps, 13.3 min)
- [x] M1-2: 学習ロス曲線から過学習転換点を特定 → run_metrics.jsonl に50点記録済み
- [x] M1-3a: base model 外部評価 (9.3h)
- [x] M1-3b: @50 外部評価 (8.6h) — 参考値、論文テーブル外
- [x] **M1-3c: @100 外部評価 (~9h)**
- [x] **M1-3d: @500 外部評価 (~9h)**
- [x] M1-4: 9B と 35B の品質推移を比較し、スケーラビリティを評価
- [ ] **M1-5: 生成された MLX 結果ディレクトリ (`runs/mlx_qlora_*`) を CUDA 側のワークスペースに転送・マージ**
- [ ] **M1-6: CUDA側で `make compare-mlx` を実行し、クロスプラットフォーム統合レポートを作成**

**完了条件**:
- 9B/35B 対称な3点の ARC-Easy / HellaSwag / TruthfulQA スコアが揃っている
- 35B base のスコアが 9B を上回っている (スケール効果の確認)
- ファインチューニング後の品質推移が 9B と同様のパターンを示す
- MLX のメトリクス（`run_metrics.jsonl`, `summary.json`）が CUDA 側に正常に統合され、プラットフォーム間比較レポートが生成されている

**実測時間**: 変換 ~1h (完了) + 学習 13.3min (完了) + 評価完了 ~36h (全4条件完了) + **マージ・統合レポート生成残り ~30min**

---

### Milestone 2: PyTorch Baseline vs. TG-LoRA 基本性能評価 (Track A - Linux 専任) — 完了

**目的**: CUDA環境かつ同一データ・同一予算で Baseline QLoRA と TG-LoRA の基本性能を比較する。

**2-1: PyTorch Baseline**
```yaml
# configs/9b_baseline_suffix_only_last25.yaml
data: data/train.jsonl       # 5K版
max_steps: 1500              # backward_passes 予算は TG-LoRA と同一
learning_rate: 2e-4
lora_r: 16, lora_alpha: 32
max_seq_len: 2048
active_layer_strategy: last_25_percent
dropout: 0.0
```

**2-2: TG-LoRA Paper PoC**
```yaml
# configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml
data: data/train.jsonl       # 5K版
max_cycles: 適宜             # backward_passes 予算を Baseline と同等に設定
K: 3, N: 5, alpha: 0.3, beta: 0.8
learning_rate: 2e-4
lora_r: 16, lora_alpha: 32
max_seq_len: 2048
active_layer_strategy: last_25_percent
relative_update_cap: 0.005
```

**タスク**:
- [x] M2-1: 5Kデータでの CUDA Baseline/TG-LoRA 実行（3-seed replicationスイープにて内包）
- [x] M2-2: 両モデルの best checkpoint を外部評価 (G3)
- [x] M2-3: 学習ロス比較グラフを作成

**完了条件**: Baseline vs TG-LoRA の外部ベンチマーク比較表と3-seed summaryが揃っている。

---

### Milestone 3: G1 内部効率 3-Seed 検証 (Track A - Linux 専任) — 完了 / strict G1 は未達

**目的**: 5Kデータセットでの TG-LoRA の効率優位性を 3 シードで検証する。
**理由**: 統計的有意性の最低条件。レビューで「再現性は？」に答える必須実験。

**タスク**:
- [x] M3-1: SEEDS='42 43 44' で Baseline + TG-LoRA 各 3 実行
- [x] M3-2: 各シードで以下を記録:
  - loss_reduction / wall_minute
  - total_backward_passes
  - best_valid_loss
  - gpu_peak_memory_mb
  - acceptance_rate (TG-LoRA のみ)
- [x] M3-3: aggregate_summary.json を生成
- [x] M3-4: G1 Gate 判定

**G1 通過条件**:
- 全シードで TG loss_red/wall_min > Baseline
- 集計平均で ≥ 1.25x (5K での最適化結果に基づく)
- best_valid_loss で有意な劣化がないこと

**結果**: strict wall-clock G1 は FAIL。5K では PCIe / validation / cache transfer 系の固定コストにより wall-clock は `0.98x` 付近。一方、best valid loss は 3-seed で一貫して改善し、Component 2 の追加 ablation では `reduction_rate` の改善が確認済み。

---

### Milestone 4: G3 外部品質評価 (Track A - Linux 専任) — 完了

**目的**: 効率向上が品質を犠牲にしていないことを確認する。
**実施結果**: 3-seed downstream summary を採用。単一 best-checkpoint 比較は補助 artifact として扱う。

**タスク**:
- [x] M4-1: 3-seed downstream 評価結果を収集
- [x] M4-2: `external_eval_3seeds_summary.json` を生成
- [x] M4-3: 相対低下の計算
- [x] M4-4: G3 Gate 判定

**G3 通過条件**:
- 集計平均相対低下 < 1%
- 単一タスク相対低下 < 3%

**結果**: PASS。3-seed aggregate drop ≈ 0.00%、HellaSwag 0.52%、TruthfulQA MC2 0.55%、ARC-Easy は TG-LoRA 優位。

---

### Milestone 5: G2 メモリフロンティア検証 (Track A - Linux 専任) — 完了

**結果**: G2 PASS (frontier at 1536/2048、30.8% VRAM削減、4.6 GB freed、3-seed confirmed)

---

### Milestone 6: アブレーション + 因果アトリビューション — 3条件分離実験

**目的**: TG-LoRA の各コンポーネントの寄与を分離する
**前提**: Milestone 3 完了 (3-seed 結果あり)
**背景**: 従来の2条件比較（Baseline vs TG-LoRA）では、Layer-Prefix Feature Cacheの寄与と
trajectory extrapolationの寄与が混在していた。メモリ削減3.32 GB (30.8%) とフロンティア拡張は
キャッシュ単体でも得られる可能性があるため、3条件で厳密に分離する。また、オプティマイザの momentum リセットに伴う confound を解消し、公平な評価を行う。

**3条件比較設計 (G4 Cache Isolation Ablation)**:

| 条件 | Cache | Extrapolation | Optimizer Lifecycle Policy | Config | 目的 |
|---|---|---|---|---|---|
| A: Baseline | なし | なし | `persistent` | `9b_baseline_suffix_only_last25.yaml` | 標準 QLoRA 基準 |
| B: Cache-only | あり | なし (N=0) | `persistent` | `9b_baseline_with_prefix_cache.yaml` | キャッシュ単体の効果 |
| C: TG-LoRA | あり | あり (N=1) | `persistent` | `9b_tg_lora_prefix_feature_cache_paper_poc.yaml` | キャッシュ + 外挿の合算効果 |

**帰属分離**:
- **キャッシュ効果** = B - A（メモリ削減、PCIe転送レイテンシの影響）
- **TG-LoRA効果** = C - B（収束品質改善、trajectory extrapolation の純寄与）

**Confound Resolution**:
以前の設計では、条件BおよびCにおいてサイクルごとにオプティマイザを再作成 (`recreate_per_cycle`) していたため、条件Aの標準的な persistent AdamW (momentum保持) との間で収束特性に confound (交絡) が発生していた。現在は `policy="persistent"` を導入し、同一のオプティマイザ・ライフサイクル管理下でアブレーションを再実行・再計測中。

また、trajectory extrapolation の真の有効性（Lookahead optimizer や単純な学習率チューニングに対する優位性）を証明するため、以下の追加アブレーションも検討・実験中：
1. **Cache-only + tuned LR**: 外挿と同等量のステップ進行をLRスケーリングで模倣できるか検証。
2. **Cache-only + Lookahead**: 既存のLookahead型最適化との収束比較。
3. **Random Direction**: 外挿ではなくランダムな摂動方向への更新＋Accept/Rollbackで同様の収束改善が得られるか（予測の一貫性の検証）。直近のコミットでEMA consistencyに基づく適応的ホライズン選択や cosine horizon ablation 実験が追加され、これらへの実証が進んでいる。

**Component 2 Runtime Ablation Status (2026-06-03)**:

- 完了済み cosine-N ablation: `runs/cosine_n_ablation_20260603_021730`
  - 固定N `reduction_rate`: `0.625`
  - cosine-driven `N` `reduction_rate`: `0.752066`
  - rollback rate: 3-seed とも `0.0`
  - selected `N` distribution: `{1: 3, 3: 1, 5: 1, 10: 2, 20: 3}` が全 seed で一致
  - fixed-N 比 wall-clock: `0.9929x`
  - 診断: cosine-driven `N` は安全に backward replacement を増やすが、validation eval 固定コストが wall-clock を相殺している。
- 完了済み validation-skip diagnostic: `runs/cosine_n_skip_ablation_20260603_083151`
  - 実装コミット: `e37fa00 feat(tg-lora): add cosine-gated validation skip`
  - 現在の最新コミット: `43fb24c test: resolve gpt2 tokenizer download dependency and remove obsolete GPU verification test`
  - `accept_eval_examples=1`
  - high-confidence cycle は post-extrapolation eval を skip、mid-confidence は間引き、low-confidence または `N=20` は強制 eval。
  - cosine-driven `N` は fixed-N に対して `reduction_rate` を `0.54945` から `0.71407` へ改善。
  - skipped cycle の rollback は 0。post-extrapolation rollback は低 confidence で eval を残した cycle に限定。
  - fixed-N 比 wall-clock は `1.00006x` で、2x以上の wall-clock speedup は未達。
  - 診断: post-eval skip は安全だが削減量が小さく、pilot validation と scheduled full eval が残支配コスト。
- preliminary final-eval-only smoke: `runs/cosine_n_skip_final_eval_only_20260603_132236`
  - seed 42 only。`EVAL_POINTS=1`
  - baseline wall-clock `833.7s`
  - fixed-N wall-clock `805.6s`
  - cosine-N wall-clock `806.2s`
  - cosine-N vs baseline wall-clock ratio `0.9670x`
  - cosine-N vs fixed-N wall-clock ratio `1.0007x`
  - cosine-N `reduction_rate` `0.71698`, rollback `0.0`
  - 診断: scheduled full eval 削減は効く。cosine-N 固有の wall-clock 優位には pilot validation 削減が次に必要。

**実行**: `make paper-memory-cache-ablation EXISTING_TG_SUITE=runs/paper_memory_one_shot_suite_20260531_192119`
- 条件Cは既存5K suiteの結果を再利用（再実行不要）
- 条件A+Bの6ラン: ~2h（GPU 1台）

**期待される分析**:
1. GPU peak memory: A ≈ 10782, B ≈ 7459 → メモリ削減はキャッシュに起因
2. Valid loss: B ≈ A → キャッシュは収束に影響せず、C < B → TG-LoRA固有の改善
3. Wall-clock: B vs A → PCIe転送オーバーヘッドの分離

---

---

### Milestone 7: 廃止 (M6 に統合)

因果アトリビューションは M6 の ablation (TG-full vs TG-no-cache) で代替。
warm/cold と cache on/off の比較は同一情報を異なる角度から見るのみ。

---

### Milestone 9: Priorベース低次元係数学習のオフライン検証 (2026-06-05決定)

**目的**: 新設計である「Prior推定＋低次元係数学習」の成立性を、実機実装（学習ループへの結合）の前にオフラインの軌跡データを用いて数学的・数値的に検証する。

**タスク**:
- [ ] M9-1: 既存の学習軌跡（ウェイト差分）データから、方向 $v$ とスケール $w_{\text{traj}}$ を prior として抽出するオフラインスクリプトの作成
- [ ] M9-2: 補助方向を直交化し、低次元空間 $\{\alpha, \beta_j\}$ を定義
- [ ] M9-3: 有限差分による方向微分および数値正規化（単位化・無次元化・直交化）の誤差と安定性の評価
- [ ] M9-4: 実際の軌跡がこの低次元部分空間にどの程度射影（近似）できるかの検証
- [ ] M9-5: オフライン検証レポートの作成と実装判断

**完了条件**:
- 有限差分による方向微分の数値条件が十分に安定していることが確認できること
- 軌跡の低次元表現の近似誤差が十分に小さいことが確認できること

---

### Milestone 8: 論文執筆

**前提**: Milestone 3-5 の Gate 判定で Claim Level が確定後

**許可される即時執筆**:
- [ ] Intro / Problem Statement
- [ ] Methodology (velocity tracking, bounded extrapolation, safeguards)
- [ ] Systems / Cache Optimization (activation cache, Layer-Prefix Feature Cache)
- [ ] Experiment Protocol

**Gate 判定後に執筆**:
- [ ] Experimental Results (確定した Claim Level に基づく)
- [ ] Discussion / Limitations
- [ ] Final title, abstract strongest claim, conclusion wording

**許可されない記述** (C2 未達成時):
- 「革命的」という表現
- frontier separation の確定

---

## 5. 実行優先順位と依存関係 (最小構成)

```
M0 (インフラ) ─────────── 完了 ──────────────────────────┐
                                                         │
M1 (35B eval 残り2点) ─── 進行中 ────────────────────────┤
                                                         │
M2+M3 (3-seed BL vs TG) ── CUDA必須 ────────────────────┤  M8 (論文執筆)
           │                                            │     ↑
           ├─→ M4 (品質 eval, best×2) ──────────────────┤     │
           │                                            │     │
           ├─→ M6 (ablation, no-cache×1) ───────────────┤     │
           │                                            │     │
           └─→ M9 (Priorベース学習オフライン検証) ───────┘     │

M5 (G2 frontier) ──── 完了 ──── 独立
M7 ──── 廃止 (M6に統合)
```

**現在のクリティカルパス**:
1. M1 35B評価マージおよび統合レポート作成
2. **M9: Priorベース低次元係数学習のオフライン検証（実装前検証）** （決定事項に基づき優先配置）
3. M6 Component 2 fixed-cost ablationの継続（またはM9検証結果を受けた設計への移行）

---


## 6. リスクと見直しポイント

| リスク | 影響 | 検出タイミング | 対応 |
|---|---|---|---|
| cosine-gated skip で低 consistency cycle の rollback が増える | 品質劣化 | M6 skip ablation 完了時 | `N=20` 強制 eval 維持、high/mid threshold を上げる |
| データ 5K でも過学習 | M1 やり直し | M1 完了時 | 10K に拡大または steps 削減 |
| Track A と Track B の結果矛盾 | 論文記載に影響 | M4 完了時 | 設定差を明示、Track A を main に据える |
| post-eval skip 後も wall-clock が改善しない | C2 runtime claim が弱まる | M6 skip ablation 完了時 | pilot validation と scheduled full eval を削る固定コスト分解へ移行 |
| Priorベース学習の有限差分フォールバックで数値的不安定性が発生 | 勾配 $dv, dw$ の不正確さによる学習崩壊 | M9 オフライン検証時 | 正規化パラメータ（差分ステップサイズ $\epsilon$、直交化手法）の調整、または別の差分近似法の導入 |
| 低次元部分空間 $\{\alpha, \beta_j\}$ が実際の軌跡を近似できない | 学習効率の改善が頭打ちのまま | M9 オフライン検証時 | 基底（補助方向）の選択数 $j$ や抽出ウィンドウ幅のチューニング |
| レビューで「中間点が足りない」指摘 | 実験追加 | レビュー後 | 学習 loss 曲線で補完、必要なら @250 を追加 (~9h) |

**見直しタイミング**:
- M1 完了時: 35B 品質推移が 9B と同パターンか確認
- 次回 M6 fixed-cost ablation 完了時: Component 2 の wall-clock claim を固定し、結果 snapshot と writing map を更新

---

## 7. 推定スケジュール (最小構成・実測値反映)

### 実測値

| 作業 | 実測時間 | 備考 |
|---|---|---|
| Qwen3.6-35B-A3B 変換 | ~1h | 完了 |
| 35B QLoRA 学習 500 steps | **13.3 min** | avg 73.3 tok/s, peak 38.3GB |
| 9B base eval (3タスク) | **3.9h** | 完了 |
| 9B adapter @100 eval | **4.3h** | 完了 |
| 9B adapter @500 eval | **5.2h** | 完了 |
| 35B base eval (3タスク) | **9.3h** | 完了 |
| 35B adapter @50 eval | **8.6h** | 完了 (参考値) |
| 35B adapter eval (推定) | **~9h** | @100, @500 |

### 最小構成スケジュール

| Milestone | 推定時間 | 状態 | レビュー耐性 |
|---|---|---|---|
| M0: インフラ修正 | - | **完了** | - |
| M1: 35B eval 残り2点 | **~18h** (Mac) | **進行中** | 9B/35B 対称3点 |
| M2+M3: 3-seed BL vs TG | - | **完了** | G1 部分達成、valid loss 改善は強いが strict wall-clock は未達 |
| M4: 品質 eval 3-seed | - | **完了** | G3 PASS |
| M5: G2 frontier | - | **完了** | G2 PASS |
| M6: Component 2 / validation-cost ablation | seed42 smoke 完了 | **部分完了** | cosine-driven `N` は PASS。final-eval-only smoke で scheduled full eval 支配を確認。次は3-seed拡張と pilot validation 削減 |
| M7: 因果アトリビューション | - | **M6に統合** | - |
| M8: 論文執筆 | 並行 | 並行可 | - |

**合計残り**: 35B eval 残り + M6 fixed-cost ablation。M6 は final-eval-only 3-seed
拡張と pilot validation 固定コスト削減に焦点を移す。

---

## 8. 既存証拠の取り扱い

### Main evidence として使用

| 証拠 | 出典 | 用途 |
|---|---|---|
| 3-seed one-shot 1024 aggregate | Track A | C0/C1 の内部効率証拠 |
| G3 external eval (3-seed aggregate drop ≈ 0.00%) | Track A | 品質保持の直接証拠 |
| MLX eval backend | Track B | 評価インフラとして両 Track で活用 |

### 参考証拠 (main table には入れない)

| 証拠 | 出典 | 理由 |
|---|---|---|
| MLX Baseline 500step 外部評価 | Track B | データ 1K の過学習結果、設定が異なる |
| MLX smoke test | Track B | 評価バックエンドの動作確認用 |
| deleted worktree accel sweep | Track A | provenance が弱い |
| paper_memory_suite seed42 (cache=false) | Track A | shakedown run、main ではない |

### 廃棄

| 証拠 | 理由 |
|---|---|
| `reports/eval/adapter_100_eval_*` | データ 1K 過学習、参考値として残すが論文には使わない |
| `reports/eval/adapter_500_eval_*` | 同上 |

---

## 9. 関連ドキュメント

| ドキュメント | 内容 | 本計画との関係 |
|---|---|---|
| [paper_experiment_plan.md](paper_experiment_plan.md) | Gate 定義、Claim Ladder、Shortest Path | 本計画の基礎。本計画が上位 |
| [eval_plan_and_status.md](eval_plan_and_status.md) | MLX Baseline 評価の詳細経緯 | M1 の参照元 |
| [paper_results_snapshot.md](paper_results_snapshot.md) | Track A の canonical evidence 数値 | M3-M5 の参照元 |
| [evaluation.md](evaluation.md) | 評価方法論 | 評価実行時のガイド |
| [hyperparameters.md](hyperparameters.md) | ハイパラ一覧 | Config 設定時の参照 |
| [runbook.md](runbook.md) | 運用手順 | 実行時の参照 |

本計画と `paper_experiment_plan.md` が矛盾する場合は、**本計画を優先**する。
