# TG-LoRA ハイパーパラメータ完全ガイド

## はじめに

TG-LoRA (Tangent-Gradient LoRA) は、LoRAの重み更新を「方向と速度」で管理し、
学習した方向に沿って勝手に先へ進む（外挿する）ことで、少ない計算量で効率よく学習する手法です。

このドキュメントは、`configs/9b_tg_lora.yaml` の全項目を順に説明します。

---

## 1. 実験設定

```yaml
experiment:
  name: tg_lora_9b_mvp
  seed: 42
```

| パラメータ | 説明                                                         |
| ---------- | ------------------------------------------------------------ |
| `name`     | 実験名。ログや保存先のフォルダ名に使われる。自由に決めてOK   |
| `seed`     | 乱数のタネ。同じseedなら同じ結果が再現できる。42は慣習的な値 |

---

## 2. モデル設定

```yaml
model:
  name_or_path: Qwen/Qwen3.5-9B
  dtype: bfloat16
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  device: cuda:0
```

| パラメータ               | 説明                                                               |
| ------------------------ | ------------------------------------------------------------------ |
| `name_or_path`           | HuggingFaceのモデル名またはローカルパス                            |
| `dtype`                  | モデルの基本精度。`bf16`（半精度）が推奨。メモリを通常の半分に節約 |
| `load_in_4bit`           | `true`で4bit量子化（QLoRA）。12GB GPUで9Bモデルを動かすために必要  |
| `bnb_4bit_quant_type`    | 量子化方式。`nf4`（NormalFloat4）が標準。通常のFP4より精度が良い   |
| `bnb_4bit_compute_dtype` | 計算時の精度。bf16で計算して4bitで保存するイメージ                 |
| `device`                 | どのGPUを使うか。`cuda:0` = 1枚目、`cuda:1` = 2枚目                |

**初心者向け解説**: 9Bパラメータのモデルは本来18GB必要ですが、4bit量子化により約2.5GBに圧縮できます。
RTX 3060 12GBでも動くのはこのおかげです。

---

## 3. LoRA設定

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
```

| パラメータ       | 説明                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `r`              | LoRAのランク（表現力）。**大きいほど学べることは増えるが、メモリも増える**。16は標準的な値 |
| `alpha`          | LoRAの出力を何倍にするか。**alpha/r が実質的な学習率の倍率**になる。r=16, alpha=32なら2倍  |
| `dropout`        | ランダムに無効化する割合。過学習防止。0.05 = 5%を無効化                                    |
| `target_modules` | LoRAを適用する層。`all-linear`は全ての線形層（ほぼ全層）に適用                             |

**初心者向け解説**:

- `r`は「追加する小さな行列のサイズ」。元の重みは凍結して、この小さな行列だけ学習する
- `alpha/r`が大きいとLoRAの更新が強く反映される。小さいと元のモデルの知識を保ちやすい
- `all-linear`は一番効果的だが、メモリが足りない場合は`["q_proj", "v_proj"]`などに減らすことも可能

---

## 4. データ設定

```yaml
data:
  train_path: data/train.jsonl
  valid_quick_path: data/valid_quick.jsonl
  valid_full_path: data/valid_full.jsonl
  gold_test_path: data/gold_test.jsonl
  max_seq_len: 2048
```

| パラメータ         | 説明                                                                      |
| ------------------ | ------------------------------------------------------------------------- |
| `train_path`       | 学習データ。1行1サンプルのJSONL形式                                       |
| `valid_quick_path` | 素早い評価用データ（少量）。毎サイクルのaccept/rollback判定に使う         |
| `valid_full_path`  | 本格的な評価用データ（全量）。最良モデルの選択に使う                      |
| `gold_test_path`   | 最終テスト用データ（参考用。学習中は使わない）                            |
| `max_seq_len`      | 1サンプルの最大トークン数。**大きいほど長文を学べるが、メモリ消費が激増** |

**初心者向け解説**:

- `max_seq_len=2048`で1サンプルあたり最大2048トークン（約1500語）
- これを1024に減らすとメモリ消費が約半分になるが、長い文章が切れる
- valid_quickは「サクッと良くなったか確認する」用、valid_fullは「じっくり評価する」用

---

## 5. 学習設定

```yaml
training:
  batch_size: 1
  grad_accumulation: 8
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_cycles: 500
  gradient_checkpointing: true
  max_grad_norm: 1.0
  optimizer_lifecycle: recreate_per_cycle
```

| パラメータ | 説明 |
| --- | --- |
| `batch_size` | 一度に処理するサンプル数。1 = 1件ずつ処理（メモリ節約） |
| `grad_accumulation` | 勾配を何回分ためてから更新するか。**実質バッチサイズ = batch_size × grad_accumulation**。8なら実質バッチ8 |
| `learning_rate` | 学習率。1回の更新でどれくらい重みを変えるか。**一番重要なハイパーパラメータ**。大きすぎると発散、小さすぎると学習が遅い |
| `weight_decay` | 重みの減衰。過学習防止。0.0 = 減衰なし（LoRAでは小さい値で十分なことが多い） |
| `max_cycles` | TG-LoRAの最大サイクル数。1サイクル = pilot学習 + 外挿 + accept/rollback |
| `gradient_checkpointing` | `true`でメモリ節約。計算速度は落ちるが、GPUメモリが大幅に節約できる |
| `max_grad_norm` | 勾配の最大値。これを超えるとクリップ（打ち切る）。学習が暴走するのを防ぐ安全装置 |
| `optimizer_lifecycle` | `recreate_per_cycle` が mainline。`reuse_state_reset_experimental` は AdamW state の再確保を避ける実験モード |
| `trainable_lora_scope` | `all` が通常。`last_25_percent` は最後の25%の LoRA 層だけを trainable に固定する experimental comparison mode |
| `prefix_feature_cache_experimental` | `true` で suffix-only trainable mode に切り替え、前半 prefix の hidden states を CPU RAM に事前キャッシュする実験モード |

**初心者向け解説**:

- `grad_accumulation`は「8回分の学習をためて一気に更新」する仕組み。GPUメモリが足りなくても大きなバッチと同じ効果が得られる
- `learning_rate`は2e-4 (= 0.0002)がQLoRAの定番。1e-4だと慎重、5e-4だと積極的
- `gradient_checkpointing`は「計算結果を捨てて、必要な時に再計算する」仕組み。メモリ30-40%節約

`optimizer_lifecycle: reuse_state_reset_experimental` は、optimizer を持ち越しつつ exp_avg / exp_avg_sq / step を in-place でゼロ化し、fresh optimizer に近い挙動を保ったまま allocator churn を減らすための実験機能です。論文 mainline の仮説とは分離して扱います。

`trainable_lora_scope: last_25_percent` は、比較実験用の suffix-only 制約です。これは prefix cache 実験だけでなく、cache を使わない baseline 側にも適用できるので、問題設定を揃えた apples-to-apples 比較が可能になります。

`prefix_feature_cache_experimental: true` は、後半25%の LoRA 層だけを trainable に固定し、prefix 側の hidden states を事前計算して再利用する experimental mode です。効果は optimizer ではなく forward の構造的スキップにあります。現在の mainline が使う activation cache は「1 cycle 内の post-extrap eval を短絡する一時 cache」ですが、この mode は「train/valid 全体の prefix 特徴量を CPU RAM に常駐させる precompute cache」です。prefix 出力の決定論性を保つため、現状は `lora.dropout=0.0` を前提にしています。

現行の experimental config は amortization を優先し、`train` 側の precompute はデフォルトで無効、`valid_quick` / `valid_full` の cache を優先します。precompute 時間も run metrics の wall-clock に含めて解釈してください。

`prefix_feature_cache_dir` は cache blob の永続保存先です。同じ dataset path、model、seed、LoRA 設定、split layer、max sequence length なら次回 run で再利用されます。`prefix_feature_cache_force_rebuild: true` を指定すると既存 blob を無視して再構築します。

---

## 6. TG-LoRA設定（このプロジェクトの核心）

```yaml
tg_lora:
  K_initial: 3
  K_candidates: [2, 3, 5, 8]

  N_initial: 5
  N_candidates: [1, 3, 5, 10, 20]

  alpha_initial: 0.3
  alpha_min: 0.03
  alpha_max: 1.5
  alpha_log_sigma: 0.15

  beta_initial: 0.8
  beta_candidates: [0.5, 0.8, 0.9, 0.95]

  lr_initial: 0.0005
  lr_min: 1.0e-5
  lr_max: 0.001
  lr_accept_boost: 1.2
  lr_reject_decay: 0.5

  relative_update_cap: 0.005

  active_layer_strategy: last_25_percent
  force_top_layers_only: true
  enable_random_walk: false
  enable_convergence_adaptation: true
  k_explore_prob: 0.0
  n_explore_prob: 0.0
  beta_explore_prob: 0.0
  strategy_explore_prob: 0.0
  random_middle_layers: 2
```

### 6-1. K（pilot steps）: サイクル内の学習回数

```yaml
K_initial: 3        # 初期値
K_candidates: [2, 3, 5, 8]  # コントローラーが選べる候補
```

**何をするか**: 1サイクルの中で「ちゃんと学習する（pilot）」回数。

**例え**: K=3なら、「3歩歩いて、その方向が良さそうなら、その方向にさらに飛ぶ」。
K=8なら、「8歩慎重に歩いてから飛ぶ」。

**トレードオフ**:

- Kが小さい → 1サイクルが速いが、方向の精度が低い
- Kが大きい → 方向は正確だが、1サイクルに時間がかかる

**コントローラーの動き**: `enable_random_walk: true` の時だけ、Kはlrと連動して適応的に変化する。paper-PoC の既定設定では K は固定される。

- Accept時: 20%の確率でKを減らす（攻撃的に成功しているなら、もっと速く進める）
- Reject時: 30%の確率でKを増やす（失敗したなら、もっと慎重に探る）

### 6-1b. enable_convergence_adaptation: 収束判定で守りに入るか

```yaml
enable_convergence_adaptation: true
```

**何をするか**: random walk を有効にしたまま、`DeltaTracker.convergence_trend() >= 0` の時に lr を減らし、K を増やす能動制御を入れるかどうか。

**使いどころ**:

- `true` → 古い adaptive branch の挙動を再現する。stalling と見なした時に守りへ寄せる。
- `false` → reward / penalize による更新は残しつつ、この追加ヒューリスティクスだけを切る。

**なぜ重要か**: adaptive run で lr が単調減少し K=8 に寄る場合、原因が extrapolation そのものではなくこのヒューリスティクスである可能性がある。したがって `alpha_max` を先にいじるより、このフラグの ablation を先に行う方が識別力が高い。

### 6-2. N（extrapolation steps）: 外挿の飛び幅

```yaml
N_initial: 5        # 初期値
N_candidates: [1, 3, 5, 10, 20]  # 候補
```

**何をするか**: pilotで見つけた方向に沿って「何歩分先へ飛ぶか」。

**例え**: K=3で3歩歩いた後、N=5なら「あと5歩分、さっきの方向にそのまま進む」。
つまり3歩の学習で8歩分（3+5）の効果を狙う。

**トレードオフ**:

- Nが小さい → 安全だが、節約効果が小さい
- Nが大きい → 大きく進めるが、外れるとrollback（無駄になる）

**コントローラーの動き**: `enable_random_walk: true` の時だけ、Accept時は30%の確率でNを増やす。
Reject時は50%の確率でNを減らす。
**外挿が成功しているならもっと飛びたい、失敗しているなら控えたい、という適応**。

### 6-3. alpha（外挿の歩幅）: 1歩の大きさ

```yaml
alpha_initial: 0.3   # 初期値
alpha_min: 0.03      # 最小値（これ以上は縮めない）
alpha_max: 1.5       # 最大値（これ以上は広げない）
alpha_log_sigma: 0.15  # ランダムウォークの散らし具合
```

**何をするか**: 外挿1歩あたりの更新の強さ。

**例え**: N=5が「5歩進む」なら、alphaは「1歩の歩幅」。
alpha=0.3なら小股、alpha=1.5なら大股。

**トレードオフ**:

- alphaが小さい → 保守的。外れるリスクは低いが進みが遅い
- alphaが大きい → 積極的。大きく進めるが外れる確率も高い

**コントローラーの動き**: `enable_random_walk: true` の時にだけ有効。**TG-LoRAで最も適応的に調整されるパラメータ**。

- Accept時: alpha × 1.1（10%増やす）
- Reject時: alpha × 0.5（半分にする）
- 毎サイクル: 対数正規分布でランダムに微調整（sigma=0.15は小幅な揺れ）

### 6-4. beta（velocity smoothing）: 方向の滑らかさ

```yaml
beta_initial: 0.8    # 初期値
beta_candidates: [0.5, 0.8, 0.9, 0.95]
```

**何をするか**: 過去の更新方向をどれくらい覚えておくか（EMAの係数）。

**例え**: beta=0.8なら「過去の方向を80%、今回の方向を20%混ぜる」。
beta=0.95なら「過去の方向を95%重視（なめらかな動き）」。
beta=0.5なら「過去と今回を半々（敏感に反応）」。

**トレードオフ**:

- betaが大きい → 方向が安定するが、急な変化に追従できない
- betaが小さい → 最新の方向に敏感だが、ノイズに振れやすい

**コントローラーの動き**: `enable_random_walk: true` の時に 15% の確率でランダムに候補から選び直す。
Accept/Rejectのフィードバックなし。

### 6-5. lr（学習率）: 攻めと守りの切り替え

```yaml
lr_initial: 0.0005   # 初期値（5e-4 = デフォルトより高め）
lr_min: 1.0e-5       # 最小値
lr_max: 0.001        # 最大値
lr_accept_boost: 1.2 # Accept時の倍率
lr_reject_decay: 0.5 # Reject時の倍率
```

**何をするか**: 各サイクルのpilot学習で使う学習率。`enable_random_walk: true` なら**動的に変化する**。paper-PoC の既定設定では固定値として扱う。

**例え**: 「調子が良ければもっと大胆に、失敗したら慎重に」というモード切替。

- Accept → 「うまくいってる！もっと攻めよう」→ lr増加
- Reject → 「失敗した…もっと慎重にいこう」→ lr減少 + K増加

**なぜ固定じゃないのか**:
lrを高くして攻撃的に進めることと、Kを増やして慎重に探ることは本質的に同じ軸です。

- **高lr + 小K** = 大胆に少しだけ探る。速いが粗い
- **低lr + 大K** = 慎重にたくさん探る。遅いが正確

コントローラーはAccept/Rejectに基づいて、この軸上を自動で行き来します。

**トレードオフ**:

- lrが大きい → 1歩が大きく、早く進むが外れやすい
- lrが小さい → 確実に進むが、時間がかかる

**コントローラーの動き**:

- Accept時: lr × 1.2（20%増やす）、Kを減らす可能性
- Reject時: lr × 0.5（半分にする）、Kを増やす可能性
- **lrとKは逆方向に連動**: 攻めている時はK小、守っている時はK大

### 6-6. relative_update_cap: 更新量の上限

```yaml
relative_update_cap: 0.005  # 0.5%
```

**何をするか**: 外挿で重みをどれくらい変えてもいいかの上限。
「元の重みに対して0.5%以上は変えない」という安全装置。

**例え**: 「1回の外挿で、どんなに頑張っても0.5%までしか動かさないよ」というリミッター。

**大きくする**: 大胆な更新が可能になるが、壊れるリスクも増える。
**小さくする**: 安全だが、外挿の効果が薄くなる。

alpha が `alpha_max` に張り付く run でも、実効更新がこの cap で先にクリップされているなら `alpha_max` を上げても挙動は変わらない。飽和を見た時は `alpha_max` と `relative_update_cap` のどちらが実効制約かを先に切り分ける。

### 6-7. active_layer_strategy: どの層を外挿するか

```yaml
active_layer_strategy: last_25_percent
force_top_layers_only: true
```

**何をするか**: モデルには何十もの層があるが、外挿を全層に適用するのは無駄。
どの層を外挿するかを選ぶ戦略。

**選べる戦略**:

| 戦略名                          | 説明                                                    |
| ------------------------------- | ------------------------------------------------------- |
| `last_25_percent`               | 最後の25%の層だけ。出力に近い層は微調整に敏感           |
| `last_25_percent_plus_random_2` | 最後25% + ランダムに2層。ablation / exploratory mode 用 |
| `middle_random`                 | ランダムに選ぶ。ablation / exploratory mode 用          |
| `lisa_like_weighted`            | 過去の報酬で重み付け。良かった層を優先                  |

`force_top_layers_only: true` を有効にすると、実験経路では `last_25_percent` に固定される。これは activation cache を成立させ、評価コスト削減の仮定を壊さないための制約。

---

## 7. 評価設定

```yaml
eval:
  quick_eval_examples: 32
  full_eval_every_cycles: 10
  rollback_tolerance: 0.005
  moving_avg_window: 3
  soft_accept_temperature: 0.0
```

| パラメータ                | 説明                                                                                      |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| `quick_eval_examples`     | 毎サイクルの素早い評価に使うサンプル数。32 = 32件でlossを測る。**小さいほど速いが不正確** |
| `full_eval_every_cycles`  | 何サイクルごとに全件評価するか。10 = 10サイクルに1回じっくり評価                          |
| `rollback_tolerance`      | 外挿後のlossが「どれくらい悪化するまで許容するか」。0.005 = 0.5%の悪化まではOKとみなす    |
| `moving_avg_window`       | accept/reject 判定時に参照する recent accepted loss の窓幅。評価ノイズを平滑化する        |
| `soft_accept_temperature` | 0より大きい時、境界付近の悪化を確率的に受理する。0.0 は無効                               |

**初心者向け解説**:

- `rollback_tolerance`は「少しくらい悪くなっても、たぶん全体としては良くなるだろう」という許容幅
- `moving_avg_window`は「直近数回の調子」で基準線を作る。1回だけの lucky / unlucky batch を受理判定に使わないための工夫
- `soft_accept_temperature: 0.0` は現在の paper-PoC 既定値。まずは deterministic に評価し、soft accept は ablation で足す
- 0.001（厳しい）: 外挿が少しでも悪化したら却下。安全だが外挿の機会を逃す
- 0.01（緩い）: 1%の悪化まではOK。大胆に外挿を試すが、品質が落ちるリスクあり

---

## 8. ログ設定

```yaml
logging:
  backend: mlflow
  log_every_cycles: 1
  save_every_cycles: 25
  run_dir: runs/${experiment.name}
```

| パラメータ          | 説明                                                   |
| ------------------- | ------------------------------------------------------ |
| `backend`           | ログの保存先。`mlflow`は実験管理ツール                 |
| `log_every_cycles`  | 何サイクルごとにログ出力するか                         |
| `save_every_cycles` | 何サイクルごとにモデルを保存するか                     |
| `run_dir`           | 結果の保存先。`${experiment.name}`は実験名に置換される |

---

## パラメータ関係の全体図

```text
1サイクルの流れ:
┌─────────────────────────────────────────────────┐
│  1. Pilot学習 (K回の通常LoRA学習)                  │
│     ← learning_rate, batch_size, grad_accumulation │
│                                                   │
│  2. 方向を計算 (dW = 今の重み - 前の重み)            │
│     ← beta で過去の方向を混ぜる (velocity)           │
│                                                   │
│  3. Quick評価 (pilot後のlossを測る)                 │
│     ← quick_eval_examples                         │
│                                                   │
│  4. 外挿 (見つけた方向にN歩、歩幅alphaで進む)         │
│     ← N, alpha, active_layer_strategy              │
│     ← relative_update_cap が安全装置                │
│                                                   │
│  5. Quick評価 (外挿後のlossを測る)                   │
│     ← quick_eval_examples                         │
│                                                   │
│  6. Accept / Rollback 判定                         │
│     ← rollback_tolerance                          │
│     Accept → alpha増、N増                          │
│     Reject → alpha減、N減、前の重みに戻す             │
│                                                   │
│  7. 定期的にFull評価                                │
│     ← full_eval_every_cycles                      │
└─────────────────────────────────────────────────┘
```

## sweepで探索しているパラメータ（影響度順）

| 優先度 | パラメータ                | 理由                                                                            |
| ------ | ------------------------- | ------------------------------------------------------------------------------- |
| 高     | `learning_rate`           | 学習の速さと安定性に直結。最も影響が大きい                                      |
| 高     | `rollback_tolerance`      | 外挿のaccept/reject率を決める。TG-LoRAの効率に直結                              |
| 中     | `K_initial` × `N_initial` | 1サイクルの粒度。`enable_random_walk: false` では初期値がそのまま実験条件になる |
| 低     | `alpha_initial`           | コントローラーが適応的に調整するので、初期値の影響は比較的小さい                |

adaptive branch では、`alpha_max` sweep の前に `enable_convergence_adaptation=true/false` を切る方が有益なことが多い。lr/K 漂流が主因なら、alpha の探索だけでは結論が出ない。
