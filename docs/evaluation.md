# 評価ガイド

## 評価の3層構造

### 1. 学習中評価 (Training Metrics)

学習ループ内で自動実行される:

- **loss**: cross-entropy loss の推移
- **grad_norm**: 勾配ノルム（学習安定性）
- **velocity_stats**: TG-LoRA固有 — 速度ベクトルの統計
- **extrapolation_error**: 外挿の精度

設定: `configs/*.yaml` の `logging.log_every_steps`

### 2. チェックポイント評価 (Quick/Full Eval)

`eval.quick_eval_examples` 件でのloss計算:

```yaml
eval:
  quick_eval_examples: 32
  full_eval_every_cycles: 10
  rollback_tolerance: 0.005
  moving_avg_window: 3
  soft_accept_temperature: 0.0
```

- `quick_eval_examples`: 毎サイクルの accept / rollback 判定に使う軽量評価
- `full_eval_every_cycles`: ベストモデル保存と early stopping に使う重い評価
- `moving_avg_window`: recent accepted loss の移動平均で判定基準を平滑化し、false accept を抑える
- `soft_accept_temperature`: 境界付近の悪化を確率的に受理する実験用パラメータ。現行 paper-PoC では 0.0 で無効

### 3. ベンチマーク評価 (lm-evaluation-harness)

学習完了後に標準ベンチマークで定量評価。当プロジェクトでは
MLXバックエンド付きのカスタム lm-evaluation-harness を使用する。

```bash
# MLXバックエンドで評価（デフォルト）
make eval ADAPTER_PATH=runs/mlx_qlora_baseline_500

# ベースモデルのみ（アダプタなし）
make eval-base

# アダプタ付きで評価
make eval-mlx ADAPTER_PATH=runs/mlx_qlora_baseline_500
```

## カスタム lm-evaluation-harness

`~/lm-evaluation-harness` に MLX バックエンドを追加したフォークを配置している。
Apple Silicon (MLX) と NVIDIA (HF) の両方で動作する。

### セットアップ

```bash
# 初回: フォークをeditable install
make install

# 手動の場合
pip install -e ~/lm-evaluation-harness
```

### 使用方法

```bash
# MLXモデル + アダプタ
lm_eval --model mlx \
  --model_args "model=.cache/mlx_models/Qwen--Qwen3.5-9B,adapter_path=runs/mlx_qlora_baseline_500" \
  --tasks arc_easy,hellaswag,truthfulqa_mc2 \
  --batch_size 1

# MLXモデルのみ（アダプタなし）
lm_eval --model mlx \
  --model_args "model=.cache/mlx_models/Qwen--Qwen3.5-9B" \
  --tasks arc_easy,hellaswag,truthfulqa_mc2

# HFバックエンド（CUDA環境、従来方式）
lm_eval --model hf \
  --model_args "pretrained=Qwen/Qwen3.5-9B,dtype=float16" \
  --tasks arc_easy,hellaswag,truthfulqa_mc2
```

### MLXバックエンドの仕組み

`lm_eval/models/mlx_lm.py` に `MLXLMEval` クラスを実装している:

- `loglikelihood` — ARC/HellaSwag/TruthfulQA 等のmultiple-choiceタスク
- `loglikelihood_rolling` — perplexity計算
- `generate_until` — GSM8K等の生成タスク
- LoRAアダプタを `adapter_path` で直接ロード可能（マージ不要）
- `--model mlx` で `MODEL_MAPPING` に登録済み

### なぜカスタムフォークを使うのか

1. **MLXネイティブ推論**: 4-bit量子化モデルをApple Siliconで高速評価。PyTorch HF経由よりも2-10x高速
2. **LoRA直接評価**: アダプタのマージ・保存が不要。safetensorsを直接ロード
3. **クロスプラットフォーム**: `--model mlx` でApple Silicon、`--model hf` でNVIDIA。同じタスク・指標を共有
4. **メモリ効率**: 9Bモデルを4-bit MLXで約5.5GBで評価可能

## Makefile ターゲット一覧

| ターゲット | 説明 | 使用例 |
| --- | --- | --- |
| `eval` | MLXバックエンドで評価（デフォルト） | `make eval ADAPTER_PATH=runs/...` |
| `eval-mlx` | MLXモデル+アダプタ評価 | `make eval-mlx ADAPTER_PATH=runs/...` |
| `eval-base` | ベースモデルのみ評価 | `make eval-base` |
| `eval-lora` | HF経由で評価（CUDA、従来方式） | `make eval-lora ADAPTER_PATH=runs/...` |

変数:

| 変数 | デフォルト | 説明 |
| --- | --- | --- |
| `LM_EVAL_HARNESS` | `~/lm-evaluation-harness` | カスタムフォークのパス |
| `LM_EVAL_MODEL` | 自動解決 | MLXモデルパス |
| `EVAL_TASKS` | `arc_easy,hellaswag,truthfulqa_mc2` | 評価タスク |
| `EVAL_OUTPUT` | `reports/eval` | 出力ディレクトリ |

## ベンチマーク一覧

| タスク         | 内容                 | 指標        | タイプ         | 備考       |
| -------------- | -------------------- | ----------- | -------------- | ---------- |
| arc_easy       | 科学推論 (Easy)      | acc_norm    | loglikelihood  | 25-shot    |
| arc_challenge  | 科学推論 (Challenge) | acc_norm    | loglikelihood  | 25-shot    |
| hellaswag      | 常識推論             | acc_norm    | loglikelihood  | 10-shot    |
| gsm8k          | 算数 (8K)            | exact_match | generate_until | 5-shot CoT |
| truthfulqa_mc2 | 事実正確性           | mc2         | loglikelihood  | 0-shot     |
| mmlu           | 知識全般             | acc         | loglikelihood  | 5-shot     |

初期検証では `arc_easy,hellaswag,truthfulqa_mc2` の3つで十分。
`gsm8k` はgenerate_untilタスクのため推論時間が長い。

## 比較実験の設計

| 条件 | 設定 |
| --- | --- |
| Baseline | QLoRA (`configs/9b_baseline.yaml`) |
| TG-LoRA | deterministic mainline (`configs/9b_tg_lora_paper_poc.yaml`) |
| Ablation | adaptive branch (`configs/9b_tg_lora_adaptive_k5.yaml`) |
| Ablation | adaptive no-conv (`configs/9b_tg_lora_adaptive_k5_no_conv.yaml`) |
| データ | 同一 (Dolly 3k train / 300 valid) |
| モデル | Qwen3.5-9B |
| 評価 | 同一ベンチマークスイート |

`configs/9b_tg_lora.yaml` は current mainline だが moving target でもある。比較表や論文図表を作る時は、固定名の config を使う。

## 比較実験の衛生条件

- `quick_eval_examples` を揃える。32 vs 64 のような差があると accept / rollback ノイズの量が変わる。
- full eval cadence を揃える。`best_valid_loss` の更新頻度が違うと best checkpoint 比較が歪む。
- completed run のみを比較対象にする。`run_footer` がなく `best_model` も保存されていない run は途中観測として扱う。
- adaptive K branch は planned cycle budget ではなく、`run_metrics.jsonl` の `total_backward_passes` を使って比較する。
- deterministic mainline と historical adaptive branch を同じ「TG-LoRA」とだけ書いて混ぜない。config 名まで明記する。

`scripts/run_ablation_suite.sh` は、上の衛生条件を満たすように baseline / paper-PoC / adaptive / adaptive-no-conv を同じ launcher から起動するための下準備である。

## 結果の解釈

期待される比較ポイント:

1. **学習効率**: 同じステップ数でのloss収束速度
2. **最終性能**: ベンチマークスコアの差
3. **安定性**: loss曲線の振動、ロールバック発生回数、false accept の有無
4. **計算コスト**: wall-clock time、GPU使用率、cache hit rate

TG-LoRA では `loss_after <= loss_pilot` だけでなく、recent accepted loss の移動平均も判断材料として使う。したがって quick eval のログを読む時は、単発の loss だけでなく recent trend を見る必要がある。

adaptive branch を読む時はさらに、`enable_convergence_adaptation` が有効だったかを確認する。有効なら lr 減衰と K 増加が extrapolation 自体とは独立に働く。

## トラブルシューティング

### MLXモデルが見つからない

```bash
# モデルを変換
make convert-mlx

# パス確認
make convert-mlx-path
```

### OOM (Out of Memory)

```bash
# MLXバックエンドは4-bit量子化で9Bを約5.5GBで動かす
# HFバックエンドの場合はbatch_sizeを下げる
bash scripts/run_eval.sh MODEL --batch-size 1
```

### LoRAアダプタの評価（HF経由、CUDA環境向け）

`eval-lora` はPEFTでマージしてからHFバックエンドで評価する。
MLX環境では `eval-mlx` または `eval` を使うこと。
