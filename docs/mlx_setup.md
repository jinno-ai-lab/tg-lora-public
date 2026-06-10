# MLX 4-bit QLoRA セットアップガイド

Apple Silicon (M1/M2/M3/M4) 向けMLXネイティブ4bit量子化QLoRA学習の手順。
MPS (PyTorch + bitsandbytes) との切り替え比較が可能。

## 前提

- Apple Silicon Mac (M1以降)
- 統合メモリ 32GB以上推奨 (Qwen3.5-9B 4bitで ~28GB消費)
- Python 3.11+ ARM64ネイティブ

## クイックスタート

```bash
# 1. venv有効化
source .venv/bin/activate

# 2. 学習データ準備 (未実施の場合)
make download-data
make prepare-data

# 3. MLX用データ準備
make mlx-data

# 4. モデル変換 (初回のみ、~10分)
make convert-mlx

# 5. スモークテスト (20ステップ、~30秒) ← 推奨
make train-mlx-smoke

# 6. 短い学習 (ステップ数指定)
make train-mlx MLX_ITERS=20
```

## OOM対策: Metal resource lifetime の短縮

長時間学習で問題になるのは、単純なbyte OOMだけではなく、Metalの `MTLBuffer` / descriptor が増え続ける resource lifetime 問題。プロセス分割やcooldownではなく、`make train-mlx` は次の対策を入れた連続学習 runner を使う。

- MLX import前に command buffer の大きさを制限する。
- lazy graph traversal の幅を制限する。
- Qwen3.5 `GatedDeltaNet` の学習時 recurrent graph を chunked custom VJP にする。
- loss/token集計をMLX arrayではなくPython scalarにして、step間でgraphを保持しない。
- cacheは毎step破壊せず、上限付きで再利用する。

詳細な切り分け、PR/Issue状況、upstream MLXで必要なallocator修正方針は [MLX Metal OOM 根本原因調査メモ](./mlx_metal_oom_root_cause.md) を参照。

検証済みの改善:

| window | 修正前 Peak | 修正後 Peak |
| --- | ---: | ---: |
| step 40 | 76.5GB | 17.5GB |
| step 70 | 137.0GB | 26.3GB |
| step 90 | 117.5GB | 23.6GB |

修正後は `MLX_GATED_DELTA_CHUNK=512`, `max_seq_length=2048`, `grad_accumulation_steps=8` で 100 step 完走し、`runs/mlx_verify_100step_chunk512_vjp/adapters.safetensors` を保存した。

関連:

- [mlx#3327](https://github.com/ml-explore/mlx/issues/3327): int32 shape-product overflow。長時間学習のresource leakとは別件。
- [mlx#3524](https://github.com/ml-explore/mlx/pull/3524): #3327 の修正。mainにmerge済み。
- [mlx-lm#1185](https://github.com/ml-explore/mlx-lm/issues/1185): Qwen3.5 LoRA training の resource 問題。
- [mlx#3464](https://github.com/ml-explore/mlx/pull/3464): maintainerが `MLX_MAX_OPS_PER_BUFFER` / `MLX_BFS_MAX_WIDTH` と big-MTLBuffer allocator を示唆。

## 各ターゲットの説明

### `make convert-mlx`

HuggingFaceモデルをMLX 4bit形式に変換。モデルごとに1回だけ実行。

- 入力: `Qwen/Qwen3.5-9B` (HF Hub から自動ダウンロード)
- 出力: `.cache/mlx_models/Qwen--Qwen3.5-9B/` (~4.7GB)
- `--` (ダッシュ2つ) は `Qwen/Qwen3.5-9B` の `/` を置換したもの
- モデル指定変更: `make convert-mlx BASE_MODEL=Qwen/Qwen2.5-7B`

### `make mlx-data`

MLX-LMが読める形式 (`train.jsonl` + `valid.jsonl`) でデータを用意。
既存の `data/train.jsonl` と `data/valid_quick.jsonl` へのシンボリックリンクを作成。

### `make train-mlx`

MLX 4bit QLoRA学習。長時間学習時のMetal resource増殖を避けるため、`mlx/scripts/train_lora_fixed.py` を使う。プロセス分割やcooldownではなく、MLX import前の command-buffer / graph traversal 制限と、step境界でのlazy graph参照破棄で対処する。

デフォルトのMetal resource制限:

- `MLX_MAX_OPS_PER_BUFFER=4`
- `MLX_MAX_MB_PER_BUFFER=32`
- `MLX_BFS_MAX_WIDTH=4`
- `MLX_GATED_DELTA_CHUNK=512`

より安定寄りにする場合:

```bash
make train-mlx \
  MLX_MAX_OPS_PER_BUFFER=2 \
  MLX_MAX_MB_PER_BUFFER=16 \
  MLX_BFS_MAX_WIDTH=2 \
  MLX_GATED_DELTA_CHUNK=512
```

PR #3524 相当の int32 shape-product overflow ガードも有効化されるが、これは長時間学習のMetal resource問題とは別件。

デフォルトパラメータ (MPS baseline と同じ):
- LoRA r=16, alpha=32 (scale=2.0), 全32層
- batch_size=1, grad_accumulation=8
- learning_rate=2e-4, linear decay
- max_seq_length=2048

ステップ数変更: `make train-mlx MLX_ITERS=20`

### `make train-mlx-smoke`

動作確認用の20ステップ実行。~30秒で完了。

### `make compare-mlx`

MPS baseline と MLX baseline の結果を比較。

```bash
make compare-mlx \
  BASELINE_RUN=runs/qlora_9b_baseline_20260526_120000 \
  MLX_RUN=runs/mlx_qlora_20260527_003000
```

## MPS との切り替え

同じデータ・同じハイパーパラメータで、バックエンドだけ切り替えて比較できます。

```bash
# MPS (PyTorch + bitsandbytes 4bit)
make train-baseline

# MLX (Apple Silicon ネイティブ 4bit)
make train-mlx
```

### 環境変数による切り替え

PyTorch側のスクリプトは `TG_LORA_BACKEND` 環境変数でデバイスを強制変更できます:

```bash
# 通常 (MPS自動検出)
make train-baseline

# MLXモード (PyTorch側はCPUにフォールスルー)
TG_LORA_BACKEND=mlx make train-baseline
```

## 出力場所

| バックエンド | 出力ディレクトリ | 内容 |
|---|---|---|
| MPS | `runs/qlora_9b_baseline_*` | adapter + run_metrics.jsonl |
| MLX | `runs/mlx_qlora_*` | adapters.safetensors |
| MLX (smoke) | `runs/mlx_smoke_*` | adapters.safetensors |

## アーキテクチャ比較

| 項目 | MPS + bitsandbytes | MLX |
|---|---|---|
| 量子化 | NF4 (bitsandbytes) | INT4 group-wise (MLX独自) |
| メモリモデル | CPU-GPU間コピー | 統合メモリ (ゼロコピー) |
| 推論速度 | ~30-50 tok/s | ~35-70 tok/s |
| 学習対応 | 安定 (ただし低速) | bounded runnerで対応 |
| インストール | bitsandbytes >= 0.41 | mlx-lm >= 0.21 |

## トラブルシューティング

### `make convert-mlx` が遅い

9Bモデルの変換は~10分かかります。HF Hubからのダウンロードが含まれる場合はさらに時間がかかります。
2回目以降はHFキャッシュが使われます。

### メモリ不足 (OOM)

まず command buffer / graph 幅を下げる。

```bash
make train-mlx \
  MLX_MAX_OPS_PER_BUFFER=2 \
  MLX_MAX_MB_PER_BUFFER=16 \
  MLX_BFS_MAX_WIDTH=2 \
  MLX_GATED_DELTA_CHUNK=512
```

それでも同じ場所で落ちる場合は、1 step の実ピークが大きすぎる可能性が高い。`max_seq_length`、`num_layers`、`grad_accumulation_steps` を下げて切り分ける。詳細は [MLX Metal OOM 根本原因調査メモ](./mlx_metal_oom_root_cause.md)。

### `ModuleNotFoundError: No module named 'mlx'`

```bash
pip install "mlx-lm[train]>=0.21"
```

### Qwen3.5がサポートされているか確認

```bash
python -c "from mlx_lm import models; import os; print([f for f in os.listdir(os.path.dirname(models.__file__)) if 'qwen' in f])"
# qwen3_5.py が含まれていればOK
```
