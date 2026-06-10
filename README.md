# TG-LoRA: Tangent-Gradient LoRA

勾配速度ベクトルの外挿によるLoRA学習効率化手法。

## Quick Start

```bash
# 環境セットアップ
make install

# 公開データセットのダウンロードと前処理
make download-data
make prepare-data

# この 12GB CUDA マシンの標準既定値は 1024 token
# Apple Silicon の MLX 導線は 2048 token 既定

# ベースライン学習
make train-baseline

# TG-LoRA学習
make train-tg-lora

# 評価
make eval
```

## アルゴリズム概要

TG-LoRAは標準LoRA学習に以下を追加する:

1. **Velocity Tracking** — LoRA重み更新の速度ベクトルを追跡
2. **Extrapolation** — 速度から次ステップの重みを外挿予測
3. **Layer Sampling** — 重要度に応じてレイヤーをサンプリング
4. **Rollback** — 学習不安定時に自動ロールバック
5. **Adaptive Control (Optional)** — ランダムウォークや収束適応でK, N, alpha, beta, lrを探索的に動かせる

## Experiment Surfaces

- `configs/9b_tg_lora.yaml`: current mainline。deterministic な paper-PoC 既定設定。
- `configs/9b_tg_lora_paper_poc.yaml`: paper-PoC の固定名コピー。比較実験で moving target を避けたい時に使う。
- `configs/9b_tg_lora_adaptive_k5.yaml`: historical adaptive branch。`enable_random_walk=true`、`K_initial=5`、ランダム層戦略あり。
- `configs/9b_tg_lora_adaptive_k5_no_conv.yaml`: adaptive branch から `enable_convergence_adaptation` だけを切った ablation 用設定。
- `configs/9b_tg_lora_optimizer_reuse_experimental.yaml`: optimizer を再生成せず、AdamW state を in-place zero reset して再利用する experimental surface。
- `configs/9b_tg_lora_prefix_feature_cache_experimental.yaml`: suffix-only trainable mode。後半25%の LoRA だけを学習対象に固定し、前半 prefix hidden states を CPU RAM に事前展開して train/eval の forward を短絡する experimental surface。現設定は amortization を優先して `train` cache を切り、`valid_quick` / `valid_full` の cache を主対象にしている。cache blob は `training.prefix_feature_cache_dir` 配下へ永続化されるので、同一 dataset / 同一 split 条件の2回目以降は disk hit で再利用される。
- `configs/9b_baseline_suffix_only_last25.yaml`: apples-to-apples 比較用の suffix-only baseline。LoRA trainable scope だけを最後の25%に合わせ、cache を使わない標準 QLoRA 対照。
- `scripts/run_ablation_suite.sh`: baseline / paper-PoC / adaptive / adaptive-no-conv を同じ eval hygiene で起動する launcher。
- `scripts/benchmark_optimizer_lifecycle.py`: recreate-per-cycle と reuse-state-reset の steady-state overhead を比較する benchmark。
- `scripts/benchmark_prefix_cache.py`: prefix feature cache の cold/warm 比較を 2 連続で実行し、persistent cache reuse の build/load 差分を `summary.json` に集約する benchmark。

current mainline と historical adaptive branch は別物として扱う。adaptive K の run は planned cycle budget ではなく、実測の `total_backward_passes` を使って解釈する。

optimizer lifecycle 実験は論文 mainline とは別扱いにする。狙いは optimizer state drift の解消ではなく、cycle ごとの AdamW state 再確保を避けることにある。

prefix feature cache 実験も論文 mainline とは別扱いにする。これは現行 mainline の一時 activation cache を拡張したものではなく、前半層を固定した suffix-only 学習へ問題設定そのものを切り替える experimental mode である。CPU RAM を forward cache の保管先に使う代わりに、`lora.dropout=0.0` と `training.trainable_lora_scope=last_25_percent` を前提にする。さらに wall-clock 評価を歪めないため、precompute 時間も run metrics に含める。現在は build 後の prefix feature cache を disk に保存するので、同じ config と dataset を再実行した2回目は build をスキップして load だけで進む。

cache の cold/warm 差分を測るときは `make compare-prefix-coldwarm CACHE_DIR=.cache/prefix_feature_cache_compare_smoke ...` を使う。1回目が cold、2回目が同じ persistent cache dir を warm reuse する。

同じ検証を summary.json 付きで自動化したいときは `make bench-prefix-cache BUDGET=32 MAX_SEQ_LEN=256 QUICK_EVAL_EXAMPLES=4 EVAL_POINTS=1 CACHE_DIR=.cache/prefix_feature_cache_benchmark_bp32_s256_e1` を使う。

## プロジェクト構成

詳細は [AGENTS.md](AGENTS.md) を参照。

## モデル

**Qwen3.5-9B** — ハイブリッドアーキテクチャ（Gated DeltaNet + Gated Attention）。
32層（24 DeltaNet + 8 Attention層）、4096 hidden、248K vocab。
4bit QLoRAでRTX3060 12GBに収まる構成。

## 初期検証データ

| データ                | 用途             | 規模                              |
| --------------------- | ---------------- | --------------------------------- |
| Dolly 15k (subset)    | SFT学習・検証    | train 3k / valid 300              |
| Capybara              | 拡張学習データ   | ~16k                              |
| lm-evaluation-harness | ベンチマーク評価 | ARC, HellaSwag, GSM8K, TruthfulQA |

## Docker

Docker による再現可能な開発環境（要 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)）。

```bash
# イメージビルド
docker compose build

# テスト実行
docker compose run --rm tg-lora pytest tests/ -v

# インタラクティブセッション
docker compose run --rm tg-lora bash

# 学習（GPU使用）
docker compose run --rm tg-lora make train-tg-lora
```

データ (`data/`)、実験出力 (`runs/`)、モデルキャッシュはボリュームマウントされるため、コンテナを再ビルドしても保持されます。

## 評価

```bash
# lm-evaluation-harness による標準ベンチマーク
make eval

# 学習中のquick eval
# configs/*.yaml の eval.quick_eval_examples で制御
```
