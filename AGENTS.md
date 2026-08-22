# AGENTS.md — TG-LoRA 運用指示書

## Loop halt pre-flight（必須）

§4 / MS-008 軸は awaiting_ratification 中 — axis work の前に必ず `make loop-halt-check`
を実行すること（判定は `verdict rc=77`=SKIP → 何も produce しない / `verdict rc=0`=PROCEED。
make の非ゼロ終了は SKIP ではなく pre-flight の故障（BROKEN）。正本:
`scripts/loop_halt_guard.py` + `loop_axis_state.json`）。

## プロジェクト概要

**TG-LoRA** (Tangent-Gradient LoRA): 勾配速度ベクトルの外挿によるLoRA学習の効率化手法。

- 学習中のLoRA重み変化の速度（velocity）を追跡
- 外挿（extrapolation）で次ステップの重みを予測し、学習効率を向上
- レイヤーサンプリングで計算コストを削減
- ロールバック機構で不安定な学習を自動回復
- ランダムウォークによるハイパーパラメータ（K, N, alpha, beta）の適応探索

## ディレクトリ構成

```text
tg-lora/
├── configs/              # Hydra/OmegaConf 学習設定
│   ├── 9b_baseline.yaml  # QLoRA ベースライン
│   └── 9b_tg_lora.yaml   # TG-LoRA 実験設定
├── data/                 # データセット（git管理外）
│   ├── seed/             # シードプロンプト
│   ├── generated/        # LLM生成データ
│   └── filtered/         # フィルタ済みデータ
├── scripts/              # 運用スクリプト
│   ├── download_data.py  # データセットダウンロード
│   ├── prepare_data.py   # データ前処理・分割
│   ├── inspect_model.py  # モデル構造検査・LoRA対象モジュール推奨
│   ├── compare_runs.py   # ベースライン/TG-LoRA比較レポート生成
│   └── run_eval.sh       # lm-evaluation-harness 実行
├── src/
│   ├── tg_lora/          # コアアルゴリズム
│   │   ├── velocity.py           # 勾配速度追跡
│   │   ├── extrapolator.py       # 重み外挿
│   │   ├── delta_tracker.py      # 重み差分追跡
│   │   ├── layer_sampler.py      # レイヤーサンプリング
│   │   ├── rollback_manager.py   # ロールバック制御
│   │   ├── random_walk_controller.py  # ハイパーパラメータ適応探索
│   │   ├── lora_state.py         # LoRA状態管理
│   │   └── metrics.py            # 学習メトリクス
│   ├── (data/)           # ← private pipeline は public mirror で意図的 strip 済み
│   │                       #   （src/data は不在。依存 tests ~130件 = pre-existing 失敗・非回帰）
│   ├── training/         # 学習ループ
│   │   ├── train_baseline_qlora.py
│   │   ├── train_tg_lora.py
│   │   ├── trainer_loop.py
│   │   └── loss.py
│   ├── eval/             # 評価
│   │   ├── eval_loss.py          # Loss評価
│   │   ├── eval_task.py          # タスク評価
│   │   └── eval_format.py        # 出力フォーマット評価
│   └── utils/            # ユーティリティ
│       ├── io.py
│       ├── logging.py
│       ├── memory.py
│       ├── run_metrics.py       # JSONL構造化ログ
│       └── seed.py
│   └── model/             # モデル読み込み・LoRAユーティリティ
│       ├── load_model.py        # 4bit量子化モデル読み込み
│       └── lora_utils.py        # LoRAパラメータ操作
├── tests/                # ユニットテスト
├── reports/              # 実験レポート（git管理外）
├── runs/                 # MLflow等ログ（git管理外）
├── docs/                 # ドキュメント
│   ├── paper/            # 論文作成ハブ（入口のみ。正本は複製しない）
│   │   ├── README.md     # 論文作業の入口
│   │   ├── 01_inputs.md  # 論文執筆に必要な入力資料の正本リンク集
│   │   ├── 02_source_data.md  # 論文の主張を支える raw artifact へのリンク集
│   │   └── 03_writing_map.md  # sectionごとの参照先マップ
│   ├── paper_experiment_plan.md   # claim ladder / gate の正本
│   ├── paper_results_snapshot.md  # 論文転記用 canonical numbers の正本
│   ├── eval_plan_and_status.md    # 現在の gate status の正本
│   └── paper_docs_index.md        # 旧入口。docs/paper/README.md への redirect
├── Makefile              # 共通コマンド
└── pyproject.toml        # パッケージ定義
```

## 論文作成ハブ

論文作成に入るときの AI 向け入口は [docs/paper/README.md](docs/paper/README.md) です。

- `docs/paper/` は一元ハブだが、数値・設定・artifact 本体を複製しない
- 正本は既存の `docs/`, `configs/`, `data/`, `runs/`, `scripts/` に残す
- `docs/paper/01_inputs.md` は執筆に必要な入力資料の正本リンク集
- `docs/paper/02_source_data.md` は論文の主張を支える raw artifact へのリンク集
- `docs/paper/03_writing_map.md` は section ごとの参照先マップ
- 既存の [docs/paper_docs_index.md](docs/paper_docs_index.md) は互換用 redirect として扱う

論文向けの編集や要約を行うときは、まず `docs/paper/README.md` から辿り、同じ情報を別ファイルへ複製しないこと。

## 開発フロー

### 環境構築

```bash
make setup             # conda環境作成 + 全依存インストール（要conda）
# または既存venvの場合:
make install
```

### 初回セットアップ（必須）

```bash
make inspect           # Qwen3.5-9Bの層構造・LoRA対象モジュール名を確認
make download-data     # 公開データセットダウンロード
make prepare-data      # データ前処理・JSONL変換・分割
```

### 初期検証フェーズ

1. **公開データセットで動作確認** — Dolly 15k, Capybara
2. **ベースラインQLoRA** — 標準学習と比較する基準
3. **TG-LoRAの挙動確認** — velocity追跡、外挿の効果測定
4. **lm-evaluation-harnessで定量評価** — ARC, HellaSwag, GSM8K, TruthfulQA

### 現行路線と品質状況（2026-08-22 40系定期品質分析）

**現行路線 = 第6期 Progressive Freezing + Activation Matching**（SYSTEM_CONSTITUTION.md / GOAL §1.6）。
第1期〜第5期（velocity 外挿 / 漸進ランク ZO / B-filter / PSA 転換）は帰無基準により**棄却または保留済み**。
旧「Component 2（Prior-based Subspace Learning）設計移行・現在最優先」記述（2026-06-05）は
この優先順位が撤回された遺物 — `scripts/collect_true_gradients.py` / `offline_tg_w_validation.py`
は保留置きのツールとして残存するが、新規作業の優先指示ではない。

**現状（2026-08-22時点）要約:**
- §4 verdict: **SHIP landed**（2026-07-29 `section4_landed_decision.json` — 両 leg TIES at full budget
  〔homogeneous cand 1.6947 vs surr 1.6960 / heterogeneous cand 1.7180 vs surr 1.7191〕+
  P1 品質保持 SURPASSES 両 leg）
- 分類Aタスク（コードで解決可能）: **枯渇** — MS-PF2 4/4・MS-PF3 2/2・MS-PF4 1/1 の全 Cat-A 変換完了。
- テスト（2026-08-22 commit 858273b 実測・`--continue-on-collection-errors`）:
  **4991 passed / 102 failed / 10 skipped / 3 xfailed / 17 collection errors**。
  失敗・エラーは全て private `src.data` 剥離 + `peft` 等ヘビー依存不在に由来する **pre-existing・非回帰**。
  品質 canary は `tests/test_cli_help_smoke.py`（**43 passed / 3 xfailed**）。
- 品質保証体制: `make ci` = ruff（src/tests/scripts/mlx）+ spine anchor 159 本 + gates-ci + pytest。
- コミット比率（直近30件）: コード系（feat+fix+test+refactor）16/30 = **53%**（改善水準 ≥50%）。
- Loop halt pre-flight: **SKIP 判定**（`awaiting_ratification`）→ §4/MS-008 軸の axis work 禁止（冒頭節参照）。

**次の最適行動:**
- **人間（operator）**: (1) `loop_axis_state.json` の 3 trigger を ratify（新規 9B deposit /
  closeout 承認 / 新 MS 軸 open）。(2) post-SHIP default 変更 D1/D2/D3 の決定
  （`specs/oper-decision-surface/default-change-proposal.md` — 決定まで mainline freeze は OFF のまま）。
  (3) private repo での production-baseline 絶対 loss 計測（Cat-C・本 mirror から実行不可）。
- **AI**: 分類A枯渇 + halt SKIP 中 = 新規 axis work は行わない。§4 verdict run の再発火は
  TIES の再現のみで新情報を生まない。`PURPOSE.md` 次の一手への追記は freeze guard
  （`tests/test_purpose_next_steps_freeze.py`）で CI 強制阻止済み。

---
**本節は2026-08-22時点の40系定期品質分析（実測値: pytest フル実行 / git log / make loop-halt-check）に基づく。**


### データ戦略

| フェーズ | データ | 規模 | 目的 |
| -------- | ------------------ | ------------------------ | ---------------- |
| 現在 (論文) | Dolly 15k (5K subset) | train: 5,000 / valid: 500 / test: 500 | 論文実験 (seed=42, 3-way split) |
| 拡張検証 | Capybara | ~16k | SFT動作確認 |
| 本番 | 自社データ | TBD | 本番品質の学習 |

### 評価戦略

- **学習中**: loss, grad_norm, velocity stats (quick eval 64件)
- **チェックポイント**: full eval (valid全件)
- **最終評価**: lm-evaluation-harness (ARC, HellaSwag, GSM8K, TruthfulQA)

## コマンド一覧

```bash
make setup             # conda環境 + 全依存（推奨）
make inspect           # モデル構造確認・LoRA対象モジュール一覧
make download-data     # 公開データセットダウンロード
make prepare-data      # データ前処理・JSONL変換・分割
make train-baseline    # QLoRAベースライン学習
make train-tg-lora     # TG-LoRA学習
make eval              # lm-evaluation-harness実行
make test              # ユニットテスト
make lint              # コード品質チェック
```

## 技術スタック

- **モデル**: Qwen3.5-9B (Track A, CUDA 4bit QLoRA) / Qwen3.6-35B-A3B (Track B, MLX 4bit QLoRA, MoE)
- **学習**: PyTorch + Transformers + PEFT + bitsandbytes (Track A) / MLX (Track B)
- **設定**: Hydra + OmegaConf
- **実験管理**: RunMetrics (JSONL) — MLflow依存ありだが未統合
- **評価**: lm-evaluation-harness
- **データ**: HuggingFace Datasets

## 制約・ルール

- `data/`, `runs/`, `reports/` はgit管理外
- 実験設定は `configs/` に YAML で管理
- モデルチェックポイントは `runs/<experiment_name>/` に保存
- 公開データセットの利用条件を遵守する
- 初期検証では自社データを使用しない

## Process exit codes（trainer → control plane）

トレーナ(`src/training/train_tg_lora.py` / `train_baseline_qlora.py`)が吐くプロセス終了コードの**生産者側**契約。分類器(`scripts/frontier_report.determine_status`)はこれを読んで run を `completed` / `oom` / `failed` に分類する。

| exit code | 意味 | 再試行可否 |
|-----------|------|-----------|
| `0` | 正常終了（summary あり） | — |
| `2` | **実故障**（`numerical_instability` / `cuda_error`）。fault checkpoint 保存後に終了 | ❌ 再試行で再現する実障害。縮小再試行の対象外 |
| `3` (`OOM_EXIT_CODE`) | **繰延可能な GPU OOM**。graceful handler が OOM を捕獲し fault checkpoint を保存済み = バッチ縮小で再開可能 | ✅ **defer and retry**（batch / seq_len を縮小して再実行） |
| `137` | kernel OOM-killer による SIGKILL（graceful でない強制殺） | ⚠️ OOM だが checkpoint は保証されない |

- **`3` の位置づけ**: graceful handler が OOM を捕獲できた（=復旧 checkpoint あり・縮小再開安全）ことのシグナル。`2` と明示的に区別するのは、OOM は「縮小すれば通る」繰延可能故障だが、数値不安定/CUDA-error は「縮小しても再現する」実故障だから — これを exit code だけで区別できなければ OOM 対策は void に繰延される。
- **`3` を「defer and retry」と読む制御系の解釈は、operator / AI-Hub control plane の domain**（本 repo 外）。ここが保証するのは、生産者が上記の区別された code を**確実に吐く**こと、と分類器がそれを `oom` として**読む**こと（log text "out of memory" / `\bOOM\b` を副次経路として併用）。
- **両 trainer で実装済み（symmetric）**: `train_tg_lora.py`（`numerical_instability` / `cuda_error` / `oom` の 3 値）と `train_baseline_qlora.py`（`cuda_error` / `oom` の 2 値・`is_gpu_oom_error()` で判定）が共に `src/utils/device.fault_exit_code()` 経由で区別された code を吐く。かつて baseline の graceful-OOM handler は fault checkpoint を保存した後 bare `raise`（exit 1）していたため、上記契約を**文書は両 trainer について謳うのに実装は TG のみ**という doc-vs-impl drift だった — 現在は閉包。`tests/test_fault_exit_contract.py` が両 trainer について static guard で pin する（bare `raise` / `SystemExit(2)` 硬 encoding への退行を妨ぐ）。

## Qwen3.5-9B 固有の注意点

- **LoRA target**: `target_modules="all-linear"` を指定（ハイブリッド構造の全Linear層に自動適用）
  - 手動で層名を列挙するとDeltaNet層が漏れる。`all-linear`でPEFTが自動検出
- **trust_remote_code 不要**: Qwen3.5はtransformersコアにネイティブ統合済み
- **ライブラリ要件**: 最新のtransformers/peft/accelerateが必要（`make setup`で対応）
- **ハイブリッド構造**: 32層中24層がGated DeltaNet（線形注意）、8層が標準Attention
- **マルチモーダル**: VLMアーキテクチャ。`AutoModelForCausalLM`で言語モデル部分のみ取得可能
- **thinking mode**: デフォルトで思考出力を生成。SFTではnon-thinkingモードを使用
- **VRAM**: 9Bの4bit QLoRAはRTX3060 12GBに収まる（LoRA分を含めて約8-10GB）

## LLM Wiki

Read `docs/llm-wiki/index.md` for high-level repository memory. Treat purpose, specs, and source files as canonical.

論文作業については LLM Wiki よりも `docs/paper/README.md` とそこから辿る canonical source を優先すること。

## 自律動作ガイドライン (Autonomy Guidelines)

本プロジェクトを推進する AI エージェントは、ユーザーの逐次的な指示を待つことなく、自律的かつ能動的にタスクを推進する責任を持ちます。以下のデフォルト動作ルールを遵守してください。

### 1. セッション開始時の自己診断 (Auto-Diagnostics)
* 新しいセッションが開始された、または次の指示が不明確な場合、直ちに以下のコマンドを実行してプロジェクトの現状と次に必要なステップを自律的に確認してください。
  ```bash
  make check-status
  ```
* 出力された診断結果に基づいて、データセットの準備状況、過去の実験結果の有無、および未完了のマイルストーンを確認します。

### 2. 能動的なタスクの進行 (Proactive Execution)
* `make check-status` の診断結果から「次に何を行うべきか」を決定し、ユーザーへその提案を行い、可能であれば自律的に実験コマンド（例：`make prepare-data` や `make paper-memory`）を実行してください。
* 指示待ちにならず、常にマイルストーンの進捗を進めることを最優先とします。

