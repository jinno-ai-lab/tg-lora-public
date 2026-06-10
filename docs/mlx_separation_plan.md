# MLXコード分離・整理計画書 (MLX Restructuring Plan)

## 1. 目的

現在、MLX関連のコード（学習、評価、変換、パッチ等）は、プロジェクトの `scripts/`, `src/utils/`, `src/eval/`, `tests/` に分散しており、PyTorch/CUDAベースの本体コードと混在しています。
Mac（Apple Silicon）におけるMLXネイティブ学習（Track B: Qwen3.6-35B-A3B等を含む）と、サーバー環境（Track A: Qwen3.5-9B, CUDA）での学習をより明確に分離するため、MLX関連コードを `mlx/` ディレクトリ配下に集約し、本体コードから隔離します。

これにより、以下のメリットが得られます：
- MLX固有のMetal OOM対策パッチやハックコードが本体の `src/` から切り離され、見通しが良くなる。
- Track B（MLX 4-bit QLoRA, MoE）での今後の機能拡張（expert routing等）が、`mlx/` 内で自己完結的に実施可能になる。
- 実行プラットフォームごとの依存関係の切り分けが明確になる。

---

## 2. 新しいディレクトリ構成

MLX関連コードをルート直下の `mlx/` ディレクトリに以下の通り再配置します。

```text
tg-lora/
├── mlx/
│   ├── scripts/             # MLX用各種スクリプト
│   │   ├── convert_model.py       # 旧 scripts/convert_mlx_model.py
│   │   ├── eval_downstream.py     # 旧 scripts/eval_downstream_mlx.py
│   │   ├── eval_llm_jp_eval.py    # 旧 scripts/eval_llm_jp_eval_mlx.py
│   │   ├── run_eval.py            # 旧 scripts/run_mlx_eval.py
│   │   ├── run_lora_guarded.py    # 旧 scripts/run_mlx_lora_guarded.py
│   │   ├── train_lora_fixed.py    # 旧 scripts/train_mlx_lora_fixed.py
│   │   └── train_lora_upstream.py # 旧 scripts/train_mlx_lora.py
│   ├── src/                 # MLX用モジュール・パッチ
│   │   ├── eval/
│   │   │   └── mlx_lm_backend.py  # 旧 src/eval/mlx_lm_backend.py
│   │   └── utils/
│   │       ├── gated_delta_patch.py  # 旧 src/utils/mlx_gated_delta_patch.py
│   │       └── shape_guard.py        # 旧 src/utils/mlx_shape_guard.py
│   └── tests/               # MLX関連のテストコード
│       ├── test_gated_delta_patch.py # 旧 tests/test_mlx_gated_delta_patch.py
│       └── test_shape_guard.py       # 旧 tests/test_mlx_shape_guard.py
```

---

## 3. 最新化と修正内容

### 3.1 インポートパスの変更
移動後の各スクリプト、モジュール、テストコードにおける `sys.path` 追加およびインポート定義を、新しいディレクトリ構成に合わせて修正します。
特に `mlx/` 配下のコードは `tg-lora/` のルートディレクトリを `sys.path` に含めることで、`mlx.src.utils.shape_guard` などの形式で互いをインポートできるようにします。

### 3.2 Makefile の修正
`Makefile` 内の MLX 関連ターゲットを、移動後の新しいスクリプトパスを参照するように更新します。
- `convert-mlx`
- `train-mlx`
- `train-mlx-continuous`
- `train-mlx-upstream`
- `train-mlx-smoke`
- `mlx-data`
- `eval-downstream-mlx`
- `eval-llm-jp-eval-mlx`
- `eval-mlx`

### 3.3 ドキュメントの修正
`docs/mlx_setup.md` に記載されている実行コマンドやスクリプトパスの記述を最新の配置に合わせて修正します。

---

## 4. 実行手順

1. **ディレクトリの作成**: `mlx/scripts/`, `mlx/src/eval/`, `mlx/src/utils/`, `mlx/tests/` ディレクトリを作成する。
2. **ファイルの移動**: 対象となるファイルを新しいパスへ移動する。
3. **インポートパス・コードの修正**: 移動したPythonファイルのインポートパス、および `Makefile`、`docs/mlx_setup.md` を更新する。
4. **動作検証 (Verification)**:
   - `make test` でMLX関連のユニットテストが動作することを確認。
   - `make train-mlx-smoke` でスモークテストが正常に動作することを確認。
   - 不要になった元のファイルを削除。
