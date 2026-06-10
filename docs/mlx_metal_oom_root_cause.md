# MLX Metal 長時間学習 OOM / resource leak 調査と修正

作成日: 2026-05-28

対象: `Qwen/Qwen3.5-9B` を MLX 4bit 形式に変換し、MLX で LoRA / QLoRA 学習する経路。

## 結論

tg-lora の MLX 経路は `src/training/train_tg_lora.py` の TG-LoRA 実装ではなく、`Makefile` の `train-mlx` から `scripts/train_mlx_lora_fixed.py` を呼び、内部で `mlx_lm` の LoRA utilities を使う QLoRA baseline 経路である。したがって今回の OOM 解析対象は TG-LoRA の extrapolation / rollback ではなく、この MLX QLoRA runner と MLX Metal backend の境界。

今回直すべき本体は PR [mlx#3524](https://github.com/ml-explore/mlx/pull/3524) の int32 shape overflow ではない。#3524 は issue [mlx#3327](https://github.com/ml-explore/mlx/issues/3327) の「2^31 要素以上の shape product が int32 で wrap して巨大 allocation に見える」問題を直すもの。これは `18446744069414600704 bytes` のような wrapped allocation を出すが、長時間 LoRA 学習で step を重ねると落ちる現象とは別系統。

今回の `Peak 76.5GB / 137.0GB` の主因は、MLX core の allocator cache ではなく、`mlx_lm.models.qwen3_5.GatedDeltaNet` の学習時 recurrent path である。

`GatedDeltaNet` は推論時には高速な custom Metal kernel (`gated_delta_kernel`) を使うが、学習時は gradient を取るために `gated_delta_ops()` へ落ちる。この関数は系列長 `T` ぶん recurrent step を Python/MLX graph として展開し、backward 用に各 token の recurrent `state` を保持する。Qwen3.5-9B の linear attention state は 1 layer あたり約 2.1MB で、linear attention layer が 24 層あるため、長いサンプルでは state 履歴だけで数十 GB になる。そこに backward intermediate と LoRA/optimizer graph が重なり、step 後には解放されるが step 中の peak が 76GB/137GB まで跳ねていた。

したがって、これは「step 後に active memory が単調増加する leak」ではなく、「1 step 内の autograd live-set が大きすぎる」問題である。実測でも step 後の `Active` は約 5.3GB に戻っていた。

関連する MLX resource 問題として、MLX maintainer は [mlx#3464](https://github.com/ml-explore/mlx/pull/3464) で「too many buffers are created」と説明し、短期回避として `MLX_MAX_OPS_PER_BUFFER` / `MLX_BFS_MAX_WIDTH`、長期解として「big MTLBuffer を作ってその中から suballocate する allocator 改善」を挙げている。

このリポジトリでは upstream MLX allocator 全体を書き換えるのではなく、現ユースケースで peak を作っている Qwen3.5 GatedDelta training graph を直接小さくするため、以下を修正した。

- `scripts/train_mlx_lora_fixed.py`
  - MLX import 前に `MLX_MAX_OPS_PER_BUFFER=4`, `MLX_MAX_MB_PER_BUFFER=32`, `MLX_BFS_MAX_WIDTH=4` を設定。
  - `losses += lvalue`, `n_tokens += toks` のように MLX array を step 間で積み上げず、`mx.eval()` 直後に Python scalar へ変換。
  - validation / test も upstream `evaluate()` ではなく、batch ごとに scalar 化する `evaluate_fixed()` を使う。
  - `mx.synchronize()` 後に `batch`, `lvalue`, `toks` 参照を破棄し、Python GC を明示。
  - `mx.set_cache_limit(0)` を強制しない。cache を完全無効化すると MTLBuffer 再利用が効かず descriptor churn が増えるため、上限付き cache と閾値超過時 evict にする。
  - `src.utils.mlx_gated_delta_patch.install()` を読み込み、Qwen3.5 の `gated_delta_update` に local patch を当てる。
- `src/utils/mlx_gated_delta_patch.py`
  - 学習時の `gated_delta_update(..., use_kernel=False)` を chunked custom VJP に置き換える。
  - forward は既存の高速 `gated_delta_kernel` を使う。
  - backward は chunk ごとに `gated_delta_ops()` を再計算する。
  - token 全体の recurrent state 履歴を保持せず、chunk 境界だけを保持する。
  - default chunk は `MLX_GATED_DELTA_CHUNK=512`。
- `Makefile`
  - `train-mlx` / `train-mlx-smoke` を上記 runner に向ける。
  - `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`, `MLX_BFS_MAX_WIDTH`, `MLX_GATED_DELTA_CHUNK` を make 変数として上書き可能にした。

## 分かっていること / まだ分かっていないこと

分かっていること:

- tg-lora の PyTorch baseline / TG-LoRA loop は `forward_backward()` 後に optimizer step を行う通常のPyTorch経路で、今回のMLX Metal OOMの直接対象ではない。
- MLX runner の旧実装には、training loss/token を MLX array のまま report interval まで保持する経路があった。
- このrunnerには一時的に `mx.set_cache_limit(0)` 強制が入っており、`Makefile` の `--cache-limit-ratio 0.15` と矛盾していた。
- upstream `evaluate()` は `all_losses += losses * toks` / `ntokens += toks` を MLX array として保持するため、validationでも同じ参照保持パターンが残っていた。

MLX core 側でまだ分かっていないこと:

- MLX C++ allocator の `num_resources_` が step ごとに単調増加しているか。
- command buffer completion handler が保持する input buffer 数が step 内でどれだけ増えているか。
- byte OOM が先か、Metal resource descriptor limit が先か。

この3点は Python の `active/cache/peak memory` だけでは確定できない。完全に確定するには MLX C++ 側に `num_resources_`, cache hit/miss, command buffer captured buffer count の instrumentation が必要。ただし、今回観測された 76GB/137GB peak は GatedDelta training graph の chunked VJP で再現よく下がったため、現ユースケースの OOM には allocator descriptor leak より GatedDelta live-set の寄与が支配的だった。

## 原因の切り分け

### 1. #3327 / #3524 は別問題

`mlx#3327` の再現条件は、出力要素数が `2^31` を超える shape で `flatten`, `reshape`, `take`, `conv_general` などを通した場合に、shape が signed int32 として負数化すること。

特徴:

- 境界は `2^31` 要素。
- 失敗 allocation size が `2^64 - x` のような wrap 値になる。
- host 側の shape / size accounting のバグで、Metal allocator の長時間回収問題ではない。
- `mlx#3524` は 2026-05-21 に main へ merge 済み。

今回の長時間学習 OOM / resource limit では、毎 step の shape が `2^31` 要素境界に到達している証拠はない。したがって #3524 は取り込むべき安全修正だが、本件の根本修正ではない。

### 2. cache retention だけでもない

`mlx#3350` は Metal caching allocator が再利用できない buffer pool を大きく保持する問題。`mx.set_cache_limit()` や `mx.clear_cache()` はここに効く。

ただし、この学習では以下の対策だけでは解決しなかった。

- `mx.clear_cache()` を毎 step 実行
- `mx.synchronize()` 後に `mx.clear_cache()`
- `mx.set_cache_limit(0)`
- `mx.set_memory_limit()`
- `mx.compile` 無効化

`cache_limit=0` は一見強いが、実際には freed buffer を即 release して次 step で再度 `newBuffer` するため、descriptor 数と driver 側 allocation churn を増やす。長時間学習では「cache を消せば直る」ではなく、「作る MTLBuffer 数を減らし、step 間で lazy graph 参照を残さない」必要がある。

### 3. MLX source で見た該当箇所

`mlx/backend/metal/allocator.cpp`:

- `MetalAllocator::malloc()` は cache hit しなければ `heap_->newBuffer()` または `device_->newBuffer()` で新しい `MTLBuffer` を作る。
- 新規 buffer ごとに `num_resources_++` され、`resource_limit_` を超えると `Resource limit (...) exceeded`。
- `free()` は cache 上限未満なら `MTLBuffer` を release せず cache に戻す。
- `clear_cache()` は cached buffer を release するが、command buffer が保持している resource はそこで消えない。

`mlx/backend/metal/device.cpp`:

- `CommandEncoder` は `buffer_ops_` と `buffer_sizes_` を持つ。
- `needs_commit()` は `MLX_MAX_OPS_PER_BUFFER` と `MLX_MAX_MB_PER_BUFFER` で commit 境界を決める。
- 1 command buffer に多く詰めるほど、完了まで生存する input/output/temporary resource が増える。

`mlx/transforms.cpp`:

- lazy graph は BFS で tape 化される。
- `MLX_BFS_MAX_WIDTH` が大きいほど一度に広く graph を展開し、同時 live temporary が増える。

この3点が `mlx#3464` の maintainer comment と一致する。

### 4. 76.5GB / 137.0GB peak の直接原因

`mlx_lm/models/qwen3_5.py`:

```python
out, state = gated_delta_update(
    q,
    k,
    v,
    a,
    b,
    self.A_log,
    self.dt_bias,
    state,
    mask,
    use_kernel=not self.training,
)
```

学習時は `self.training == True` なので `use_kernel=False` になり、`mlx_lm/models/gated_delta.py` の `gated_delta_ops()` が使われる。

```python
ys = []
for t in range(T):
    y, state = _gated_delta_step_ops(...)
    ys.append(y)
y = mx.stack(ys, axis=1)
```

この loop は differentiable だが、MLX autograd は backward のために token ごとの recurrent `state` を保持する。

Qwen3.5-9B の実 config:

- `linear_num_value_heads = 32`
- `linear_value_head_dim = 128`
- `linear_key_head_dim = 128`
- state shape: `(B, 32, 128, 128)`
- dtype: `float32`
- 1 state: `32 * 128 * 128 * 4 = 2,097,152 bytes`、約 2.1MB
- linear attention layer: 24 層

したがって、state 履歴だけで概算:

- `T=512`: `2.1MB * 512 * 24 = 約 25.8GB`
- `T=984`: `2.1MB * 984 * 24 = 約 49.5GB`

実際には `q/k/v/g/beta`, `conv_out`, backward intermediate, loss/optimizer graph も重なるため、step window peak は 76.5GB / 137.0GB まで増える。step 後に `Active` が約 5.3GB に戻るのは、この peak が永続 leak ではなく、1 step 内の live-set explosion であることと整合する。

## 適用したローカル修正の意味

### `MLX_MAX_OPS_PER_BUFFER`

1 command buffer に入る primitive dispatch 数を減らす。これにより、command buffer 完了まで保持される temporary buffer 群を小さくする。

### `MLX_MAX_MB_PER_BUFFER`

1 command buffer に入る buffer footprint を抑える。large temporary が多い step で command buffer を早めに切る。

### `MLX_BFS_MAX_WIDTH`

lazy graph traversal の幅を抑える。wide graph を一度に展開して大量の temporary を同時に live にする挙動を避ける。

### MLX array accumulator の廃止

upstream training loop は `losses += lvalue`, `n_tokens += toks` のように MLX array を Python 変数として保持する。これは report interval 内で小さい計算 graph を保持し続けるため、長時間学習の resource lifetime を悪化させる。

修正後は:

```python
mx.eval(lvalue, toks, grad_accum)
loss_f = float(lvalue)
toks_n = int(toks)
del lvalue, toks

losses += loss_f
n_tokens += toks_n
```

集計値は Python scalar として保持し、MLX graph を step 境界で切る。

### cache は「無効化」ではなく「上限付き再利用」

毎 step `mx.clear_cache()` や `mx.set_cache_limit(0)` は短期的に active bytes を下げるように見えるが、次 step で同じサイズの buffer を再作成するため、Metal descriptor / resource churn を増やす。

今回の方針は、cache を bounded reuse に使い、閾値を超えた場合だけ evict すること。

### GatedDelta は chunked custom VJP にする

修正後の `gated_delta_update` は、学習時に系列を `MLX_GATED_DELTA_CHUNK` token ごとに分割する。各 chunk の forward は既存の高速 Metal kernel を使い、VJP だけ ops 実装で再計算する。

```python
for start in range(0, seq_len, chunk):
    y_chunk, state = chunk_with_mask(
        q[:, start:end],
        k[:, start:end],
        v[:, start:end],
        g[:, start:end],
        beta[:, start:end],
        state,
        mask[:, start:end],
    )
    ys.append(y_chunk)
```

これにより backward に必要な full-token state 履歴を保持せず、chunk 境界 state だけを保持する。`chunk=512` は今回のデータでは速度と peak のバランスがよかった。`chunk=64` / `256` は peak はさらに低いが、custom VJP 境界と recurrent recompute のコストが大きく実用速度ではなかった。

## 現在のコマンド

```bash
make train-mlx
```

主な上書き:

```bash
make train-mlx \
  MLX_MAX_OPS_PER_BUFFER=2 \
  MLX_MAX_MB_PER_BUFFER=16 \
  MLX_BFS_MAX_WIDTH=2 \
  MLX_GATED_DELTA_CHUNK=512
```

より安定寄りにするなら数値を下げる。速さ寄りにするなら上げる。ただし、上げるほど 1 command buffer 内の live resource が増える。

## upstream での本当の長期修正

今回の 76GB/137GB peak を upstream で直すなら、第一候補は MLX core allocator ではなく `mlx-lm` の Qwen3.5 `GatedDeltaNet` training implementation である。理想形は、forward の custom Metal kernel に対応する memory-efficient custom VJP / backward kernel を実装し、training でも token ごとの recurrent state 履歴を autograd graph に全部保持しないこと。

別問題として、`Resource limit (499000) exceeded` のような descriptor-count 問題を MLX core で直すなら、`MetalAllocator` を「小さい配列ごとに個別 `MTLBuffer` を作る」設計から、「大きな `MTLBuffer` / heap を確保し、その内部 offset を suballocate する」設計へ変える必要がある。ただしこれは小さい patch ではない。

- `allocator::Buffer` が現在は実質 `MTL::Buffer*` だけを持つ。
- backend 側は `a.buffer().ptr()` を `MTL::Buffer*` として扱い、`a.offset()` を Metal kernel に渡す。
- suballocator 化するには、allocation ごとに parent buffer + byte offset + size を厳密に管理し、既存の view offset と合成する必要がある。
- hazard tracking / fence / residency set / cache の単位も見直しが必要。

したがって、このリポジトリで今すぐ高速学習を動かす目的では、MLX本体の allocator 大改修より、Qwen3.5 GatedDelta の chunked custom VJP と、command buffer / graph width / Python参照保持の抑制を組み合わせる方が実用的。

## 検証状況

### PR #3524 / issue #3327

MLX main (`0.32.0.dev20260528+2165dc0`) を local build し、#3524 の shape-product overflow guard が入っていることを確認した。ただし、この修正は今回の 76GB/137GB peak とは別問題。

### 固定前

`max_seq_length=2048`, `grad_accumulation_steps=8`, `steps_per_report=10`:

| run | step | Peak | Active | Cache | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed runner, GatedDelta未修正 | 40 | 76.5GB | 5.3GB | 7.9GB | step後にActiveは戻る |
| fixed runner, GatedDelta未修正 | 70 | 137.0GB | 5.4GB | 7.9GB | 旧100-step run |
| fixed runner, GatedDelta未修正 | 90 | 117.5GB | 5.4GB | 8.1GB | 旧100-step run |

`max_seq_length=512` でも step 40 peak は 67.7GB だった。入力長だけの問題ではなく、GatedDelta recurrent graph の state 履歴が本体。

### 固定後

`MLX_GATED_DELTA_CHUNK=512`, `max_seq_length=2048`, `grad_accumulation_steps=8`:

| step | Peak | Active | Cache | It/sec |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 9.2GB | 5.4GB | 4.3GB | 0.058 |
| 20 | 9.4GB | 5.4GB | 1.6GB | 0.057 |
| 30 | 10.1GB | 5.4GB | 0.0GB | 0.052 |
| 40 | 17.5GB | 5.3GB | 5.2GB | 0.035 |
| 50 | 8.8GB | 5.4GB | 7.9GB | 0.631 |
| 60 | 10.1GB | 5.4GB | 3.1GB | 0.688 |
| 70 | 26.3GB | 5.4GB | 3.2GB | 0.241 |
| 80 | 8.8GB | 5.3GB | 8.2GB | 0.617 |
| 90 | 23.6GB | 5.4GB | 5.5GB | 0.122 |
| 100 | 13.5GB | 5.4GB | 3.5GB | 0.067 |

100 step run は完走し、`runs/mlx_verify_100step_chunk512_vjp/adapters.safetensors` を保存した。以前 76.5GB だった step 40 window は 17.5GB、以前 137.0GB だった step 70 window は 26.3GB、以前 117.5GB だった step 90 window は 23.6GB まで下がった。

### 本来あるべき上限

このマシンの `max_recommended_working_set_size` は `60,129,542,144 bytes`、約 60.1GB。`--memory-limit-ratio 0.8` を使うなら、実質目標は約 48.1GB。したがって、76.5GB / 137.0GB は設定上限を大きく超えた異常値であり、少なくとも 48GB 未満、悪くても 60GB 未満に収めるべき。
