# Post-§4 Research Axis — PSA (§3.2) Reactivation Question

> **位置づけ**: §4 arc（Progressive Freezing のコスト削減研究問い）は
> [§4 terminal verdict](section4_terminal_verdict.md) で **terminal（TIES / cost=0.0 / SHIP-as-quality-preservation）**
> として閉じた。本ファイルはその §4 verdict §4 が命名した前方路 **(C)「§4 arc を離れる新規 Cat-A デリバラブル」**
> のうち、**PSA（Prior-based Subspace Amplification）副路線（GOAL §3.2）の再活性化** を、
> §7 の科学誠実性規律の下で定義する。GOAL.md と矛盾すれば GOAL.md を正とする。

---

## 1. なぜ PSA か（§4 からの pivot）

PSA は Progressive Freezing と**直交**する（GOAL §3.2）。PSA は backward graph を変えず、
各ステップの勾配を**安定的な優位方向 `v_PSA`（per-tensor PC1）に沿って増幅**して品質を上げる路線で、
コスト削減（§4 の問い）ではなく**品質向上**を狙う。よって §4 の null（TIES / cost=0.0）は PSA の前提を侵さず、
真に独立な次の研究 axis である。実装は既存（`src/tg_lora/psa.py`、`enable_psa` route で dormant）。

## 2. 研究問い（§7 の下で定式化）

> **PSA の per-tensor PC1 prior `v_PSA` は、勾配を「ランダム方向に沿って増幅する surrogate」を有意に超えて
> 品質（valid_loss）を改善するか?**

GOAL §4 統計の歯止め「ランダム順サロゲートを超えた削減・性能だけを有効と認定」の PSA 版。
 surrogate（ランダム単位ベクトル沿いの増幅）を超えなければ、PSA の prior 抽出は null
（勾配ノイズを注入するだけ）と診断し、§4 と同様に正直に閉じる。

## 3. §7 受入基準（GPU-free 前提 gate — 本 iter で閉じた）

PSA を 9B target-scale で走らせる（Cat-C / GPU run）**前**に、prior が信号か Marchenko-Pastur ノイズかを
GPU-free で閉じる。これが無ければ 9B run は「ノイズを増幅しているだけ」の可能性を排除できない。

`src/tg_lora/psa_null_baseline.py::prior_vs_surrogate_alignment` が測る metric:

```
prior_alignment     = mean_g [ <g, v_PSA>² / ||g||² ]        # prior が捉える勾配エネルギー割合
surrogate_alignment = mean_{g, v_rand} [ <g, v_rand>² / ||g||² ]
alignment_ratio     = prior_alignment / surrogate_alignment
```

- **NULL（iid ノイズ）**: `alignment_ratio ≈ 1.0`（実測 0.90–0.98 across regimes）。prior はランダム方向と同値 → PSA は null。
- **SIGNAL（planted spike）**: `alignment_ratio >> 1.0`（実測 ≈274、`cos(v_PSA, u)=0.9998`）。prior は実方向を捉える。

`layer_delta_analysis.rank1_z` は rank-1 **固有値**優位性の MP null を持つが、PSA が増幅する**方向ベクトル**の
surrogate は別問題（固有値 spike があっても eigenvector が増幅対象勾配と整合するとは限らない）。本 leaf がその区別を閉じる。
両軸の null/signal assertion は実測値で校正され、実装 mutation（`**2` 削除 / prior 無視）で RED 化を確認済み
（`tests/test_psa_null_baseline.py`、10 tests / GPU-free）。

## 4. 何が残るか（Cat-C hand-off）

`alignment_ratio` の実値を**本番 9B run の実 ΔW history + 実勾配**で測ることのみが target-scale で未達成。
これは §4 verdict と同じ Cat-C（9B GPU + src.data-free verdict worker で閉じうる境界）。
 surrogate を下回る、または ≈1.0 なら PSA は §4 と同じく null として記録して閉じる。
 9B run は本 public mirror の `--data-file` offline rail（`079e8f1`/`e135736`）で自己完結再現可能。

## 5. Provenance

- metric + surrogate: `src/tg_lora/psa_null_baseline.py`（`_random_like_with_norm` パターンの単位ベクトル版）。
- behavior lock: `tests/test_psa_null_baseline.py`（null no-false-positive + signal detected + random-prior discrimination、mutation-proven）。
- 関連 null: `src/tg_lora/layer_delta_analysis.py`（rank-1 固有値 z-score）、`tests/test_rank1_null_calibration.py`。
- §4 terminal context: [section4_terminal_verdict.md](section4_terminal_verdict.md) §4 (C)。
