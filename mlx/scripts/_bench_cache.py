"""実測: mlx/src/activation_cache_mlx.py の cache speedup (controlled, steady-state).

Method: run N batches through `eval_and_cache` (prefix+suffix, caches all).
Then run M calls of `eval_from_cache_with_model` (suffix only, prefix reused).
Speedup = (mean time per reuse, EXCLUDING the first reuse which is a cold
suffix path / warmup) / (mean time per full eval_and_cache).

Excluding the first reuse removes the warmup / first-call Metal state
allocation, giving the steady-state "suffix runs over cached hidden" time.
N=20 batches, M=5 reuses.
"""
import sys, os, time, statistics
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "4")
os.environ.setdefault("MLX_GATED_DELTA_CHUNK", "512")
sys.path.insert(0, ".")

from mlx.src.utils.gated_delta_patch import install as igp
from mlx.src.utils.shape_guard import install as isg
isg(); igp()

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
import types

from mlx.src.activation_cache_mlx import ActivationCache

print("Loading Qwen3.5-9B...", flush=True)
model, tokenizer = load(".cache/mlx_models/Qwen--Qwen3.5-9B",
                        tokenizer_config={"trust_remote_code": True})

args = types.SimpleNamespace(data="data_mlx_jsonex", train=True, test=False,
                             valid=True, mask_prompt=False,
                             prompt_feature="", completion_feature="")
train, valid, _ = load_dataset(args, tokenizer)
ds = CacheDataset(valid)
it = iterate_batches(dataset=ds, batch_size=1, max_seq_length=256,
                      loop=False, comm_group=mx.distributed.init())
N = 20
batches = [next(it) for _ in range(N)]
batch_dicts = [
    {"input_ids": b[0].astype(mx.int32), "labels": b[1].astype(mx.int32),
     "attention_mask": None}
    for b in batches
]
SPLIT = 24
M = 5  # reuses

# === Warmup: one full forward to settle Metal state ===
print("\nWarmup (1 full forward)...", flush=True)
_ = ActivationCache().eval_and_cache(model, batch_dicts[:1], SPLIT)

# === Time eval_and_cache across N batches ===
print(f"\n=== A) eval_and_cache ({N} batches, prefix+suffix each, caches all) ===", flush=True)
cache = ActivationCache()
# Warmup the cache itself once (first call = Metal state init for the cache path)
_ = cache.eval_and_cache(model, batch_dicts[:1], SPLIT); cache.clear()
mx.eval(cache._batches[0].hidden_states) if cache._batches else None

t0 = time.perf_counter()
mean_pilot, n_cached = cache.eval_and_cache(model, batch_dicts, SPLIT)
mx.eval(mean_pilot)
t_full = time.perf_counter() - t0
print(f"  num_batches cached: {n_cached}, mean pilot loss: {mean_pilot:.4f}")
print(f"  total: {t_full*1000:.0f} ms  -> per batch (prefix+suffix): {t_full/N*1000:.0f} ms/batch", flush=True)

# === Time eval_from_cache: M reuses (suffix only) ===
print(f"\n=== B) eval_from_cache_with_model x{M} (suffix only, prefix reused) ===", flush=True)
reuse_times_ms = []
for i in range(M):
    mx.eval(cache._batches[0].hidden_states)  # keep cache hot
    t0 = time.perf_counter()
    lf = cache.eval_from_cache_with_model(model)
    mx.eval(lf)
    dt = (time.perf_counter() - t0) * 1000
    reuse_times_ms.append(dt)
    print(f"  reuse {i+1}/{M}: {dt:.0f} ms  (loss={lf:.4f})", flush=True)

# === Speedup: reuse[1:] (steady state) vs full per-batch ===
print(f"\n=== Speedup (steady state = reuses 2..{M}) ===", flush=True)
steady = reuse_times_ms[1:]  # skip first (warmup)
print(f"  reuse times (ms): {[f'{x:.0f}' for x in reuse_times_ms]}")
print(f"  steady reuses: {[f'{x:.0f}' for x in steady]}")
mean_reuse = statistics.mean(steady)
std_reuse = statistics.pstdev(steady) if len(steady) > 1 else 0.0
full_per_batch_ms = t_full / N * 1000
print(f"  full per batch (prefix+suffix): {full_per_batch_ms:.0f} ms")
print(f"  reuse per call (suffix only, steady): {mean_reuse:.0f} ± {std_reuse:.0f} ms")
speedup = (1 - mean_reuse / full_per_batch_ms) * 100
print(f"  cache speedup (steady state): {speedup:.1f}% per eval")
print(f"  (= prefix skipped; suffix retains the rest)")

print("\n=== RESULT ===", flush=True)
print(f"  full eval: {full_per_batch_ms:.0f} ms/batch (prefix+suffix)")
print(f"  cached reuse: {mean_reuse:.0f} ± {std_reuse:.0f} ms/eval (suffix only)")
print(f"  speedup: {speedup:.1f}%  (architectural cache benefit, real measurement)")
