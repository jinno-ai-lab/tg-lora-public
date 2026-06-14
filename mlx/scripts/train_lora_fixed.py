#!/usr/bin/env python3
"""MLX LoRA training with bounded Metal resource lifetime.

The long-training failure is not fixed by sleeping or restarting the process.
It comes from two resource-lifetime issues:

1. MLX can encode a wide lazy graph into large Metal command buffers, keeping
   many temporary MTLBuffers alive until the command buffer completes.
2. The upstream training loop accumulates loss/token MLX arrays across steps,
   which can retain lazy graph references longer than needed.

The environment knobs below must be set before importing MLX because MLX reads
them once during startup.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

# Defaults are intentionally below MLX's built-in Mac defaults because Qwen3.5
# LoRA creates wide lazy graphs and many short-lived Metal buffers. Callers can
# override them in the environment after validating their machine.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "4")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "32")
os.environ.setdefault("MLX_BFS_MAX_WIDTH", "4")
os.environ.setdefault("MLX_GATED_DELTA_CHUNK", "512")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim_base  # noqa: E402
import numpy as np  # noqa: E402
from mlx.nn.utils import average_gradients  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402

# Install the shape-overflow guard from PR #3524 before importing mlx_lm.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mlx.src.utils.gated_delta_patch import install as install_gated_delta_patch  # noqa: E402
from mlx.src.utils.shape_guard import install as install_shape_guard  # noqa: E402

install_shape_guard()
install_gated_delta_patch()

from mlx_lm.tuner.datasets import CacheDataset  # noqa: E402
from mlx_lm.tuner.trainer import (  # noqa: E402
    TrainingArgs,
    default_loss,
    grad_checkpoint,
    iterate_batches,
)
from mlx_lm.tuner.utils import (  # noqa: E402
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from mlx_lm.utils import load, save_config  # noqa: E402
from mlx.src.dynfreeze_mlx import DynamicFreezeController  # noqa: E402
from mlx.src.gold_eval_mlx import evaluate_json_extraction_run_mlx  # noqa: E402
from mlx.src.activation_cache_mlx import ActivationCache  # noqa: E402


def _save_checkpoint(
    model,
    optimizer,
    args,
    step: int,
    best_val_loss: float,
    trained_tokens: int,
    total_elapsed: float,
) -> None:
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    ckpt_dir = Path(args.adapter_file).parent
    step_file = ckpt_dir / f"{step:07d}_adapters.safetensors"
    mx.save_safetensors(str(step_file), adapter_weights)
    mx.save_safetensors(str(args.adapter_file), adapter_weights)

    opt_flat = tree_flatten(optimizer.state)
    opt_path = ckpt_dir / f"{step:07d}_optimizer.safetensors"
    opt_dict = {k: v for k, v in opt_flat if hasattr(v, "shape")}
    if opt_dict:
        mx.save_safetensors(str(opt_path), opt_dict)

    meta = {
        "step": step,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "trained_tokens": trained_tokens,
        "elapsed_seconds": round(total_elapsed, 3),
        "adapter_file": str(step_file),
    }
    meta_path = ckpt_dir / f"{step:07d}_training_state.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Step {step}: checkpoint saved to {ckpt_dir}", flush=True)


def _find_latest_checkpoint(adapter_dir: Path) -> dict | None:
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        return None
    states = sorted(adapter_dir.glob("*_training_state.json"))
    if not states:
        return None
    latest = states[-1]
    with open(latest) as f:
        meta = json.load(f)
    step = meta.get("step", 0)
    opt_path = adapter_dir / f"{step:07d}_optimizer.safetensors"
    adapter_path = adapter_dir / f"{step:07d}_adapters.safetensors"
    if not adapter_path.exists():
        return None
    meta["_opt_path"] = str(opt_path) if opt_path.exists() else None
    meta["_adapter_path"] = str(adapter_path)
    return meta


def _arg(args, name: str, default=None):
    return getattr(args, name, default)


def _parser_has_option(parser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def _add_parser_arg_if_missing(parser, option: str, **kwargs) -> None:
    if not _parser_has_option(parser, option):
        parser.add_argument(option, **kwargs)


def _clear_cache_if_needed(threshold: int | None) -> None:
    if threshold and mx.get_cache_memory() > threshold:
        mx.clear_cache()


def _metal_memory_gb() -> tuple[float, float, float]:
    return (
        mx.get_peak_memory() / 1e9,
        mx.get_active_memory() / 1e9,
        mx.get_cache_memory() / 1e9,
    )


def evaluate_fixed(
    model,
    dataset,
    batch_size: int,
    num_batches: int,
    max_seq_length: int = 2048,
    loss=default_loss,
    iterate_batches=iterate_batches,
    clear_cache_threshold: int | None = None,
) -> float:
    """Evaluate without carrying MLX scalar graphs across validation batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            comm_group=mx.distributed.init(),
        ),
    ):
        losses, toks = loss(model, *batch)
        mx.eval(losses, toks)
        total_loss += float(losses) * int(toks)
        total_tokens += int(toks)
        del losses, toks, batch
        mx.synchronize()
        gc.collect()
        _clear_cache_if_needed(clear_cache_threshold)

    if total_tokens == 0:
        raise ValueError("Validation produced zero tokens.")
    return total_loss / total_tokens


def train_fixed(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs | None = None,
    loss=None,
    iterate_batches=iterate_batches,
    max_grad_norm: float = 1.0,
    start_step: int = 0,
    init_best_val_loss: float = float("inf"),
    tokenizer=None,
    dynfreeze=None,
    gold_records=None,
    cycle_steps: int = 8,
    gold_eval_every: int = 5,
    gold_eval_max_examples: int = 50,
    gold_eval_max_tokens: int = 128,
    split_layer: int = 0,
):
    """Train without keeping evaluated MLX graphs alive across step boundaries."""
    if args is None:
        args = TrainingArgs()
    if loss is None:
        loss = default_loss

    print(f"Starting training..., iters: {args.iters}")
    total_start_time = time.perf_counter()
    if start_step > 0:
        print(f"Resuming from step {start_step}", flush=True)
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    # ── Metal memory setup ──────────────────────────────────────
    # Keep a bounded cache so repeated temporary shapes can reuse MTLBuffers.
    # Forcing cache_limit=0 increases newBuffer churn and makes descriptor
    # growth harder to distinguish from real active-memory pressure.
    if mx.metal.is_available():
        max_rec = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(int(max_rec * _arg(args, "wired_limit_ratio", 0.8)))

        cache_limit_ratio = _arg(args, "cache_limit_ratio", None)
        memory_limit_ratio = _arg(args, "memory_limit_ratio", None)
        if cache_limit_ratio is not None:
            mx.set_cache_limit(int(max_rec * cache_limit_ratio))
        if memory_limit_ratio is not None:
            mx.set_memory_limit(int(max_rec * memory_limit_ratio))

        if rank == 0:
            cache_gb = mx.get_cache_memory() / 1e9
            active_gb = mx.get_active_memory() / 1e9
            wired_gb = mx.device_info().get("max_recommended_working_set_size", 0) / 1e9
            print(
                f"Metal: wired={wired_gb:.1f}GB, "
                f"cache={cache_gb:.2f}GB, active={active_gb:.2f}GB",
                flush=True,
            )

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, loss)

    grad_accum_steps = args.grad_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")

    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        mx.eval(lvalue)
        if not math.isfinite(float(lvalue)):
            return lvalue, toks, None
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            if max_grad_norm and max_grad_norm > 0:
                grad, grad_norm = optim_base.clip_grad_norm(grad, max_grad_norm)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    model.train()
    losses = 0.0  # Python float, not MLX array
    n_tokens = 0  # Python int
    steps = 0
    trained_tokens = 0
    train_time = 0.0
    grad_accum = None
    best_val_loss = init_best_val_loss
    last_val_loss = init_best_val_loss if math.isfinite(init_best_val_loss) else None
    # Main training loop
    for it, batch in zip(
        range(start_step + 1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
            comm_group=world,
        ),
    ):
        tic = time.perf_counter()

        # Validation
        if val_dataset and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            val_loss = evaluate_fixed(
                model=model,
                dataset=val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
                clear_cache_threshold=args.clear_cache_threshold,
            )
            best_val_loss = min(best_val_loss, val_loss)
            last_val_loss = val_loss
            model.train()
            mx.synchronize()
            gc.collect()
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: Val loss {val_loss:.3f}, Val took {val_time:.3f}s",
                    flush=True,
                )
            tic = time.perf_counter()

        # ── Forward + backward ──────────────────────────────────
        lvalue, toks, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )

        # ── Per-step graph release ──────────────────────────────
        # 1. Materialize the computation graph outputs.
        mx.eval(lvalue, toks, grad_accum)

        # 2. Extract to Python scalars immediately so the MLX arrays
        #    and their computation graphs can be released.
        loss_f = float(lvalue)
        toks_n = int(toks)
        del lvalue, toks

        if not math.isfinite(loss_f):
            print(
                f"WARNING: Non-finite loss at iter {it}, discarding batch", flush=True
            )
            grad_accum = None
            del batch
            gc.collect()
            _clear_cache_if_needed(args.clear_cache_threshold)
            continue

        losses += loss_f
        n_tokens += toks_n
        steps += 1

        # 3. Evaluate the model / optimizer / RNG state so the latest
        #    parameter updates are concrete (not lazy).
        mx.eval(model.state, optimizer.state, mx.random.state)

        # 4. Wait for all Metal command buffers to complete.
        mx.synchronize()

        # 5. Drop input batch references before Python GC. The evaluated output
        #    arrays no longer need the input graph after synchronize().
        del batch

        # 6. Force Python GC so dangling MLX array wrappers that hold Metal
        #    buffer refs are collected before optional cache eviction.
        gc.collect()

        # 7. Evict cache only when explicitly requested. Clearing every step
        #    prevents buffer reuse and recreates many MTLBuffer descriptors.
        _clear_cache_if_needed(args.clear_cache_threshold)

        train_time += time.perf_counter() - tic

        # ── Guard cycle boundary (dynfreeze r_A + freeze/unfreeze + gold) ──
        if it % cycle_steps == 0 and (dynfreeze is not None or gold_records):
            cycle = it // cycle_steps
            guard_fields: dict = {}
            if dynfreeze is not None:
                dynfreeze.compute_r_A(model, cycle)
                _release = dynfreeze.decide_unfreeze(cycle)
                dynfreeze.apply_unfreeze(model, _release)
                _freeze = dynfreeze.decide_freeze(cycle)
                dynfreeze.apply_freeze(model, _freeze, cycle)
                guard_fields["guard_block_size"] = dynfreeze.block_size
                guard_fields["guard_block_layers"] = ",".join(
                    str(_l) for _l in dynfreeze.frozen_block
                )
                for _li, _hist in dynfreeze.r_A_history.items():
                    if _hist:
                        guard_fields[f"guard_r_A_L{_li}"] = _hist[-1]

            # Real validation loss at the cycle boundary (§5.2 trigger needs it).
            # PREFIX/SUFFIX CACHE: the prefix (layers[:split_layer]) is computed
            # ONCE per cycle via eval_and_cache; the post eval reuses the cached
            # hidden state (suffix-only) — the TG-LoRA architecture that makes
            # the freeze's speed benefit testable. See activation_cache_mlx.py.
            if val_dataset is not None and split_layer > 0 and tokenizer is not None:
                # Build a small list of cache-compatible dict batches from val.
                _val_cache_batches: list[dict] = []
                _batches_taken = 0
                for _vb in iterate_batches(
                    dataset=val_dataset,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length,
                    loop=False,
                    comm_group=mx.distributed.init(),
                ):
                    if _batches_taken >= args.val_batches:
                        break
                    _inp, _tgt = _vb
                    _val_cache_batches.append(
                        {
                            "input_ids": _inp.astype(mx.int32),
                            "labels": _tgt.astype(mx.int32),
                            "attention_mask": None,
                        }
                    )
                    _batches_taken += 1
                if _val_cache_batches:
                    _cycle_cache = ActivationCache()
                    _pilot_loss, _n = _cycle_cache.eval_and_cache(
                        model, _val_cache_batches, split_layer,
                    )
                    model.train()
                    # 2nd eval reusing the cached prefix (suffix only) — this
                    # is the architectural speed benefit (prefix amortized).
                    _post_loss = _cycle_cache.eval_from_cache_with_model(model)
                    model.train()
                    _vl = _pilot_loss  # both equal by construction; use pilot
                    _post_reuse_loss = _post_loss
                    last_val_loss = _vl
                    best_val_loss = min(best_val_loss, _vl)
                else:
                    _post_reuse_loss = None
                    _vl = None
            else:
                # Fallback: full eval (no cache — e.g. split_layer=0 or no tokenizer).
                if val_dataset is not None:
                    _vl = evaluate_fixed(
                        model=model, dataset=val_dataset, loss=loss,
                        batch_size=args.batch_size, num_batches=args.val_batches,
                        max_seq_length=args.max_seq_length,
                        iterate_batches=iterate_batches,
                        clear_cache_threshold=args.clear_cache_threshold,
                    )
                    model.train()
                    last_val_loss = _vl
                    best_val_loss = min(best_val_loss, _vl)
                _post_reuse_loss = None

            gold_fields: dict = {}
            if gold_records and tokenizer is not None and (
                cycle % gold_eval_every == 0 or it == args.iters
            ):
                _gs = evaluate_json_extraction_run_mlx(
                    model,
                    tokenizer,
                    gold_records,
                    max_examples=gold_eval_max_examples,
                    max_tokens=gold_eval_max_tokens,
                )
                gold_fields = {f"gold_{k}": v for k, v in _gs.items()}
                if rank == 0:
                    print(
                        f"Guard cycle {cycle} (iter {it}): "
                        f"block={guard_fields.get('guard_block_size', 0)} "
                        f"loss_valid={last_val_loss} "
                        f"gold_combined={_gs.get('combined')}",
                        flush=True,
                    )
            elif rank == 0 and guard_fields:
                print(
                    f"Guard cycle {cycle} (iter {it}): "
                    f"block={guard_fields.get('guard_block_size')} "
                    f"loss_valid={last_val_loss}",
                    flush=True,
                )

            if rank == 0:
                _metrics_path = Path(args.adapter_file).parent / "run_metrics.jsonl"
                _elapsed = time.perf_counter() - total_start_time
                _record = {
                    "type": "step",
                    "mode": "baseline",
                    "step": it,
                    "cycle": cycle,
                    "loss_train": (losses / steps) if steps else None,
                    "loss_valid": last_val_loss,
                    "elapsed_seconds": round(_elapsed, 3),
                    **guard_fields,
                    **gold_fields,
                }
                with open(_metrics_path, "a") as _mf:
                    _mf.write(json.dumps(_record) + "\n")

        # Report training loss
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = losses / steps
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = n_tokens / train_time
            trained_tokens += n_tokens
            peak_mem, active_mem, cache_mem = _metal_memory_gb()
            mx.reset_peak_memory()

            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"LR {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tok/sec {tokens_sec:.3f}, "
                    f"Peak {peak_mem:.1f}GB, "
                    f"Active {active_mem:.1f}GB, "
                    f"Cache {cache_mem:.1f}GB",
                    flush=True,
                )

                metrics_path = Path(args.adapter_file).parent / "run_metrics.jsonl"
                total_elapsed = time.perf_counter() - total_start_time
                with open(metrics_path, "a") as f_metrics:
                    f_metrics.write(
                        json.dumps(
                            {
                                "type": "step",
                                "mode": "baseline",
                                "step": it,
                                "cycle": it // cycle_steps,
                                "loss_train": train_loss,
                                "loss_valid": last_val_loss,
                                "lr": learning_rate,
                                "tokens_per_sec": tokens_sec,
                                "elapsed_seconds": round(total_elapsed, 3),
                                "gpu_peak_mb": peak_mem * 1024.0,
                                "tg_lora_accepted": None,
                                "tg_lora_cosine_sim": None,
                                "tg_lora_raw_delta_cosine_sim": None,
                                "tg_lora_predicted_consistency": None,
                                "tg_lora_short_long_norm_ratio": None,
                                "tg_lora_reduction_rate": None,
                                "tg_lora_K": None,
                                "tg_lora_N": None,
                                "tg_lora_proposed_N": None,
                                "tg_lora_alpha": None,
                                "tg_lora_beta": None,
                                "tg_lora_lr": None,
                                "tg_lora_cache_built": None,
                                "tg_lora_cache_eligible": None,
                                "tg_lora_cache_hit": None,
                                "tg_lora_validation_forwards": None,
                                "tg_lora_post_extrapolation_eval": None,
                                "tg_lora_rollback_triggered": None,
                                "psa_regime": None,
                                "psa_regime_transitions": None,
                                "act_regime": None,
                                "act_stable_fraction": None,
                                "act_cosine_latest": None,
                                "act_cosine_mean": None,
                            }
                        )
                        + "\n"
                    )

            losses = 0.0
            n_tokens = 0
            steps = 0
            train_time = 0.0

        # Save checkpoint
        if it % args.steps_per_save == 0 and rank == 0:
            _save_checkpoint(
                model, optimizer, args,
                step=it,
                best_val_loss=best_val_loss,
                trained_tokens=trained_tokens,
                total_elapsed=time.perf_counter() - total_start_time,
            )

    # Save final weights
    if rank == 0:
        _save_checkpoint(
            model, optimizer, args,
            step=args.iters,
            best_val_loss=best_val_loss,
            trained_tokens=trained_tokens,
            total_elapsed=time.perf_counter() - total_start_time,
        )

        total_wall_seconds = time.perf_counter() - total_start_time
        summary_path = Path(args.adapter_file).parent / "summary.json"
        with open(summary_path, "w") as f_sum:
            json.dump(
                {
                    "seed": getattr(args, "seed", 42),
                    "wall_seconds": total_wall_seconds,
                    "best_valid_loss": best_val_loss
                    if best_val_loss != float("inf")
                    else None,
                    "tokens_per_sec": trained_tokens / total_wall_seconds
                    if total_wall_seconds > 0
                    else 0.0,
                    "gpu_peak_mb": mx.get_peak_memory() / (1024 * 1024)
                    if mx.metal.is_available()
                    else 0.0,
                    "max_seq_length": args.max_seq_length,
                    "total_steps": args.iters,
                },
                f_sum,
                indent=2,
            )


def main() -> None:
    import types

    import yaml as _yaml

    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    print(
        "MLX scheduler: "
        f"MAX_OPS={os.environ.get('MLX_MAX_OPS_PER_BUFFER', 'default')} "
        f"MAX_MB={os.environ.get('MLX_MAX_MB_PER_BUFFER', 'default')} "
        f"BFS_WIDTH={os.environ.get('MLX_BFS_MAX_WIDTH', 'default')} "
        f"GATED_DELTA_CHUNK={os.environ.get('MLX_GATED_DELTA_CHUNK', 'default')}",
        flush=True,
    )
    parser = build_parser()
    _add_parser_arg_if_missing(
        parser,
        "--wired-limit-ratio",
        type=float,
        default=None,
        help="Ratio of max recommended Metal working set used as wired limit.",
    )
    _add_parser_arg_if_missing(
        parser,
        "--memory-limit-ratio",
        type=float,
        default=None,
        help="Ratio of max recommended Metal working set used as MLX memory limit.",
    )
    _add_parser_arg_if_missing(
        parser,
        "--cache-limit-ratio",
        type=float,
        default=None,
        help="Ratio of max recommended Metal working set used as MLX cache limit.",
    )
    _add_parser_arg_if_missing(
        parser,
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable).",
    )
    ns_args = parser.parse_args()
    args = vars(ns_args)
    config = args.get("config")
    if config:
        with open(config, "r") as f:
            config = _yaml.load(f, _yaml.SafeLoader)
        for k, v in config.items():
            if args.get(k, None) is None:
                args[k] = v
    local_defaults = {
        **CONFIG_DEFAULTS,
        "wired_limit_ratio": 0.8,
        "memory_limit_ratio": 0.8,
        "cache_limit_ratio": 0.15,
    }
    for k, v in local_defaults.items():
        if args.get(k, None) is None:
            args[k] = v

    args = types.SimpleNamespace(**args)
    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    print("Loading datasets")
    from mlx_lm.tuner.datasets import load_dataset

    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    if args.train:
        print("Training")
        mx.random.seed(args.seed)
        model.freeze()

        if args.num_layers > len(model.layers):
            raise ValueError(
                f"Requested {args.num_layers} layers but model has {len(model.layers)}."
            )

        if args.fine_tune_type == "full":
            for layer in model.layers[-max(args.num_layers, 0) :]:
                layer.unfreeze()
            args.lora_parameters = None
        elif args.fine_tune_type in ("lora", "dora"):
            linear_to_lora_layers(
                model,
                args.num_layers,
                args.lora_parameters,
                use_dora=(args.fine_tune_type == "dora"),
            )

        if args.resume_adapter_file is not None:
            print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        print_trainable_parameters(model)

        adapter_path = Path(args.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_path / "adapters.safetensors"
        save_config(vars(args), adapter_path / "adapter_config.json")

        training_args = types.SimpleNamespace(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            clear_cache_threshold=args.clear_cache_threshold,
            grad_accumulation_steps=args.grad_accumulation_steps,
            wired_limit_ratio=args.wired_limit_ratio,
            memory_limit_ratio=args.memory_limit_ratio,
            cache_limit_ratio=args.cache_limit_ratio,
            seed=args.seed,
        )

        lr = (
            build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
        )
        import mlx.optimizers as optim

        opt_name = args.optimizer.lower()
        opt_config = args.optimizer_config.get(opt_name, {})
        opt_classes = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "muon": optim.Muon,
            "sgd": optim.SGD,
            "adafactor": optim.Adafactor,
        }
        opt_class = opt_classes[opt_name]
        optimizer = opt_class(learning_rate=lr, **opt_config)

        # --- Resume from checkpoint if available ---
        resume_step = 0
        resume_best_val = float("inf")
        ckpt = _find_latest_checkpoint(adapter_path)
        if ckpt is not None and ckpt.get("step", 0) > 0:
            resume_step = ckpt["step"]
            resume_best_val = ckpt.get("best_val_loss") or float("inf")
            adapter_weights_path = ckpt.get("_adapter_path")
            if adapter_weights_path:
                print(f"Restoring adapter weights from {adapter_weights_path}")
                model.load_weights(adapter_weights_path, strict=False)
            opt_path = ckpt.get("_opt_path")
            if opt_path:
                print(f"Restoring optimizer state from {opt_path}")
                opt_state_flat = mx.load(opt_path)
                from mlx.utils import tree_unflatten
                optimizer.state = tree_unflatten(list(opt_state_flat.items()))
                mx.eval(optimizer.state)
            print(
                f"Resuming from step {resume_step} "
                f"(best_val_loss={resume_best_val:.4f})",
                flush=True,
            )
            mx.random.seed(args.seed + resume_step)

        # --- Guard experiment (M10) setup ---
        dynfreeze = None
        if getattr(args, "dynfreeze_enabled", False):
            _n_total = len(model.layers)
            _n_lora = getattr(args, "num_layers", 0)
            _lora_idx = list(range(_n_total - _n_lora, _n_total))
            dynfreeze = DynamicFreezeController(
                settled_ratio=getattr(args, "dynfreeze_settled_ratio", 0.15),
                window=getattr(args, "dynfreeze_window", 5),
                stir_interval=getattr(args, "dynfreeze_stir_interval", 10),
                unfreeze_ratio=getattr(args, "dynfreeze_unfreeze_ratio", 0.5),
                min_trainable=getattr(args, "dynfreeze_min_trainable", 2),
                epsilon_ratio=getattr(args, "dynfreeze_epsilon_ratio", 0.01),
                a_mask_ratio=getattr(args, "dynfreeze_a_mask_ratio", 0.1),
                all_layer_indices=_lora_idx,
            )
            print(
                f"Guard enabled: LoRA layers L{_lora_idx[0]}..L{_lora_idx[-1]} "
                f"(settled_ratio={dynfreeze._settled_ratio}, "
                f"window={dynfreeze._window}, stir={dynfreeze._stir_interval}, "
                f"min_trainable={dynfreeze._min_trainable})",
                flush=True,
            )

        gold_records = []
        if getattr(args, "gold_eval_enabled", False) and getattr(
            args, "gold_test_path", ""
        ):
            with open(args.gold_test_path) as _gf:
                gold_records = [json.loads(_l) for _l in _gf if _l.strip()]
            print(
                f"Gold eval: {len(gold_records)} records from {args.gold_test_path}",
                flush=True,
            )

        _cycle_steps = int(getattr(args, "cycle_steps", 8) or 8)

        train_fixed(
            model=model,
            optimizer=optimizer,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set) if valid_set else None,
            args=training_args,
            max_grad_norm=args.max_grad_norm,
            start_step=resume_step,
            init_best_val_loss=resume_best_val,
            tokenizer=tokenizer,
            dynfreeze=dynfreeze,
            gold_records=gold_records,
            cycle_steps=_cycle_steps,
            gold_eval_every=int(getattr(args, "gold_eval_every_cycles", 5) or 5),
            gold_eval_max_examples=int(
                getattr(args, "gold_eval_max_examples", 50) or 50
            ),
            gold_eval_max_tokens=int(
                getattr(args, "gold_eval_max_new_tokens", 128) or 128
            ),
            split_layer=int(getattr(args, "split_layer", 0) or 0),
        )

    if args.test:
        print("Testing")
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)
        test_loss = evaluate_fixed(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            clear_cache_threshold=args.clear_cache_threshold,
        )
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


if __name__ == "__main__":
    main()
