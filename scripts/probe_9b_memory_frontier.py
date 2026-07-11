#!/usr/bin/env python
"""Empirical GPU memory-frontier probe for the suffix-only 9B config.

Answers a single, decisive, *data-independent* question that the real 9B §4
verdict depends on:

    **Does the ``configs/9b_baseline_suffix_only_last25.yaml`` memory stack
    (4-bit nf4 + ``trainable_lora_scope: last_25_percent`` + gradient
    checkpointing) let a real ``Qwen/Qwen3.5-9B`` QLoRA forward+backward+step
    fit on a 12 GB GPU at ``seq_len >= 1024`` — where the *standard* 9B config
    is proven to OOM?**

Why this probe exists
--------------------
The recurring AI-Hub feedback demands a real 9B target-scale A/B verdict bound
to ``configs/9b_baseline_suffix_only_last25.yaml``. The standing block argument
rests on an OOM claim measured only for the *standard* config (``9b_tg_lora`` /
``9b_baseline``): 9B 4-bit QLoRA OOMs at ``seq_len >= 512`` on the 12 GB RTX
3060 (CE-loss ``logits.float()`` + eval batching). The suffix-only config has
*different* memory characteristics (last-25%-scope cuts LoRA grad/optim state
~4x; gradient checkpointing is on) that have **never been measured at target
scale**. The full verdict run additionally needs the private ``src.data``
pipeline (absent on this public mirror) to produce training data, so the loss
frontier (``make paper-memory-frontier-sweep``) cannot run here. This probe
isolates the *physical* memory question from the *data-pipeline* question: it
reuses the trainer's exact public model-loading path
(``load_base_model`` → ``apply_lora`` → ``configure_trainable_lora_scope``) and
bypasses **only** the data-loading step, feeding a synthetic batch. It therefore
measures the same per-step peak the real trainer would hit.

What it does NOT exercise
-------------------------
The config's headline ``prefix_feature_cache_experimental`` +
``prefix_feature_cache_offload_prefix_to_cpu`` levers *additionally* stream
frozen-prefix activations to CPU and are not exercised by a plain forward pass.
So this probe is an **upper bound** on per-step GPU memory: if it FITS, the
verdict is physically producible; if it OOMs, prefix-offload remains the one
untested recovery lever (noted in the report).

Output
------
JSON (``--json`` / ``--output``) with per-``seq_len`` peak allocated/reserved
MiB, fit/OOM, the max fit ``seq_len``, total GPU MiB, and the config levers
attributed. A human summary prints the same, including an explicit verdict line
for ``seq_len == 1024`` (the full §4 threshold).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

logger = logging.getLogger("probe-9b-memory")

# seq_len at/above which a verdict counts as the "full §4" target-scale result
# (GOAL §4 / TASK-0154 REQ-003). Used only for the human verdict line.
FULL_SECTION4_SEQ_LEN = 1024


class PeakMemoryOOM(RuntimeError):
    """Raised by the GPU step-runner when a training step hits CUDA OOM.

    Lifted out of ``torch.cuda.OutOfMemoryError`` so the sweep logic (and its
    unit tests) do not depend on a live CUDA context.
    """


# ---------------------------------------------------------------------------
# Pure / orchestrable core (unit-tested without a GPU)
# ---------------------------------------------------------------------------


@dataclass
class FrontierMeasurement:
    seq_len: int
    peak_alloc_mb: float | None
    peak_reserved_mb: float | None
    fit: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "peak_alloc_mb": self.peak_alloc_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "fit": self.fit,
            "error": self.error,
        }


@dataclass
class FrontierResult:
    gpu_name: str
    gpu_total_mb: float
    seq_lens: list[int]
    measurements: list[FrontierMeasurement]
    max_fit_seq_len: int | None
    full_section4_producible: bool | None
    config_levers: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_total_mb": self.gpu_total_mb,
            "seq_lens": self.seq_lens,
            "measurements": [m.to_dict() for m in self.measurements],
            "max_fit_seq_len": self.max_fit_seq_len,
            "full_section4_producible": self.full_section4_producible,
            "config_levers": self.config_levers,
            "note": self.note,
        }


def sweep_seq_lens(
    seq_lens: list[int],
    batch_builder: Callable[[int], dict[str, torch.Tensor]],
    step_runner: Callable[[dict[str, torch.Tensor]], tuple[float, float]],
    cleanup: Callable[[int], None] | None = None,
) -> list[FrontierMeasurement]:
    """Run ``step_runner`` once per ``seq_len``, capturing OOM as fit=False.

    Parameters
    ----------
    seq_lens:
        Probed in the given order (callers pass ascending so the fit frontier
        is characterized from small to large; once a seq_len OOMs, larger ones
        are still attempted and recorded as OOM rather than silently skipped).
    batch_builder:
        ``seq_len -> batch dict`` (``{input_ids, attention_mask, labels}``).
    step_runner:
        ``batch -> (peak_alloc_mb, peak_reserved_mb)``; must raise
        :class:`PeakMemoryOOM` on CUDA OOM.
    cleanup:
        Invoked between attempts (e.g. ``torch.cuda.empty_cache()``).
    """
    measurements: list[FrontierMeasurement] = []
    for seq_len in seq_lens:
        if cleanup is not None:
            cleanup(seq_len)
        try:
            batch = batch_builder(seq_len)
            peak_alloc, peak_reserved = step_runner(batch)
        except PeakMemoryOOM as exc:
            measurements.append(
                FrontierMeasurement(seq_len, None, None, False, error=f"OOM: {exc}")
            )
            continue
        measurements.append(
            FrontierMeasurement(seq_len, peak_alloc, peak_reserved, True, error=None)
        )
    return measurements


def max_fit_seq_len(measurements: list[FrontierMeasurement]) -> int | None:
    """Largest ``seq_len`` with ``fit=True``, or ``None`` if none fit."""
    fit = [m.seq_len for m in measurements if m.fit]
    return max(fit) if fit else None


def full_section4_producible(
    measurements: list[FrontierMeasurement], threshold: int = FULL_SECTION4_SEQ_LEN
) -> bool | None:
    """Whether any measured ``seq_len >= threshold`` fit.

    ``None`` (not ``False``) when no seq_len at/above the threshold was probed,
    so the verdict is "unmeasured" rather than "failed".
    """
    above = [m for m in measurements if m.seq_len >= threshold]
    if not above:
        return None
    return any(m.fit for m in above)


def config_levers_from_cfg(cfg: Any) -> dict[str, Any]:
    """Pull the memory-relevant config knobs for honest attribution."""
    model = getattr(cfg, "model", {}) or {}
    training = getattr(cfg, "training", {}) or {}
    return {
        "model": model.get("name_or_path"),
        "load_in_4bit": model.get("load_in_4bit"),
        "bnb_4bit_quant_type": model.get("bnb_4bit_quant_type"),
        "dtype": model.get("dtype"),
        "trainable_lora_scope": training.get("trainable_lora_scope", "all"),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "prefix_feature_cache_experimental": bool(
            training.get("prefix_feature_cache_experimental", False)
        ),
        "prefix_feature_cache_offload_prefix_to_cpu": bool(
            training.get("prefix_feature_cache_offload_prefix_to_cpu", False)
        ),
        "prefix_feature_cache_offload_exercised": False,
    }


def format_report(result: FrontierResult) -> str:
    """Human-readable summary with an explicit seq1024 verdict line."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("9B suffix-only memory-frontier probe")
    lines.append("=" * 64)
    lines.append(f"  GPU: {result.gpu_name}  ({result.gpu_total_mb:.0f} MiB total)")
    lines.append(f"  model/levers: {result.config_levers.get('model')} | "
                 f"scope={result.config_levers.get('trainable_lora_scope')} | "
                 f"4bit={result.config_levers.get('load_in_4bit')} | "
                 f"grad_ckpt={result.config_levers.get('gradient_checkpointing')}")
    lines.append("")
    lines.append(f"  {'seq_len':>8}  {'status':>6}  {'peak_alloc_MiB':>15}  "
                 f"{'peak_reserved_MiB':>17}  {'headroom_MiB':>13}")
    for m in result.measurements:
        if m.fit and m.peak_alloc_mb is not None:
            headroom = result.gpu_total_mb - m.peak_alloc_mb
            lines.append(
                f"  {m.seq_len:>8}  {'FIT':>6}  {m.peak_alloc_mb:>15.1f}  "
                f"{(m.peak_reserved_mb or 0.0):>17.1f}  {headroom:>13.1f}"
            )
        else:
            lines.append(f"  {m.seq_len:>8}  {'OOM':>6}  {'-':>15}  {'-':>17}  {'-':>13}")
    lines.append("")
    lines.append(f"  max fit seq_len: {result.max_fit_seq_len}")
    prod = result.full_section4_producible
    if prod is None:
        verdict = f"UNMEASURED (no seq_len >= {FULL_SECTION4_SEQ_LEN} probed)"
    elif prod:
        verdict = (f"YES — a seq_len >= {FULL_SECTION4_SEQ_LEN} training step fit; "
                   f"the full §4 verdict is physically producible on this GPU "
                   f"(the remaining blocker is the data pipeline, not memory)")
    else:
        verdict = (f"NO — every seq_len >= {FULL_SECTION4_SEQ_LEN} OOMs; "
                   f"the full §4 verdict is NOT physically producible on this GPU "
                   f"with this config's base forward/backward. "
                   f"Prefix-cache CPU offload is the one untested recovery lever.")
    lines.append(f"  full §4 (seq>={FULL_SECTION4_SEQ_LEN}) producible: {verdict}")
    if result.note:
        lines.append("")
        lines.append(f"  note: {result.note}")
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU step-runner + batch builder (the real, faithful path)
# ---------------------------------------------------------------------------


def build_synthetic_batch(
    vocab_size: int,
    pad_token_id: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a fully-supervised LM batch (labels == input_ids, no -100 masking).

    Uses random token ids in ``[0, vocab_size)`` with the pad id excluded so the
    batch is non-degenerate. ``labels == input_ids`` supervises every position —
    the *pessimal* (full-loss) case, matching the documented OOM driver
    (CE-loss over the full vocab for every token).
    """
    low = 0
    high = max(vocab_size, 1)
    input_ids = torch.randint(low, high, (batch_size, seq_len), device=device)
    if 0 <= pad_token_id < high:
        input_ids[input_ids == pad_token_id] = (pad_token_id + 1) % high
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def make_gpu_step_runner(
    model: torch.nn.Module,
    lr: float,
    grad_accumulation: int,
):
    """Return a ``step_runner(batch) -> (peak_alloc_mb, peak_reserved_mb)``.

    Mirrors ``src.training.trainer_loop.forward_backward`` + ``optimizer_step``:
    ``compute_loss`` here is the HF-internal CE (``model(labels=...).loss``),
    which is exactly what the real trainer computes and the documented OOM
    driver (``logits.float()`` upcast inside HF). Peak stats are reset
    immediately before the step so the reading is the per-step high-water mark
    *including* the already-resident model + optimizer state.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    def run_step(batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        try:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_accumulation
            loss.backward()
            optimizer.step()
        except BaseException as exc:  # noqa: BLE001 — re-raise as PeakMemoryOOM
            if _is_oom(exc):
                # Free the graph tensors that OOM left allocated before raising,
                # so the next (smaller) attempt starts from a clean cache.
                try:
                    del outputs, loss  # noqa: F821 — best-effort if bound
                except NameError:
                    pass
                raise PeakMemoryOOM(str(exc)) from exc
            raise
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
        del outputs, loss
        return peak_alloc, peak_reserved

    return run_step, optimizer


def gpu_cleanup(_seq_len: int) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_seq_lens(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seq_lens = [int(p) for p in parts]
    if not seq_lens:
        raise ValueError("--seq-lens must contain at least one value")
    return seq_lens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Empirical 9B suffix-only GPU memory-frontier probe."
    )
    parser.add_argument(
        "--config",
        default="configs/9b_baseline_suffix_only_last25.yaml",
        help="Config to bind to the GPU (default: the suffix-only baseline).",
    )
    parser.add_argument(
        "--seq-lens",
        default="256,512,768,1024,1280,1536,2048",
        help="Comma-separated seq_lens to probe, ascending.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=8,
        help="Matches the config's grad_accumulation (memory-equivalent to 1).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON evidence to stdout.")
    parser.add_argument("--output", default=None, help="Write the JSON report to this path.")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not torch.cuda.is_available():
        logger.error("CUDA not available — this probe measures GPU memory. Aborting.")
        return 2

    cfg = OmegaConf.load(args.config)
    seq_lens = parse_seq_lens(args.seq_lens)

    # Late import: the public model-loading helpers (no src.data dependency).
    from src.model.load_model import apply_lora, load_base_model, load_tokenizer
    from src.model.lora_utils import configure_trainable_lora_scope

    device = torch.device("cuda")
    gpu_total_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2
    gpu_name = torch.cuda.get_device_name(device)

    logger.info("Loading tokenizer + base model (4-bit) for %s ...", cfg.model.name_or_path)
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = apply_lora(model, cfg)
    scope = cfg.training.get("trainable_lora_scope", "all")
    active_names, _active_indices = configure_trainable_lora_scope(model, scope)
    logger.info(
        "LoRA scope=%s -> %d trainable params across %d names",
        scope,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        len(active_names),
    )
    model.train()

    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def batch_builder(seq_len: int) -> dict[str, torch.Tensor]:
        return build_synthetic_batch(
            vocab_size, pad_token_id, seq_len, args.batch_size, device
        )

    run_step, _optimizer = make_gpu_step_runner(model, args.lr, args.grad_accumulation)

    # Warmup at the smallest seq_len to materialize optimizer state + cuDNN/cuBLAS
    # workspaces so they are resident *before* peak stats are reset per step.
    logger.info("Warmup step at seq_len=%d (materialize optimizer/cuda state) ...", seq_lens[0])
    try:
        warm_batch = batch_builder(seq_lens[0])
        run_step(warm_batch)
        del warm_batch
        gpu_cleanup(seq_lens[0])
    except PeakMemoryOOM:
        logger.warning("Warmup itself OOMed at seq_len=%d — recording as-is.", seq_lens[0])

    measurements = sweep_seq_lens(seq_lens, batch_builder, run_step, gpu_cleanup)

    result = FrontierResult(
        gpu_name=gpu_name,
        gpu_total_mb=gpu_total_mb,
        seq_lens=seq_lens,
        measurements=measurements,
        max_fit_seq_len=max_fit_seq_len(measurements),
        full_section4_producible=full_section4_producible(measurements),
        config_levers=config_levers_from_cfg(cfg),
        note=(
            "Per-step peak of a plain forward+backward+AdamW step under the "
            "config's base levers. prefix_feature_cache_offload_prefix_to_cpu "
            "(if set) is NOT exercised here — this is an upper bound on "
            "per-step GPU memory; the actual config may fit more."
        ),
    )

    report = format_report(result)
    print(report, flush=True)

    payload = result.to_dict()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Wrote JSON report to %s", args.output)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    # Non-zero exit if the full §4 threshold is unreachable, so a caller/CI can
    # branch on it. UNMEASURED (None) exits 0 (no claim either way).
    if result.full_section4_producible is False:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
