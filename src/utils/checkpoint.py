import logging
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from src.tg_lora.cycle_state import CycleState
from src.tg_lora.delta_tracker import DeltaTracker
from src.tg_lora.dynamic_freeze import DynFreezeState
from src.tg_lora.random_walk_controller import ControllerState
from src.tg_lora.velocity import Velocity
from src.utils.atomic_save import _atomic_torch_save
from src.utils.tensor_artifact import load_tensor_artifact

logger = logging.getLogger(__name__)


def _sanitize_tensors(tensor_dict: dict[str, torch.Tensor], label: str) -> None:
    """Replace NaN/Inf in loaded tensors with zeros and log a warning."""
    bad_keys: list[str] = []
    for k, v in tensor_dict.items():
        if not torch.isfinite(v).all():
            bad_keys.append(k)
            tensor_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if bad_keys:
        logger.warning(
            "Sanitized non-finite values in %s for keys: %s", label, bad_keys
        )


def _atomic_publish_checkpoint_dir(tmp_dir: Path, save_dir: Path) -> None:
    """Publish a fully-written *tmp_dir* as *save_dir* without ever leaving a
    torn or half-swapped destination.

    Each step is an atomic POSIX directory rename (``os.replace``). On the
    overwrite path (e.g. ``best_model/``) the prior destination is moved aside
    to a PID-suffixed backup BEFORE the new one is renamed in, so a fault
    between the two renames leaves the destination either at its new (complete)
    value or restored to its prior value — never empty, never a mix of old and
    new files. The costly weight bytes are written fully into *tmp_dir* by
    ``save_pretrained`` (→ ``safetensors.save_file``, which writes directly to
    its target with no temp+rename) before this is ever called, so a torn
    ``adapter_model.safetensors`` — the costliest artifact to lose to a torn
    write, and one that sat OUTSIDE the ``_atomic_torch_save`` guarantee — can
    never be what resume's ``load_file`` loads.
    """
    backup_dir = save_dir.parent / f"{save_dir.name}.old.{os.getpid()}"
    swap_landed = False
    try:
        if save_dir.exists():
            os.replace(save_dir, backup_dir)
        os.replace(tmp_dir, save_dir)
        swap_landed = True
    finally:
        if swap_landed:
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            # Swap did not land: restore the prior destination if we moved it
            # aside, so it reflects the last complete checkpoint rather than a
            # gap, and remove the orphan temp holding the (never-published) new
            # bytes so a crashed run does not litter the checkpoint directory.
            if backup_dir.exists() and not save_dir.exists():
                os.replace(backup_dir, save_dir)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)


def save_checkpoint(model: torch.nn.Module, tokenizer, save_dir: Path) -> None:
    """Save model and tokenizer to *save_dir* atomically, with readback verify.

    Used by both baseline and TG-LoRA trainers for periodic saves and
    best-model persistence. ``model.save_pretrained`` ultimately calls
    ``safetensors.save_file``, which writes directly to its destination with no
    temp+rename — so a SIGINT (or OOM kill) during the multi-MB
    ``adapter_model.safetensors`` dump would otherwise leave a torn weight file
    that resume's ``load_file`` (``train_tg_lora.py``) would crash on or
    silently restore as corrupt. To close that gap the full save is staged in a
    PID-suffixed sibling temp dir and published by
    :func:`_atomic_publish_checkpoint_dir` only once complete; a fault
    mid-stage (``except BaseException`` catches ``KeyboardInterrupt``/
    ``SystemExit`` too, mirroring ``_atomic_torch_save``) removes the orphan
    temp so resume never sees a partial adapter.
    """
    save_dir = Path(save_dir)
    tmp_dir = save_dir.parent / f"{save_dir.name}.tmp.{os.getpid()}"
    # Clear an orphan temp left by a prior crashed run with the same PID before
    # reusing it (the PID suffix makes cross-run reuse unlikely, not impossible).
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        # Readback verification on the staged temp BEFORE it is published, so a
        # save that silently produced nothing is surfaced (and, for an empty
        # result, still published to preserve prior behavior).
        file_count = sum(1 for _ in tmp_dir.iterdir())
        if file_count == 0:
            logger.warning("Checkpoint save produced an empty directory: %s", tmp_dir)
    except BaseException:
        # Never publish a partial checkpoint: the destination is untouched and
        # the orphan temp is removed.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    _atomic_publish_checkpoint_dir(tmp_dir, save_dir)


_CYCLE_DIR_RE = re.compile(r"^checkpoint-cycle-(\d+)$")

# Baseline QLoRA trainer writes ``checkpoint-<global_step>`` (no ``-cycle-``
# token). Disjoint from ``_CYCLE_DIR_RE``: ``checkpoint-cycle-5`` has no digits
# immediately after ``checkpoint-`` so this never matches it, and a cycle run_dir
# never contains bare ``checkpoint-<N>`` dirs — so the step and cycle pruners can
# never touch each other's recovery points.
_STEP_DIR_RE = re.compile(r"^checkpoint-(\d+)$")

# trajectory_delta_artifacts/{mode}_{anchor}_{cycle|step}_NNNNNN.pt — the integer
# suffix (group 1) orders them; anchor name is variable-length so we anchor on
# the trailing _cycle/_step token, not the mode/anchor prefix.
_ARTIFACT_KEY_RE = re.compile(r"_(?:cycle|step)_(\d+)\.pt$")
_TRAJECTORY_ARTIFACT_SUBDIR = "trajectory_delta_artifacts"


def _sorted_cycle_checkpoint_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(cycle, dir)`` pairs for ``checkpoint-cycle-<N>`` under *run_dir*,
    oldest cycle first."""
    found: list[tuple[int, Path]] = []
    for p in Path(run_dir).iterdir():
        if p.is_dir():
            m = _CYCLE_DIR_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def _sorted_step_checkpoint_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(step, dir)`` pairs for ``checkpoint-<N>`` under *run_dir*,
    oldest step first.

    The baseline QLoRA trainer (``train_baseline_qlora``) writes
    ``checkpoint-<global_step>`` every ``save_every_steps`` — a different naming
    scheme from the TG-LoRA loop's ``checkpoint-cycle-<N>``, which the cycle
    guard's :data:`_CYCLE_DIR_RE` deliberately does not match. Sibling of
    :func:`_sorted_cycle_checkpoint_dirs`; non-numeric siblings
    (``best_model``, ``oom_checkpoint``) are ignored, never pruned. A missing
    *run_dir* yields ``[]`` (safe noop), mirroring
    :func:`_sorted_trajectory_delta_artifact_files`.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if p.is_dir():
            m = _STEP_DIR_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def _sorted_trajectory_delta_artifact_files(
    run_dir: Path,
) -> list[tuple[int, Path]]:
    """Return ``(cycle_or_step, file)`` pairs for ``*.pt`` trajectory-delta
    artifacts under ``run_dir/trajectory_delta_artifacts/``, oldest key first.

    Filenames are ``{mode}_{anchor_kind}_{cycle|step}_NNNNNN.pt`` (see
    ``artifact_file_name``); the integer suffix orders them. Files without a
    parseable suffix (e.g. a future manifest) are left alone — unsortable, so
    never pruned. Multiple files per key (the pilot + speculative anchors saved
    each cycle) group together and survive/fall as a unit.
    """
    art_dir = Path(run_dir) / _TRAJECTORY_ARTIFACT_SUBDIR
    if not art_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for p in art_dir.iterdir():
        if p.is_file():
            m = _ARTIFACT_KEY_RE.search(p.name)
            if m:
                found.append((int(m.group(1)), p))
    found.sort(key=lambda t: (t[0], t[1].name))
    return found


def _bound_keyed_paths(
    run_dir: Path,
    keyed_paths: list[tuple[int, Path]],
    keep_last: int,
    min_free_bytes: float,
    *,
    remove,
) -> list[Path]:
    """Count-bound then disk-floor a list of ``(key, Path)`` pairs.

    Shared by the checkpoint-dir and trajectory-artifact pruners so the two
    on-disk growth vectors of the M10.3 disk-death class share one proven
    policy. ``keyed_paths`` must be oldest-key-first (the caller's sort; ties
    keep the caller's order). ``keep_last`` bounds the number of **distinct
    keys** that survive: every path whose key is older than the newest
    ``keep_last`` keys is removed. ``min_free_bytes`` then removes whole oldest
    keys until the filesystem floor is met, but never below the single newest
    key — so one recovery point's worth of data always survives.

    ``remove`` is the per-item destructor (``shutil.rmtree`` for dirs, ``unlink``
    for files). Returns removed Paths, oldest-first. Idempotent; safe no-op on
    an empty list. Knobs are clamped non-negative, so ``0`` disables either
    guard (the safe default preserving unbounded behavior for unrelated runs).
    """
    if not keyed_paths:
        return []

    keep_last = max(0, int(keep_last))
    min_free_bytes = max(0.0, float(min_free_bytes))
    removed: list[Path] = []

    distinct_keys = sorted({key for key, _ in keyed_paths})
    surviving = list(keyed_paths)

    # 1) Count bound: drop every path whose key predates the newest keep_last keys.
    if keep_last > 0 and keep_last < len(distinct_keys):
        cut_keys = set(distinct_keys[: len(distinct_keys) - keep_last])
        kept: list[tuple[int, Path]] = []
        for key, path in surviving:
            if key in cut_keys:
                remove(path)
                removed.append(path)
            else:
                kept.append((key, path))
        surviving = kept

    # 2) Disk floor: remove whole oldest keys until the floor is met, but never
    #    below the single newest key.
    if min_free_bytes > 0:
        by_key: dict[int, list[Path]] = {}
        for key, path in surviving:
            by_key.setdefault(key, []).append(path)
        live_keys = sorted(by_key)
        while (
            len(live_keys) > 1
            and shutil.disk_usage(str(run_dir)).free < min_free_bytes
        ):
            oldest = live_keys.pop(0)
            for path in by_key[oldest]:
                remove(path)
                removed.append(path)

    return removed


def prune_checkpoint_cycles(
    run_dir: Path,
    keep_last: int = 0,
    min_free_disk_gb: float = 0.0,
) -> list[Path]:
    """Bound on-disk checkpoint growth by removing old ``checkpoint-cycle-*`` dirs.

    Two independent guards; either may be ``0`` to disable (the safe default,
    preserving today's unbounded behavior for arbitrary configs):

    - ``keep_last``: when > 0, retain only the newest ``keep_last`` checkpoint
      dirs and delete the rest. This is the primary bound against the M10.3
      disk-death class — a run saving every cycle otherwise accumulates one dir
      per cycle forever.
    - ``min_free_disk_gb``: when > 0, if free space on *run_dir*'s filesystem is
      below this floor, delete the oldest checkpoint dirs first until the floor
      is met. This is emergency reclamation: it MAY go below ``keep_last``, but
      never deletes the single newest checkpoint, so one recovery point always
      survives.

    Cycle order is read from the integer suffix of ``checkpoint-cycle-<N>``.
    Returns the dirs removed, oldest-first. Idempotent; safe with no dirs.
    """
    run_dir = Path(run_dir)
    min_free_bytes = max(0.0, float(min_free_disk_gb)) * (1024 ** 3)
    return _bound_keyed_paths(
        run_dir,
        _sorted_cycle_checkpoint_dirs(run_dir),
        keep_last,
        min_free_bytes,
        remove=lambda p: shutil.rmtree(p, ignore_errors=True),
    )


def prune_step_checkpoints(
    run_dir: Path,
    keep_last: int = 0,
    min_free_disk_gb: float = 0.0,
) -> list[Path]:
    """Bound on-disk growth of the baseline trainer's ``checkpoint-<step>`` dirs.

    :func:`prune_checkpoint_cycles` (the first M10.3 vector) matches only
    ``checkpoint-cycle-*``. But the baseline QLoRA trainer
    (``train_baseline_qlora``) writes ``checkpoint-<global_step>`` every
    ``save_every_steps`` into *run_dir* and never removes old ones — the same
    unbounded per-save accumulation class, on the other training entrypoint,
    with a naming scheme the cycle regex deliberately does not match. A long
    baseline run or a sweep re-introduces the M10.3 incident class here unless
    this pruner fires.

    Identical contract to :func:`prune_checkpoint_cycles`: ``keep_last`` retains
    only the newest ``keep_last`` step dirs; ``min_free_disk_gb`` reclaims oldest
    first when the filesystem is low, never below the single newest step. Either
    may be ``0`` to disable (the safe default preserving unbounded behavior for
    runs that have not opted in). The two are disjoint by construction — see
    :data:`_STEP_DIR_RE` — so a cycle run_dir handed here prunes nothing.
    Returns dirs removed, oldest-first. Idempotent; safe with no dirs.
    """
    run_dir = Path(run_dir)
    min_free_bytes = max(0.0, float(min_free_disk_gb)) * (1024 ** 3)
    return _bound_keyed_paths(
        run_dir,
        _sorted_step_checkpoint_dirs(run_dir),
        keep_last,
        min_free_bytes,
        remove=lambda p: shutil.rmtree(p, ignore_errors=True),
    )


def prune_trajectory_delta_artifacts(
    run_dir: Path,
    keep_last: int = 0,
    min_free_disk_gb: float = 0.0,
) -> list[Path]:
    """Bound on-disk growth of ``run_dir/trajectory_delta_artifacts/*.pt``.

    The cycle guard (:func:`prune_checkpoint_cycles`) only matches
    ``checkpoint-cycle-*`` dirs. But ``save_trajectory_delta_artifacts`` writes
    one ``.pt`` per anchor (pilot + speculative) every
    ``trajectory_delta_artifact_interval`` cycles (default 1) into
    ``trajectory_delta_artifacts/`` and never removes old ones — a second
    unbounded per-cycle accumulation vector on the same autonomous runs the
    cycle guard protects. With ``save_every_cycles: 1`` and 120 cycles that is
    up to ~240 delta-tensor files growing linearly and never reclaimed, and the
    cycle guard's ``min_free_disk_gb`` floor never sees them (different path).

    Same contract as :func:`prune_checkpoint_cycles`: ``keep_last`` retains only
    the newest ``keep_last`` cycle/step keys' files (all anchors for those keys,
    kept together); ``min_free_disk_gb`` reclaims oldest keys first when the
    filesystem is low, never below the single newest key. Either may be ``0`` to
    disable (the safe default preserving unbounded behavior for unrelated runs).
    Returns files removed, oldest-first. Idempotent; safe with no dir/files.
    """
    run_dir = Path(run_dir)
    min_free_bytes = max(0.0, float(min_free_disk_gb)) * (1024 ** 3)
    return _bound_keyed_paths(
        run_dir,
        _sorted_trajectory_delta_artifact_files(run_dir),
        keep_last,
        min_free_bytes,
        remove=lambda p: p.unlink(missing_ok=True),
    )


def prune_checkpoint_cycles_from_cfg(cfg, run_dir: Path) -> list[Path]:
    """Read the pruning knobs from ``cfg.logging`` and prune ``checkpoint-cycle-*``.

    The config-driven entry point wired into the periodic-save path of
    ``train_tg_lora``. Extracted from an inline read+call so the config->prune
    coupling is the unit-testable seam: an untested inline block was a silent
    "protection exists but never fires" risk (a renamed key would leave the
    guard inert while every isolated ``prune_checkpoint_cycles`` test still
    passed). Returns the pruned dirs, oldest-first; empty when both knobs are
    off (the safe default).

    ``cfg`` may be an OmegaConf ``DictConfig`` (prod) or a plain dict (tests);
    both expose ``.get``.
    """
    logging_cfg = cfg.get("logging", {}) if cfg is not None else {}
    keep_last = int(logging_cfg.get("keep_last_checkpoints", 0))
    min_free = float(logging_cfg.get("min_free_disk_gb", 0.0))
    if keep_last <= 0 and min_free <= 0.0:
        return []
    return prune_checkpoint_cycles(
        run_dir, keep_last=keep_last, min_free_disk_gb=min_free
    )


def prune_step_checkpoints_from_cfg(cfg, run_dir: Path) -> list[Path]:
    """Read the pruning knobs from ``cfg.logging`` and prune ``checkpoint-<step>``.

    The baseline entrypoint's mirror of :func:`prune_checkpoint_cycles_from_cfg`.
    Targets the baseline trainer's ``checkpoint-<global_step>`` dirs — the
    cycle guard's regex never matches them, so without this seam the same
    ``keep_last_checkpoints`` / ``min_free_disk_gb`` knobs that bound the TG-LoRA
    path are inert for ``make train-baseline`` (the "protection exists but
    doesn't fire on this code path" class). Reads the SAME knobs so a single
    opt-in bounds every on-disk vector across both training entrypoints; the
    baseline configs ship default-off, so this is a no-op for them until they
    opt in. Returns pruned dirs, oldest-first; empty when both knobs are off.

    ``cfg`` may be an OmegaConf ``DictConfig`` (prod) or a plain dict (tests);
    both expose ``.get``.
    """
    logging_cfg = cfg.get("logging", {}) if cfg is not None else {}
    keep_last = int(logging_cfg.get("keep_last_checkpoints", 0))
    min_free = float(logging_cfg.get("min_free_disk_gb", 0.0))
    if keep_last <= 0 and min_free <= 0.0:
        return []
    return prune_step_checkpoints(
        run_dir, keep_last=keep_last, min_free_disk_gb=min_free
    )


def prune_trajectory_delta_artifacts_from_cfg(cfg, run_dir: Path) -> list[Path]:
    """Read the pruning knobs from ``cfg.logging`` and prune trajectory artifacts.

    Mirrors :func:`prune_checkpoint_cycles_from_cfg` but targets the
    ``trajectory_delta_artifacts/*.pt`` vector. Reads the SAME
    ``keep_last_checkpoints`` / ``min_free_disk_gb`` knobs so a single opt-in
    bounds BOTH on-disk growth vectors of the M10.3 class. The M10 configs
    already set these knobs AND ``save_trajectory_delta_artifacts: true``, so
    this fires for them with no config change — closing the gap where the knobs
    were "on" but inert for artifacts (the cycle guard never scanned that path).
    Returns pruned files, oldest-first; empty when both knobs are off (the safe
    default that preserves unbounded behavior for unrelated runs).

    ``cfg`` may be an OmegaConf ``DictConfig`` (prod) or a plain dict (tests);
    both expose ``.get``.
    """
    logging_cfg = cfg.get("logging", {}) if cfg is not None else {}
    keep_last = int(logging_cfg.get("keep_last_checkpoints", 0))
    min_free = float(logging_cfg.get("min_free_disk_gb", 0.0))
    if keep_last <= 0 and min_free <= 0.0:
        return []
    return prune_trajectory_delta_artifacts(
        run_dir, keep_last=keep_last, min_free_disk_gb=min_free
    )


def save_periodic_cycle_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    checkpoint_dir: Path,
    run_dir: Path,
    cfg,
    training_state: "TrainingState",
    *,
    log_artifact=None,
) -> list[Path]:
    """Periodic-save path: persist a cycle checkpoint, log it, then prune old dirs.

    The exact save -> state -> artifact -> prune sequence the TG-LoRA training
    loop runs every ``save_every_cycles``. Extracted from an inline block so the
    M10.3 disk-death guard — the config-driven prune step — is a named,
    importable, unit-tested seam rather than an anonymous call inside a
    multi-thousand-line loop. A silent drop or mis-guard of the prune step here
    re-opens the unbounded ``checkpoint-cycle-*`` accumulation class while every
    isolated ``prune_checkpoint_cycles`` test stays green: the protection would
    exist but never fire in the real save path.

    Order matters: ``save_checkpoint`` writes model+tokenizer, then
    ``save_training_state`` adds ``training_state.pt`` into the same dir, so the
    artifact logged next carries the full checkpoint. Pruning runs last and
    never touches the just-written (newest) dir. ``log_artifact`` is optional so
    the seam is exercisable without a logger; the trainer passes the bound
    ``mlf.log_artifact`` (which no-ops when MLflow is disabled). Returns the dirs
    pruned, oldest-first; empty when both ``cfg.logging`` knobs are off (the safe
    default that preserves unbounded behavior for unrelated runs).
    """
    save_checkpoint(model, tokenizer, checkpoint_dir)
    save_training_state(training_state, checkpoint_dir / "training_state.pt")
    if log_artifact is not None:
        log_artifact(checkpoint_dir, "checkpoints")
    return prune_checkpoint_cycles_from_cfg(cfg, run_dir)


@dataclass
class TrainingState:
    """All state needed to resume a TG-LoRA training job from a checkpoint."""

    cycle_state: CycleState
    controller_state: ControllerState
    velocity: Velocity
    delta_tracker: DeltaTracker
    cycle_offset: int = 0
    adapter_checkpoint_dir: str | None = None
    train_batch_position: int = 0
    accepted_valid_history: list[float] | None = None
    dynfreeze_state: DynFreezeState | None = None
    # Best full-eval loss/perplexity seen so far — gates the run_dir/"best_model"
    # save via a PLAIN improvement check, which is distinct from
    # ``cycle_state.best_loss`` (that one carries the §5.3 ``min_delta`` margin
    # and drives early-stopping). Must survive resume: without it the first
    # post-resume full eval sees ``best_full_eval_loss=inf`` and unconditionally
    # overwrites the genuinely-best pre-fault "best_model" artifact — a silent
    # state-loss on fault-resume, sibling to the fixed ``dynfreeze_state`` gap.
    best_full_eval_loss: float = float("inf")
    best_full_eval_perplexity: float | None = None
    # Warmup phase (two-phase gate). ``warmup_released`` flips True once the
    # cosine-predicted consistency holds for ``warmup_release_count`` consecutive
    # cycles; while False the loop runs pilot-only and bypasses
    # adapt_to_convergence / adapt_to_acceleration / extrapolation. It is NOT
    # monotonic: the M9 subspace-accept path intentionally resets it to False
    # mid-run to re-accumulate direction, so a checkpoint can legitimately catch
    # it in *either* phase. Must survive resume: without it a checkpoint taken
    # mid-production loads as warmup_released=False and the resumed run silently
    # drops back into the warmup phase — re-disabling convergence/acceleration
    # adaptation and extrapolation until the gate re-fires. Sibling resume
    # state-loss to the fixed ``best_full_eval_*`` / ``dynfreeze_state`` gaps.
    warmup_released: bool = False
    warmup_cos_consecutive: int = 0
    # LAWA (mandatory baseline, GOAL §3.3) snapshot window — the serialized
    # ``LAWAAverager.state_dict()`` (window_size / start_cycle / counters + the
    # CPU LoRA-snapshot buffer). Must survive resume: without it a fault-resume
    # rebuilds the averager empty, ``is_ready`` is False, and the LAWA
    # comparison plus LAWA-averaged JSON eval are silently skipped until
    # ``start_cycle`` worth of new snapshots re-accumulate — the resumed
    # headline-quality baseline measured over a different (post-fault-only)
    # window. Sibling resume-state-loss to the fixed dynfreeze / best_full_eval
    # / warmup gaps. ``None`` = LAWA disabled or a pre-fix checkpoint.
    lawa_state: dict | None = None
    # Best LAWA loss observed across the run — the caller-scoped minimum of the
    # GOAL §3.3 mandatory-baseline comparison (``evaluate_with_lawa``), reported
    # in the run summary JSON. Sibling tracker to ``best_full_eval_loss``: the
    # LAWA fault-resume fix (``0eb6fdb``) persisted the snapshot ``lawa_state``
    # window but left ``best_lawa_loss`` un-persisted, so a fault/periodic resume
    # reset it to ``inf`` and the post-resume ``best_lawa_loss`` headline
    # reflected only post-resume cycles (the fix's own PURPOSE note flagged this
    # as "best_lawa_loss も inf にリセット"). Must survive resume for a faithful
    # run-end headline. Mirrors the ``best_full_eval_*`` legacy path.
    best_lawa_loss: float = float("inf")
    # Linearity-budget target steps already fired (250/500/.../1500). The
    # training loop's ``_check_and_save_linearity_budget_checkpoint`` fires a
    # mandatory full eval + ``checkpoint-{target}`` save + a step-aligned
    # ``is_step_aligned_full_eval`` record in ``run_metrics.jsonl`` exactly once
    # per target, guarded by ``target not in triggered_target_steps``. Must
    # survive resume: without it a fault/periodic resume resets the set to empty
    # and the first post-resume cycle re-fires EVERY already-crossed target —
    # redundant full evals, re-saved ``checkpoint-{target}`` dirs, and DUPLICATE
    # ``aligned_target`` records corrupting the linearity-budget vs-baseline
    # comparison dataset (a downstream reader keyed on ``step`` sees the
    # post-resume value twice / the pre-resume value overwritten). Sibling
    # resume-state-loss to the fixed dynfreeze / best_full_eval / warmup / lawa
    # gaps. Modeled as a list (sorted at save) so the legacy path mirrors
    # ``accepted_valid_history``; the trainer restores it to a ``set`` on resume.
    triggered_target_steps: list[int] | None = None
    # Activation-fingerprint regime inventory (GOAL §4 step 1) run-wide state —
    # the serialized ``ActivationFingerprintTracker.state_dict()`` (full cosine
    # series + per-regime counts + classification window + current regime). Must
    # survive resume: without it a fault/periodic resume rebuilds the tracker
    # empty and the run-end summary's ``activation_regime_inventory`` /
    # ``stable_fraction`` reflect only post-resume steps — a silent
    # resume-state-loss sibling to the fixed LAWA (``lawa_state``) / dynfreeze /
    # best_full_eval gaps. ``None`` = activation-regime disabled
    # (``activation_regime_enabled: false``) or a pre-fix checkpoint.
    act_regime_state: dict | None = None
    # Run-wide efficiency-accounting counters (GOAL §5 / P3 cost accounting) —
    # a snapshot of the ``train_tg_lora`` caller-scoped tallies that accumulate
    # across the whole run and feed the run-end summary (cache hit-rate,
    # validation_forwards_total, post-extrapolation eval accounting, subspace-ZO
    # / alpha-line step tallies, future-work projection ratios). Must survive
    # resume: without it a fault/periodic resume rebuilds every counter at
    # zero/empty and the run-end cost report reflects only post-resume cycles —
    # a silent resume-state-loss sibling to the fixed LAWA (``lawa_state``) /
    # act-regime (``act_regime_state``) / dynfreeze gaps. Plain dict so mixed
    # int / float / list / dict counter types round-trip uniformly. ``None`` =
    # a pre-fix checkpoint (every counter resumes at its zero/empty init, the
    # pre-fix behavior — no fabricated data).
    efficiency_accounting: dict | None = None
    # PSA (Prior-based Subspace Amplification, GOAL §1.5 / §3.3 baseline route)
    # subspace-prior run-wide state — the serialized ``PSAPrior.state_dict()``
    # (per-step incremental ``_delta_history`` ring buffer + extracted PC1
    # ``priors`` that drive production gradient amplification + the L2-reg
    # ``_prev_priors`` anchor + the unbounded ``_prior_cosines`` stability series
    # + the ``should_update`` timing). Must survive resume: without it a
    # fault/periodic resume rebuilds ``psa_prior`` empty, amplification is
    # silently off until 2 deltas re-accumulate and the next extract fires, and
    # the run-end ``layer_delta_analysis`` (gated on ``history_count >= 2``) is
    # omitted entirely if the residual run is short — a silent resume-state-loss
    # sibling to the fixed LAWA (``lawa_state``) / act-regime
    # (``act_regime_state``) / efficiency-accounting gaps. ``None`` = PSA
    # disabled (``enable_psa: false`` — currently every config in this mirror)
    # or a pre-fix checkpoint (the prior rebuilds empty, the pre-fix behavior —
    # no fabricated priors).
    psa_state: dict | None = None
    # PSA regime detector (GOAL §1.5 / §3.3, the ``RegimeDetector`` that gates
    # the PSA prior reset on a stable→transition shift) run-wide state — the
    # serialized ``RegimeDetector.state_dict()`` (loss / velocity classification
    # windows + current regime + the run-wide transition count). Must survive
    # resume: without it a fault/periodic resume rebuilds the detector fresh and
    # the per-cycle ``psa_regime_transitions`` (persisted to ``run_metrics.jsonl``)
    # resets to 0 — a silent resume-state-loss sibling to the fixed PSA prior
    # (``psa_state``) / activation-regime (``act_regime_state``) gaps. ``None`` =
    # PSA disabled (``enable_psa: false`` — currently every config in this mirror)
    # or a pre-fix checkpoint (the detector rebuilds fresh, the pre-fix behavior —
    # no fabricated transitions).
    psa_regime_state: dict | None = None
    # Async prefix-cache swap completion cycles (GOAL §3.3 cache-ablation
    # apparatus) — the caller-scoped cycle at which the ``valid_quick`` /
    # ``valid_full`` loader was swapped to the asynchronously-built cached
    # dataset (``async_builder.poll()`` succeeds mid-run), reported in the
    # run-end summary as ``async_cache_swap_cycle_valid_quick`` /
    # ``async_cache_swap_cycle_valid_full``. Must survive resume: without it a
    # fault/periodic resume resets both to ``None`` (the caller init) and the
    # run-end summary silently drops both swap-cycle fields — a silent
    # resume-state-loss sibling to the fixed act-regime (``act_regime_state``)
    # / efficiency-accounting / LAWA gaps. ``None`` = async cache building
    # disabled (no current config enables it) or the swap had not yet fired, or
    # a pre-fix checkpoint (the field stays ``None``, the pre-fix behavior — no
    # fabricated cycle).
    swap_cycle_vq: int | None = None
    swap_cycle_vf: int | None = None
    # Progressive Freeze (GOAL §1.6 / design §4.1) cumulative frozen-layer set —
    # the serialized ``ProgressiveFreezeController.state_dict()`` (sorted frozen
    # layer indices + most-recent index). Must survive resume: the training loop
    # rebuilds the controller fresh from config and restores LoRA weights from
    # safetensors (weights only — NOT the freeze's ``requires_grad=False`` flag),
    # and the cycle loop's ``layers_due_at(cycle)`` gate fires only for cycles
    # ``>= cycle_offset``, so the pre-fault cumulative freezes are never
    # re-applied. Without this the frozen layers silently re-train (undoing the
    # cost reduction that defines Progressive Freezing) AND the run-summary
    # footer's ``frozen_layers`` — the Tier-2 §4 order-verdict arm provenance —
    # reports only post-fault freezes. Sibling resume-state-loss to dynfreeze /
    # LAWA / warmup (the controller's own state_dict docstring states why the
    # Level-2 ``xin`` caches are not persisted: a separate Phase-3 axis not on
    # the Tier-2 valid_loss path). ``None`` = progressive-freeze disabled or a
    # pre-fix checkpoint (the set rebuilds empty, the pre-fix behavior — no
    # fabricated freezes).
    progressive_freeze_state: dict | None = None


@dataclass
class BaselineTrainingState:
    """State needed to resume baseline QLoRA training from a checkpoint."""

    global_step: int = 0
    best_loss: float = float("inf")
    best_step: int = 0
    stale_steps: int = 0
    train_batch_position: int = 0
    adapter_checkpoint_dir: str | None = None


def save_baseline_training_state(
    state: BaselineTrainingState,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "global_step": state.global_step,
        "best_loss": state.best_loss,
        "best_step": state.best_step,
        "stale_steps": state.stale_steps,
        "train_batch_position": state.train_batch_position,
        "adapter_checkpoint_dir": state.adapter_checkpoint_dir,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    _atomic_torch_save(blob, path)
    logger.info(
        "Saved baseline training state to %s (step %d)", path, state.global_step
    )


def load_baseline_training_state(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline training state not found: {path}")
    blob = load_tensor_artifact(path)
    state = BaselineTrainingState(
        global_step=blob.get("global_step", 0),
        best_loss=blob.get("best_loss", float("inf")),
        best_step=blob.get("best_step", 0),
        stale_steps=blob.get("stale_steps", 0),
        train_batch_position=blob.get("train_batch_position", 0),
        adapter_checkpoint_dir=blob.get("adapter_checkpoint_dir"),
    )
    return {
        "state": state,
        "optimizer_state_dict": blob.get("optimizer_state_dict"),
        "scheduler_state_dict": blob.get("scheduler_state_dict"),
    }


def save_training_state(state: TrainingState, path: Path) -> None:
    """Serialize training state to disk for later recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    velocity_tensors = None
    if state.velocity._state is not None:
        velocity_tensors = {k: v.cpu() for k, v in state.velocity._state.items()}
    velocity_short_tensors = None
    if state.velocity._short_state is not None:
        velocity_short_tensors = {
            k: v.cpu() for k, v in state.velocity._short_state.items()
        }
    velocity_long_tensors = None
    if state.velocity._long_state is not None:
        velocity_long_tensors = {
            k: v.cpu() for k, v in state.velocity._long_state.items()
        }

    delta_tensors = None
    if state.delta_tracker._history:
        delta_tensors = [
            {k: v.cpu() for k, v in h.items()} for h in state.delta_tracker._history
        ]

    blob = {
        "cycle_state": state.cycle_state.summary(),
        "controller_state": state.controller_state.summary(),
        "velocity_tensors": velocity_tensors,
        "velocity_short_tensors": velocity_short_tensors,
        "velocity_long_tensors": velocity_long_tensors,
        "velocity_magnitudes": state.velocity.magnitudes,
        "velocity_max_history": state.velocity._max_history,
        "velocity_beta_short": state.velocity.beta_short,
        "velocity_beta_long": state.velocity.beta_long,
        "velocity_update_count": state.velocity.update_count,
        "delta_tensors": delta_tensors,
        "delta_norm_history": state.delta_tracker.norm_history,
        "delta_max_history": state.delta_tracker._max_history,
        "cycle_offset": state.cycle_offset,
        "adapter_checkpoint_dir": state.adapter_checkpoint_dir,
        "train_batch_position": state.train_batch_position,
        "accepted_valid_history": state.accepted_valid_history,
        "best_full_eval_loss": state.best_full_eval_loss,
        "best_full_eval_perplexity": state.best_full_eval_perplexity,
        "warmup_released": state.warmup_released,
        "warmup_cos_consecutive": state.warmup_cos_consecutive,
        "lawa_state": (
            # Deep-copy the snapshot buffer so the checkpoint is independent of
            # the live averager's deque; tensors are already CPU at record time.
            {
                "window_size": state.lawa_state["window_size"],
                "start_cycle": state.lawa_state["start_cycle"],
                "cycle": state.lawa_state["cycle"],
                "recorded_count": state.lawa_state["recorded_count"],
                "buffer": [
                    dict(snapshot) for snapshot in state.lawa_state["buffer"]
                ],
            }
            if state.lawa_state is not None
            else None
        ),
        "best_lawa_loss": state.best_lawa_loss,
        # Sorted so the serialized form is deterministic across save order;
        # ``None`` on LAWA-disabled / pre-fix checkpoints. Passed through as-is
        # (already a list at construction — see ``accepted_valid_history``).
        "triggered_target_steps": state.triggered_target_steps,
        "act_regime_state": state.act_regime_state,
        "efficiency_accounting": state.efficiency_accounting,
        "psa_state": state.psa_state,
        "psa_regime_state": state.psa_regime_state,
        "swap_cycle_vq": state.swap_cycle_vq,
        "swap_cycle_vf": state.swap_cycle_vf,
        "progressive_freeze_state": state.progressive_freeze_state,
        "dynfreeze_state": (
            {
                "frozen_layer_indices": state.dynfreeze_state.frozen_layer_indices,
                "r_A_history": state.dynfreeze_state.r_A_history,
                "frozen_since_cycle": state.dynfreeze_state.frozen_since_cycle,
                "prev_A_fro": state.dynfreeze_state.prev_A_fro,
                "median_A": state.dynfreeze_state.median_A,
                "epsilon": state.dynfreeze_state.epsilon,
                # §4 release-cooldown map (layer_idx → cycle last released). Must
                # survive the round-trip: a run that checkpoints while a layer is
                # mid-cooldown and resumes otherwise loses the cooldown and §3
                # silently re-freezes the just-released layer on its stale 0.0
                # r_A history — the §4 reversible release undone on resume.
                "released_at": state.dynfreeze_state.released_at,
            }
            if state.dynfreeze_state is not None
            else None
        ),
    }
    _atomic_torch_save(blob, path)
    logger.info("Saved training state to %s (cycle %d)", path, state.cycle_state.cycle)


class CheckpointIntegrityError(RuntimeError):
    """A resume checkpoint exists on disk but is torn/truncated and won't load.

    The load-side counterpart to the atomic-save guarantee. The atomic write
    helpers (:func:`src.utils.atomic_save._atomic_torch_save` for single-file
    artifacts and :func:`_atomic_publish_checkpoint_dir` for the staged LoRA
    adapter dir) make it impossible for the TRAINING PROCESS to publish a torn
    destination — a mid-commit fault leaves the destination either fully
    reflecting the new state or still at its prior, loadable value, never torn.
    So a checkpoint that trips this error could NOT have been torn by an
    in-process fault; it reached the loader from a source the atomic guarantee
    does not govern:

      * a checkpoint written before the atomic helpers landed (commits
        ``510b0d1`` for ``training_state.pt`` and ``620372c`` for the adapter
        dir), still sitting in an old run dir,
      * external corruption — disk-full during a non-atomic copy/backup of the
        run dir, an NFS hiccup, a manual edit, or ``kill -9`` mid-``cp``.

    Resume intentionally does NOT silently fall back to a fresh start: that
    would hide the lost training progress (a GOAL.md honesty break). Instead
    this fails loud, with the original loader error chained (``raise ... from
    exc``), so the operator can delete or restore the corrupt file deliberately.
    Silently restarting would defeat the resume guarantee the 12-site
    persistence axis went to the trouble of capturing.
    """


# ``torch.load(weights_only=True)`` torn-file signatures, captured empirically
# against torch 2.1.1 on truncated / empty / garbage inputs (pinned in
# ``tests/test_checkpoint_integrity.py``):
#   truncated zip archive -> RuntimeError("PytorchStreamReader failed reading
#                                       zip archive: failed finding central
#                                       directory")
#   <8-byte / non-zip     -> RuntimeError("... not a ZIP archive")
#   empty file            -> EOFError
#   non-pickle garbage    -> pickle.UnpicklingError("Weights only load failed. ...")
# ``RuntimeError`` is matched on MESSAGE (not type) so a genuine deserialization
# bug that happens to raise ``RuntimeError`` is NOT masked — it re-raises
# unchanged with its original traceback.
_TORCH_ARCHIVE_CORRUPTION_MARKERS = (
    "failed reading zip archive",
    "not a ZIP archive",
)


def _is_torch_load_corruption(exc: BaseException) -> bool:
    """True if *exc* is the signature ``torch.load`` raises on a torn / empty /
    garbage file (as opposed to a real deserialization bug)."""
    if isinstance(exc, (EOFError, pickle.UnpicklingError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        return any(marker in msg for marker in _TORCH_ARCHIVE_CORRUPTION_MARKERS)
    return False


def _is_safetensors_corruption(exc: BaseException) -> bool:
    """True if *exc* is the signature ``safetensors.torch.load_file`` raises on
    a torn / garbage ``.safetensors`` file. ``safetensors`` is a rigid format —
    it either loads or the bytes are corrupt / truncated / version-mismatched —
    so a ``SafetensorError`` on a file that exists is corruption, full stop."""
    try:
        from safetensors import SafetensorError
    except ImportError:  # pragma: no cover - safetensors is a core dep; be robust
        return type(exc).__module__.startswith("safetensors")
    return isinstance(exc, SafetensorError)


def load_training_state(path: Path) -> TrainingState:
    """Load training state from a previously saved checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training state not found: {path}")

    try:
        blob = load_tensor_artifact(path)
    except (EOFError, pickle.UnpicklingError, RuntimeError) as exc:
        # The load-side counterpart to the atomic-save guarantee. A torn /
        # truncated / empty ``training_state.pt`` raises a corruption-signature
        # error (see :func:`_is_torch_load_corruption`); diagnose it as
        # :class:`CheckpointIntegrityError` so resume fails with an actionable
        # message instead of an opaque ``EOFError``/``RuntimeError`` traceback.
        # A ``RuntimeError`` that does NOT match the corruption signature is a
        # genuine deserialization bug — re-raise it UNCHANGED so it is not masked.
        if _is_torch_load_corruption(exc):
            raise CheckpointIntegrityError(
                f"Training-state checkpoint at {path} exists but is torn or "
                f"corrupt and cannot be loaded for resume "
                f"({type(exc).__name__}: {exc}). The atomic-save helper makes "
                f"it impossible for the training process to WRITE a torn "
                f"destination, so this file predates that helper (commit "
                f"510b0d1) or was corrupted externally (disk-full during a "
                f"non-atomic copy/backup, NFS, manual edit, kill -9 mid-"
                f"transfer). Delete or restore it from a known-good checkpoint "
                f"before resuming; resume intentionally does NOT silently "
                f"restart from scratch, which would hide the lost progress."
            ) from exc
        raise

    # Restore CycleState — from_dict accepts both summary() and legacy checkpoint keys
    cycle_state = CycleState.from_dict(blob["cycle_state"])

    # Restore ControllerState
    controller_state = ControllerState.from_dict(blob["controller_state"])

    # Restore Velocity
    velocity = Velocity(
        max_history=blob.get("velocity_max_history", 100),
        beta_short=blob.get("velocity_beta_short"),
        beta_long=blob.get("velocity_beta_long"),
    )
    velocity_tensors = blob.get("velocity_tensors")
    if velocity_tensors is not None:
        _sanitize_tensors(velocity_tensors, "velocity")
        velocity._state = velocity_tensors
    velocity_short_tensors = blob.get("velocity_short_tensors")
    if velocity_short_tensors is not None:
        _sanitize_tensors(velocity_short_tensors, "velocity_short")
        velocity._short_state = velocity_short_tensors
    velocity_long_tensors = blob.get("velocity_long_tensors")
    if velocity_long_tensors is not None:
        _sanitize_tensors(velocity_long_tensors, "velocity_long")
        velocity._long_state = velocity_long_tensors
    velocity._magnitude_history = blob.get("velocity_magnitudes", [])
    velocity._update_count = blob.get(
        "velocity_update_count",
        len(velocity._magnitude_history),
    )

    # Restore DeltaTracker
    delta_tracker = DeltaTracker(max_history=blob.get("delta_max_history", 100))
    delta_tensors = blob.get("delta_tensors")
    if delta_tensors:
        for i, h in enumerate(delta_tensors):
            _sanitize_tensors(h, f"delta_history[{i}]")
        delta_tracker._history = delta_tensors
    delta_tracker._norm_history = blob.get("delta_norm_history", [])
    # Recompute last_stats from latest history entry
    if delta_tracker._history:
        from src.tg_lora.delta_tracker import _compute_stats

        delta_tracker._last_stats = _compute_stats(delta_tracker._history[-1])

    # Restore DynFreezeState
    dynfreeze_raw = blob.get("dynfreeze_state")
    dynfreeze_state = None
    if dynfreeze_raw is not None:
        dynfreeze_state = DynFreezeState(
            frozen_layer_indices=dynfreeze_raw.get("frozen_layer_indices", []),
            r_A_history=dynfreeze_raw.get("r_A_history", {}),
            frozen_since_cycle=dynfreeze_raw.get("frozen_since_cycle", 0),
            prev_A_fro=dynfreeze_raw.get("prev_A_fro", {}),
            median_A=dynfreeze_raw.get("median_A", 0.0),
            epsilon=dynfreeze_raw.get("epsilon", 1e-6),
            # Absent on pre-fix checkpoints → no active cooldown (the only sane
            # reading of a checkpoint that predates the field). Mirrors the
            # controller's own legacy tolerance in ``load_state_dict``.
            released_at=dynfreeze_raw.get("released_at", {}),
        )

    return TrainingState(
        cycle_state=cycle_state,
        controller_state=controller_state,
        velocity=velocity,
        delta_tracker=delta_tracker,
        cycle_offset=blob.get("cycle_offset", 0),
        adapter_checkpoint_dir=blob.get("adapter_checkpoint_dir"),
        train_batch_position=blob.get("train_batch_position", 0),
        accepted_valid_history=blob.get("accepted_valid_history"),
        # Absent on pre-fix checkpoints → inf/None, the only sane reading of a
        # checkpoint that predates the field (mirrors the dynfreeze legacy path).
        best_full_eval_loss=blob.get("best_full_eval_loss", float("inf")),
        best_full_eval_perplexity=blob.get("best_full_eval_perplexity"),
        # Absent on pre-fix checkpoints → not-yet-released warmup (False/0), the
        # only sane reading of a checkpoint that predates the fields: a True
        # default would skip warmup on a checkpoint that was genuinely warming
        # up, while False merely re-runs a brief warmup on an old mid-production
        # checkpoint — backward-compatible with the pre-fix behavior. Mirrors the
        # best_full_eval_* / dynfreeze legacy tolerance.
        warmup_released=blob.get("warmup_released", False),
        warmup_cos_consecutive=blob.get("warmup_cos_consecutive", 0),
        # Absent on pre-fix checkpoints → None, the only sane reading of a
        # checkpoint that predates the field: the resume path treats a missing
        # window as 'start fresh' (the pre-fix behavior), not a fabricated
        # non-empty window. Mirrors the dynfreeze/best_full_eval legacy path.
        lawa_state=blob.get("lawa_state"),
        # Absent on pre-fix checkpoints → inf, the only sane reading of a
        # checkpoint that predates the field (the headline is recomputed from
        # post-resume cycles, the pre-fix behavior — not a fabricated low).
        # Mirrors the best_full_eval_* / lawa_state legacy paths.
        best_lawa_loss=blob.get("best_lawa_loss", float("inf")),
        # Absent on pre-fix checkpoints → None, the only sane reading of a
        # checkpoint that predates the field: the resume path treats a missing
        # set as 'no target yet fired' and the loop re-fires from the resumed
        # equivalent-step count (the pre-fix behavior). The trainer converts the
        # list back to a set; ``None`` → empty set. Mirrors the
        # accepted_valid_history / best_full_eval_* / best_lawa_loss legacy paths.
        triggered_target_steps=blob.get("triggered_target_steps"),
        # Absent on pre-fix checkpoints → None, the only sane reading of a
        # checkpoint that predates the field: the resume path treats a missing
        # regime inventory as 'start empty' (the pre-fix behavior). Mirrors the
        # lawa_state / triggered_target_steps legacy paths.
        act_regime_state=blob.get("act_regime_state"),
        # Absent on pre-fix checkpoints → None, the only sane reading of a
        # checkpoint that predates the field: the resume path treats a missing
        # accounting bag as 'all counters at zero/empty init' (the pre-fix
        # behavior — no fabricated tallies). Mirrors the lawa_state /
        # act_regime_state / triggered_target_steps legacy paths.
        efficiency_accounting=blob.get("efficiency_accounting"),
        # Absent on pre-fix checkpoints / ``enable_psa: false`` runs → None, the
        # only sane reading of a checkpoint that predates the field or predates
        # PSA: the resume path treats a missing prior as 'start fresh' (the
        # pre-fix behavior, no fabricated priors). Mirrors the act_regime_state /
        # efficiency_accounting / lawa_state legacy paths.
        psa_state=blob.get("psa_state"),
        # Absent on pre-fix checkpoints / ``enable_psa: false`` runs → None, the
        # only sane reading of a checkpoint that predates the field or predates
        # PSA: the resume path treats a missing regime state as 'start fresh'
        # (the pre-fix behavior, no fabricated transitions). Mirrors the psa_state
        # / act_regime_state legacy paths.
        psa_regime_state=blob.get("psa_regime_state"),
        # Absent on pre-fix checkpoints / async-cache-disabled runs → None, the
        # only sane reading of a checkpoint that predates the field or predates
        # a swap firing: the resume path treats a missing cycle as 'no swap yet'
        # (the pre-fix behavior, no fabricated cycle). Mirrors the psa_state /
        # act_regime_state legacy paths.
        swap_cycle_vq=blob.get("swap_cycle_vq"),
        swap_cycle_vf=blob.get("swap_cycle_vf"),
        # Absent on pre-fix checkpoints / progressive-freeze-disabled runs →
        # None, the only sane reading of a checkpoint that predates the field:
        # the resume path treats a missing frozen set as 'start empty' (the
        # pre-fix behavior, no fabricated freezes) and skips refreeze. Mirrors
        # the psa_state / act_regime_state / lawa_state legacy paths.
        progressive_freeze_state=blob.get("progressive_freeze_state"),
        dynfreeze_state=dynfreeze_state,
    )


def load_adapter_weights(adapter_dir: str | Path) -> dict:
    """Load the LoRA adapter state dict from *adapter_dir* with load-side
    integrity diagnosis.

    Symmetric to :func:`save_checkpoint`'s atomic staging + directory publish:
    a torn ``adapter_model.safetensors`` (a pre-fix checkpoint or external
    corruption — the costlier artifact to lose to a torn write) is diagnosed as
    :class:`CheckpointIntegrityError` rather than crashing resume with an opaque
    ``SafetensorError``. ``safetensors`` is imported lazily so this module does
    not hard-depend on it at import time (mirroring the lazy import the resume
    path already used). Returns the state dict for
    ``peft.set_peft_model_state_dict``.

    Raises ``FileNotFoundError`` unchanged when the adapter file is absent (the
    dir exists but the weights do not) — that is a missing-file condition, not
    corruption, and is left to the caller to handle.
    """
    from safetensors.torch import load_file

    adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
    try:
        return load_file(adapter_path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        if _is_safetensors_corruption(exc):
            raise CheckpointIntegrityError(
                f"LoRA adapter checkpoint at {adapter_path} exists but is torn "
                f"or corrupt and cannot be loaded for resume "
                f"({type(exc).__name__}: {exc}). The atomic staging + publish "
                f"in save_checkpoint makes it impossible for the training "
                f"process to WRITE a torn adapter (commit 620372c), so this "
                f"file predates that helper or was corrupted externally. "
                f"Restore it from a known-good checkpoint before resuming."
            ) from exc
        raise
