"""Static guard: the trainer→classifier OOM-defer exit-code contract is wired.

Closes the producer-side half of the AI-Hub feedback gap — "the GPU lock's value
depends on the control plane reading exit code 3 as 'defer and retry' rather than
a training fault". On this public mirror the control plane / autonomous loop is
AI-Hub infrastructure (not in-repo), so the verifiable in-repo realization is the
PRODUCER half: BOTH trainers (``train_tg_lora`` and ``train_baseline_qlora``) must
emit a distinct exit code for a deferrable OOM, and the in-repo classifier
(``scripts/frontier_report.determine_status``) must recognize it. Before this
contract, a *handled* TG-LoRA OOM (fault checkpoint saved, deferrable) logged
"GPU OOM at cycle N" and exited 2 — neither of which the classifier recognized —
so it was misread as a generic "failed" run and the defer/retry signal was lost in
classification. The baseline trainer had the symmetric defect: its graceful-OOM
handler saved a checkpoint and bare-`raise`d the original exception (exit 1), so
it was keyable only off the log line — violating the contract AGENTS.md documents
for BOTH entrypoints.

Five invariants are pinned (each mutation-verifiable):

1. ``src/utils/device.py`` defines ``OOM_EXIT_CODE = 3`` — the single source of
   truth for the contract value.
2. ``src/training/train_tg_lora.py`` routes the fault exit through
   ``fault_exit_code`` (OOM→3, numerical/CUDA→2), NOT a hardcoded ``SystemExit(2)``.
3. ``src/training/train_baseline_qlora.py`` routes its graceful fault exit through
   ``fault_exit_code`` (OOM→3 / cuda_error→2) too, NOT a bare ``raise`` that
   collapses both into a generic exit 1. Symmetric with the TG-LoRA trainer.
4. ``scripts/frontier_report.py`` recognizes ``OOM_EXIT_CODE`` in
   ``determine_status`` so the classifier reads the producer's signal.
5. ``AGENTS.md`` documents the contract so the operator/control-plane side has a
   spec to read exit 3 as "defer and retry".

If a future change reverts EITHER trainer to a bare exit/``raise``, drops the
classifier branch, or removes the AGENTS.md section, this guard fails loud — so
the defer/retry signal cannot silently regress back into the void.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_PY = REPO_ROOT / "src" / "utils" / "device.py"
TRAINER_PY = REPO_ROOT / "src" / "training" / "train_tg_lora.py"
BASELINE_PY = REPO_ROOT / "src" / "training" / "train_baseline_qlora.py"
FRONTIER_PY = REPO_ROOT / "scripts" / "frontier_report.py"
AGENTS_MD = REPO_ROOT / "AGENTS.md"


def test_device_defines_oom_exit_code() -> None:
    """The contract value is defined once, in the device leaf."""
    text = DEVICE_PY.read_text()
    assert "OOM_EXIT_CODE = 3" in text, (
        "src/utils/device.py must define OOM_EXIT_CODE = 3 (the contract value)"
    )
    assert "def fault_exit_code(" in text, (
        "src/utils/device.py must define fault_exit_code(fault_reason)"
    )


def test_trainer_routes_fault_exit_through_helper() -> None:
    """The trainer must NOT hardcode SystemExit(2) for the fault path.

    It routes through fault_exit_code so a deferrable OOM exits OOM_EXIT_CODE (3)
    while numerical/CUDA faults stay at 2. Reverting to a bare SystemExit(2)
    collapses the two and erases the defer/retry signal.
    """
    text = TRAINER_PY.read_text()
    assert "from src.utils.device import fault_exit_code" in text, (
        "train_tg_lora.py must import fault_exit_code from src.utils.device"
    )
    assert "raise SystemExit(fault_exit_code(fault_reason))" in text, (
        "train_tg_lora.py fault exit must route through fault_exit_code(fault_reason), "
        "not a hardcoded exit code — OOM must be distinguishable from a real fault"
    )
    assert "raise SystemExit(2)" not in text, (
        "train_tg_lora.py must not hardcode SystemExit(2) for the fault path; "
        "route through fault_exit_code instead"
    )


def test_baseline_routes_fault_exit_through_helper() -> None:
    """The baseline trainer's graceful fault handler must NOT bare-`raise`.

    It must classify the fault (OOM vs cuda_error) with ``is_gpu_oom_error`` and
    route the exit through ``fault_exit_code`` so a deferrable OOM emits
    ``OOM_EXIT_CODE`` (3) while a real CUDA fault emits 2 — symmetric with the
    TG-LoRA trainer. AGENTS.md documents the contract for BOTH entrypoints;
    before this the baseline saved a fault checkpoint then bare-`raise`d the
    original exception (exit 1), so its handled OOM was keyable only off the log
    line and the documented contract was violated.
    """
    text = BASELINE_PY.read_text()
    assert "from src.utils.device import fault_exit_code, is_gpu_oom_error" in text, (
        "train_baseline_qlora.py graceful-fault handler must import fault_exit_code "
        "and is_gpu_oom_error from src.utils.device"
    )
    assert 'reason = "oom" if is_gpu_oom_error(exc) else "cuda_error"' in text, (
        "train_baseline_qlora.py must classify the fault via is_gpu_oom_error(exc) "
        "so a deferrable OOM is distinguished from a real CUDA fault"
    )
    assert "raise SystemExit(fault_exit_code(reason))" in text, (
        "train_baseline_qlora.py graceful-fault exit must route through "
        "fault_exit_code(reason), not a bare `raise` that collapses a deferrable "
        "OOM and a real CUDA fault into a generic exit 1"
    )


def test_classifier_recognizes_oom_exit_code() -> None:
    """determine_status must read the producer's OOM exit code as 'oom'."""
    text = FRONTIER_PY.read_text()
    assert "OOM_EXIT_CODE = 3" in text, (
        "frontier_report.py must pin its local OOM_EXIT_CODE = 3 (kept equal to "
        "src.utils.device.OOM_EXIT_CODE by the classifier-constant sync test)"
    )
    assert "exit_code == OOM_EXIT_CODE" in text, (
        "determine_status must recognize exit_code == OOM_EXIT_CODE as 'oom' — "
        "otherwise a handled OOM is misclassified as 'failed'"
    )
    # The log-text backstop: the bare \bOOM\b acronym pattern (the trainers log
    # "GPU OOM …" / "OOM checkpoint saved to …"). Without it only exit 137 or the
    # literal "out of memory" substring classifies as OOM.
    assert r"\bOOM\b" in text, (
        "frontier_report.py must include the \\bOOM\\b log pattern so a handled "
        "OOM is also recognizable from its log line"
    )


def test_agents_md_documents_exit_code_contract() -> None:
    """AGENTS.md must document the producer exit-code contract.

    The control-plane interpretation (read 3 as 'defer and retry') is the
    operator/AI-Hub domain; this just requires the PRODUCER side to document what
    codes it emits so that interpretation has a spec to key off.
    """
    text = AGENTS_MD.read_text()
    assert "Process exit codes" in text, (
        "AGENTS.md must document a 'Process exit codes' section (the contract spec)"
    )
    assert "OOM_EXIT_CODE" in text or "exit code 3" in text or "exit 3" in text, (
        "AGENTS.md must name the OOM-defer exit code (3 / OOM_EXIT_CODE) in the contract"
    )
    assert "defer" in text.lower(), (
        "AGENTS.md must state the 'defer and retry' semantics for the OOM exit code"
    )


def test_trainer_classifies_device_dispatch_oom_before_cuda_error() -> None:
    """A torch 2.12 device-dispatch CUDA OOM surfaces as torch.AcceleratorError.

    That is a RuntimeError sibling that is NOT a torch.cuda.OutOfMemoryError, so
    the TG-LoRA trainer's dedicated OOM handler (``except
    torch.cuda.OutOfMemoryError``) does NOT catch it — it falls through to the
    generic ``except RuntimeError`` handler. That handler must classify it as the
    deferrable ``oom`` (exit 3) via ``is_gpu_oom_error`` BEFORE the
    ``_is_cuda_error`` branch: an AcceleratorError OOM message carries "CUDA", so
    the cuda_error check (exit 2, fatal) would otherwise swallow a deferrable OOM
    and a contended-GPU OOM during TG-LoRA training would crash instead of
    deferring. Same sibling-exception class ``train_baseline_qlora.py`` enforces
    and the freeze-ci-9b harness fixed (``is_cuda_oom``). Static + mutation-safe:
    drop the branch, or reorder it after ``_is_cuda_error``, and this fails loud.
    """
    text = TRAINER_PY.read_text()
    marker = "except RuntimeError as exc:"
    start = text.find(marker)
    assert start != -1, (
        "train_tg_lora.py must have an `except RuntimeError as exc:` fault handler"
    )
    rest = text[start:]
    # The block ends at the next sibling `except`/`finally` at the same indent.
    block_end = len(rest)
    for kw in ("\n    except ", "\n    finally:"):
        idx = rest.find(kw, 1)
        if idx != -1:
            block_end = min(block_end, idx)
    block = rest[:block_end]

    oom_idx = block.find("is_gpu_oom_error(exc)")
    assert oom_idx != -1, (
        "The RuntimeError handler must classify an OOM via is_gpu_oom_error(exc) "
        "so a torch 2.12 AcceleratorError device-dispatch OOM (a RuntimeError "
        "sibling the dedicated OOM handler misses) routes to `oom` (exit 3), not "
        "cuda_error (exit 2)"
    )
    cuda_idx = block.find("_is_cuda_error(exc)")
    assert cuda_idx != -1, (
        "The RuntimeError handler must retain its _is_cuda_error(exc) cuda_error branch"
    )
    assert oom_idx < cuda_idx, (
        "is_gpu_oom_error(exc) must be checked BEFORE _is_cuda_error(exc): an "
        "AcceleratorError OOM message contains 'CUDA', so the cuda_error branch "
        "would otherwise reclassify a deferrable OOM as a fatal fault (exit 2)"
    )
    assert 'fault_reason = "oom"' in block, (
        'The is_gpu_oom_error branch must set fault_reason = "oom" (exit 3 defer)'
    )


def test_trainer_eval_skip_handler_catches_device_dispatch_oom() -> None:
    """The final-full-eval OOM-skip handler must catch the device-dispatch path too.

    A bare ``except torch.cuda.OutOfMemoryError`` (the pre-fix form) misses the
    torch 2.12 AcceleratorError sibling, so an eval-time device-dispatch OOM
    would crash an otherwise-successful run instead of gracefully skipping the
    eval (the handler's documented intent: "Training results are still valid").
    It must catch the RuntimeError family and gate on ``is_gpu_oom_error`` so any
    OOM path skips the eval while a non-OOM RuntimeError still propagates. Static
    + mutation-safe: revert to the bare OOM except, or drop the gate, → fail.
    """
    text = TRAINER_PY.read_text()
    needle = "Final full eval skipped: OOM on 12GB GPU."
    lines = text.splitlines()
    try:
        target = next(i for i, ln in enumerate(lines) if needle in ln)
    except StopIteration:
        raise AssertionError(
            "eval-skip OOM handler with its canonical log line must remain"
        )
    # Nearest preceding line that IS an except clause (lstrip'd). A substring
    # rfind would match the word "except" inside the handler's own comment
    # (``except torch.cuda.OutOfMemoryError``), so scan whole lines instead.
    try:
        except_no = next(
            i for i in range(target - 1, -1, -1) if lines[i].lstrip().startswith("except ")
        )
    except StopIteration:
        raise AssertionError("an `except` clause must own the eval-skip log line")
    except_line = lines[except_no]
    assert "RuntimeError" in except_line, (
        "the eval-skip OOM handler must catch the RuntimeError family (so the "
        "torch.AcceleratorError device-dispatch OOM sibling is caught), not only "
        "torch.cuda.OutOfMemoryError"
    )
    span = "\n".join(lines[except_no:target])
    assert "is_gpu_oom_error" in span, (
        "the eval-skip handler must gate on is_gpu_oom_error so a device-dispatch "
        "OOM skips the eval while a non-OOM RuntimeError still propagates"
    )
