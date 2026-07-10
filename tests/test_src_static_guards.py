"""Static guard: the whole ``src/`` tree must be ``ruff``-clean (zero findings).

Extends the per-file pin in ``test_train_tg_lora_static_guards.py`` (which locks
``src/training/train_tg_lora.py`` to zero findings) to the entire ``src/`` tree.
The two guards share one rationale: the audit long described lint debt as
"pre-existing, not a regression" prose, and prose drifts. Commit a4d7c26 made
that drift measurable by pinning the training entry point; this test pins the
rest of the source tree the same way so a future ambiguous-name comprehension
variable, unused import, or re-introduced dead local fails CI here instead of
re-stating a stale lint count in PURPOSE.md.

The cleanup this locks in removed 11 ``src/`` findings, all behavior-preserving:
two unused imports (F401), three dead locals (F841 — computed-but-never-read,
the same dead-state class a4d7c26 removed from the trainer), and six ambiguous
``l`` loop variables (E741). It also surfaced and fixed a real latent
``NameError`` in ``scripts/`` (``incregments`` typo, F821) — covered by
``tests/test_analyze_trajectory_deltas.py`` — which is exactly the defect class
a clean ``src/`` tree keeps out of the core path.

Scope is ``src/`` only: ``scripts/`` and ``tests/`` still carry documented
pre-existing lint debt (``make lint`` checks all three and remains red on those
two). Pinning ``src/`` first keeps the highest-value slice — the imported core
— regression-proof without a high-churn bulk cleanup of the lower-priority
slices. Runs ``ruff`` via subprocess and skips when ``ruff`` is absent.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "src"


def _run_ruff(
    target: Path, select: Sequence[str] = ()
) -> subprocess.CompletedProcess[str]:
    ruff = shutil.which("ruff")
    cmd: list[str] = [ruff, "check"]
    if select:
        # ruff's default rule set is E4/E7/E9/F — it deliberately EXCLUDES the
        # pycodestyle W rules. ``select`` opts a specific rule back in so a guard
        # can police a defect class the default-clean check is blind to (W605).
        cmd += ["--select", ",".join(select)]
    cmd.append(str(target))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_src_tree_is_ruff_clean() -> None:
    """``ruff check src/`` must report zero findings across the default rules.

    Guards the ``src/`` lint-clean invariant: a non-zero count regresses it
    (re-introduced dead local, unused import, ambiguous name) and makes the
    audit's standing claim false. See the module docstring for the cleanup this
    locks in and the deliberate ``src/``-only scope.

    Scope note: this runs ruff's *default* rule set (E4/E7/E9/F), which excludes
    the pycodestyle ``W`` rules — so it does NOT catch invalid escape sequences
    (W605). Those are policed separately by
    :func:`test_src_tree_has_no_invalid_escape_sequences`, because a ``\\*`` in a
    docstring passes this default check while Python emits a ``DeprecationWarning``
    on every import (a ``SyntaxWarning`` from 3.12).
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the cleanliness guard")

    assert TARGET.is_dir(), f"src/ tree not found at {TARGET}"
    proc = _run_ruff(TARGET)
    assert proc.returncode == 0, (
        "src/ must be ruff-clean across the default rule set (E4/E7/E9/F) — a "
        "non-zero count regresses the src/ lint-clean invariant and makes the "
        "audit's standing claim false. ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )


def test_src_tree_has_no_invalid_escape_sequences() -> None:
    """No ``src/`` file may contain an invalid escape sequence (ruff W605).

    Companion to :func:`test_src_tree_is_ruff_clean`. That test's default rule
    set excludes W605, so a ``\\*`` / ``\\d`` / ``\\m`` in a docstring passes it
    silently while Python emits a ``DeprecationWarning`` on every import (a hard
    ``SyntaxWarning`` from 3.12). One such site did exactly that for months — a
    ``L\\*`` in ``run_metrics.record_full_eval_loss``'s docstring rendered as the
    literal ``L\\*`` (not the intended ``L*``) and fired the warning on every
    test run, all while the default-clean guard stayed green. This test selects
    W605 explicitly to close that blind spot and keep the escape-sequence class
    out of the imported core for good.
    """
    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH; cannot enforce the escape-sequence guard")

    assert TARGET.is_dir(), f"src/ tree not found at {TARGET}"
    proc = _run_ruff(TARGET, select=("W605",))
    assert proc.returncode == 0, (
        "src/ must contain no invalid escape sequences (W605) — each one is a "
        "DeprecationWarning now and a SyntaxWarning from Python 3.12. "
        "ruff output:\n" + (proc.stdout + proc.stderr).strip()
    )


def test_no_bare_torch_save_in_src() -> None:
    """The only ``torch.save`` call in ``src/`` lives in the atomic-save helper.

    A bare ``torch.save(blob, path)`` truncates the destination before writing,
    so an interruption mid-dump (OOM kill, SIGINT during a multi-hundred-MB
    write) leaves a torn file that breaks the next load — losing a run
    (``training_state.pt``), corrupting a large analysis artifact
    (``trajectory_delta_artifacts/*.pt``), or forcing an expensive rebuild of a
    multi-GB prefix-feature cache. Every on-disk artifact that must survive an
    interruption routes through :func:`src.utils.atomic_save._atomic_torch_save`
    (PID-suffixed temp + ``os.replace``) instead, so a load never sees a partial
    file. ``save_pretrained`` (the LoRA/tokenizer checkpoint path) is unaffected
    — it writes via the safetensors library, not ``torch.save``.

    This guard pins that discipline with an AST scan: any ``torch.save(...)``
    call outside :mod:`src.utils.atomic_save` reintroduces the mid-save
    truncation hazard the resume-state persistence axis exists to prevent. AST
    (not a text grep) so docstring and code-comment mentions of ``torch.save``
    do not false-positive — only real call expressions are counted.
    """
    atomic_leaf = TARGET / "utils" / "atomic_save.py"
    assert atomic_leaf.is_file(), (
        f"atomic-save helper missing at {atomic_leaf}; the single torch.save "
        "publish point that every on-disk artifact routes through must exist"
    )

    offenders: list[tuple[Path, int]] = []
    for src_file in sorted(TARGET.rglob("*.py")):
        tree = ast.parse(
            src_file.read_text(encoding="utf-8"), filename=str(src_file)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                offenders.append((src_file, node.lineno))

    off_src = [(f, ln) for f, ln in offenders if f != atomic_leaf]
    assert not off_src, (
        "Every on-disk artifact in src/ must persist via "
        "src.utils.atomic_save._atomic_torch_save — a bare "
        "torch.save(<path>) outside that helper reintroduces the mid-save "
        "truncation hazard (torn checkpoint / torn trajectory artifact / torn "
        "prefix-feature cache) the resume-state axis exists to prevent. "
        "Offending sites:\n" + "\n".join(f"  {f}:{ln}" for f, ln in off_src)
    )


def _call_name(func: ast.expr) -> str | None:
    """Reduce a Call's func node to a bare name (``load_training_state`` or
    ``model.foo`` → ``foo``), or ``None`` for non-name call targets."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_resume_path_wires_integrity_checked_loaders() -> None:
    """The (shifted) ``train_tg_lora`` resume seam must route both checkpoint
    loads through the integrity-checked helpers, load-before-apply, unwrapped.

    Sibling philosophy to :func:`test_no_bare_torch_save_in_src`: the load-side
    integrity helpers (:func:`~src.utils.checkpoint.load_training_state` and the
    adapter restore via :func:`~src.training.train_tg_lora._restore_adapter_weights`)
    only protect resume if the resume seam actually CALLS them, in order, and
    does NOT swallow the :class:`~src.utils.checkpoint.CheckpointIntegrityError`
    they raise on a torn checkpoint. The resume block physically moved when
    ``train_tg_lora`` shifted out of ``scripts/`` into ``src/training/`` and
    again as the file grew (the steering feedback flagged the line-number drift),
    so a future edit could silently reroute resume around the helpers —
    re-introducing the opaque ``EOFError`` / ``SafetensorError`` crash on a torn
    checkpoint — or wrap them in a broad ``except`` that swallows the fail-loud
    guarantee into a silent restart (a GOAL.md honesty break: it hides lost
    progress). This AST guard pins the wiring against that drift:

      * the ``_restore_adapter_weights`` helper calls ``load_adapter_weights``
        (the integrity gate) BEFORE it applies — so a torn adapter raises with
        zero model mutation;
      * the ``if resume_path is not None:`` seam calls BOTH ``load_training_state``
        and ``_restore_adapter_weights``; and
      * neither critical load is enclosed by any ``try``/``except`` — an outer
        try around the whole seam OR an inner try around just the one load call
        could swallow a torn-checkpoint ``CheckpointIntegrityError`` into a
        silent restart.

    AST-parse of the source (not an import): ``src.training.train_tg_lora`` is
    un-importable on the public mirror (private ``src.data`` + absent ``peft``),
    so the guard mirrors :func:`test_no_bare_torch_save_in_src`'s source-parse
    approach.
    """
    trainer = TARGET / "training" / "train_tg_lora.py"
    assert trainer.is_file(), f"trainer entrypoint missing at {trainer}"
    tree = ast.parse(trainer.read_text(encoding="utf-8"), filename=str(trainer))

    # Parent map so the no-swallowing check can climb ancestors of the seam.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.tlsparent = parent  # type: ignore[attr-defined]

    module_funcs = {
        n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    assert "train_tg_lora" in module_funcs, "train_tg_lora entrypoint missing"
    assert "_restore_adapter_weights" in module_funcs, (
        "_restore_adapter_weights helper missing — the resume adapter-restore "
        "seam (load_adapter_weights before apply) must be a named, importable "
        "helper pinned by tests/test_resume_adapter_integrity.py, not an "
        "untested inline statement ordering inside the loop"
    )

    # (1) The helper itself must load BEFORE it applies.
    helper = module_funcs["_restore_adapter_weights"]
    helper_calls = [
        (_call_name(c.func), c.lineno)
        for c in ast.walk(helper)
        if isinstance(c, ast.Call)
    ]
    load_linenos = [ln for name, ln in helper_calls if name == "load_adapter_weights"]
    apply_linenos = [
        ln
        for name, ln in helper_calls
        if name in ("apply_state", "set_peft_model_state_dict")
    ]
    assert load_linenos and apply_linenos, (
        "_restore_adapter_weights must call both load_adapter_weights (the "
        "integrity gate) and apply_state (the model mutation)"
    )
    assert min(load_linenos) < min(apply_linenos), (
        "_restore_adapter_weights must call load_adapter_weights BEFORE "
        "apply_state — reordering lets a torn adapter_model.safetensors "
        "half-apply the model instead of raising CheckpointIntegrityError"
    )

    # (2) Locate the resume seam inside train_tg_lora (if resume_path is not None:).
    fn = module_funcs["train_tg_lora"]
    resume_if: ast.If | None = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "resume_path"
            and node.test.ops
            and isinstance(node.test.ops[0], ast.IsNot)
            and node.test.comparators
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        ):
            resume_if = node
            break
    assert resume_if is not None, (
        "resume seam (if resume_path is not None:) missing from train_tg_lora — "
        "the fault/periodic resume path must exist"
    )

    seam_calls = {
        _call_name(c.func)
        for c in ast.walk(resume_if)
        if isinstance(c, ast.Call)
    }
    assert "load_training_state" in seam_calls, (
        "resume seam must route training_state.pt through load_training_state "
        "(the torn-checkpoint → CheckpointIntegrityError gate), not a raw "
        "torch.load"
    )
    assert "_restore_adapter_weights" in seam_calls, (
        "resume seam must restore the adapter via _restore_adapter_weights (the "
        "load-before-apply integrity gate), not an inline safetensors path"
    )

    # (3) Neither critical load may be enclosed by a try/except — an outer try
    #     around the whole seam OR an inner try around just the one load call
    #     could swallow a torn-checkpoint CheckpointIntegrityError into a silent
    #     restart (the GOAL.md honesty break). Climb EACH critical call's
    #     ancestors to the function root; any Try on the path is a swallower.
    critical_calls = [
        c
        for c in ast.walk(resume_if)
        if isinstance(c, ast.Call)
        and _call_name(c.func) in ("load_training_state", "_restore_adapter_weights")
    ]
    assert critical_calls, (
        "resume seam must contain the load_training_state and "
        "_restore_adapter_weights calls checked above"
    )
    for call_node in critical_calls:
        swallower: ast.AST | None = None
        cur: ast.AST | None = getattr(call_node, "tlsparent", None)
        while cur is not None and cur is not fn:
            if isinstance(cur, ast.Try):
                swallower = cur
                break
            cur = getattr(cur, "tlsparent", None)
        assert swallower is None, (
            f"resume critical load {_call_name(call_node.func)}(...) at line "
            f"{call_node.lineno} must NOT be enclosed in a try/except — a "
            "handler there could swallow a torn-checkpoint "
            "CheckpointIntegrityError into a silent restart, hiding lost "
            f"progress. Found enclosing try at line {swallower.lineno}"
        )


def test_baseline_resume_path_wires_integrity_checked_loaders() -> None:
    """The ``train_baseline`` resume seam must route BOTH checkpoint loads
    through the integrity-checked helpers, load-before-apply, unwrapped.

    Sibling of :func:`test_resume_path_wires_integrity_checked_loaders` for the
    OTHER training entrypoint (``train_baseline_qlora.train_baseline``). The
    write side is already atomic for the baseline path
    (``save_baseline_training_state`` → ``_atomic_torch_save``; ``save_checkpoint``
    → the atomic directory publish), so this guard pins the symmetric LOAD side:
    that resume routes ``training_state.pt`` through ``load_baseline_training_state``
    (torn → :class:`~src.utils.checkpoint.CheckpointIntegrityError`) AND the LoRA
    adapter through ``load_adapter_weights`` (load-before-apply, torn →
    ``CheckpointIntegrityError``) — never a raw ``torch.load`` / inline
    ``safetensors.load_file`` — and that neither critical load is swallowed by a
    ``try``/``except`` that could turn a torn-checkpoint fail-loud into a silent
    restart (a GOAL.md honesty break that hides lost progress). Without this the
    baseline path is the asymmetric weak link a torn checkpoint would crash
    opaquely on, the very gap the load-side integrity axis closed for the TG-LoRA
    path. The baseline trainer is un-importable on the public mirror (absent
    ``peft`` + private ``src.data``), so — like the TG-LoRA guard — this is an
    AST source-parse, not an import.

    The baseline path has TWO ``if resume_path is not None:`` blocks (one restores
    optimizer/scheduler/scalars, one restores adapter weights + batch position),
    so this guard checks BOTH seams rather than the single seam the TG-LoRA path
    has.
    """
    trainer = TARGET / "training" / "train_baseline_qlora.py"
    assert trainer.is_file(), f"baseline trainer entrypoint missing at {trainer}"
    tree = ast.parse(trainer.read_text(encoding="utf-8"), filename=str(trainer))

    # Parent map so the no-swallowing check can climb ancestors of each seam.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.tlsparent = parent  # type: ignore[attr-defined]

    module_funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "train_baseline" in module_funcs, "train_baseline entrypoint missing"
    fn = module_funcs["train_baseline"]

    # Collect EVERY `if resume_path is not None:` seam inside train_baseline
    # (there are two — see the docstring).
    resume_seams: list[ast.If] = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "resume_path"
            and node.test.ops
            and isinstance(node.test.ops[0], ast.IsNot)
            and node.test.comparators
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        ):
            resume_seams.append(node)
    assert resume_seams, (
        "resume seam (if resume_path is not None:) missing from train_baseline "
        "— the fault/periodic resume path must exist"
    )

    seam_calls = [
        (_call_name(c.func), c.lineno)
        for seam in resume_seams
        for c in ast.walk(seam)
        if isinstance(c, ast.Call)
    ]
    seam_call_names = {name for name, _ in seam_calls}

    # (1) training_state.pt routes through the integrity-checked loader.
    assert "load_baseline_training_state" in seam_call_names, (
        "resume seam must route training_state.pt through "
        "load_baseline_training_state (the torn-checkpoint → "
        "CheckpointIntegrityError gate), not a raw torch.load"
    )

    # (2) The LoRA adapter routes through the integrity-checked loader, BEFORE
    #     the apply — so a torn adapter_model.safetensors raises with zero model
    #     mutation. Mirrors _restore_adapter_weights's load-before-apply invariant.
    load_linenos = [ln for name, ln in seam_calls if name == "load_adapter_weights"]
    apply_linenos = [ln for name, ln in seam_calls if name == "set_peft_model_state_dict"]
    assert load_linenos and apply_linenos, (
        "resume seam must restore the adapter via load_adapter_weights (the "
        "integrity gate) and set_peft_model_state_dict (the model mutation), "
        "not an inline safetensors.load_file"
    )
    assert min(load_linenos) < min(apply_linenos), (
        "resume seam must call load_adapter_weights BEFORE "
        "set_peft_model_state_dict — reordering lets a torn "
        "adapter_model.safetensors half-apply the model instead of raising "
        "CheckpointIntegrityError"
    )

    # (3) Neither critical load may be enclosed by a try/except (same rationale
    #     as the TG-LoRA guard). Climb EACH critical call's ancestors to the
    #     function root; any Try on the path is a swallower.
    critical_calls = [
        c
        for seam in resume_seams
        for c in ast.walk(seam)
        if isinstance(c, ast.Call)
        and _call_name(c.func)
        in ("load_baseline_training_state", "load_adapter_weights")
    ]
    assert critical_calls, "resume seam must contain the integrity-checked loads"
    for call_node in critical_calls:
        swallower: ast.AST | None = None
        cur: ast.AST | None = getattr(call_node, "tlsparent", None)
        while cur is not None and cur is not fn:
            if isinstance(cur, ast.Try):
                swallower = cur
                break
            cur = getattr(cur, "tlsparent", None)
        assert swallower is None, (
            f"resume critical load {_call_name(call_node.func)}(...) at line "
            f"{call_node.lineno} must NOT be enclosed in a try/except — a "
            "handler there could swallow a torn-checkpoint "
            "CheckpointIntegrityError into a silent restart, hiding lost "
            f"progress. Found enclosing try at line {swallower.lineno}"
        )


# ---------------------------------------------------------------------------
# On-disk resume-state-loss axis: RunMetrics must NOT truncate on resume.
# The per-cycle ``run_metrics.jsonl`` is the on-disk 13th accumulator the 12/12
# caller-scoped resume-state sites protect — if RunMetrics reopens it in ``"wb"``
# at resume, every pre-resume cycle record is lost. Both trainers pass
# ``append=resume_path is not None``; these guards pin that wiring. AST
# source-parsed (trainers un-importable on the public mirror).
# ---------------------------------------------------------------------------

_TRAINERS = [
    ("train_tg_lora", TARGET / "training" / "train_tg_lora.py"),
    ("train_baseline", TARGET / "training" / "train_baseline_qlora.py"),
]


def _metrics_append_kwarg(tree: ast.Module, fn_name: str) -> ast.expr | None:
    """Return the ``append=...`` value of the RunMetrics(...) call inside the
    trainer function, or ``None`` if the construction or keyword is absent."""
    module_funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    fn = module_funcs.get(fn_name)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RunMetrics"
        ):
            for kw in node.keywords:
                if kw.arg == "append":
                    return kw.value
    return None


def _is_resume_path_is_not_none(value: ast.expr) -> bool:
    """True iff ``value`` is the AST shape ``resume_path is not None``."""
    return (
        isinstance(value, ast.Compare)
        and isinstance(value.left, ast.Name)
        and value.left.id == "resume_path"
        and value.ops
        and isinstance(value.ops[0], ast.IsNot)
        and value.comparators
        and isinstance(value.comparators[0], ast.Constant)
        and value.comparators[0].value is None
    )


@pytest.mark.parametrize(
    "fn_name,trainer", _TRAINERS, ids=[t[0] for t in _TRAINERS]
)
def test_run_metrics_constructed_append_resume_aware(
    fn_name: str, trainer: Path
) -> None:
    """Each trainer must build RunMetrics with ``append=resume_path is not None``.

    Pins the on-disk resume-continuity fix: when ``train_tg_lora`` /
    ``train_baseline`` resume into the same ``run_dir`` (fault or periodic
    resume), RunMetrics opens the existing ``run_metrics.jsonl`` in append mode
    and carries ``run_id`` + wall-clock forward instead of truncating it.
    Without this, every caller-scoped accumulator the 12/12 resume-state-loss
    axis restores is silently undermined — the advisor / deposit / break-even
    gate read the metrics FILE, so they would see only post-resume records and
    the run-end summary would not reflect the full run. A future edit that drops
    the ``append=`` keyword (or wires it to a constant) reopens that truncation
    gap and fails here.
    """
    assert trainer.is_file(), f"trainer entrypoint missing at {trainer}"
    tree = ast.parse(trainer.read_text(encoding="utf-8"), filename=str(trainer))

    append_value = _metrics_append_kwarg(tree, fn_name)
    assert append_value is not None, (
        f"{trainer.name}: {fn_name}(...) must construct RunMetrics(...) with an "
        "`append=` keyword (the resume-continuity fix) — found no append= "
        "keyword on the RunMetrics call"
    )
    assert _is_resume_path_is_not_none(append_value), (
        f"{trainer.name}: RunMetrics(...) must be built with "
        "`append=resume_path is not None` — any other value (a hard-coded "
        "True/False) breaks resume continuity for one of the run paths"
    )
