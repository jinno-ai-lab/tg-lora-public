"""Operator-facing distinct error classes for the freeze-ci-9b entrypoint family.

Sibling leaf of :mod:`src.utils.atomic_save` (the atomic-write leaf) and
:mod:`src.utils.checkpoint_integrity` (the load-diagnosis leaf). Those leaves
closed the *internal* silent-corruption chokepoints — a mid-commit fault could
publish a torn destination, a torn checkpoint could crash resume with an opaque
``EOFError``. This leaf closes the **operator-input** chokepoint: the three
freeze-ci-9b entrypoints
(``scripts/replay_freeze_validloss_ci.py`` /
``scripts/run_freeze_validloss_ci_9b.py`` /
``scripts/launch_freeze_ci_9b_full.py``) accept four *distinct* classes of
operator error (missing config, malformed YAML, AppConfig validation failure,
malformed eval results) that previously surfaced as an undifferentiated mix of
``FileNotFoundError`` / ``yaml.YAMLError`` / ``pydantic.ValidationError`` /
``json.JSONDecodeError`` tracebacks — so an operator staring at a CI log could
not tell in one read *which* input was wrong. Each class now raises its own
:class:`OperatorError` subtype with a single, well-shaped, grep-friendly
message and the common exit code :data:`EXIT_OPERATOR_ERROR` (78 =
``sysexits.h`` ``EX_CONFIG``), emitted via :func:`emit_operator_error`.

This is deliberately a **zero-dependency leaf** (stdlib only). ``yaml`` /
``pydantic`` / ``omegaconf`` are referenced under ``TYPE_CHECKING`` only, so the
4 subtypes *wrap* (never subclass) the third-party exceptions: importing this
module drags no heavy dependency along and has no import side effects — the same
property that factored the atomic-save / checkpoint-integrity leaves out of the
heavyweight training-resume stack. The three entrypoints import the wrappers
and a thin outer ``try/except``; nothing else changes.

See ``specs/freeze-ci-operator-errors/`` (requirements / architecture /
interfaces) for the frozen contract this leaf implements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, Protocol, runtime_checkable

if TYPE_CHECKING:
    import yaml
    from pydantic import ValidationError

__all__ = [
    "EXIT_OPERATOR_ERROR",
    "OperatorError",
    "MissingConfigError",
    "MalformedYAMLError",
    "AppConfigValidationError",
    "MalformedEvalResultsError",
    "raise_missing_config",
    "raise_malformed_yaml",
    "raise_app_config_validation",
    "raise_malformed_eval_results",
    "emit_operator_error",
]

#: ``sysexits.h`` ``EX_CONFIG`` (BSD/Linux) — configuration file error.
#:
#: The single exit status for ALL four :class:`OperatorError` subtypes. It is
#: deliberately distinct from argparse's usage-error exit 2, the worker
#: contract's ``EXIT_DONE/UNEXPECTED/CUDA_DOWN/INCOMPLETE_RESUME`` (0/1/2/3)
#: and ``EXIT_GPU_TEMPFAIL`` (75) pinned in ``ad8c84a``, so an operator (or the
#: launcher's ``classify_exit_code``) can tell a configuration problem from a
#: transient host / GPU event by the code alone. 🔵 REQ-003, REQ-705
EXIT_OPERATOR_ERROR: int = 78


# ===========================================================================
# OperatorError hierarchy
# ===========================================================================


class OperatorError(Exception):
    """Base class for all operator-input failures.

    🔵 REQ-001, REQ-002, NFR-301

    Subclasses represent the four distinct operator-error axes the freeze-ci-9b
    entrypoint family must surface (AI_HUB_MAKE_RUN_FEEDBACK "operator-facing
    follow-up: implement distinct handling for missing config, malformed YAML,
    AppConfig validation failures, and malformed eval results"). The hierarchy
    mirrors the :mod:`src.utils.atomic_save` / :mod:`src.utils.checkpoint_integrity`
    leaf pattern — one file, one responsibility, imported by all three
    entrypoints.

    The leaf itself is torch-free / pydantic-free / omegaconf-free:
    ``yaml.YAMLError`` and ``pydantic.ValidationError`` are wrapped, not
    subclassed, so the leaf imports in any test environment without dragging the
    heavy dependencies along.
    """

    #: Default exit status for all subclasses. 🔵 REQ-003
    default_exit_status: int = EXIT_OPERATOR_ERROR

    def __init__(self, detail: str, *, exit_status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        # REQ-301a: per-instance exit_status override (future hook). Today every
        # subtype exits 78; the hook is accepted so a later per-class code lands
        # without touching call sites.
        self.exit_status: int = (
            EXIT_OPERATOR_ERROR if exit_status is None else exit_status
        )

    def __str__(self) -> str:
        """Human-readable form: ``"<ClassName>: <detail>"``.

        🔵 REQ-002, NFR-202 — the class-name prefix makes a CI log greppable
        (``grep '^MissingConfigError:'``).
        """
        return f"{type(self).__name__}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable form for ``--json`` mode.

        🔵 REQ-002, REQ-501, EDGE-102 — returns
        ``{"error": <class name>, "detail": <str>, "exit_status": 78}``. Callers
        serialize via ``json.dumps(...)`` (no indent) so the output is one line
        with no embedded newline except the trailing one.
        """
        return {
            "error": type(self).__name__,
            "detail": self.detail,
            "exit_status": self.exit_status,
        }


class MissingConfigError(OperatorError):
    """Class 1: ``--config`` or ``--samples-file`` path does not exist.

    🔵 REQ-101, REQ-102, EDGE-001 — wraps ``FileNotFoundError`` (and an
    ``IsADirectoryError`` for ``--config <directory>``).
    """


class MalformedYAMLError(OperatorError):
    """Class 2: YAML parse failure on an existing ``--config`` file.

    🔵 REQ-201, REQ-202, EDGE-002 — wraps ``yaml.YAMLError``. The PyYAML
    ``__str__`` (line/column info) is preserved verbatim in :attr:`detail`.
    """


class AppConfigValidationError(OperatorError):
    """Class 3: Pydantic schema violation on a parsed config.

    🔵 REQ-301, REQ-302, REQ-303 — wraps ``pydantic.ValidationError``.
    :attr:`detail` carries the config-class name, the error count, and the first
    error's ``loc / msg / type`` triple — enough for an operator to fix the
    config in one pass (NFR-201).
    """


class MalformedEvalResultsError(OperatorError):
    """Class 4: ``<samples_file>`` JSON parse failure or schema violation.

    🔵 REQ-401, REQ-402, EDGE-003, EDGE-004 — replay-script specific. A missing
    samples file is :class:`MissingConfigError`; a JSON parse failure, a missing
    required key, or a type mismatch is this class.
    """


# ===========================================================================
# 4 wrapper functions (raise-only)
# ===========================================================================


class _PathLike(Protocol):
    """Anything ``str(path)`` or ``Path(path)`` can produce.

    🔵 Existing pattern (``Path | str``) across the 3 entrypoints.
    """

    def __fspath__(self) -> str: ...


def raise_missing_config(
    path: _PathLike,
    *,
    kind: str = "config",
) -> NoReturn:
    """Build and raise :class:`MissingConfigError` for a missing path.

    🔵 REQ-101, REQ-102, EDGE-001

    ``kind`` distinguishes ``--config`` (default) from ``--samples-file`` in the
    emitted message: the replay script passes ``kind="samples file"``. If
    ``path`` is an existing directory (EDGE-001), the message is suffixed with
    ``(is a directory)``.
    """
    p = str(path)
    suffix = ""
    try:
        from pathlib import Path  # local import keeps the leaf zero-dep

        if Path(p).is_dir():
            suffix = " (is a directory)"
    except OSError:
        # Broken symlink / permission error: treat as missing.
        pass
    raise MissingConfigError(f"{kind} not found: {p}{suffix}")


def raise_malformed_yaml(
    path: _PathLike,
    exc: "yaml.YAMLError",
) -> NoReturn:
    """Build and raise :class:`MalformedYAMLError` from a PyYAML error.

    🔵 REQ-201, REQ-202, EDGE-002 — the PyYAML ``__str__`` (line/column info) is
    preserved verbatim in the :class:`MalformedYAMLError` ``detail`` field.
    """
    raise MalformedYAMLError(f"yaml parse error in {path}: {exc}")


def raise_app_config_validation(
    config_class: str,
    exc: "ValidationError",
) -> NoReturn:
    """Build and raise :class:`AppConfigValidationError` from a Pydantic error.

    🔵 REQ-301, REQ-302, REQ-303

    ``config_class`` is the Pydantic model name (``"TGLoRAConfig"`` /
    ``"BaselineConfig"`` / ...). The :class:`pydantic.ValidationError` ``.errors()``
    list is summarized as ``<N> errors; first: <loc> <msg> (<type>)`` — the
    ``first`` slot is always present even when there is only one error, so the
    operator sees a single, well-shaped line, and ``<N>`` matches
    ``len(exc.errors())`` (REQ-302).
    """
    errors = exc.errors()
    n = len(errors)
    first = errors[0] if errors else {"loc": (), "msg": "unknown", "type": "unknown"}
    loc = ".".join(str(x) for x in first.get("loc", ())) or "<root>"
    msg = first.get("msg", "unknown")
    typ = first.get("type", "unknown")
    raise AppConfigValidationError(
        f"schema validation failed for {config_class}: {n} errors; "
        f"first: {loc} {msg} ({typ})"
    )


def raise_malformed_eval_results(
    reason: str,
    detail: str = "",
) -> NoReturn:
    """Build and raise :class:`MalformedEvalResultsError` for a samples-file failure.

    🔵 REQ-401, REQ-402, EDGE-003, EDGE-004

    ``reason`` is one of:
      * ``"json parse error"`` (with ``detail`` carrying the JSON error),
      * ``"missing key: <key>"``,
      * ``"invalid type for <field>: expected <type>, got <actual>"``.
    """
    if detail:
        raise MalformedEvalResultsError(f"{reason}: {detail}")
    raise MalformedEvalResultsError(reason)


# ===========================================================================
# Output renderer (emitter)
# ===========================================================================


@runtime_checkable
class _SupportsWriteText(Protocol):
    """Minimal stream protocol: ``write(str)`` + ``flush``.

    🔵 Replaces explicit ``sys.stderr`` / ``sys.stdout`` references so the
    emitter is unit-testable with ``io.StringIO``.
    """

    def write(self, s: str, /) -> int: ...

    def flush(self) -> None: ...


def emit_operator_error(
    exc: OperatorError,
    *,
    json_mode: bool = False,
    stdout: _SupportsWriteText | None = None,
    stderr: _SupportsWriteText | None = None,
) -> None:
    """Render an OperatorError to the right stream given the caller's mode.

    🔵 REQ-003, REQ-501, REQ-502, NFR-202, EDGE-102, EDGE-103

    In human mode (``json_mode=False``, default), writes
    ``"<ClassName>: <detail>\\n"`` to ``stderr`` (or the supplied ``stderr``
    stream for tests).

    In JSON mode (``json_mode=True``), writes
    ``{"error": "<ClassName>", "detail": "<detail>", "exit_status": 78}\\n`` to
    ``stdout`` (or the supplied ``stdout`` stream) and leaves ``stderr`` empty.
    The output is a single line: callers MUST use ``json.dumps(...)`` (no
    ``indent=2``) so ``"\\n" not in payload`` except the trailing newline —
    pinned by EDGE-102.

    ANSI color codes are never emitted (NFR-203, EDGE-103).
    """
    import json
    import sys as _sys

    out = stdout if stdout is not None else _sys.stdout
    err = stderr if stderr is not None else _sys.stderr
    if json_mode:
        out.write(json.dumps(exc.to_dict()) + "\n")
        out.flush()
    else:
        err.write(str(exc) + "\n")
        err.flush()
