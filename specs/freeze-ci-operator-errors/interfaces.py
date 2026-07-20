# freeze-ci-operator-errors 型定義
#
# 作成日: 2026-07-20
# 関連設計: architecture.md
# 関連要件: requirements.md
# 関連データフロー: dataflow.md
#
# 信頼性レベル:
# - 🔵 青信号: 既存 leaf module パターン + 既存 test pattern から導出した型定義
# - 🟡 黄信号: 既存 `--json` mode pattern からの妥当な推測による型定義
# - 🔴 赤信号: 将来拡張（REQ-301a）の推測による型定義
#
# NOTE: This is a **spec / design** type-definition file, not a runnable module.
# The actual implementation lives at `src/utils/cli_errors.py` and is produced
# by the TASK-0179 commit that follows this spec. The Protocol / TypedDict
# shapes here are the contract the implementation must satisfy.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, Protocol, runtime_checkable

# Forward references for the exception types we WRAP but do not import.
# Runtime: leaf does NOT import yaml/pydantic (zero-dep leaf pattern).
# Type-check only: mypy/pyright can resolve the wrapper signatures.
if TYPE_CHECKING:
    import yaml
    from pydantic import ValidationError

# ========================================
# Module-level constants
# ========================================

#: sysexits.h `EX_CONFIG` (BSD/Linux) — configuration file error.
#: Used as the exit code for ALL four OperatorError subtypes.
#: Distinct from argparse error 2, worker EXIT_* (0/1/2/3/75), and
#: the launcher's existing 4-way exit-code contract (`ad8c84a`).
EXIT_OPERATOR_ERROR: int = 78  # 🔵 REQ-003, REQ-705

# ========================================
# OperatorError hierarchy
# ========================================


class OperatorError(Exception):
    """Base class for all operator-input failures.

    🔵 REQ-001, REQ-002, NFR-301

    Subclasses represent the four distinct operator-error axes the
    freeze-ci-9b entrypoint family must surface (AI_HUB_MAKE_RUN_FEEDBACK
    "operator-facing follow-up: implement distinct handling for missing
    config, malformed YAML, AppConfig validation failures, and malformed
    eval results"). The hierarchy mirrors the `atomic_save.py` /
    `checkpoint_integrity.py` leaf pattern — one file, one responsibility,
    imported by all three entrypoints.

    The leaf itself is torch-free / pydantic-free / omegaconf-free:
    ``yaml.YAMLError`` and ``pydantic.ValidationError`` are wrapped, not
    subclassed, so the leaf can be imported in any test environment
    without dragging the heavy dependencies along.
    """

    #: Default exit status for all subclasses. 🔵 REQ-003
    default_exit_status: int = EXIT_OPERATOR_ERROR

    def __init__(self, detail: str, *, exit_status: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        # REQ-301a 🔴: per-instance exit_status override (future hook).
        # The TASK-0179 implementation should accept this without using it.
        self.exit_status: int = (
            EXIT_OPERATOR_ERROR if exit_status is None else exit_status
        )

    def __str__(self) -> str:
        """Human-readable form: ``"<ClassName>: <detail>"``.

        🔵 REQ-002, NFR-202
        """
        return f"{type(self).__name__}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable form for ``--json`` mode.

        🔵 REQ-002, REQ-501, EDGE-102
        Returns ``{"error": <class name>, "detail": <str>, "exit_status": 78}``.
        Callers must serialize via ``json.dumps(...)`` (no indent), so the
        output is a single line — the contract pinned by ``EDGE-102``.
        """
        return {
            "error": type(self).__name__,
            "detail": self.detail,
            "exit_status": self.exit_status,
        }


class MissingConfigError(OperatorError):
    """Class 1: ``--config`` or ``--samples-file`` path does not exist.

    🔵 REQ-101, REQ-102, EDGE-001
    Wraps ``FileNotFoundError`` (and ``IsADirectoryError`` for
    ``--config <directory>``).
    """


class MalformedYAMLError(OperatorError):
    """Class 2: YAML parse failure on an existing ``--config`` file.

    🔵 REQ-201, REQ-202, EDGE-002
    Wraps ``yaml.YAMLError``. The PyYAML ``__str__`` (line/column info)
    is preserved verbatim in :attr:`detail`.
    """


class AppConfigValidationError(OperatorError):
    """Class 3: Pydantic schema violation on parsed YAML.

    🔵 REQ-301, REQ-302, REQ-303
    Wraps ``pydantic.ValidationError``. :attr:`detail` carries the
    ``config_class`` name, the error count, and the first error's
    ``loc / msg / type`` triple — enough for an operator to fix the
    config in one pass (NFR-201).
    """


class MalformedEvalResultsError(OperatorError):
    """Class 4: ``<samples_file>`` JSON parse failure or schema violation.

    🔵 REQ-401, REQ-402, EDGE-003, EDGE-004
    Replay-script specific. Distinguishes the four sub-failure modes
    (file not found → ``MissingConfigError``; JSON parse / missing key /
    type mismatch → this class).
    """


# ========================================
# 4 wrapper functions (raise-only)
# ========================================


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

    ``kind`` distinguishes ``--config`` (default) from ``--samples-file``
    in the emitted message: the replay script passes
    ``kind="samples file"``. If ``path`` is an existing directory
    (EDGE-001), the message is suffixed with ``(is a directory)``.
    """
    p = str(path)
    suffix = ""
    try:
        from pathlib import Path  # local import keeps leaf zero-dep
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

    🔵 REQ-201, REQ-202, EDGE-002
    The PyYAML ``__str__`` (line/column info) is preserved verbatim in
    the :class:`MalformedYAMLError` ``detail`` field.
    """
    raise MalformedYAMLError(f"yaml parse error in {path}: {exc}")


def raise_app_config_validation(
    config_class: str,
    exc: "ValidationError",
) -> NoReturn:
    """Build and raise :class:`AppConfigValidationError` from a Pydantic error.

    🔵 REQ-301, REQ-302, REQ-303
    ``config_class`` is the Pydantic model name (``"TGLoRAConfig"`` /
    ``"BaselineConfig"`` / ...). The :class:`pydantic.ValidationError`
    ``.errors()`` list is summarized as ``<N> errors; first: <loc>
    <msg> (<type>)`` — the ``first`` slot is always present even when
    there is only one error, so the operator sees a single, well-shaped
    line.
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

    * ``"json parse error"`` (with ``detail`` carrying the JSON error)
    * ``"missing key: <key>"``
    * ``"invalid type for <field>: expected <type>, got <actual>"``
    """
    if detail:
        raise MalformedEvalResultsError(f"{reason}: {detail}")
    raise MalformedEvalResultsError(reason)


# ========================================
# Output renderer (emitter)
# ========================================


@runtime_checkable
class _SupportsWriteText(Protocol):
    """Minimal stream protocol: ``write(str)`` + flush.

    🔵 Replaces explicit ``sys.stderr`` / ``sys.stdout`` references so
    the emitter is unit-testable with ``io.StringIO``.
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
    ``"<ClassName>: <detail>\\n"`` to ``stderr`` (or the supplied
    ``stderr`` stream for tests).

    In JSON mode (``json_mode=True``), writes
    ``{"error": "<ClassName>", "detail": "<detail>", "exit_status": 78}\\n``
    to ``stdout`` (or the supplied ``stdout`` stream) and leaves
    ``stderr`` empty. The output is a single line: callers MUST use
    ``json.dumps(...)`` (no ``indent=2``) so ``"\\n" not in payload``
    except the trailing newline — pinned by ``EDGE-102``.

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


# ========================================
# Entrypoint-side helpers (optional, recommended)
# ========================================


def install_outer_try_except(
    json_mode: bool,
) -> None:
    """Install a process-level ``sys.excepthook`` that maps OperatorError → exit 78.

    🟡 Optional convenience for entrypoints that prefer declarative
    installation over wrapping ``main()`` in ``try/except``. The TASK-0179
    implementation may use either pattern; this is the spec for the
    hook itself.

    🔵 REQ-003 + leaf pattern.
    """
    import sys as _sys

    def _hook(exc_type: type, exc_value: BaseException, _tb: Any) -> None:
        if isinstance(exc_value, OperatorError):
            emit_operator_error(exc_value, json_mode=json_mode)
            _sys.exit(EXIT_OPERATOR_ERROR)
        # Defer to default handler for non-OperatorError.
        _sys.excepthook(exc_type, exc_value, _tb)

    _sys.excepthook = _hook


# ========================================
# Launcher extension (worker-exit classification)
# ========================================


#: Suffix added to ``scripts/launch_freeze_ci_9b_full.py::classify_exit_code``
#: to map the new 78 → ``FATAL`` ("operator_error") branch.
#:
#: 🔵 interview-record.md A6 + ad8c84a worker contract
#:
#: Pseudocode for the launcher-side extension (NOT a function to call —
#: the launcher is torch-free by design and inlines its own constants):
#:
#:     if code == EXIT_OPERATOR_ERROR:  # 78
#:         return Decision(Action.FATAL, "operator_error", 0.0)
#:
#: This is added BEFORE the existing ``code < 0`` (signal-kill) branch
#: in ``classify_exit_code`` so the 78 path takes precedence over
#: the signal-kill RETRY path. Pin: tests/test_worker_launcher_exit_contract.py.
LAUNCHER_OPERATOR_ERROR_BRANCH: str = (
    "if code == EXIT_OPERATOR_ERROR: return Decision(Action.FATAL, 'operator_error', 0.0)"
)


# ========================================
# 信頼性レベルサマリー
# ========================================
#
# - 🔵 青信号: 14件 (87.5%)
# - 🟡 黄信号: 1件 (6.25%)  — `install_outer_try_except` convenience
# - 🔴 赤信号: 1件 (6.25%)  — REQ-301a `exit_status` override hook
#
# 品質評価: 高品質
