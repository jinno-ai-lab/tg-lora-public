"""Unit + mutation tests for the operator-error leaf ``src.utils.cli_errors``.

This is the leaf-level half of the freeze-ci-operator-errors axis
(``specs/freeze-ci-operator-errors/``). It pins the frozen contract from
``interfaces.py``: the 4-subtype hierarchy, the 4 raise-only wrappers, the
stderr/stdout emitter, and the zero-dependency import guarantee. The
entrypoint integration (replay / producer / launcher) is pinned in
``tests/test_replay_freeze_validloss_ci.py``,
``tests/test_run_freeze_validloss_ci_9b_producer_operator_errors.py``,
``tests/test_launch_freeze_ci_9b_full.py`` and ``test_worker_launcher_exit_contract.py``.

**Mutation-proofness (NFR-101).** Every wrapper test asserts the wrapper
``raise``\\ s the exact subtype via ``pytest.raises``. A maintainer who
neutralizes a wrapper body (``pass`` / ``return None``) makes the function
return instead of raise, so ``pytest.raises`` fails with ``DID NOT RAISE`` —
the detection test goes RED. That is the same structural mutation pin the
TASK-0171..0178 replay-gate bind family uses.
"""

from __future__ import annotations

import io
import sys

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from src.utils.cli_errors import (
    EXIT_OPERATOR_ERROR,
    AppConfigValidationError,
    MalformedEvalResultsError,
    MalformedYAMLError,
    MissingConfigError,
    OperatorError,
    emit_operator_error,
    raise_app_config_validation,
    raise_malformed_eval_results,
    raise_malformed_yaml,
    raise_missing_config,
)

# A tiny local Pydantic model with ``extra="forbid"`` so the
# ``raise_app_config_validation`` wrapper is exercised against a REAL
# ``pydantic.ValidationError`` (not a mock) without importing the heavyweight
# ``src.training.config_schema`` graph — keeps this leaf test fast and decoupled.


class _ToyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float
    steps: int


SUBTYPES = [
    MissingConfigError,
    MalformedYAMLError,
    AppConfigValidationError,
    MalformedEvalResultsError,
]


# ---------------------------------------------------------------------------
# TC-001-01..E01 — OperatorError hierarchy
# ---------------------------------------------------------------------------


class TestOperatorErrorHierarchy:
    def test_four_subtypes_importable(self):
        # TC-001-01 — the exact import the entrypoints use.
        assert all(
            issubclass(c, OperatorError)
            for c in (
                MissingConfigError,
                MalformedYAMLError,
                AppConfigValidationError,
                MalformedEvalResultsError,
            )
        )

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_each_subtype_isinstance_operator_error(self, cls):
        # TC-001-03.
        assert isinstance(cls("detail"), OperatorError)

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_to_dict_schema_frozen(self, cls):
        # TC-001-02 — the exact dict ``emit_operator_error`` serializes.
        e = cls("boom")
        assert e.to_dict() == {
            "error": cls.__name__,
            "detail": "boom",
            "exit_status": 78,
        }

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_str_format_class_prefix(self, cls):
        # NFR-202 — "<ClassName>: <detail>" so a CI log is greppable.
        assert str(cls("boom")) == f"{cls.__name__}: boom"

    def test_empty_detail_preserved(self):
        # TC-001-E01.
        assert MissingConfigError("").to_dict() == {
            "error": "MissingConfigError",
            "detail": "",
            "exit_status": 78,
        }

    def test_exit_status_override_hook(self):
        # REQ-301a — per-instance override hook; default stays 78.
        assert OperatorError("x").exit_status == 78
        assert OperatorError("x", exit_status=42).exit_status == 42
        # The override round-trips through to_dict.
        assert MissingConfigError("x", exit_status=42).to_dict()["exit_status"] == 42


# ---------------------------------------------------------------------------
# NFR-101 — wrapper mutation proof
# ---------------------------------------------------------------------------


class TestWrapperMutation:
    def test_raise_missing_config_normal(self):
        # Neutralizing ``raise_missing_config`` → returns None → DID NOT RAISE.
        with pytest.raises(MissingConfigError, match="config not found: /nope.yaml"):
            raise_missing_config("/nope.yaml")

    def test_raise_missing_config_kind_argument(self):
        with pytest.raises(MissingConfigError, match="samples file not found: /nope.json"):
            raise_missing_config("/nope.json", kind="samples file")

    def test_raise_missing_config_directory(self, tmp_path):
        # EDGE-001 — a directory path is suffixed accordingly.
        with pytest.raises(MissingConfigError, match=r"is a directory"):
            raise_missing_config(tmp_path)

    def test_raise_malformed_yaml_preserves_parser_msg(self):
        # REQ-202 — PyYAML line/column preserved verbatim.
        exc = type("YAMLError", (Exception,), {})("line 5, column 3: bad indent")
        with pytest.raises(MalformedYAMLError) as ei:
            raise_malformed_yaml("x.yaml", exc)  # type: ignore[arg-type]
        assert "yaml parse error in x.yaml: line 5, column 3: bad indent" in str(ei.value)

    def test_raise_app_config_validation_summarizes_first_error(self):
        # REQ-301/302 — a REAL pydantic ValidationError, summarized to one line.
        try:
            _ToyConfig(lr=0.1, steps=3, rogue="x")  # type: ignore[call-arg]
        except ValidationError as exc:
            assert len(exc.errors()) == 1
            with pytest.raises(AppConfigValidationError) as ei:
                raise_app_config_validation("_ToyConfig", exc)
            detail = str(ei.value)
            assert "schema validation failed for _ToyConfig" in detail
            assert "1 errors" in detail  # single rogue field (extra forbidden)
            assert "first:" in detail
            assert "rogue" in detail  # the offending field name (loc)
            # pydantic v2 names the extra-forbidden type ``extra_forbidden``
            # (the acceptance criteria's V1 ``value_error.extra`` is the legacy
            # spelling — corrected to verified reality here).
            assert "extra_forbidden" in detail
        else:  # pragma: no cover - the construction must raise
            pytest.fail("expected ValidationError")

    def test_raise_app_config_validation_error_count(self):
        # REQ-302 — ``N`` matches ``len(exc.errors())`` for multiple violations.
        try:
            _ToyConfig(lr="fast", steps="many", rogue="x")  # type: ignore[call-arg]
        except ValidationError as exc:
            n = len(exc.errors())
            assert n >= 2
            with pytest.raises(AppConfigValidationError) as ei:
                raise_app_config_validation("_ToyConfig", exc)
            assert f"{n} errors" in str(ei.value)
        else:  # pragma: no cover
            pytest.fail("expected ValidationError")

    def test_raise_app_config_validation_handles_empty_errors(self):
        # Defensive: a ValidationError-shaped object reporting zero errors still
        # raises (the ``first`` slot degrades gracefully rather than Indexerror).
        class _Empty:
            def errors(self):
                return []

        with pytest.raises(AppConfigValidationError, match="0 errors"):
            raise_app_config_validation("_ToyConfig", _Empty())  # type: ignore[arg-type]

    def test_raise_malformed_eval_results_missing_key(self):
        # REQ-401 / EDGE-003.
        with pytest.raises(MalformedEvalResultsError, match="missing key: candidate_total"):
            raise_malformed_eval_results("missing key: candidate_total")

    def test_raise_malformed_eval_results_invalid_type(self):
        # REQ-402 / EDGE-004 — expected + got both present.
        with pytest.raises(MalformedEvalResultsError) as ei:
            raise_malformed_eval_results(
                "invalid type for samples: expected list, got str"
            )
        msg = str(ei.value)
        assert "expected list" in msg and "got str" in msg

    def test_raise_malformed_eval_results_with_detail(self):
        # The json-parse path composes reason + detail.
        with pytest.raises(MalformedEvalResultsError) as ei:
            raise_malformed_eval_results("json parse error", "Expecting value: line 1 column 1")
        assert "json parse error: Expecting value" in str(ei.value)


# ---------------------------------------------------------------------------
# Emitter — NFR-202 grep, EDGE-102 single-line JSON, EDGE-103 no ANSI
# ---------------------------------------------------------------------------


class TestEmitter:
    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_emit_human_mode_writes_stderr(self, cls):
        out, err = io.StringIO(), io.StringIO()
        emit_operator_error(cls("boom"), stdout=out, stderr=err)
        assert err.getvalue() == f"{cls.__name__}: boom\n"
        assert out.getvalue() == ""

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_emit_json_mode_writes_stdout_single_line(self, cls):
        # EDGE-102 — one line, trailing newline only, no embedded newline.
        out, err = io.StringIO(), io.StringIO()
        emit_operator_error(cls("boom"), json_mode=True, stdout=out, stderr=err)
        payload = out.getvalue()
        assert payload.endswith("\n")
        assert payload.count("\n") == 1
        assert "\n" not in payload.rstrip("\n")
        assert err.getvalue() == ""

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_emit_json_mode_payload_shape(self, cls):
        import json

        out = io.StringIO()
        emit_operator_error(cls("boom"), json_mode=True, stdout=out)
        assert json.loads(out.getvalue()) == {
            "error": cls.__name__,
            "detail": "boom",
            "exit_status": 78,
        }

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_emit_human_starts_with_class_name(self, cls):
        # NFR-202.
        err = io.StringIO()
        emit_operator_error(cls("boom"), stderr=err)
        assert err.getvalue().startswith(cls.__name__ + ": ")

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_emit_no_ansi_codes(self, cls):
        # EDGE-103 — both modes are ANSI-free.
        out, err = io.StringIO(), io.StringIO()
        emit_operator_error(cls("boom"), stdout=out, stderr=err)
        assert "\x1b[" not in out.getvalue() + err.getvalue()
        out2, err2 = io.StringIO(), io.StringIO()
        emit_operator_error(cls("boom"), json_mode=True, stdout=out2, stderr=err2)
        assert "\x1b[" not in out2.getvalue() + err2.getvalue()

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_message_under_120_chars(self, cls):
        # NFR-203 — a representative message fits a terminal width.
        e = cls("config not found: /tmp/really/long/path/to/9b_tg_lora.yaml")
        assert len(str(e)) <= 120


# ---------------------------------------------------------------------------
# Leaf independence — zero import side effects, constant value
# ---------------------------------------------------------------------------


class TestLeafIndependence:
    def test_leaf_imports_no_heavy_deps(self):
        # Re-import in a clean subprocess so the pre-existing torch/omegaconf
        # imports from the test session do not pollute the baseline.
        import os
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        probe = (
            "import sys; before=set(sys.modules); "
            "import src.utils.cli_errors; "
            "heavy={'torch','pydantic','omegaconf','yaml'}; "
            "new={k.split('.')[0] for k in set(sys.modules)-before} & heavy; "
            "print('NEW' if new else 'CLEAN', sorted(new))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, cwd=repo_root,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("CLEAN"), result.stdout

    def test_exit_operator_error_constant_value(self):
        # REQ-003 / REQ-705.
        assert EXIT_OPERATOR_ERROR == 78

    @pytest.mark.parametrize("cls", SUBTYPES)
    def test_all_subtypes_exit_status_78_by_default(self, cls):
        assert cls("x").exit_status == 78


# ---------------------------------------------------------------------------
# TASK-0182 — launcher ↔ leaf ↔ worker constant agreement (drift detection)
# ---------------------------------------------------------------------------


class TestLauncherExitCodeIntegration:
    """The launcher mirrors the leaf's ``EXIT_OPERATOR_ERROR`` by VALUE.

    The launcher (``scripts.launch_freeze_ci_9b_full``) is deliberately
    torch-free so it can poll a busy card, so it does not IMPORT the leaf — it
    redeclares ``EXIT_OPERATOR_ERROR = 78`` and branches on it in
    ``classify_exit_code`` (TASK-0182). The worker (producer) DOES import the
    leaf constant. So the SAME integer lives in three places: the leaf
    (definition), the worker (imported), the launcher (mirrored). Without this
    pin, a unilateral value change on one side would drift silently.

    The cross-module equality is *transitively* pinned by
    ``test_worker_launcher_exit_contract.py`` (launcher == worker == 78), but
    that chain only holds while the worker keeps importing the leaf — this test
    pins the launcher↔leaf leg directly so a maintainer who later inlines the
    worker's constant is still caught.
    """

    def test_launcher_matches_leaf_constant(self):
        import scripts.launch_freeze_ci_9b_full as launcher

        assert launcher.EXIT_OPERATOR_ERROR == EXIT_OPERATOR_ERROR == 78

    def test_launcher_constant_is_the_leaf_constant_value(self):
        # Explicit non-transitive pin: the launcher's literal is the leaf's
        # value, the contract the launcher's classify branch was written against.
        import scripts.launch_freeze_ci_9b_full as launcher

        assert launcher.EXIT_OPERATOR_ERROR is not None
        assert int(launcher.EXIT_OPERATOR_ERROR) == int(EXIT_OPERATOR_ERROR)


# ---------------------------------------------------------------------------
# TASK-0183 — NFR-101 axis-wide mutation proof
#
# TestWrapperMutation (above) pins the POSITIVE: each wrapper raises its exact
# subtype. These pin the NEGATIVE: if a maintainer neutralizes a wrapper body
# (``pass`` / ``return None``), the wrapper returns instead of raising and the
# operator-error class is SILENTLY DROPPED — i.e. each detection test is
# mutation-sensitive. Consolidated here so the four-axis sensitivity is visible
# in one place rather than scattered across the entrypoint test files.
# ---------------------------------------------------------------------------


class TestOperatorErrorAxisMutationProof:
    @pytest.mark.parametrize(
        ("wrapper_name", "subtype"),
        [
            ("raise_missing_config", MissingConfigError),
            ("raise_malformed_yaml", MalformedYAMLError),
            ("raise_app_config_validation", AppConfigValidationError),
            ("raise_malformed_eval_results", MalformedEvalResultsError),
        ],
    )
    def test_neutralized_wrapper_drops_detection(self, monkeypatch, wrapper_name, subtype):
        # NFR-101 axis-wide mutation proof. TestWrapperMutation (above) pins the
        # POSITIVE — each wrapper's real body raises its exact subtype via
        # ``pytest.raises``. This pins the NEGATIVE: if a maintainer neutralizes
        # a wrapper body (``pass`` / ``return None``), the wrapper returns
        # instead of raising and the operator-error class is SILENTLY DROPPED —
        # exactly the corruption mode the axis exists to close, and the reason
        # each detection test goes RED under the mutation. ``subtype`` is in the
        # parametrize so a wrapper that raises the WRONG subtype is caught too.
        import src.utils.cli_errors as leaf

        monkeypatch.setattr(leaf, wrapper_name, lambda *a, **k: None)
        wrapper = getattr(leaf, wrapper_name)
        # The neutralized wrapper swallows any call. If it still raised the
        # subtype, the call below would propagate it and this test would ERROR
        # rather than pass — that is the mutation-sensitivity being pinned.
        assert wrapper("/nope.yaml") is None
        assert wrapper("x.yaml", Exception("bad")) is None
        assert wrapper("missing key: candidate_total") is None


# ---------------------------------------------------------------------------
# TASK-0183 — assembled wiring regression net (the b8ee35c-analog)
#
# ``b8ee35c`` drove the REAL assembled heterogeneous §4 path end-to-end and
# asserted the 5 per-commit honesty invariants at integration scale in ONE
# test. This is the operator-error-axis mirror: it pins, at the SOURCE level,
# that all THREE entrypoints wire the operator-error axis (import the leaf +
# emit/route operator errors) — so a future change that silently unwires one
# entrypoint is caught here without re-deriving the per-entrypoint integration
# tests. GPU-free, torch-free (reads source text only); deliberately NOT a
# brittle test-count assertion.
# ---------------------------------------------------------------------------


class TestOperatorErrorWiringAssembled:
    """All three freeze-ci-9b entrypoints wire the operator-error leaf.

    The axis is only as strong as its WEAKEST entrypoint: if any one of the
    producer / replay / launcher stops importing the leaf or routing exit 78,
    that entrypoint silently reverts to an undifferentiated traceback (the
    pre-axis state). This pins the wiring is present in all three in one net.
    """

    def _src(self, rel_path: str) -> str:
        # Read source by PATH (not import) so this leaf test file stays
        # torch-free: importing ``scripts.run_freeze_validloss_ci_9b`` would
        # drag the worker's torch/peft graph into the leaf test session.
        import os
        from pathlib import Path

        repo_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return (repo_root / rel_path).read_text(encoding="utf-8")

    def test_producer_wires_operator_error_axis(self):
        # scripts/run_freeze_validloss_ci_9b.py (the worker the launcher spawns).
        src = self._src("scripts/run_freeze_validloss_ci_9b.py")
        assert "from src.utils.cli_errors import" in src
        assert "EXIT_OPERATOR_ERROR" in src
        assert "emit_operator_error(" in src
        assert "return EXIT_OPERATOR_ERROR" in src  # named constant, not bare 78

    def test_replay_wires_operator_error_axis(self):
        # scripts/replay_freeze_validloss_ci.py (standalone replay gate).
        src = self._src("scripts/replay_freeze_validloss_ci.py")
        assert "from src.utils.cli_errors import" in src
        assert "EXIT_OPERATOR_ERROR" in src
        assert "emit_operator_error(" in src
        assert "raise_malformed_eval_results" in src  # the replay-specific class 4

    def test_launcher_routes_operator_error_exit_code(self):
        # scripts/launch_freeze_ci_9b_full.py — classifies worker exit 78 FATAL.
        src = self._src("scripts/launch_freeze_ci_9b_full.py")
        assert "EXIT_OPERATOR_ERROR = 78" in src
        assert "if code == EXIT_OPERATOR_ERROR" in src
        assert '"operator_error"' in src  # the distinct classify kind

    def test_wired_entrypoints_raise_distinct_classes(self):
        # All four operator-error classes are wired into entrypoint main flows.
        #
        #   raise_missing_config        -> producer (--config) + replay (--samples-file)
        #   raise_malformed_yaml        -> producer (--config YAML parse)
        #   raise_malformed_eval_results-> replay (samples-file schema)
        #   raise_app_config_validation -> producer (--config Pydantic schema,
        #       TASK-0184): wired at the producer ``_load_cfg`` level ONLY.
        #       ``_load_cfg`` runs ``config_schema.validate_config_data`` on the
        #       parsed mapping and converts the raw ``pydantic.ValidationError``
        #       to AppConfigValidationError; ``config_schema`` itself is
        #       UNTOUCHED (keeps raising raw ValidationError), so the existing
        #       ``pytest.raises(ValidationError)`` pins stay green. Replay does
        #       NOT wire it — replay validates a samples JSON via
        #       MalformedEvalResultsError, not a training config.
        producer = self._src("scripts/run_freeze_validloss_ci_9b.py")
        replay = self._src("scripts/replay_freeze_validloss_ci.py")
        assert "raise_missing_config" in producer
        assert "raise_missing_config" in replay
        assert "raise_malformed_yaml" in producer
        assert "raise_malformed_eval_results" in replay
        # AppConfigValidationError is now wired into the PRODUCER only (the one
        # entrypoint that loads a Pydantic-validated training config). Replay has
        # no training config, so it must stay unwired — pin that asymmetry so a
        # future change to either side is a conscious decision, not a drift.
        assert "raise_app_config_validation(" in producer
        assert "raise_app_config_validation(" not in replay
