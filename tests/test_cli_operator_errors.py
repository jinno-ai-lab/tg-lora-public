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
