"""Operator-error entrypoint integration for the 9B producer (TASK-0181).

Pins the producer side of the freeze-ci-operator-errors axis
(``specs/freeze-ci-operator-errors/``). The producer reads ``--config`` via
``OmegaConf.load`` in :func:`scripts.run_freeze_validloss_ci_9b._load_cfg`,
which converts the two REAL ``OmegaConf.load`` failure modes — a missing file
(``FileNotFoundError``) and unparseable YAML (``yaml.YAMLError``) — to
:class:`MissingConfigError` / :class:`MalformedYAMLError`, plus an explicit
0-byte check (EDGE-002; OmegaConf parses an empty file to ``{}`` rather than
raising). ``main``'s outer try/except emits the error and exits 78.

Two honest notes reflected below (and in acceptance-criteria.md):

* The producer DOES Pydantic-validate the config (TASK-0184): ``_load_cfg``
  runs the parsed mapping through ``config_schema.validate_config_data`` and
  converts a ``pydantic.ValidationError`` → :class:`AppConfigValidationError`
  (exit 78) AT THE PRODUCER LEVEL ONLY. ``config_schema`` itself is untouched
  — it keeps raising the raw ``pydantic.ValidationError``, so the existing
  ``pytest.raises(ValidationError)`` pins in test_config_schema /
  test_script_config_validation stay green. The ``AppConfigValidationError``
  wrapper is still pinned directly below against a REAL
  ``pydantic.ValidationError`` (the unit the producer conversion calls), and
  the ``_load_cfg`` / ``main`` integration tests pin the end-to-end exit-78
  path for an extra field (TC-301-01), a missing required field (TC-301-02),
  and a type mismatch (TC-301-03).
* In the no-ledger / non-sealed path the CUDA / free-memory guards fire
  BEFORE the config load (the documented "guards fire before config load"
  invariant). The config load therefore happens first only on the
  ``--ledger <existing>`` path, so the ``main()`` integration tests pass a
  dummy ``--ledger`` to reach the config-load branch on a GPU-busy host.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from scripts.run_freeze_validloss_ci_9b import _load_cfg, main
from src.utils.cli_errors import (
    AppConfigValidationError,
    MalformedYAMLError,
    MissingConfigError,
    raise_app_config_validation,
)


class _ToyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float
    steps: int


def _dummy_ledger(tmp_path: Path) -> Path:
    """An existing (empty) ledger file so ``main`` takes the config-load-first
    branch instead of the GPU-guard-first branch."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    return ledger


# ---------------------------------------------------------------------------
# _load_cfg — the real operator-error conversion (GPU-independent)
# ---------------------------------------------------------------------------


class TestLoadCfgConversion:
    def test_missing_config_raises_missing_config(self, tmp_path):
        with pytest.raises(MissingConfigError, match="config not found"):
            _load_cfg(tmp_path / "absent.yaml")

    def test_missing_config_message_names_path(self, tmp_path):
        with pytest.raises(MissingConfigError) as ei:
            _load_cfg("/nonexistent/9b_tg_lora.yaml")
        assert "config not found: /nonexistent/9b_tg_lora.yaml" in str(ei.value)

    def test_directory_config_is_flagged(self, tmp_path):
        # EDGE-001.
        with pytest.raises(MissingConfigError, match=r"is a directory"):
            _load_cfg(tmp_path)

    def test_malformed_yaml_raises_malformed_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: bar\n\tbaz: qux\n")  # tab indent -> ScannerError
        with pytest.raises(MalformedYAMLError, match="yaml parse error"):
            _load_cfg(bad)

    def test_malformed_yaml_preserves_line_column(self, tmp_path):
        # REQ-202 — PyYAML's line/column info is preserved verbatim.
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: bar\n\tbaz: qux\n")
        with pytest.raises(MalformedYAMLError) as ei:
            _load_cfg(bad)
        detail = str(ei.value)
        assert "line" in detail and "column" in detail

    def test_empty_file_raises_malformed_yaml(self, tmp_path):
        # EDGE-002 — OmegaConf parses a 0-byte file to {} rather than raising,
        # so _load_cfg checks stat().st_size explicitly.
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(MalformedYAMLError, match="file is empty"):
            _load_cfg(empty)


# ---------------------------------------------------------------------------
# main() — exit 78 + emit (via the --ledger config-load-first path)
# ---------------------------------------------------------------------------


class TestProducerMainOperatorError:
    def test_missing_config_exits_78(self, capsys, tmp_path):
        ledger = _dummy_ledger(tmp_path)
        assert main(["--config", "/nonexistent/9b_tg_lora.yaml",
                     "--ledger", str(ledger)]) == 78
        err = capsys.readouterr().err
        assert "MissingConfigError: config not found" in err

    def test_malformed_yaml_exits_78(self, capsys, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: bar\n\tbaz: qux\n")
        ledger = _dummy_ledger(tmp_path)
        assert main(["--config", str(bad), "--ledger", str(ledger)]) == 78
        assert "MalformedYAMLError: yaml parse error" in capsys.readouterr().err

    def test_empty_config_exits_78(self, capsys, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        ledger = _dummy_ledger(tmp_path)
        assert main(["--config", str(empty), "--ledger", str(ledger)]) == 78
        assert "file is empty" in capsys.readouterr().err

    def test_operator_error_stderr_has_class_name_line(self, capsys, tmp_path):
        # NFR-202 — a line greppable by class name.
        ledger = _dummy_ledger(tmp_path)
        main(["--config", "/nonexistent/9b_tg_lora.yaml", "--ledger", str(ledger)])
        err = capsys.readouterr().err
        assert any(line.startswith("MissingConfigError: ") for line in err.splitlines())

    def test_operator_error_no_ansi_codes(self, capsys, tmp_path):
        # EDGE-103.
        ledger = _dummy_ledger(tmp_path)
        main(["--config", "/nonexistent/9b_tg_lora.yaml", "--ledger", str(ledger)])
        assert "\x1b[" not in capsys.readouterr().err

    def test_json_mode_operator_error_stdout_single_line(self, capsys, tmp_path):
        # REQ-502 / EDGE-102 — --json emits a single JSON line on stdout.
        ledger = _dummy_ledger(tmp_path)
        assert main(["--config", "/nonexistent/9b_tg_lora.yaml",
                     "--ledger", str(ledger), "--json"]) == 78
        captured = capsys.readouterr()
        out, err = captured.out, captured.err
        assert out.endswith("\n")
        assert out.count("\n") == 1
        loaded = json.loads(out)
        assert loaded["error"] == "MissingConfigError"
        assert loaded["exit_status"] == 78
        assert "config not found" in loaded["detail"]
        assert err == ""

    def test_argparse_error_exit_code_unchanged(self):
        # REQ-601 / TC-704-05 — argparse usage error still exits 2 (parse_args
        # runs OUTSIDE the operator-error try/except).
        with pytest.raises(SystemExit) as ei:
            main(["--no-such-flag"])
        assert ei.value.code == 2


# ---------------------------------------------------------------------------
# REQ-301 — AppConfigValidationError at the leaf wrapper (REAL pydantic error)
#
# The producer does not Pydantic-validate (it reads OmegaConf directly), so the
# AppConfigValidationError axis is delivered + pinned at the wrapper that any
# future Pydantic-validating caller would use.
# ---------------------------------------------------------------------------


class TestAppConfigValidationWrapper:
    def test_extra_forbidden_summarized(self):
        try:
            _ToyConfig(lr=0.1, steps=3, rogue="x")  # type: ignore[call-arg]
        except ValidationError as exc:
            with pytest.raises(AppConfigValidationError) as ei:
                raise_app_config_validation("_ToyConfig", exc)
            detail = str(ei.value)
            assert "schema validation failed for _ToyConfig" in detail
            assert "1 errors" in detail
            assert "rogue" in detail
            assert "extra_forbidden" in detail  # pydantic v2 type (not V1 value_error.extra)
        else:  # pragma: no cover
            pytest.fail("expected ValidationError")

    def test_error_count_matches_pydantic(self):
        # REQ-302 — N == len(exc.errors()).
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


# ---------------------------------------------------------------------------
# REQ-301 — _load_cfg Pydantic validation gate (TC-301-01/02/03/B01, TASK-0184)
#
# _load_cfg now runs the parsed mapping through config_schema.validate_config_data
# and converts pydantic.ValidationError → AppConfigValidationError (exit 78). The
# fixtures below are minimal REAL-shape BaselineConfig YAMLs (the producer's
# default config family — the same schema the training loop uses) with one
# violation injected, so the first error is deterministic and the tests exercise
# the genuine schema, not a toy.
# ---------------------------------------------------------------------------

# A minimal config that PASSES BaselineConfig validation. Each test injects ONE
# violation into the relevant section so the first error is deterministic.
_VALID_BASELINE = """\
experiment: {name: t, seed: 1}
model: {name_or_path: m}
lora: {r: 1, alpha: 1, dropout: 0.0}
data: {train_path: a, valid_quick_path: b, valid_full_path: c}
training: {batch_size: 1, grad_accumulation: 1, learning_rate: 1.0e-4, max_steps: 10}
logging: {run_dir: r}
"""

_EXTRA_FIELD = "logging:\n  run_dir: r\n  rogue_extra: 1"
_MISSING_REQ = "logging: {}"
_TYPE_MISMATCH = "lora:\n  r: not-an-int\n  alpha: 1\n  dropout: 0.0"


def _write_cfg(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(body)
    return cfg


class TestLoadCfgPydanticValidation:
    """_load_cfg converts a Pydantic schema violation into
    AppConfigValidationError (the producer side of REQ-301)."""

    def test_extra_forbidden_field_raises_app_config(self, tmp_path):
        # TC-301-01 — an undeclared field under logging is rejected.
        body = _VALID_BASELINE.replace("logging: {run_dir: r}", _EXTRA_FIELD)
        with pytest.raises(AppConfigValidationError, match="schema validation failed") as ei:
            _load_cfg(_write_cfg(tmp_path, body))
        detail = str(ei.value)
        assert "BaselineConfig" in detail
        assert "1 errors" in detail
        assert "logging.rogue_extra" in detail
        assert "extra_forbidden" in detail  # pydantic v2 type

    def test_missing_required_field_raises_app_config(self, tmp_path):
        # TC-301-02 — logging has no required run_dir.
        body = _VALID_BASELINE.replace("logging: {run_dir: r}", _MISSING_REQ)
        with pytest.raises(AppConfigValidationError, match="schema validation failed") as ei:
            _load_cfg(_write_cfg(tmp_path, body))
        detail = str(ei.value)
        assert "logging.run_dir" in detail
        assert "missing" in detail  # pydantic v2 error type

    def test_type_mismatch_raises_app_config(self, tmp_path):
        # TC-301-03 — lora.r is a string, not an int.
        body = _VALID_BASELINE.replace(
            "lora: {r: 1, alpha: 1, dropout: 0.0}", _TYPE_MISMATCH
        )
        with pytest.raises(AppConfigValidationError, match="schema validation failed") as ei:
            _load_cfg(_write_cfg(tmp_path, body))
        detail = str(ei.value)
        assert "lora.r" in detail
        assert "int_parsing" in detail  # pydantic v2 type-mismatch type

    def test_baseline_config_class_name_in_detail(self, tmp_path):
        # TC-301-B01 — the producer's default family is BaselineConfig; its name
        # appears in the message so an operator knows which schema failed.
        body = _VALID_BASELINE.replace("logging: {run_dir: r}", _EXTRA_FIELD)
        with pytest.raises(AppConfigValidationError) as ei:
            _load_cfg(_write_cfg(tmp_path, body))
        assert "schema validation failed for BaselineConfig" in str(ei.value)

    def test_tg_lora_config_class_dispatch(self, tmp_path):
        # The class name follows ValidationError.title automatically: a
        # tg_lora-bearing config resolves to TGLoRAConfig (no dispatch dup). The
        # params are deliberately incomplete so validation fails under
        # TGLoRAConfig — proving the class name tracks the dispatched model.
        body = _VALID_BASELINE + "tg_lora: {K_initial: 1}\n"
        with pytest.raises(AppConfigValidationError) as ei:
            _load_cfg(_write_cfg(tmp_path, body))
        assert "schema validation failed for TGLoRAConfig" in str(ei.value)

    def test_valid_config_loads_unchanged(self, tmp_path):
        # Happy path: a schema-valid config returns the OmegaConf struct with NO
        # mutation (the producer reads cfg.model / cfg.lora / cfg.training
        # directly). This is the zero-regression guard for the new gate.
        cfg = _load_cfg(_write_cfg(tmp_path, _VALID_BASELINE))
        assert cfg.model.name_or_path == "m"
        assert cfg.lora.r == 1
        assert cfg.training.learning_rate == 1.0e-4

    def test_validation_is_load_bearing(self, tmp_path, monkeypatch):
        # Mutation proof: neutralize the producer's validate_config_data and the
        # schema-invalid config must NO LONGER raise AppConfigValidationError —
        # proving the 78 path is produced by the gate, not by accident.
        import scripts.run_freeze_validloss_ci_9b as mod

        body = _VALID_BASELINE.replace("logging: {run_dir: r}", _EXTRA_FIELD)
        monkeypatch.setattr(mod, "validate_config_data", lambda data: None)
        cfg = _load_cfg(_write_cfg(tmp_path, body))
        # Without the gate the config loads without raising (OmegaConf keeps the
        # extra field; the producer reads the fields it needs regardless).
        assert cfg.model.name_or_path == "m"


class TestProducerMainAppConfigValidation:
    """main() exits 78 + emits AppConfigValidationError for a schema-invalid
    config (the --ledger config-load-first path, GPU-independent)."""

    def _run(self, tmp_path, body, *extra):
        ledger = _dummy_ledger(tmp_path)
        return main(
            ["--config", str(_write_cfg(tmp_path, body)), "--ledger", str(ledger), *extra]
        )

    def test_extra_field_exits_78(self, capsys, tmp_path):
        # TC-301-01.
        assert self._run(tmp_path, _VALID_BASELINE.replace("logging: {run_dir: r}", _EXTRA_FIELD)) == 78
        err = capsys.readouterr().err
        assert "AppConfigValidationError:" in err
        assert "schema validation failed" in err

    def test_missing_required_field_exits_78(self, capsys, tmp_path):
        # TC-301-02 (producer side; the launcher assembled mirror is
        # test_main_mirrors_worker_operator_error_exit in test_launch_freeze_ci_9b_full).
        assert self._run(tmp_path, _VALID_BASELINE.replace("logging: {run_dir: r}", _MISSING_REQ)) == 78
        assert "missing" in capsys.readouterr().err

    def test_type_mismatch_exits_78(self, capsys, tmp_path):
        # TC-301-03.
        body = _VALID_BASELINE.replace("lora: {r: 1, alpha: 1, dropout: 0.0}", _TYPE_MISMATCH)
        assert self._run(tmp_path, body) == 78
        assert "int_parsing" in capsys.readouterr().err

    def test_app_config_error_stderr_has_class_name_line(self, capsys, tmp_path):
        # NFR-202 — a line greppable by class name.
        self._run(tmp_path, _VALID_BASELINE.replace("logging: {run_dir: r}", _EXTRA_FIELD))
        err = capsys.readouterr().err
        assert any(line.startswith("AppConfigValidationError: ") for line in err.splitlines())

    def test_json_mode_app_config_error_single_line(self, capsys, tmp_path):
        # REQ-501 / EDGE-102 — --json emits a single JSON line on stdout, stderr empty.
        body = _VALID_BASELINE.replace("logging: {run_dir: r}", _EXTRA_FIELD)
        assert self._run(tmp_path, body, "--json") == 78
        captured = capsys.readouterr()
        out, err = captured.out, captured.err
        assert out.endswith("\n")
        assert out.count("\n") == 1
        loaded = json.loads(out)
        assert loaded["error"] == "AppConfigValidationError"
        assert loaded["exit_status"] == 78
        assert "schema validation failed" in loaded["detail"]
        assert err == ""
