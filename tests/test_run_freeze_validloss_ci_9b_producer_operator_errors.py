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

* The producer does NOT Pydantic-validate the config (it reads the OmegaConf
  struct directly), so :class:`AppConfigValidationError` is exercised at the
  leaf-wrapper level against a REAL ``pydantic.ValidationError`` — not wired
  into ``config_schema.load_and_validate_config`` (doing so would break the
  existing ``pytest.raises(ValidationError)`` pins in test_config_schema /
  test_script_config_validation).
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
