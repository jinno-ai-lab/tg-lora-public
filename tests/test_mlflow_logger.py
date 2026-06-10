"""Tests for src/utils/mlflow_logger.py — TASK-0022, TASK-0057.

All mlflow calls are mocked so no server is required.
"""

from __future__ import annotations

import re
import types
from unittest.mock import MagicMock, patch


import pytest

from src.utils.mlflow_logger import MLflowLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_mlflow() -> types.ModuleType:
    """Create a fake ``mlflow`` module with the subset we use."""
    fake = types.ModuleType("mlflow")
    fake.start_run = MagicMock(return_value=MagicMock())
    fake.end_run = MagicMock()
    fake.log_params = MagicMock()
    fake.log_metrics = MagicMock()
    fake.log_artifact = MagicMock()
    fake.set_tracking_uri = MagicMock()
    fake.set_experiment = MagicMock()
    fake.set_tag = MagicMock()

    # mlflow.pyfunc.log_model
    fake_pyfunc = types.ModuleType("mlflow.pyfunc")
    fake_pyfunc.log_model = MagicMock()
    fake.pyfunc = fake_pyfunc

    return fake


# ---------------------------------------------------------------------------
# Test: log_params
# ---------------------------------------------------------------------------


class TestLogParams:
    def test_log_params_delegates_to_mlflow(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            logger = MLflowLogger(enabled=True, experiment_name="test")
            with logger:
                logger.log_params({"lr": 0.001, "batch_size": 4})
        fake.log_params.assert_called_once_with({"lr": 0.001, "batch_size": 4})

    def test_log_params_noop_when_disabled(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            logger = MLflowLogger(enabled=False)
            logger.log_params({"lr": 0.001})
        fake.log_params.assert_not_called()


# ---------------------------------------------------------------------------
# Test: log_metrics
# ---------------------------------------------------------------------------


class TestLogMetrics:
    def test_log_metrics_with_step(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            logger = MLflowLogger(enabled=True)
            with logger:
                logger.log_metrics({"loss": 2.5}, step=10)
        fake.log_metrics.assert_called_once_with({"loss": 2.5}, step=10)

    def test_log_metrics_noop_when_disabled(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            logger = MLflowLogger(enabled=False)
            logger.log_metrics({"loss": 2.5}, step=1)
        fake.log_metrics.assert_not_called()


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_enter_starts_run(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, run_name="my-run") as mlf:
                assert mlf.enabled
            fake.start_run.assert_called_once_with(run_name="my-run")
            fake.end_run.assert_called_once_with("FINISHED")

    def test_exit_on_exception_marks_failed(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            try:
                with MLflowLogger(enabled=True):
                    raise ValueError("boom")
            except ValueError:
                pass
        fake.end_run.assert_called_once_with("FAILED")

    def test_context_manager_noop_when_disabled(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=False) as mlf:
                assert not mlf.enabled
        fake.start_run.assert_not_called()
        fake.end_run.assert_not_called()


# ---------------------------------------------------------------------------
# Test: disabled fallback
# ---------------------------------------------------------------------------


class TestDisabledFallback:
    def test_enabled_false_means_noop(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=False)
            assert not mlf.enabled
            mlf.log_params({"x": 1})
            mlf.log_metrics({"y": 2.0}, step=0)
            mlf.set_tag("k", "v")
        fake.log_params.assert_not_called()
        fake.log_metrics.assert_not_called()
        fake.set_tag.assert_not_called()


# ---------------------------------------------------------------------------
# Test: import error fallback
# ---------------------------------------------------------------------------


class TestImportErrorFallback:
    def test_mlflow_none_means_disabled(self):
        with patch("src.utils.mlflow_logger._mlflow", None):
            mlf = MLflowLogger(enabled=True)
            assert not mlf.enabled
            # All calls should be silent no-ops
            mlf.log_params({"x": 1})
            mlf.log_metrics({"y": 2.0}, step=0)
            mlf.log_artifact("/tmp/fake")
            mlf.set_tag("k", "v")

    def test_import_error_does_not_raise(self):
        """If mlflow was never importable, constructing MLflowLogger must not raise."""
        with patch("src.utils.mlflow_logger._mlflow", None):
            mlf = MLflowLogger(enabled=True, experiment_name="test")
            assert not mlf.enabled


# ---------------------------------------------------------------------------
# Test: log_artifact / set_tag
# ---------------------------------------------------------------------------


class TestOtherMethods:
    def test_log_artifact(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_artifact("/tmp/model.safetensors", "checkpoints")
        fake.log_artifact.assert_called_once_with(
            "/tmp/model.safetensors", "checkpoints"
        )

    def test_set_tag(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.set_tag("mode", "tg_lora")
        fake.set_tag.assert_called_once_with("mode", "tg_lora")

    def test_tracking_uri_set_when_provided(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            MLflowLogger(enabled=True, tracking_uri="http://localhost:5000")
        fake.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_experiment_name_set_when_provided(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            MLflowLogger(enabled=True, experiment_name="my-experiment")
        fake.set_experiment.assert_called_once_with("my-experiment")


# ---------------------------------------------------------------------------
# Test: log_model
# ---------------------------------------------------------------------------


class TestLogModel:
    def test_log_model_delegates_to_pyfunc(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_model("my_model_obj", artifact_path="checkpoint")
        fake.pyfunc.log_model.assert_called_once_with(
            "checkpoint", python_model="my_model_obj"
        )

    def test_log_model_default_artifact_path(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_model("my_model_obj")
        fake.pyfunc.log_model.assert_called_once_with(
            "model", python_model="my_model_obj"
        )

    def test_log_model_noop_when_disabled(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=False)
            mlf.log_model("my_model_obj")
        fake.pyfunc.log_model.assert_not_called()

    def test_log_model_noop_when_mlflow_unavailable(self):
        with patch("src.utils.mlflow_logger._mlflow", None):
            mlf = MLflowLogger(enabled=True)
            # Should silently no-op without error
            mlf.log_model("my_model_obj")


# ---------------------------------------------------------------------------
# Test: TASK-0057 auto-generated metadata
# ---------------------------------------------------------------------------


class TestAutoMetadata:
    """TASK-0057: run name, tags, and description auto-generated from config."""

    def test_run_name_format_with_K_N(self):
        """Run name follows {experiment_name}_{timestamp}_{K}-{N}."""
        fake = _make_fake_mlflow()
        config = {"experiment_name": "test_exp", "K": 3, "N": 5}
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, config=config):
                pass
        run_name = fake.start_run.call_args.kwargs["run_name"]
        assert run_name.startswith("test_exp_")
        assert run_name.endswith("_3-5")
        middle = run_name[len("test_exp_") : -len("_3-5")]
        assert re.match(r"\d{8}_\d{6}", middle)

    def test_run_name_without_K_N(self):
        """When K/N missing, run name is {experiment_name}_{timestamp}."""
        fake = _make_fake_mlflow()
        config = {"experiment_name": "baseline"}
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, config=config):
                pass
        run_name = fake.start_run.call_args.kwargs["run_name"]
        assert re.match(r"baseline_\d{8}_\d{6}$", run_name)

    def test_hyperparameter_tags(self):
        """K, N, alpha, beta, lr are logged as hp.* tags."""
        fake = _make_fake_mlflow()
        config = {
            "experiment_name": "test",
            "K": 3,
            "N": 5,
            "alpha": 0.3,
            "beta": 0.8,
            "lr": 0.0005,
        }
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, config=config):
                pass
        tag_calls = {c[0][0]: c[0][1] for c in fake.set_tag.call_args_list}
        assert tag_calls.get("hp.K") == "3"
        assert tag_calls.get("hp.N") == "5"
        assert tag_calls.get("hp.alpha") == "0.3"
        assert tag_calls.get("hp.beta") == "0.8"
        assert tag_calls.get("hp.lr") == "0.0005"

    def test_description_tag_set(self):
        """Config summary is stored as mlflow.note tag."""
        fake = _make_fake_mlflow()
        config = {"experiment_name": "exp1", "K": 3, "N": 5, "lr": 0.001}
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, config=config):
                pass
        tag_calls = {c[0][0]: c[0][1] for c in fake.set_tag.call_args_list}
        assert "mlflow.note" in tag_calls
        note = tag_calls["mlflow.note"]
        assert "experiment=exp1" in note
        assert "K=3" in note
        assert "N=5" in note
        assert "lr=0.001" in note

    def test_metadata_skipped_when_disabled(self):
        """When enabled=False, no metadata is generated."""
        fake = _make_fake_mlflow()
        config = {"experiment_name": "test", "K": 3, "N": 5}
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=False, config=config):
                pass
        fake.start_run.assert_not_called()
        fake.set_tag.assert_not_called()

    def test_config_overrides_explicit_run_name(self):
        """Config experiment_name takes priority over explicit run_name."""
        fake = _make_fake_mlflow()
        config = {"experiment_name": "auto_name", "K": 2, "N": 4}
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, run_name="manual", config=config):
                pass
        run_name = fake.start_run.call_args.kwargs["run_name"]
        assert run_name.startswith("auto_name_")
        assert run_name.endswith("_2-4")

    def test_falls_back_to_explicit_run_name_without_config(self):
        """Without config, explicit run_name is used as before."""
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, run_name="manual"):
                pass
        run_name = fake.start_run.call_args.kwargs["run_name"]
        assert run_name == "manual"

    def test_partial_config_only_sets_available_tags(self):
        """Only config keys present become tags."""
        fake = _make_fake_mlflow()
        config = {"experiment_name": "test", "K": 3}
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True, config=config):
                pass
        tag_calls = {c[0][0]: c[0][1] for c in fake.set_tag.call_args_list}
        assert "hp.K" in tag_calls
        assert "hp.N" not in tag_calls


# ---------------------------------------------------------------------------
# Test: TASK-0097 constructor validation
# ---------------------------------------------------------------------------


class TestMLflowLoggerValidation:
    """TASK-0097: MLflowLogger.__init__ rejects invalid parameter values."""

    def test_init_rejects_empty_tracking_uri(self):
        with pytest.raises(ValueError, match="tracking_uri must be a non-empty string"):
            MLflowLogger(enabled=True, tracking_uri="")

    def test_init_rejects_empty_experiment_name(self):
        with pytest.raises(ValueError, match="experiment_name must be a non-empty string"):
            MLflowLogger(enabled=True, experiment_name="")

    def test_init_rejects_empty_run_name(self):
        with pytest.raises(ValueError, match="run_name must be a non-empty string"):
            MLflowLogger(enabled=True, run_name="")

    def test_init_rejects_non_dict_config(self):
        with pytest.raises(ValueError, match="config must be a dict"):
            MLflowLogger(enabled=True, config="not a dict")

    def test_init_rejects_list_config(self):
        with pytest.raises(ValueError, match="config must be a dict"):
            MLflowLogger(enabled=True, config=[1, 2, 3])

    def test_init_accepts_none_params(self):
        mlf = MLflowLogger(
            enabled=False,
            tracking_uri=None,
            experiment_name=None,
            run_name=None,
            config=None,
        )
        assert not mlf.enabled

    def test_init_accepts_valid_params(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(
                enabled=True,
                tracking_uri="http://localhost:5000",
                experiment_name="exp",
                run_name="run1",
                config={"key": "val"},
            )
            assert mlf.enabled
