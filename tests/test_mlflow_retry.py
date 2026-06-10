"""Tests for MLflow retry logic and error enhancement — TASK-0059.

All mlflow calls are mocked so no server is required.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch


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

    fake_pyfunc = types.ModuleType("mlflow.pyfunc")
    fake_pyfunc.log_model = MagicMock()
    fake.pyfunc = fake_pyfunc

    return fake


# ---------------------------------------------------------------------------
# Test: retry on ConnectionError
# ---------------------------------------------------------------------------


class TestRetryOnConnectionError:
    def test_retries_on_connection_error_and_succeeds(self):
        """After 2 ConnectionErrors, the 3rd attempt succeeds."""
        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = [
            ConnectionError("refused"),
            ConnectionError("refused"),
            None,
        ]
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
        assert fake.log_metrics.call_count == 3

    def test_retries_exhausted_gives_up_gracefully(self):
        """After 3 ConnectionErrors, the method returns without raising."""
        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = ConnectionError("refused")
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
        assert fake.log_metrics.call_count == 3

    def test_retry_on_timeout_error(self):
        """TimeoutError is also retried."""
        fake = _make_fake_mlflow()
        fake.log_params.side_effect = [TimeoutError("timed out"), None]
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_params({"lr": 0.01})
        assert fake.log_params.call_count == 2

    def test_retry_on_os_error(self):
        """OSError (e.g. network unreachable) is also retried."""
        fake = _make_fake_mlflow()
        fake.set_tag.side_effect = [OSError("network unreachable"), None]
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.set_tag("key", "value")
        assert fake.set_tag.call_count == 2


# ---------------------------------------------------------------------------
# Test: non-retryable exceptions fail immediately
# ---------------------------------------------------------------------------


class TestNonRetryableExceptions:
    def test_value_error_fails_immediately(self):
        """Non-retryable exceptions should be caught and logged, not retried."""
        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = ValueError("bad data")
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
        # Called exactly once — no retries
        assert fake.log_metrics.call_count == 1

    def test_type_error_fails_immediately(self):
        fake = _make_fake_mlflow()
        fake.log_params.side_effect = TypeError("wrong type")
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_params({"x": 1})
        assert fake.log_params.call_count == 1

    def test_runtime_error_fails_immediately(self):
        fake = _make_fake_mlflow()
        fake.log_artifact.side_effect = RuntimeError("unexpected")
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_artifact("/tmp/model.bin")
        assert fake.log_artifact.call_count == 1


# ---------------------------------------------------------------------------
# Test: error logs include operation type and target name
# ---------------------------------------------------------------------------


class TestErrorLogContent:
    def test_error_includes_operation_and_metrics_name(self, caplog):
        """Error log should mention the operation and metric key names."""
        import logging

        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = ConnectionError("refused")
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
            caplog.at_level(logging.ERROR),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0, "acc": 0.5}, step=1)
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        msg = errors[0].message
        assert "log_metrics" in msg
        assert "loss,acc" in msg

    def test_error_includes_operation_and_param_keys(self, caplog):
        """Error log should mention operation and param key names."""
        import logging

        fake = _make_fake_mlflow()
        fake.log_params.side_effect = ValueError("bad")
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            caplog.at_level(logging.ERROR),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_params({"lr": 0.01, "batch": 4})
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        msg = errors[0].message
        assert "log_params" in msg
        assert "lr,batch" in msg

    def test_warning_on_retry_includes_operation(self, caplog):
        """Retry warnings should include operation type."""
        import logging

        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = [ConnectionError("refused"), None]
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
            caplog.at_level(logging.WARNING),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "log_metrics" in warnings[0].message
        assert "loss" in warnings[0].message

    def test_error_includes_artifact_name(self, caplog):
        """Error log for log_artifact should include the file path."""
        import logging

        fake = _make_fake_mlflow()
        fake.log_artifact.side_effect = ConnectionError("down")
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
            caplog.at_level(logging.ERROR),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_artifact("/tmp/checkpoint.bin")
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        msg = errors[0].message
        assert "log_artifact" in msg
        assert "/tmp/checkpoint.bin" in msg


# ---------------------------------------------------------------------------
# Test: training loop never stops
# ---------------------------------------------------------------------------


class TestTrainingLoopResilience:
    def test_all_methods_swallow_exceptions(self):
        """None of the public methods should raise, even on repeated errors."""
        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = ConnectionError("down")
        fake.log_params.side_effect = ConnectionError("down")
        fake.log_artifact.side_effect = ConnectionError("down")
        fake.set_tag.side_effect = ConnectionError("down")
        fake.pyfunc.log_model.side_effect = ConnectionError("down")

        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
                mlf.log_params({"lr": 0.01})
                mlf.log_artifact("/tmp/model.bin")
                mlf.set_tag("key", "value")
                mlf.log_model("model", "checkpoint")
        # If we get here without an exception, the training loop is safe.

    def test_no_exception_propagates_from_log_metrics(self):
        """log_metrics must never raise into the training loop."""
        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = ConnectionError("down")
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep"),
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
                # No exception raised

    def test_disabled_logger_still_noop(self):
        """Disabled logger is still a silent no-op even after retry changes."""
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=False)
            mlf.log_params({"x": 1})
            mlf.log_metrics({"y": 2.0}, step=0)
            mlf.log_artifact("/tmp/fake")
            mlf.set_tag("k", "v")
        fake.log_params.assert_not_called()
        fake.log_metrics.assert_not_called()
        fake.set_tag.assert_not_called()


# ---------------------------------------------------------------------------
# Test: exponential backoff timing
# ---------------------------------------------------------------------------


class TestBackoffTiming:
    def test_backoff_delays_increase(self):
        """Verify exponential backoff: 1s, 2s between retries."""
        fake = _make_fake_mlflow()
        fake.log_metrics.side_effect = [
            ConnectionError("down"),
            ConnectionError("down"),
            None,
        ]
        with (
            patch("src.utils.mlflow_logger._mlflow", fake),
            patch("src.utils.mlflow_logger.time.sleep") as mock_sleep,
        ):
            mlf = MLflowLogger(enabled=True)
            with mlf:
                mlf.log_metrics({"loss": 1.0}, step=1)
        # sleep called after attempt 1 and attempt 2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # 2^0 * 1.0
        mock_sleep.assert_any_call(2.0)  # 2^1 * 1.0
