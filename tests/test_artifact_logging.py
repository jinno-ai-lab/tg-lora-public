"""Tests for MLflow artifact logging integration — TASK-0056.

Verifies:
- log_artifact is called after save_checkpoint with correct paths
- Artifact logging is skipped when mlflow is disabled
- compare_runs.py log_reports_to_mlflow logs generated files
"""

import types
from unittest.mock import MagicMock, call, patch


from src.utils.mlflow_logger import MLflowLogger


def _make_fake_mlflow() -> types.ModuleType:
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


class TestCheckpointArtifactLogging:
    """Criterion 1 & 3: log_artifact called after save_checkpoint, skipped when disabled."""

    def test_log_artifact_called_with_checkpoint_path(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True) as mlf:
                mlf.log_artifact("/runs/best_model", "checkpoints")
        fake.log_artifact.assert_called_once_with("/runs/best_model", "checkpoints")

    def test_log_artifact_skipped_when_disabled(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=False)
            mlf.log_artifact("/runs/best_model", "checkpoints")
        fake.log_artifact.assert_not_called()

    def test_multiple_artifact_calls_after_each_checkpoint(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True) as mlf:
                mlf.log_artifact("/runs/best_model", "checkpoints")
                mlf.log_artifact("/runs/checkpoint-cycle-25", "checkpoints")
        assert fake.log_artifact.call_count == 2
        fake.log_artifact.assert_any_call("/runs/best_model", "checkpoints")
        fake.log_artifact.assert_any_call("/runs/checkpoint-cycle-25", "checkpoints")

    def test_periodic_checkpoint_artifact_logged(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True) as mlf:
                for cycle in [25, 50, 75]:
                    mlf.log_artifact(f"/runs/checkpoint-cycle-{cycle}", "checkpoints")
        assert fake.log_artifact.call_count == 3


class TestCompareRunsArtifactLogging:
    """Criterion 2 & 4: compare_runs logs reports as MLflow artifacts."""

    def test_log_reports_to_mlflow_logs_all_files(self, tmp_path):
        (tmp_path / "comparison_20260101.txt").write_text("report")
        (tmp_path / "comparison_20260101.md").write_text("# report")
        (tmp_path / "loss_comparison.png").write_bytes(b"\x89PNG")

        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            from scripts.compare_runs import log_reports_to_mlflow

            log_reports_to_mlflow(tmp_path)

        assert fake.log_artifact.call_count == 3
        logged_names = [c.args[0] for c in fake.log_artifact.call_args_list]
        assert str(tmp_path / "comparison_20260101.txt") in logged_names
        assert str(tmp_path / "comparison_20260101.md") in logged_names
        assert str(tmp_path / "loss_comparison.png") in logged_names

    def test_log_reports_to_mlflow_no_files(self, tmp_path):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            from scripts.compare_runs import log_reports_to_mlflow

            log_reports_to_mlflow(tmp_path)

        fake.log_artifact.assert_not_called()

    def test_log_reports_to_mlflow_uses_comparison_report_path(self, tmp_path):
        report_file = tmp_path / "report.json"
        report_file.write_text("{}")

        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            from scripts.compare_runs import log_reports_to_mlflow

            log_reports_to_mlflow(tmp_path)

        fake.log_artifact.assert_called_once_with(
            str(report_file),
            "comparison_report",
        )

    def test_log_reports_to_mlflow_skips_when_mlflow_unavailable(self, tmp_path):
        (tmp_path / "report.txt").write_text("data")
        with patch("src.utils.mlflow_logger._mlflow", None):
            from scripts.compare_runs import log_reports_to_mlflow

            log_reports_to_mlflow(tmp_path)


class TestArtifactLoggingIntegrationPattern:
    """Verify the integration pattern: save_checkpoint then log_artifact."""

    def test_disabled_logger_noops_on_all_methods(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            mlf = MLflowLogger(enabled=False)
            mlf.log_artifact("/tmp/ckpt", "checkpoints")
            mlf.log_artifact("/tmp/report.json", "comparison_report")
        fake.log_artifact.assert_not_called()

    def test_artifact_path_parameter_passed_through(self):
        fake = _make_fake_mlflow()
        with patch("src.utils.mlflow_logger._mlflow", fake):
            with MLflowLogger(enabled=True) as mlf:
                mlf.log_artifact("/runs/best_model", "checkpoints")
                mlf.log_artifact("/reports/loss.png", "comparison_report")
        calls = fake.log_artifact.call_args_list
        assert calls[0] == call("/runs/best_model", "checkpoints")
        assert calls[1] == call("/reports/loss.png", "comparison_report")
