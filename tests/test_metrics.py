import csv
import json

import pytest

from tg_lora.metrics import MetricsRecorder


class TestCSVRoundTrip:
    def test_write_and_read_csv(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(
            1, train_loss=0.5, valid_loss=0.4, K=3, N=5, alpha=1.0,
            beta=0.9, lr=1e-4, accepted=True,
            active_layer_strategy="last_25_percent", velocity_magnitude=0.1,
        )
        recorder.record_cycle(
            2, train_loss=0.3, valid_loss=0.35, K=3, N=5, alpha=1.0,
            beta=0.9, lr=1e-4, accepted=False,
            active_layer_strategy="middle_random", velocity_magnitude=0.05,
        )

        csv_files = list(tmp_path.glob("metrics_*.csv"))
        assert len(csv_files) == 1

        with open(csv_files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["cycle"] == "1"
        assert float(rows[0]["train_loss"]) == pytest.approx(0.5)
        assert float(rows[0]["valid_loss"]) == pytest.approx(0.4)
        assert rows[0]["accepted"] == "True"
        assert rows[0]["active_layer_strategy"] == "last_25_percent"
        assert rows[1]["cycle"] == "2"
        assert rows[1]["accepted"] == "False"


class TestJSONSummary:
    def test_summary_structure(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5, valid_loss=0.4)
        recorder.record_cycle(2, train_loss=0.3, valid_loss=0.35)
        recorder.record_cycle(3, train_loss=0.2, valid_loss=0.3)

        path = recorder.write_summary()
        assert path.exists()
        assert path.name.startswith("summary_")
        assert path.suffix == ".json"

        with open(path) as f:
            data = json.load(f)

        assert "total_cycles" in data
        assert "final" in data
        assert "best" in data
        assert "aggregate" in data
        assert "history" in data
        assert data["total_cycles"] == 3
        assert data["final"]["cycle"] == 3
        assert data["best"]["train_loss"] == pytest.approx(0.2)
        assert data["best"]["valid_loss"] == pytest.approx(0.3)

    def test_aggregate_stats(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=1.0, valid_loss=0.8, reduction_rate=0.5, acceptance_rate=0.6)
        recorder.record_cycle(2, train_loss=0.5, valid_loss=0.4, reduction_rate=0.6, acceptance_rate=0.7)
        recorder.record_cycle(3, train_loss=0.2, valid_loss=0.2, reduction_rate=0.7, acceptance_rate=0.8)

        summary = recorder.get_summary()
        agg = summary["aggregate"]
        assert agg["train_loss"]["mean"] == pytest.approx((1.0 + 0.5 + 0.2) / 3.0)
        assert agg["train_loss"]["min"] == pytest.approx(0.2)
        assert agg["train_loss"]["max"] == pytest.approx(1.0)
        assert agg["valid_loss"]["mean"] == pytest.approx((0.8 + 0.4 + 0.2) / 3.0)
        assert agg["valid_loss"]["min"] == pytest.approx(0.2)
        assert agg["valid_loss"]["max"] == pytest.approx(0.8)
        assert agg["final_reduction_rate"] == pytest.approx(0.7)
        assert agg["final_acceptance_rate"] == pytest.approx(0.8)


class TestAppendMode:
    def test_csv_appends_per_cycle(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5)
        recorder.record_cycle(2, train_loss=0.3)
        recorder.record_cycle(3, train_loss=0.1)

        csv_files = list(tmp_path.glob("metrics_*.csv"))
        assert len(csv_files) == 1

        with open(csv_files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert float(rows[0]["train_loss"]) == pytest.approx(0.5)
        assert float(rows[1]["train_loss"]) == pytest.approx(0.3)
        assert float(rows[2]["train_loss"]) == pytest.approx(0.1)


class TestEmptyState:
    def test_summary_with_no_cycles(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        summary = recorder.get_summary()
        assert summary["total_cycles"] == 0
        assert summary["final"] == {}
        assert summary["history"] == []
        assert summary["aggregate"]["train_loss"]["mean"] == 0.0
        assert summary["aggregate"]["train_loss"]["min"] == 0.0
        assert summary["aggregate"]["train_loss"]["max"] == 0.0
        assert summary["best"]["train_loss"] == 0.0
        assert summary["best"]["valid_loss"] == 0.0

    def test_write_summary_empty(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        path = recorder.write_summary()
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["total_cycles"] == 0


class TestMultipleCycles:
    def test_history_preserves_order(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        for i in range(5):
            recorder.record_cycle(i + 1, train_loss=1.0 / (i + 1))

        summary = recorder.get_summary()
        assert len(summary["history"]) == 5
        for i, row in enumerate(summary["history"]):
            assert row["cycle"] == i + 1
            assert row["train_loss"] == pytest.approx(1.0 / (i + 1))

    def test_final_is_last_recorded_cycle(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5, K=3, N=5)
        recorder.record_cycle(2, train_loss=0.3, K=4, N=6)

        summary = recorder.get_summary()
        assert summary["final"]["cycle"] == 2
        assert summary["final"]["K"] == 4
        assert summary["final"]["N"] == 6


class TestNonFiniteHandling:
    def test_nan_train_loss_becomes_zero(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=float("nan"))
        summary = recorder.get_summary()
        assert summary["final"]["train_loss"] == 0.0

    def test_inf_train_loss_becomes_zero(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=float("inf"))
        summary = recorder.get_summary()
        assert summary["final"]["train_loss"] == 0.0

    def test_negative_inf_train_loss_becomes_zero(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=float("-inf"))
        summary = recorder.get_summary()
        assert summary["final"]["train_loss"] == 0.0

    def test_nan_valid_loss_becomes_zero(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5, valid_loss=float("nan"))
        summary = recorder.get_summary()
        assert summary["final"]["valid_loss"] == 0.0

    def test_inf_valid_loss_becomes_zero(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5, valid_loss=float("inf"))
        summary = recorder.get_summary()
        assert summary["final"]["valid_loss"] == 0.0

    def test_none_valid_loss_stored_as_zero(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5, valid_loss=None)
        summary = recorder.get_summary()
        assert summary["final"]["valid_loss"] == 0.0

    def test_mixed_finite_and_nonfinite(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5)
        recorder.record_cycle(2, train_loss=float("nan"))
        recorder.record_cycle(3, train_loss=0.2)

        summary = recorder.get_summary()
        assert summary["aggregate"]["train_loss"]["min"] == pytest.approx(0.0)
        assert summary["aggregate"]["train_loss"]["max"] == pytest.approx(0.5)

    def test_nonfinite_fields_in_csv(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=float("nan"), valid_loss=float("inf"), alpha=float("nan"))

        csv_files = list(tmp_path.glob("metrics_*.csv"))
        with open(csv_files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert float(rows[0]["train_loss"]) == 0.0
        assert float(rows[0]["valid_loss"]) == 0.0
        assert float(rows[0]["alpha"]) == 0.0


class TestOutputDir:
    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        recorder = MetricsRecorder(output_dir=nested)
        recorder.record_cycle(1, train_loss=0.5)
        assert nested.exists()

    def test_write_summary_returns_path(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5)
        path = recorder.write_summary()
        assert path.parent == tmp_path
        assert path.exists()

    def test_get_summary_does_not_write(self, tmp_path):
        recorder = MetricsRecorder(output_dir=tmp_path)
        recorder.record_cycle(1, train_loss=0.5)
        recorder.get_summary()
        json_files = list(tmp_path.glob("summary_*.json"))
        assert len(json_files) == 0
