"""Unit tests for src/data/provenance.py."""

from datetime import datetime, timezone

from src.data.provenance import create_provenance


class TestCreateProvenance:
    def test_returns_all_required_fields(self):
        result = create_provenance(
            seed_id="seed-001",
            generator_model="qwen-9b",
        )
        expected_keys = {
            "seed_data_id",
            "source_type",
            "generator_model",
            "closed_model_used",
            "review_status",
            "quality_score",
            "intended_use",
            "created_at",
        }
        assert set(result.keys()) == expected_keys

    def test_default_values(self):
        result = create_provenance(
            seed_id="seed-001",
            generator_model="qwen-9b",
        )
        assert result["seed_data_id"] == "seed-001"
        assert result["generator_model"] == "qwen-9b"
        assert result["source_type"] == "open_model_generated"
        assert result["closed_model_used"] is False
        assert result["review_status"] == "auto_passed"
        assert result["quality_score"] == 0.0
        assert result["intended_use"] == "train"

    def test_custom_values(self):
        result = create_provenance(
            seed_id="seed-042",
            generator_model="gpt-4",
            source_type="closed_model_generated",
            closed_model_used=True,
            quality_score=0.85,
            review_status="human_approved",
            intended_use="eval",
        )
        assert result["seed_data_id"] == "seed-042"
        assert result["generator_model"] == "gpt-4"
        assert result["source_type"] == "closed_model_generated"
        assert result["closed_model_used"] is True
        assert result["quality_score"] == 0.85
        assert result["review_status"] == "human_approved"
        assert result["intended_use"] == "eval"

    def test_created_at_is_valid_iso_format(self):
        result = create_provenance(
            seed_id="seed-001",
            generator_model="qwen-9b",
        )
        dt = datetime.fromisoformat(result["created_at"])
        assert dt.tzinfo is not None

    def test_created_at_is_recent_utc(self):
        result = create_provenance(
            seed_id="seed-001",
            generator_model="qwen-9b",
        )
        dt = datetime.fromisoformat(result["created_at"])
        now = datetime.now(timezone.utc)
        delta = (now - dt).total_seconds()
        assert abs(delta) < 5

    def test_quality_score_zero(self):
        result = create_provenance(
            seed_id="s1",
            generator_model="m",
            quality_score=0.0,
        )
        assert result["quality_score"] == 0.0

    def test_quality_score_max(self):
        result = create_provenance(
            seed_id="s1",
            generator_model="m",
            quality_score=1.0,
        )
        assert result["quality_score"] == 1.0
