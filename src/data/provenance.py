from datetime import datetime, timezone


def create_provenance(
    seed_id: str,
    generator_model: str,
    source_type: str = "open_model_generated",
    closed_model_used: bool = False,
    quality_score: float = 0.0,
    review_status: str = "auto_passed",
    intended_use: str = "train",
) -> dict:
    return {
        "seed_data_id": seed_id,
        "source_type": source_type,
        "generator_model": generator_model,
        "closed_model_used": closed_model_used,
        "review_status": review_status,
        "quality_score": quality_score,
        "intended_use": intended_use,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
