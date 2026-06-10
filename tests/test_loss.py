"""Unit tests for src/training/loss.py."""

import pytest
import torch
from unittest.mock import MagicMock

from src.training.loss import compute_loss


class TestComputeLoss:
    def test_returns_model_loss(self):
        """compute_loss forwards batch to model and returns loss tensor."""
        model = MagicMock()
        expected_loss = torch.tensor(2.5)
        model.return_value = MagicMock(loss=expected_loss)

        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.randint(0, 100, (2, 8)),
        }

        result = compute_loss(model, batch)

        assert result is expected_loss
        model.assert_called_once_with(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def test_forwards_all_batch_keys(self):
        """compute_loss passes input_ids, attention_mask, and labels to model."""
        model = MagicMock()
        model.return_value = MagicMock(loss=torch.tensor(1.0))

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }

        compute_loss(model, batch)

        model.assert_called_once_with(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def test_raises_on_missing_input_ids(self):
        model = MagicMock()
        batch = {
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.randint(0, 100, (2, 8)),
        }
        with pytest.raises(KeyError, match="input_ids"):
            compute_loss(model, batch)

    def test_raises_on_missing_attention_mask(self):
        model = MagicMock()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8)),
        }
        with pytest.raises(KeyError, match="attention_mask"):
            compute_loss(model, batch)

    def test_raises_on_missing_labels(self):
        model = MagicMock()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        with pytest.raises(KeyError, match="labels"):
            compute_loss(model, batch)

    def test_extra_keys_allowed(self):
        model = MagicMock()
        model.return_value = MagicMock(loss=torch.tensor(1.0))
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "extra_key": "ignored",
        }
        result = compute_loss(model, batch)
        assert result is not None

    def test_empty_batch_raises(self):
        model = MagicMock()
        with pytest.raises(KeyError, match="input_ids"):
            compute_loss(model, {})
