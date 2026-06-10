import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.eval_loss import EvalLossResult, eval_loss, eval_loss_detailed


class _DictDataset(Dataset):
    def __init__(self, n=4):
        self.input_ids = torch.randint(0, 10, (n, 4))
        self.attention_mask = torch.ones(n, 4, dtype=torch.long)
        self.labels = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = input_ids.float()
        logits = self.linear(x)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return type("Out", (), {"loss": loss})()
        return type("Out", (), {"loss": torch.tensor(0.0)})()


def _make_loader(n=4):
    return DataLoader(_DictDataset(n), batch_size=2)


class _MaskedMixDataset(Dataset):
    def __init__(self):
        self.input_ids = torch.randint(0, 10, (3, 4))
        self.attention_mask = torch.ones(3, 4, dtype=torch.long)
        self.labels = torch.tensor([-100, 1, 0], dtype=torch.long)

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def test_eval_loss_preserves_training_mode():
    model = _TinyModel()

    # Case 1: model in training mode before eval
    model.train()
    assert model.training
    loader = _make_loader()
    _ = eval_loss(model, loader, device="cpu", max_batches=1)
    assert model.training  # should still be True

    # Case 2: model in eval mode before eval
    model.eval()
    assert not model.training
    loader = _make_loader()
    _ = eval_loss(model, loader, device="cpu", max_batches=1)
    assert not model.training  # should still be False


def test_eval_loss_computes_average_loss():
    model = _TinyModel()
    torch.manual_seed(42)
    loader = _make_loader(n=4)

    loss = eval_loss(model, loader, device="cpu")

    assert isinstance(loss, float)
    assert loss > 0.0

    # Verify it's truly an average by running batches individually
    torch.manual_seed(42)
    dataset = _DictDataset(n=4)
    loader2 = DataLoader(dataset, batch_size=2)

    individual_losses = []
    with torch.no_grad():
        model.eval()
        for batch in loader2:
            logits = model.linear(batch["input_ids"].float())
            loss_val = nn.functional.cross_entropy(logits, batch["labels"])
            individual_losses.append(loss_val.item())

    expected_avg = sum(individual_losses) / len(individual_losses)
    assert abs(loss - expected_avg) < 1e-5


def test_eval_loss_respects_max_batches():
    model = _TinyModel()
    loader = _make_loader(n=20)  # 10 batches of 2

    loss = eval_loss(model, loader, device="cpu", max_batches=2)

    assert isinstance(loss, float)
    assert loss > 0.0


def test_eval_loss_respects_max_examples_exactly():
    model = _TinyModel()
    loader = _make_loader(n=5)  # batches: 2,2,1

    loss = eval_loss(model, loader, device="cpu", max_examples=3)

    assert isinstance(loss, float)
    assert loss > 0.0


def test_eval_loss_empty_dataloader():
    model = _TinyModel()
    loader = DataLoader(_DictDataset(n=0), batch_size=2)

    loss = eval_loss(model, loader, device="cpu")

    assert math.isnan(loss)


def test_eval_loss_dropout_disabled_during_eval():
    class _DropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(p=0.99)
            self.linear = nn.Linear(4, 2)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.drop(input_ids.float())
            logits = self.linear(x)
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
                return type("Out", (), {"loss": loss})()
            return type("Out", (), {"loss": torch.tensor(0.0)})()

    model = _DropoutModel()
    model.eval()
    loader = _make_loader(n=4)

    loss1 = eval_loss(model, loader, device="cpu")
    loss2 = eval_loss(model, loader, device="cpu")

    # With eval mode, dropout is disabled so losses should be identical
    assert abs(loss1 - loss2) < 1e-6


def test_eval_loss_device_none_uses_model_device():
    model = _TinyModel()
    loader = _make_loader(n=2)
    # device=None should auto-detect from model parameters
    loss = eval_loss(model, loader, device=None, max_batches=1)
    assert isinstance(loss, float)
    assert loss > 0.0


def test_eval_loss_skips_all_masked_batches():
    model = _TinyModel()
    dataset = _MaskedMixDataset()

    mixed_loader = DataLoader(dataset, batch_size=1)
    valid_loader = DataLoader(torch.utils.data.Subset(dataset, [1, 2]), batch_size=1)

    mixed_loss = eval_loss(model, mixed_loader, device="cpu")
    valid_loss = eval_loss(model, valid_loader, device="cpu")

    assert math.isfinite(mixed_loss)
    assert abs(mixed_loss - valid_loss) < 1e-6


# ---- eval_loss_detailed tests ----


def test_eval_loss_detailed_returns_correct_avg():
    model = _TinyModel()
    torch.manual_seed(42)
    loader = _make_loader(n=4)

    result = eval_loss_detailed(model, loader, device="cpu")

    assert isinstance(result, EvalLossResult)
    assert result.num_batches == 2
    assert result.avg_loss > 0.0

    # avg should match eval_loss
    torch.manual_seed(42)
    loader2 = _make_loader(n=4)
    expected = eval_loss(model, loader2, device="cpu")
    assert abs(result.avg_loss - expected) < 1e-5


def test_eval_loss_detailed_perplexity():
    model = _TinyModel()
    loader = _make_loader(n=4)

    result = eval_loss_detailed(model, loader, device="cpu")

    assert result.perplexity == math.exp(result.avg_loss)
    assert result.perplexity > 1.0


def test_eval_loss_detailed_min_max():
    model = _TinyModel()
    loader = _make_loader(n=10)

    result = eval_loss_detailed(model, loader, device="cpu")

    assert result.min_loss <= result.avg_loss
    assert result.max_loss >= result.avg_loss
    assert result.min_loss > 0.0
    assert result.max_loss > 0.0


def test_eval_loss_detailed_respects_max_batches():
    model = _TinyModel()
    loader = _make_loader(n=20)

    result = eval_loss_detailed(model, loader, device="cpu", max_batches=2)

    assert result.num_batches == 2


def test_eval_loss_detailed_rejects_mixed_limits():
    model = _TinyModel()
    loader = _make_loader(n=4)

    with pytest.raises(ValueError, match="at most one"):
        eval_loss_detailed(model, loader, device="cpu", max_batches=1, max_examples=2)


def test_eval_loss_detailed_empty_dataloader():
    model = _TinyModel()
    loader = DataLoader(_DictDataset(n=0), batch_size=2)

    result = eval_loss_detailed(model, loader, device="cpu")

    assert math.isnan(result.avg_loss)
    assert result.num_batches == 0
    assert result.perplexity == float("inf")
    assert math.isnan(result.min_loss)
    assert math.isnan(result.max_loss)


def test_eval_loss_detailed_preserves_training_mode():
    model = _TinyModel()

    model.train()
    assert model.training
    loader = _make_loader()
    eval_loss_detailed(model, loader, device="cpu", max_batches=1)
    assert model.training

    model.eval()
    assert not model.training
    eval_loss_detailed(model, loader, device="cpu", max_batches=1)
    assert not model.training


def test_eval_loss_detailed_skips_all_masked_batches():
    model = _TinyModel()
    dataset = _MaskedMixDataset()

    mixed_loader = DataLoader(dataset, batch_size=1)
    valid_loader = DataLoader(torch.utils.data.Subset(dataset, [1, 2]), batch_size=1)

    mixed = eval_loss_detailed(model, mixed_loader, device="cpu")
    valid = eval_loss_detailed(model, valid_loader, device="cpu")

    assert mixed.num_batches == valid.num_batches
    assert abs(mixed.avg_loss - valid.avg_loss) < 1e-6


def test_eval_loss_result_repr():
    r = EvalLossResult(avg_loss=2.5, num_batches=10, min_loss=1.0, max_loss=4.0)
    s = repr(r)
    assert "avg_loss=2.5000" in s
    assert "ppl=" in s
    assert "batches=10" in s


def test_eval_loss_result_perplexity_inf_for_large_loss():
    r = EvalLossResult(avg_loss=200.0, num_batches=1, min_loss=200.0, max_loss=200.0)
    assert r.perplexity == float("inf")


def test_eval_loss_result_perplexity_inf_for_nan_loss():
    r = EvalLossResult(
        avg_loss=float("nan"),
        num_batches=0,
        min_loss=float("nan"),
        max_loss=float("nan"),
    )
    assert r.perplexity == float("inf")


def test_eval_loss_result_perplexity_inf_for_pos_inf_loss():
    r = EvalLossResult(
        avg_loss=float("inf"), num_batches=0, min_loss=float("inf"), max_loss=float("inf")
    )
    assert r.perplexity == float("inf")


def test_eval_loss_result_perplexity_inf_for_neg_inf_loss():
    r = EvalLossResult(
        avg_loss=float("-inf"), num_batches=0, min_loss=float("-inf"), max_loss=float("-inf")
    )
    assert r.perplexity == float("inf")


def test_eval_loss_result_perplexity_boundary_at_100():
    r = EvalLossResult(avg_loss=100.0, num_batches=1, min_loss=100.0, max_loss=100.0)
    assert r.perplexity == float("inf")


def test_eval_loss_result_perplexity_just_below_threshold():
    r = EvalLossResult(avg_loss=99.9, num_batches=1, min_loss=99.9, max_loss=99.9)
    assert math.isfinite(r.perplexity)
    assert r.perplexity == math.exp(99.9)


def test_eval_loss_result_perplexity_negative_loss():
    r = EvalLossResult(avg_loss=-5.0, num_batches=1, min_loss=-5.0, max_loss=-5.0)
    assert math.isfinite(r.perplexity)
    assert r.perplexity == math.exp(-5.0)
    assert 0.0 < r.perplexity < 1.0


# ---- TASK-0097: constructor validation ----


class TestEvalLossResultValidation:
    """TASK-0097: EvalLossResult.__init__ rejects invalid parameter values."""

    def test_init_rejects_nan_avg_loss(self):
        with pytest.raises(ValueError, match="avg_loss must be a finite number"):
            EvalLossResult(avg_loss=float("nan"), num_batches=1, min_loss=1.0, max_loss=2.0)

    def test_init_rejects_inf_avg_loss(self):
        with pytest.raises(ValueError, match="avg_loss must be a finite number"):
            EvalLossResult(avg_loss=float("inf"), num_batches=1, min_loss=1.0, max_loss=2.0)

    def test_init_rejects_negative_num_batches(self):
        with pytest.raises(ValueError, match="num_batches must be a non-negative integer"):
            EvalLossResult(avg_loss=1.0, num_batches=-1, min_loss=1.0, max_loss=1.0)

    def test_init_rejects_inverted_min_max(self):
        with pytest.raises(ValueError, match="min_loss must not exceed max_loss"):
            EvalLossResult(avg_loss=1.5, num_batches=1, min_loss=2.0, max_loss=1.0)

    def test_init_rejects_nan_min_loss(self):
        with pytest.raises(ValueError, match="min_loss must be a finite number"):
            EvalLossResult(avg_loss=1.0, num_batches=1, min_loss=float("nan"), max_loss=2.0)

    def test_init_rejects_nan_max_loss(self):
        with pytest.raises(ValueError, match="max_loss must be a finite number"):
            EvalLossResult(avg_loss=1.0, num_batches=1, min_loss=1.0, max_loss=float("nan"))

    def test_init_allows_nan_when_zero_batches(self):
        r = EvalLossResult(
            avg_loss=float("nan"),
            num_batches=0,
            min_loss=float("nan"),
            max_loss=float("nan"),
        )
        assert r.num_batches == 0

    def test_init_allows_inf_when_zero_batches(self):
        r = EvalLossResult(
            avg_loss=float("inf"),
            num_batches=0,
            min_loss=float("inf"),
            max_loss=float("inf"),
        )
        assert r.num_batches == 0
