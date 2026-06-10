import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from src.eval.eval_format import _infer_required_keys, eval_format_compliance
from src.eval.eval_task import eval_task_performance


class _TinyGenModel(nn.Module):
    """Minimal model that returns logits compatible with model.generate()."""

    def __init__(self, vocab_size=10):
        super().__init__()
        self.linear = nn.Linear(4, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = input_ids.float()
        logits = self.linear(x)
        return type("Out", (), {"logits": logits, "loss": torch.tensor(0.0)})()

    def generate(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        max_new = kwargs.get("max_new_tokens", 4)
        batch_size = input_ids.shape[0]
        # Return input_ids + some new tokens
        new_tokens = torch.randint(0, self.vocab_size, (batch_size, max_new))
        return torch.cat([input_ids, new_tokens], dim=1)

    def eval(self):
        nn.Module.eval(self)
        return self

    def train(self, mode=True):
        nn.Module.train(self, mode)
        return self


class _TokenizedOutput(dict):
    """Dict subclass with .to() for mocking tokenizer output."""

    def to(self, device):
        return self


def _make_tokenizer():
    tok = MagicMock()

    def _call(text, **kwargs):
        return _TokenizedOutput(
            {
                "input_ids": torch.randint(0, 10, (1, 4)),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        )

    tok.side_effect = _call
    tok.decode = MagicMock(return_value='{"answer": "42"}')
    return tok


def _write_jsonl(records, path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---- eval_task_performance tests ----


def test_eval_task_performance_runs_with_metric_fn():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()

    records = [
        {"prompt": "What is 2+2?", "completion": "4"},
        {"prompt": "What is 3+3?", "completion": "6"},
    ]

    call_log = []

    def my_metric(expected, predicted):
        call_log.append((expected, predicted))
        return 1.0 if expected == predicted else 0.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        _write_jsonl(records, f.name)
        test_path = f.name

    try:
        with patch("src.eval.eval_task.load_jsonl", return_value=records):
            result = eval_task_performance(
                model, tokenizer, test_path, device="cpu", metric_fn=my_metric
            )

        assert result["total"] == 2
        assert "mean_score" in result
        assert "scores" in result
        assert len(result["scores"]) == 2
        assert len(call_log) == 2
    finally:
        os.unlink(test_path)


def test_eval_task_performance_without_metric_fn():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()

    records = [{"prompt": "Hello", "completion": "World"}]

    with patch("src.eval.eval_task.load_jsonl", return_value=records):
        result = eval_task_performance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert result["total"] == 1
    assert "mean_score" not in result


def test_eval_task_performance_skips_empty_prompts():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()

    records = [
        {"prompt": "", "completion": "empty"},
        {"prompt": "valid", "completion": "answer"},
    ]

    with patch("src.eval.eval_task.load_jsonl", return_value=records):
        result = eval_task_performance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert result["total"] == 1  # only the valid prompt


def test_eval_task_performance_saves_output():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()

    records = [{"prompt": "test", "completion": "output"}]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        output_path = f.name

    try:
        with patch("src.eval.eval_task.load_jsonl", return_value=records):
            with patch("src.eval.eval_task.save_jsonl") as mock_save:
                result = eval_task_performance(
                    model,
                    tokenizer,
                    "dummy.jsonl",
                    output_path=output_path,
                    device="cpu",
                )
                mock_save.assert_called_once()

        assert result["total"] == 1
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


# ---- eval_format_compliance tests ----


def test_eval_format_compliance_valid_json():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()
    tokenizer.decode = MagicMock(return_value='{"answer": "42", "confidence": 0.9}')

    records = [
        {"prompt": "Q1?", "completion": '{"answer": "42", "confidence": 0.9}'},
        {"prompt": "Q2?", "completion": '{"answer": "99", "confidence": 0.8}'},
    ]

    with patch("src.eval.eval_format.load_jsonl", return_value=records):
        result = eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert result["total"] == 2
    assert result["valid_json"] == 2
    assert result["json_rate"] == 1.0


def test_eval_format_compliance_invalid_json():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()
    tokenizer.decode = MagicMock(return_value="not valid json at all")

    records = [
        {"prompt": "Q1?", "completion": "something"},
    ]

    with patch("src.eval.eval_format.load_jsonl", return_value=records):
        result = eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert result["total"] == 1
    assert result["valid_json"] == 0
    assert result["json_rate"] == 0.0


def test_eval_format_compliance_empty_records():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()

    with patch("src.eval.eval_format.load_jsonl", return_value=[]):
        result = eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert result["total"] == 0
    assert result["json_rate"] == 0.0
    assert result["key_compliance_rate"] == 0.0


def test_eval_format_compliance_mixed_json():
    model = _TinyGenModel()

    call_count = [0]

    def alternating_decode(token_ids, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return '{"answer": "valid"}'
        return "not json"

    tokenizer = _make_tokenizer()
    tokenizer.decode = alternating_decode

    records = [
        {"prompt": "Q1?", "completion": '{"answer": "a"}'},
        {"prompt": "Q2?", "completion": "bad"},
    ]

    with patch("src.eval.eval_format.load_jsonl", return_value=records):
        result = eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert result["total"] == 2
    assert result["valid_json"] == 1
    assert result["json_rate"] == 0.5


# ---- _infer_required_keys tests ----


def test_infer_required_keys_common_keys():
    records = [
        {"completion": '{"answer": "a", "confidence": 0.9}'},
        {"completion": '{"answer": "b", "confidence": 0.8}'},
        {"completion": '{"answer": "c", "confidence": 0.7}'},
    ]

    keys = _infer_required_keys(records)

    assert "answer" in keys
    assert "confidence" in keys


def test_infer_required_keys_below_threshold():
    records = [
        {"completion": '{"answer": "a", "rare_key": 1}'},
        {"completion": '{"answer": "b"}'},
        {"completion": '{"answer": "c"}'},
    ]

    keys = _infer_required_keys(records)

    assert "answer" in keys
    assert "rare_key" not in keys  # appears in 1/3 (< 50%) of records


def test_infer_required_keys_empty_records():
    keys = _infer_required_keys([])
    assert keys == []


def test_infer_required_keys_uses_output_fallback():
    records = [
        {"output": '{"result": "x"}'},
        {"output": '{"result": "y"}'},
    ]

    keys = _infer_required_keys(records)

    assert "result" in keys


def test_infer_required_keys_handles_invalid_json():
    records = [
        {"completion": "not json"},
        {"completion": '{"answer": "a"}'},
        {"completion": '{"answer": "b"}'},
    ]

    keys = _infer_required_keys(records)

    assert "answer" in keys


def test_eval_format_compliance_skips_empty_prompts():
    model = _TinyGenModel()
    tokenizer = _make_tokenizer()
    tokenizer.decode = MagicMock(return_value='{"answer": "42"}')

    records = [
        {"prompt": "", "completion": "should be skipped"},
        {"prompt": "Valid question?", "completion": '{"answer": "yes"}'},
        {"prompt": "  ", "completion": "also has whitespace only prompt"},
    ]

    with patch("src.eval.eval_format.load_jsonl", return_value=records):
        result = eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")

    # Empty string "" is skipped, but whitespace "  " is truthy and evaluated
    assert result["total"] == 2


def test_eval_format_compliance_restores_model_state_on_exception():
    """Regression: model.train() must be called even if evaluation raises."""
    model = _TinyGenModel()
    model.train()
    tokenizer = _make_tokenizer()

    records = [{"prompt": "Q?", "completion": "a"}]

    def _raise(*a, **kw):
        raise RuntimeError("boom")

    with patch("src.eval.eval_format.load_jsonl", return_value=records):
        with patch.object(tokenizer, "__call__", side_effect=_raise):
            try:
                eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")
            except RuntimeError:
                pass

    assert model.training is True


def test_eval_task_performance_restores_model_state_on_exception():
    """Regression: model.train() must be called even if evaluation raises."""
    model = _TinyGenModel()
    model.train()
    tokenizer = _make_tokenizer()

    records = [{"prompt": "Q?", "completion": "a"}]

    def _raise(*a, **kw):
        raise RuntimeError("boom")

    with patch("src.eval.eval_task.load_jsonl", return_value=records):
        with patch.object(tokenizer, "__call__", side_effect=_raise):
            try:
                eval_task_performance(model, tokenizer, "dummy.jsonl", device="cpu")
            except RuntimeError:
                pass

    assert model.training is True


def test_eval_format_compliance_preserves_eval_mode():
    model = _TinyGenModel()
    model.eval()
    tokenizer = _make_tokenizer()
    tokenizer.decode = MagicMock(return_value='{"answer": "42"}')

    records = [{"prompt": "Valid question?", "completion": '{"answer": "yes"}'}]

    with patch("src.eval.eval_format.load_jsonl", return_value=records):
        eval_format_compliance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert model.training is False


def test_eval_task_performance_preserves_eval_mode():
    model = _TinyGenModel()
    model.eval()
    tokenizer = _make_tokenizer()
    tokenizer.decode = MagicMock(return_value="answer")

    records = [{"prompt": "Valid question?", "completion": "answer"}]

    with patch("src.eval.eval_task.load_jsonl", return_value=records):
        eval_task_performance(model, tokenizer, "dummy.jsonl", device="cpu")

    assert model.training is False
