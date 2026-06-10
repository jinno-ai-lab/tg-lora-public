"""Unit tests for src/data/generate_open_data.py."""

import torch
from unittest.mock import MagicMock, patch

import pytest

from src.data.generate_open_data import generate_open_data


class _BatchEncoding(dict):
    """Minimal stand-in for transformers BatchEncoding with .to() support."""

    def __init__(self, data):
        super().__init__(data)
        self._data = data

    def to(self, device):
        return self

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


@pytest.fixture
def seed_records(tmp_path):
    """Create a small seed JSONL file."""
    from src.utils.io import save_jsonl

    records = [
        {"id": "seed_000001", "prompt": "Hello, world!"},
        {"id": "seed_000002", "text": "Another prompt here"},
        {"id": "seed_000003"},
        {"prompt": "Fourth prompt"},
    ]
    seed_path = tmp_path / "seeds.jsonl"
    save_jsonl(records, str(seed_path))
    return str(seed_path)


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    tok.vocab_size = 100
    enc = _BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )
    tok.return_value = enc
    tok.decode.return_value = " generated text here"
    return tok


def _make_mock_model():
    model = MagicMock()
    output_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    model.generate.return_value = output_ids
    return model


class TestGenerateOpenData:
    @patch("src.data.generate_open_data.AutoModelForCausalLM")
    @patch("src.data.generate_open_data.AutoTokenizer")
    def test_generates_records_from_seed_prompts(
        self, mock_tok_cls, mock_model_cls, tmp_path, seed_records
    ):
        """generate_open_data reads seeds, generates completions, writes output."""
        mock_tok = _make_mock_tokenizer()
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_model = _make_mock_model()
        mock_model_cls.from_pretrained.return_value = mock_model

        output_path = str(tmp_path / "generated.jsonl")

        generate_open_data(
            model_name="fake-model",
            seed_path=seed_records,
            output_path=output_path,
            device="cpu",
            provenance=True,
        )

        from src.utils.io import load_jsonl

        results = load_jsonl(output_path)
        assert len(results) == 3
        assert all("text" in r for r in results)
        assert all("prompt" in r for r in results)
        assert all("completion" in r for r in results)
        assert all("provenance" in r for r in results)

    @patch("src.data.generate_open_data.AutoModelForCausalLM")
    @patch("src.data.generate_open_data.AutoTokenizer")
    def test_no_provenance_when_disabled(
        self, mock_tok_cls, mock_model_cls, tmp_path, seed_records
    ):
        """When provenance=False, records should not contain provenance field."""
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _make_mock_model()

        output_path = str(tmp_path / "generated_no_prov.jsonl")

        generate_open_data(
            model_name="fake-model",
            seed_path=seed_records,
            output_path=output_path,
            device="cpu",
            provenance=False,
        )

        from src.utils.io import load_jsonl

        results = load_jsonl(output_path)
        for r in results:
            assert "provenance" not in r

    @patch("src.data.generate_open_data.AutoModelForCausalLM")
    @patch("src.data.generate_open_data.AutoTokenizer")
    def test_skips_records_without_prompt(
        self, mock_tok_cls, mock_model_cls, tmp_path, seed_records
    ):
        """Records without 'prompt' or 'text' fields should be skipped."""
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _make_mock_model()

        output_path = str(tmp_path / "skip_test.jsonl")

        generate_open_data(
            model_name="fake-model",
            seed_path=seed_records,
            output_path=output_path,
            device="cpu",
        )

        from src.utils.io import load_jsonl

        results = load_jsonl(output_path)
        assert len(results) == 3

    @patch("src.data.generate_open_data.AutoModelForCausalLM")
    @patch("src.data.generate_open_data.AutoTokenizer")
    def test_sets_pad_token_when_none(
        self, mock_tok_cls, mock_model_cls, tmp_path, seed_records
    ):
        """If tokenizer has no pad_token, it should be set to eos_token."""
        mock_tok = _make_mock_tokenizer()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = _make_mock_model()

        output_path = str(tmp_path / "pad_test.jsonl")

        generate_open_data(
            model_name="fake-model",
            seed_path=seed_records,
            output_path=output_path,
            device="cpu",
        )

        assert mock_tok.pad_token == mock_tok.eos_token

    @patch("src.data.generate_open_data.AutoModelForCausalLM")
    @patch("src.data.generate_open_data.AutoTokenizer")
    def test_model_called_with_correct_generate_params(
        self, mock_tok_cls, mock_model_cls, tmp_path, seed_records
    ):
        """model.generate should be called with expected generation parameters."""
        mock_tok = _make_mock_tokenizer()
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_model = _make_mock_model()
        mock_model_cls.from_pretrained.return_value = mock_model

        output_path = str(tmp_path / "params_test.jsonl")

        generate_open_data(
            model_name="fake-model",
            seed_path=seed_records,
            output_path=output_path,
            max_new_tokens=128,
            temperature=0.5,
            top_p=0.8,
            device="cpu",
        )

        mock_model.generate.assert_called()
        gen_call = mock_model.generate.call_args
        assert gen_call.kwargs["max_new_tokens"] == 128
        assert gen_call.kwargs["temperature"] == 0.5
        assert gen_call.kwargs["top_p"] == 0.8
        assert gen_call.kwargs["do_sample"] is True

    @patch("src.data.generate_open_data.AutoModelForCausalLM")
    @patch("src.data.generate_open_data.AutoTokenizer")
    def test_device_none_auto_detects(
        self, mock_tok_cls, mock_model_cls, tmp_path, seed_records
    ):
        """When device=None, should auto-detect cuda/cpu."""
        mock_tok = _make_mock_tokenizer()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = _make_mock_model()
        mock_model_cls.from_pretrained.return_value = mock_model

        output_path = str(tmp_path / "auto_device.jsonl")

        with patch("src.utils.device.detect_device", return_value=torch.device("cpu")):
            generate_open_data(
                model_name="fake-model",
                seed_path=seed_records,
                output_path=output_path,
                device=None,
            )

        # Should have called from_pretrained with device_map="cpu"
        mock_model_cls.from_pretrained.assert_called_once()
        assert mock_model_cls.from_pretrained.call_args.kwargs["device_map"] == "cpu"
