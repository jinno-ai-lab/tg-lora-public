import logging

import torch
from torch.utils.data import Dataset

from src.data.schema import validate_records
from src.utils.io import load_jsonl

logger = logging.getLogger(__name__)

_CHATML_ASSISTANT_MARKER = "<|im_start|>assistant\n"


def _resolve_text(rec: dict) -> str:
    text = rec.get("text", "")
    if not text:
        text = rec.get("prompt", "") + rec.get("completion", "")
    return text


def _prompt_char_boundary(rec: dict, text: str) -> int:
    marker_idx = text.find(_CHATML_ASSISTANT_MARKER)
    if marker_idx != -1:
        return marker_idx + len(_CHATML_ASSISTANT_MARKER)

    prompt_text = rec.get("prompt")
    if prompt_text:
        return len(prompt_text.rstrip())

    return 0


def _token_count(tokenizer, text: str, max_seq_len: int) -> int:
    if not text:
        return 0
    enc = tokenizer(
        text,
        max_length=max_seq_len,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    return int(enc["input_ids"].shape[-1])


def _prompt_token_count(rec: dict, text: str, tokenizer, max_seq_len: int) -> int:
    marker_idx = text.find(_CHATML_ASSISTANT_MARKER)
    if marker_idx != -1:
        return _token_count(tokenizer, text[: marker_idx + len(_CHATML_ASSISTANT_MARKER)], max_seq_len)

    prompt_text = rec.get("prompt")
    if prompt_text:
        return _token_count(tokenizer, prompt_text.rstrip(), max_seq_len)

    return 0


def encode_record(
    rec: dict,
    tokenizer,
    max_seq_len: int = 512,
    *,
    train_on_prompt: bool = False,
) -> dict[str, torch.Tensor]:
    text = _resolve_text(rec)

    encode_kwargs = {
        "max_length": max_seq_len,
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }
    try:
        enc = tokenizer(text, return_offsets_mapping=True, **encode_kwargs)
    except TypeError:
        enc = tokenizer(text, **encode_kwargs)

    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    if not train_on_prompt:
        prompt_boundary = _prompt_char_boundary(rec, text)
        if prompt_boundary > 0 and "offset_mapping" in enc:
            offsets = enc["offset_mapping"].squeeze(0)
            prompt_mask = (attention_mask == 1) & (offsets[:, 1] <= prompt_boundary)
            labels[prompt_mask] = -100
        else:
            prompt_tokens = _prompt_token_count(rec, text, tokenizer, max_seq_len)
            if prompt_tokens > 0:
                labels[: min(prompt_tokens, labels.shape[0])] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def analyze_record_supervision(
    rec: dict,
    tokenizer,
    max_seq_len: int = 512,
    *,
    train_on_prompt: bool = False,
) -> dict[str, int | float | bool]:
    item = encode_record(
        rec,
        tokenizer,
        max_seq_len,
        train_on_prompt=train_on_prompt,
    )
    attention_mask = item["attention_mask"] == 1
    labels = item["labels"]
    total_tokens = int(attention_mask.sum().item())
    supervised_tokens = int(((labels != -100) & attention_mask).sum().item())
    prompt_tokens = int(((labels == -100) & attention_mask).sum().item())
    return {
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "prompt_tokens": prompt_tokens,
        "all_masked": supervised_tokens == 0,
        "prompt_token_ratio": (prompt_tokens / total_tokens) if total_tokens > 0 else 0.0,
    }


class LoraDataset(Dataset):
    _CHATML_ASSISTANT_MARKER = _CHATML_ASSISTANT_MARKER

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_seq_len: int = 512,
        source_path: str | None = None,
        *,
        train_on_prompt: bool = False,
    ):
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a non-empty list")
        self._all_masked_warned = False
        if tokenizer is None:
            raise ValueError("tokenizer must not be None")
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive int, got {max_seq_len}")
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.source_path = source_path
        self.train_on_prompt = train_on_prompt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        item = encode_record(
            rec,
            self.tokenizer,
            self.max_seq_len,
            train_on_prompt=self.train_on_prompt,
        )
        labels = item["labels"]

        if not self._all_masked_warned and (labels == -100).all():
            logger.warning(
                "Sample %d has all labels masked (-100) after prompt masking. "
                "Long prompt + truncation can cause NaN loss.",
                idx,
            )
            self._all_masked_warned = True

        return item

    def _prompt_char_boundary(self, rec: dict, text: str) -> int:
        return _prompt_char_boundary(rec, text)

    def _prompt_token_count(self, rec: dict, text: str) -> int:
        return _prompt_token_count(rec, text, self.tokenizer, self.max_seq_len)

    def _token_count(self, text: str) -> int:
        return _token_count(self.tokenizer, text, self.max_seq_len)


def _drop_all_masked_records(
    records: list[dict],
    tokenizer,
    max_seq_len: int,
    *,
    train_on_prompt: bool = False,
    source_path: str | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    dropped = 0
    for rec in records:
        supervision = analyze_record_supervision(
            rec,
            tokenizer,
            max_seq_len,
            train_on_prompt=train_on_prompt,
        )
        if supervision["all_masked"]:
            dropped += 1
            continue
        filtered.append(rec)

    if dropped:
        logger.warning(
            "Dropped %d all-masked records from %s at max_seq_len=%d",
            dropped,
            source_path or "<in-memory>",
            max_seq_len,
        )
    return filtered


def load_dataset(
    path: str,
    tokenizer,
    max_seq_len: int = 512,
    *,
    validate: bool = False,
    train_on_prompt: bool = False,
) -> LoraDataset:
    records = load_jsonl(path)
    if validate:
        records, _summary = validate_records(records)
    if not train_on_prompt:
        records = _drop_all_masked_records(
            records,
            tokenizer,
            max_seq_len,
            train_on_prompt=train_on_prompt,
            source_path=path,
        )
    return LoraDataset(
        records,
        tokenizer,
        max_seq_len,
        source_path=path,
        train_on_prompt=train_on_prompt,
    )
