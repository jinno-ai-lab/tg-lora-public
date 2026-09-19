import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.provenance import create_provenance
from src.utils.io import load_jsonl, save_jsonl

logger = logging.getLogger("tg-lora")


def generate_open_data(
    model_name: str,
    seed_path: str,
    output_path: str,
    max_seq_len: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str | None = None,
    provenance: bool = True,
) -> None:
    if device is None:
        from src.utils.device import detect_device
        device = str(detect_device())

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    seed_records = load_jsonl(seed_path)

    generated = []
    for i, seed in enumerate(tqdm(seed_records, desc="Generating")):
        prompt = seed.get("prompt", seed.get("text", ""))
        if not prompt:
            continue

        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        completion = tokenizer.decode(
            outputs[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        text = prompt + completion

        record = {
            "text": text,
            "prompt": prompt,
            "completion": completion,
        }

        if provenance:
            record["provenance"] = create_provenance(
                seed_id=seed.get("id", f"seed_{i:06d}"),
                generator_model=model_name,
                source_type="open_model_generated",
            )

        generated.append(record)

    save_jsonl(generated, output_path)
    logger.info(f"Generated {len(generated)} records -> {output_path}")
