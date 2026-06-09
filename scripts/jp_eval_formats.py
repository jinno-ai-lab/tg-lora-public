"""Japanese LLM evaluation task formatters.

Shared prompt formatting functions for JGLUE tasks (JCommonsenseQA, JNLI, JSQuAD)
used by both PyTorch and MLX evaluation scripts.
"""
from __future__ import annotations


def format_jcommonsenseqa(item: dict) -> tuple[str, str]:
    """Format a JCommonsenseQA item into a prompt and expected answer."""
    prompt = f"質問: {item['question']}\n選択肢:\n"
    for i in range(5):
        prompt += f"- {i}: {item[f'choice{i}']}\n"
    prompt += "回答は選択肢の番号（0, 1, 2, 3, 4）のみで答えてください。\n回答: "
    return prompt, str(item["label"])


def format_jnli(item: dict) -> tuple[str, str]:
    """Format a JNLI item into a prompt and expected answer."""
    premise = item.get("premise", item.get("sentence1", ""))
    hypothesis = item.get("hypothesis", item.get("sentence2", ""))
    prompt = f"前提: {premise}\n仮説: {hypothesis}\n"
    prompt += "前提と仮説の関係は、含意（entailment）、矛盾（contradiction）、中立（neutral）のどれですか？\n"
    prompt += "回答は「含意」、「矛盾」、「中立」のいずれかで答えてください。\n回答: "

    label_map = {
        "entailment": "含意",
        "contradiction": "矛盾",
        "neutral": "中立",
        0: "含意",
        1: "矛盾",
        2: "中立",
    }
    lbl = item.get("label", "")
    if isinstance(lbl, int) and 0 <= lbl < 3:
        expected = ["含意", "矛盾", "中立"][lbl]
    else:
        expected = label_map.get(lbl, "中立")
    return prompt, expected


def format_jsquad(item: dict) -> tuple[str, str]:
    """Format a JSQuAD item into a prompt and expected answer."""
    context = item.get("context", "")
    question = item.get("question", "")
    prompt = f"文脈: {context}\n質問: {question}\n"
    prompt += "質問に対する回答を文脈から抽出して短く答えてください。\n回答: "

    answers = item.get("answers", {})
    expected = ""
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if texts:
            expected = texts[0]
    elif isinstance(answers, list) and answers:
        first_ans = answers[0]
        if isinstance(first_ans, dict):
            expected = first_ans.get("text", "")
        else:
            expected = str(first_ans)
    return prompt, expected
