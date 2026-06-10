from pathlib import Path

import orjson


def save_json(obj: dict | list, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def load_json(path: str | Path) -> dict | list:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_jsonl(records: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b"\n")


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    return records
