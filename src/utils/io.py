import os
from pathlib import Path

import orjson


def _atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Atomically write *data* bytes to *path* — temp-in-same-dir + ``os.replace``.

    Mirrors :func:`src.utils.atomic_save._atomic_torch_save` (the sole
    ``torch.save`` publish point, pinned by ``test_no_bare_torch_save_in_src``):
    write to a PID-suffixed temp in the SAME directory, then rename into place
    with :func:`os.replace` — atomic on POSIX same-filesystem — so the
    destination either fully reflects the new content or is left at its prior
    value, NEVER torn. A kill mid-write (OOM-kill, SIGINT, a host recycling a
    worktree, or an operator harvesting the file during a multi-hour 9B run's
    final deposit) therefore can never leave a half-written JSON that fails to
    parse or, worse, parses to a silently-truncated verdict.

    The ``except BaseException`` cleanup removes the orphan temp on ANY exit
    (including ``KeyboardInterrupt`` / ``SystemExit``, which ``except Exception``
    would miss) and re-raises the original — see :mod:`tests.test_atomic_save`
    for the BaseException-specific pin. This is the JSON/text analogue of the
    torch-artifact atomicity guarantee, and the sole publish point every
    :func:`save_json` / :func:`save_jsonl` write (and the 9B deposit / run-log
    writes) routes through — so a JSON artifact survives a mid-write fault
    exactly as a checkpoint does.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.parent / f"{p.name}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, p)
    except BaseException:
        # Never publish a partial file: if the rename did not land, the prior
        # file (if any) must remain intact. Best-effort-remove the orphan temp
        # so a crashed run does not litter the output directory. ``BaseException``
        # (not ``Exception``) so a Ctrl-C / SystemExit mid-write also cleans up
        # rather than leaving a ``.tmp.<pid>`` behind on every interrupt.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _atomic_write_text(
    path: str | Path, data: str, *, encoding: str = "utf-8"
) -> None:
    """Atomically write *data* text to *path* (UTF-8 by default).

    Thin text wrapper over :func:`_atomic_write_bytes`; see that docstring for
    the atomicity contract. Used by the 9B deposit + run-log writes so the
    citable §4 verdict and its loss-curve witness survive a mid-write kill.
    """
    _atomic_write_bytes(path, data.encode(encoding))


def save_json(obj: dict | list, path: str | Path) -> None:
    _atomic_write_bytes(path, orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def load_json(path: str | Path) -> dict | list:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_jsonl(records: list[dict], path: str | Path) -> None:
    # Buffer the full payload then publish atomically — a streaming write could
    # be interrupted mid-file and leave a torn (possibly parseable-to-a-
    # truncation) JSONL artifact. The on-disk bytes are identical to the prior
    # per-record loop form.
    data = b"".join(orjson.dumps(rec) + b"\n" for rec in records)
    _atomic_write_bytes(path, data)


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    return records
