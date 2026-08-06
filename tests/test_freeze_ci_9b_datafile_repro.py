"""§7 independent-reproducibility artifact pin: the 9B verdict setup reproduces
from the **public mirror alone** via the offline ``--data-file`` rail — no live
dataset fetch, no private ``src.data``.

Why this file exists
--------------------
The §4 arc is DONE in this checkout (both citable faithful TIES harvested via
the *streaming* ``datasets.load_dataset`` path; SHIP landed ``6c75fda``). But
"citable" ≠ "independently reproducible" (GOAL §7): a deposit produced by
streaming ``databricks/databricks-dolly-15k`` at run time hinges on a live HF
hub fetch of a dataset the hub could revise or withhold — so a reproducer with
only this public mirror (and a firewall / offline host) could NOT boot the
verdict run. Commit ``079e8f1`` added the ``--data-file`` ingestion rail to close
exactly that gap: a local Dolly-shaped JSONL is read by the SAME honesty guards
as the streaming path, so a local dump is drop-in equivalent (same capped +
seeded-shuffled record set).

This test pins the **fired evidence** that the rail works end-to-end at real 9B
target scale — a citable self-contained §7 artifact (AI-Hub feedback option A):

* ``tests/fixtures/freeze_validloss_ci_9b_datafile_repro.json`` — a REAL
  ``Qwen/Qwen3.5-9B`` QLoRA A/B deposit produced via ``--data-file`` on the
  public mirror's RTX 3060 (``proxy_scale=False``, ``reduced_budget=True`` — a
  reproducibility smoke, NOT a verdict re-derivation; the §4 verdicts stay the
  harvested streaming deposits).
* ``tests/fixtures/freeze_validloss_ci_9b_datafile_repro_ledger.jsonl`` — the
  resume ledger whose fingerprint stamps ``data_file = <local path>``: the
  machine-verifiable proof the run ingested a LOCAL dump (offline source), the
  property the streaming-path ledgers do NOT carry (their fingerprint predates
  ``079e8f1`` and records no ``data_file``).

This REFUTES, with fired-run evidence, the "9B run is structurally DATA-blocked"
narrative that drove three consecutive ``no-op-with-blocker`` iterations
(TASK-0236 / 0237 / 0238): data + model + GPU were all available, and the
``--data-file`` run booted and deposited. The structural half of that refutation
(producer is src.data-free) lives in
``test_9b_verdict_producer_self_contained.py``; THIS file is the fired-artifact
half.

Scope
-----
Read-only assertions over the two committed fixtures. It does NOT re-fire the
run (that is a GPU smoke, opt-in via the procedure documented in the matching
TASK note). If the fixtures are regenerated, the assertions still hold as long
as the run used ``--data-file`` against a real 9B — which is the whole point.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEPOSIT = FIXTURES / "freeze_validloss_ci_9b_datafile_repro.json"
LEDGER = FIXTURES / "freeze_validloss_ci_9b_datafile_repro_ledger.jsonl"

# The verdict vocabulary result_to_json emits (kept loose — the §7 artifact's
# job is to prove the rail deposits a WELL-FORMED real-9B A/B, not to assert a
# specific reduced-budget verdict magnitude, which is not citable as §4).
_VERDICTS = {"SURPASSES", "TIES", "INCONCLUSIVE", "DEGRADES", "SURPASSED_BY"}


def _load_deposit() -> dict:
    return json.loads(DEPOSIT.read_text(encoding="utf-8"))


def _ledger_fingerprints() -> list[dict]:
    """Every ``fingerprint`` object stamped across the resume-ledger entries."""
    fps: list[dict] = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        fp = entry.get("fingerprint")
        if isinstance(fp, dict):
            fps.append(fp)
    return fps


def test_datafile_repro_deposit_is_a_real_9b_target_scale_ab() -> None:
    """The §7 artifact is a real ``Qwen/Qwen3.5-9B`` A/B deposit (not a proxy,
    not a stub): real target scale, both arms populated, a well-formed verdict.

    This is the fired evidence that the offline ``--data-file`` rail produces a
    citable real-scale deposit — the run booted end-to-end on the public mirror.
    """
    deposit = _load_deposit()
    assert deposit["model"] == "Qwen/Qwen3.5-9B", (
        "§7 artifact must be a real 9B run, not a proxy/stub"
    )
    assert deposit["proxy_scale"] is False, (
        "§7 artifact must be real target scale (proxy_scale=False)"
    )
    assert deposit["citable_as_target_scale"] is True
    assert deposit["reduced_budget"] is True, (
        "§7 reproducibility smoke is reduced-budget by design (NOT a §4 verdict)"
    )
    assert deposit["dataset"] == "databricks/databricks-dolly-15k"
    # A real A/B: both arms ran and emitted per-seed valid_loss samples.
    n_cand = deposit["n_candidate"]
    n_surr = deposit["n_surrogate"]
    assert n_cand >= 1 and n_surr >= 1
    assert len(deposit["candidate_losses"]) == n_cand
    assert len(deposit["surrogate_losses"]) == n_surr
    assert all(isinstance(x, (int, float)) for x in deposit["candidate_losses"])
    assert all(isinstance(x, (int, float)) for x in deposit["surrogate_losses"])
    assert deposit["verdict"] in _VERDICTS, deposit["verdict"]


def test_datafile_repro_ledger_fingerprints_the_offline_source() -> None:
    """The §7 load-bearing property: the ledger fingerprint stamps a non-null
    ``data_file`` — the run ingested a LOCAL Dolly dump (offline source), so it
    is reproducible from the public mirror alone with no live dataset fetch.

    The streaming-path ledgers (``freeze_validloss_ci_9b_full_ledger.jsonl`` et
    al.) record NO ``data_file`` (their fingerprint predates ``079e8f1``); this
    ledger MUST, because the artifact's entire purpose is offline reproduction.
    """
    fps = _ledger_fingerprints()
    assert fps, "resume ledger must stamp at least one fingerprint"
    # The run-config HEADER fingerprint carries the full config incl. data_file
    # (the per-arm entries stamp a minimal arm-identity fingerprint without it —
    # that is by design: the header establishes the run's ingestion source).
    # ... and the run that produced THIS artifact used a local dump (non-null).
    offline = [fp for fp in fps if fp.get("data_file")]
    assert offline, (
        "no ledger fingerprint stamps a non-null data_file — the §7 artifact "
        "was not produced via the offline --data-file rail"
    )
    df = offline[0]["data_file"]
    assert isinstance(df, str) and df.endswith(".jsonl"), (
        f"data_file must be a local JSONL path, got {df!r}"
    )
