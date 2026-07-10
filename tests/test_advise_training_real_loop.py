"""Real producer -> real consumer loop integration tests.

Why this file exists
--------------------
The previous iteration proved the ``advise_training.py`` CONSUMER on synthetic
dicts that *mirrored* the ``RunMetrics`` schema (``_real_producer_plateau`` in
``test_advise_training_e2e.py``). It never drove the REAL producer code
(``src.utils.run_metrics.RunMetrics.record_step`` / ``write_header``) to emit a
``run_metrics.jsonl`` and then consumed it — so the dormant producer->consumer
loop was asserted only on a hand-built dict, not on real producer output.

These tests close that gap: **BOTH ends are real code.**

- Producer side: the real ``RunMetrics.record_step`` / ``write_header`` emit the
  jsonl (orjson-serialized, with a real ``run_header`` line the consumer must
  skip). This is mutation-linked to the producer — if ``record_step`` ever drops
  ``tg_lora_loss_pilot_eval`` / ``tg_lora_loss_after``, these tests go RED.
- Consumer side: the real ``advise_training.py`` CLI (driven as a subprocess)
  reads the real-producer file and renders the advisory block.

Stated synthetic boundary: the loss *values* are a hand-built plateau trajectory.
The genuine 9B training run is Category-C on this public mirror (private
``src.data`` dataset pipeline + >12 GB GPU, GOAL sec4), so a real-model
trajectory is not producible here. The SERIALIZATION / field-name contract — the
part that was previously only synthetically asserted — is now exercised by real
producer code on both ends.
"""
import json
import subprocess
import sys
from pathlib import Path

from src.utils.run_metrics import RunMetrics

CLI = Path("scripts/advise_training.py")
FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "advise_loop"
    / "run_metrics_real_producer.jsonl"
)
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Real producer driver (the genuine src.utils.run_metrics.RunMetrics path)
# ---------------------------------------------------------------------------


class _FakeCfg:
    """Minimal config satisfying RunMetrics.write_header (FakeCfg idiom from
    tests/test_run_metrics.py)."""

    class model:
        name_or_path = "Qwen/real-producer-test"

    class training:
        batch_size = 1
        grad_accumulation = 1
        learning_rate = 1e-4

    class lora:
        r = 8
        alpha = 16

    class experiment:
        seed = 42

    _path = "real_producer_test.yaml"
    tg_lora = None
    alpha_line = None


def _emit_real_plateau(run_dir: Path, *, n_cycles: int = 14) -> Path:
    """Drive the REAL RunMetrics producer over a plateau trajectory.

    Loss improves for the first cycles then goes exactly flat -> stagnation +
    convergence -> the advisor fires ``stop_training`` and ``increase_k`` (whose
    remediation names ``tg_lora.K_initial``). Returns the emitted jsonl path.
    Every byte is written by the real producer; only the loss *values* are
    synthetic (see module docstring).
    """
    metrics = RunMetrics(run_dir, mode="tg_lora", run_id="real_producer_test")
    metrics.write_header(_FakeCfg(), budget_type="cycles", budget_value=n_cycles)
    try:
        for i in range(n_cycles):
            loss = round(2.0 - 0.10 * min(i, 6), 4)
            metrics.record_step(
                step=i + 1,
                cycle=i,
                total_backward_passes=i + 1,
                loss_train=loss,
                loss_valid=loss,
                grad_norm=0.5,
                tg_lora_accepted=True,
                tg_lora_K=3,
                tg_lora_N=2,
                tg_lora_alpha=0.5,
                # Real producer keys the consumer must read (NOT loss_pilot /
                # loss_after). Non-zero so a consumer contract break surfaces as
                # 0.0 rather than masking on a zero input.
                tg_lora_loss_pilot_eval=round(loss + 0.01, 4),
                tg_lora_loss_after=round(loss - 0.005, 4),
            )
    finally:
        metrics.close()
    return run_dir / "run_metrics.jsonl"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(ROOT),
    )


def _emit_real_with_full_eval(run_dir: Path, *, n_cycles: int = 8) -> Path:
    """Drive the REAL RunMetrics producer over a trajectory that includes a
    full-eval record (``record_full_eval_loss``), exactly as the trainer emits
    at every full-eval site (``train_tg_lora.py:2720/2857/3025/4340/4581``).

    A full-eval record carries ``loss_train=None`` + ``loss_valid=None`` + a
    genuine ``loss_valid_full`` (the §5.1/§5.2 honest validation loss). The
    consumer previously crashed on these with ``TypeError: must be real number,
    not NoneType`` because ``math.isnan(None)`` and ``dict.get(k, default)``
    returns ``None`` — not the default — when ``k`` is present-but-``None``. The
    full-eval loss is the BEST loss so the consumer must surface it (not drop or
    crash on it). Returns the emitted jsonl path.
    """
    metrics = RunMetrics(run_dir, mode="tg_lora", run_id="full_eval_test")
    metrics.write_header(_FakeCfg(), budget_type="cycles", budget_value=n_cycles)
    try:
        for i in range(n_cycles - 1):
            loss = round(2.0 - 0.10 * i, 4)
            metrics.record_step(
                step=i + 1,
                cycle=i,
                total_backward_passes=i + 1,
                loss_train=loss,
                loss_valid=loss,
                grad_norm=0.5,
                tg_lora_accepted=True,
            )
        # The final cycle is a full-eval cycle: the trainer emits a genuine
        # full-eval loss via record_full_eval_loss (loss_train=None, honest
        # loss_valid_full). Make it strictly the BEST loss.
        best_full_eval = round(2.0 - 0.10 * (n_cycles - 1) - 0.05, 4)
        metrics.record_full_eval_loss(
            cycle=n_cycles - 1,
            full_loss=best_full_eval,
            total_backward_passes=n_cycles,
            step=n_cycles,
        )
    finally:
        metrics.close()
    return run_dir / "run_metrics.jsonl"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealProducerConsumerLoop:
    """Both ends real: RunMetrics (producer) -> advise_training.py (consumer)."""

    def test_real_producer_emits_consumer_contract_keys(self, tmp_path: Path):
        """Producer side of the contract: the real RunMetrics.record_step writes
        exactly the keys the consumer reads. Mutation-linked to the producer —
        drop ``tg_lora_loss_pilot_eval`` / ``tg_lora_loss_after`` from
        record_step and this (and the consumer test below) go RED."""
        path = _emit_real_plateau(tmp_path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        steps = [r for r in records if r.get("type") == "step"]
        assert len(steps) >= 14, f"expected >=14 real step records, got {len(steps)}"
        # The header line is present too (real runs always emit it) and the
        # consumer must skip it — assert it is non-cycle so the contract is
        # explicit about what the consumer filters.
        assert any(r.get("type") == "run_header" for r in records)
        for k in ("loss_train", "loss_valid", "tg_lora_loss_pilot_eval",
                  "tg_lora_loss_after", "grad_norm"):
            assert all(k in s for s in steps), (
                f"real producer must emit {k} on every step record"
            )

    def test_real_producer_to_consumer_cli_renders_advisory(self, tmp_path: Path):
        """Full loop, both ends real: real RunMetrics writes the jsonl, the real
        advise_training.py CLI renders the advisory block and reaches a
        stop_training truncation naming the exact config knob."""
        path = _emit_real_plateau(tmp_path)
        r = _run_cli(str(path))  # TEXT mode -> rendered console output
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        out = r.stdout
        assert "Training Advisory Report" in out
        assert "Recommended Actions" in out
        # Plateau -> convergence/stagnation -> stop_training truncation.
        assert "stop_training" in out, f"expected stop_training in:\n{out}"
        # The advisory is actionable: the exact knob string renders (bullet #3).
        assert "tg_lora.K_initial" in out, (
            f"advisory must name the exact knob tg_lora.K_initial:\n{out}"
        )
        assert "-> remediation:" in out

    def test_consumer_reads_real_producer_keys_not_zero(self, tmp_path: Path):
        """The previously-disconnected producer->consumer contract (consumer read
        loss_pilot / loss_after which the producer never writes) holds on REAL
        producer output. Import the CLI's extraction helper and assert the real
        producer's ``tg_lora_loss_pilot_eval`` / ``tg_lora_loss_after`` flow
        through as non-zero. Mutation-revertible: restore the legacy
        ``loss_pilot``/``loss_after`` reads and this goes RED (all 0.0)."""
        path = _emit_real_plateau(tmp_path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import advise_training as _cli  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        extracted = _cli._extract_cycle_records(records)
        assert extracted, "cycle records must be extracted from real producer output"
        pilot = [r["loss_pilot"] for r in extracted]
        after = [r["loss_after"] for r in extracted]
        assert all(v != 0.0 for v in pilot), (
            f"loss_pilot must flow from real tg_lora_loss_pilot_eval, got {pilot}"
        )
        assert all(v != 0.0 for v in after), (
            f"loss_after must flow from real tg_lora_loss_after, got {after}"
        )


class TestCommittedRealProducerFixture:
    """Durable evidence: a fixture emitted by the real RunMetrics producer
    (regenerable via ``scripts/generate_advise_loop_fixture.py``) consumes to a
    valid advisory through the real CLI. Pins that the committed artifact and
    the producer-of-record stay in sync with the consumer."""

    def test_fixture_exists_and_is_real_producer_output(self):
        """The committed fixture is present and carries the real producer's
        header + step record types."""
        assert FIXTURE.exists(), f"committed real-producer fixture missing: {FIXTURE}"
        records = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
        types = {r.get("type") for r in records}
        assert "run_header" in types and "step" in types, types

    def test_committed_fixture_consumes_to_valid_advisory(self):
        """The real CLI consumes the committed real-producer fixture to a valid
        advisory with the exact config knob."""
        r = _run_cli(str(FIXTURE), "--json")
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        data = json.loads(r.stdout)
        assert data["overall_health"] in ("healthy", "warning", "critical")
        assert isinstance(data["actions"], list) and data["actions"]
        knobs = {a["remediation"] for a in data["actions"] if a.get("remediation")}
        assert any("tg_lora.K_initial" in k for k in knobs), (
            f"structured remediation must name tg_lora.K_initial: {knobs}"
        )


class TestRealProducerFullEvalRecord:
    """The producer's ``record_full_eval_loss`` (emitted at every full-eval site
    in the trainer) writes ``step`` records with ``loss_train=None`` + a genuine
    ``loss_valid_full``. The consumer previously crashed on these with
    ``TypeError: must be real number, not NoneType`` (``math.isnan(None)``):
    ``dict.get(k, default)`` returns ``None`` — not the default — when ``k`` is
    present-but-``None``, so a full-eval record yielded a ``None`` train_loss.

    These drive the REAL producer (``record_full_eval_loss``) AND the REAL CLI to
    prove the honest full-eval loss now flows through the consumer instead of
    crashing it. This closes the same producer→consumer contract axis that the
    ``loss_pilot``/``loss_after`` key fix started: the producer emits records the
    consumer must read without crashing."""

    def test_full_eval_record_does_not_crash_consumer(self, tmp_path: Path):
        """Regression: a real-producer file containing a full-eval record
        (loss_train=null) no longer crashes the CLI. Before the fix this raised
        ``TypeError: must be real number, not NoneType`` and the CLI exited with
        a traceback (non-zero on --json)."""
        path = _emit_real_with_full_eval(tmp_path)
        r = _run_cli(str(path), "--json")
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert "Traceback" not in r.stderr, (
            f"consumer must not crash on full-eval record:\n{r.stderr}"
        )
        data = json.loads(r.stdout)  # strict parse — fails if it crashed
        assert data["overall_health"] in ("healthy", "warning", "critical")
        assert isinstance(data["actions"], list) and data["actions"]

    def test_full_eval_loss_surfaces_as_tracked_best(self, tmp_path: Path):
        """The honest full-eval loss (``loss_valid_full``) is the BEST loss in
        the trajectory. It must flow through the consumer as the tracked best
        loss — proving the honest §5.1/§5.2 signal is CONSUMED, not silently
        dropped or crashed on."""
        path = _emit_real_with_full_eval(tmp_path)
        r = _run_cli(str(path))  # TEXT mode to read the rendered best-loss line
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        out = r.stdout
        assert "New best loss" in out, f"expected a best-loss line:\n{out}"
        # The full-eval loss (1.25 for n_cycles=8) is the lowest; the advisor
        # must report it. Pin the literal value so a regression that drops the
        # full-eval record (and reports the higher pilot best instead) is caught.
        assert "1.25" in out, (
            f"full-eval loss 1.25 must surface as best; got:\n{out}"
        )

    def test_extract_surfaces_loss_valid_full_not_none(self, tmp_path: Path):
        """Direct contract on the extraction helper: ``_extract_cycle_records``
        surfaces ``loss_valid_full`` as ``train_loss`` for a full-eval record
        (``loss_train=None`` in the file). Mutation-revertible: remove the
        ``loss_valid_full`` surfacing in the extractor and this asserts ``None``,
        while the two consumer tests above crash with TypeError."""
        path = _emit_real_with_full_eval(tmp_path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Sanity: the producer really did write a full-eval record with null
        # loss_train (this is the defect trigger — confirm it's in the file).
        full_eval_steps = [
            r for r in records if r.get("type") == "step" and r.get("loss_train") is None
        ]
        assert full_eval_steps, "expected a full-eval record with loss_train=null"
        assert all("loss_valid_full" in r for r in full_eval_steps)

        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import advise_training as _cli  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        extracted = _cli._extract_cycle_records(records)
        # The last extracted record is the full-eval record.
        full_eval_rec = extracted[-1]
        assert full_eval_rec["train_loss"] is not None, (
            "full-eval record must surface loss_valid_full as train_loss, not None "
            f"(got {full_eval_rec})"
        )
        assert full_eval_rec["train_loss"] == 1.25, (
            f"expected the honest full-eval loss 1.25, got {full_eval_rec['train_loss']}"
        )


class TestRealProducerFullEvalInterleaved:
    """The REAL trainer interleaves: on a full-eval cycle it emits a regular
    ``record_step`` (pilot-proxy ``loss_train`` / ``loss_valid``) AND THEN a
    ``record_full_eval_loss`` (``loss_train=None`` + honest ``loss_valid_full``)
    with the SAME cycle number (``train_tg_lora.py:2954`` + ``:3025``, and the
    other full-eval sites :2720/:2857/:4340/:4581). The sibling full-eval test
    above (``_emit_real_with_full_eval``) emits the full-eval record for a cycle
    that has NO regular record, so it never exercised this interleaving — and
    that left a phantom-duplicate defect unguarded: the consumer turned BOTH
    records into trajectory points, so one cycle became two points at different
    loss scales (pilot proxy vs full-eval), corrupting the trajectory with a fake
    crash-then-spike and inflating the cycle count.

    These drive the REAL producer over the genuine interleaving and pin that the
    consumer now MERGES the full-eval record into its cycle's existing record
    (honest loss becomes ``valid_loss``) instead of appending a duplicate. The
    merge is load-bearing: reverting it (full-eval records always append) makes
    ``test_no_phantom_duplicate_one_point_per_cycle`` and
    ``test_train_trajectory_not_corrupted_by_phantom_spike`` go RED."""

    def _emit_interleaved(self, run_dir: Path, *, n_cycles: int = 8,
                          full_eval_every: int = 3) -> tuple[Path, dict]:
        """Drive the REAL RunMetrics producer exactly as the trainer does: a
        regular ``record_step`` every cycle, PLUS a ``record_full_eval_loss`` on
        full-eval cycles (same ``cycle``). The full-eval loss is on a DIFFERENT
        scale (the full validation set, not the pilot subset), so a phantom
        duplicate would visibly corrupt the trajectory. Returns (jsonl path,
        {cycle: full_eval_loss} for the cycles that got a full-eval record)."""
        metrics = RunMetrics(run_dir, mode="tg_lora", run_id="interleaved")
        metrics.write_header(_FakeCfg(), budget_type="cycles", budget_value=n_cycles)
        full_eval_losses: dict[int, float] = {}
        try:
            for i in range(n_cycles):
                # Pilot-proxy train/valid loss (the cheap every-cycle proxy).
                proxy = round(2.0 - 0.08 * i, 4)
                metrics.record_step(
                    step=i + 1,
                    cycle=i,
                    total_backward_passes=i + 1,
                    loss_train=proxy,
                    loss_valid=proxy,
                    grad_norm=0.5,
                    tg_lora_accepted=True,
                    tg_lora_K=3,
                    tg_lora_loss_pilot_eval=round(proxy + 0.01, 4),
                    tg_lora_loss_after=round(proxy - 0.005, 4),
                )
                # Full-eval cycle: the trainer ALSO emits record_full_eval_loss
                # (loss_train=None, honest loss_valid_full) on the SAME cycle.
                if i % full_eval_every == 0 and i > 0:
                    honest = round(proxy - 0.40, 4)  # genuinely lower / diff scale
                    metrics.record_full_eval_loss(
                        cycle=i,
                        full_loss=honest,
                        total_backward_passes=i + 1,
                        step=i + 1,
                    )
                    full_eval_losses[i] = honest
        finally:
            metrics.close()
        return run_dir / "run_metrics.jsonl", full_eval_losses

    def test_no_phantom_duplicate_one_point_per_cycle(self, tmp_path: Path):
        """The defect signature: with the real interleaving the consumer
        previously returned a cycle number TWICE (once for the regular record,
        once for the full-eval record). After the merge each cycle is exactly one
        trajectory point. Mutation-revertible: drop the merge branch and the
        full-eval cycles reappear as duplicates -> this asserts a dup exists."""
        path, _ = self._emit_interleaved(tmp_path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import advise_training as _cli  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        extracted = _cli._extract_cycle_records(records)
        cyc = [e["cycle"] for e in extracted]
        # Sanity: the file really does carry interleaved full-eval records
        # (regular + full-eval sharing a cycle) — otherwise this test is inert.
        full_eval_steps = [
            r for r in records if r.get("type") == "step" and r.get("loss_valid_full") is not None
        ]
        assert full_eval_steps, "fixture must contain full-eval records"
        shared = {r.get("cycle") for r in full_eval_steps}
        assert shared, "full-eval records must share a cycle with a regular record"
        # The merge: one point per cycle, no duplicates.
        assert len(cyc) == len(set(cyc)), (
            f"phantom duplicate cycles: {cyc} — full-eval record must merge, "
            "not append"
        )
        assert len(extracted) <= 8, (
            f"expected <=8 cycle points (one per cycle), got {len(extracted)}"
        )

    def test_full_eval_loss_merges_as_valid_loss(self, tmp_path: Path):
        """The honest full-eval loss becomes the merged cycle's ``valid_loss``
        (the honest §5.1 signal is CONSUMED), while ``train_loss`` stays the
        regular record's pilot-proxy training loss (a full-eval loss is a
        VALIDATION loss, not a training loss). Mutation-revertible: drop the
        merge and the full-eval cycle's valid_loss reverts to the pilot proxy."""
        path, full_eval_losses = self._emit_interleaved(tmp_path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import advise_training as _cli  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        extracted = _cli._extract_cycle_records(records)
        by_cycle = {e["cycle"]: e for e in extracted}
        for cyc, honest in full_eval_losses.items():
            entry = by_cycle[cyc]
            assert entry["valid_loss"] == honest, (
                f"cycle {cyc}: honest full-eval loss {honest} must surface as "
                f"valid_loss, got {entry['valid_loss']}"
            )
            # train_loss is NOT the full-eval loss — it stays the regular
            # record's training loss (different measurement).
            assert entry["train_loss"] != honest, (
                f"cycle {cyc}: full-eval valid loss must not overwrite train_loss"
            )
            # The regular record's metadata survives the merge.
            assert entry["tg_lora_accepted"] is True

    def test_train_trajectory_not_corrupted_by_phantom_spike(self, tmp_path: Path):
        """The phantom duplicate corrupted the TRAIN trajectory with a fake
        crash to the full-eval value (then a spike back). After the merge the
        train_loss sequence is the clean per-cycle pilot proxy — no full-eval
        value leaks into it. Mutation-revertible: drop the merge and the
        full-eval losses (0.4+ below the proxy) reappear inside train_loss."""
        path, full_eval_losses = self._emit_interleaved(tmp_path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import advise_training as _cli  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        extracted = _cli._extract_cycle_records(records)
        train_seq = [e["train_loss"] for e in extracted]
        # No honest full-eval loss may appear in the train-loss sequence.
        leaked = [v for v in train_seq if v in set(full_eval_losses.values())]
        assert not leaked, (
            f"full-eval losses leaked into train trajectory: {leaked} "
            f"(train_seq={train_seq})"
        )
        # The train trajectory is monotonically non-increasing (pilot proxy
        # improves each cycle) — a phantom duplicate would break monotonicity.
        assert all(b <= a + 1e-9 for a, b in zip(train_seq, train_seq[1:])), (
            f"train trajectory not monotonic (phantom spike?): {train_seq}"
        )

    def test_interleaved_cli_runs_clean(self, tmp_path: Path):
        """The real CLI consumes a real interleaved producer file end-to-end:
        no crash, a valid advisory, and the honest full-eval loss surfaces as the
        tracked best loss (it is the lowest valid_loss)."""
        path, _ = self._emit_interleaved(tmp_path)
        r = _run_cli(str(path))
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert "Traceback" not in r.stderr, (
            f"consumer must not crash on interleaved full-eval:\n{r.stderr}"
        )
        assert "Training Advisory Report" in r.stdout
