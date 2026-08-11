"""Run-verify test for the committed ``artifact_manifest.json``.

This is the portable structural-integrity guard for the §4 evidence base: it
recomputes the live manifest from ``tests/fixtures`` and diffs it against the
committed one. Unlike the AI-Hub-co-located run-verify template (which skips in
CI / a plain checkout, leaving only a weak structural fallback), this test is
**pure-local** — no GPU, no network, no AI-Hub co-location — so it runs
everywhere and is the first line of defense against evidence-byte drift.

Scope is structural (``path`` / ``sha256`` / ``size``) by design: content /
semantic axes are owned by :mod:`src.tg_lora.freeze_evidence_hash` (the in-band
hash over measurement keys), and re-deriving them here would be circular.

Every drift class is mutation-proven on a throwaway copy of the fixtures dir so
the real evidence bytes are never mutated by the suite.

Producer loading
----------------
The producer is loaded by **file path** (``importlib``), NOT via
``from scripts.generate_artifact_manifest import ...``. This worktree co-exists
with ``/home/jinno/ai-hub/scripts/__init__.py`` — a *regular* package sitting on
``PYTHONPATH=/home/jinno/ai-hub`` — which shadows the worktree's ``scripts/``
directory (the worktree ships no ``scripts/__init__.py``, so ``import scripts``
resolves to the AI-Hub copy and terminates the namespace search there, breaking
every ``from scripts.* import``). File-path loading is shadow-proof and mirrors
how the producer actually runs (``python scripts/generate_artifact_manifest.py``),
so this test stays green both in a plain checkout and inside the AI-Hub
workspace without depending on import-order luck.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import orjson
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRODUCER_PATH = _REPO_ROOT / "scripts" / "generate_artifact_manifest.py"
REAL_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


def _load_producer():
    """Load the producer module from its file path (shadow-proof; see above)."""
    spec = importlib.util.spec_from_file_location("_am_producer", _PRODUCER_PATH)
    module = importlib.util.module_from_spec(spec)
    # exec_module runs the producer's own repo-root bootstrap, so its
    # ``from src.utils.io import ...`` resolves under the worktree's ``src``.
    spec.loader.exec_module(module)
    return module


_am = _load_producer()
MANIFEST_FILENAME = _am.MANIFEST_FILENAME
SCHEMA_VERSION = _am.SCHEMA_VERSION
build_manifest = _am.build_manifest
manifest_path = _am.manifest_path
verify_manifest = _am.verify_manifest
write_manifest = _am.write_manifest
iter_artifact_files = _am.iter_artifact_files

from src.utils.io import load_json  # noqa: E402

REAL_MANIFEST = REAL_FIXTURES / MANIFEST_FILENAME


@pytest.fixture
def fixtures_copy(tmp_path: Path) -> Path:
    """A throwaway copy of the real fixtures dir + manifest for mutation tests.

    Copies bytes only — never touches the real evidence. Tests mutate this copy
    and assert :func:`verify_manifest` flags the drift.
    """
    copy = tmp_path / "fixtures"
    shutil.copytree(REAL_FIXTURES, copy)
    return copy


# --------------------------------------------------------------------------- #
# The real guard — runs in CI / plain checkout (no skip).
# --------------------------------------------------------------------------- #
def test_committed_manifest_matches_live_tree() -> None:
    """The committed manifest must equal the manifest recomputed from the tree.

    This is the load-bearing structural-integrity assertion. It fails on any
    tampered fixture byte, any added-but-unregistered fixture, any deleted-but-
    still-referenced fixture, or any hand-edit to the committed manifest's
    counts/fields.
    """
    ok, diffs = verify_manifest(REAL_FIXTURES)
    assert ok, "manifest drift:\n  - " + "\n  - ".join(diffs)
    assert diffs == []

    # Structural equality: the producer's view of the live tree, reconstructed
    # independently of the committed file, must equal the committed file's dict.
    assert build_manifest(REAL_FIXTURES) == load_json(REAL_MANIFEST)


def test_committed_manifest_is_byte_canonical() -> None:
    """The committed manifest must equal the producer's canonical serialization.

    Pins deterministic regeneration: a hand-reorder / reformat / re-encoding of
    the committed JSON flips this RED even if the parsed dict is unchanged.
    """
    canonical = orjson.dumps(build_manifest(REAL_FIXTURES), option=orjson.OPT_INDENT_2)
    assert canonical == REAL_MANIFEST.read_bytes()


def test_manifest_covers_every_fixture_file_and_vice_versa() -> None:
    """No fixture file is unregistered; no entry points at a missing file."""
    live_paths = set(iter_artifact_files(REAL_FIXTURES))
    manifest = load_json(REAL_MANIFEST)
    registered = {a["path"] for a in manifest["artifacts"]}

    assert live_paths == registered, (
        f"unregistered={live_paths - registered} "
        f"missing-from-tree={registered - live_paths}"
    )
    assert manifest["artifact_count"] == len(registered)


def test_manifest_entries_are_path_sorted() -> None:
    """Stable ordering is what makes the manifest diffable & regen byte-stable."""
    manifest = load_json(REAL_MANIFEST)
    paths = [a["path"] for a in manifest["artifacts"]]
    assert paths == sorted(paths)


# --------------------------------------------------------------------------- #
# Mutation proofs — each drift class flips verify_manifest RED.
# --------------------------------------------------------------------------- #
def test_verify_catches_tampered_fixture_bytes(fixtures_copy: Path) -> None:
    """A single flipped byte in any fixture must surface as a sha256 mismatch."""
    target = fixtures_copy / "freeze_validloss_ci_9b_full.json"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF  # flip the first byte
    target.write_bytes(bytes(data))

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any(
        "sha256 mismatch" in d and "freeze_validloss_ci_9b_full.json" in d
        for d in diffs
    )


def test_verify_catches_size_drift(fixtures_copy: Path) -> None:
    """A wrong ``size`` in the manifest (bytes intact) must surface."""
    out = manifest_path(fixtures_copy)
    manifest = load_json(out)
    manifest["artifacts"][0]["size"] = manifest["artifacts"][0]["size"] + 1
    out.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any("size mismatch" in d for d in diffs)


def test_verify_catches_tampered_sha256_in_manifest(fixtures_copy: Path) -> None:
    """A forged ``sha256`` in the manifest (bytes intact) must surface."""
    out = manifest_path(fixtures_copy)
    manifest = load_json(out)
    manifest["artifacts"][0]["sha256"] = "0" * 64
    out.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any("sha256 mismatch" in d for d in diffs)


def test_verify_catches_untracked_added_fixture(fixtures_copy: Path) -> None:
    """A new fixture file not registered in the manifest must surface."""
    (fixtures_copy / "sneaky_unregistered_deposit.json").write_bytes(b'{"x":1}')

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any(
        "untracked artifact" in d and "sneaky_unregistered_deposit.json" in d
        for d in diffs
    )


def test_verify_catches_deleted_fixture(fixtures_copy: Path) -> None:
    """A referenced fixture removed from the tree must surface as a stale entry."""
    (fixtures_copy / "freeze_validloss_ci_9b_surrogate.json").unlink()

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any("stale manifest entry" in d for d in diffs)


def test_verify_catches_artifact_count_drift(fixtures_copy: Path) -> None:
    """A wrong ``artifact_count`` header must surface even if the list is right."""
    out = manifest_path(fixtures_copy)
    manifest = load_json(out)
    manifest["artifact_count"] = manifest["artifact_count"] + 1
    out.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any("artifact_count mismatch" in d for d in diffs)


def test_verify_catches_schema_version_drift(fixtures_copy: Path) -> None:
    """A ``schema_version`` bump without a producer change must surface."""
    out = manifest_path(fixtures_copy)
    manifest = load_json(out)
    manifest["schema_version"] = SCHEMA_VERSION + 1
    out.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    ok, diffs = verify_manifest(fixtures_copy)
    assert not ok
    assert any("schema_version drift" in d for d in diffs)


def test_verify_catches_missing_manifest(tmp_path: Path) -> None:
    """An absent manifest must fail loud (not silently pass)."""
    empty = tmp_path / "no_manifest_here"
    empty.mkdir()
    ok, diffs = verify_manifest(empty)
    assert not ok
    assert any("manifest missing" in d for d in diffs)


def test_write_manifest_is_idempotent_and_deterministic(fixtures_copy: Path) -> None:
    """Two consecutive regenerations are byte-identical and leave verify green.

    Note the ``base_dir`` field is location-aware (the manifest stamps where it
    lives), so the FIRST rewrite of a copied fixture dir legitimately differs
    from the copied original — but a SECOND rewrite must equal the first, which
    is the determinism property that makes the manifest regenerable & diffable.
    """
    out = manifest_path(fixtures_copy)
    write_manifest(fixtures_copy)
    first = out.read_bytes()

    write_manifest(fixtures_copy)
    second = out.read_bytes()
    assert first == second, "manifest is not deterministically reproducible"

    ok, diffs = verify_manifest(fixtures_copy)
    assert ok, diffs
