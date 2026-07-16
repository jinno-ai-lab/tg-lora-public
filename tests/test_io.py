import os

import pytest

from src.utils.io import save_json, load_json, save_jsonl, load_jsonl


class TestSaveLoadJson:
    def test_roundtrip_dict(self, tmp_path):
        data = {"key": "value", "num": 42}
        path = tmp_path / "test.json"
        save_json(data, path)
        assert load_json(path) == data

    def test_roundtrip_list(self, tmp_path):
        data = [{"a": 1}, {"b": 2}]
        path = tmp_path / "list.json"
        save_json(data, path)
        assert load_json(path) == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.json"
        save_json({"x": 1}, path)
        assert load_json(path) == {"x": 1}

    def test_string_path(self, tmp_path):
        path = str(tmp_path / "str.json")
        save_json({"ok": True}, path)
        assert load_json(path) == {"ok": True}


class TestSaveLoadJsonl:
    def test_roundtrip_records(self, tmp_path):
        records = [{"text": "hello"}, {"text": "world", "id": 2}]
        path = tmp_path / "test.jsonl"
        save_jsonl(records, path)
        assert load_jsonl(path) == records

    def test_empty_list_writes_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        save_jsonl([], path)
        assert load_jsonl(path) == []
        assert path.exists()

    def test_skips_blank_lines_on_load(self, tmp_path):
        path = tmp_path / "gaps.jsonl"
        path.write_bytes(b'{"a":1}\n\n{"b":2}\n\n')
        assert load_jsonl(path) == [{"a": 1}, {"b": 2}]

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        save_jsonl([{"x": 1}], path)
        assert load_jsonl(path) == [{"x": 1}]

    def test_string_path(self, tmp_path):
        path = str(tmp_path / "str.jsonl")
        save_jsonl([{"ok": True}], path)
        assert load_jsonl(path) == [{"ok": True}]

    def test_unicode_roundtrip(self, tmp_path):
        records = [{"text": "こんにちは世界"}, {"text": "日本語テスト"}]
        path = tmp_path / "unicode.jsonl"
        save_jsonl(records, path)
        assert load_jsonl(path) == records


class TestAtomicJsonWrites:
    """A mid-write kill never leaves a torn JSON destination.

    ``save_json`` / ``save_jsonl`` route through
    :func:`src.utils.io._atomic_write_bytes` (temp-in-same-dir + ``os.replace``),
    the JSON analogue of :func:`src.utils.atomic_save._atomic_torch_save`. This
    pins that guarantee at the helper level the same way
    :mod:`tests.test_atomic_save` pins the torch-artifact path: inject an
    interrupt at the ``os.replace`` publish boundary and assert (a) the prior,
    still-loadable destination is intact (the torn new value was never
    published), (b) no orphan PID-suffixed temp litters the directory, and
    (c) the original interrupt type re-raises. Parametrized over ``OSError``
    (an ``Exception`` subclass — the common fault) AND ``KeyboardInterrupt`` /
    ``SystemExit`` (``BaseException`` subclasses — the ``except BaseException``
    cleanup path a narrowed ``except Exception`` would silently drop).
    """

    @pytest.mark.parametrize("interrupt", [OSError, KeyboardInterrupt, SystemExit])
    def test_save_json_prior_destination_survives_mid_publish(
        self, tmp_path, monkeypatch, interrupt
    ):
        path = tmp_path / "out.json"
        save_json({"v": 1}, path)
        assert load_json(path) == {"v": 1}

        def _boom(_src, _dst):
            raise interrupt("simulated mid-publish interrupt")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(interrupt):
            save_json({"v": 2}, path)

        # The prior, still-loadable destination is intact with the OLD value —
        # the torn (v2) state was never published.
        assert load_json(path) == {"v": 1}
        # No orphan PID-suffixed temp left behind (the load-bearing assertion
        # for the ``except BaseException`` cleanup over ``except Exception``).
        assert not list(tmp_path.glob("out.json.tmp.*"))

    @pytest.mark.parametrize("interrupt", [OSError, KeyboardInterrupt, SystemExit])
    def test_save_jsonl_prior_destination_survives_mid_publish(
        self, tmp_path, monkeypatch, interrupt
    ):
        path = tmp_path / "out.jsonl"
        save_jsonl([{"v": 1}], path)
        assert load_jsonl(path) == [{"v": 1}]

        def _boom(_src, _dst):
            raise interrupt("simulated mid-publish interrupt")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(interrupt):
            save_jsonl([{"v": 2}, {"v": 3}], path)

        assert load_jsonl(path) == [{"v": 1}]
        assert not list(tmp_path.glob("out.jsonl.tmp.*"))

    def test_save_json_fresh_save_interrupt_publishes_nothing(
        self, tmp_path, monkeypatch
    ):
        # No prior destination: an interrupted fresh save must publish nothing
        # at all (no empty/torn file), and clean the orphan temp.
        path = tmp_path / "fresh.json"

        def _boom(_src, _dst):
            raise OSError("simulated mid-publish interrupt")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            save_json({"v": 1}, path)

        assert not path.exists()
        assert not list(tmp_path.glob("fresh.json.tmp.*"))

    def test_save_jsonl_empty_list_still_atomic_and_round_trips(self, tmp_path):
        # The buffered-join form must still produce a valid empty file (the
        # prior per-record loop wrote nothing) and round-trip to [].
        path = tmp_path / "empty.jsonl"
        save_jsonl([], path)
        assert path.exists()
        assert load_jsonl(path) == []
        assert not list(tmp_path.glob("empty.jsonl.tmp.*"))  # temp cleaned

