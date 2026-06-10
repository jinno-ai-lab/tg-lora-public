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
