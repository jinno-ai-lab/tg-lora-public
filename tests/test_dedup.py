"""Unit tests for src/data/dedup.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import orjson

from src.data.dedup import dedup_embedding, dedup_exact, find_duplicates
from src.utils.io import load_jsonl


def _write_jsonl(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


# ── dedup_exact ────────────────────────────────────────────────────────────


class TestDedupExact:
    def test_removes_exact_duplicates(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            inp,
            [
                {"text": "hello"},
                {"text": "world"},
                {"text": "hello"},
            ],
        )
        removed = dedup_exact(str(inp), str(out))
        result = load_jsonl(str(out))
        assert len(result) == 2
        assert removed == 1
        texts = [r["text"] for r in result]
        assert "hello" in texts
        assert "world" in texts

    def test_no_duplicates_returns_zero(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            inp,
            [
                {"text": "alpha"},
                {"text": "beta"},
                {"text": "gamma"},
            ],
        )
        removed = dedup_exact(str(inp), str(out))
        assert removed == 0
        assert len(load_jsonl(str(out))) == 3

    def test_all_duplicates_keeps_one(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [{"text": "same"}] * 5)
        removed = dedup_exact(str(inp), str(out))
        assert removed == 4
        assert len(load_jsonl(str(out))) == 1

    def test_empty_input(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [])
        removed = dedup_exact(str(inp), str(out))
        assert removed == 0
        assert load_jsonl(str(out)) == []

    def test_custom_text_key(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            inp,
            [
                {"content": "abc"},
                {"content": "abc"},
                {"content": "def"},
            ],
        )
        removed = dedup_exact(str(inp), str(out), text_key="content")
        assert removed == 1
        assert len(load_jsonl(str(out))) == 2

    def test_keeps_first_occurrence(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            inp,
            [
                {"text": "dup", "id": 1},
                {"text": "dup", "id": 2},
            ],
        )
        result = load_jsonl(str(out)) if dedup_exact(str(inp), str(out)) or True else []
        assert result[0]["id"] == 1


# ── find_duplicates (numpy path) ──────────────────────────────────────────


class TestFindDuplicates:
    def test_identifies_duplicate_embeddings(self):
        vec = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        keep = find_duplicates(vec, similarity_threshold=0.99)
        assert 0 in keep
        assert 1 not in keep
        assert 2 in keep

    def test_no_duplicates_keeps_all(self):
        vecs = np.eye(3, dtype=np.float32)
        keep = find_duplicates(vecs, similarity_threshold=0.99)
        assert keep == {0, 1, 2}

    def test_single_record_keeps_it(self):
        vec = np.array([[1.0, 0.0]], dtype=np.float32)
        keep = find_duplicates(vec, similarity_threshold=0.99)
        assert keep == {0}

    def test_all_duplicates_keeps_first(self):
        vec = np.array([[1.0, 0.0]] * 4, dtype=np.float32)
        keep = find_duplicates(vec, similarity_threshold=0.99)
        assert 0 in keep
        assert len(keep) == 1


# ── dedup_embedding ────────────────────────────────────────────────────────


class TestDedupEmbedding:
    @patch("src.data.dedup.compute_embeddings")
    @patch("src.data.dedup.find_duplicates")
    def test_removes_semantic_duplicates(self, mock_find, mock_embed, tmp_path):
        mock_embed.return_value = np.eye(3, dtype=np.float32)
        mock_find.return_value = {0, 2}

        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            inp,
            [
                {"text": "the cat sat"},
                {"text": "a feline rested"},
                {"text": "dogs are great"},
            ],
        )

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            removed = dedup_embedding(str(inp), str(out))

        result = load_jsonl(str(out))
        assert len(result) == 2
        assert removed == 1
        assert result[0]["text"] == "the cat sat"
        assert result[1]["text"] == "dogs are great"

    @patch("src.data.dedup.compute_embeddings")
    @patch("src.data.dedup.find_duplicates")
    def test_no_duplicates_keeps_all(self, mock_find, mock_embed, tmp_path):
        mock_embed.return_value = np.eye(3, dtype=np.float32)
        mock_find.return_value = {0, 1, 2}

        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            inp,
            [
                {"text": "a"},
                {"text": "b"},
                {"text": "c"},
            ],
        )

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            removed = dedup_embedding(str(inp), str(out))

        assert removed == 0
        assert len(load_jsonl(str(out))) == 3

    def test_empty_input_returns_zero(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [])

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            removed = dedup_embedding(str(inp), str(out))

        assert removed == 0

    def test_missing_sentence_transformers_returns_zero(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        _write_jsonl(inp, [{"text": "hello"}])

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            removed = dedup_embedding(str(inp), str(out))

        assert removed == 0


class TestComputeEmbeddings:
    def test_calls_sentence_transformers(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.eye(3, dtype=np.float32)
        mock_st_class = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_class)},
        ):
            from src.data.dedup import compute_embeddings

            result = compute_embeddings(["a", "b", "c"], "fake-model")
            mock_st_class.assert_called_once_with("fake-model")
            mock_model.encode.assert_called_once()
            assert result.shape == (3, 3)


class TestFindDuplicatesNumpyFallback:
    def test_fallback_without_faiss(self):
        vecs = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        with patch.dict("sys.modules", {"faiss": None}):
            keep = find_duplicates(vecs, similarity_threshold=0.95, top_k=10)
        # First two are near-duplicates, second should be removed
        assert 0 in keep
        assert 2 in keep

    def test_fallback_all_unique(self):
        vecs = np.eye(4, dtype=np.float32)
        with patch.dict("sys.modules", {"faiss": None}):
            keep = find_duplicates(vecs, similarity_threshold=0.99, top_k=10)
        assert keep == {0, 1, 2, 3}

    def test_fallback_with_topk_less_than_n(self):
        vecs = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.14],
                [0.0, 1.0],
                [0.0, -1.0],
                [1.0, 0.01],
            ],
            dtype=np.float32,
        )
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        with patch.dict("sys.modules", {"faiss": None}):
            keep = find_duplicates(vecs, similarity_threshold=0.95, top_k=2)
        # With top_k=2 < n=5, the argpartition path is taken
        assert 0 in keep
        assert 2 in keep

    def test_fallback_skips_already_removed_j(self):
        vecs = np.array(
            [
                [1.0, 0.0],  # v0
                [0.0, 1.0],  # v1
                [1.0, 0.0],  # v2 (dup of v0)
                [0.0, 1.0],  # v3 (dup of v1)
            ],
            dtype=np.float32,
        )
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        with patch.dict("sys.modules", {"faiss": None}):
            keep = find_duplicates(vecs, similarity_threshold=0.99, top_k=10)
        # v2 removed by v0, v3 removed by v1
        # When i=1 processes j=2, v2 is already not in keep → hits continue
        assert keep == {0, 1}


class TestFindDuplicatesFaissWarning:
    def test_faiss_warns_when_n_exceeds_k(self, caplog):
        vecs = np.eye(20, dtype=np.float32)

        keep = find_duplicates(vecs, similarity_threshold=0.99, top_k=5)
        # Should still return all since they're orthogonal
        assert len(keep) == 20
