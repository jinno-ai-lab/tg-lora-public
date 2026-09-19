import logging

import numpy as np

from src.utils.io import load_jsonl, save_jsonl

logger = logging.getLogger("tg-lora")


def dedup_exact(input_path: str, output_path: str, text_key: str = "text") -> int:
    records = load_jsonl(input_path)
    seen = set()
    unique = []
    for rec in records:
        text = rec.get(text_key, "")
        if text not in seen:
            seen.add(text)
            unique.append(rec)
    save_jsonl(unique, output_path)
    removed = len(records) - len(unique)
    logger.info(f"Exact dedup: {len(records)} -> {len(unique)} (removed {removed})")
    return removed


def compute_embeddings(
    texts: list[str],
    embedding_model: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def find_duplicates(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.95,
    top_k: int = 200,
) -> set[int]:
    """Return indices of records to keep (duplicates removed)."""
    n = len(embeddings)
    keep = set(range(n))

    try:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        k = min(n, top_k)
        similarities, faiss_indices = index.search(embeddings.astype(np.float32), k=k)

        for i in range(n):
            if i not in keep:
                continue
            for rank in range(k):
                j = int(faiss_indices[i][rank])
                if j <= i or j not in keep:
                    continue
                if similarities[i][rank] >= similarity_threshold:
                    keep.discard(j)

        if n > k:
            logger.warning(
                f"FAISS dedup is approximate (k={k} for n={n} records). "
                "Some near-duplicates beyond top-k may be missed."
            )

    except ImportError:
        logger.warning("faiss not installed, using numpy for dedup")
        sim_matrix = embeddings @ embeddings.T

        k = min(n, top_k)
        for i in range(n):
            if i not in keep:
                continue
            row = sim_matrix[i]
            if k >= n:
                for j in range(i + 1, n):
                    if j not in keep:
                        continue
                    if row[j] >= similarity_threshold:
                        keep.discard(j)
            else:
                top_positions = np.argpartition(-row, k)[:k]
                for pos in top_positions:
                    j = int(pos)
                    if j <= i or j not in keep:
                        continue
                    if row[j] >= similarity_threshold:
                        keep.discard(j)

    return keep


def dedup_embedding(
    input_path: str,
    output_path: str,
    text_key: str = "text",
    similarity_threshold: float = 0.95,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> int:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping embedding dedup")
        return 0

    records = load_jsonl(input_path)
    if not records:
        return 0

    texts = [rec.get(text_key, "") for rec in records]
    embeddings = compute_embeddings(texts, embedding_model)
    keep = find_duplicates(embeddings, similarity_threshold)

    unique = [records[i] for i in sorted(keep)]
    save_jsonl(unique, output_path)
    removed = len(records) - len(unique)
    logger.info(f"Embedding dedup: {len(records)} -> {len(unique)} (removed {removed})")
    return removed
