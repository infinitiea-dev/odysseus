"""Tests for VectorRAG efficiency changes.

Covers the batched existence check in add_documents_batch (one ChromaDB get
instead of one per document, with identical dedup behavior) and the single
count() call in search. Uses lightweight fakes so neither ChromaDB nor an
embedding backend is required.
"""
import numpy as np

from src.rag_vector import VectorRAG, VECTOR_WEIGHT, KEYWORD_WEIGHT


def _doc_id(text: str) -> str:
    return f"doc_{hash(text) % 10**16}"


class FakeCollection:
    def __init__(self, existing_ids=None):
        self._existing = set(existing_ids or [])
        self.get_calls = 0
        self.added = []          # list of id-lists passed to add()
        self.count_calls = 0
        self._count = len(self._existing)
        self.query_calls = 0

    def count(self):
        self.count_calls += 1
        return self._count

    def get(self, ids=None, where=None, include=None):
        self.get_calls += 1
        if ids is not None:
            return {"ids": [i for i in ids if i in self._existing]}
        return {"ids": list(self._existing)}

    def add(self, ids, embeddings, documents, metadatas):
        self.added.append(list(ids))
        self._existing.update(ids)
        self._count = len(self._existing)

    def query(self, query_embeddings, n_results, where=None, include=None):
        self.query_calls += 1
        self.last_n_results = n_results
        # Return two canned docs.
        return {
            "ids": [["a", "b"]],
            "distances": [[0.1, 0.4]],
            "documents": [["hello world alpha", "unrelated beta"]],
            "metadatas": [[{"filename": "a.txt"}, {"filename": "b.txt"}]],
        }


class FakeModel:
    model = "fake"
    url = "local://fake"

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype="float32")


def _make_rag(collection):
    """Build a VectorRAG without running its heavy __init__."""
    rag = object.__new__(VectorRAG)
    rag._collection = collection
    rag._model = FakeModel()
    rag._healthy = True
    rag.persist_directory = "data/chroma"
    return rag


def test_add_documents_batch_single_get_and_dedup():
    docs = [
        ("alpha text", {"source": "x"}),
        ("beta text", {"source": "y"}),
        ("alpha text", {"source": "z"}),  # intra-batch duplicate of the first
    ]
    # "beta text" already lives in the collection → should be skipped.
    col = FakeCollection(existing_ids={_doc_id("beta text")})
    rag = _make_rag(col)

    result = rag.add_documents_batch(docs)

    # Existence is checked in exactly ONE round-trip, not one per document.
    assert col.get_calls == 1
    # beta skipped (already present); both alphas added (intra-batch dup preserved).
    assert result["added_count"] == 2
    assert col.added and len(col.added[0]) == 2
    assert all(i == _doc_id("alpha text") for i in col.added[0])
    assert result["total_count"] == 3


def test_add_documents_batch_all_new():
    col = FakeCollection()
    rag = _make_rag(col)
    result = rag.add_documents_batch([("one", {"s": 1}), ("two", {"s": 2})])
    assert col.get_calls == 1
    assert result["added_count"] == 2


def test_search_calls_count_once():
    col = FakeCollection(existing_ids={"a", "b"})
    rag = _make_rag(col)
    results = rag.search("hello world alpha", k=2)
    assert col.count_calls == 1          # collapsed from 3 calls
    assert col.query_calls == 1
    assert len(results) == 2
    # Hybrid score = vector_sim*W_v + keyword*W_k; first doc shares words with query.
    top = results[0]
    expected_vec = 1.0 - 0.1
    assert top["vector_similarity"] == round(expected_vec, 4)
    assert top["similarity"] >= results[1]["similarity"]


def test_search_empty_collection_short_circuits():
    col = FakeCollection()  # count == 0
    rag = _make_rag(col)
    assert rag.search("anything") == []
    assert col.query_calls == 0
