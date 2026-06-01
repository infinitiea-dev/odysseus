"""Tests for ChatProcessor._hybrid_retrieve (keyword/BM25 path).

Guards the BM25 scoring after hoisting the constant avg_len computation out of
the per-memory inner loop (an O(N^2) -> O(N) change that must not alter ranking).
"""
import time

from src.chat_processor import ChatProcessor


def _proc():
    # _hybrid_retrieve only touches memory_vector (None here → keyword-only path);
    # the other managers are unused for this method.
    return ChatProcessor(memory_manager=None, personal_docs_manager=None,
                         memory_vector=None, skills_manager=None)


def _mem(mid, text):
    return {"id": mid, "text": text, "category": "fact", "timestamp": time.time()}


def test_keyword_retrieval_ranks_relevant_first():
    proc = _proc()
    mems = [
        _mem("1", "The user's favorite programming language is Rust"),
        _mem("2", "The user lives in Tokyo Japan near the coast"),
        _mem("3", "The user enjoys hiking mountains on weekends"),
    ]
    out = proc._hybrid_retrieve("which programming language does the user prefer", mems, k=2)
    assert out, "expected at least one relevant memory"
    assert out[0]["id"] == "1"
    assert len(out) <= 2


def test_empty_query_returns_nothing():
    proc = _proc()
    assert proc._hybrid_retrieve("", [_mem("1", "anything")], k=3) == []


def test_no_memories_returns_nothing():
    proc = _proc()
    assert proc._hybrid_retrieve("anything at all", [], k=3) == []


def test_irrelevant_query_filtered_out():
    proc = _proc()
    mems = [_mem("1", "The user's favorite programming language is Rust")]
    # No meaningful token overlap → gated out by the relevance threshold.
    assert proc._hybrid_retrieve("quantum helicopter umbrella", mems, k=3) == []
