"""Unit tests for src/llm_core.py caching and request-prep internals.

These cover the speed/efficiency rework of the LLM core: the bounded TTL+LRU
response cache, the memoized pure helpers, the shared message-prep, and the
deterministic cache key. No network is required.
"""
import importlib

import pytest

import src.llm_core as llm


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty response cache."""
    llm._response_cache.clear()
    yield
    llm._response_cache.clear()


# ── TTL + LRU response cache ──

def test_cache_roundtrip():
    llm._set_cached_response("k", "v")
    assert llm._get_cached_response("k") == "v"
    assert llm._get_cached_response("missing") is None


def test_cache_evicts_oldest_over_capacity():
    n = llm.LLMConfig.CACHE_MAX
    for i in range(n + 5):
        llm._set_cached_response(f"k{i}", f"v{i}")
    assert len(llm._response_cache) == n
    # The first 5 inserted should have been evicted from the front.
    assert llm._get_cached_response("k0") is None
    assert llm._get_cached_response("k4") is None
    assert llm._get_cached_response(f"k{n + 4}") == f"v{n + 4}"


def test_cache_recently_read_entry_survives_eviction():
    """A read marks an entry MRU, so it should outlive newer-but-untouched keys."""
    n = llm.LLMConfig.CACHE_MAX
    for i in range(n):
        llm._set_cached_response(f"k{i}", f"v{i}")
    # Touch the oldest key so it becomes most-recently-used.
    assert llm._get_cached_response("k0") == "v0"
    # Insert one more, forcing a single eviction from the front.
    llm._set_cached_response("new", "newval")
    assert llm._get_cached_response("k0") == "v0"        # survived
    assert llm._get_cached_response("k1") is None        # evicted instead


def test_cache_ttl_expiry(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(llm.time, "time", lambda: t["now"])
    llm._set_cached_response("k", "v")
    assert llm._get_cached_response("k") == "v"
    t["now"] += llm.LLMConfig.CACHE_TTL + 1
    assert llm._get_cached_response("k") is None
    # Expired entry is dropped, not left lingering.
    assert "k" not in llm._response_cache


# ── Cache key determinism ──

def test_cache_key_stable_across_dict_order():
    a = [{"role": "user", "content": "hi"}]
    b = [{"content": "hi", "role": "user"}]  # same data, different key order
    assert llm._get_cache_key("u", "m", a, 0.5, 100) == llm._get_cache_key("u", "m", b, 0.5, 100)


def test_cache_key_handles_multimodal_content():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]}]
    k1 = llm._get_cache_key("u", "m", msgs, 0.0, 0)
    k2 = llm._get_cache_key("u", "m", msgs, 0.0, 0)
    assert k1 == k2 and isinstance(k1, str) and len(k1) == 64


def test_cache_key_varies_on_inputs():
    base = [{"role": "user", "content": "hi"}]
    k = llm._get_cache_key("u", "m", base, 0.5, 100)
    assert k != llm._get_cache_key("u", "m", base, 0.9, 100)      # temperature
    assert k != llm._get_cache_key("u", "m", base, 0.5, 200)      # max_tokens
    assert k != llm._get_cache_key("u", "m2", base, 0.5, 100)     # model
    assert k != llm._get_cache_key("u2", "m", base, 0.5, 100)     # url


# ── Memoized pure helpers (parity with documented behavior) ──

@pytest.mark.parametrize("url,expected", [
    ("https://api.anthropic.com/v1/messages", "anthropic"),
    ("https://openrouter.ai/api/v1/chat/completions", "openrouter"),
    ("https://api.groq.com/openai/v1/chat/completions", "groq"),
    ("https://api.openai.com/v1/chat/completions", "openai"),
    ("http://localhost:11434/v1/chat/completions", "openai"),
    ("", "openai"),
])
def test_detect_provider(url, expected):
    assert llm._detect_provider(url) == expected


@pytest.mark.parametrize("model,expected", [
    ("o1-preview", True),
    ("gpt-5", True),
    ("openai/gpt-4.5", True),
    ("gpt-4o", False),
    ("claude-sonnet-4", False),
    ("", False),
])
def test_uses_max_completion_tokens(model, expected):
    assert llm._uses_max_completion_tokens(model) is expected


@pytest.mark.parametrize("model,expected", [
    ("qwen3-32b", True),
    ("deepseek-r1", True),
    ("QwQ-32B", True),
    ("gpt-4o", False),
    ("", False),
])
def test_supports_thinking(model, expected):
    assert llm._supports_thinking(model) is expected


@pytest.mark.parametrize("url,expected", [
    ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
    ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
    ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com/v1/messages"),
    ("https://api.anthropic.com/v1/messages/", "https://api.anthropic.com/v1/messages"),
])
def test_normalize_anthropic_url(url, expected):
    assert llm._normalize_anthropic_url(url) == expected


def test_host_key():
    assert llm._host_key("https://api.openai.com/v1/chat/completions") == "https://api.openai.com"
    # No scheme/netloc → returned unchanged.
    assert llm._host_key("not-a-url") == "not-a-url"


def test_helpers_are_memoized():
    # lru_cache exposes cache_info(); a repeated call should register a hit.
    llm._detect_provider.cache_clear()
    llm._detect_provider("https://api.openai.com/v1")
    llm._detect_provider("https://api.openai.com/v1")
    assert llm._detect_provider.cache_info().hits >= 1


# ── Shared message preparation ──

def test_prepare_messages_consolidates_system():
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "B"},
    ]
    out = llm._prepare_messages(msgs)
    assert out[0] == {"role": "system", "content": "A\n\nB"}
    assert out[1] == {"role": "user", "content": "hello"}
    assert len(out) == 2


def test_prepare_messages_strips_metadata():
    msgs = [{"role": "user", "content": "hi", "id": 7, "timestamp": "now"}]
    out = llm._prepare_messages(msgs)
    assert out == [{"role": "user", "content": "hi"}]


def test_prepare_messages_no_system():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert llm._prepare_messages(msgs) == msgs


def test_build_openai_payload_stream_options_gated():
    p = llm._build_openai_payload("m", [], 0.7, 0, stream=True, provider="openai")
    assert p["stream"] is True and p["stream_options"] == {"include_usage": True}
    pg = llm._build_openai_payload("m", [], 0.7, 0, stream=True, provider="groq")
    assert "stream_options" not in pg
    # Non-stream path adds neither stream nor stream_options.
    pn = llm._build_openai_payload("m", [], 0.7, 0)
    assert "stream" not in pn and "stream_options" not in pn


def test_build_openai_payload_token_key_selection():
    assert "max_completion_tokens" in llm._build_openai_payload("gpt-5", [], 0.7, 100)
    assert "max_tokens" in llm._build_openai_payload("gpt-4o", [], 0.7, 100)
    # max_tokens omitted entirely when not positive.
    assert "max_tokens" not in llm._build_openai_payload("gpt-4o", [], 0.7, 0)
