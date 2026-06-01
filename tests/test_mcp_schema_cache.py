"""Tests for McpManager.get_all_openai_schemas caching.

The schema list is assembled on every agent request, so it is cached on the
same signal as get_tool_descriptions_for_prompt (disabled map + server count).
"""
from src.mcp_manager import McpManager


def _mgr_with_one_server():
    mgr = McpManager()
    # An NPX-style builtin is included in function-calling schemas; use a plain
    # external server id so it is always included regardless of is_builtin().
    mgr._tools["srv1"] = [
        {"name": "do_thing", "description": "does a thing", "input_schema": {"type": "object", "properties": {}}},
        {"name": "do_other", "description": "does another", "input_schema": {"type": "object", "properties": {}}},
    ]
    mgr._connections["srv1"] = {"status": "connected", "name": "Server One"}
    return mgr


def test_schemas_built_correctly():
    mgr = _mgr_with_one_server()
    schemas = mgr.get_all_openai_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"mcp__srv1__do_thing", "mcp__srv1__do_other"}
    assert all(s["type"] == "function" for s in schemas)


def test_repeated_call_returns_cached_object():
    mgr = _mgr_with_one_server()
    first = mgr.get_all_openai_schemas()
    second = mgr.get_all_openai_schemas()
    assert first is second  # identical object → served from cache


def test_disabled_map_filters_and_keys_cache():
    mgr = _mgr_with_one_server()
    full = mgr.get_all_openai_schemas()
    filtered = mgr.get_all_openai_schemas({"srv1": {"do_other"}})
    full_names = {s["function"]["name"] for s in full}
    filtered_names = {s["function"]["name"] for s in filtered}
    assert "mcp__srv1__do_other" in full_names
    assert "mcp__srv1__do_other" not in filtered_names
    assert filtered_names == {"mcp__srv1__do_thing"}
    # Different disabled maps must not collide in the cache.
    assert full is not filtered


def test_cache_invalidates_when_server_count_changes():
    mgr = _mgr_with_one_server()
    first = mgr.get_all_openai_schemas()
    # Simulate a new server connecting.
    mgr._tools["srv2"] = [
        {"name": "thing2", "description": "d", "input_schema": {"type": "object", "properties": {}}},
    ]
    mgr._connections["srv2"] = {"status": "connected", "name": "Server Two"}
    refreshed = mgr.get_all_openai_schemas()
    assert refreshed is not first
    assert "mcp__srv2__thing2" in {s["function"]["name"] for s in refreshed}
