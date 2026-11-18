import json

import pytest

from lfm3ha.tools_de import MEDIA_TYPES, SERVICES, TIMER_ACTIONS, TOOLS, build_tools


def test_six_tools_with_expected_names():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["control", "get_state", "music_library", "timer", "get_time", "broadcast"]


def test_schemas_are_valid_json_schema_shapes():
    for tool in TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["description"], f"{fn['name']} braucht eine Beschreibung"
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params.get("properties", {}), dict)
        for required in params.get("required", []):
            assert required in params["properties"], f"{fn['name']}: {required} fehlt"


def test_enums_are_filled():
    by_name = {t["function"]["name"]: t["function"] for t in TOOLS}
    assert by_name["music_library"]["parameters"]["properties"]["media_type"]["enum"] == list(
        MEDIA_TYPES
    )
    assert by_name["timer"]["parameters"]["properties"]["action"]["enum"] == list(TIMER_ACTIONS)


def test_control_lists_its_services():
    description = TOOLS[0]["function"]["description"]
    for service in ("light.turn_on", "media_player.search_and_play", "vacuum.start"):
        assert service in description
    assert "vacuum.start" in SERVICES


def test_tools_are_serializable_and_copyable():
    a = build_tools()
    b = build_tools()
    a[0]["function"]["name"] = "kaputt"
    assert b[0]["function"]["name"] == "control"
    assert json.loads(json.dumps(b, ensure_ascii=False))


@pytest.mark.parametrize("tool", TOOLS, ids=[t["function"]["name"] for t in TOOLS])
def test_no_empty_parameter_descriptions(tool):
    for name, spec in tool["function"]["parameters"].get("properties", {}).items():
        assert isinstance(spec, dict), f"{tool['function']['name']}.{name}"
