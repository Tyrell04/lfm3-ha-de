import json

from lfm3ha.evaluate import (
    aggregate,
    call_signature,
    gold_calls,
    normalize_args,
    parse_tool_calls,
    score,
)

GERMAN_CALL = """<tool_call>
<function=control>
<parameter=actions>
[{"service": "light.turn_on", "entity_id": ["light.wohnzimmer_decke"]}]
</parameter>
</function>
</tool_call>"""


def test_parses_single_call():
    calls = parse_tool_calls(GERMAN_CALL)
    assert len(calls) == 1
    assert calls[0]["name"] == "control"
    assert calls[0]["arguments"]["actions"][0]["service"] == "light.turn_on"


def test_parses_multiple_calls_and_scalars():
    text = (
        "<tool_call>\n<function=timer>\n<parameter=action>\nstart\n</parameter>\n"
        "<parameter=minutes>\n10\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["timer", "get_time"]
    assert calls[0]["arguments"] == {"action": "start", "minutes": 10}
    assert calls[1]["arguments"] == {}


def test_ignores_prose_and_malformed_blocks():
    assert parse_tool_calls("Ich mache das Licht an.") == []
    assert parse_tool_calls("<tool_call><function=control>") == []


def test_entity_id_order_and_types_do_not_matter():
    a = {"actions": [{"service": "light.turn_off", "entity_id": ["a", "b"], "data": {}}]}
    b = {"actions": [{"service": "light.turn_off", "entity_id": ["b", "a"]}]}
    assert call_signature({"name": "control", "arguments": a}) == call_signature(
        {"name": "control", "arguments": b}
    )


def test_normalize_drops_empty_values_and_coerces_numbers():
    args = normalize_args({"minutes": "10", "name": "", "zone": None, "extra": "x"})
    assert args == {"minutes": 10, "extra": "x"}


def test_gold_calls_reads_first_assistant_turn():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Licht an"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "control",
                        "arguments": {"actions": [{"service": "light.turn_on", "entity_id": ["light.x"]}]},
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "Erledigt."},
    ]
    gold = gold_calls(messages)
    assert gold[0]["name"] == "control"
    assert score(gold, GERMAN_CALL.replace("wohnzimmer_decke", "x"))["exact"] is True


def test_wrong_function_is_not_exact():
    gold = [{"name": "get_state", "arguments": {"entity_ids": ["sensor.x"]}}]
    result = score(gold, GERMAN_CALL)
    assert result["parse"] is True
    assert result["function"] is False
    assert result["exact"] is False


def test_chat_rows_need_text_without_tool_call():
    assert score([], "Guten Morgen!")["answer"] is True
    assert score([], GERMAN_CALL)["answer"] is False
    assert score([], "   ")["answer"] is False


def test_aggregate_reports_rates_and_categories():
    rows = [
        {
            "category": "control",
            "gold": [{"name": "control", "arguments": {}}],
            "prediction": GERMAN_CALL,
            "scores": score([{"name": "control", "arguments": {}}], GERMAN_CALL),
        },
        {
            "category": "chat",
            "gold": [],
            "prediction": "Hallo!",
            "scores": score([], "Hallo!"),
        },
    ]
    metrics = aggregate(rows)
    assert metrics["overall"]["n"] == 2
    assert metrics["tool_calls"]["n"] == 1
    assert metrics["chat"]["exact"] == 1.0
    assert json.loads(json.dumps(metrics))
