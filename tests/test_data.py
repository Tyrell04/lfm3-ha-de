import json

from lfm3ha.data import (
    _path_to_messages,
    categories,
    deduplicate,
    is_usable,
    mix,
    normalize_messages,
    stats,
)

RAW_ROW = [
    {"role": "system", "content": "Du bist der Sprachassistent."},
    {"role": "user", "content": "Lass den Saugroboter saugen"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "control",
                    "arguments": '{"actions": [{"service": "vacuum.start", '
                    '"entity_id": ["vacuum.haus_staubsauger"]}]}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"success": true}'},
    {"role": "assistant", "content": "Staubsauger legt los."},
]


def test_arguments_string_is_parsed_into_mapping():
    messages = normalize_messages(RAW_ROW)
    arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["actions"][0]["service"] == "vacuum.start"


def test_none_content_becomes_empty_string():
    messages = normalize_messages(RAW_ROW)
    assert messages[2]["content"] == ""


def test_bad_json_arguments_degrade_to_empty_mapping():
    row = [
        {"role": "system", "content": "s"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "control", "arguments": "{kaputt"}}],
        },
    ]
    messages = normalize_messages(row)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {}


def test_is_usable_rejects_missing_system_and_empty_answer():
    assert is_usable(normalize_messages(RAW_ROW))
    assert not is_usable(normalize_messages(RAW_ROW[1:]))
    empty = normalize_messages(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "  "},
        ]
    )
    assert not is_usable(empty)


def test_categories_are_stable():
    messages = normalize_messages(RAW_ROW)
    assert categories(messages) == "control"
    assert categories([{"role": "system", "content": "s"}, {"role": "assistant", "content": "Hi"}]) == "chat"


def test_deduplicate_and_stats():
    a = {"messages": normalize_messages(RAW_ROW), "source": "ha", "category": "control", "case": ""}
    b = {
        "messages": normalize_messages(RAW_ROW),
        "source": "ha",
        "category": "control",
        "case": "",
    }
    rows = deduplicate([a, b])
    assert len(rows) == 1
    summary = stats(rows)
    assert summary["rows"] == 1
    assert summary["tool_rows"] == 1


def test_mix_keeps_chat_share_bounded():
    ha = [
        {"messages": [{"role": "system", "content": "s"}], "category": "control", "source": "ha", "case": ""}
        for _ in range(100)
    ]
    chat = [
        {"messages": [{"role": "system", "content": "s"}], "category": "chat", "source": "oasst2", "case": ""}
        for _ in range(50)
    ]
    merged = mix(ha, chat, ratio=0.1)
    chat_share = sum(1 for r in merged if r["category"] == "chat") / len(merged)
    assert 0.05 <= chat_share <= 0.12


def test_thread_path_alternates_roles():
    path = [
        {"role": "prompter", "text": "Wie funktioniert eine Wärmepumpe im Altbau?"},
        {
            "role": "assistant",
            "text": "Eine Wärmepumpe entzieht der Außenluft Wärme und hebt sie auf ein "
            "höheres Temperaturniveau, das für Heizkörper ausreicht.",
        },
        {"role": "prompter", "text": "Und was kostet das im Jahr ungefähr?"},
    ]
    messages = _path_to_messages(path)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_bad_oasst_answer_ends_the_thread():
    path = [
        {"role": "prompter", "text": "Zeichne mir bitte ein Bild von einem Fuchs."},
        {
            "role": "assistant",
            "text": "Als KI kann ich leider keine Bilder erstellen, dafür brauchst du ein anderes Werkzeug.",
        },
        {"role": "prompter", "text": "Schade, dann beschreib ihn wenigstens."},
    ]
    messages = _path_to_messages(path)
    # Ohne brauchbare Antwort bleibt kein vollständiger Turn übrig.
    assert [m["role"] for m in messages] == ["system"]


def test_trailing_question_is_trimmed():
    path = [
        {"role": "prompter", "text": "Warum ist der Himmel blau?"},
        {
            "role": "assistant",
            "text": "Weil die Luft kurze Wellenlängen stärker streut als lange – daher "
            "erscheint der Himmel blau, abends dann rot.",
        },
        {"role": "prompter", "text": "Und warum ist er nachts schwarz?"},
    ]
    messages = _path_to_messages(path)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_tool_row_payload_is_json_serializable():
    messages = normalize_messages(RAW_ROW)
    assert json.loads(json.dumps(messages, ensure_ascii=False))
