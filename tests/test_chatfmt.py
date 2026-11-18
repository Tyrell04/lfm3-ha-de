"""Vertragstest: rendert unsere Rows mit dem echten LFM3-Chat-Template.

Ohne Netz oder ohne ``jinja2`` wird übersprungen – der Test braucht weder
Transformers noch eine GPU.
"""

from __future__ import annotations

import json
import re
import urllib.request

import pytest

from lfm3ha.chatfmt import IGNORE_INDEX, encode_example
from lfm3ha.tools_de import TOOLS

TEMPLATE_URL = "https://huggingface.co/LiquidAI/LFM3-Omni-2B/raw/main/chat_template.jinja"

MESSAGES = [
    {"role": "system", "content": "Du bist der Sprachassistent eines Smart Homes."},
    {"role": "user", "content": "Mach das Licht im Wohnzimmer an"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "control",
                    "arguments": {
                        "actions": [
                            {
                                "service": "light.turn_on",
                                "entity_id": ["light.wohnzimmer_decke"],
                            }
                        ]
                    },
                },
            }
        ],
    },
]


def _load_template() -> str | None:
    try:
        with urllib.request.urlopen(TEMPLATE_URL, timeout=20) as response:  # noqa: S310
            text = response.read().decode("utf-8")
    except Exception:
        return None
    # {% generation %} ist eine Transformers-Erweiterung, kein Jinja2.
    return re.sub(r"{%-?\s*(?:end)?generation\s*-?%}", "", text)


def test_template_renders_native_tool_calls():
    jinja2 = pytest.importorskip("jinja2")
    template = _load_template()
    if not template:
        pytest.skip("Chat-Template nicht erreichbar (offline)")

    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    rendered = env.from_string(template).render(
        messages=MESSAGES,
        tools=TOOLS,
        bos_token="<|startoftext|>",
        preserve_thinking=False,
        enable_thinking=False,
    )

    assert "<tools>" in rendered, "Werkzeugdefinitionen fehlen im Systemprompt"
    assert "<tool_call>" in rendered
    assert "<function=control>" in rendered
    assert "<parameter=actions>" in rendered
    assert "light.wohnzimmer_decke" in rendered
    assert "Du bist der Sprachassistent" in rendered


class StubTokenizer:
    """Minimaler Tokenizer, der das Maskenverhalten prüfbar macht."""

    pad_token_id = 0

    def __init__(self, ids, mask):
        self.ids = ids
        self.mask = mask

    def apply_chat_template(self, messages, tools=None, tokenize=True, **kwargs):
        if kwargs.get("return_dict"):
            return {"input_ids": list(self.ids), "assistant_masks": list(self.mask)}
        return list(self.ids)


def test_labels_mask_everything_but_assistant_tokens():
    tok = StubTokenizer(ids=[10, 11, 12, 13], mask=[0, 0, 1, 1])
    encoded = encode_example(tok, MESSAGES, TOOLS, max_len=64)
    assert encoded["input_ids"] == [10, 11, 12, 13]
    assert encoded["labels"] == [IGNORE_INDEX, IGNORE_INDEX, 12, 13]


def test_rows_with_only_masked_tokens_are_dropped():
    tok = StubTokenizer(ids=[10, 11], mask=[0, 0])
    assert encode_example(tok, MESSAGES, TOOLS, max_len=64) is None


def test_truncation_that_cuts_the_answer_is_dropped():
    tok = StubTokenizer(ids=[10, 11, 12, 13], mask=[0, 0, 1, 1])
    assert encode_example(tok, MESSAGES, TOOLS, max_len=2) is None
