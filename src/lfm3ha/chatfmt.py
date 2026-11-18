"""Chat-Template, Label-Maskierung und Collator.

Trainingsformat und Produktion müssen zeichengenau identisch sein. Deshalb
rendern wir jede Zeile mit dem Chat-Template des Modells
(``chat_template.jinja`` aus dem HF-Repo) und maskieren alles außer den
Assistenten-Turns mit ``-100``.

LFM3 markiert Assistenten-Bereiche mit ``{% generation %}``, daher liefert
``return_assistant_tokens_mask=True`` die Maske direkt. Für Modelle ohne
``generation``-Block gibt es einen Fallback über Präfix-Diffs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

IGNORE_INDEX = -100


def load_text_tokenizer(model_id: str):
    """Tokenizer für den Textpfad – auch bei Omni-Modellen.

    Omni-Modelle bringen einen ``AutoProcessor`` mit; der Tokenizer darin ist
    derselbe, den das Template für Text benutzt.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return processor.tokenizer


def _decode_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
    return tools or None


def _as_ids(result) -> list[int]:
    """``apply_chat_template`` liefert je nach Version Liste oder Dict."""
    if isinstance(result, dict):
        return list(result["input_ids"])
    return list(result)


def _spans_by_diff(tok, messages: list[dict[str, Any]], tools) -> list[tuple[int, int]]:
    """Fallback: Assistenten-Bereiche über Präfixlängen bestimmen."""
    spans = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        before = _as_ids(
            tok.apply_chat_template(
                messages[:i], tools=tools, tokenize=True, add_generation_prompt=False
            )
        )
        after = _as_ids(
            tok.apply_chat_template(
                messages[: i + 1], tools=tools, tokenize=True, add_generation_prompt=False
            )
        )
        spans.append((len(before), len(after)))
    return spans


def encode_example(
    tok,
    messages: Sequence[dict[str, Any]],
    tools: Any = None,
    max_len: int = 4096,
) -> dict[str, list[int]] | None:
    """Rendert eine Konversation zu ``input_ids`` + ``labels``.

    Gibt ``None`` zurück, wenn nach der Kürzung kein Assistenten-Token mehr
    übrig ist (die Zeile wäre ein reiner Prompt ohne Lernsignal).
    """
    messages = [dict(m) for m in messages]
    tool_list = _decode_tools(tools)

    try:
        enc = tok.apply_chat_template(
            messages,
            tools=tool_list,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = list(enc["input_ids"])
        mask = enc.get("assistant_masks")
        if mask is not None:
            mask = list(mask)
    except Exception:
        ids, mask = None, None

    if ids is None:
        ids = _as_ids(
            tok.apply_chat_template(
                messages, tools=tool_list, tokenize=True, add_generation_prompt=False
            )
        )
        mask = [0] * len(ids)
        for start, end in _spans_by_diff(tok, messages, tool_list):
            for i in range(start, min(end, len(ids))):
                mask[i] = 1

    if len(ids) > max_len:
        ids = ids[:max_len]
        mask = mask[:max_len]

    labels = [tok_id if m else IGNORE_INDEX for tok_id, m in zip(ids, mask, strict=False)]
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def build_example(tok, row: dict[str, Any], max_len: int):
    """Row aus dem HF-Dataset (``messages`` als JSON-String) -> Tensoren."""
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    return encode_example(tok, messages, row.get("tools"), max_len=max_len)


class PadCollator:
    """Padding-Collator; ``labels`` behalten ``-100``."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = width - n
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
            attention.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def render_preview(tok, messages, tools=None) -> str:
    """Gerenderter Prompt – für Tests und zum Draufschauen."""
    return tok.apply_chat_template(
        messages, tools=_decode_tools(tools), tokenize=False, add_generation_prompt=False
    )
