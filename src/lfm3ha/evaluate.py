"""Auswertung: Werkzeugaufrufe parsen und gegen Gold vergleichen.

Metriken (bewusst getrennt, damit man sieht *wo* ein Modell scheitert):

* ``parse``     – Ausgabe enthält mindestens einen gültigen ``<tool_call>``
* ``function``  – Multimenge der Funktionsnamen stimmt
* ``exact``     – Namen **und** Argumente stimmen (Reihenfolge egal)
* ``answer``    – bei Rows ohne Werkzeug: nicht-leere deutsche Antwort

Argumente werden normalisiert, bevor verglichen wird: ``entity_id``-Listen
gelten als Mengen, ``"3"`` und ``3`` sind gleich, ``null``/``""`` fallen weg.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_\-]+)>\s*(.*?)</function>\s*</tool_call>",
    re.S,
)
PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_\-]+)>\s*(.*?)\s*</parameter>", re.S)

# Schlüssel, deren Listen als Mengen verglichen werden
SET_KEYS = {"entity_ids", "entity_id"}


def _coerce(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Liest LFM3-Werkzeugaufrufe aus einer Modellantwort."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text or ""):
        name, body = match.group(1), match.group(2)
        args: dict[str, Any] = {}
        for p_name, p_value in PARAM_RE.findall(body):
            args[p_name] = _coerce(p_value)
        calls.append({"name": name, "arguments": args})
    return calls


def normalize_args(args: Any) -> Any:
    """Vergleichsform der Argumente."""
    if isinstance(args, str):
        args = _coerce(args)
    if isinstance(args, list):
        return sorted((normalize_args(a) for a in args), key=_sort_key)
    if isinstance(args, dict):
        out = {}
        for key, value in args.items():
            if value is None or value == "":
                continue
            normalized = normalize_args(value)
            if normalized in ({}, [], ""):
                continue
            if key in SET_KEYS:
                if not isinstance(normalized, list):
                    normalized = [normalized]
                normalized = sorted({str(x) for x in normalized})
            out[key] = normalized
        return out
    if isinstance(args, str) and args.isdigit():
        return int(args)
    return args


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def call_signature(call: dict[str, Any]) -> str:
    return json.dumps(
        {"name": call["name"], "arguments": normalize_args(call.get("arguments", {}))},
        sort_keys=True,
        ensure_ascii=False,
    )


def gold_calls(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Erster Assistenten-Turn mit Werkzeugaufrufen."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return [
                {
                    "name": call["function"]["name"],
                    "arguments": call["function"].get("arguments", {}),
                }
                for call in msg["tool_calls"]
            ]
    return []


def score(gold: list[dict[str, Any]], prediction: str) -> dict[str, Any]:
    predicted = parse_tool_calls(prediction)
    result = {
        "parse": False,
        "function": False,
        "exact": False,
        "answer": False,
    }

    if not gold:
        # Reine Chat-Zeile: richtige Antwort ist Text ohne Werkzeugaufruf.
        result["answer"] = bool((prediction or "").strip()) and not predicted
        result["parse"] = result["answer"]
        result["function"] = result["answer"]
        result["exact"] = result["answer"]
        return result

    if not predicted:
        return result

    result["parse"] = True
    gold_names = Counter(c["name"] for c in gold)
    pred_names = Counter(c["name"] for c in predicted)
    result["function"] = gold_names == pred_names
    result["exact"] = Counter(call_signature(c) for c in gold) == Counter(
        call_signature(c) for c in predicted
    )
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fasst Einzelurteile zu Metriken zusammen – gesamt und je Kategorie."""
    tool_rows = [r for r in rows if r["gold"]]
    chat_rows = [r for r in rows if not r["gold"]]

    def rate(items, key):
        return round(sum(1 for i in items if i["scores"][key]) / len(items), 4) if items else None

    def block(items):
        return {
            "n": len(items),
            "parse": rate(items, "parse"),
            "function": rate(items, "function"),
            "exact": rate(items, "exact"),
        }

    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in tool_rows:
        by_category.setdefault(row.get("category") or "?", []).append(row)

    return {
        "overall": block(rows),
        "tool_calls": block(tool_rows),
        "chat": block(chat_rows),
        "by_category": {
            key: block(items)
            for key, items in sorted(by_category.items(), key=lambda kv: -len(kv[1]))
        },
    }
