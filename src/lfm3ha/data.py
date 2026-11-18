"""Datenaufbereitung für das deutsche Home-Assistant-Modell.

Zwei Quellen werden zu einem Chat-Datensatz im LFM3-Format verschmolzen:

1. ``mkenfenheuer/homeassistant-tool-calling`` – synthetische Gespräche mit
   echten Home-Assistant-Werkzeugaufrufen in sechs Sprachen. Wir behalten
   ``locale == "de"``. Das frühere deutsche Einzeldataset
   (``homeassistant-tool-calling-de``) ist dort aufgegangen.
2. ``OpenAssistant/oasst2`` – deutsche Threads für Sprachgefühl und
   Antwortstil. Bewusst nur ein kleiner Anteil, sonst verliert das Modell die
   Werkzeugdisziplin (Format-Drift).

Beide Quellen sind lizenziert (CC-BY-4.0 bzw. Apache-2.0); die Datenkarte in
``docs/DATASET_CARD.md`` nennt die Details.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

HA_DATASET = "mkenfenheuer/homeassistant-tool-calling"
OASST_DATASET = "OpenAssistant/oasst2"

SYSTEM_CHAT_DE = (
    "Du bist ein hilfsbereiter, freundlicher Assistent. "
    "Antworte immer auf Deutsch und in ganzen Sätzen."
)

# Zeichen, die auf kaputte Serialisierung hindeuten.
_BROKEN = re.compile(r"(\{\{|\}\}|\\u0000|<\|endoftext\|>|None)")

# Typische OASST2-Floskeln, die kein gutes Deutsch und keine gute Antwort sind.
_REFUSALS = (
    "als ki",
    "als sprachmodell",
    "als konversations-ki",
    "als textbasiertes",
    "ich kann keine bilder",
    "ich habe keine möglichkeit",
    "ich bin nur ein",
)


def _good_answer(text: str) -> bool:
    """Filtert OASST2-Antworten, die Stil oder Inhalt kaputt machen."""
    lowered = text.lower()
    if not 40 <= len(text) <= 1200:
        return False
    return not any(phrase in lowered for phrase in _REFUSALS)


# --------------------------------------------------------------------------- #
# Normalisierung
# --------------------------------------------------------------------------- #
def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macht aus einem Roh-Row template-taugliche Nachrichten.

    Wichtigster Schritt: ``tool_calls[].function.arguments`` liegt im Datensatz
    als JSON-**String** vor. Das LFM3-Template bricht bei Strings mit einer
    Ausnahme ab ("Tool call arguments must be a mapping"), also parsen wir hier.
    """
    out: list[dict[str, Any]] = []
    for raw in messages:
        msg = dict(raw)
        content = msg.get("content")
        msg["content"] = "" if content is None else content

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            parsed_calls = []
            for call in tool_calls:
                call = dict(call)
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                call["function"] = fn
                parsed_calls.append(call)
            msg["tool_calls"] = parsed_calls
        out.append(msg)
    return out


def is_usable(messages: list[dict[str, Any]]) -> bool:
    """Mindestanforderung an einen Trainingsrow."""
    if not messages or messages[0].get("role") != "system":
        return False
    roles = [m.get("role") for m in messages]
    if "assistant" not in roles:
        return False
    last_assistant = [m for m in messages if m.get("role") == "assistant"][-1]
    answer = (last_assistant.get("content") or "").strip()
    if not answer and not last_assistant.get("tool_calls"):
        return False
    if answer and (len(answer) < 2 or _BROKEN.search(answer)):
        return False
    for m in messages:
        if m.get("role") == "user" and not (m.get("content") or "").strip():
            return False
    return True


def categories(messages: list[dict[str, Any]]) -> str:
    """Grobe Aufgabenklasse – für die Auswertung nach Kategorie."""
    names = [
        call["function"]["name"]
        for m in messages
        for call in (m.get("tool_calls") or [])
    ]
    if not names:
        return "chat"
    return "+".join(sorted(set(names)))


# --------------------------------------------------------------------------- #
# Quelle 1: Home Assistant
# --------------------------------------------------------------------------- #
def load_ha_de(
    limit: int | None = None,
    streaming: bool = False,
    splits: tuple[str, ...] = ("train", "test"),
):
    """Lädt den deutschen Teil des HA-Datensatzes."""
    from datasets import load_dataset

    result = {}
    for split in splits:
        ds = load_dataset(HA_DATASET, split=split, streaming=streaming)
        ds = ds.filter(lambda r: r["locale"] == "de")
        rows = []
        for row in ds:
            messages = normalize_messages(row["messages"])
            if not is_usable(messages):
                continue
            meta = row.get("meta") or {}
            rows.append(
                {
                    "messages": messages,
                    "locale": "de",
                    "source": "homeassistant-tool-calling",
                    "category": str(meta.get("cat") or categories(messages)),
                    "case": str(meta.get("case") or ""),
                }
            )
            if limit and len(rows) >= limit:
                break
        result[split] = rows
    return result


# --------------------------------------------------------------------------- #
# Quelle 2: OASST2 (deutsche Threads)
# --------------------------------------------------------------------------- #
def _best_path(node: dict[str, Any], children: dict[str, list[dict[str, Any]]]):
    """Folgt vom Knoten aus jeweils der bestbewerteten Antwort."""
    path = [node]
    while children.get(node["message_id"]) and len(path) < 6:
        node = max(
            children[node["message_id"]],
            key=lambda m: (m.get("rank") or 0, m.get("review_count") or 0),
        )
        path.append(node)
    return path


def _iter_threads(root: dict[str, Any], children: dict[str, list[dict[str, Any]]]):
    """Alle brauchbaren Gesprächsfäden eines Wurzelknotens.

    OASST2 ist ein Baum: unter einer Frage hängen mehrere Antworten, unter
    jeder wieder mehrere. Eine Wurzel kann also mehrere unabhängige Gespräche
    tragen – wir laufen jede Antwort der zweiten Ebene einmal ab, statt nur den
    besten Zweig zu nehmen. Sonst bliebe vom deutschen Teil kaum etwas übrig.
    """
    replies = children.get(root["message_id"], [])
    if not replies:
        return
    ordered = sorted(
        replies, key=lambda m: (m.get("rank") or 0, m.get("review_count") or 0), reverse=True
    )
    for reply in ordered:
        yield [root, *_best_path(reply, children)]


def _path_to_messages(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_CHAT_DE}]
    for msg in path:
        role = "user" if msg["role"] == "prompter" else "assistant"
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if role == "assistant" and not _good_answer(text):
            # Schlechte Antwort beendet den Faden: alles danach baut darauf auf.
            break
        if messages[-1]["role"] == role:  # gleiche Rolle zweimal -> ersetzen
            if len(text) > len(messages[-1]["content"]):
                messages[-1]["content"] = text
            continue
        messages.append({"role": role, "content": text})
    # Endet der Faden auf einer Frage, ist der letzte Assistenten-Turn die
    # Antwort – die abschließende Frage wird verworfen.
    while len(messages) > 1 and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages


def load_oasst2_de(limit: int = 4000, min_rank: int = 0) -> list[dict[str, Any]]:
    """Baut aus dem OASST2-Baum deutsche, mehrstufige Gespräche."""
    from datasets import load_dataset

    frames = []
    for split in ("train", "validation"):
        ds = load_dataset(OASST_DATASET, split=split)
        for row in ds:
            if row.get("lang") != "de" or row.get("deleted"):
                continue
            if row.get("review_result") is False:
                continue
            if (row.get("rank") or 0) < min_rank:
                continue
            if row.get("synthetic"):
                continue  # maschinelle Turns meiden – wir wollen echtes Deutsch
            text = (row.get("text") or "").strip()
            if len(text) < 3 or _BROKEN.search(text):
                continue
            frames.append(row)

    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots: list[dict[str, Any]] = []
    for msg in frames:
        if msg.get("parent_id") is None:
            roots.append(msg)
        else:
            children[msg["parent_id"]].append(msg)

    random.Random(42).shuffle(roots)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in _iter_threads(root, children):
            messages = _path_to_messages(path)
            if len(messages) < 3:
                continue
            if not any(m["role"] == "assistant" for m in messages):
                continue
            if not is_usable(messages):
                continue
            fingerprint = _fingerprint({"messages": messages})
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append(
                {
                    "messages": messages,
                    "locale": "de",
                    "source": "oasst2",
                    "category": "chat",
                    "case": "",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


# --------------------------------------------------------------------------- #
# Zusammenbau
# --------------------------------------------------------------------------- #
def _fingerprint(row: dict[str, Any]) -> str:
    """Stabiler Hash der Konversation (über Läufe hinweg reproduzierbar)."""
    text = "".join(str(m.get("content") or "") for m in row["messages"])
    text += "".join(
        json.dumps(c.get("function"), sort_keys=True, ensure_ascii=False)
        for m in row["messages"]
        for c in (m.get("tool_calls") or [])
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for row in rows:
        fp = _fingerprint(row)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(row)
    return out


def mix(
    ha_rows: list[dict[str, Any]],
    chat_rows: list[dict[str, Any]],
    ratio: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Mischt Chat-Anteil bei, ohne die Werkzeugverteilung zu verschieben."""
    if not chat_rows or ratio <= 0:
        return ha_rows
    wanted = min(len(chat_rows), int(len(ha_rows) * ratio / max(1e-9, 1 - ratio)))
    rng = random.Random(seed)
    picked = rng.sample(chat_rows, wanted) if wanted < len(chat_rows) else chat_rows
    combined = ha_rows + picked
    rng.shuffle(combined)
    return combined


def to_hf(rows: list[dict[str, Any]], tools_json: str):
    """Wandelt Zeilen in ein HF-Dataset (messages als JSON-Spalte)."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "messages": [json.dumps(r["messages"], ensure_ascii=False) for r in rows],
            "tools": [tools_json if r["category"] != "chat" else "" for r in rows],
            "locale": [r["locale"] for r in rows],
            "source": [r["source"] for r in rows],
            "category": [r["category"] for r in rows],
            "case": [r["case"] for r in rows],
        }
    )


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    turn_counts: list[int] = []
    for row in rows:
        counts[row["category"]] += 1
        turn_counts.append(len(row["messages"]))
    return {
        "rows": len(rows),
        "by_category": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "avg_messages": round(sum(turn_counts) / max(1, len(turn_counts)), 2),
        "tool_rows": sum(1 for r in rows if r["category"] != "chat"),
    }
