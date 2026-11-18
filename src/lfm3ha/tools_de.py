"""Die sechs Werkzeuge des Home-Assistant-Assistenten (deutsch).

Die Schemas bilden ab, was das Modell im Training sieht und was zur Laufzeit
an ``apply_chat_template(..., tools=...)`` übergeben wird. Wer hier etwas
ändert, muss neu trainieren – die Werkzeugliste steht wörtlich im Prompt.

Die Struktur ist an die Semantik des Datensatzes
``mkenfenheuer/homeassistant-tool-calling`` angelehnt: ``control`` ruft
Home-Assistant-Dienste auf, ``get_state`` liest Zustände, ``music_library``
liest die Medienbibliothek, ``timer`` steuert Timer auf dem Sprachgerät,
``get_time`` liefert Uhrzeit/Datum, ``broadcast`` macht eine Durchsage.
"""

from __future__ import annotations

import copy
from typing import Any

# Dienste, die das Modell aufrufen darf. Bewusst als Text in der Beschreibung,
# nicht als Enum: Home Assistant kennt deutlich mehr Dienste, und ein Enum würde
# das Modell bei Sonderfällen blockieren.
SERVICES: tuple[str, ...] = (
    "light.turn_on",
    "light.turn_off",
    "switch.turn_on",
    "switch.turn_off",
    "fan.turn_on",
    "fan.turn_off",
    "fan.set_percentage",
    "cover.open_cover",
    "cover.close_cover",
    "cover.set_cover_position",
    "climate.set_temperature",
    "climate.turn_on",
    "climate.turn_off",
    "humidifier.turn_on",
    "humidifier.turn_off",
    "humidifier.set_humidity",
    "lock.lock",
    "lock.unlock",
    "media_player.media_play",
    "media_player.media_pause",
    "media_player.media_next_track",
    "media_player.media_previous_track",
    "media_player.volume_set",
    "media_player.volume_up",
    "media_player.volume_down",
    "media_player.search_and_play",
    "media_player.seek_relative",
    "todo.add_item",
    "todo.complete_item",
    "todo.remove_item",
    "vacuum.start",
    "vacuum.return_to_base",
)

MEDIA_TYPES: tuple[str, ...] = (
    "playlist",
    "artist",
    "album",
    "track",
    "radio",
    "podcast",
)

TIMER_ACTIONS: tuple[str, ...] = (
    "start",
    "cancel",
    "cancel_all",
    "pause",
    "resume",
    "add",
    "subtract",
    "status",
)


def build_tools() -> list[dict[str, Any]]:
    """Frische Kopie der Werkzeugliste (OpenAI-Format)."""
    duration = {"type": "integer", "minimum": 0}
    return copy.deepcopy(
        [
            {
                "type": "function",
                "function": {
                    "name": "control",
                    "description": (
                        "Steuert ein oder mehrere Geräte über Home-Assistant-Dienste. "
                        "Jede Aktion nennt einen Dienst (domain.dienst) und entweder "
                        "entity_id(s) oder – bei Lichtern – einen Bereich. Musik wird mit "
                        "media_player.search_and_play gesucht und abgespielt, Spulen mit "
                        "media_player.seek_relative, Listen ändern todo.add_item, "
                        "todo.complete_item und todo.remove_item. Erlaubte Dienste: "
                        + ", ".join(SERVICES)
                        + "."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "service": {
                                            "type": "string",
                                            "description": "Dienst im Format domain.dienst",
                                        },
                                        "entity_id": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Betroffene entity_ids",
                                        },
                                        "area": {
                                            "type": "string",
                                            "description": (
                                                "Statt entity_id: alle Lichter dieses "
                                                'Bereichs, "*" für das ganze Haus'
                                            ),
                                        },
                                        "data": {
                                            "type": "object",
                                            "description": (
                                                "Optionale Parameter, z. B. brightness_pct, "
                                                "brightness_step_pct, color_name, "
                                                "color_temp_kelvin, position, temperature, "
                                                "humidity, percentage, volume_level, "
                                                "is_volume_muted, shuffle, repeat "
                                                "(off, one, all), code, query, media_type ("
                                                + ", ".join(MEDIA_TYPES)
                                                + "), artist, seconds, item"
                                            ),
                                        },
                                    },
                                    "required": ["service"],
                                },
                            }
                        },
                        "required": ["actions"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_state",
                    "description": (
                        "Liefert Zustand und wichtige Attribute von Geräten und Sensoren – "
                        "entweder für entity_ids oder für alle Geräte einer Domain in einem "
                        'Bereich (domain + area, "*" für das ganze Haus, optional '
                        "device_class, z. B. window). Wetter liefert den Rest des Tages "
                        "(later_today) und die Vorhersage (forecast), To-do-Listen ihre "
                        "offenen Einträge."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_ids": {"type": "array", "items": {"type": "string"}},
                            "domain": {"type": "string"},
                            "area": {"type": "string"},
                            "device_class": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "music_library",
                    "description": (
                        "Liest die Medienbibliothek eines Players (entity_id): Playlists, "
                        "Künstler, Alben, Titel, Sender oder Podcasts (media_type), optional "
                        "gefiltert nach search. Die Namen sind die query für "
                        "media_player.search_and_play."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "media_type": {"type": "string", "enum": list(MEDIA_TYPES)},
                            "search": {"type": "string"},
                        },
                        "required": ["entity_id", "media_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "timer",
                    "description": (
                        "Timer auf dem Sprachgerät: starten (start mit hours, minutes, "
                        "seconds, optional name und command für einen Befehl nach Ablauf), "
                        "abbrechen (cancel, cancel_all), pausieren (pause), fortsetzen "
                        "(resume), verlängern (add), verkürzen (subtract), abfragen (status)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": list(TIMER_ACTIONS)},
                            "hours": duration,
                            "minutes": duration,
                            "seconds": duration,
                            "name": {"type": "string"},
                            "command": {"type": "string"},
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Liefert die aktuelle Uhrzeit, den Wochentag und das Datum.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "broadcast",
                    "description": "Spricht eine Durchsage auf allen Sprachgeräten im Haus.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            },
        ]
    )


TOOLS: list[dict[str, Any]] = build_tools()
TOOL_NAMES: list[str] = [t["function"]["name"] for t in TOOLS]


def tools_json() -> str:
    """Kompakte JSON-Darstellung – identisch zu dem, was das Template rendert."""
    import json

    return json.dumps(TOOLS, ensure_ascii=False)
