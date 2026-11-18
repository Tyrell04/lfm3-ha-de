---
license: cc-by-4.0
language:
  - de
task_categories:
  - text-generation
tags:
  - home-assistant
  - tool-calling
  - function-calling
  - smart-home
  - lfm3
  - liquid-ai
  - german
size_categories:
  - 10K<n<100K
---

# Home Assistant Tool Calling – LFM3, Deutsch

29.048 Trainingszeilen · 500 Validierung · 300 Test · deutschsprachig.

Deutschsprachiges Trainingsset für Werkzeugaufrufe eines Smart-Home-
Assistenten, aufbereitet für **Liquid AI LFM3** (Chat-Template mit
`<tool_call>`-Blöcken). Erzeugt von
[lfm3-ha-de](https://github.com/Tyrell04/lfm3-ha-de).

Dieses Repository enthält **aufbereitete Daten**, keinen neuen Inhalt: es
führt zwei öffentliche Datensätze zusammen und bringt sie in das Format, das
LFM3 beim Training erwartet.

## Quellen

| Quelle | Rolle | Lizenz |
|---|---|---|
| [`mkenfenheuer/homeassistant-tool-calling`](https://huggingface.co/datasets/mkenfenheuer/homeassistant-tool-calling), `locale == "de"` | Aufgabe: Befehl → Werkzeugaufruf | CC-BY-4.0 |
| [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2), deutsche Threads | Sprachgefühl, Antwortstil | Apache-2.0 |

Das frühere Dataset `mkenfenheuer/homeassistant-tool-calling-de` existiert
nicht mehr eigenständig – es ist im mehrsprachigen Datensatz aufgegangen und
leitet dorthin weiter. Deshalb wird dort über die Spalte `locale` gefiltert.

## Aufbau

| Spalte | Typ | Bedeutung |
|---|---|---|
| `messages` | string (JSON) | Konversation als Liste von Nachrichten |
| `tools` | string (JSON) | Werkzeugschemas; leer bei reinen Chat-Zeilen |
| `locale` | string | immer `de` |
| `source` | string | `homeassistant-tool-calling` oder `oasst2` |
| `category` | string | Werkzeugklasse (`control`, `get_state`, `timer`, …) oder `chat` |
| `case` | string | Fallname des Generators, z. B. `command:i_vacuum` |

`messages` folgt dem OpenAI-Schema: `system` → `user` → `assistant`
(mit `tool_calls`) → `tool` → `assistant`. Die Argumente der Werkzeugaufrufe
sind **Objekte**, keine JSON-Strings – das LFM3-Template lehnt Strings ab.

## Verarbeitung

1. Home-Assistant-Zeilen mit `locale == "de"` behalten.
2. `tool_calls[].function.arguments` von JSON-String zu Mapping parsen.
3. Zeilen ohne brauchbare Antwort verwerfen (leer, kaputte Serialisierung).
4. OASST2: Baum zu Gesprächen zusammensetzen, dabei Nachrichten mit
   `deleted`, negativem `review_result`, negativem `rank` und synthetische
   Turns aussortieren.
5. Duplikate über einen Hash der Konversation entfernen.
6. Chat-Zeilen beimischen (Zielanteil 8 %, gedeckelt durch das Angebot:
   OASST2-de liefert nach Filterung 1.228 Gespräche, das sind 4,2 %) – genug
   für den Ton, wenig genug, dass die Werkzeugdisziplin nicht leidet.
7. Train/Validation/Test trennen; Test stammt aus dem Test-Split der
   HA-Quelle und wird nicht angefasst.

## Nutzung

```python
import json
from datasets import load_dataset

ds = load_dataset("REPO_ID")
row = ds["train"][0]
messages = json.loads(row["messages"])
tools = json.loads(row["tools"]) if row["tools"] else None
```

## Grenzen

* Die HA-Gespräche sind **synthetisch**. Geräte, Namen und Bereiche sind
  zufällig erzeugt; das Modell lernt, die Geräteliste im Systemprompt zu
  lesen, nicht einen konkreten Haushalt.
* OASST2 ist eine Spende-Sammlung: Stil und Qualität schwanken. Der Anteil
  ist bewusst klein.
* Keine Audio-Daten. Für Sprachsteuerung muss vorgelagert ASR laufen (oder
  ein Omni-Modell auf Audio trainiert werden).
* Der Systemprompt ist lang (Geräteliste + Werkzeugschemas ≈ 2.000 Token).
  Das ist Produktionsrealität, kostet aber Trainingszeit.

## Lizenz und Namensnennung

Die aufbereiteten Daten stehen unter **CC-BY-4.0** (Lizenz der HA-Quelle);
OASST2-Beiträge stehen unter **Apache-2.0**. Bei Weitergabe bitte beide
Quellen nennen:

* mKenfenheuer, *Home Assistant Tool Calling* (CC-BY-4.0)
* OpenAssistant, *OASST2* (Apache-2.0)
* sowie dieses Repository als Aufbereitung.
