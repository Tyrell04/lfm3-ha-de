# Herkunft und Lizenzen

Dieses Repository enthält **eigenen Code** (MIT, siehe `LICENSE`). Es lädt
fremde Modelle und Daten, für die andere Lizenzen gelten.

## Basismodell

* `LiquidAI/LFM3-Omni-2B` bzw. `LiquidAI/LFM3-1.7B`
* Lizenz: **LFM Open License v1.0** – https://huggingface.co/LiquidAI/LFM3-Omni-2B/blob/main/LICENSE
* Das Fine-Tune erbt diese Lizenz. Kommerzielle Nutzung ist laut Lizenz
  erlaubt, solange die Bedingungen eingehalten werden (u. a. Namensnennung).

## Trainingsdaten

* **Home Assistant Tool Calling** – `mkenfenheuer/homeassistant-tool-calling`
  Lizenz **CC-BY-4.0**. Enthält synthetische Gespräche in sechs Sprachen;
  dieses Projekt nutzt ausschließlich `locale == "de"`.
  Das frühere Dataset `mkenfenheuer/homeassistant-tool-calling-de` ist dort
  aufgegangen und leitet auf denselben Datensatz weiter.
  Projekt: https://github.com/mKenfenheuer/homeassistant-llm

* **OpenAssistant OASST2** – `OpenAssistant/oasst2`
  Lizenz **Apache-2.0**. Genutzt wird eine gefilterte Teilmenge deutscher
  Threads (keine als gelöscht oder minderwertig markierten Nachrichten, keine
  synthetischen Turns).

## Werkzeugschemas

Die sechs Werkzeugschemas in `src/lfm3ha/tools_de.py` sind eigenständig
formuliert. Sie bilden dieselbe Funktionalität ab wie das Home-Assistant-
Projekt oben, dessen Quellcode **keine Lizenzdatei** enthält – es wurde
deshalb kein Code daraus übernommen.

## Zitieren

Wenn du dieses Projekt oder den Datensatz nutzt, nenne bitte:

```bibtex
@misc{schulz2026lfm3hade,
  title  = {lfm3-ha-de: German Home Assistant tool calling with LFM3},
  author = {Schulz, Marc Leon},
  year   = {2026},
  url    = {https://github.com/Tyrell04/lfm3-ha-de}
}
```
