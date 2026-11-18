# lfm3-ha-de 🇩🇪🏠

**Ein deutschsprachiger Home-Assistant-Assistent auf Liquid AI LFM3.**
Befehl rein, Werkzeugaufruf raus – trainiert auf echten Smart-Home-
Konversationen, nicht auf Übersetzungen.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tyrell04/lfm3-ha-de/blob/main/notebooks/colab_train.ipynb)
[![CI](https://github.com/Tyrell04/lfm3-ha-de/actions/workflows/ci.yml/badge.svg)](https://github.com/Tyrell04/lfm3-ha-de/actions/workflows/ci.yml)
[![Modell](https://img.shields.io/badge/🤗%20Modell-marc04%2Flfm3--ha--de-yellow)](https://huggingface.co/marc04/lfm3-ha-de)
[![Datensatz](https://img.shields.io/badge/🤗%20Daten-marc04%2Fhomeassistant--tool--calling--lfm3--de-yellow)](https://huggingface.co/datasets/marc04/homeassistant-tool-calling-lfm3-de)

```text
„Mach das Licht im Wohnzimmer an."
        ↓
<tool_call>
<function=control>
<parameter=actions>
[{"service": "light.turn_on", "entity_id": ["light.wohnzimmer_decke"]}]
</parameter>
</function>
</tool_call>
```

---

## Was hier drin ist

| Baustein | Datei | Zweck |
|---|---|---|
| Werkzeugschemas | `src/lfm3ha/tools_de.py` | die sechs HA-Werkzeuge, deutsche Beschreibungen |
| Datenaufbereitung | `src/lfm3ha/data.py` | HA-`locale=de` + OASST2-de → LFM3-Chatformat |
| Chat/Template | `src/lfm3ha/chatfmt.py` | Rendering mit dem echten LFM3-Template, Label-Maskierung |
| Training | `src/lfm3ha/train.py` | LoRA-SFT, GPU-adaptiv (bf16/fp16, optional 4-bit) |
| Auswertung | `src/lfm3ha/evaluate.py`, `infer.py` | parse / function / exact je Kategorie |
| Upload | `src/lfm3ha/push_model.py` | Adapter verschmelzen, Modellkarte, HF-Push |
| Colab | `notebooks/colab_train.ipynb` | ein Klick, Ende-zu-Ende |

## Warum LFM3 und nicht LFM2.5

LFM3 hat ein **eigenes Werkzeugformat** – eine Obermenge von `qwen3_coder`:

```text
<tool_call><function=name><parameter=arg>wert</parameter></function></tool_call>
```

LFM2/LFM2.5 nutzen dagegen `<|tool_call_start|>[name(arg="wert")]<|tool_call_end|>`.
Liquid Fine-Tune unterstützt offiziell bisher LFM2.5; für LFM3 ist der
Transformers+PEFT-Weg (offizielles Cookbook-Notebook) der vorgesehene. Genau
den setzt dieses Repo um – inklusive der Fallstricke, die dort nicht stehen
(siehe [Troubleshooting](docs/TROUBLESHOOTING.md)).

## Modellwahl

| Basismodell | Größe | Wann |
|---|---|---|
| `LiquidAI/LFM3-Omni-2B` | 2,17 B (4,3 GB) | Standard hier: ein Modell für Text **und** später Audio |
| `LiquidAI/LFM3-1.7B` | 1,7 B (3,5 GB) | wenn es nur Text sein soll – kleiner, schneller, weniger VRAM |

Beide teilen Backbone und Chat-Template, ein Wechsel ist ein Flag:
`--base-model LiquidAI/LFM3-1.7B`.

## Schnellstart

### Auf Colab (empfohlen)

1. [Notebook öffnen](https://colab.research.google.com/github/Tyrell04/lfm3-ha-de/blob/main/notebooks/colab_train.ipynb)
2. Laufzeit → **GPU** (T4 reicht für einen Smoke-Test; L4/A100 für den echten Lauf)
3. Colab-Secret `HF_TOKEN` anlegen (Write-Token von huggingface.co/settings/tokens)
4. Zellen von oben nach unten ausführen

### Lokal

```bash
git clone https://github.com/Tyrell04/lfm3-ha-de.git
cd lfm3-ha-de
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
# 1) Daten bauen (Smoke-Test: --limit 200)
lfm3ha prepare --limit 200 --oasst-limit 200

# 2) Trainieren (Smoke-Test)
lfm3ha train --dataset data/lfm3-ha-de --epochs 1 --max-train-rows 200

# 3) Messen
lfm3ha eval --adapter outputs/lfm3-ha-de --dataset data/lfm3-ha-de --limit 50

# 4) Reden
lfm3ha chat --adapter outputs/lfm3-ha-de
```

Ohne `--dataset` nimmt die CLI den veröffentlichten Datensatz
`marc04/homeassistant-tool-calling-lfm3-de`.

## Der Datensatz in Zahlen

Gemessen am deutschen Anteil der Quelle (100 zufällige Zeilen, Tokenizer von
LFM3-Omni-2B, Werkzeugschemas im Prompt):

| Kennzahl | Wert |
|---|---|
| Prompt-Länge p50 | 2.290 Token |
| p90 / p99 / max | 3.073 / 3.815 / 5.173 Token |
| Anteil Assistenten-Token | 2–12 % (der Rest ist Geräteliste + Schemas) |
| Standard `max_seq_len` | **4.096** (kürzt < 1 % der Zeilen ab) |

Der lange Prompt ist Absicht: in Produktion steht die echte Geräteliste
drin. Er kostet Rechenzeit, aber genau daran lernt das Modell, Namen auf
`entity_id`s abzubilden statt zu raten.

## Training in Kurzform

```bash
lfm3ha prepare                     # baut data/lfm3-ha-de (+ optional HF-Push)
lfm3ha train --epochs 2            # LoRA r=16, alpha=32, LR 2e-4, cosine
lfm3ha eval --out outputs/metrics.json
lfm3ha push --adapter outputs/lfm3-ha-de --repo-id marc04/lfm3-ha-de
```

| Hyperparameter | Wert | Begründung |
|---|---|---|
| LoRA r / alpha / dropout | 16 / 32 / 0.05 | offizielles Cookbook; reicht für Format + Vokabular |
| Zielmodule | `q,k,v,o_proj`, `in_proj/out_proj`, `w1,w2,w3`, `gate_proj` | Attention, gated Convolutions, MLP |
| Lernrate | 2e-4 | LoRA-üblich, mit cosine + 3 % Warmup |
| Epochen | 2 | mehr overfittet die synthetischen Häuser |
| Batch | 8 × 4 (A100/L4) · 1 × 16 (T4) | effektiv 32 |
| Precision | bf16, sonst fp16 | T4 kann kein bf16 |
| Labels | nur Assistenten-Turns | Systemprompt und Werkzeugantworten werden nicht gelernt |

Details und Begründungen: [docs/TRAINING.md](docs/TRAINING.md).

## Ergebnisse

> Wird nach dem ersten vollständigen Lauf gefüllt. Die Spalten sind bewusst
> getrennt – „parse“ kann hoch sein, während „exact“ einbricht.

| Modell | n | parse | function | exact |
|---|---|---|---|---|
| LFM3-Omni-2B (Basismodell, ohne Training) | — | — | — | — |
| + LoRA (dieses Repo) | — | — | — | — |

Reproduzieren:

```bash
lfm3ha eval --base-model LiquidAI/LFM3-Omni-2B --limit 300 --out outputs/base.json
lfm3ha eval --adapter outputs/lfm3-ha-de --limit 300 --out outputs/lora.json
```

## Grenzen

* **Nur Deutsch.** Kein Multilingual-Training, keine Übersetzungsaufgabe.
* **Synthetische Haushalte.** Das Modell lernt das Schema, nicht deine Geräte.
* **Prompt-Abhängigkeit.** Ohne Geräteliste im Systemprompt sinkt die
  Qualität stark – das ist so gewollt und entspricht der Laufzeit.
* **Kein Audio.** Für Sprache kommt ASR davor (oder ein Audio-Training des
  Omni-Modells, siehe [docs/TRAINING.md](docs/TRAINING.md#audio-später)).
* **Keine Sicherheitsgarantie.** Werkzeugaufrufe müssen serverseitig gegen
  Berechtigungen geprüft werden – ein LLM ist keine Firewall.

## Struktur

```text
configs/            Standardwerte
docs/               Datenkarte, Modellkarte, Training, Troubleshooting
notebooks/          Colab-Notebook
src/lfm3ha/         Paket (CLI: lfm3ha)
tests/              Unit-Tests + Template-Vertragstest
```

## Lizenz

Code: MIT. Basismodell: LFM Open License v1.0. Daten: CC-BY-4.0 /
Apache-2.0. Vollständig in [NOTICE.md](NOTICE.md).
