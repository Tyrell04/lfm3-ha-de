# Training

## Hardware

| GPU | VRAM | Einstellung | Recheneinheiten/Std. | Voller Lauf (geschätzt) | Einheiten pro Lauf |
|---|---|---|---|---|---|
| Colab T4 (frei) | 16 GB | `--batch-size 1 --grad-accum 16` fp16 | ~1,1 | 20–24 h | ~23–28 (bzw. gratis) |
| Colab L4 | 24 GB | `--batch-size 4 --grad-accum 8` bf16 | ~1,6 | 9–11 h | ~15–18 |
| Colab A100 40 GB | 40 GB | Standard (`8 × 4`) bf16 | ~5,3 | 3,5–4,5 h | ~19–24 |
| Colab A100 80 GB | 80 GB | Standard (`8 × 4`) bf16 | ~7,0 | 3–4 h | ~21–28 |
| Colab G4 (Pro+) | 96 GB | Standard | ~8,8 | 2–2,5 h | ~18–22 |
| Colab H100 (Pro+) | 80 GB | Standard | ~18 | 1,5–2 h | ~27–36 |
| RTX 3090/4090 (lokal) | 24 GB | `--batch-size 4 --grad-accum 8` | – | 18–22 h | – |

> Google veröffentlicht die Verbrauchsraten nicht; die Spalte sind
> Nutzermessungen (Stand 2026). Die Dauern sind aus der Tokenmenge geschätzt:
> 29.048 Zeilen × 2 Epochen × ~2.400 Token ≈ 140 Mio. Token, also rund
> 1,6 · 10¹⁸ FLOP bei 6 · N · D. Beides zusammen auf ±50 % genau – miss es
> einmal selbst nach (siehe unten), statt dich darauf zu verlassen.

**Was das für die Wahl heißt.** Ein Unit-Preis sagt nichts; entscheidend sind
Einheiten *pro fertigem Lauf*. Die Verbrauchsrate wächst ungefähr so schnell wie
der Durchsatz, deshalb liegen L4 (~16) und A100 (~21) nah beieinander, obwohl
die A100 die 2,4-fache Rate hat. Die A100 gewinnt trotzdem: bei 4 Stunden ist
der Lauf in einer Sitzung fertig, bei 10 Stunden auf der L4 brauchst du
`--resume` und zwei Anläufe. G4 und H100 sind für ein 2B-LoRA-Gehirn Geld-
verbrennung – der Engpass ist Speicherbandbreite und Fixkosten, nicht Rechen-
leistung. T4 ist gratis, aber 20+ Stunden fp16 mit Abbruchrisiko – nur für den
Smoke-Test.

**Empfehlung:** A100 40 GB für den echten Lauf, L4 wenn dir Einheiten wichtiger
sind als Zeit, T4 nur für `SMOKE = True`. Pro (100 Einheiten/Monat) reicht für
beide Varianten; Pro+ (600) brauchst du nur, wenn du mehrere Läufe vergleichst.

**Selbst nachmessen** (dauert eine Minute, ersetzt jede Schätzung):

1. In Colab links *Ressourcen* öffnen – dort steht der Einheitenstand.
2. Stand notieren, Smoke-Lauf starten (`SMOKE = True`, 400 Zeilen), Stand
   erneut notieren.
3. `Einheiten pro Stunde ≈ ΔEinheiten / (ΔMinuten / 60)`; hochrechnen mit
   `Laufzeit ≈ Smoke-Laufzeit × 29.048 × Epochen / 400`.

T4 kann **kein bf16**. Das Skript erkennt das (`torch.cuda.is_bf16_supported()`)
und schaltet auf fp16. Wenn du das erzwingst, bekommst du NaN-Loss.

## Daten

```bash
lfm3ha prepare                      # voller Datensatz → data/lfm3-ha-de
lfm3ha prepare --limit 300 --oasst-limit 100   # Smoke-Test in Minuten
lfm3ha prepare --chat-ratio 0            # ohne OASST2 (maximale Werkzeugdisziplin)
lfm3ha prepare --chat-ratio 0.15         # mehr Sprachgefühl, mehr Format-Drift
```

Das Ergebnis ist ein `DatasetDict` mit `train` / `validation` / `test`,
`stats.json` und – auf Wunsch – einem Push zu Hugging Face
(`--push-to-hub marc04/homeassistant-tool-calling-lfm3-de`).

Wichtig: Der Test-Split kommt aus dem Test-Split der Quelle und wird beim
Training nie angefasst.

## Training

```bash
lfm3ha train --epochs 2
```

Was passiert:

1. Modell laden (`AutoModelForMultimodalLM` bei Omni, sonst
   `AutoModelForCausalLM`), Precision nach GPU.
2. LoRA-Ziele **am geladenen Modell** ermitteln – nicht geraten. Bei
   Omni-Modellen nur Module unter `language_model`, damit Audio- und
   Vision-Encoder unverändert bleiben.
3. Jede Zeile mit dem echten Chat-Template rendern und nur
   Assistenten-Token als Labels behalten.
4. Trainer mit `group_by_length` (ähnlich lange Sequenzen in einem Batch).

### Warum nur Assistenten-Turns?

Der Systemprompt enthält Geräteliste und Werkzeugschemas – zusammen rund
2.000 Token. Würde man darauf trainieren, lernt das Modell, den Prompt zu
reproduzieren, und die eigentliche Antwort verwässert. Die Maske spart
außerdem Loss-Berechnung: bei diesen Zeilen sind nur 2–12 % der Token
relevant.

### Sequenzlänge

Gemessen: p50 2.290, p90 3.073, max 5.173 Token. Standard ist 4.096 – damit
werden unter 1 % der Zeilen abgeschnitten. Abgeschnittene Zeilen, bei denen
die Antwort komplett wegfällt, werden verworfen und geloggt:

```text
→ Training: 27431 Zeilen (37 zu lang / verworfen)
```

Wenn VRAM knapp ist, ist `--max-seq-len 3072` der günstigste Hebel: ~10 %
der Zeilen fallen weg, der Rest trainiert weiter.

### Lernrate und Epochen

2e-4 mit cosine ist der Wert aus dem offiziellen Cookbook. Zwei Epochen sind
für ~28k synthetische Beispiele genug; mehr führt dazu, dass das Modell
Gerätenamen auswendig lernt, die es nie wieder sieht. Wenn `eval_loss` ab
Epoche 2 steigt: eine Epoche, `--learning-rate 1e-4`.

### Overfitting erkennen

Der Validierungsverlust ist bei dieser Aufgabe wenig aussagekräftig – der
Loss fällt auch dann, wenn das Modell nur das Format lernt. Aussagekräftig ist
`exact` aus der Auswertung:

```bash
lfm3ha eval --adapter outputs/lfm3-ha-de/checkpoint-1000 --limit 200
```

## Auswertung

```bash
lfm3ha eval --adapter outputs/lfm3-ha-de --limit 300 --out outputs/metrics.json
```

Drei Metriken, bewusst getrennt:

| Metrik | Frage |
|---|---|
| `parse` | Ist die Ausgabe überhaupt ein gültiger `<tool_call>`? |
| `function` | Stimmen die Funktionsnamen? |
| `exact` | Stimmen Namen **und** Argumente? |

`entity_id`-Listen werden als Mengen verglichen, `"3"` und `3` sind gleich,
leere Felder fallen weg. Das verhindert, dass ein semantisch korrekter Aufruf
an der Reihenfolge scheitert.

Zusätzlich wird nach `category` aufgeschlüsselt – `control` und `get_state`
sind leicht, `compound` (mehrere Aktionen) und `relative` („etwas lauter")
sind die harten Fälle.

## Veröffentlichen

```bash
lfm3ha eval --out outputs/metrics.json
lfm3ha push --adapter outputs/lfm3-ha-de --repo-id marc04/lfm3-ha-de
```

`push` verschmilzt den Adapter mit dem Basismodell (braucht CPU-RAM in
Höhe des Modells, also ~10 GB bei 2B in bf16 → Colab „High-RAM" oder
`--adapter-only`), schreibt die Modellkarte mit den Messwerten und lädt hoch.

## Audio später

Das Omni-Modell kann Sprache verstehen und erzeugen – dieser Datensatz ist
aber reiner Text. Für einen echten Sprachassistenten:

1. Deutschen Befehl aufnehmen (oder OHF-Voice / eigenes Mikrofon).
2. ASR davorschalten – das Omni-Modell kann das direkt, oder Whisper.
3. Auf dem Transkript läuft dieses Modell.

Willst du direkt Audio → Werkzeugaufruf trainieren, ist das offizielle
Cookbook-Notebook `lfm3_omni_sft_with_transformers.ipynb` der Startpunkt: es
zeigt den Collator für Audio-Embeddings und maskiert ebenfalls nur die
Antwort. Der Rest (Werkzeuge, Daten, Metriken) bleibt identisch.
