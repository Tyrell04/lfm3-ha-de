# Training

## Hardware

| GPU | VRAM | Einstellung | Dauer (Schätzung, siehe Hinweis) |
|---|---|---|---|
| Colab T4 (frei) | 16 GB | `--batch-size 1 --grad-accum 16` fp16 | Smoke-Test ja, voller Lauf knapp |
| Colab L4 | 24 GB | `--batch-size 4 --grad-accum 8` bf16 | voller Lauf machbar |
| Colab A100 | 40 GB | Standard (`8 × 4`) bf16 | komfortabel |
| RTX 3090/4090 | 24 GB | `--batch-size 4 --grad-accum 8` | gut |

> Die offizielle Liquid-Angabe für ein ähnliches Setup: eine Epoche über
> ~6.600 Audio-Beispiele auf einer A100 in ~10 Minuten. Hier ist der Prompt
> deutlich länger (Geräteliste + Schemas), dafür sind die Beispiele Text –
> rechne mit Größenordnung 1–3 Stunden für den vollen deutschen Datensatz auf
> einer A100 und messe es selbst.

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
