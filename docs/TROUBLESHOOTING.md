# Troubleshooting

Die Fehler, die bei diesem Setup wirklich auftreten – mit Ursache und Fix.

## `Tool call arguments must be a mapping, got a JSON-encoded string`

**Ursache:** Das LFM3-Template ruft `raise_exception`, wenn
`tool_calls[].function.arguments` ein String ist. Der HA-Datensatz speichert
genau das (OpenAI-Format).

**Fix:** `lfm3ha.data.normalize_messages()` parst die Strings. Wenn du eigene
Daten einspielst: `arguments = json.loads(arguments)` vor
`apply_chat_template`.

## Die Ausgabe enthält `<|tool_call_start|>` statt `<tool_call>`

**Ursache:** Du mischst LFM2- und LFM3-Formate. Die Datensätze von
liquid-finetune sind für LFM2/LFM2.5 gebaut.

| Modellfamilie | Format |
|---|---|
| LFM2 / LFM2.5 | `<\|tool_call_start\|>[name(arg="wert")]<\|tool_call_end\|>` |
| LFM3 | `<tool_call><function=name><parameter=arg>wert</parameter></function></tool_call>` |

**Fix:** Für LFM3 dieses Repo nutzen (Template rendert das Format selbst, wenn
du `tool_calls` übergibst). Für LFM2.5 liquid-finetune.

## `ValueError: Your setup doesn't support bf16`

**Ursache:** T4 (Compute Capability 7.5) kann kein bf16.

**Fix:** `train.py` erkennt das automatisch. Wenn du selbst schreibst:
`bf16=False, fp16=True`. Bei fp16 zusätzlich `max_grad_norm` klein halten
(0.3) und Warmup nicht weglassen – sonst NaNs.

**Achtung, gemessen auf einer Colab-T4 (torch 2.14):**
`torch.cuda.is_bf16_supported()` meldet dort **`True`**, obwohl die Karte
kein bf16 kann. Wer sich auf das Flag verlässt, trainiert in emuliertem bf16.
`pick_dtype()` fragt deshalb `get_device_capability()` und nimmt bf16 erst ab
8.0 – nicht das Flag.

## `ImportError: Found an incompatible version of torchao`

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

**Ursache:** Colab bringt `torchao 0.10` mit, `peft ≥ 0.21` verlangt > 0.16.
Der Fehler fliegt beim Anhängen des Adapters – mitten in `get_peft_model`,
also nach Modell-Download und Tokenisierung.

**Fix:** `pip install -U "torchao>=0.16"` (oder `pip uninstall -y torchao`,
wenn du keine quantisierten Adapter brauchst). Das Notebook macht das in der
Setup-Zelle; `check_peft_env()` bricht sonst vorher mit genau diesem Hinweis ab.

## `TypeError: TrainingArguments.__init__() got an unexpected keyword argument`

**Ursache:** transformers 5 hat `warmup_ratio` und `group_by_length` aus
`TrainingArguments` entfernt (`max_length`-bezogene Felder ebenfalls
umsortiert). Der Lauf stirbt, bevor der erste Schritt läuft.

**Fix:** `warmup_steps` selbst rechnen (`ratio × Schritte`); `group_by_length`
weglassen. Beides ist im Repo erledigt.

## `KeyError: 'messages'` beim Tokenisieren

**Ursache:** `load_dataset("data/lfm3-ha-de")` auf einem `save_to_disk`-Ordner.
`datasets` hält `state.json` für eine Datendatei und liefert je Split **eine**
Zeile, deren Spalten die Schlüssel von `state.json` sind (`_data_files`,
`_fingerprint`, …). Der Fehler bleibt über Neustarts, weil `datasets` das
Ergebnis zusätzlich in `~/.cache/huggingface/datasets` ablegt.

**Fix:** lokal `load_from_disk` benutzen – `lfm3ha.data.load_any_dataset()`
entscheidet anhand des Pfads. Wenn es schon passiert ist:
`rm -rf ~/.cache/huggingface/datasets/lfm3-ha-de`.

## Alle Metriken 0 %, obwohl das Modell sinnvoll antwortet

**Ursache:** `Lfm3OmniForConditionalGeneration.generate` gibt ein
`Lfm3OmniGenerateOutput` zurück (`sequences`, `sequence_lengths`,
`audio_codes`), keinen Tensor. `output[0][prompt_len:]` darauf ergibt eine
leere Sequenz.

**Fix:** `lfm3ha.infer.extract_new_tokens()` benutzen oder selbst
`result.sequences[0][prompt_len:result.sequence_lengths[0]]` schneiden.
Gegenprobe: Logits des letzten Prompt-Tokens anschauen – wenn dort `Hallo`
mit Abstand vorn liegt, ist das Modell in Ordnung und nur das Auslesen kaputt.

## `ModuleNotFoundError: No module named 'transformers.models.lfm3_omni'`

**Ursache:** `trust_remote_code=True` fehlt oder die Transformers-Version ist
zu alt. LFM3-Omni braucht **5.20.2**.

**Fix:** `pip install "transformers==5.20.2"` und überall
`trust_remote_code=True` setzen.

## `AutoModelForMultimodalLM` existiert nicht

**Ursache:** ältere Transformers-Version.

**Fix:** Version pinnen. Für reine Textmodelle (`LFM3-1.7B`) geht
`AutoModelForCausalLM` – `train.py` wählt anhand der Architektur.

## 4-bit-Training bricht ab / `bitsandbytes` fehlt

**Ursache:** `bitsandbytes` ist Linux/CUDA-only und funktioniert mit
Custom-Code-Modellen nicht immer.

**Fix:** Ohne `--load-in-4bit` trainieren (LoRA in bf16 braucht bei 2B nur
~8 GB) oder auf eine größere GPU wechseln. 4-bit ist hier eine Notlösung, kein
Standard.

## `push` stirbt mit `Killed`

**Ursache:** `merge_adapter` lädt das volle Modell in CPU-RAM (bei 2B bf16
~10 GB inkl. Kopie beim Verschmelzen). Colab-Standard hat 12 GB.

**Fix:** Colab „High-RAM" aktivieren, oder `--adapter-only` pushen (Adapter
~50 MB, Basismodell wird referenziert).

## `datasets-server` liefert 502

**Ursache:** Der Viewer der Hugging-Face-Datasets ist ein separater Dienst und
hat Aussetzer.

**Fix:** `load_dataset(...)` lädt die Parquet-Dateien direkt und ist davon
nicht betroffen. Nur die Vorschau im Browser leidet.

## `load_dataset` liefert `input_ids` statt `messages`

**Ursache:** Beim Tokenisieren (`Dataset.map`) schreibt `datasets`
`cache-*.arrow`-Dateien **neben** die Daten. Werden die mit hochgeladen,
findet `load_dataset` sie zuerst und liefert die tokenisierten
Zwischenstände.

**Fix:** `lfm3ha prepare --push-to-hub` löscht sie vor dem Upload und
ignoriert sie zusätzlich beim Hochladen. Bei manuellem Upload:
`huggingface-cli upload … --exclude "*/cache-*.arrow"`.

## `homeassistant-tool-calling-de` lässt sich nicht laden

**Ursache:** Der Datensatz wurde umbenannt und leitet auf
`mkenfenheuer/homeassistant-tool-calling` weiter (dort Spalte `locale`).

**Fix:** Dieses Repo filtert bereits korrekt:
`ds.filter(lambda r: r["locale"] == "de")`.

## Das Modell antwortet auf Deutsch, ruft aber kein Werkzeug auf

Typische Ursachen, in dieser Reihenfolge prüfen:

1. **Geräteliste fehlt** im Systemprompt → das Modell erfindet oder antwortet
   nur sprachlich. Prompt mit `build_system_prompt()` bauen.
2. **Werkzeuge nicht übergeben** → `tools=TOOLS` bei
   `apply_chat_template` vergessen. Das Template schreibt die Schemas dann
   nicht in den Prompt.
3. **Zu viel OASST2** → `--chat-ratio` senken (0.08 → 0.03) und neu
   trainieren. Chat-Daten belohnen lange Antworten, Werkzeugdaten kurze.
4. **Zu wenige Epochen** → `exact` auf `control` prüfen; wenn selbst das
   niedrig ist, war das Training zu kurz.

## `exact` ist hoch, aber HA führt nichts aus

Kein Modellproblem: Die `entity_id` muss im echten Haus existieren. Prüfe die
Geräteliste, die du in den Prompt schreibst – das Modell nutzt nur, was es
sieht.

## Training läuft, Loss fällt nicht

- Lernrate zu klein? LoRA braucht 1e-4 bis 3e-4, nicht 2e-5 (das ist
  Full-Finetune-Territorium).
- Labels komplett `-100`? Dann ist die Maske kaputt. `encode_example` gibt
  `None` zurück und die Zeile wird verworfen – prüfe die Logzeile
  „zu lang / verworfen".
- Falsches Template: `tok.chat_template` muss die LFM3-Vorlage sein, nicht
  eine generische. `load_text_tokenizer()` prüft das nicht – bei eigenen
  Tokenizern `print(tok.chat_template[:200])`.
