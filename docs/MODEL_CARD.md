---
library_name: transformers
license: other
license_name: lfm1.0
license_link: https://huggingface.co/{{base_model}}/blob/main/LICENSE
base_model: {{base_model}}
language:
  - de
pipeline_tag: text-generation
tags:
  - liquid
  - lfm3
  - home-assistant
  - tool-calling
  - function-calling
  - smart-home
  - german
datasets:
  - {{dataset}}
---

# LFM3 Home Assistant (Deutsch)

Deutschsprachiger Home-Assistant-Assistent als LoRA-Fine-Tune von
[`{{base_model}}`](https://huggingface.co/{{base_model}}). Das Modell bildet
getippte Befehle auf die sechs Werkzeuge `control`, `get_state`,
`music_library`, `timer`, `get_time` und `broadcast` ab.

| Eigenschaft | Wert |
|---|---|
| Basismodell | `{{base_model}}` |
| Training | LoRA (r=16, alpha=32), nur Assistenten-Turns |
| Datensatz | [`{{dataset}}`](https://huggingface.co/datasets/{{dataset}}) |
| Adapter | `{{adapter}}` |
| Stand | {{today}} |

## Nutzung

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{{repo_id}}"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype="bfloat16", device_map="auto", trust_remote_code=True
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Mach das Licht im Wohnzimmer an"},
]
inputs = tok.apply_chat_template(
    messages,
    tools=TOOLS,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
out = model.generate(**inputs.to(model.device), do_sample=False, max_new_tokens=256)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

`TOOLS` und der Systemprompt stehen im Trainingsrepo
[{{repo}}](https://github.com/{{repo}}) – `src/lfm3ha/tools_de.py` und
`src/lfm3ha/infer.py` (`build_system_prompt`, `DEMO_HOUSE`).

Antwortformat:

```text
<tool_call>
<function=control>
<parameter=actions>
[{"service": "light.turn_on", "entity_id": ["light.wohnzimmer_decke"]}]
</parameter>
</function>
</tool_call>
```

## Messwerte

{{metrics}}

## Grenzen

* **Nur Deutsch.** Andere Sprachen wurden nicht trainiert.
* **Prompt-Abhängig.** Die Geräteliste im Systemprompt ist Teil der Aufgabe.
  Ohne sie erfindet das Modell `entity_id`s – das ist der häufigste Fehler
  beim Einsatz.
* **Synthetische Haushalte.** Gelernt wird das Schema, nicht deine Geräte.
* **Keine Sicherheitsgarantie.** Werkzeugaufrufe serverseitig gegen
  Berechtigungen prüfen.
* **Kein Audio.** Für Sprache ASR davorschalten.

## Lizenz

Basismodell: **LFM Open License v1.0** (siehe Lizenzdatei des Basismodells),
dieses Fine-Tune erbt sie. Trainingsdaten: CC-BY-4.0
(`mkenfenheuer/homeassistant-tool-calling`) und Apache-2.0
(`OpenAssistant/oasst2`). Details: [NOTICE](https://github.com/{{repo}}/blob/main/NOTICE.md).
