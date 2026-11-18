"""Inferenz: deutsches Zuhause -> Werkzeugaufruf.

Enthält den Systemprompt-Bauer (gleiche Form wie im Training), das Laden von
Basismodell + Adapter und die Generationsschleife für die Auswertung.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .evaluate import gold_calls, score
from .tools_de import TOOLS

DEFAULT_VERBOSITY = "medium"

VERBOSITY_DE = {
    "short": "Antworte so knapp wie möglich, mit wenigen Worten.",
    "medium": "Antworte kurz, meist in einem Satz.",
    "long": "Antworte in ganzen Sätzen und nenne die wichtigsten Details.",
}

SYSTEM_DE = (
    "Du bist der Sprachassistent eines Smart Homes mit Home Assistant. "
    "Antworte immer auf Deutsch und freundlich. {verbosity}\n"
    "Steuere Geräte mit dem Werkzeug control. Frage Zustände und Messwerte mit "
    "get_state ab, bevor du Fragen dazu beantwortest.\n"
    "Verwende nur entity_ids aus der folgenden Liste. Gibt es ein Gerät nicht, "
    "sage das, statt eine entity_id zu erfinden.\n"
    "Meint der Nutzer alle Lichter eines Bereichs, gib statt entity_id den Bereich "
    'an (area), für das ganze Haus area "*".\n'
    "Timer stellst du mit timer, Uhrzeit und Datum liefert get_time, Durchsagen an "
    "alle Lautsprecher machst du mit broadcast.\n"
    "\n"
    "Geräte (entity_id | Name | Bereich):\n{devices}"
)


@dataclass(frozen=True)
class Device:
    entity_id: str
    name: str
    area: str


def build_system_prompt(
    devices: Iterable[Device], verbosity: str = DEFAULT_VERBOSITY
) -> str:
    lines = "\n".join(f"{d.entity_id} | {d.name} | {d.area}" for d in devices)
    return SYSTEM_DE.format(
        verbosity=VERBOSITY_DE.get(verbosity, VERBOSITY_DE[DEFAULT_VERBOSITY]),
        devices=lines,
    )


DEMO_HOUSE: tuple[Device, ...] = (
    Device("light.wohnzimmer_decke", "Deckenlampe", "Wohnzimmer"),
    Device("light.wohnzimmer_stehlampe", "Stehlampe", "Wohnzimmer"),
    Device("light.kueche_arbeitslicht", "Arbeitslicht", "Küche"),
    Device("switch.wohnzimmer_kaffeemaschine", "Kaffeemaschine", "Wohnzimmer"),
    Device("cover.wohnzimmer_rollo", "Rollo", "Wohnzimmer"),
    Device("climate.wohnzimmer_thermostat", "Thermostat", "Wohnzimmer"),
    Device("media_player.wohnzimmer_speaker", "Wohnzimmer Speaker", "Wohnzimmer"),
    Device("sensor.wohnzimmer_temperatur", "Temperatur", "Wohnzimmer"),
    Device("sensor.kueche_luftfeuchtigkeit", "Luftfeuchtigkeit", "Küche"),
    Device("weather.bremen", "Bremen", "Garten"),
    Device("lock.haus_haustuer", "Haustür", "Flur"),
    Device("vacuum.wohnzimmer_saugroboter", "Saugroboter", "Wohnzimmer"),
    Device("todo.einkaufsliste", "Einkaufsliste", "-"),
)


# --------------------------------------------------------------------------- #
# Modell
# --------------------------------------------------------------------------- #
def load_for_inference(base_model: str, adapter: str | None = None, load_in_4bit: bool = False):
    from transformers import AutoTokenizer

    from .train import load_model

    model, dtype, _ = load_model(base_model, load_in_4bit)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    # Omni-Modelle bringen einen Processor mit; für Text genügt der Tokenizer.
    try:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor

        tok = AutoProcessor.from_pretrained(base_model, trust_remote_code=True).tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok, dtype


def generate_text(model, tok, messages: list[dict[str, Any]], max_new_tokens: int = 256) -> str:
    import torch

    inputs = tok.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Auswertung
# --------------------------------------------------------------------------- #
def run_eval(model, tok, dataset, limit: int = 500, max_new_tokens: int = 256) -> dict:
    """Generiert Antworten und vergleicht sie mit den Gold-Aufrufen."""
    import json as _json

    from .evaluate import aggregate

    rows = []
    for i, row in enumerate(dataset):
        if i >= limit:
            break
        messages = row["messages"]
        if isinstance(messages, str):
            messages = _json.loads(messages)
        gold = gold_calls(messages)
        # Prompt bis zum ersten Assistenten-Turn mit Werkzeugaufruf – genau die
        # Situation, in der das Modell in Home Assistant steht.
        prompt: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                break
            prompt.append(message)
        prediction = generate_text(model, tok, prompt, max_new_tokens)
        rows.append(
            {
                "category": row.get("category") or "",
                "gold": gold,
                "prediction": prediction,
                "scores": score(gold, prediction),
            }
        )
    return aggregate(rows)
