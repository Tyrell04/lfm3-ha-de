"""LoRA-SFT für LFM3 (Omni oder Text) auf deutschen Home-Assistant-Daten.

Der Ablauf folgt dem offiziellen Liquid-Cookbook: Transformer-Modell laden,
LoRA auf die Sprachmodell-Projektionen, nur auf Assistenten-Turns trainieren.
Zusätzlich ist alles hier so gebaut, dass es auch auf einer einzelnen
Colab-GPU läuft – inklusive fp16-Fallback auf T4 (kein bf16).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chatfmt import PadCollator, build_example, load_text_tokenizer
from .data import load_any_dataset
from .tools_de import TOOLS

# Projektionen, die LoRA bekommt. in_proj/out_proj sind die gated short
# convolutions der LFM3-Blöcke, w1/w2/w3 das MLP, gate_proj der Per-Head-Gate.
LORA_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
    "gate_proj",
)


@dataclass
class TrainConfig:
    base_model: str = "LiquidAI/LFM3-Omni-2B"
    dataset: str = "marc04/homeassistant-tool-calling-lfm3-de"
    output_dir: str = "outputs/lfm3-ha-de"
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 4
    max_seq_len: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    seed: int = 42
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    max_train_rows: int | None = None
    max_eval_rows: int = 500
    report_to: str = "none"
    resume: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Modell
# --------------------------------------------------------------------------- #
def pick_dtype(torch):
    """bf16 nur ab Ampere (Compute Capability 8.0), sonst fp16.

    ``torch.cuda.is_bf16_supported()`` meldet auf einer T4 (CC 7.5) mit
    torch 2.14 ``True``, obwohl Turing kein bf16 kann – das Training läuft
    dann in emuliertem bf16 und liefert NaN. Die Compute Capability lügt nicht.
    """
    if torch.cuda.is_available():
        try:
            major = torch.cuda.get_device_capability(0)[0]
        except Exception:  # exotische Backends ohne Capability-Auskunft
            major = 0
        return torch.bfloat16 if major >= 8 else torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


TORCHAO_MIN = (0, 16)


def check_peft_env() -> None:
    """Abbruch mit klarer Anleitung, wenn das Umfeld peft blockiert.

    Colab bringt ``torchao 0.10`` mit; peft ≥ 0.21 verlangt > 0.16 und wirft
    beim Anhängen des Adapters einen ``ImportError`` mitten aus dem Nichts.
    Lieber vorher prüfen als hinterher einen Traceback zu entziffern.
    """
    try:
        from importlib.metadata import version

        raw = version("torchao")
    except Exception:  # nicht installiert – der Normalfall außerhalb Colabs
        return
    digits = [int(p) for p in raw.split(".")[:2] if p.isdigit()]
    if len(digits) == 2 and tuple(digits) < TORCHAO_MIN:
        raise SystemExit(
            f"torchao {raw} ist zu alt für peft (verlangt > 0.16.0).\n"
            "Fix: pip install -U 'torchao>=0.16'  – oder: pip uninstall -y torchao"
        )


def load_model(model_id: str, load_in_4bit: bool = False):
    """Lädt LFM3 – Omni über ``AutoModelForMultimodalLM``, sonst causal LM."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    check_peft_env()

    dtype = pick_dtype(torch)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    is_omni = "Omni" in (config.architectures or [""])[0] or "omni" in type(config).__name__.lower()

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = dtype
        kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "auto"

    if is_omni:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs), dtype, is_omni
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs), dtype, is_omni


def discover_lora_targets(model, prefer_language_model: bool) -> str:
    """Findet die Projektionsnamen im Modell und gibt einen Regex zurück.

    Statt Modulpfade zu raten (die sich zwischen LFM3-Varianten unterscheiden),
    zählen wir sie am geladenen Modell. Bei Omni-Modellen beschränken wir uns
    auf ``language_model``, damit Audio-/Vision-Encoder unangetastet bleiben.
    """
    targets = [
        name
        for name, module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in LORA_LEAVES and hasattr(module, "weight")
    ]
    if prefer_language_model:
        language = [t for t in targets if "language_model" in t]
        targets = language or targets
    if not targets:
        raise RuntimeError(
            "Keine LoRA-Zielmodule gefunden – Modellarchitektur prüfen "
            f"(erwartet: {', '.join(LORA_LEAVES)})"
        )
    return "|".join(re.escape(t) for t in targets)


def attach_lora(model, cfg: TrainConfig, prefer_language_model: bool):
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=discover_lora_targets(model, prefer_language_model),
    )
    model = get_peft_model(model, peft_config)
    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:  # manche Custom-Architekturen kennen das nicht
            print(f"! Gradient Checkpointing nicht aktivierbar: {exc}")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- #
# Daten
# --------------------------------------------------------------------------- #
def tokenize_split(dataset, tok, max_len: int, limit: int | None, num_proc: int = 2):
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    def _fn(row):
        encoded = build_example(tok, row, max_len)
        if encoded is None:
            return {"input_ids": [], "labels": [], "length": 0}
        encoded["length"] = len(encoded["input_ids"])
        return encoded

    tokenized = dataset.map(
        _fn,
        remove_columns=dataset.column_names,
        num_proc=max(1, num_proc),
        desc="Tokenisieren",
    )
    before = len(tokenized)
    tokenized = tokenized.filter(lambda r: len(r["input_ids"]) > 0, num_proc=1)
    dropped = before - len(tokenized)
    return tokenized, dropped


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_text_tokenizer(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print(f"→ Basismodell: {cfg.base_model}")
    model, dtype, is_omni = load_model(cfg.base_model, cfg.load_in_4bit)
    print(f"→ dtype={dtype}, omni={is_omni}, 4bit={cfg.load_in_4bit}")

    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    model = attach_lora(model, cfg, prefer_language_model=is_omni)

    print(f"→ Datensatz: {cfg.dataset}")
    raw = load_any_dataset(cfg.dataset)
    train_ds, dropped = tokenize_split(
        raw["train"], tok, cfg.max_seq_len, cfg.max_train_rows
    )
    eval_ds = None
    if "validation" in raw and len(raw["validation"]) > 0:
        eval_ds, eval_dropped = tokenize_split(
            raw["validation"], tok, cfg.max_seq_len, cfg.max_eval_rows, num_proc=1
        )
        if len(eval_ds) == 0:
            eval_ds = None
        else:
            print(f"→ Validation: {len(eval_ds)} Zeilen ({eval_dropped} zu lang)")
    print(f"→ Training: {len(train_ds)} Zeilen ({dropped} zu lang / verworfen)")

    # Transformers 5 kennt ``warmup_ratio`` nicht mehr – nur Schritte.
    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    warmup_steps = max(1, round(cfg.warmup_ratio * steps_per_epoch * cfg.epochs))
    print(f"→ {steps_per_epoch * cfg.epochs} Schritte, davon {warmup_steps} Warmup")

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=max(1, cfg.batch_size),
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        report_to=cfg.report_to,
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=True if cfg.resume else None)

    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    (out / "training_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "tools_de.json").write_text(
        json.dumps(TOOLS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✓ Adapter gespeichert: {out}")
    return out
