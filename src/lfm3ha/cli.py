"""Kommandozeile: ``lfm3ha prepare | train | eval | push | chat``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data as data_mod
from .tools_de import tools_json

DEFAULT_DATASET_REPO = "marc04/homeassistant-tool-calling-lfm3-de"


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def cmd_prepare(args: argparse.Namespace) -> int:
    from datasets import DatasetDict

    print("→ Lade Home-Assistant-Datensatz (locale=de) …")
    ha = data_mod.load_ha_de(
        limit=args.limit, streaming=args.streaming, splits=("train", "test")
    )
    print(f"  train={len(ha['train'])}  test={len(ha['test'])}")

    chat_rows: list[dict] = []
    if args.chat_ratio > 0 and args.oasst_limit > 0:
        print("→ Lade OASST2 (deutsche Threads) …")
        chat_rows = data_mod.load_oasst2_de(limit=args.oasst_limit)
        print(f"  {len(chat_rows)} Gespräche")

    train_rows = data_mod.mix(
        data_mod.deduplicate(ha["train"]), chat_rows, args.chat_ratio
    )
    test_rows = data_mod.deduplicate(ha["test"])

    # Validierung aus dem Trainingsanteil abzweigen (Test bleibt unberührt).
    if args.validation_rows and len(train_rows) > args.validation_rows * 2:
        import random

        rng = random.Random(42)
        shuffled = train_rows[:]
        rng.shuffle(shuffled)
        valid_rows = shuffled[: args.validation_rows]
        train_rows = shuffled[args.validation_rows :]
    else:
        valid_rows = []

    splits = {
        "train": data_mod.to_hf(train_rows, tools_json()),
        "test": data_mod.to_hf(test_rows, tools_json()),
    }
    if valid_rows:
        splits["validation"] = data_mod.to_hf(valid_rows, tools_json())
    ds = DatasetDict(splits)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    summary = {
        "train": data_mod.stats(train_rows),
        "validation": data_mod.stats(valid_rows),
        "test": data_mod.stats(test_rows),
        "chat_ratio": args.chat_ratio,
        "sources": {
            "home_assistant": data_mod.HA_DATASET,
            "chat": data_mod.OASST_DATASET if chat_rows else None,
        },
    }
    (out / "stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["train"], indent=2, ensure_ascii=False))

    if args.push_to_hub:
        _push_dataset(out, args.push_to_hub, args.private, summary)
    return 0


def _push_dataset(out: Path, repo_id: str, private: bool, summary: dict) -> None:
    from huggingface_hub import HfApi

    # ``Dataset.map`` (z. B. beim Tokenisieren) legt cache-*.arrow neben die
    # Daten. Werden die mit hochgeladen, liest ``load_dataset`` anschließend
    # die tokenisierten Zwischenstände statt der Konversationen.
    for stale in out.rglob("cache-*.arrow"):
        stale.unlink()

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    card = Path("docs/DATASET_CARD.md")
    if card.exists():
        text = card.read_text(encoding="utf-8").replace("REPO_ID", repo_id)
        (out / "README.md").write_text(text, encoding="utf-8")
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["*/cache-*.arrow", "cache-*.arrow"],
        commit_message="Upload LFM3 Home-Assistant DE Trainingsset",
    )
    print(f"✓ Datensatz: https://huggingface.co/datasets/{repo_id}")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    cfg = TrainConfig(
        base_model=args.base_model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        load_in_4bit=args.load_in_4bit,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        report_to=args.report_to,
        resume=args.resume,
    )
    train(cfg)
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    from .data import load_split
    from .infer import load_for_inference, run_eval

    print(f"→ Modell: {args.base_model}  Adapter: {args.adapter or '—'}")
    model, tok, _ = load_for_inference(
        args.base_model, args.adapter, load_in_4bit=args.load_in_4bit
    )
    ds = load_split(args.dataset, args.split)
    metrics = run_eval(model, tok, ds, limit=args.limit, max_new_tokens=args.max_new_tokens)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✓ {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# push
# --------------------------------------------------------------------------- #
def cmd_push(args: argparse.Namespace) -> int:
    from .push_model import load_metrics, merge_adapter, push, write_card

    out = Path(args.out_dir)
    if not args.adapter_only:
        print(f"→ Verschmelze Adapter {args.adapter} mit {args.base_model} …")
        merge_adapter(args.base_model, args.adapter, out)
    else:
        import shutil

        out.mkdir(parents=True, exist_ok=True)
        for item in Path(args.adapter).iterdir():
            if item.is_file():
                shutil.copy2(item, out / item.name)

    write_card(
        out,
        repo_id=args.repo_id,
        base_model=args.base_model,
        dataset=args.dataset,
        repo=args.repo,
        adapter=args.adapter,
        metrics=load_metrics(args.metrics),
    )
    url = push(out, args.repo_id, private=args.private, token=args.token)
    print(f"✓ Modell: {url}")
    return 0


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
def cmd_chat(args: argparse.Namespace) -> int:
    from .infer import DEMO_HOUSE, build_system_prompt, generate_text, load_for_inference

    model, tok, _ = load_for_inference(
        args.base_model, args.adapter, load_in_4bit=args.load_in_4bit
    )
    system = build_system_prompt(DEMO_HOUSE)
    messages = [{"role": "system", "content": system}]
    for prompt in args.prompt or [
        "Mach das Licht im Wohnzimmer an.",
        "Wie warm ist es im Wohnzimmer?",
        "Spiel ruhige Musik im Wohnzimmer.",
        "Stell einen Timer auf 10 Minuten.",
    ]:
        turn = messages + [{"role": "user", "content": prompt}]
        answer = generate_text(model, tok, turn, args.max_new_tokens)
        print(f"\n🧑 {prompt}\n🤖 {answer}")
        messages = turn + [{"role": "assistant", "content": answer}]
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
CONFIG_MAP: dict[tuple[str, str], str] = {
    ("model", "base"): "base_model",
    ("model", "max_seq_len"): "max_seq_len",
    ("lora", "r"): "lora_r",
    ("lora", "alpha"): "lora_alpha",
    ("lora", "dropout"): "lora_dropout",
    ("training", "epochs"): "epochs",
    ("training", "learning_rate"): "learning_rate",
    ("training", "batch_size"): "batch_size",
    ("training", "grad_accum"): "grad_accum",
    ("training", "warmup_ratio"): "warmup_ratio",
    ("training", "max_grad_norm"): "max_grad_norm",
    ("training", "save_steps"): "save_steps",
    ("training", "eval_steps"): "eval_steps",
    ("data", "chat_ratio"): "chat_ratio",
    ("data", "chat_limit"): "oasst_limit",
    ("data", "validation_rows"): "validation_rows",
    ("publish", "model"): "repo_id",
}


def flatten_config(path: str | Path) -> dict[str, object]:
    """YAML-Konfiguration auf CLI-Dests abbilden.

    Die Werte werden zu **Defaults**; wer ein Flag explizit setzt, gewinnt.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    flat: dict[str, object] = {}
    for (section, key), dest in CONFIG_MAP.items():
        value = (raw.get(section) or {}).get(key)
        if value is not None:
            flat[dest] = value
    return flat


def _set_defaults(sub: argparse.ArgumentParser, defaults: dict[str, object]) -> None:
    known = {action.dest for action in sub._actions}  # noqa: SLF001
    sub.set_defaults(**{k: v for k, v in defaults.items() if k in known})


def build_parser(defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    p = argparse.ArgumentParser(
        prog="lfm3ha",
        description="LFM3-Fine-Tune für Home Assistant auf Deutsch.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML-Datei mit Defaults (z. B. configs/default.yaml)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Trainingsset bauen und optional pushen")
    prep.add_argument("--out-dir", default="data/lfm3-ha-de")
    prep.add_argument("--limit", type=int, default=None, help="nur N HA-Zeilen (Smoke-Test)")
    prep.add_argument("--streaming", action="store_true")
    prep.add_argument("--oasst-limit", type=int, default=6000)
    prep.add_argument("--chat-ratio", type=float, default=0.08)
    prep.add_argument("--validation-rows", type=int, default=500)
    prep.add_argument("--push-to-hub", default=None)
    prep.add_argument("--private", action="store_true")
    prep.set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="LoRA-SFT starten")
    tr.add_argument("--base-model", default="LiquidAI/LFM3-Omni-2B")
    tr.add_argument("--dataset", default=DEFAULT_DATASET_REPO)
    tr.add_argument("--output-dir", default="outputs/lfm3-ha-de")
    tr.add_argument("--epochs", type=float, default=2.0)
    tr.add_argument("--learning-rate", type=float, default=2e-4)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--max-seq-len", type=int, default=4096)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=32)
    tr.add_argument("--warmup-ratio", type=float, default=0.03)
    tr.add_argument("--max-grad-norm", type=float, default=0.3)
    tr.add_argument("--load-in-4bit", action="store_true")
    tr.add_argument("--max-train-rows", type=int, default=None)
    tr.add_argument("--max-eval-rows", type=int, default=500)
    tr.add_argument("--logging-steps", type=int, default=10)
    tr.add_argument("--save-steps", type=int, default=200)
    tr.add_argument("--eval-steps", type=int, default=200)
    tr.add_argument("--report-to", default="none")
    tr.add_argument("--resume", action="store_true")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Werkzeugaufrufe gegen Testset messen")
    ev.add_argument("--base-model", default="LiquidAI/LFM3-Omni-2B")
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--dataset", default=DEFAULT_DATASET_REPO)
    ev.add_argument("--split", default="test")
    ev.add_argument("--limit", type=int, default=300)
    ev.add_argument("--max-new-tokens", type=int, default=256)
    ev.add_argument("--load-in-4bit", action="store_true")
    ev.add_argument("--out", default="outputs/metrics.json")
    ev.set_defaults(func=cmd_eval)

    pu = sub.add_parser("push", help="Modell zu Hugging Face hochladen")
    pu.add_argument("--adapter", required=True)
    pu.add_argument("--base-model", default="LiquidAI/LFM3-Omni-2B")
    pu.add_argument("--out-dir", default="outputs/merged")
    pu.add_argument("--repo-id", required=True)
    pu.add_argument("--dataset", default=DEFAULT_DATASET_REPO)
    pu.add_argument("--repo", default="Tyrell04/lfm3-ha-de")
    pu.add_argument("--metrics", default="outputs/metrics.json")
    pu.add_argument("--adapter-only", action="store_true")
    pu.add_argument("--private", action="store_true")
    pu.add_argument("--token", default=None)
    pu.set_defaults(func=cmd_push)

    ch = sub.add_parser("chat", help="Demo-Konversation im Beispielhaus")
    ch.add_argument("--base-model", default="LiquidAI/LFM3-Omni-2B")
    ch.add_argument("--adapter", default=None)
    ch.add_argument("--load-in-4bit", action="store_true")
    ch.add_argument("--max-new-tokens", type=int, default=256)
    ch.add_argument("--prompt", action="append")
    ch.set_defaults(func=cmd_chat)

    for parser in (prep, tr, ev, pu, ch):
        _set_defaults(parser, defaults)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = flatten_config(known.config) if known.config else {}
    args = build_parser(defaults).parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
