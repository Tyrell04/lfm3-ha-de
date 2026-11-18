"""LoRA-Adapter mit dem Basismodell verschmelzen und zu Hugging Face pushen.

Zwei Modi:

* ``--merge`` (Standard): volles Modell, direkt in Transformers nutzbar.
* ``--adapter-only``: nur der Adapter (~50 MB) plus Basismodell-Verweis.

Die Modellkarte kommt aus ``docs/MODEL_CARD.md``; Platzhalter werden per
``str.replace`` gefüllt (nicht ``format``), damit JSON-Beispiele im Code-Block
mit geschweiften Klammern stehen bleiben dürfen.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

CARD_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "MODEL_CARD.md"


def _metrics_block(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return (
            "_Noch keine Messwerte. Nach dem Training "
            "`lfm3ha eval --adapter outputs/lfm3-ha-de` ausführen._\n"
        )
    lines = [
        "| Kategorie | n | parse | function | exact |",
        "|---|---|---|---|---|",
    ]
    for name, values in metrics.get("by_category", {}).items():
        lines.append(
            f"| `{name}` | {values['n']} | {values['parse']} | "
            f"{values['function']} | {values['exact']} |"
        )
    overall = metrics.get("tool_calls", {})
    lines.append("")
    lines.append(
        f"Gesamt (Werkzeug-Rows, n={overall.get('n')}): "
        f"parse {overall.get('parse')}, function {overall.get('function')}, "
        f"exact {overall.get('exact')}."
    )
    lines.append("")
    lines.append(
        "Metriken: `parse` = gültiger `<tool_call>`, `function` = richtige "
        "Funktionsnamen, `exact` = Namen und Argumente."
    )
    return "\n".join(lines) + "\n"


def merge_adapter(base_model: str, adapter: str, out_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from .train import load_model

    model, _, _ = load_model(base_model)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(str(out_dir))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def write_card(
    out_dir: Path,
    repo_id: str,
    base_model: str,
    dataset: str,
    repo: str,
    adapter: str,
    metrics: dict[str, Any] | None,
) -> Path:
    if not CARD_SOURCE.exists():
        raise FileNotFoundError(f"Modellkarte nicht gefunden: {CARD_SOURCE}")
    card = CARD_SOURCE.read_text(encoding="utf-8")
    for key, value in {
        "{{repo_id}}": repo_id,
        "{{base_model}}": base_model,
        "{{dataset}}": dataset,
        "{{repo}}": repo,
        "{{adapter}}": adapter,
        "{{today}}": date.today().isoformat(),
        "{{metrics}}": _metrics_block(metrics),
    }.items():
        card = card.replace(key, value)
    path = out_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def push(
    out_dir: Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload LFM3 Home-Assistant DE",
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def load_metrics(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
