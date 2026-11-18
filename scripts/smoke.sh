#!/usr/bin/env bash
# Kleiner Ende-zu-Ende-Test ohne GPU: Daten bauen, Zeilen tokenisieren.
# Braucht `datasets` und `transformers` (kein torch).
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON="${PYTHON:-python3}"

echo "→ Daten (20 Zeilen, gestreamt)"
"$PYTHON" -m lfm3ha.cli prepare --limit 20 --oasst-limit 0 --streaming --out-dir data/smoke

echo "→ Tokenisierung mit dem echten LFM3-Tokenizer"
"$PYTHON" - <<'PY'
from datasets import load_from_disk
from lfm3ha.chatfmt import load_text_tokenizer
from lfm3ha.train import tokenize_split

ds = load_from_disk("data/smoke")
tok = load_text_tokenizer("LiquidAI/LFM3-Omni-2B")
tokenized, dropped = tokenize_split(ds["train"], tok, max_len=4096, limit=10)
assert len(tokenized) == 10, f"nur {len(tokenized)} Zeilen"
row = tokenized[0]
answer_tokens = sum(1 for label in row["labels"] if label != -100)
assert answer_tokens > 0, "keine Assistenten-Token markiert"
print(f"OK: {len(tokenized)} Zeilen, {answer_tokens} Antwort-Token, {dropped} verworfen")
PY

echo "✓ Smoke-Test durch"
