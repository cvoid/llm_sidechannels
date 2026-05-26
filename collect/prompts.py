from __future__ import annotations

import json
from pathlib import Path

_DEFAULT = Path(__file__).parent / "data" / "exp1_prompts.jsonl"


def load_prompts(path: Path | None = None) -> list[str]:
    path = path or _DEFAULT
    with open(path) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]
