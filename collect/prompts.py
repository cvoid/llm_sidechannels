from __future__ import annotations

import json
from pathlib import Path

_DEFAULT = Path(__file__).parent / "data" / "exp1_prompts.jsonl"


def load_prompts(path: Path = _DEFAULT) -> list[str]:
    with open(path) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]
