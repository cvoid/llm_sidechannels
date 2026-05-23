from __future__ import annotations

import json
from pathlib import Path

import pytest

from collect.prompts import load_prompts


def test_load_prompts_count(tmp_path: Path) -> None:
    p = tmp_path / "prompts.jsonl"
    p.write_text("\n".join(json.dumps({"id": i, "text": f"prompt {i}"}) for i in range(7)))
    assert len(load_prompts(p)) == 7


def test_load_prompts_returns_strings(tmp_path: Path) -> None:
    p = tmp_path / "prompts.jsonl"
    p.write_text(json.dumps({"id": 0, "text": "hello world"}))
    prompts = load_prompts(p)
    assert prompts == ["hello world"]


def test_load_prompts_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "prompts.jsonl"
    p.write_text(json.dumps({"id": 0, "text": "a"}) + "\n\n" + json.dumps({"id": 1, "text": "b"}) + "\n")
    assert len(load_prompts(p)) == 2


def test_default_path_loads_50_prompts() -> None:
    prompts = load_prompts()
    assert len(prompts) == 50
    assert all(isinstance(p, str) and len(p) > 0 for p in prompts)
