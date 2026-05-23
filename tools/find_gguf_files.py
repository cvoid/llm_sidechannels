"""Print GGUF blob paths for ollama-managed models.

Usage:
    python tools/find_gguf_files.py
    python tools/find_gguf_files.py --models-dir /path/to/ollama/models
    python tools/find_gguf_files.py qwen2.5:7b-instruct-q4_K_M qwen2.5:0.5b-instruct-q8_0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SYSTEM_MODELS_DIR = pathlib.Path("/usr/share/ollama/.ollama/models")

EXP1_MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:0.5b-instruct-q8_0",
]


def find_blob(model_tag: str, models_dir: pathlib.Path) -> pathlib.Path:
    if ":" not in model_tag:
        raise ValueError(f"expected model:tag format, got {model_tag!r}")
    model, tag = model_tag.split(":", 1)
    manifest_path = models_dir / "manifests/registry.ollama.ai/library" / model / tag
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}\n"
            f"  Is the model pulled? Run: ollama pull {model_tag}"
        )
    data: dict[str, list[dict[str, str]]] = json.loads(manifest_path.read_text())
    digest = next(
        (
            layer["digest"]
            for layer in data["layers"]
            if layer["mediaType"] == "application/vnd.ollama.image.model"
        ),
        None,
    )
    if digest is None:
        raise RuntimeError(f"no model layer found in manifest for {model_tag}")
    blob_path = models_dir / "blobs" / digest.replace(":", "-")
    if not blob_path.exists():
        raise FileNotFoundError(f"blob missing: {blob_path}")
    return blob_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "models",
        nargs="*",
        default=EXP1_MODELS,
        metavar="model:tag",
        help="models to locate (default: exp1 target + draft pair)",
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=SYSTEM_MODELS_DIR,
        help=f"ollama models directory (default: {SYSTEM_MODELS_DIR})",
    )
    args = parser.parse_args()

    ok = True
    for model_tag in args.models:
        try:
            path = find_blob(model_tag, args.models_dir)
            print(f"{model_tag}: {path}")
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"ERROR {model_tag}: {e}", file=sys.stderr)
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
