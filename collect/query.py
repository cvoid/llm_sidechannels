from __future__ import annotations

import json

import httpx


def send(
    prompt: str,
    host: str,
    port: int = 8443,
    temperature: float = 0.3,
    system_prompt: str = "",
    timeout: float = 120.0,
    model: str = "qwen2.5:7b-instruct-q4_K_M",
) -> str:
    url = f"https://{host}:{port}/v1/chat/completions"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    chunks: list[str] = []
    with httpx.Client(verify=False, timeout=timeout) as client:
        with client.stream(
            "POST",
            url,
            json={"model": model, "messages": messages, "temperature": temperature, "stream": True},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                data = json.loads(payload)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
    return "".join(chunks)
