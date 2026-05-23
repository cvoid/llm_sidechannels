from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from collect.query import send


def _make_sse_stream(tokens: list[str]) -> list[str]:
    lines: list[str] = []
    for token in tokens:
        data = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(data)}")
    lines.append("data: [DONE]")
    return lines


def test_send_concatenates_tokens() -> None:
    sse_lines = _make_sse_stream(["Hello", ", ", "world"])
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = iter(sse_lines)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=mock_client):
        result = send("test prompt", "localhost")

    assert result == "Hello, world"


def test_send_skips_empty_delta() -> None:
    lines = [
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"delta": {"content": "hi"}}]}',
        "data: [DONE]",
    ]
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=mock_client):
        result = send("test", "localhost")

    assert result == "hi"


def test_send_includes_system_prompt() -> None:
    sse_lines = _make_sse_stream(["ok"])
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = iter(sse_lines)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=mock_client):
        send("hello", "localhost", system_prompt="You are a doctor.")

    _, kwargs = mock_client.stream.call_args
    messages = kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a doctor."
