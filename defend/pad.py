"""Packet padding defense proxy.

Adds random or fixed-size padding to each SSE data event before forwarding,
obscuring per-iteration TLS record sizes from a passive network observer.

Two modes:
  random  Add a uniform random number of bytes in [0, max_pad] to each event.
          Wei et al. (arXiv:2411.01076) §5.2: at max_pad=512, reduces
          fingerprinting accuracy by ~70% with ~8.7x payload overhead.
  fixed   Pad every event to a constant size, fully collapsing size variation.
          Defeats the attack entirely but at up to 230x payload overhead.

Padding is injected as an SSE comment line (":p=<bytes>") inside each event,
before its blank-line terminator. SSE clients ignore comment lines, so the
JSON payload is received unchanged.

Example -- original event:
    data: {"choices": [{"delta": {"content": "tok"}}]}\n
    \n

After padding with 20 bytes:
    data: {"choices": [{"delta": {"content": "tok"}}]}\n
    :p=xxxxxxxxxxxxxxx\n
    \n

Architecture:
    client -> nginx:8445 (TLS) -> this proxy:8083 -> llama.cpp:8080

The undefended path (:8443) and aggregation proxy (:8444) are unchanged.

Usage:
    uv run python -m defend.pad --mode random --max-pad 512
    uv run python -m defend.pad --mode fixed --fixed-size 1500
    sudo nginx -s reload

Profile against the padded endpoint:
    uv run python tools/profile.py --port 8445 --out-dir data/raw_defend/pad512/temp_0.3
"""
from __future__ import annotations

import argparse
import logging
import random
from collections.abc import AsyncIterator

import aiohttp
from aiohttp import web

log = logging.getLogger(__name__)

_PAD_CHAR = b"x"
_COMMENT_OVERHEAD = 4  # len(b":p=\n")


def _pad_event(event: bytes, mode: str, max_pad: int, fixed_size: int) -> bytes:
    """Inject an SSE comment line into event to add padding bytes.

    The comment line ":p=<chars>\\n" is inserted before the blank-line
    terminator, adding exactly `extra` bytes to the event size.
    """
    if mode == "random":
        extra = random.randint(0, max_pad)
    else:
        extra = max(0, fixed_size - len(event))

    if extra < _COMMENT_OVERHEAD:
        return event

    # event = b"data: ...\n\n"
    # result = b"data: ...\n" + b":p=xxx\n" + b"\n"
    comment = b":p=" + _PAD_CHAR * (extra - _COMMENT_OVERHEAD) + b"\n"
    return event[:-1] + comment + b"\n"


async def _iter_sse_events(content: aiohttp.StreamReader) -> AsyncIterator[bytes]:
    """Yield one complete SSE event (including its trailing blank line) at a time."""
    buf: list[bytes] = []
    async for raw_line in content:
        buf.append(raw_line)
        if raw_line.rstrip(b"\r\n") == b"":
            if buf:
                yield b"".join(buf)
                buf.clear()
    if buf:
        yield b"".join(buf)


async def _proxy(
    request: web.Request,
    upstream: str,
    mode: str,
    max_pad: int,
    fixed_size: int,
) -> web.StreamResponse:
    body = await request.read()
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding",
                             "connection", "keep-alive")
    }

    async with aiohttp.ClientSession() as session:
        async with session.request(
            request.method,
            f"{upstream}{request.path_qs}",
            headers=forward_headers,
            data=body,
        ) as upstream_resp:
            resp_headers = {
                k: v for k, v in upstream_resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-length",
                                     "connection", "keep-alive")
            }
            response = web.StreamResponse(
                status=upstream_resp.status,
                headers=resp_headers,
            )
            await response.prepare(request)

            async for event in _iter_sse_events(upstream_resp.content):
                if event.lstrip().startswith(b"data:"):
                    await response.write(_pad_event(event, mode, max_pad, fixed_size))
                else:
                    await response.write(event)

            await response.write_eof()
    return response


def make_app(upstream: str, mode: str, max_pad: int, fixed_size: int) -> web.Application:
    app = web.Application()

    async def handler(request: web.Request) -> web.StreamResponse:
        return await _proxy(request, upstream, mode, max_pad, fixed_size)

    app.router.add_route("*", r"/{path_info:.*}", handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["random", "fixed"], default="random",
                        help="padding mode: random [0, max_pad] or fixed size (default: random)")
    parser.add_argument("--max-pad", type=int, default=512,
                        help="max bytes to add per event in random mode (default: 512)")
    parser.add_argument("--fixed-size", type=int, default=1500,
                        help="target event size in bytes in fixed mode (default: 1500)")
    parser.add_argument("--port", type=int, default=8083,
                        help="port to listen on (default: 8083)")
    parser.add_argument("--upstream", default="http://127.0.0.1:8080",
                        help="upstream llama.cpp base URL (default: http://127.0.0.1:8080)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "random":
        log.info("padding proxy  mode=random  max_pad=%d  port=%d  upstream=%s",
                 args.max_pad, args.port, args.upstream)
    else:
        log.info("padding proxy  mode=fixed  fixed_size=%d  port=%d  upstream=%s",
                 args.fixed_size, args.port, args.upstream)

    app = make_app(args.upstream, args.mode, args.max_pad, args.fixed_size)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
