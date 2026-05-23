"""Token aggregation defense proxy.

Sits between nginx and llama.cpp. Buffers `batch_size` SSE data events
before forwarding them as a single write, collapsing N per-iteration TLS
records into one. The attacker observes one large chunk instead of N
individual per-iteration boundaries, degrading fingerprinting accuracy.

Wei et al. (arXiv:2411.01076) §5.1: aggregating over N iterations reduces
attack accuracy by up to 50% with no change to payload size. A batch_size
of 4-8 gives the best accuracy/latency tradeoff in their evaluation.

Architecture:
    client -> nginx:8444 (TLS) -> this proxy:8082 -> llama.cpp:8080

The existing undefended path (nginx:8443 -> llama.cpp:8080) is unchanged.

Usage:
    uv run python -m defend.aggregate --batch-size 4
    sudo nginx -s reload   # after installing serve/nginx_defend.conf
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator

import aiohttp
from aiohttp import web

log = logging.getLogger(__name__)


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
    batch_size: int,
) -> web.StreamResponse:
    body = await request.read()
    # Strip hop-by-hop headers before forwarding upstream.
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

            batch: list[bytes] = []
            async for event in _iter_sse_events(upstream_resp.content):
                if event.lstrip().startswith(b"data:"):
                    batch.append(event)
                    if len(batch) >= batch_size:
                        await response.write(b"".join(batch))
                        batch.clear()
                else:
                    # Non-data SSE lines (comments, retry directives, blank
                    # preamble): flush any buffered events first, then pass
                    # through immediately so the client doesn't stall.
                    if batch:
                        await response.write(b"".join(batch))
                        batch.clear()
                    await response.write(event)

            # Flush any remaining buffered events at end of stream.
            if batch:
                await response.write(b"".join(batch))

            await response.write_eof()
    return response


def make_app(upstream: str, batch_size: int) -> web.Application:
    app = web.Application()

    async def handler(request: web.Request) -> web.StreamResponse:
        return await _proxy(request, upstream, batch_size)

    app.router.add_route("*", r"/{path_info:.*}", handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=4,
                        help="SSE data events to buffer per write (default: 4)")
    parser.add_argument("--port", type=int, default=8082,
                        help="port to listen on (default: 8082)")
    parser.add_argument("--upstream", default="http://127.0.0.1:8080",
                        help="upstream llama.cpp base URL (default: http://127.0.0.1:8080)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("aggregation proxy  batch_size=%d  port=%d  upstream=%s",
             args.batch_size, args.port, args.upstream)

    app = make_app(args.upstream, args.batch_size)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
