"""Constant-rate chunk streaming defense proxy.

Buffers the complete SSE response body, then retransmits it as fixed-size
chunks with a configurable inter-chunk delay. An observer sees packets of
constant size at regular intervals, destroying both per-iteration byte-size
variation and timing correlation.

Two modes depending on --interval-ms:

  0 (burst)   Buffer the whole response, send everything at once. All bytes
              arrive in a single burst; the feature extractor groups them into
              one iteration, reducing the signal to total response length only.

  N > 0       Buffer then stream one chunk every N ms. Observer sees [chunk_size,
              chunk_size, ...] for the feature vector, which is constant across
              all prompts. Accuracy should approach chance (1/n_classes).

Trade-off: the client receives no bytes until the model finishes generating,
adding latency equal to full generation time (typically 20-60 s).

Architecture:
    client -> nginx:8446 (TLS) -> this proxy:8084 -> llama.cpp:8080

Usage:
    uv run python -m defend.cbr --chunk-size 512 --interval-ms 20
    sudo nginx -s reload
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import aiohttp
from aiohttp import web

log = logging.getLogger(__name__)


async def _proxy(
    request: web.Request,
    upstream: str,
    chunk_size: int,
    interval_ms: float,
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
            # Buffer the entire response before transmitting anything.
            raw = await upstream_resp.read()

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

            if interval_ms == 0:
                # Burst: send all buffered bytes in one write. TCP may
                # fragment into MTU-sized segments but they all arrive in
                # rapid succession, collapsing to a single feature-extractor
                # iteration.
                await response.write(raw)
            else:
                delay = interval_ms / 1000.0
                for offset in range(0, len(raw), chunk_size):
                    await response.write(raw[offset: offset + chunk_size])
                    await asyncio.sleep(delay)

            await response.write_eof()
    return response


def make_app(upstream: str, chunk_size: int, interval_ms: float) -> web.Application:
    app = web.Application()

    async def handler(request: web.Request) -> web.StreamResponse:
        return await _proxy(request, upstream, chunk_size, interval_ms)

    app.router.add_route("*", r"/{path_info:.*}", handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="bytes per transmitted chunk (default: 512)")
    parser.add_argument("--interval-ms", type=float, default=20.0,
                        help="ms between chunks; 0 = burst mode (default: 20)")
    parser.add_argument("--port", type=int, default=8084,
                        help="port to listen on (default: 8084)")
    parser.add_argument("--upstream", default="http://127.0.0.1:8080",
                        help="upstream llama.cpp base URL (default: http://127.0.0.1:8080)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("cbr proxy  chunk_size=%d  interval_ms=%.1f  port=%d  upstream=%s",
             args.chunk_size, args.interval_ms, args.port, args.upstream)

    app = make_app(args.upstream, args.chunk_size, args.interval_ms)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
