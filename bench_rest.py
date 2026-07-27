"""
Benchmark vLLM's REST (OpenAI-compatible) streaming completions endpoint.

Start the server first (separate terminal):

    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096

Then run:

    python bench_rest.py --base-url http://localhost:8000 \
        --num-requests 100 --concurrency 10 --prompt-tokens 128 --max-tokens 128
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp

from common import (
    MODEL_NAME,
    RequestResult,
    WorkloadRequest,
    build_workload,
    save_json,
    summarize,
)


async def send_one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    req: WorkloadRequest,
    model: str,
    timeout_s: float,
) -> RequestResult:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    result = RequestResult(request_id=req.request_id, success=False)
    result.start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    output_tokens = 0
    prompt_tokens = 0

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                choices = chunk.get("choices", [])
                text_piece = choices[0].get("text", "") if choices else ""

                if text_piece:
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        result.inter_token_latencies_s.append(now - last_token_time)
                    last_token_time = now
                    output_tokens += 1

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)

            result.end_time = time.perf_counter()
            if first_token_time is None:
                result.error = "No tokens received"
                return result

            result.ttft_s = first_token_time - result.start_time
            result.e2e_latency_s = result.end_time - result.start_time
            result.output_tokens = output_tokens
            result.prompt_tokens = prompt_tokens or len(req.prompt.split())
            result.success = True

    except asyncio.TimeoutError:
        result.error = "timeout"
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"

    return result


async def run_benchmark(
    base_url: str,
    model: str,
    workload: list[WorkloadRequest],
    concurrency: int,
    timeout_s: float,
) -> tuple[list[RequestResult], float]:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def bound_send(req: WorkloadRequest):
            async with sem:
                return await send_one_request(session, base_url, req, model, timeout_s)

        start = time.perf_counter()
        tasks = [asyncio.create_task(bound_send(r)) for r in workload]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        wall = time.perf_counter() - start

    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 25, 50],
                     help="one or more concurrency levels to sweep")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--output", default="results/rest_results.json")
    args = ap.parse_args()

    all_summaries = []
    for c in args.concurrency:
        print(f"\n=== REST | concurrency={c} ===")
        workload = build_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            seed=1000 + c,  # vary seed slightly per level, kept deterministic
        )
        results, wall = asyncio.run(
            run_benchmark(args.base_url, args.model, workload, c, args.timeout_s)
        )
        summary = summarize(results, wall, protocol="rest", concurrency=c)
        print(json.dumps(summary, indent=2))
        all_summaries.append(summary)

    save_json(all_summaries, args.output)


if __name__ == "__main__":
    main()
