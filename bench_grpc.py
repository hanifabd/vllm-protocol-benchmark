"""
Benchmark vLLM's gRPC serving endpoint (vllm serve --grpc), added in recent
vLLM releases and backed by the external `smg-grpc-servicer` / proto stubs
in `smg-grpc-proto`.

*** RUN discover_grpc_schema.py FIRST ***
That script prints the exact service name, RPC method name, and message
field names for whatever version of smg-grpc-proto you have installed.

Start the server first (separate terminal):

    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
        --grpc \
        --host 0.0.0.0 --port 8010 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096

Then run:

    python bench_grpc.py --target localhost:8010 \
        --num-requests 100 --concurrency 10 --prompt-tokens 128 --max-tokens 128
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import grpc

from common import (
    MODEL_NAME,
    RequestResult,
    WorkloadRequest,
    build_workload,
    save_json,
    summarize,
)

from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc


async def send_one_request(
    stub,
    req: WorkloadRequest,
    model: str,
    timeout_s: float,
    debug: bool = False,
) -> RequestResult:
    result = RequestResult(request_id=req.request_id, success=False)

    request = vllm_engine_pb2.GenerateRequest(
        request_id=str(req.request_id),
        text=req.prompt,
        sampling_params=vllm_engine_pb2.SamplingParams(
            temperature=0.0,
            max_tokens=req.max_tokens,
        ),
        stream=True,
    )

    first_token_time: float | None = None
    last_token_time: float | None = None
    output_tokens = 0
    prompt_tokens = len(req.prompt.split())  # fallback estimation

    result.start_time = time.perf_counter()
    try:
        msg_i = 0
        call = stub.Generate(request, timeout=timeout_s)

        async for response in call:
            now = time.perf_counter()
            msg_i += 1

            # Capture TTFT on the VERY FIRST response message (matches REST SSE behavior)
            if first_token_time is None:
                first_token_time = now
                last_token_time = now

            if debug:
                which = "chunk" if response.HasField("chunk") else (
                    "complete" if response.HasField("complete") else "UNKNOWN")
                if which == "chunk":
                    print(f"  [msg {msg_i}] chunk: len(token_ids)={len(response.chunk.token_ids)} "
                          f"prompt_tokens={response.chunk.prompt_tokens} "
                          f"completion_tokens={response.chunk.completion_tokens}")
                elif which == "complete":
                    print(f"  [msg {msg_i}] complete: prompt_tokens={response.complete.prompt_tokens} "
                          f"completion_tokens={response.complete.completion_tokens} "
                          f"len(output_ids)={len(response.complete.output_ids)} "
                          f"finish_reason={response.complete.finish_reason!r}")

            if response.HasField("chunk"):
                chunk = response.chunk
                n_new_tokens = len(chunk.token_ids)

                if n_new_tokens > 0:
                    if last_token_time is not None and last_token_time != now:
                        gap = (now - last_token_time) / n_new_tokens
                        result.inter_token_latencies_s.extend([gap] * n_new_tokens)
                    last_token_time = now
                    output_tokens += n_new_tokens
                elif chunk.completion_tokens > 0:
                    output_tokens = max(output_tokens, chunk.completion_tokens)

                if chunk.prompt_tokens:
                    prompt_tokens = chunk.prompt_tokens

            elif response.HasField("complete"):
                complete = response.complete
                if complete.prompt_tokens:
                    prompt_tokens = complete.prompt_tokens
                if complete.completion_tokens:
                    output_tokens = max(output_tokens, complete.completion_tokens)

        result.end_time = time.perf_counter()

        if first_token_time is None:
            result.error = "No gRPC response received from server"
            return result

        result.ttft_s = first_token_time - result.start_time
        result.e2e_latency_s = result.end_time - result.start_time
        result.output_tokens = output_tokens
        result.prompt_tokens = prompt_tokens
        result.success = True

    except grpc.RpcError as e:
        result.error = f"grpc.RpcError: {e.code()} {e.details()}"
        if debug:
            print(f"Request {req.request_id} failed with gRPC error: {e}")
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"
        if debug:
            print(f"Request {req.request_id} failed with exception: {e}")

    return result


async def run_benchmark(
    target: str,
    model: str,
    workload: list[WorkloadRequest],
    concurrency: int,
    timeout_s: float,
    debug: bool = False,
) -> tuple[list[RequestResult], float]:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async with grpc.aio.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    ) as channel:
        stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)

        async def bound_send(req: WorkloadRequest):
            async with sem:
                is_first = debug and req.request_id == 0
                if is_first:
                    print(f"\n--- debug: request_id=0 (prompt_tokens={req.max_tokens} max_tokens) ---")
                return await send_one_request(stub, req, model, timeout_s, debug=is_first)

        start = time.perf_counter()
        tasks = [asyncio.create_task(bound_send(r)) for r in workload]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        wall = time.perf_counter() - start

    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="localhost:8010", help="host:port of the gRPC server")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--output", default="results/grpc_results.json")
    ap.add_argument("--debug", action="store_true",
                    help="dump raw per-message field values for request_id=0 in each concurrency level")
    args = ap.parse_args()

    all_summaries = []
    for c in args.concurrency:
        print(f"\n=== gRPC | concurrency={c} ===")
        workload = build_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            seed=1000 + c,
        )
        results, wall = asyncio.run(
            run_benchmark(args.target, args.model, workload, c, args.timeout_s, debug=args.debug)
        )
        summary = summarize(results, wall, protocol="grpc", concurrency=c)
        print(json.dumps(summary, indent=2))
        all_summaries.append(summary)

    save_json(all_summaries, args.output)


if __name__ == "__main__":
    main()