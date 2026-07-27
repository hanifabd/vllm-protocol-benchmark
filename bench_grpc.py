"""
Benchmark vLLM's gRPC serving endpoint (vllm serve --grpc), added in recent
vLLM releases and backed by the external `smg-grpc-servicer` / proto stubs
in `smg-grpc-proto`.
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

# TODO (1) Resolved: Imports matching discover_grpc_schema.py
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc


async def send_one_request(
    stub: vllm_engine_pb2_grpc.VllmEngineStub,
    req: WorkloadRequest,
    model: str,
    timeout_s: float,
) -> RequestResult:
    result = RequestResult(request_id=req.request_id, success=False)

    # TODO (2) Resolved: Build SamplingParams & GenerateRequest matching schema
    sampling_params = vllm_engine_pb2.SamplingParams(
        max_tokens=req.max_tokens,
        temperature=0.0,
    )

    request = vllm_engine_pb2.GenerateRequest(
        request_id=req.request_id,
        text=req.prompt,
        sampling_params=sampling_params,
        stream=True,
    )

    first_token_time = None
    last_token_time = None
    output_tokens = 0
    prompt_tokens = len(req.prompt.split())  # fallback if not returned in response

    result.start_time = time.perf_counter()
    try:
        # TODO (3) Resolved: RPC method `Generate` and `GenerateResponse` stream field extraction
        call = stub.Generate(request, timeout=timeout_s)
        async for response in call:
            now = time.perf_counter()

            # Handle streaming chunks
            if response.HasField("chunk"):
                chunk = response.chunk
                
                # Each chunk delivers new token ID(s)
                num_new_tokens = len(chunk.token_ids)
                if num_new_tokens > 0:
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        result.inter_token_latencies_s.append(now - last_token_time)
                    
                    last_token_time = now
                    output_tokens += num_new_tokens

                if chunk.prompt_tokens:
                    prompt_tokens = chunk.prompt_tokens

            # Handle completion signal if present
            elif response.HasField("complete"):
                complete = response.complete
                if complete.prompt_tokens:
                    prompt_tokens = complete.prompt_tokens
                if complete.completion_tokens:
                    output_tokens = complete.completion_tokens

        result.end_time = time.perf_counter()
        if first_token_time is None:
            result.error = "No tokens received"
            return result

        result.ttft_s = first_token_time - result.start_time
        result.e2e_latency_s = result.end_time - result.start_time
        result.output_tokens = output_tokens
        result.prompt_tokens = prompt_tokens
        result.success = True

    except grpc.RpcError as e:
        result.error = f"grpc.RpcError: {e.code()} {e.details()}"
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"

    return result


async def run_benchmark(
    target: str,
    model: str,
    workload: list[WorkloadRequest],
    concurrency: int,
    timeout_s: float,
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
        # Resolved: Correct stub name `VllmEngineStub`
        stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)

        async def bound_send(req: WorkloadRequest):
            async with sem:
                return await send_one_request(stub, req, model, timeout_s)

        start = time.perf_counter()
        tasks = [asyncio.create_task(bound_send(r)) for r in workload]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        wall = time.perf_counter() - start

    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="localhost:8001", help="host:port of the gRPC server")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--output", default="results/grpc_results.json")
    args = ap.parse_args()

    all_summaries = []
    for c in args.concurrency:
        print(f"\n=== gRPC | concurrency={c} ===")
        workload = build_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            seed=1000 + c,  # same seed scheme as bench_rest.py -> same prompts
        )
        results, wall = asyncio.run(
            run_benchmark(args.target, args.model, workload, c, args.timeout_s)
        )
        summary = summarize(results, wall, protocol="grpc", concurrency=c)
        print(json.dumps(summary, indent=2))
        all_summaries.append(summary)

    save_json(all_summaries, args.output)


if __name__ == "__main__":
    main()