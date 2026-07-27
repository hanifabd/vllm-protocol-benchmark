"""
Benchmark vLLM gRPC serving endpoint.

Usage:

    python bench_grpc.py \
        --target localhost:8010 \
        --num-requests 100 \
        --concurrency 1 5 10 25 50 \
        --prompt-tokens 128 \
        --max-tokens 128 \
        --output results/grpc_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import grpc

from common import (
    MODEL_NAME,
    RequestResult,
    WorkloadRequest,
    build_workload,
    save_json,
    summarize,
)

from smg_grpc_proto import (
    vllm_engine_pb2,
    vllm_engine_pb2_grpc,
)


# ============================================================
# Configuration
# ============================================================

# Possible protobuf request message names.
REQUEST_MESSAGE_CANDIDATES = [
    "GenerateRequest",
    "CompletionRequest",
    "GenerateTextRequest",
    "Request",
]

# Possible gRPC stub class names.
STUB_CANDIDATES = [
    "VllmEngineStub",
    "VLLMEngineStub",
    "VllmEngineServiceStub",
    "VLLMEngineServiceStub",
    "EngineStub",
]

# Possible RPC method names.
RPC_CANDIDATES = [
    "Generate",
    "GenerateStream",
    "Completion",
    "Complete",
    "Chat",
    "ChatCompletion",
]


# ============================================================
# Proto helpers
# ============================================================

def get_message_class():
    """
    Find the request protobuf class available in the installed
    smg_grpc_proto package.
    """

    for name in REQUEST_MESSAGE_CANDIDATES:
        message_class = getattr(vllm_engine_pb2, name, None)

        if message_class is not None:
            print(f"[gRPC] Using request message: {name}")
            return message_class

    available = [
        name
        for name in dir(vllm_engine_pb2)
        if not name.startswith("_")
    ]

    raise RuntimeError(
        "Could not find a supported request protobuf message.\n"
        f"Available protobuf symbols:\n{available}"
    )


def get_stub_class():
    """
    Find the gRPC Stub class available in the installed
    smg_grpc_proto package.
    """

    for name in STUB_CANDIDATES:
        stub_class = getattr(vllm_engine_pb2_grpc, name, None)

        if stub_class is not None:
            print(f"[gRPC] Using stub: {name}")
            return stub_class

    available = [
        name
        for name in dir(vllm_engine_pb2_grpc)
        if not name.startswith("_")
    ]

    raise RuntimeError(
        "Could not find a supported gRPC Stub class.\n"
        f"Available gRPC symbols:\n{available}"
    )


def get_available_fields(message_class) -> set[str]:
    """
    Return protobuf field names.
    """

    return {
        field.name
        for field in message_class.DESCRIPTOR.fields
    }


def print_schema():
    """
    Print request protobuf fields and available RPC methods.
    """

    print("\n========== gRPC SCHEMA ==========")

    print("\nProtobuf messages:")

    for name in dir(vllm_engine_pb2):
        if name.startswith("_"):
            continue

        obj = getattr(vllm_engine_pb2, name)

        if hasattr(obj, "DESCRIPTOR"):
            descriptor = getattr(obj, "DESCRIPTOR", None)

            if descriptor is not None and hasattr(descriptor, "fields"):
                fields = [
                    field.name
                    for field in descriptor.fields
                ]

                print(f"  {name}: {fields}")

    print("\ngRPC stubs:")

    for name in dir(vllm_engine_pb2_grpc):
        if name.startswith("_"):
            continue

        obj = getattr(vllm_engine_pb2_grpc, name)

        if isinstance(obj, type):
            methods = [
                method
                for method in dir(obj)
                if not method.startswith("_")
            ]

            print(f"  {name}: {methods}")

    print("\n==================================\n")


# ============================================================
# Request construction
# ============================================================

def build_grpc_request(
    message_class,
    req: WorkloadRequest,
    model: str,
):
    """
    Build the protobuf request dynamically.

    Different smg-grpc-proto versions may expose different fields.

    This function checks the installed protobuf schema and only
    sets fields that actually exist.
    """

    available_fields = get_available_fields(message_class)

    kwargs: dict[str, Any] = {}

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt_fields = [
        "prompt",
        "text",
        "input",
        "query",
    ]

    for field in prompt_fields:
        if field in available_fields:
            kwargs[field] = req.prompt
            break

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    # IMPORTANT:
    # Your error indicates that `model` is NOT a field in
    # GenerateRequest.
    #
    # Therefore we only set it if the installed proto actually
    # supports it.

    model_fields = [
        "model",
        "model_name",
        "model_id",
    ]

    for field in model_fields:
        if field in available_fields:
            kwargs[field] = model
            break

    # --------------------------------------------------------
    # Max tokens
    # --------------------------------------------------------

    max_tokens_fields = [
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
    ]

    for field in max_tokens_fields:
        if field in available_fields:
            kwargs[field] = req.max_tokens
            break

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if "temperature" in available_fields:
        kwargs["temperature"] = 0.0

    # --------------------------------------------------------
    # Top P
    # --------------------------------------------------------

    if "top_p" in available_fields:
        kwargs["top_p"] = 1.0

    # --------------------------------------------------------
    # Stream
    # --------------------------------------------------------

    if "stream" in available_fields:
        kwargs["stream"] = True

    # --------------------------------------------------------
    # Build message
    # --------------------------------------------------------

    if not kwargs:
        raise RuntimeError(
            "Could not map WorkloadRequest to protobuf request.\n"
            f"Available fields: {sorted(available_fields)}"
        )

    print_once = getattr(build_grpc_request, "_printed", False)

    if not print_once:
        print("\n[gRPC] Request fields detected:")
        print(
            json.dumps(
                {
                    "available_fields": sorted(available_fields),
                    "using_fields": kwargs,
                },
                indent=2,
                default=str,
            )
        )

        build_grpc_request._printed = True

    try:
        return message_class(**kwargs)

    except Exception as exc:

        raise RuntimeError(
            "Failed to construct gRPC protobuf request.\n"
            f"Available fields: {sorted(available_fields)}\n"
            f"Attempted fields: {kwargs}\n"
            f"Original error: {exc}"
        ) from exc


# ============================================================
# RPC discovery
# ============================================================

def find_rpc_method(stub):
    """
    Find a supported RPC method dynamically.
    """

    for name in RPC_CANDIDATES:
        method = getattr(stub, name, None)

        if method is not None:
            print(f"[gRPC] Using RPC method: {name}")
            return method

    available = [
        name
        for name in dir(stub)
        if not name.startswith("_")
    ]

    raise RuntimeError(
        "Could not find supported gRPC RPC method.\n"
        f"Available methods:\n{available}"
    )


# ============================================================
# Response parsing
# ============================================================

def extract_text_from_chunk(chunk) -> str:
    """
    Extract generated text from a response chunk.

    Supports several possible response schemas.
    """

    # Direct string-like fields
    fields = [
        "text",
        "token",
        "content",
        "delta",
        "generated_text",
        "output",
    ]

    for field in fields:

        if not hasattr(chunk, field):
            continue

        value = getattr(chunk, field)

        if isinstance(value, str):
            return value

        # Handle nested objects such as delta.content
        if hasattr(value, "content"):
            content = getattr(value, "content")

            if isinstance(content, str):
                return content

        if hasattr(value, "text"):
            text = getattr(value, "text")

            if isinstance(text, str):
                return text

    return ""


def extract_token_count(
    chunk,
    field_names: list[str],
) -> int | None:
    """
    Extract integer token count from a protobuf response.
    """

    for field in field_names:

        if not hasattr(chunk, field):
            continue

        value = getattr(chunk, field)

        if isinstance(value, int):
            return value

    return None


# ============================================================
# Single request
# ============================================================

async def send_one_request(
    stub,
    request_message_class,
    req: WorkloadRequest,
    model: str,
    timeout_s: float,
) -> RequestResult:

    result = RequestResult(
        request_id=req.request_id,
        success=False,
    )

    # --------------------------------------------------------
    # Build protobuf request
    # --------------------------------------------------------

    try:

        request = build_grpc_request(
            message_class=request_message_class,
            req=req,
            model=model,
        )

    except Exception as exc:

        result.error = (
            f"Request construction error: "
            f"{type(exc).__name__}: {exc}"
        )

        return result

    # --------------------------------------------------------
    # Benchmark timing
    # --------------------------------------------------------

    first_token_time = None
    last_token_time = None

    output_tokens = 0

    prompt_tokens = len(
        req.prompt.split()
    )

    result.start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Find RPC
        # ----------------------------------------------------

        rpc = find_rpc_method(stub)

        # ----------------------------------------------------
        # Call RPC
        # ----------------------------------------------------

        call = rpc(
            request,
            timeout=timeout_s,
        )

        # ----------------------------------------------------
        # Streaming response
        # ----------------------------------------------------

        async for chunk in call:

            now = time.perf_counter()

            # -----------------------------------------------
            # Extract generated text
            # -----------------------------------------------

            text_piece = extract_text_from_chunk(
                chunk
            )

            if text_piece:

                # First generated chunk
                if first_token_time is None:

                    first_token_time = now

                else:

                    if last_token_time is not None:

                        result.inter_token_latencies_s.append(
                            now - last_token_time
                        )

                last_token_time = now

                # ------------------------------------------------
                # IMPORTANT:
                #
                # A chunk != necessarily one token.
                #
                # This is only an approximation.
                # ------------------------------------------------

                output_tokens += len(
                    text_piece.split()
                )

            # -----------------------------------------------
            # Prompt token count
            # -----------------------------------------------

            detected_prompt_tokens = extract_token_count(
                chunk,
                [
                    "prompt_tokens",
                    "input_tokens",
                    "num_prompt_tokens",
                ],
            )

            if detected_prompt_tokens:
                prompt_tokens = detected_prompt_tokens

            # -----------------------------------------------
            # Output token count
            # -----------------------------------------------

            detected_output_tokens = extract_token_count(
                chunk,
                [
                    "completion_tokens",
                    "output_tokens",
                    "generated_tokens",
                    "num_generated_tokens",
                ],
            )

            if detected_output_tokens is not None:
                output_tokens = detected_output_tokens

        # ----------------------------------------------------
        # End timing
        # ----------------------------------------------------

        result.end_time = time.perf_counter()

        # ----------------------------------------------------
        # No response
        # ----------------------------------------------------

        if first_token_time is None:

            result.error = (
                "No generated text received "
                "from gRPC stream"
            )

            return result

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        result.ttft_s = (
            first_token_time
            - result.start_time
        )

        result.e2e_latency_s = (
            result.end_time
            - result.start_time
        )

        result.output_tokens = output_tokens

        result.prompt_tokens = prompt_tokens

        result.success = True

    except grpc.aio.AioRpcError as e:

        result.end_time = time.perf_counter()

        result.error = (
            f"grpc.AioRpcError: "
            f"{e.code()} "
            f"{e.details()}"
        )

    except grpc.RpcError as e:

        result.end_time = time.perf_counter()

        result.error = (
            f"grpc.RpcError: "
            f"{e.code()} "
            f"{e.details()}"
        )

    except asyncio.TimeoutError:

        result.end_time = time.perf_counter()

        result.error = (
            "Request timed out"
        )

    except Exception as e:

        result.end_time = time.perf_counter()

        result.error = (
            f"{type(e).__name__}: {e}"
        )

    return result


# ============================================================
# Benchmark
# ============================================================

async def run_benchmark(
    target: str,
    model: str,
    workload: list[WorkloadRequest],
    concurrency: int,
    timeout_s: float,
) -> tuple[list[RequestResult], float]:

    sem = asyncio.Semaphore(
        concurrency
    )

    results: list[RequestResult] = []

    # --------------------------------------------------------
    # Create gRPC channel
    # --------------------------------------------------------

    async with grpc.aio.insecure_channel(
        target,
        options=[
            (
                "grpc.max_send_message_length",
                64 * 1024 * 1024,
            ),
            (
                "grpc.max_receive_message_length",
                64 * 1024 * 1024,
            ),
        ],
    ) as channel:

        # ----------------------------------------------------
        # Create stub
        # ----------------------------------------------------

        stub_class = get_stub_class()

        stub = stub_class(
            channel
        )

        # ----------------------------------------------------
        # Find request message
        # ----------------------------------------------------

        request_message_class = (
            get_message_class()
        )

        # ----------------------------------------------------
        # Bounded request
        # ----------------------------------------------------

        async def bound_send(
            request: WorkloadRequest,
        ):

            async with sem:

                return await send_one_request(
                    stub=stub,
                    request_message_class=request_message_class,
                    req=request,
                    model=model,
                    timeout_s=timeout_s,
                )

        # ----------------------------------------------------
        # Start benchmark
        # ----------------------------------------------------

        start = time.perf_counter()

        tasks = [
            asyncio.create_task(
                bound_send(req)
            )
            for req in workload
        ]

        # ----------------------------------------------------
        # Collect completed requests
        # ----------------------------------------------------

        for coro in asyncio.as_completed(
            tasks
        ):

            result = await coro

            results.append(
                result
            )

        wall = (
            time.perf_counter()
            - start
        )

    return results, wall


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Benchmark vLLM gRPC endpoint"
        )
    )

    ap.add_argument(
        "--target",
        default="localhost:8010",
        help=(
            "gRPC server host:port"
        ),
    )

    ap.add_argument(
        "--model",
        default=MODEL_NAME,
        help=(
            "Model name. Only used if "
            "the protobuf supports a model field."
        ),
    )

    ap.add_argument(
        "--num-requests",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[
            1,
            5,
            10,
            25,
            50,
        ],
    )

    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--max-tokens",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
    )

    ap.add_argument(
        "--output",
        default=(
            "results/grpc_results.json"
        ),
    )

    ap.add_argument(
        "--show-schema",
        action="store_true",
        help=(
            "Print installed protobuf schema "
            "and exit."
        ),
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # Print schema
    # --------------------------------------------------------

    if args.show_schema:

        print_schema()

        return

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    import os

    os.makedirs(
        os.path.dirname(
            args.output
        ) or ".",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Benchmark all concurrency levels
    # --------------------------------------------------------

    all_summaries = []

    for concurrency in args.concurrency:

        print(
            f"\n=== gRPC | "
            f"concurrency={concurrency} ==="
        )

        # ----------------------------------------------------
        # Build workload
        # ----------------------------------------------------

        workload = build_workload(
            num_requests=args.num_requests,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            seed=1000 + concurrency,
        )

        # ----------------------------------------------------
        # Run benchmark
        # ----------------------------------------------------

        results, wall = asyncio.run(
            run_benchmark(
                target=args.target,
                model=args.model,
                workload=workload,
                concurrency=concurrency,
                timeout_s=args.timeout_s,
            )
        )

        # ----------------------------------------------------
        # Summarize
        # ----------------------------------------------------

        summary = summarize(
            results,
            wall,
            protocol="grpc",
            concurrency=concurrency,
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        all_summaries.append(
            summary
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        all_summaries,
        args.output,
    )

    print(
        f"\nResults saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()