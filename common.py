"""
Shared utilities for the vLLM REST vs gRPC benchmark.

Both bench_rest.py and bench_grpc.py import from here so that the two
protocols are hit with *identical* workloads (same prompts, same token
targets, same concurrency schedule) -> apples-to-apples comparison.
"""
from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

# A small pool of source sentences we recombine to build prompts of
# controllable, roughly-known token length without needing an external
# dataset (ShareGPT etc.) downloaded up front. Feel free to swap this for
# real ShareGPT/random-token prompts later -- what matters for the
# REST-vs-gRPC comparison is that both protocols see the *same* prompts.
_LOREM_POOL = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    "Large language models are trained on massive corpora of text data. "
    "Kubernetes orchestrates containerized workloads across a cluster of nodes. "
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose. "
    "The stock market fluctuated wildly after the central bank announcement. "
    "In distributed systems, consensus protocols ensure nodes agree on state. "
    "The chef carefully plated the dish with a drizzle of balsamic reduction. "
    "Quantum computers exploit superposition to explore many states at once. "
    "The novel's protagonist wrestled with questions of identity and memory. "
    "Renewable energy sources are becoming increasingly cost competitive. "
).split(". ")


def build_prompt(target_tokens: int, seed: int) -> str:
    """Builds a pseudo-random prompt roughly `target_tokens` words long.

    Word count is a rough proxy for token count (good enough for controlling
    relative prompt sizes across a benchmark sweep; exact token count is
    measured downstream via the server's own tokenizer/usage stats).
    """
    rng = random.Random(seed)
    words: List[str] = []
    while len(words) < target_tokens:
        sentence = rng.choice(_LOREM_POOL)
        words.extend(sentence.split())
    words = words[:target_tokens]
    return "Continue the following passage in a natural way:\n" + " ".join(words)


@dataclass
class WorkloadRequest:
    request_id: int
    prompt: str
    max_tokens: int


def build_workload(
    num_requests: int,
    prompt_tokens: int = 128,
    max_tokens: int = 128,
    seed: int = 1234,
) -> List[WorkloadRequest]:
    reqs = []
    for i in range(num_requests):
        reqs.append(
            WorkloadRequest(
                request_id=i,
                prompt=build_prompt(prompt_tokens, seed=seed + i),
                max_tokens=max_tokens,
            )
        )
    return reqs


# ---------------------------------------------------------------------------
# Per-request result + aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: int
    success: bool
    ttft_s: Optional[float] = None          # time to first token
    e2e_latency_s: Optional[float] = None   # full request wall time
    inter_token_latencies_s: List[float] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    # raw send/finish timestamps, useful for computing achieved throughput
    start_time: float = 0.0
    end_time: float = 0.0


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def summarize(results: List[RequestResult], wall_clock_s: float, protocol: str, concurrency: int) -> dict:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2es = [r.e2e_latency_s for r in ok if r.e2e_latency_s is not None]
    all_itls = [itl for r in ok for itl in r.inter_token_latencies_s]

    # TPOT: mean time-per-output-token *excluding* the first token, per request
    tpots = []
    for r in ok:
        if r.output_tokens > 1 and r.e2e_latency_s is not None and r.ttft_s is not None:
            decode_time = r.e2e_latency_s - r.ttft_s
            n_decode_tokens = r.output_tokens - 1
            if n_decode_tokens > 0:
                tpots.append(decode_time / n_decode_tokens)

    total_output_tokens = sum(r.output_tokens for r in ok)
    total_input_tokens = sum(r.prompt_tokens for r in ok)

    summary = {
        "protocol": protocol,
        "concurrency": concurrency,
        "num_requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(failed),
        "benchmark_duration_s": round(wall_clock_s, 4),
        "request_throughput_req_s": round(len(ok) / wall_clock_s, 4) if wall_clock_s > 0 else 0,
        "output_token_throughput_tok_s": round(total_output_tokens / wall_clock_s, 4) if wall_clock_s > 0 else 0,
        "total_token_throughput_tok_s": round((total_output_tokens + total_input_tokens) / wall_clock_s, 4) if wall_clock_s > 0 else 0,
        "ttft_ms": _stats_block(ttfts, scale=1000),
        "tpot_ms": _stats_block(tpots, scale=1000),
        "itl_ms": _stats_block(all_itls, scale=1000),
        "e2e_latency_ms": _stats_block(e2es, scale=1000),
        "errors": [r.error for r in failed][:5],  # sample of errors
    }
    return summary


def _stats_block(values: List[float], scale: float = 1.0) -> dict:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p99": None, "min": None, "max": None}
    scaled = [v * scale for v in values]
    return {
        "mean": round(statistics.mean(scaled), 3),
        "median": round(statistics.median(scaled), 3),
        "p90": round(_percentile(scaled, 90), 3),
        "p99": round(_percentile(scaled, 99), 3),
        "min": round(min(scaled), 3),
        "max": round(max(scaled), 3),
    }


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved -> {path}")
