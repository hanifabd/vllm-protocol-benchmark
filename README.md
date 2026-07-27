# vLLM REST vs gRPC Benchmark — Qwen2.5-0.5B-Instruct

## Why gRPC needs one extra step

vLLM v0.26.0's gRPC server (`vllm serve --grpc`) is backed by proto
definitions published *outside* the vLLM repo, in `smg-grpc-proto` /
`smg-grpc-servicer`, which is still under active development. That means
the exact RPC method name and message field names can differ between
patch versions. Rather than hand you a client hardcoded against field
names I can't verify against your exact installed version, `bench_grpc.py`
has 3 clearly marked `TODO` blocks — `discover_grpc_schema.py` prints
everything you need to fill them in, in about 2 minutes.

The REST script (`bench_rest.py`) is complete and needs no edits — vLLM's
OpenAI-compatible `/v1/completions` endpoint is stable.

## 1. Install

```bash
python -m venv venv && source venv/bin/activate

pip install "vllm==0.26.0"
pip install aiohttp grpcio grpcio-tools smg-grpc-proto
```

## 2. Start the REST server (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

## 3. Start the gRPC server (same or a second GPU)

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --grpc \
    --host 0.0.0.0 --port 8010 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

> Run REST and gRPC as separate server processes (ideally sequentially, or
> on separate GPUs) so one benchmark doesn't steal GPU time from the other
> and skew results.

## 4. Discover the gRPC schema, fill in the TODOs

```bash
python discover_grpc_schema.py
```

This prints every service, RPC method, and message field defined in your
installed `smg-grpc-proto`. Open `bench_grpc.py` and fix the 3 marked
spots (request message construction, RPC method call, response chunk
field access) to match.

## 5. Run both benchmarks

```bash
# REST
python bench_rest.py --base-url http://localhost:8000 \
    --num-requests 100 \
    --concurrency 1 5 10 25 50 \
    --prompt-tokens 128 --max-tokens 128 \
    --output results/rest_results.json

# gRPC (after filling in the TODOs)
python bench_grpc.py --target localhost:8001 \
    --num-requests 100 \
    --concurrency 1 5 10 25 50 \
    --prompt-tokens 128 --max-tokens 128 \
    --output results/grpc_results.json
```

Both scripts sweep the same concurrency levels against the same
deterministically-generated prompts, so the two JSON result files are
directly comparable.

## 6. Compare

```bash
python compare_results.py \
    --rest results/rest_results.json \
    --grpc results/grpc_results.json \
    --csv results/comparison.csv
```

Prints a side-by-side table and writes `results/comparison.csv` with, per
concurrency level and protocol:

- request throughput (req/s), output token throughput (tok/s)
- TTFT mean / p99
- TPOT mean / p99
- ITL mean
- end-to-end latency mean / p99
- success/failure counts

## What each metric tells you

| Metric | What it isolates |
|---|---|
| TTFT | Time to first token — sensitive to serialization/connection overhead, most likely place to see REST vs gRPC differ |
| TPOT / ITL | Decode-loop speed — should be ~identical between protocols; a gap here signals a client-side streaming/parsing bug, not a real protocol difference |
| Throughput (req/s, tok/s) | Overall system capacity at a given concurrency |
| p99 vs mean | Tail latency — where transport overhead compounds under load |

Repeat the prompt-tokens/max-tokens sweep with short (e.g. 32/32) and long
(e.g. 512/512) values — protocol overhead is most visible on short
request/short response workloads, and gets swamped by GPU compute time on
long ones.
