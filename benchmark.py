import argparse
import asyncio
import time
import json
import numpy as np
from tabulate import tabulate
import aiohttp
import grpc

# Import vLLM gRPC generated stubs with fallback chain and Pylance type suppressions
try:
    import vllm.entrypoints.grpc_server.vllm_engine_pb2 as vllm_engine_pb2  # type: ignore
    import vllm.entrypoints.grpc_server.vllm_engine_pb2_grpc as vllm_engine_pb2_grpc  # type: ignore
except ImportError:
    try:
        from vllm.entrypoints.grpc import vllm_engine_pb2, vllm_engine_pb2_grpc  # type: ignore
    except ImportError:
        try:
            import vllm.entrypoints.grpc.protocol.vllm_engine_pb2 as vllm_engine_pb2  # type: ignore
            import vllm.entrypoints.grpc.protocol.vllm_engine_pb2_grpc as vllm_engine_pb2_grpc  # type: ignore
        except ImportError:
            vllm_engine_pb2 = None
            vllm_engine_pb2_grpc = None


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain quantum computing and its potential impact on modern cryptography in detail."
MAX_TOKENS = 128


class MetricCollector:
    def __init__(self):
        self.ttfts = []
        self.tpots = []
        self.total_latencies = []
        self.total_tokens = []

    def add_result(self, ttft, tpot, total_time, tokens_generated):
        self.ttfts.append(ttft)
        self.tpots.append(tpot)
        self.total_latencies.append(total_time)
        self.total_tokens.append(tokens_generated)

    def summary(self, duration):
        if not self.total_latencies:
            return {}
        
        return {
            "RPS": len(self.total_latencies) / duration,
            "Total Tokens/s": sum(self.total_tokens) / duration,
            "Mean TTFT (ms)": np.mean(self.ttfts) * 1000,
            "P95 TTFT (ms)": np.percentile(self.ttfts, 95) * 1000,
            "Mean TPOT (ms)": np.mean(self.tpots) * 1000,
            "P95 TPOT (ms)": np.percentile(self.tpots, 95) * 1000,
            "Mean Latency (s)": np.mean(self.total_latencies),
        }


# --- REST Streaming Worker ---
async def worker_rest(session: aiohttp.ClientSession, url: str, metrics: MetricCollector):
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.0
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    async with session.post(url, json=payload) as response:
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                data = json.loads(line[6:])
                if data["choices"][0]["text"]:
                    token_count += 1

    end_time = time.perf_counter()
    
    if token_count > 0 and first_token_time:
        ttft = first_token_time - start_time
        gen_duration = end_time - first_token_time
        tpot = gen_duration / token_count if token_count > 1 else gen_duration
        metrics.add_result(ttft, tpot, end_time - start_time, token_count)


# --- gRPC Streaming Worker ---
async def worker_grpc(stub, metrics: MetricCollector):
    if vllm_engine_pb2 is None:
        raise ImportError("Could not locate vLLM gRPC protobuf modules in current python environment.")

    request = vllm_engine_pb2.GenerateRequest(
        model=MODEL_NAME,
        prompt=PROMPT,
        stream=True,
        sampling_params=vllm_engine_pb2.SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.0
        )
    )
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    async for response in stub.Generate(request):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        
        if response.text:
            token_count += 1

    end_time = time.perf_counter()
    
    if token_count > 0 and first_token_time:
        ttft = first_token_time - start_time
        gen_duration = end_time - first_token_time
        tpot = gen_duration / token_count if token_count > 1 else gen_duration
        metrics.add_result(ttft, tpot, end_time - start_time, token_count)


# --- Benchmark Suite Runners ---
async def run_rest_suite(host: str, port: int, concurrency: int, total_requests: int):
    url = f"http://{host}:{port}/v1/completions"
    metrics = MetricCollector()
    connector = aiohttp.TCPConnector(limit=concurrency)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup request
        await worker_rest(session, url, MetricCollector())
        
        start = time.perf_counter()
        tasks = [worker_rest(session, url, metrics) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        
    return metrics.summary(duration)


async def run_grpc_suite(host: str, port: int, concurrency: int, total_requests: int):
    target = f"{host}:{port}"
    metrics = MetricCollector()
    
    async with grpc.aio.insecure_channel(target) as channel:
        stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
        
        # Warmup request
        await worker_grpc(stub, MetricCollector())
        
        start = time.perf_counter()
        tasks = [worker_grpc(stub, metrics) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        
    return metrics.summary(duration)


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM Protocol (Sequential)")
    parser.add_argument("--protocol", choices=["rest", "grpc"], required=True, help="Protocol to benchmark")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, help="Server port (Default: 8000 for REST, 50051 for gRPC)")
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 4, 8], help="List of concurrency levels")
    args = parser.parse_args()

    port = args.port or (8000 if args.protocol == "rest" else 50051)

    print("==================================================")
    print(f"  vLLM Single-Protocol Benchmark")
    print(f"  Protocol: {args.protocol.upper()}")
    print(f"  Endpoint: {args.host}:{port}")
    print(f"  Model:    {MODEL_NAME}")
    print("==================================================\n")

    for concurrency in args.concurrency_levels:
        num_requests = concurrency * 10
        print(f"Testing Concurrency Level = {concurrency} ({num_requests} total requests)...")
        
        if args.protocol == "rest":
            results = asyncio.run(run_rest_suite(args.host, port, concurrency, num_requests))
        else:
            results = asyncio.run(run_grpc_suite(args.host, port, concurrency, num_requests))
            
        table_data = [[metric, f"{value:.2f}"] for metric, value in results.items()]
        print(tabulate(table_data, headers=["Metric", f"Value ({args.protocol.upper()})"], tablefmt="grid"))
        print("\n")


if __name__ == "__main__":
    main()