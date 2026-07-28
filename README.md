### Executive Summary: vLLM Serve Protocol Comparison (gRPC vs. REST API)

This evaluation compares **gRPC** and **REST API** performance for `vllm serve` across workloads ranging from **1 to 50** concurrent requests (100 total requests per test, 100% success rate).

---

### Key Takeaways & Performance Insights

1. **Protocols Perform Identically at Low Load ($C=1$):**
* At single-request concurrency, performance differences are negligible ($0.83\text{ ms}$ lower end-to-end latency for gRPC and $<0.02\text{ req/s}$ difference in throughput). Either protocol works equally well for light workloads.


2. **gRPC Handles High Concurrency Significantly Better ($C=50$):**
* **Faster First Token (TTFT):** At 50 concurrent requests, gRPC starts returning tokens **$41.06\text{ ms}$ faster on average** (a $43.59\%$ latency reduction from $94.19\text{ ms}$ down to $53.13\text{ ms}$).
* **Faster Completion (End-to-End Latency):** Full request completion is **$51.14\text{ ms}$ faster on average** under gRPC ($338.27\text{ ms}$ vs. $389.41\text{ ms}$, representing a $13.13\%$ speedup).
* **Higher Capacity (Throughput):** gRPC processes **$19.96$ more requests per second** ($147.06\text{ req/s}$ vs. $127.10\text{ req/s}$, a $15.70\%$ improvement) and generates **$3,646$ more tokens per second**.


3. **Token Generation Speed Remains Unchanged:**
* Time-per-output-token (TPOT) is virtually identical across both protocols ($\sim 2.25\text{ ms}$ for gRPC vs. $\sim 2.49\text{ ms}$ for REST at $C=50$). This confirms that gRPC’s performance advantage comes from lower network and connection overhead, not faster GPU generation.



---

### Test Environment & Hardware Specifications

The benchmark was executed using the following hardware and software setup:

* **LLM Model:** Qwen/Qwen2.5-0.5B-Instruct

* **GPU:** 1x NVIDIA GeForce RTX 4090 (24 GB VRAM)


* **CPU:** AMD EPYC 7352 24-Core Processor (48 threads)


* **System Memory:** 251.55 GB RAM


* **OS & CUDA:** Linux (Ubuntu 22.04 LTS), NVIDIA Driver 580.159.04, CUDA 13.0


* **Software Stack:** vLLM `0.26.0`, PyTorch `2.11.0+cu130`, gRPC `1.83.0`, Python `3.14.6`


---

### Complete Metric Breakdown

| Concurrency | Protocol | Throughput (req/s) | Throughput Gain | Mean First Token (TTFT) *(Lower is Better)* | First Token Advantage | Mean Full Response (E2E) *(Lower is Better)* | Total Time Saved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | **gRPC** | **4.28** | +0.02 req/s *(+0.36%)* | **12.22 ms** | **0.35 ms faster** *(2.78%)* | **233.61 ms** | **0.83 ms faster** *(0.36%)* |
|  | REST | 4.26 |  | 12.57 ms |  | 234.44 ms |  |
| **5** | **gRPC** | **18.53** | +0.66 req/s *(+3.67%)* | **19.41 ms** | **8.17 ms faster** *(29.62%)* | **269.59 ms** | **9.79 ms faster** *(3.50%)* |
|  | REST | 17.87 |  | 27.58 ms |  | 279.38 ms |  |
| **10** | **gRPC** | **35.83** | +1.28 req/s *(+3.71%)* | **25.88 ms** | **8.79 ms faster** *(25.34%)* | **278.61 ms** | **10.21 ms faster** *(3.53%)* |
|  | REST | 34.55 |  | 34.67 ms |  | 288.82 ms |  |
| **25** | **gRPC** | **82.62** | +7.53 req/s *(+10.03%)* | **35.02 ms** | **24.33 ms faster** *(41.00%)* | **301.48 ms** | **29.46 ms faster** *(8.90%)* |
|  | REST | 75.09 |  | 59.34 ms |  | 330.93 ms |  |
| **50** | **gRPC** | **147.06** | +19.96 req/s *(+15.70%)* | **53.13 ms** | **41.06 ms faster** *(43.59%)* | **338.27 ms** | **51.14 ms faster** *(13.13%)* |
|  | REST | 127.10 |  | 94.19 ms |  | 389.41 ms |  |

---

### Operational Recommendation

* **Use gRPC for Production & Scaled Deployments:** Ideal for streaming services or multi-user applications where low initial response time (TTFT) and high throughput are essential under heavy concurrent traffic.
* **Use REST for Rapid Prototyping & Simple Integrations:** Best suited for low-concurrency internal testing ($C \le 5$) where standard HTTP/JSON tooling provides developer convenience with no noticeable speed penalty.