### Executive Summary: vLLM Serve Protocol Comparison (gRPC vs. REST API)

This analysis evaluates performance data comparing **gRPC** and **REST API** protocols for `vllm serve` across concurrency levels ranging from **1 to 50** concurrent requests (with 100 successful requests per test run and 0 failures).

---

### Key Findings

1. **gRPC Outperforms REST Significantly under High Concurrency:**
* **Throughput:** At low concurrency ($C=1$), both protocols perform virtually identically (**$0.36\%$** gain for gRPC). However, as concurrency scales to $50$, **gRPC achieves $15.70\%$ higher request throughput** ($147.06$ req/s vs. $127.10$ req/s) and **$24.02\%$ higher token throughput** ($18,823.32$ vs. $15,177.15$ output tokens/s).
* **Time-to-First-Token (TTFT):** gRPC significantly reduces response initiation latency as concurrent load grows. At $C=50$, gRPC delivers a **$43.59\%$ faster mean TTFT** ($53.13\text{ ms}$ vs. $94.19\text{ ms}$) and a **$37.58\%$ faster P99 TTFT** ($79.63\text{ ms}$ vs. $127.57\text{ ms}$).
* **End-to-End (E2E) Latency:** At maximum load ($C=50$), gRPC delivers a **$13.13\%$ faster mean total response time** ($338.27\text{ ms}$ vs. $389.41\text{ ms}$) and a **$15.61\%$ faster P99 E2E time** ($354.74\text{ ms}$ vs. $420.34\text{ ms}$).


2. **Per-Token Generation Latency (TPOT & ITL) remains Stable:**
* Both protocols exhibit nearly identical Time-Per-Output-Token (TPOT) and Inter-Token Latency (ITL) values across all concurrency levels (e.g., $\sim 2.25\text{ ms}$ for gRPC vs. $\sim 2.49\text{ ms}$ for REST at $C=50$). The throughput advantage of gRPC stems primarily from lower connection handling overhead during initial request establishment rather than token generation speed itself.



---

### Test Environment & Server Specifications

The benchmark execution was conducted under the following hardware and environment setup:

* **CPU:** AMD EPYC 7352 24-Core Processor (24 physical cores, 48 logical threads).


* **GPU:** 1x NVIDIA GeForce RTX 4090 (24 GB VRAM, Compute Capability 8.9).


* **System Memory:** 251.55 GB RAM.


* **Operating System:** Linux (Ubuntu 22.04 LTS, Kernel 6.8.0-117-generic).


* **Driver & CUDA:** NVIDIA Driver 580.159.04, CUDA Driver max 13.0, NVCC Toolkit 12.8.


* **Software Stack:**
* **vLLM Version:** 0.26.0


* **PyTorch Version:** 2.11.0+cu130


* **gRPC Library Version:** 1.83.0


* **Python Version:** 3.14.6





---

### Key Performance Comparison Table (with gRPC Improvements)

| Concurrency | Protocol | Req Throughput (req/s) | Output Tok Throughput (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean E2E Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **gRPC** <br>

<br> **REST** <br>

<br> *(gRPC Gain)* | **4.28** <br>

<br> 4.26 <br>

<br> *(+0.36%)* | **547.76** <br>

<br> 545.61 <br>

<br> *(+0.40%)* | **12.22** <br>

<br> 12.57 <br>

<br> *(2.78% faster)* | **15.40** <br>

<br> 15.31 <br>

<br> *(-0.56%)* | **233.61** <br>

<br> 234.44 <br>

<br> *(0.36% faster)* |
| **5** | **gRPC** <br>

<br> **REST** <br>

<br> *(gRPC Gain)* | **18.53** <br>

<br> 17.87 <br>

<br> *(+3.67%)* | **2,371.87** <br>

<br> 2,287.46 <br>

<br> *(+3.69%)* | **19.41** <br>

<br> 27.58 <br>

<br> *(29.62% faster)* | **25.01** <br>

<br> 35.77 <br>

<br> *(30.08% faster)* | **269.59** <br>

<br> 279.38 <br>

<br> *(3.50% faster)* |
| **10** | **gRPC** <br>

<br> **REST** <br>

<br> *(gRPC Gain)* | **35.83** <br>

<br> 34.55 <br>

<br> *(+3.71%)* | **4,586.03** <br>

<br> 4,421.81 <br>

<br> *(+3.71%)* | **25.88** <br>

<br> 34.67 <br>

<br> *(25.34% faster)* | **36.05** <br>

<br> 45.05 <br>

<br> *(19.98% faster)* | **278.61** <br>

<br> 288.82 <br>

<br> *(3.53% faster)* |
| **25** | **gRPC** <br>

<br> **REST** <br>

<br> *(gRPC Gain)* | **82.62** <br>

<br> 75.09 <br>

<br> *(+10.03%)* | **10,575.25** <br>

<br> 9,611.59 <br>

<br> *(+10.03%)* | **35.02** <br>

<br> 59.34 <br>

<br> *(41.00% faster)* | **64.48** <br>

<br> 85.16 <br>

<br> *(24.28% faster)* | **301.48** <br>

<br> 330.93 <br>

<br> *(8.90% faster)* |
| **50** | **gRPC** <br>

<br> **REST** <br>

<br> *(gRPC Gain)* | **147.06** <br>

<br> 127.10 <br>

<br> *(+15.70%)* | **18,823.32** <br>

<br> 15,177.15 <br>

<br> *(+24.02%)* | **53.13** <br>

<br> 94.19 <br>

<br> *(43.59% faster)* | **79.63** <br>

<br> 127.57 <br>

<br> *(37.58% faster)* | **338.27** <br>

<br> 389.41 <br>

<br> *(13.13% faster)* |

---

### Strategic Recommendation

* **Adopt gRPC for High-Throughput & Production Workloads:** gRPC is strongly recommended for high-concurrency production deployments where low latency (specifically TTFT, up to **$43.59\%$ faster**) and maximum inference throughput (**$15.70\%$ to $24.02\%$ higher**) are required.
* **Use REST for Internal Prototyping & Easy Integration:** REST API remains suitable for low-concurrency environments ($C \le 5$) or simple integration scenarios where HTTP/JSON standard tooling is preferred and performance differences are negligible ($\le 3.7\%$).