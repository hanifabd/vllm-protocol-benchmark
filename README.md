### Executive Summary: vLLM REST API vs. gRPC Benchmark Comparison

This benchmark evaluates the performance of **vLLM** using **gRPC** versus **REST API** protocols across increasing concurrency levels ($1, 5, 10, 25, 50$). Each test run executed $100$ total requests with a $100\%$ success rate.

---

## Server & Environment Specifications

### Hardware
* **CPU:** AMD Ryzen 9 7950X (16 Physical Cores / 32 Threads)[cite: 1]
* **GPU:** 1x NVIDIA GeForce RTX 4090 (24 GB VRAM)[cite: 1]
* **System Memory:** 124.91 GB RAM[cite: 1]

### System & Driver Info
* **Operating System:** Linux Ubuntu 22.04 LTS (Kernel 6.8.0-107-generic)[cite: 1]
* **NVIDIA Driver:** 580.126.20[cite: 1]
* **CUDA / Toolkit:** CUDA 13.0 (NVCC Toolkit 12.8)[cite: 1]
* **Python Version:** 3.14.6[cite: 1]

### Core Dependencies & Software
* **vLLM:** 0.26.0[cite: 1]
* **PyTorch:** 2.11.0+cu130[cite: 1]
* **gRPC:** 1.83.0[cite: 1]
* **aiohttp:** 3.14.3[cite: 1]
* **smg_grpc_proto:** 0.4.14[cite: 1]

## LLM Models
* **Models:** Qwen/Qwen2.5-0.5B-Instruct

---

### Key Takeaways

1. **Overall Request Throughput & Latency are Comparable:**
Both REST and gRPC perform very similarly across all tested concurrency levels. At the highest concurrency ($50$), gRPC achieves **$165.10\text{ req/s}$** compared to REST's **$160.51\text{ req/s}$** ($\sim 2.8\%$ throughput advantage for gRPC).
2. **High-Concurrency Latency Edge for gRPC:**
At higher concurrency levels ($25$ and $50$), gRPC demonstrates improved **Time to First Token (TTFT)** and lower overall **End-to-End (E2E) latency**:
* **TTFT Mean at Concurrency 50:** gRPC is **$13.4\%$ faster** ($27.83\text{ ms}$ vs. $32.13\text{ ms}$).
* **TTFT P99 at Concurrency 50:** gRPC is **$14.5\%$ faster** ($36.30\text{ ms}$ vs. $42.45\text{ ms}$).
* **E2E Mean Latency at Concurrency 50:** gRPC is **$2.9\%$ faster** ($300.54\text{ ms}$ vs. $309.42\text{ ms}$).


3. **Inter-Token Latency (ITL):**
Inter-Token Latency is nearly identical across protocols, scaling smoothly from $\sim 1.72\text{ ms} - 1.73\text{ ms}$ at concurrency 1 to $\sim 2.14\text{ ms} - 2.18\text{ ms}$ at concurrency 50, with gRPC holding a negligible $\sim 1-2\%$ efficiency edge.
4. **Data Logging Anomalies in gRPC Run:**
* **Missing TPOT:** Time Per Output Token (`tpot_mean_ms`, `tpot_p99_ms`) was not recorded (`NaN`) for gRPC.
* **Token Throughput Discrepancy:** The `output_tok_throughput_s` column for gRPC mirrors its request throughput ($1:1$ ratio), whereas REST correctly reports token throughput scaling up to **$20,543.99\text{ tok/s}$** at concurrency 50. This indicates a metric collection artifact in the gRPC benchmarker rather than a hardware limitation.



---

### Performance Comparison Table

| Concurrency | Protocol | Req Throughput (req/s) | Output Tok Throughput (tok/s) | TTFT Mean (ms) | TTFT P99 (ms) | ITL Mean (ms) | E2E Mean (ms) | E2E P99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | gRPC | **4.47** | 4.47* | **4.71** | **5.69** | **1.72** | **223.79** | **225.32** |
|  | REST | 4.44 | 567.97 | 5.15 | 6.81 | 1.73 | 225.26 | 226.71 |
| **5** | gRPC | **19.72** | 19.72* | 8.03 | 13.85 | **1.93** | **253.52** | 259.71 |
|  | REST | 19.66 | 2,515.73 | **6.98** | **10.05** | 1.95 | 254.09 | **257.28** |
| **10** | gRPC | 38.69 | 38.69* | 11.45 | 17.84 | **1.94** | 258.35 | 265.75 |
|  | REST | **38.80** | 4,966.18 | **8.73** | **13.34** | 1.96 | **257.28** | **262.19** |
| **25** | gRPC | **90.13** | 90.13* | **15.97** | **22.77** | **2.05** | **276.68** | **283.91** |
|  | REST | 88.99 | 11,389.01 | 17.39 | 23.82 | 2.07 | 279.84 | 285.27 |
| **50** | gRPC | **165.10** | 165.10* | **27.83** | **36.30** | **2.14** | **300.54** | **310.97** |
|  | REST | 160.51 | 20,543.99 | 32.13 | 42.45 | 2.18 | 309.42 | 313.98 |

**Note: gRPC token throughput values reflect a benchmarking recording issue.*

---

### Recommendation

* **For High-Scale/High-Concurrency Workloads:** Choose **gRPC**. It offers superior network efficiency, lower TTFT overhead ($13-14\%$ faster under load), and higher request throughput capacity.
* **For Simplicity & Integrations:** Choose **REST API**. The performance difference at low-to-medium concurrency ($1-10$) is negligible, and REST provides standard compatibility with standard client ecosystems.