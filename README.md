# vLLM REST vs. gRPC Benchmark Suite (GPU)

This repository provides an automated benchmark harness to compare the throughput, latency, and streaming performance of **REST (HTTP/1.1 + JSON)** versus **gRPC (HTTP/2 + Protobuf)** endpoints on [vLLM](https://github.com/vllm-project/vllm).

The benchmark target model used is **`Qwen/Qwen2.5-0.5B-Instruct`** executing on **CUDA / GPU**.

---

## 📋 Table of Contents
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Benchmarking Workflow](#-benchmarking-workflow)
  - [1. Benchmark REST Protocol](#1-benchmark-rest-protocol)
  - [2. Benchmark gRPC Protocol](#2-benchmark-grpc-protocol)
- [Metrics Measured](#-metrics-measured)
- [Command Line Arguments](#-command-line-arguments)

---

## ⚙️ Prerequisites
- Python `3.10` or higher
- NVIDIA GPU with Compute Capability 7.0+ (e.g., T4, RTX 3080/4090, A100, H100)
- NVIDIA Driver installed with CUDA `12.x` support (`nvidia-smi` accessible)
- System VRAM: Minimum 4 GB free VRAM

---

## 📦 Installation

1. **Clone or navigate to the project directory:**
```bash
   git clone 
   cd vllm-protocol-benchmark
```

2. **Create and activate an isolated Python virtual environment:**
```bash
python3 -m venv vllm-env
source vllm-env/bin/activate

```


3. **Install Dependencies:**
Install vLLM with standard CUDA acceleration along with client benchmarking dependencies:
```bash
# Upgrade core build tools
pip install --upgrade pip setuptools wheel

# Install from requirements.txt
pip install -r requirements.txt

```



---

## 📂 Project Structure

```text
vllm-benchmark/
├── benchmark.py    # Async benchmarking harness
├── requirements.txt       # Project dependencies
└── README.md              # Documentation and execution guide

```

---

## 🚀 Benchmarking Workflow

To ensure clean isolates and prevent VRAM allocation conflicts, **run servers sequentially** (one protocol at a time).

---

### 1. Benchmark REST Protocol

#### **Terminal 1: Start the REST Server (GPU)**

```bash
source vllm-env/bin/activate

# Launch OpenAI-compatible API server on GPU (Port 8000)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --device cuda \
    --gpu-memory-utilization 0.80

```

*Wait until you see: `Application startup complete.*`

#### **Terminal 2: Run the Benchmark Client**

```bash
source vllm-env/bin/activate

python benchmark.py --protocol rest --port 8000

```

#### **Terminal 1: Stop Server**

Press `Ctrl + C` in Terminal 1 to terminate the process and completely clear GPU VRAM before testing gRPC.

---

### 2. Benchmark gRPC Protocol

#### **Terminal 1: Start the gRPC Server (GPU)**

```bash
source vllm-env/bin/activate

# Launch gRPC server on GPU (Port 50051)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.grpc_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 50051 \
    --device cuda \
    --gpu-memory-utilization 0.80

```

*Wait until you see: `vLLM gRPC server started on 127.0.0.1:50051`.*

#### **Terminal 2: Run the Benchmark Client**

```bash
source vllm-env/bin/activate

python benchmark.py --protocol grpc --port 50051

```

#### **Terminal 1: Stop Server**

Press `Ctrl + C` in Terminal 1.

---

## 📊 Metrics Measured

The test harness evaluates performance under multiple concurrency tiers (`1`, `4`, `8` concurrent streams):

| Metric | Description |
| --- | --- |
| **RPS** | Requests Per Second completed by the endpoint. |
| **Total Tokens/s** | Total output token generation velocity across all concurrent streams. |
| **Mean TTFT (ms)** | Time to First Token. Measures transport + initial GPU prompt evaluation latency. |
| **P95 TTFT (ms)** | 95th percentile tail latency for initial token delivery. |
| **Mean TPOT (ms)** | Time Per Output Token. Average inter-token latency during streaming. |
| **P95 TPOT (ms)** | 95th percentile inter-token latency (measures streaming jitter). |
| **Mean Latency (s)** | Total end-to-end request duration. |

---

## 🛠️ Command Line Arguments

You can customize the benchmark parameters using flags on `benchmark_single.py`:

```bash
python benchmark_single.py [FLAGS]

```

| Flag | Default | Description |
| --- | --- | --- |
| `--protocol` | *(Required)* | Protocol to test (`rest` or `grpc`). |
| `--host` | `127.0.0.1` | Endpoint host IP address. |
| `--port` | `8000` (REST) / `50051` (gRPC) | Endpoint port. |
| `--concurrency-levels` | `1 4 8` | Space-separated concurrency tiers to execute. |

### Example with Higher GPU Concurrency:

```bash
python benchmark_single.py --protocol grpc --port 50051 --concurrency-levels 1 4 16 32 64

```