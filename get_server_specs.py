"""
Collect server/hardware specification -- CPU, RAM, OS, and especially GPU
details (model, VRAM, driver, CUDA, compute capability, current
utilization/power/clocks) -- so that a rest_results.json / grpc_results.json
run can always be traced back to the exact hardware it was produced on.

No non-stdlib dependencies are required. If `nvidia-smi` isn't on PATH (no
NVIDIA GPU, or driver not installed), the script still runs and simply
reports gpu.available = False. If `torch` happens to be importable in the
current environment (it will be, alongside vllm), a few extra fields are
added, but it's optional.

Usage:
    python get_server_specs.py
    python get_server_specs.py --output results/server_specs.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

try:
    from common import save_json
except ImportError:  # allow running this file standalone, outside the repo
    def save_json(obj, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"Saved -> {path}")


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# CPU / RAM / OS
# ---------------------------------------------------------------------------

def get_cpu_info() -> dict:
    info = {
        "model": None,
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
    }

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                text = f.read()
            match = re.search(r"model name\s*:\s*(.+)", text)
            if match:
                info["model"] = match.group(1).strip()
            physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", text))
            core_ids = set(re.findall(r"core id\s*:\s*(\d+)", text))
            if physical_ids and core_ids:
                info["physical_cores"] = len(physical_ids) * len(core_ids)
        except OSError:
            pass
    elif platform.system() == "Darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        phys = _run(["sysctl", "-n", "hw.physicalcpu"])
        info["physical_cores"] = int(phys) if phys and phys.isdigit() else None
    elif platform.system() == "Windows":
        info["model"] = platform.processor()

    return info


def get_memory_info() -> dict:
    info = {"total_gb": None}
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                text = f.read()
            match = re.search(r"MemTotal:\s*(\d+)\s*kB", text)
            if match:
                info["total_gb"] = round(int(match.group(1)) / (1024 ** 2), 2)
        except OSError:
            pass
    elif platform.system() == "Darwin":
        total = _run(["sysctl", "-n", "hw.memsize"])
        if total and total.isdigit():
            info["total_gb"] = round(int(total) / (1024 ** 3), 2)
    return info


def get_os_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# GPU (the main event)
# ---------------------------------------------------------------------------

# Fields we'd *like* from nvidia-smi, roughly newest/most-detailed first.
# Older driver versions may not support every field, so we retry with
# progressively smaller field sets rather than failing outright.
_QUERY_FIELD_TIERS = [
    [
        "index", "name", "uuid", "driver_version",
        "memory.total", "memory.used", "memory.free",
        "utilization.gpu", "utilization.memory",
        "temperature.gpu", "power.draw", "power.limit",
        "clocks.sm", "clocks.max.sm",
        "pcie.link.gen.current", "pcie.link.width.current",
        "compute_cap",
    ],
    [
        "index", "name", "uuid", "driver_version",
        "memory.total", "memory.used", "memory.free",
        "utilization.gpu", "utilization.memory",
        "temperature.gpu", "power.draw", "power.limit",
    ],
    ["index", "name", "driver_version", "memory.total"],
]

_NUMERIC_FIELDS = {
    "memory.total", "memory.used", "memory.free",
    "utilization.gpu", "utilization.memory",
    "temperature.gpu", "power.draw", "power.limit",
    "clocks.sm", "clocks.max.sm",
    "pcie.link.gen.current", "pcie.link.width.current",
}


def _parse_nvidia_smi_query(fields: list[str]) -> list[dict] | None:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    out = _run(cmd)
    if out is None:
        return None

    gpus = []
    for line in out.splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(fields):
            continue  # malformed/unsupported field set, let caller retry smaller tier
        row = dict(zip(fields, values))
        for f in fields:
            if f in _NUMERIC_FIELDS:
                try:
                    row[f] = float(row[f]) if "." in row[f] else int(row[f])
                except ValueError:
                    row[f] = None
        gpus.append(row)
    return gpus if gpus else None


def _get_cuda_driver_version() -> str | None:
    """Max CUDA version supported by the installed driver (from the
    plain `nvidia-smi` banner, not available via --query-gpu)."""
    out = _run(["nvidia-smi"])
    if not out:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return match.group(1) if match else None


def _get_nvcc_version() -> str | None:
    """CUDA *toolkit* version (nvcc), which can differ from the driver's
    max-supported CUDA version above -- useful when debugging build/runtime
    mismatches."""
    out = _run(["nvcc", "--version"])
    if not out:
        return None
    match = re.search(r"release ([\d.]+)", out)
    return match.group(1) if match else None


def _get_torch_gpu_info() -> dict | None:
    """Optional extra corroboration if torch happens to be importable in
    the current environment (it will be, next to vllm)."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None

    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
    return info


def get_gpu_info() -> dict:
    if _run(["which", "nvidia-smi"]) is None and _run(["nvidia-smi", "-L"]) is None:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}

    gpus = None
    for fields in _QUERY_FIELD_TIERS:
        gpus = _parse_nvidia_smi_query(fields)
        if gpus is not None:
            break

    if gpus is None:
        return {"available": False, "reason": "nvidia-smi found but query failed"}

    result = {
        "available": True,
        "count": len(gpus),
        "cuda_driver_max_version": _get_cuda_driver_version(),
        "nvcc_toolkit_version": _get_nvcc_version(),
        "gpus": gpus,
    }

    torch_info = _get_torch_gpu_info()
    if torch_info is not None:
        result["torch"] = torch_info

    return result


# ---------------------------------------------------------------------------
# Library versions relevant to the benchmark (vllm, grpc, aiohttp)
# ---------------------------------------------------------------------------

def get_package_versions() -> dict:
    versions = {}
    for pkg in ("vllm", "torch", "aiohttp", "grpc", "smg_grpc_proto"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return versions


def collect_server_specs() -> dict:
    return {
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "packages": get_package_versions(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="results/server_specs.json")
    args = ap.parse_args()

    specs = collect_server_specs()
    print(json.dumps(specs, indent=2))
    save_json(specs, args.output)


if __name__ == "__main__":
    main()
