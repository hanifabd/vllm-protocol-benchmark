"""
Combine results/rest_results.json and results/grpc_results.json into a
side-by-side comparison table (and optionally a chart).

Usage:
    python compare_results.py \
        --rest results/rest_results.json \
        --grpc results/grpc_results.json \
        --csv results/comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def flatten(summary: dict) -> dict:
    row = {
        "protocol": summary["protocol"],
        "concurrency": summary["concurrency"],
        "success": summary["successful_requests"],
        "failed": summary["failed_requests"],
        "req_throughput_s": summary["request_throughput_req_s"],
        "output_tok_throughput_s": summary["output_token_throughput_tok_s"],
        "ttft_mean_ms": summary["ttft_ms"]["mean"],
        "ttft_p99_ms": summary["ttft_ms"]["p99"],
        "tpot_mean_ms": summary["tpot_ms"]["mean"],
        "tpot_p99_ms": summary["tpot_ms"]["p99"],
        "itl_mean_ms": summary["itl_ms"]["mean"],
        "e2e_mean_ms": summary["e2e_latency_ms"]["mean"],
        "e2e_p99_ms": summary["e2e_latency_ms"]["p99"],
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest", default="results/rest_results.json")
    ap.add_argument("--grpc", default="results/grpc_results.json")
    ap.add_argument("--csv", default="results/comparison.csv")
    args = ap.parse_args()

    rest_summaries = load(args.rest)
    grpc_summaries = load(args.grpc)

    rows = [flatten(s) for s in rest_summaries] + [flatten(s) for s in grpc_summaries]
    rows.sort(key=lambda r: (r["concurrency"], r["protocol"]))

    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison table -> {out_path}\n")

    # Pretty print to console too
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(col_widths[h]) for h in headers))


if __name__ == "__main__":
    main()
