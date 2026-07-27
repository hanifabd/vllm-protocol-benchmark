"""
vLLM's gRPC server (added behind `vllm serve --grpc` in recent releases) is
backed by proto stubs published separately as `smg-grpc-proto`, which is
still evolving and not fully documented inline. Rather than hardcode RPC /
message field names that might be wrong for your installed version, run
this script FIRST -- it imports whatever proto stubs you have installed and
prints:

  - every gRPC service defined
  - every RPC method on each service (+ whether it's streaming)
  - every field on the request/response message types

Use that printout to fill in the TODOs in bench_grpc.py (there are only
~3 small spots to fill in: the request message construction, the
"first token" detection condition, and how tokens are pulled off each
streamed response chunk).

Usage:
    pip install smg-grpc-proto grpcio grpcio-tools
    python discover_grpc_schema.py
"""
from __future__ import annotations

import sys


def describe_message(descriptor, indent="    "):
    for field in descriptor.fields:
        label = {1: "optional", 2: "required", 3: "repeated"}.get(field.label, "?")
        print(f"{indent}- {field.name} : {field.type_name or field.cpp_type_name if hasattr(field,'cpp_type_name') else field.type} ({label})")


def main():
    try:
        from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
    except ImportError as e:
        print("Could not import smg_grpc_proto.vllm_engine_pb2[.grpc].")
        print("Install with:  pip install smg-grpc-proto grpcio grpcio-tools")
        print(f"Original error: {e}")
        sys.exit(1)

    file_descriptor = vllm_engine_pb2.DESCRIPTOR

    print("=" * 70)
    print(f"Proto file: {file_descriptor.name}")
    print(f"Package:    {file_descriptor.package}")
    print("=" * 70)

    print("\nMessage types found:")
    for msg_name, msg_desc in file_descriptor.message_types_by_name.items():
        print(f"\n  message {msg_name} {{")
        describe_message(msg_desc)
        print("  }")

    print("\n" + "=" * 70)
    print("Services / RPC methods:")
    print("=" * 70)
    for svc_name, svc_desc in file_descriptor.services_by_name.items():
        print(f"\nservice {svc_name} {{")
        for method in svc_desc.methods:
            streaming_in = "stream " if method.client_streaming else ""
            streaming_out = "stream " if method.server_streaming else ""
            print(
                f"    rpc {method.name}({streaming_in}{method.input_type.name}) "
                f"returns ({streaming_out}{method.output_type.name})"
            )
        print("}")

    print("\n" + "=" * 70)
    print("Client stub class available as:")
    print(f"    vllm_engine_pb2_grpc.<ServiceName>Stub(channel)")
    print("Attributes on the grpc module:", [a for a in dir(vllm_engine_pb2_grpc) if not a.startswith("_")])
    print("=" * 70)
    print("\nNext step: open bench_grpc.py and fill in the 3 TODO blocks using")
    print("the exact message/field/method names printed above.")


if __name__ == "__main__":
    main()
