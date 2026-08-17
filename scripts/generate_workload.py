#!/usr/bin/env python3
"""Generate a deterministic RFDYNW1 insert or delete workload."""

from __future__ import annotations

import argparse
import hashlib
import random
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=("insert", "delete"), required=True)
    parser.add_argument("--count", type=int, required=True, help="Full external-ID space")
    parser.add_argument("--dimension", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fraction", type=float)
    selection.add_argument("--operation-count", type=int)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 1 or args.dimension <= 0:
        raise SystemExit("--count must exceed one and --dimension must be positive")
    if args.fraction is not None:
        if not 0.0 < args.fraction < 1.0:
            raise SystemExit("--fraction must be strictly between zero and one")
        operation_count = round(args.count * args.fraction)
    else:
        operation_count = args.operation_count
    if operation_count is None or not 0 < operation_count < args.count:
        raise SystemExit("operation count must be between one and count - 1")
    if not 0 <= args.seed < 2**64:
        raise SystemExit("--seed must fit uint64")

    ids = random.Random(args.seed).sample(range(args.count), operation_count)

    operation = 1 if args.operation == "insert" else 2
    if operation == 1:
        initial_active, final_active = args.count - operation_count, args.count
    else:
        initial_active, final_active = args.count, args.count - operation_count

    header = struct.pack(
        "<8s7IQ",
        b"RFDYNW1\0",
        1,
        operation,
        args.count,
        initial_active,
        final_active,
        args.dimension,
        operation_count,
        args.seed,
    )
    payload = struct.pack(f"<{operation_count}I", *ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + payload)
    digest = hashlib.sha256(header + payload).hexdigest()
    print(f"output:{args.output.resolve()}")
    print(f"operation:{args.operation}")
    print(f"operation_count:{operation_count}")
    print(f"initial_active:{initial_active}")
    print(f"final_active:{final_active}")
    print(f"seed:{args.seed}")
    print(f"sha256:{digest}")


if __name__ == "__main__":
    main()
