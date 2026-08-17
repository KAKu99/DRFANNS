#!/usr/bin/env python3
"""Run a small, dependency-free RFANNS build/search/update experiment."""

from __future__ import annotations

import argparse
import csv
import math
import struct
import subprocess
import sys
from pathlib import Path


POINTS = 192
DIMENSION = 8
QUERIES = 24
K = 10
UPDATE_COUNT = 16
SEED = 12345


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--work-dir", type=Path, default=Path("smoke-output"))
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_vectors(path: Path, rows: list[list[float]]) -> None:
    with path.open("wb") as output:
        output.write(struct.pack("<ii", len(rows), len(rows[0])))
        for row in rows:
            output.write(struct.pack(f"<{len(row)}f", *row))


def create_inputs(directory: Path) -> tuple[Path, Path, Path, Path]:
    base = []
    for point in range(POINTS):
        base.append(
            [
                math.sin((point + 1) * (component + 2) * 0.071)
                + math.cos((point + 3) * (component + 1) * 0.037)
                + point * 0.0007
                for component in range(DIMENSION)
            ]
        )
    queries = []
    for query in range(QUERIES):
        source = base[(query * 17 + 5) % POINTS]
        queries.append([value + (query + 1) * 0.0001 for value in source])

    data_path = directory / "base.bin"
    query_path = directory / "query.bin"
    attr_path = directory / "attributes.bin"
    range_path = directory / "ranges.bin"
    write_vectors(data_path, base)
    write_vectors(query_path, queries)
    attributes = [(point * 37 + 11) % 256 for point in range(POINTS)]
    attr_path.write_bytes(struct.pack(f"<{POINTS}i", *attributes))
    with range_path.open("wb") as output:
        for query in range(QUERIES):
            low, high = ((0, 255) if query % 2 == 0 else (48, 207))
            output.write(struct.pack("<ii", low, high))
    return data_path, query_path, attr_path, range_path


def require_executables(build_dir: Path) -> dict[str, Path]:
    names = (
        "rfanns_build",
        "generate_filter_groundtruth",
        "rfanns_search",
        "rfanns_update",
        "rfanns_validate",
    )
    result = {name: (build_dir / name).resolve() for name in names}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing executables; build the project first: " + ", ".join(missing))
    return result


def assert_csv_rows(path: Path, expected: int) -> None:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != expected:
        raise RuntimeError(f"{path} contains {len(rows)} rows, expected {expected}")


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    root = Path(__file__).resolve().parent.parent
    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    work_dir = args.work_dir if args.work_dir.is_absolute() else root / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    executables = require_executables(build_dir)
    data, query, attributes, ranges = create_inputs(work_dir)

    workload_script = root / "scripts" / "generate_workload.py"
    insert_workload = work_dir / "insert.bin"
    delete_workload = work_dir / "delete.bin"
    for operation, output in (("insert", insert_workload), ("delete", delete_workload)):
        run(
            [
                sys.executable,
                str(workload_script),
                "--output",
                str(output),
                "--operation",
                operation,
                "--count",
                str(POINTS),
                "--dimension",
                str(DIMENSION),
                "--operation-count",
                str(UPDATE_COUNT),
                "--seed",
                str(SEED),
            ]
        )

    full_index = work_dir / "full.index"
    full_idmap = work_dir / "full.idmap"
    initial_index = work_dir / "insert_initial.index"
    initial_idmap = work_dir / "insert_initial.idmap"

    def build(index: Path, idmap: Path, workload: Path | None = None) -> None:
        command = [
            str(executables["rfanns_build"]),
            "--data_path",
            str(data),
            "--attr_path",
            str(attributes),
            "--index_file",
            str(index),
            "--id_map_file",
            str(idmap),
            "--attr_min",
            "0",
            "--attr_max",
            "255",
            "--num_bins",
            "16",
            "--M",
            "8",
            "--ef_construction",
            "32",
            "--threads",
            str(args.threads),
            "--seed",
            str(SEED),
            "--validate",
            "1",
        ]
        if workload is not None:
            command.extend(("--active_workload", str(workload)))
        run(command)

    build(full_index, full_idmap)
    build(initial_index, initial_idmap, insert_workload)

    full_gt = work_dir / "groundtruth_full.bin"
    delete_gt = work_dir / "groundtruth_post_delete.bin"

    def groundtruth(output: Path, workload: Path | None = None) -> None:
        command = [
            str(executables["generate_filter_groundtruth"]),
            "--data_path",
            str(data),
            "--query_path",
            str(query),
            "--range_path",
            str(ranges),
            "--attr_path",
            str(attributes),
            "--groundtruth_file",
            str(output),
            "--K",
            str(K),
            "--Nq",
            str(QUERIES),
            "--threads",
            str(args.threads),
        ]
        if workload is not None:
            command.extend(("--workload_file", str(workload)))
        run(command)

    groundtruth(full_gt)
    groundtruth(delete_gt, delete_workload)

    static_csv = work_dir / "static_search.csv"
    run(
        [
            str(executables["rfanns_search"]),
            "--data_path",
            str(data),
            "--query_path",
            str(query),
            "--range_path",
            str(ranges),
            "--groundtruth_path",
            str(full_gt),
            "--index_file",
            str(full_index),
            "--id_map_file",
            str(full_idmap),
            "--result_file",
            str(static_csv),
            "--K",
            str(K),
            "--Nq",
            str(QUERIES),
            "--ef",
            "10,20,40",
            "--seed",
            str(SEED),
        ]
    )

    post_insert_index = work_dir / "post_insert.index"
    post_insert_idmap = work_dir / "post_insert.idmap"

    def update(
        operation: str,
        index: Path,
        idmap: Path,
        workload: Path,
        groundtruth_path: Path,
        save: bool,
    ) -> None:
        manifest = work_dir / f"{operation}_evaluation.tsv"
        manifest.write_text(
            f"tag\trange_path\tgroundtruth_path\nsmoke\t{ranges}\t{groundtruth_path}\n",
            encoding="utf-8",
        )
        command = [
            str(executables["rfanns_update"]),
            "--data_path",
            str(data),
            "--attr_path",
            str(attributes),
            "--query_path",
            str(query),
            "--index_file",
            str(index),
            "--id_map_file",
            str(idmap),
            "--workload_file",
            str(workload),
            "--evaluation_manifest",
            str(manifest),
            "--result_csv",
            str(work_dir / f"{operation}_search.csv"),
            "--summary_csv",
            str(work_dir / f"{operation}_summary.csv"),
            "--per_operation_csv",
            str(work_dir / f"{operation}_latency.csv"),
            "--dynamic_attr_expand",
            "0",
            "--K",
            str(K),
            "--Nq",
            str(QUERIES),
            "--ef",
            "10,20,40",
            "--seed",
            str(SEED),
        ]
        if save:
            command.extend(
                (
                    "--save_index_file",
                    str(post_insert_index),
                    "--save_id_map_file",
                    str(post_insert_idmap),
                )
            )
        run(command)
        assert_csv_rows(work_dir / f"{operation}_search.csv", 3)
        assert_csv_rows(work_dir / f"{operation}_latency.csv", UPDATE_COUNT)

    update("insert", initial_index, initial_idmap, insert_workload, full_gt, True)
    update("delete", full_index, full_idmap, delete_workload, delete_gt, False)

    for index, idmap in ((full_index, full_idmap), (post_insert_index, post_insert_idmap)):
        run(
            [
                str(executables["rfanns_validate"]),
                "--data_path",
                str(data),
                "--index_file",
                str(index),
                "--id_map_file",
                str(idmap),
                "--seed",
                str(SEED),
            ]
        )
    assert_csv_rows(static_csv, 3)
    print(f"smoke_status:passed\noutput_directory:{work_dir.resolve()}")


if __name__ == "__main__":
    main()
