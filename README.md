# DRFANNS

We propose an algorithm for dynamic range-filtered approximate nearest neighbor search. 
It supports static index construction, filtered search, dynamic insertion, and physical deletion for vectors with one attribute.  
Supplementary experimental results for our WSDM 2027 paper are provided [here](https://github.com/KAKu99/DRFANNS/blob/main/WSDM2027DRFANNS_supplementary.pdf).

## Requirements

- C++17 compiler and CMake 3.16+
- OpenMP
- Python 3.10+ for helper scripts; NumPy for benchmark attribute generation

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRFANNS_NATIVE_ARCH=ON \
  -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use `RFANNS_NATIVE_ARCH=OFF` for portable binaries. Run a self-contained smoke
test with:

```bash
python3 scripts/smoke_experiment.py --build-dir build
```

## Dataset

DRFANNS has been evaluated with the following datasets. Each evaluation uses
10,000 query vectors and squared L2 distance.

| Dataset | Base vectors | Dimensions | Source |
| --- | ---: | ---: | --- |
| SIFT10M | 10,000,000 | 128 | [SIFT1B/BigANN](http://corpus-texmex.irisa.fr/) subset |
| Deep10M | 10,000,000 | 96 | [Deep1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) subset |
| SPACEV10M | 10,000,000 | 100 | [SPACEV1B](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) subset |
| YouTube-Video 1M | 1,000,000 | 1024 | [YouTube-8M](https://research.google.com/youtube8m/download.html) mean-RGB subset |

Dataset files are not included. SIFT10M, Deep10M, and SPACEV10M use the same
synthetic attribute vector: one independent value per base vector, sampled
uniformly from `[0, 100000)` with NumPy's `default_rng(12345)`:

```python
import numpy

rng = numpy.random.default_rng(12345)
attributes = rng.integers(0, 100000, size=10_000_000, dtype=numpy.int32)
attributes.tofile("attributes.bin")
```

For YouTube-Video, `publishedAt` is fetched from the YouTube Data API, parsed as UTC, and converted by `published_day = floor(Unix timestamp / 86400)`. 
The evaluated attribute values range from 12995 to 16714.

Convert vectors to the matrix format below and generate filtered ground truth
with `generate_filter_groundtruth`.

## Command-line tools

| Tool | Purpose |
| --- | --- |
| `rfanns_build` | Build and save an index |
| `generate_filter_groundtruth` | Generate exact filtered top-k results |
| `rfanns_search` | Evaluate filtered search |
| `rfanns_update` | Run insert or delete workloads |
| `rfanns_validate` | Validate a saved index |

Minimal static workflow:

```bash
build/rfanns_build \
  --data_path base.bin --attr_path attributes.bin \
  --index_file index.bin --id_map_file index.idmap \
  --attr_min 0 --attr_max 99999 --num_bins 2048 \
  --M 16 --ef_construction 256 --threads 8

build/generate_filter_groundtruth \
  --data_path base.bin --query_path query.bin \
  --range_path ranges.bin --attr_path attributes.bin \
  --groundtruth_file groundtruth.bin \
  --K 10 --Nq 1000 --threads 8

build/rfanns_search \
  --data_path base.bin --query_path query.bin \
  --range_path ranges.bin --groundtruth_path groundtruth.bin \
  --index_file index.bin --id_map_file index.idmap \
  --result_file results.csv \
  --K 10 --Nq 1000 --ef 50,100,200
```

Use `scripts/generate_workload.py` and `rfanns_update` for dynamic experiments.
The smoke script provides a complete insertion and deletion example.

## Input formats

Binary data uses the host representation; the supported environment is
little-endian.

| File | Layout |
| --- | --- |
| Base/query vectors | `int32 count`, `int32 dimension`, then float32 values |
| Attributes | `count` raw int32 values |
| Query ranges | one inclusive `(int32 low, int32 high)` pair per query |
| Ground truth | `int32 query_count`, `int32 K`, then uint32 IDs |
| Evaluation manifest | TSV rows: `tag`, `range_path`, `groundtruth_path` |

Distances returned by the API are squared L2 distances.


## Citation
If you use our implementation, please cite the following paper.
``` 
@inproceedings{kido2027stride,
  title={Range-Filtered Approximate Nearest Neighbor Search on Fully Dynamic Data},
  author={Kido, Keito and Amagata, Daichi and Fujita, Sumio},
  booktitle={WSDM},
  year={2027}
}
``` 

## License

The locally developed DRFANNS code is Copyright (c) 2026 KAKu99 and is released
under the MIT License. The modified hnswlib files under `third_party/hnswlib`
remain under the Apache License 2.0. See `LICENSE`,
`third_party/hnswlib/LICENSE`, and `NOTICE.md`.
