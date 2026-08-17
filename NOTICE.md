K# Notices

This repository contains the compact RFANNS reference implementation. It is
prepared as a standalone source distribution without build outputs, datasets,
indexes, logs, figures, or machine-specific runner scripts.

## RFANNS implementation (MIT)

The implementation under `include/rfanns`, `src`, `tests`, and `tools`, along
with its CMake build definition, was developed locally as the RFANNS
implementation. Its earlier Proposal directories were local snapshots of the
same development process, not third-party upstream sources.

These files are distributed under the MIT License in the root `LICENSE`.
The copyright notice for the locally developed RFANNS code is:

```text
Copyright (c) 2026 Anonymous
```

The standalone release documentation, CI configuration, and helper scripts
were added or substantially rewritten while preparing this repository. Unless
otherwise identified below, they are also covered by the root MIT License.

## hnswlib SIMD subset (Apache-2.0)

This distribution separately includes modified portions of
[hnswlib](https://github.com/nmslib/hnswlib) from upstream revision
`d9b3608c83d83b46c96e25088cb1d729b29dcfe9`:

- `hnswlib/hnswlib.h` is copied to `third_party/hnswlib/hnswlib.h`. Its
  trailing automatic includes for unused hnswlib components are removed; the
  squared-L2 space include is retained.
- `hnswlib/space_l2.h` is copied to `third_party/hnswlib/space_l2.h`. Its
  AVX-512 and AVX temporary-array horizontal sums are replaced by
  register-only tree reductions.

hnswlib is distributed upstream under the Apache License 2.0. These vendored
files are not relicensed under the root MIT License. The Apache-2.0 license text
is included in `third_party/hnswlib/LICENSE`, and modification notices are
present in the modified headers.

### FeatureDetector code within hnswlib (CC0-1.0)

The vendored `hnswlib.h` retains the upstream notice that its CPU feature
detection code is adapted from
[Mysticial/FeatureDetector](https://github.com/Mysticial/FeatureDetector),
which is released under CC0 1.0.
