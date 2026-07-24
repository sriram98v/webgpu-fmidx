# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release under the `haystackfm` name. The project was previously published as
`webgpu-fmidx` (versions 0.1.0–0.5.1); version numbering restarts at 0.1.0 under the new
name, and its earlier history is not carried over here.

### Added
- GPU-accelerated FM-index construction (suffix array, BWT, Occ table) via WebGPU compute
  shaders, alongside CPU construction.
- `count` / `locate` queries; bidirectional index with MEM / SMEM finding (CPU and GPU paths).
- Full 16-symbol IUPAC ambiguity alphabet with a pluggable `Alphabet` trait
  (`IupacDna` default, `ExactDna` for exact ACGT matching).
- WASM bindings for in-browser WebGPU use; index serialization (`to_bytes` / `from_bytes`).
- Community health files, CI (fmt / clippy / build / test on `--all-features`), and Dependabot.
- Sequence-id accessors on `FmIndex` and `BidirFmIndex` — `seq_headers()`, `seq_header(id)`,
  `seq_id(header)`. Ids are 0-based in build order and stable across serialization, so callers
  can build an `id -> label` table once at load time.
- Integer-id locate variants that allocate no header `String` per occurrence:
  `FmIndex::locate_ids`, `BidirFmIndex::locate_interval_ids`, and `find_smems_ids` /
  `find_mems_ids` returning the new `MemIds` type. `MemIds::positions` has the same
  `(seq_id, offset)` shape as `MemHit::positions` on the GPU path. The existing
  `String`-returning `locate` / `locate_interval` / `find_smems` / `find_mems` are unchanged.
- WASM bindings for the above: `locateIds`, `seqHeader`, `seqId`.

### Changed
- Licensed under Apache-2.0.

[Unreleased]: https://github.com/sriram98v/haystackfm/commits/main
