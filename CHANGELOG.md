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
- `SeqId`, a stable 0-based identifier for an indexed reference. Assigned in build order and
  preserved across `to_bytes` / `from_bytes`.
- Sequence-id accessors on `FmIndex` and `BidirFmIndex` — `seq_headers()`, `seq_header(id)`
  and `seq_id(header)`. Both directions are O(1), backed by a header map built at
  construction and rebuilt on deserialization (it is derived, so it is not serialized).
- `FmIndexError::DuplicateHeader`, raised at build time when two sequences share a header.
  Uniqueness is what makes `seq_id` an exact inverse of `seq_header`. Sequences supplied
  without a header are still auto-named `seq_{i}`, so this only fires on genuinely repeated
  names.
- WASM bindings for the accessors: `seq_header`, `seq_id`, `seq_headers`.

### Changed
- Queries report match locations as `(SeqId, offset)` rather than `(String, offset)`, so no
  header string is allocated per occurrence — the cost was per occurrence rather than per
  seed, and scaled with seed multiplicity. Affects `FmIndex::locate`, `FmIndex::locate_gpu`,
  `FmIndex::map_position`, `BidirFmIndex::locate_interval`, `Mem::positions` and
  `MemHit::positions`, and the WASM `locate`. Callers that need labels resolve ids through
  `seq_header()`, or build an `id -> label` table once from `seq_headers()`.
- Licensed under Apache-2.0.

[Unreleased]: https://github.com/sriram98v/haystackfm/commits/main
