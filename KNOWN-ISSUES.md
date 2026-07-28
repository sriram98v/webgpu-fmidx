# Known issues

Defects that are understood but not yet fixed. Each entry names the code, a reproduction
route, and what a fix has to do.

## 1. GPU MEM mode has not been ported to occurrence-level MEMs

**Where:** `shaders/mem_find.wgsl`, `process_query` MODE_MEM path.

`BidirFmIndex::find_mems` (CPU) now enumerates MEMs in the MUMmer / BWA sense — a query
interval is reported when at least one occurrence is maximal in both directions at that
occurrence. The shader still runs the previous whole-set algorithm: it right-extends until
the interval set goes empty, keeps only that longest match per start position, and applies a
single whole-set left-maximality check. So `find_mems_gpu` returns a strict subset of
`find_mems`.

**Impact:** `find_mems_gpu` and `find_mems` disagree. `find_smems_gpu` is not affected by
this entry (but see issue 2).

**Reproduction:** `query = ACGTACGTAC` against references `ACGTACGTAC` and `ACGTACT` with
`min_len = 2`. CPU returns `{(0,2), (0,6), (0,10), (4,10), (8,10)}`; the GPU returns
`{(0,10)}`. The CPU-side unit test is `find_mems_reports_all_occurrence_level_mems` in
`src/fm_index/smem.rs`.

**Current test posture:** the GPU MEM assertions in `tests/gpu_mem_parity.rs`,
`tests/gpu_mem_positions_parity.rs` and `tests/mem_iupac_parity.rs` compare the GPU against
`legacy_longest_match_per_start` in `tests/common/mod.rs` — a deliberate reimplementation of
the old whole-set algorithm — rather than against `find_mems`. They therefore assert that the
GPU still matches its documented (wrong) behavior, and will need to be pointed back at
`find_mems` as part of the port.

**What a fix must do:** emit a MEM at every occurrence-count drop rather than only at the end
of the forward extension, recover the dropped occurrences by extending with each symbol
*incompatible* with `query[j]`, and subtract the left-extendable ones. See
`collect_mem_candidates`, `drop_set` and `push_mem_candidate` in `src/fm_index/smem.rs` for
the CPU version. Note `MAX_IVS = 16` in the shader: a drop set can hold up to
`|incompatible symbols| × |ivs|` intervals, so the cap needs raising or the existing
truncation flag needs to cover it.

## 2. GPU SMEM mode lacks backward extension and the containment filter

**Where:** `shaders/mem_find.wgsl` MODE_SMEM (`i = last_j` pivot advance), and
`BidirFmIndex::find_smems_gpu` in `src/fm_index/bidir_index.rs`, which passes shader output
through with no host-side containment filter.

This is the bug the CPU `smem_raws` doc comment in `src/fm_index/smem.rs` describes as fixed:
advancing the pivot by its forward reach without backward-extending each candidate drops
SMEMs that start before the next pivot. The CPU path fixed it by collecting every
right-maximal prefix, backward-extending each independently (`collect_smem_candidates` +
`extend_left_maximally`), then filtering to the containment-maximal elements. The shader
still does the old pivot jump.

**Impact:** `find_smems_gpu` can silently drop valid SMEMs. Existing CPU/GPU SMEM parity
tests pass, but only because their corpora do not contain the triggering shape.

**Reproduction candidate:** the corpus in the `smem_drops_valid_longer_seed` unit test in
`src/fm_index/smem.rs` — the original CPU reproducer — run through `find_smems_gpu` and
compared against `find_smems`. **Not yet executed**; this entry is from reading the shader,
so confirm before acting on it.

**What a fix must do:** port `collect_smem_candidates` and `extend_left_maximally` to WGSL,
or apply the containment filter host-side after resolving intervals.

## 3. Pre-existing, unrelated

None recorded.
