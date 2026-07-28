# MEM / SMEM Finding

Exact-match queries answer "is this whole pattern present?". Seeding a read aligner instead
needs the *longest exact stretches* shared between a query and the reference — Maximal Exact
Matches (MEMs) and Super-Maximal Exact Matches (SMEMs). These require the bidirectional index,
`BidirFmIndex`.

## Build a bidirectional index

```rust
use haystackfm::{DnaSequence, BidirFmIndex, FmIndexConfig};

let refs = vec![DnaSequence::from_str("ACGTACGTACGT")?];
let config = FmIndexConfig::default();
let bidir = BidirFmIndex::build_cpu(&refs, &config)?;
```

## CPU MEM / SMEM

```rust
let query = DnaSequence::from_str("ACGT")?;

// Vec<Mem>. `min_len` filters short matches; `locate` resolves positions.
let smems = bidir.find_smems(query.as_slice(), /*min_len=*/18, /*locate=*/true);
let mems  = bidir.find_mems(query.as_slice(),  /*min_len=*/18, /*locate=*/true);
```

Both are IUPAC-aware (`N` matches any of A/C/G/T). Passing `locate = false` skips position
resolution and leaves `Mem::positions` empty — cheaper when you only need match extents and
counts.

### What counts as maximal

Maximality is judged **per occurrence**, as in MUMmer and BWA. `find_mems` reports a query
interval `[s,e)` when *at least one* of its occurrences is maximal in both directions at that
occurrence — preceded by a symbol incompatible with `query[s-1]` (or at a reference start, or
`s == 0`) and followed by a symbol incompatible with `query[e]` (or at a reference end, or
`e == query.len()`).

The distinction matters as soon as there is more than one reference:

```text
query  ACGTACGTAC
ref_a  ACGTACGTAC     holds the whole query
ref_b  ACGTACT        holds query[0..6), then T != query[6] = G

find_mems(min_len = 2)  ->  (0,2)  (0,6)  (0,10)  (4,10)  (8,10)
find_smems(min_len = 2) ->                (0,10)
```

`(0,6)` is a MEM on the strength of ref_b's occurrence alone, even though ref_a's occurrence
at the same query start extends further. Requiring *every* occurrence to be maximal would
drop it — and would collapse `find_mems` onto `find_smems`.

Two consequences worth planning for:

- `match_count` and `positions` cover only the **maximal** occurrences of a match, not every
  occurrence of the matched substring.
- The result can hold up to O(|query|²) intervals. `min_len` is the only bound on output
  size, so set it deliberately.

The SMEMs are exactly the containment-maximal MEMs, so `find_smems` output is always a subset
of `find_mems` output.

### `Mem`

```rust
pub struct Mem {
    pub query_start: usize,              // 0-based inclusive
    pub query_end:   usize,              // 0-based exclusive
    pub match_count: u32,                // number of occurrences
    pub positions:   Vec<(SeqId, u32)>,  // (seq id, offset) — empty when locate=false
}
```

`positions` identifies each reference by [`SeqId`](./concepts.md#sequence-ids-vs-headers) —
its 0-based build order, not its FASTA name. Resolve one with `bidir.seq_header(id)`, or
build a label table once from `bidir.seq_headers()` and index it by `id.index()`. This is the
hot path the id representation exists for: a conserved seed can occur in hundreds of
references, and positions are resolved per occurrence.

## GPU MEM / SMEM

> **`find_mems_gpu` is not currently equivalent to `find_mems`.** The shader still applies
> whole-set maximality and reports only the longest match per query start, so it returns a
> strict subset of the CPU result. Use the CPU path when you need occurrence-level MEMs.
> Tracked as issue 1 in [`KNOWN-ISSUES.md`](https://github.com/sriram98v/haystackfm/blob/main/KNOWN-ISSUES.md).
> `find_smems_gpu` is unaffected by this.

For batches of queries, run the GPU pipeline. Queries are passed as `&[DnaSequence]`, and
reference boundaries come from the index:

```rust
# #[cfg(feature = "gpu")]
# async fn run(bidir: &haystackfm::BidirFmIndex, query: haystackfm::DnaSequence) -> Result<(), Box<dyn std::error::Error>> {
let boundaries = bidir.seq_boundaries();     // reference-sequence boundaries
let queries = [query.clone()];               // &[DnaSequence]

// Vec<Vec<MemHit>> — the GPU context is drawn from a process-wide cache.
let smem_hits = bidir.find_smems_gpu(&queries, /*min_len=*/18, boundaries, /*max_hits_per_mem=*/1024).await?;
let mem_hits  = bidir.find_mems_gpu(&queries,  18, boundaries, 1024).await?;
// smem_hits[query_i] = Vec<MemHit> with resolved reference positions
# Ok(()) }
```

### `MemHit`

```rust
pub struct MemHit {                     // GPU result type
    pub query_start: u32,
    pub query_end:   u32,
    pub match_count: u32,
    pub positions:   Vec<(SeqId, u32)>, // same shape as Mem::positions
    pub truncated:   bool,              // true if positions capped at max_hits_per_mem
}
```

`positions` has the same `(SeqId, offset)` shape as the CPU `Mem::positions`, so the same
label-resolution code works against either path.

`max_hits_per_mem` caps how many positions each MEM resolves; when a MEM has more occurrences
than the cap, `truncated` is set. GPU results are parity-tested against the CPU
`find_mems` / `find_smems` output.
