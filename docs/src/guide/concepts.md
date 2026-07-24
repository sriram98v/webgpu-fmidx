# Concepts

A quick tour of the moving parts, so the rest of the guide has names to hang on.

## The building blocks

An FM-index is assembled from a few classic components:

- **Suffix Array (SA)** — the sorted order of all suffixes of the text. It's the backbone
  that turns "where does *P* occur" into a contiguous range.
- **Burrows–Wheeler Transform (BWT)** — a reversible permutation of the text derived from
  the SA. It clusters similar contexts together, which is what makes the index compressible
  and searchable.
- **C array** — for each symbol, the number of text characters that sort strictly before it.
  A tiny histogram (16 values for the IUPAC alphabet).
- **Occ table** — *rank* support: `Occ(c, i)` = how many times symbol `c` appears in the
  BWT up to position `i`. This is the hot data structure every query step touches.

Together, the **LF-mapping** (`C[c] + Occ(c, i)`) lets a backward search walk the text one
symbol at a time, shrinking an SA interval until it exactly covers every occurrence of the
pattern.

## Backward search

`count` and `locate` both start with a backward search: process the pattern right-to-left,
and at each step apply LF-mapping to narrow the `[lo, hi)` SA interval. When the interval is
empty the pattern is absent; otherwise its size is the occurrence count.

Because haystackfm is IUPAC-aware, a single query symbol can match several BWT symbols, so
the search tracks a **union of intervals** rather than one — see
[IUPAC Ambiguity](./iupac.md).

## Locate

`count` only needs the final interval size. `locate` must turn each SA-interval position
into a text coordinate. Storing the whole SA would defeat compression, so haystackfm samples
it every *k* positions (`sa_sample_rate`) and **LF-walks** from an unsampled position until
it lands on a sample — trading a little query time for a much smaller index.

## Sequence ids vs. headers

An index is built over several reference sequences concatenated into one text, so every
result has to say *which* reference it came from. haystackfm keeps two distinct things
apart here:

- A **header** is the FASTA name of a reference — `"chr1"`, `"NC_045512.2"`. It is what a
  human reads, and it is a `String`.
- A **`SeqId`** is that reference's position in build order — the first sequence you pass to
  the builder is `SeqId(0)`. It is a `u32`.

**Queries only ever report `SeqId`.** Locating an occurrence yields `(SeqId, offset)`, never
a header. This matters because locate cost is *per occurrence*: a seed conserved across
hundreds of references produces hundreds of hits, and cloning a header string for each one —
only for the caller to look it up again — is pure overhead. Returning the integer lets a
caller index a `Vec` directly.

Translating between the two is a separate, O(1) step:

```rust
index.seq_header(id)   // Option<&str>   — id  -> header
index.seq_id(header)   // Option<SeqId>  — header -> id
index.seq_headers()    // &[String]      — every header, indexed by SeqId::index()
```

The usual pattern is to build whatever label table you need *once* at load time from
`seq_headers()`, then index it by `SeqId::index()` per hit.

Ids are stable: they are assigned in build order and preserved across `to_bytes` /
`from_bytes`, so an id recorded against one load of an index still means the same reference
after a reload. They are *not* portable across differently-built indexes — reordering or
adding sequences renumbers everything.

Because `seq_id` must be an unambiguous inverse, **headers must be unique**: a collision
fails the build with `FmIndexError::DuplicateHeader`. Sequences supplied without a header are
auto-named `seq_{i}` in build order.

## Bidirectional index

Maximal Exact Match (MEM) and Super-Maximal Exact Match (SMEM) finding need to extend a
match in *both* directions. That requires a **bidirectional** FM-index — `BidirFmIndex` —
which maintains synchronized intervals over the forward and reverse texts. See
[MEM / SMEM Finding](./mem-smem.md).

## Where the GPU comes in

Construction (SA, BWT, Occ) and the query hot loops are all data-parallel, which maps well
to GPU compute. haystackfm implements each stage as WGSL compute shaders and keeps the CPU
implementation as the correctness oracle — every GPU path is parity-tested against it. The
full pipeline is laid out in [Architecture](../reference/architecture.md).
