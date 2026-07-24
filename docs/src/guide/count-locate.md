# Count & Locate

The two core exact-match queries. Both operate on IUPAC-encoded byte slices
(`DnaSequence::as_slice()`).

## `count`

```rust
let n = index.count(query.as_slice()); // u32 — number of occurrences
```

Runs a backward search and returns the final SA-interval size — *O(|P|)*, independent of how
many times the pattern occurs.

## `locate`

```rust
let hits = index.locate(query.as_slice()); // Vec<(SeqId, u32)>
```

Returns every occurrence as `(seq_id, offset_within_seq)`, where `seq_id` is a
[`SeqId`](./concepts.md#sequence-ids-vs-headers) — the reference's 0-based position in build
order, *not* its FASTA name — and `offset` is a `u32` within that reference. Cost is
*O(|P| + occ·k)*, where `occ` is the number of hits and `k` is `sa_sample_rate` — each hit
LF-walks at most *k* steps to the nearest SA sample.

No header string is allocated per hit. Since locate cost scales with occurrence count, that
matters most exactly where it hurts: a seed conserved across many references.

## Sequence ids and headers

`SeqId` and the FASTA header are separate things; convert between them explicitly. Both
directions are O(1).

```rust
index.seq_header(id)?;         // Option<&str>   — id -> header
index.seq_id("chr1");          // Option<SeqId>  — header -> id
index.seq_headers();           // &[String]      — all headers, indexed by SeqId::index()
```

To label results, resolve per hit:

```rust
for (id, offset) in index.locate(query.as_slice()) {
    println!("{}:{offset}", index.seq_header(id).unwrap());
}
```

Or, in a hot loop, build your own table once and index it by `SeqId::index()`:

```rust
// Done once at load time, not per occurrence.
let labels: Vec<MyLabel> = index.seq_headers().iter().map(MyLabel::parse).collect();

for (id, offset) in index.locate(query.as_slice()) {
    let label = &labels[id.index()];   // plain Vec indexing — no hashing
}
```

Headers must be unique: a duplicate fails the build with `FmIndexError::DuplicateHeader`,
which is what makes `seq_id` an exact inverse of `seq_header`. Sequences built without an
explicit header are named `seq_{i}` in build order.

Ids survive serialization — see [Serialization](./serialization.md).

## Index metadata

```rust
index.text_len();        // u32 — total length of the concatenated text
index.num_sequences();   // u32 — number of indexed sequences
```

## GPU batch locate

When you have many patterns, resolve them together on the GPU. It is IUPAC-aware and returns
results per query:

```rust
use haystackfm::gpu::locate::locate_batch_gpu;

let ctx = haystackfm::gpu::GpuContext::new().await?;
let queries: Vec<&[u8]> = vec![q1.as_slice(), q2.as_slice()];
let hits = locate_batch_gpu(&ctx, &index, &queries).await?;
// hits[i] = Vec<(seq_id, offset_within_seq)> for queries[i]
```

Requires the `gpu` feature. GPU results carry the same occurrences as the CPU `locate` output
(order is not guaranteed), reported with the same `(SeqId, offset)` shape — so the same
label-resolution code works for both. `FmIndex::locate_gpu` wraps this for a whole batch. For
when the GPU path is worthwhile, see [GPU Acceleration](./gpu.md).
