# Implementation Plan: `webgpu-fmidx`

## 1. FM-Index Background

### 1.1 What is an FM-Index

The FM-index (Full-text Minute-space index) is a compressed full-text substring index based on the Burrows-Wheeler Transform (BWT). It enables exact pattern matching in O(m) time where m is the pattern length, with a space footprint that is a small constant factor of the original text size. It was introduced by Ferragina and Manzini (2000) and is the backbone of modern short-read aligners (BWA, Bowtie2).

An FM-index for a text T of length n over alphabet Sigma is composed of:

**Suffix Array (SA):** An array SA[0..n] of integers where SA[i] is the starting position in T of the i-th lexicographically smallest suffix. Construction dominates overall build cost. For text of length n, the SA has n+1 entries (including the sentinel `$`).

**Burrows-Wheeler Transform (BWT):** Defined as BWT[i] = T[SA[i] - 1] (with wraparound: if SA[i] == 0, BWT[i] = T[n]). The BWT is a permutation of T that tends to cluster identical characters, enabling compression. It is trivially derived from the SA in O(n) time.

**C Array (First Column):** C[c] = number of characters in T that are lexicographically smaller than c. For DNA alphabet {$, A, C, G, T}, this is a 5-element array. Computed in O(n) time via a single pass counting character frequencies.

**Occ Table (Rank Structure):** Occ(c, i) = number of occurrences of character c in BWT[0..i]. This is the rank function. Combined with C, it enables the LF-mapping: LF(i) = C[BWT[i]] + Occ(BWT[i], i). The Occ table can be represented as:
- A naive 2D array of size |Sigma| x (n+1) -- O(n * |Sigma|) space, O(1) query
- A sampled checkpoint + bitvector approach -- O(n) space, O(1) query with constant-time rank
- A wavelet tree -- O(n log|Sigma|) space, O(log|Sigma|) query, supports generalized rank/select

For DNA (|Sigma| = 5), a sampled Occ table with checkpoints every k positions is the most practical: store full counts every k-th row, and use popcount on a packed bitvector for the remainder. With k=64, this gives O(n) space and O(1) rank queries.

### 1.2 Why GPU Acceleration Matters

FM-index construction is dominated by suffix array construction, which is O(n) to O(n log n) depending on algorithm. For large genomic datasets (millions of sequences, total length in the billions of bases), construction time becomes a significant bottleneck:

- **Suffix array sorting** is the primary bottleneck (>90% of build time). Comparison-based suffix sorting involves massive data movement and is memory-bandwidth bound -- exactly the regime where GPU parallelism excels.
- **Prefix sum operations** used in counting sort, radix sort, and Occ table construction are textbook GPU primitives.
- **BWT derivation** from SA is an embarrassingly parallel gather operation.
- **Occ table construction** involves per-character prefix sums -- 5 independent parallel prefix scans for DNA.

For a WebGPU/browser context, the motivation is enabling client-side index construction without server roundtrips. A user uploads FASTA/FASTQ data, the browser builds the FM-index locally, and queries run entirely client-side. GPU acceleration makes this feasible for non-trivial dataset sizes (up to hundreds of megabases within WebGPU buffer limits).

### 1.3 Parallelizable Components

| Component | Parallelism Type | GPU Suitability | Complexity |
|-----------|-----------------|-----------------|------------|
| Suffix Array (prefix doubling) | Data-parallel sort + scatter | HIGH | O(n log^2 n) work, O(log^2 n) depth |
| Suffix Array (DC3/skew) | Recursive + radix sort | MEDIUM | O(n) work, harder to parallelize |
| Suffix Array (SA-IS) | Scan + induce | MEDIUM | O(n) work, sequential dependencies |
| BWT from SA | Embarrassingly parallel gather | HIGH | O(n) work, O(1) depth |
| C array | Parallel reduction (histogram) | HIGH | O(n) work, O(log n) depth |
| Occ table | Per-character prefix sum | HIGH | O(n * |Sigma|) work, O(log n) depth |
| Rank queries | Independent lookups | HIGH (batched) | O(1) per query |

**Chosen approach: Prefix doubling for SA construction.** While DC3 and SA-IS are asymptotically superior (O(n) vs O(n log^2 n)), prefix doubling maps naturally to GPU radix sort primitives and has simple, regular memory access patterns. For the DNA alphabet (|Sigma|=5) and practical sequence lengths (up to ~128M bases in WebGPU), the log^2 n factor is at most ~17 iterations of 289 = (log2(128M))^2, each dominated by a radix sort. The radix sort itself is O(n) per iteration on a GPU. The total GPU work is O(n * log^2 n), but with massive parallelism the wall-clock time is O(log^2 n * log n) for the sort phases.

## 2. Architecture

### 2.1 Crate Structure

```
webgpu-fmidx/
|-- Cargo.toml
|-- README.md
|-- plan.md
|-- .gitignore
|-- src/
|   |-- lib.rs                    # Crate root, public API, feature gates
|   |-- alphabet.rs               # DNA alphabet encoding ($=0, A=1, C=2, G=3, T=4)
|   |-- suffix_array/
|   |   |-- mod.rs                # SA trait + factory
|   |   |-- cpu.rs                # CPU prefix-doubling SA construction
|   |   |-- gpu.rs                # GPU prefix-doubling SA construction (wgpu)
|   |-- bwt/
|   |   |-- mod.rs                # BWT trait + factory
|   |   |-- cpu.rs                # CPU BWT derivation from SA
|   |   |-- gpu.rs                # GPU BWT derivation (parallel gather)
|   |-- occ/
|   |   |-- mod.rs                # Occ table trait + factory
|   |   |-- cpu.rs                # CPU Occ table construction (prefix sums)
|   |   |-- gpu.rs                # GPU Occ table construction (parallel prefix sum)
|   |-- c_array.rs                # C array construction (small; CPU-only is fine)
|   |-- fm_index/
|   |   |-- mod.rs                # FmIndex struct, build pipeline orchestration
|   |   |-- query.rs              # count() and locate() query implementations
|   |   |-- serialize.rs          # Serialization/deserialization (bincode or custom)
|   |-- gpu/
|   |   |-- mod.rs                # GPU context management (device, queue, adapter)
|   |   |-- buffers.rs            # GPU buffer allocation, upload, download helpers
|   |   |-- pipeline.rs           # Compute pipeline creation + dispatch helpers
|   |   |-- prefix_sum.rs         # Reusable parallel prefix sum implementation
|   |   |-- radix_sort.rs         # GPU radix sort (used by SA construction)
|   |   |-- histogram.rs          # GPU parallel histogram (used by C array, radix sort)
|   |-- wasm/
|   |   |-- mod.rs                # wasm-bindgen entry points
|   |   |-- js_api.rs             # JavaScript-facing API (FmIndexBuilder, FmIndexQuery)
|   |-- error.rs                  # Error types
|-- shaders/
|   |-- prefix_sum.wgsl           # Blelloch-style inclusive/exclusive prefix sum
|   |-- radix_sort_count.wgsl     # Per-workgroup digit histogram
|   |-- radix_sort_scatter.wgsl   # Global scatter phase of radix sort
|   |-- sa_init_ranks.wgsl        # Initialize SA ranks from first character
|   |-- sa_compare_pairs.wgsl     # Compare adjacent suffixes by (rank[i], rank[i+h])
|   |-- sa_update_ranks.wgsl      # Assign new ranks from comparison results
|   |-- bwt_gather.wgsl           # BWT[i] = T[SA[i]-1] parallel gather
|   |-- occ_scan.wgsl             # Per-character prefix sum for Occ table
|   |-- histogram.wgsl            # Character frequency histogram
|-- tests/
|   |-- common/
|   |   |-- mod.rs                # Shared test utilities, known FM-index test vectors
|   |-- cpu_correctness.rs        # CPU implementation correctness tests
|   |-- gpu_correctness.rs        # GPU implementation correctness tests (requires WebGPU)
|   |-- cpu_gpu_equivalence.rs    # Property: CPU and GPU produce identical results
|   |-- query_tests.rs            # FM-index count/locate query tests
|   |-- serialize_tests.rs        # Round-trip serialization tests
|-- benches/
|   |-- construction.rs           # Criterion benchmarks: CPU vs GPU, varying sizes
|   |-- query.rs                  # Query benchmarks
|-- examples/
|   |-- build_index.rs            # CLI example: build FM-index from FASTA
|-- web/                          # Browser demo (not part of the crate)
|   |-- index.html
|   |-- main.ts                   # TypeScript demo using the WASM module
|   |-- package.json
|   |-- tsconfig.json
```

### 2.2 WASM Compilation Strategy

**Toolchain:**
- `wasm-pack` for building the WASM package with JS bindings
- `wasm-bindgen` for Rust-to-JS FFI
- `wasm-bindgen-futures` for async WebGPU operations (device request, buffer mapping)
- Target: `wasm32-unknown-unknown`

**Feature flags in `Cargo.toml`:**
```toml
[features]
default = ["cpu"]
cpu = []                  # CPU-only implementations
gpu = ["dep:wgpu"]        # GPU-accelerated implementations
wasm = ["gpu", "dep:wasm-bindgen", "dep:wasm-bindgen-futures", "dep:js-sys", "dep:web-sys"]
```

**Build command:**
```bash
# Native (CPU-only, for testing)
cargo build --features cpu

# Native with GPU (uses wgpu Vulkan/Metal backend)
cargo build --features gpu

# WASM (browser, WebGPU)
wasm-pack build --target web --features wasm
```

**Key constraint:** The `wgpu` crate compiles to WASM and uses the browser's `navigator.gpu` API when targeting `wasm32-unknown-unknown`. No additional WebGPU bindings are needed -- wgpu handles the abstraction. However, all GPU operations are async in the browser, so the build pipeline must be fully async.

### 2.3 WebGPU Integration Approach

Use the `wgpu` crate (version 24.x, matching your existing project) which provides:
- Native backends (Vulkan, Metal, DX12) for development and testing
- WebGPU backend when compiled to WASM
- Uniform API across all targets

**GPU context lifecycle:**
1. `GpuContext::new()` -- async: request adapter, request device with required limits
2. Device limits to request: `max_buffer_size` (at least 256MB), `max_storage_buffer_binding_size` (at least 128MB), `max_compute_workgroups_per_dimension` (at least 65535)
3. The `GpuContext` owns the `Device` and `Queue` and is shared across all pipeline stages
4. Each pipeline stage (SA, BWT, Occ) creates its own compute pipelines and bind groups

### 2.4 CPU/GPU Pipeline Collaboration

```
Input: Vec<DnaSequence>
        |
        v
[CPU] Concatenation + Encoding
  Concatenate all sequences with $ separators: S = s1$s2$...$sn$
  Encode to u8 array: $ -> 0, A -> 1, C -> 2, G -> 3, T -> 4
        |
        v
[CPU -> GPU] Upload encoded text to GPU storage buffer
        |
        v
[GPU] Suffix Array Construction (prefix doubling)
  - Initialize ranks from first character
  - Iterative doubling: sort by (rank[i], rank[i+h]) using radix sort
  - Detect when all ranks are unique (parallel reduction)
  - O(log^2 n) iterations, each with O(n) radix sort
        |
        v
[GPU] BWT Construction
  - Parallel gather: BWT[i] = text[(SA[i] + n) % (n+1)]  (or text[SA[i]-1] with wrap)
        |
        v
[GPU] Occ Table Construction
  - 5 parallel prefix sums (one per alphabet symbol)
  - Checkpoint every 64 positions, store packed bitvectors between checkpoints
        |
        v
[GPU -> CPU] Download BWT, Occ checkpoints, SA (sampled or full)
        |
        v
[CPU] C Array Construction
  - Single pass over BWT, accumulate character frequencies
  - Prefix sum of frequencies -> C array
  (This is trivially fast on CPU; not worth a GPU dispatch)
        |
        v
[CPU] Assemble FmIndex struct
  - BWT, C, Occ checkpoints, sampled SA
  - Ready for count() and locate() queries
```

## 3. Data Structures

### 3.1 Suffix Array Representation

```rust
/// Suffix array: SA[i] = starting position of the i-th lexicographically smallest suffix.
/// Stored as Vec<u32> for texts up to 4GB (sufficient for WebGPU buffer limits).
/// On GPU: stored as a storage buffer of u32 elements.
pub struct SuffixArray {
    data: Vec<u32>,  // SA[0..n], where n = text.len() (including sentinel)
}
```

**GPU buffer layout:** A single `storage` buffer of `n * 4` bytes, where each element is a `u32` suffix index. For a 128MB text, this is 512MB of GPU memory for the SA alone, which is within WebGPU's `maxBufferSize` on modern GPUs (typically 1-4GB).

**Sampled SA for space reduction:** After construction, store only every k-th entry (k=32 or k=64) plus the full BWT. To recover SA[i] for non-sampled positions, walk backwards through the BWT using LF-mapping until hitting a sampled position. Trade-off: k steps per locate query vs k-fold space reduction.

```rust
pub struct SampledSuffixArray {
    samples: Vec<u32>,     // SA values at every k-th position
    sample_rate: u32,      // k
}
```

### 3.2 BWT Construction

The BWT is derived from the SA via a simple gather:

```
BWT[i] = T[(SA[i] - 1 + n) % n]    for SA with sentinel
```

where n is the length of T including the sentinel `$`.

```rust
pub struct Bwt {
    data: Vec<u8>,  // BWT[0..n], encoded as alphabet indices (0-4)
}
```

**GPU buffer layout:** A single `storage` buffer of `n` bytes. Since WGSL does not support `u8` storage buffers directly, the BWT is packed 4 characters per `u32` (each character is 3 bits, but we use 8 bits for simplicity and alignment). The gather shader reads from the text buffer and SA buffer, writing to the BWT buffer.

Actually, WGSL storage buffers must be arrays of `u32` or `vec4<u32>`. So the BWT buffer stores `ceil(n/4)` u32 values, with 4 characters packed per u32 (one byte per character in the low 8 bits of each byte position). The shader packs on write and the CPU unpacks on readback.

### 3.3 Occ Table (Rank Structure)

For the DNA alphabet (|Sigma|=5), we use a **checkpoint + bitvector** scheme:

**Checkpoints:** Every `BLOCK_SIZE` positions (BLOCK_SIZE=64), store the cumulative count of each character up to that position. Each checkpoint is 5 x u32 = 20 bytes.

**Bitvectors:** Between checkpoints, store a 64-bit bitvector for each character indicating which positions in the block contain that character. For |Sigma|=5, each block needs 5 x u64 = 40 bytes of bitvectors.

**Rank query Occ(c, i):**
1. Find block index: `block = i / BLOCK_SIZE`
2. Offset within block: `off = i % BLOCK_SIZE`
3. `Occ(c, i) = checkpoint[block][c] + popcount(bitvector[block][c] & ((1 << (off+1)) - 1))`

This gives O(1) rank queries with no branching.

```rust
pub struct OccTable {
    checkpoints: Vec<[u32; 5]>,  // checkpoints[block][char] = cumulative count
    bitvectors: Vec<[u64; 5]>,   // bitvectors[block][char] = presence bits
    block_size: u32,             // 64
}
```

**GPU buffer layout for construction:**

During construction, we need a full `|Sigma| x n` matrix of prefix sums. This is too large to materialize for big texts. Instead, we use a two-pass approach:

Pass 1 (per-block): Each workgroup processes one block of BLOCK_SIZE characters. It computes the local character counts for that block. Output: `block_counts[block][char]` -- a `num_blocks x 5` array of u32.

Pass 2 (prefix sum over blocks): Parallel prefix sum over `block_counts` for each character independently. Output: `checkpoint[block][char]` = sum of `block_counts[0..block][char]`.

Pass 3 (bitvector construction): Each workgroup processes one block, setting bits in 5 u64 bitvectors based on the character at each position.

Passes 1 and 3 can be fused into a single shader.

### 3.4 C Array

```rust
pub struct CArray {
    data: [u32; 5],  // C[c] = number of characters < c in the text
}
```

Computed from the character histogram: count occurrences of each character, then exclusive prefix sum.

For DNA: `C[$]=0, C[A]=count($), C[C]=count($)+count(A), C[G]=count($)+count(A)+count(C), C[T]=count($)+count(A)+count(C)+count(G)`.

This is trivially computed on CPU from the final Occ checkpoint (the last checkpoint contains total counts for each character).

### 3.5 Complete FM-Index Structure

```rust
pub struct FmIndex {
    bwt: Bwt,
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    text_len: u32,  // length including sentinel
}
```

## 4. GPU Compute Pipeline

### 4.1 Shader Inventory

| Shader File | Purpose | Inputs | Outputs |
|------------|---------|--------|---------|
| `prefix_sum.wgsl` | Blelloch inclusive/exclusive prefix sum on u32 array | `data[]`, `n` | `data[]` (in-place) |
| `radix_sort_count.wgsl` | Per-workgroup digit histogram for radix sort | `keys[]`, `digit_shift`, `n` | `local_histograms[]` |
| `radix_sort_scatter.wgsl` | Global scatter using prefix-summed histograms | `keys_in[]`, `values_in[]`, `global_offsets[]`, `digit_shift`, `n` | `keys_out[]`, `values_out[]` |
| `sa_init_ranks.wgsl` | Initialize rank[i] = text[i] for each suffix | `text[]`, `n` | `ranks[]`, `sa[]` (identity permutation) |
| `sa_compare_pairs.wgsl` | Compare (rank[sa[i]], rank[(sa[i]+h)%n]) vs (rank[sa[i-1]], rank[(sa[i-1]+h)%n]) | `sa[]`, `ranks[]`, `h`, `n` | `flags[]` (0 if equal to predecessor, 1 if different) |
| `sa_update_ranks.wgsl` | New rank[sa[i]] = prefix_sum(flags)[i] | `sa[]`, `flags[]`, `prefix_sums[]`, `n` | `new_ranks[]` |
| `bwt_gather.wgsl` | BWT[i] = text[(SA[i]-1+n)%n] | `text[]`, `sa[]`, `n` | `bwt[]` |
| `occ_block_count.wgsl` | Count chars per block + build bitvectors | `bwt[]`, `n`, `block_size` | `block_counts[]`, `bitvectors[]` |
| `histogram.wgsl` | Global character histogram | `text[]`, `n` | `histogram[5]` |

### 4.2 Parallel Prefix Sum (Blelloch Scan)

This is the most critical GPU primitive, reused by radix sort, Occ construction, and SA rank update.

**Algorithm:** Work-efficient parallel prefix sum (Blelloch 1990). Two phases:
1. **Up-sweep (reduce):** Build a binary tree of partial sums from leaves to root. O(n) work, O(log n) depth.
2. **Down-sweep (distribute):** Propagate prefix sums from root back to leaves. O(n) work, O(log n) depth.

**For arrays larger than one workgroup:** Use a hierarchical approach:
1. Each workgroup computes prefix sums of its local segment (up to `WORKGROUP_SIZE * 2` elements)
2. The last element of each workgroup's scan is written to an auxiliary `block_sums` array
3. Recursively prefix-sum the `block_sums` array
4. A final "add block sums" pass adds the appropriate block sum to every element in each workgroup

**WGSL sketch for single-workgroup prefix sum:**

```wgsl
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

var<workgroup> temp: array<u32, 512>;  // 2 * WORKGROUP_SIZE

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) gid: vec3u) {
    let n = params.n;
    let offset = gid.x * 512u;
    // Load into shared memory
    temp[2u * lid.x] = data[offset + 2u * lid.x];
    temp[2u * lid.x + 1u] = data[offset + 2u * lid.x + 1u];
    workgroupBarrier();

    // Up-sweep
    var stride = 1u;
    for (var d = 256u; d > 0u; d >>= 1u) {
        if (lid.x < d) {
            let ai = stride * (2u * lid.x + 1u) - 1u;
            let bi = stride * (2u * lid.x + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        stride <<= 1u;
        workgroupBarrier();
    }

    // Set root to zero for exclusive scan
    if (lid.x == 0u) { temp[511u] = 0u; }
    workgroupBarrier();

    // Down-sweep
    for (var d = 1u; d < 512u; d <<= 1u) {
        stride >>= 1u;
        if (lid.x < d) {
            let ai = stride * (2u * lid.x + 1u) - 1u;
            let bi = stride * (2u * lid.x + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        workgroupBarrier();
    }

    // Write back
    data[offset + 2u * lid.x] = temp[2u * lid.x];
    data[offset + 2u * lid.x + 1u] = temp[2u * lid.x + 1u];
}
```

**Workgroup size:** 256 threads, each handling 2 elements = 512 elements per workgroup. For n elements, dispatch `ceil(n / 512)` workgroups in the first pass.

### 4.3 GPU Radix Sort

Used for suffix array construction (sorting suffixes by their rank pairs). We sort `(key, value)` pairs where key = `(rank[sa[i]], rank[(sa[i]+h)%n])` packed into a u64 (or two u32s sorted lexicographically) and value = `sa[i]`.

**Algorithm:** Least-significant-digit (LSD) radix sort with radix = 256 (8-bit digits). For 32-bit keys, 4 passes. For rank pairs (two u32 keys), sort by secondary key first, then primary key (stable sort preserves secondary ordering).

**Per pass:**
1. **Count phase:** Each workgroup counts the frequency of each digit (0-255) in its segment. Output: `local_histogram[workgroup][256]`.
2. **Prefix sum:** Exclusive prefix sum over the flattened histogram to get global scatter offsets.
3. **Scatter phase:** Each workgroup scatters its elements to the correct global positions.

**Buffer requirements per radix sort pass:**
- `keys_in[n]` -- u32 storage buffer
- `keys_out[n]` -- u32 storage buffer (double buffered)
- `values_in[n]` -- u32 storage buffer (SA indices)
- `values_out[n]` -- u32 storage buffer
- `histograms[num_workgroups * 256]` -- u32 storage buffer
- `prefix_sums[num_workgroups * 256]` -- u32 storage buffer

**Dispatch strategy:**
- Workgroup size: 256 threads
- Elements per workgroup: 1024 (4 per thread, tunable)
- Number of workgroups: `ceil(n / 1024)`

### 4.4 Parallel SA Construction (Prefix Doubling)

**Algorithm overview (Karp, Miller, Rosenberg 1972; adapted for GPU):**

```
1. Initialize:
   - SA[i] = i for all i in [0, n)
   - rank[i] = text[i] (character code) for all i

2. For h = 1, 2, 4, 8, ..., until all ranks are unique:
   a. Form sort key for each suffix i: key_i = (rank[i], rank[(i + h) % n])
   b. Stable sort SA by these keys using GPU radix sort
   c. Recompute ranks:
      - Compare adjacent sorted suffixes' key pairs
      - flag[i] = (key_{SA[i]} != key_{SA[i-1]}) ? 1 : 0, flag[0] = 1
      - new_rank[SA[i]] = prefix_sum(flag)[i]
   d. Check if max(new_rank) == n - 1 (all unique). If so, done.
   e. rank = new_rank, h *= 2

3. Return SA
```

**Complexity:**
- Each iteration: O(n) for radix sort + O(n) for comparison + O(n) for prefix sum = O(n)
- Number of iterations: O(log n) in the worst case, but typically O(log n) for random text. For repetitive DNA, could be O(log n) to O(log^2 n).
- Total: O(n log n) to O(n log^2 n) work. GPU parallelism reduces wall-clock to O(log n * sort_depth).

**GPU buffers needed:**
- `text[n]` -- packed u32 (4 chars per u32), read-only
- `sa[n]` -- u32, read-write
- `ranks[n]` -- u32, read-write
- `new_ranks[n]` -- u32, write
- `key_primary[n]` -- u32 (rank[sa[i]])
- `key_secondary[n]` -- u32 (rank[(sa[i]+h)%n])
- `flags[n]` -- u32 (comparison results)
- Radix sort temporary buffers (see section 4.3)

**Total GPU memory for SA construction:** Approximately `12n` bytes (3 x u32 arrays for sa, ranks, new_ranks) plus radix sort buffers (~`8n` bytes for double-buffered keys+values plus histograms). Roughly `20n` to `24n` bytes total. For n = 128M, this is ~2.4-3.0 GB, which is within the memory budget of most discrete GPUs but may be tight on integrated GPUs.

**Optimization: early termination.** After each iteration, check if all ranks are unique by computing the maximum rank via parallel reduction. If `max_rank == n-1`, all suffixes are distinguished and we can stop. For random DNA text, this typically converges in 10-15 iterations.

### 4.5 Parallel BWT Construction

Trivially parallel once SA is computed:

```wgsl
@compute @workgroup_size(256)
fn bwt_gather(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let sa_i = sa[i];
    let text_pos = select(sa_i - 1u, params.n - 1u, sa_i == 0u);

    // text is packed 4 chars per u32
    let word_idx = text_pos / 4u;
    let byte_idx = text_pos % 4u;
    let ch = (text_packed[word_idx] >> (byte_idx * 8u)) & 0xFFu;

    // Pack into BWT output (4 chars per u32)
    let out_word = i / 4u;
    let out_byte = i % 4u;
    // Use atomicOr since multiple threads write different bytes of the same u32
    atomicOr(&bwt_packed[out_word], ch << (out_byte * 8u));
}
```

**Note on atomics:** Since 4 threads may write to different bytes of the same u32, we use `atomicOr` on a zero-initialized buffer. Each byte position is written by exactly one thread, so there are no data races on the logical values, but the u32-level write requires atomics.

**Alternative:** Sort output indices so that threads in the same workgroup write to consecutive u32 words, avoiding atomics entirely. Since `i` is the global invocation ID and increases linearly, consecutive threads already write to consecutive positions -- we only need atomics for the 4 threads sharing the boundary u32 of each workgroup.

Actually, the simpler approach: just output to a `u32` buffer where each element holds one character (no packing on GPU). Pack on CPU readback. This wastes 3/4 of GPU memory bandwidth but avoids all complexity.

### 4.6 Parallel Occ Table Construction

**Pass 1 -- Block counts + bitvectors (fused):**

Each workgroup processes one block of 64 BWT characters. Within the workgroup, 64 threads each inspect one character and:
1. Set the appropriate bit in the workgroup-shared bitvector for that character
2. Contribute to the block's character count via workgroup-level reduction

Output: `block_counts[block_id][5]` and `bitvectors[block_id][5]`.

**Pass 2 -- Prefix sum over block counts:**

For each of the 5 characters independently, perform an exclusive prefix sum over `block_counts[*][c]`. This gives `checkpoints[block_id][c]` = total count of character `c` in all blocks before `block_id`.

This is 5 independent prefix sums, each of length `ceil(n/64)`. They can be batched into a single dispatch if we interleave the data, or run as 5 separate dispatches.

**Workgroup size for Pass 1:** 64 threads (one per block element). Since 64 is below the typical optimal workgroup size of 256, we can instead have 256 threads process 4 blocks each, but the simpler approach is fine for correctness-first development.

### 4.7 Synchronization Strategy

WebGPU does not allow synchronization between workgroups within a single dispatch. All inter-workgroup synchronization must happen via separate dispatches (command encoder submissions).

**Command encoder sequence for full FM-index build:**

```
encoder.begin_compute_pass()

// --- SA Construction ---
// Dispatch: sa_init_ranks
// For each doubling iteration:
//   Dispatch: form_sort_keys (gather rank pairs)
//   For each radix sort pass (4 passes for 32-bit keys, x2 for two keys):
//     Dispatch: radix_sort_count
//     Dispatch: prefix_sum (on histograms, possibly multi-level)
//     Dispatch: radix_sort_scatter
//   Dispatch: sa_compare_pairs
//   Dispatch: prefix_sum (on flags)
//   Dispatch: sa_update_ranks
//   Dispatch: parallel_max (reduction to check convergence)
//   // Read back max_rank to CPU to check termination
//   // If not converged, continue loop

// --- BWT Construction ---
// Dispatch: bwt_gather

// --- Occ Table Construction ---
// Dispatch: occ_block_count_and_bitvectors
// Dispatch: prefix_sum (on block_counts, per character)

encoder.end_compute_pass()
encoder.copy_buffer_to_buffer(...)  // stage results for CPU readback
queue.submit(encoder.finish())
```

**Important:** The prefix doubling loop requires reading `max_rank` back to the CPU to decide whether to continue iterating. This means we cannot encode the entire SA construction in a single command buffer. Each iteration must be submitted, the max value read back (via `map_async` on a staging buffer), and the decision made on the CPU. This introduces a CPU-GPU synchronization point per iteration.

**Optimization:** We can speculatively encode multiple iterations and use a flag buffer to skip unnecessary work in later iterations. But this is a premature optimization; the simple approach with per-iteration sync is correct and sufficient for Phase 2.

## 5. Algorithm Details

### 5.1 Suffix Array Construction: Prefix Doubling

**Formal description:**

Let T[0..n-1] be the input text of length n (including sentinel $). Define:
- `suffix(i)` = T[i..n-1] (the suffix starting at position i)
- `rank_h(i)` = the rank of `suffix(i)` when suffixes are compared by their first `h` characters only

**Initialization (h=1):**
- `rank_1(i) = T[i]` (the character code at position i)
- `SA` = any permutation (identity works)

**Doubling step (h -> 2h):**
- Sort the suffixes stably by the key pair `(rank_h(i), rank_h((i+h) % n))`
- The second component handles wraparound for suffixes near the end of the text
- After sorting, assign new ranks based on the sorted order:
  - Two suffixes get the same rank if and only if their key pairs are identical
  - This is detected by comparing adjacent elements in the sorted order

**Termination:**
- When all n ranks are distinct (max rank = n-1), the suffixes are fully sorted
- The SA is the current permutation

**Why prefix doubling is GPU-friendly:**
1. Each iteration is dominated by a stable sort, which is a well-known GPU primitive (radix sort)
2. Rank comparison and update are embarrassingly parallel scatter/gather operations
3. No recursive data dependencies within an iteration (unlike SA-IS or DC3)
4. Regular memory access patterns (no pointer chasing)

**Handling the sentinel:**
The sentinel `$` has rank 0 (lexicographically smallest). It is always the first suffix in sorted order. We can either:
- Include it in the sort (simplest, costs one extra element)
- Handle it specially (omit from sort, place at position 0)

We choose to include it for simplicity.

### 5.2 BWT Derivation from SA

Given SA[0..n-1], the BWT is:

```
BWT[i] = T[SA[i] - 1]     if SA[i] > 0
BWT[i] = T[n - 1]          if SA[i] == 0  (wrap around; this is always '$')
```

Wait -- more precisely, if we define T' = T + $ (so T' has length n and T'[n-1] = $), then:

```
BWT[i] = T'[(SA[i] - 1 + n) % n]
```

This is correct because SA[i]=0 means the suffix starting at position 0 (the entire text) is the i-th smallest. Its predecessor character wraps to T'[n-1] = $.

Complexity: O(n) work, O(1) depth -- embarrassingly parallel.

### 5.3 Rank/Select Data Structure

**Rank query (used in FM-index search):**

`Occ(c, i)` = number of occurrences of character c in BWT[0..i-1].

Using the checkpoint + bitvector structure (Section 3.3):

```rust
fn rank(&self, c: u8, i: u32) -> u32 {
    let block = (i / BLOCK_SIZE) as usize;
    let offset = i % BLOCK_SIZE;
    let checkpoint = self.checkpoints[block][c as usize];
    let bitvec = self.bitvectors[block][c as usize];
    let mask = if offset == 0 { 0 } else { (1u64 << offset) - 1 };
    checkpoint + (bitvec & mask).count_ones() as u32
}
```

Complexity: O(1) per query (popcount is a single instruction on modern CPUs and can be computed in O(1) in WGSL using `countOneBits`).

**Select query (optional, used for locate):**

`Select(c, j)` = position of the j-th occurrence of character c in BWT.

Not needed for basic FM-index count/locate. Can be implemented via binary search over rank if needed. For locate, we use the sampled SA approach instead.

### 5.4 FM-Index Query: Count

Find the number of occurrences of pattern P[0..m-1] in the text:

```rust
fn count(&self, pattern: &[u8]) -> u32 {
    let mut lo = 0u32;
    let mut hi = self.text_len;  // Inclusive range [lo, hi)

    // Backward search: process pattern from right to left
    for &c in pattern.iter().rev() {
        lo = self.c_array[c] + self.occ.rank(c, lo);
        hi = self.c_array[c] + self.occ.rank(c, hi);
        if lo >= hi {
            return 0;  // Pattern not found
        }
    }

    hi - lo  // Number of occurrences
}
```

Complexity: O(m) per query, where m = pattern length. Each step is O(1) (C array lookup + rank query).

### 5.5 FM-Index Query: Locate

Find the actual text positions of all occurrences:

```rust
fn locate(&self, pattern: &[u8]) -> Vec<u32> {
    let mut lo = 0u32;
    let mut hi = self.text_len;

    for &c in pattern.iter().rev() {
        lo = self.c_array[c] + self.occ.rank(c, lo);
        hi = self.c_array[c] + self.occ.rank(c, hi);
        if lo >= hi {
            return vec![];
        }
    }

    // For each position in [lo, hi), recover the text position
    (lo..hi).map(|i| self.resolve_sa(i)).collect()
}

fn resolve_sa(&self, mut i: u32) -> u32 {
    let mut steps = 0u32;
    loop {
        if i % self.sa_samples.sample_rate == 0 {
            return self.sa_samples.samples[(i / self.sa_samples.sample_rate) as usize] + steps;
        }
        // LF-mapping: step backward through BWT
        let c = self.bwt[i as usize];
        i = self.c_array[c] + self.occ.rank(c, i);
        steps += 1;
    }
}
```

Complexity: O(m + occ * k) where occ = number of occurrences, k = SA sample rate. Each LF-mapping step is O(1).

## 6. API Design

### 6.1 Public Rust API

```rust
// --- lib.rs ---

/// Configuration for FM-index construction.
pub struct FmIndexConfig {
    /// SA sampling rate for locate queries. Higher = less memory, slower locate.
    /// Default: 32. Set to 1 for full SA (fastest locate, most memory).
    pub sa_sample_rate: u32,
    /// Occ table block size. Default: 64.
    pub occ_block_size: u32,
    /// Whether to use GPU acceleration. Falls back to CPU if GPU unavailable.
    pub use_gpu: bool,
}

impl Default for FmIndexConfig {
    fn default() -> Self {
        Self {
            sa_sample_rate: 32,
            occ_block_size: 64,
            use_gpu: true,
        }
    }
}

/// A DNA sequence (A, C, G, T only).
pub struct DnaSequence {
    bases: Vec<u8>,  // Encoded: A=1, C=2, G=3, T=4
}

impl DnaSequence {
    /// Parse from a string of ACGT characters. Returns Err on invalid characters.
    pub fn from_str(s: &str) -> Result<Self, Error>;
    pub fn len(&self) -> usize;
}

/// The FM-index, ready for queries.
pub struct FmIndex { /* fields from Section 3.5 */ }

impl FmIndex {
    /// Build an FM-index from a set of DNA sequences.
    /// Async because GPU operations require awaiting device/buffer operations.
    pub async fn build(sequences: &[DnaSequence], config: &FmIndexConfig) -> Result<Self, Error>;

    /// Build using CPU only (synchronous).
    pub fn build_cpu(sequences: &[DnaSequence], config: &FmIndexConfig) -> Result<Self, Error>;

    /// Count occurrences of a pattern.
    pub fn count(&self, pattern: &DnaSequence) -> u32;

    /// Locate all occurrences (returns text positions).
    pub fn locate(&self, pattern: &DnaSequence) -> Vec<u32>;

    /// Serialize to bytes (for caching/storage).
    pub fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error>;

    /// Total text length (including sentinels).
    pub fn text_len(&self) -> u32;

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32;
}
```

### 6.2 JavaScript/TypeScript API via wasm-bindgen

```rust
// --- wasm/js_api.rs ---

use wasm_bindgen::prelude::*;

/// JS-facing FM-index builder.
#[wasm_bindgen]
pub struct FmIndexBuilder {
    sequences: Vec<DnaSequence>,
    config: FmIndexConfig,
}

#[wasm_bindgen]
impl FmIndexBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self;

    /// Set SA sample rate.
    pub fn sa_sample_rate(self, rate: u32) -> Self;

    /// Add a DNA sequence (ACGT string). Throws on invalid characters.
    pub fn add_sequence(&mut self, seq: &str) -> Result<(), JsValue>;

    /// Add multiple sequences at once (newline-separated or FASTA format).
    pub fn add_fasta(&mut self, fasta: &str) -> Result<(), JsValue>;

    /// Build the FM-index. Returns a Promise that resolves to FmIndexHandle.
    /// Async because GPU operations are async in the browser.
    pub async fn build(&self) -> Result<FmIndexHandle, JsValue>;
}

/// JS-facing FM-index query handle.
#[wasm_bindgen]
pub struct FmIndexHandle {
    inner: FmIndex,
}

#[wasm_bindgen]
impl FmIndexHandle {
    /// Count occurrences of pattern (ACGT string).
    pub fn count(&self, pattern: &str) -> Result<u32, JsValue>;

    /// Locate all occurrences. Returns a Uint32Array of text positions.
    pub fn locate(&self, pattern: &str) -> Result<Vec<u32>, JsValue>;

    /// Serialize to Uint8Array for storage (IndexedDB, etc).
    pub fn serialize(&self) -> Result<Vec<u8>, JsValue>;

    /// Deserialize from Uint8Array.
    #[wasm_bindgen(js_name = "deserialize")]
    pub fn from_bytes(data: &[u8]) -> Result<FmIndexHandle, JsValue>;

    /// Get the total indexed text length.
    pub fn text_len(&self) -> u32;

    /// Get the number of indexed sequences.
    pub fn num_sequences(&self) -> u32;
}
```

**TypeScript usage (from the browser):**

```typescript
import init, { FmIndexBuilder } from './pkg/webgpu_fmidx.js';

await init();

const builder = new FmIndexBuilder();
builder.add_sequence("ACGTACGTACGT");
builder.add_sequence("TGCATGCATGCA");

const index = await builder.build();  // GPU-accelerated

console.log(index.count("ACGT"));     // => 3
console.log(index.locate("ACGT"));    // => Uint32Array [0, 4, 8] (example)

// Save to IndexedDB
const bytes = index.serialize();
localStorage.setItem("fmindex", bytes);

// Restore later
const restored = FmIndexHandle.deserialize(bytes);
```

### 6.3 Input/Output Design

**Input format support:**
- Raw ACGT strings (one sequence per call)
- FASTA format (parsed on CPU, header lines stripped)
- Future: FASTQ format

**Sequence concatenation:**
Multiple sequences are concatenated with `$` separators: `s1$s2$...$sn$`. Each `$` is a sentinel that sorts before all other characters. The final `$` ensures the last suffix is the sentinel.

When resolving locate results, the text position can be mapped back to the original sequence by binary searching a precomputed array of cumulative sequence lengths.

**Output:**
- `FmIndex` struct (in-memory, ready for queries)
- Serialized binary format (for persistence/transfer)
- Query results: `u32` count, `Vec<u32>` positions

## 7. Implementation Phases

### Phase 1: CPU-Only FM-Index (Correctness Baseline)

**Goal:** Working FM-index construction and query on CPU. No GPU, no WASM. This is the reference implementation against which all GPU results are validated.

**Steps:**

1. **Project scaffolding** (Files: `Cargo.toml`, `src/lib.rs`, `src/error.rs`)
   - Initialize Cargo project with feature flags
   - Define error types
   - Dependencies: none beyond std (maybe `thiserror` for errors)
   - Risk: Low
   - Estimated complexity: Small

2. **Alphabet encoding** (File: `src/alphabet.rs`)
   - `$`=0, A=1, C=2, G=3, T=4
   - Encode/decode functions
   - `DnaSequence` struct with validation
   - Risk: Low
   - Estimated complexity: Small

3. **CPU suffix array construction** (File: `src/suffix_array/cpu.rs`)
   - Implement prefix doubling algorithm (O(n log^2 n))
   - Use `Vec<u32>` for SA and ranks
   - Standard library sort for radix sort (or implement counting sort for small alphabets)
   - Risk: Medium (correctness of the doubling logic)
   - Estimated complexity: Medium

4. **CPU BWT construction** (File: `src/bwt/cpu.rs`)
   - Trivial gather: `BWT[i] = text[(SA[i] + n - 1) % n]`
   - Risk: Low
   - Estimated complexity: Small

5. **C array construction** (File: `src/c_array.rs`)
   - Single pass character count + exclusive prefix sum
   - Risk: Low
   - Estimated complexity: Small

6. **CPU Occ table construction** (File: `src/occ/cpu.rs`)
   - Build checkpoints and bitvectors
   - Implement `rank(c, i)` query
   - Risk: Low
   - Estimated complexity: Medium

7. **FM-index assembly and queries** (File: `src/fm_index/mod.rs`, `src/fm_index/query.rs`)
   - `FmIndex::build_cpu()`
   - `count()` and `locate()` via backward search
   - SA sampling for space-efficient locate
   - Risk: Medium (backward search correctness)
   - Estimated complexity: Medium

8. **Serialization** (File: `src/fm_index/serialize.rs`)
   - Binary serialization/deserialization of FmIndex
   - Use `bincode` or hand-rolled format for WASM compatibility
   - Risk: Low
   - Estimated complexity: Small

9. **Comprehensive tests** (Files: `tests/cpu_correctness.rs`, `tests/query_tests.rs`)
   - Test with known FM-index examples (e.g., "abracadabra$", "banana$", DNA strings)
   - Property: SA is a permutation of [0..n)
   - Property: SA is sorted (suffix at SA[i] < suffix at SA[i+1])
   - Property: BWT can reconstruct original text via inverse BWT
   - Property: count("") == n, count of any single char == its frequency
   - Property: locate results are correct positions in text
   - Risk: Low
   - Estimated complexity: Medium

**Deliverable:** `cargo test` passes, CPU FM-index builds correctly, count/locate work.

### Phase 2: GPU Infrastructure + Suffix Array Construction

**Goal:** GPU-accelerated suffix array construction using WebGPU (via wgpu). Native backend first (Vulkan/Metal), WASM later.

**Steps:**

1. **GPU context module** (File: `src/gpu/mod.rs`)
   - `GpuContext` struct: adapter, device, queue
   - Async initialization with required limits
   - Feature-gated behind `#[cfg(feature = "gpu")]`
   - Risk: Medium (device capability negotiation)
   - Estimated complexity: Medium

2. **GPU buffer helpers** (File: `src/gpu/buffers.rs`)
   - Upload `Vec<u32>` to GPU storage buffer
   - Download GPU buffer to `Vec<u32>`
   - Buffer creation with usage flags (STORAGE, COPY_SRC, COPY_DST, MAP_READ)
   - Staging buffer management for readback
   - Risk: Low
   - Estimated complexity: Medium

3. **GPU prefix sum** (Files: `src/gpu/prefix_sum.rs`, `shaders/prefix_sum.wgsl`)
   - Blelloch scan, hierarchical for large arrays
   - Test: compare with CPU prefix sum on random data
   - Risk: High (off-by-one errors, workgroup boundary handling)
   - Estimated complexity: High

4. **GPU radix sort** (Files: `src/gpu/radix_sort.rs`, `shaders/radix_sort_count.wgsl`, `shaders/radix_sort_scatter.wgsl`)
   - 8-bit radix, 4 passes for 32-bit keys
   - Key-value sort (key = rank, value = SA index)
   - Test: compare with CPU sort on random data
   - Risk: High (scatter correctness, histogram accuracy)
   - Estimated complexity: High

5. **GPU SA construction** (Files: `src/suffix_array/gpu.rs`, `shaders/sa_init_ranks.wgsl`, `shaders/sa_compare_pairs.wgsl`, `shaders/sa_update_ranks.wgsl`)
   - Prefix doubling using GPU radix sort
   - Per-iteration CPU-GPU sync for convergence check
   - Test: compare GPU SA with CPU SA on identical inputs
   - Risk: High (complex multi-stage pipeline)
   - Estimated complexity: High

6. **GPU histogram** (Files: `src/gpu/histogram.rs`, `shaders/histogram.wgsl`)
   - Parallel character frequency counting
   - Used as a building block for radix sort and C array
   - Risk: Medium
   - Estimated complexity: Medium

7. **Integration tests** (File: `tests/gpu_correctness.rs`, `tests/cpu_gpu_equivalence.rs`)
   - SA from GPU == SA from CPU for various inputs
   - Test with edge cases: single character repeated, all unique, empty, single character
   - Risk: Low
   - Estimated complexity: Medium

**Deliverable:** `cargo test --features gpu` passes, GPU SA matches CPU SA.

### Phase 3: GPU BWT + Occ Table Construction

**Goal:** Complete GPU FM-index construction pipeline.

**Steps:**

1. **GPU BWT construction** (Files: `src/bwt/gpu.rs`, `shaders/bwt_gather.wgsl`)
   - Parallel gather shader
   - Test: GPU BWT == CPU BWT
   - Risk: Low (simple shader)
   - Estimated complexity: Small

2. **GPU Occ table construction** (Files: `src/occ/gpu.rs`, `shaders/occ_scan.wgsl`)
   - Block count + bitvector construction shader
   - Per-character prefix sum over block counts
   - Test: GPU Occ == CPU Occ
   - Risk: Medium (multi-pass coordination)
   - Estimated complexity: Medium

3. **GPU FM-index build pipeline** (File: `src/fm_index/mod.rs`)
   - `FmIndex::build()` async method that orchestrates GPU SA -> GPU BWT -> GPU Occ -> CPU C array -> assemble
   - Fallback to CPU if GPU unavailable
   - Risk: Medium (async orchestration)
   - Estimated complexity: Medium

4. **Correctness tests** (File: `tests/gpu_correctness.rs`)
   - Full pipeline: GPU-built FM-index produces same count/locate results as CPU-built
   - Risk: Low
   - Estimated complexity: Medium

**Deliverable:** Complete GPU-accelerated FM-index construction, verified against CPU baseline.

### Phase 4: Optimization + Benchmarks

**Goal:** Performance tuning and benchmarking.

**Steps:**

1. **Benchmark suite** (Files: `benches/construction.rs`, `benches/query.rs`)
   - Criterion benchmarks for SA construction (CPU vs GPU, varying n from 1K to 128M)
   - Query benchmarks (varying pattern length, varying number of occurrences)
   - Risk: Low
   - Estimated complexity: Medium

2. **Radix sort optimization**
   - Tune elements-per-thread, workgroup size
   - Experiment with 4-bit vs 8-bit radix
   - Coalesce memory accesses
   - Risk: Medium
   - Estimated complexity: Medium

3. **Memory optimization**
   - Reduce buffer count by reusing buffers across stages
   - Pack text more tightly (3 bits per base, 10 bases per u32)
   - Reduce SA memory by using in-place rank update
   - Risk: Medium
   - Estimated complexity: Medium

4. **Large input handling**
   - Chunk-based processing for texts exceeding GPU buffer limits
   - Build SA in chunks, merge (like external sorting)
   - Risk: High (complex merge logic)
   - Estimated complexity: High

**Deliverable:** Benchmarks showing GPU speedup over CPU, optimized buffer usage.

### Phase 5: WASM Bindings + Browser Demo

**Goal:** Ship the WASM package with JS/TS bindings and a browser demo.

**Steps:**

1. **WASM bindings** (Files: `src/wasm/mod.rs`, `src/wasm/js_api.rs`)
   - `FmIndexBuilder` and `FmIndexHandle` wasm-bindgen classes
   - FASTA parsing
   - Error conversion to JsValue
   - Risk: Medium (wasm-bindgen async, memory management)
   - Estimated complexity: Medium

2. **wasm-pack configuration** (Files: `Cargo.toml`, `.cargo/config.toml`)
   - WASM target configuration
   - Optimize for size: `wasm-opt -Oz`
   - Feature flags for WASM vs native
   - Risk: Low
   - Estimated complexity: Small

3. **Browser demo** (Files: `web/index.html`, `web/main.ts`, `web/package.json`)
   - Simple page: paste DNA sequences, build index, search patterns
   - Show construction time, index size, query results
   - Risk: Low
   - Estimated complexity: Medium

4. **Browser-based WebGPU tests** (File: `web/tests/`)
   - Use `wasm-pack test --chrome` or Playwright
   - Verify GPU path works in actual browser
   - Risk: High (browser WebGPU availability, headless testing)
   - Estimated complexity: High

5. **Documentation** (Files: `README.md`, doc comments throughout)
   - API documentation
   - Architecture overview
   - Build instructions
   - Browser compatibility notes
   - Risk: Low
   - Estimated complexity: Medium

**Deliverable:** Published WASM package, working browser demo, comprehensive docs.

## 8. Testing Strategy

### 8.1 Unit Tests

| Module | Test File | Key Properties |
|--------|-----------|---------------|
| `alphabet` | `src/alphabet.rs` (inline) | Encode/decode roundtrip, reject invalid chars |
| `suffix_array::cpu` | `tests/cpu_correctness.rs` | SA is permutation, SA is sorted, known test vectors |
| `suffix_array::gpu` | `tests/gpu_correctness.rs` | GPU SA == CPU SA |
| `bwt::cpu` | `tests/cpu_correctness.rs` | BWT can reconstruct text (inverse BWT), known vectors |
| `bwt::gpu` | `tests/gpu_correctness.rs` | GPU BWT == CPU BWT |
| `occ::cpu` | `tests/cpu_correctness.rs` | Rank values match naive counting |
| `occ::gpu` | `tests/gpu_correctness.rs` | GPU Occ == CPU Occ |
| `c_array` | `tests/cpu_correctness.rs` | C values match cumulative histogram |
| `fm_index::query` | `tests/query_tests.rs` | Count matches brute-force search, locate positions are correct |
| `fm_index::serialize` | `tests/serialize_tests.rs` | Roundtrip: serialize -> deserialize -> identical queries |
| `gpu::prefix_sum` | `tests/gpu_correctness.rs` | GPU prefix sum == CPU prefix sum, edge cases (power of 2, non-power of 2, 1 element, 0 elements) |
| `gpu::radix_sort` | `tests/gpu_correctness.rs` | GPU sort == `sort_unstable` on same input |

### 8.2 Integration Tests

- **Full pipeline (CPU):** Build FM-index from FASTA-like input, run count and locate queries, verify against brute-force `str::find` / `str::matches`.
- **Full pipeline (GPU):** Same queries, same expected results, built via GPU path.
- **CPU/GPU equivalence:** For identical inputs, `FmIndex::build_cpu()` and `FmIndex::build()` produce identical data structures (byte-for-byte identical serialization).

### 8.3 Property-Based Tests

Use `proptest` or `quickcheck`:

```rust
// Property: SA is a valid permutation
proptest! {
    fn sa_is_permutation(text in "[ACGT]{1,10000}") {
        let sa = build_sa(&encode(&text));
        let n = sa.len();
        let mut sorted = sa.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
    }
}

// Property: SA is correctly sorted
proptest! {
    fn sa_is_sorted(text in "[ACGT]{1,10000}") {
        let encoded = encode(&text);
        let sa = build_sa(&encoded);
        for i in 1..sa.len() {
            assert!(suffix(&encoded, sa[i-1]) < suffix(&encoded, sa[i]));
        }
    }
}

// Property: count matches naive search
proptest! {
    fn count_matches_naive(
        text in "[ACGT]{1,1000}",
        pattern in "[ACGT]{1,10}"
    ) {
        let fm = build_fm_index(&text);
        let expected = text.matches(&pattern).count() as u32;
        assert_eq!(fm.count(&encode(&pattern)), expected);
    }
}

// Property: locate results are valid positions
proptest! {
    fn locate_results_valid(
        text in "[ACGT]{1,1000}",
        pattern in "[ACGT]{1,10}"
    ) {
        let fm = build_fm_index(&text);
        let positions = fm.locate(&encode(&pattern));
        for pos in &positions {
            assert_eq!(&text[*pos as usize..*pos as usize + pattern.len()], pattern);
        }
        assert_eq!(positions.len(), text.matches(&pattern).count());
    }
}
```

### 8.4 Benchmark Suite

```rust
// benches/construction.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_sa_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_array");
    for size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        let text = random_dna(size);
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            &text,
            |b, t| b.iter(|| build_sa_cpu(t)),
        );
        // GPU benchmark (requires async runtime)
        group.bench_with_input(
            BenchmarkId::new("gpu", size),
            &text,
            |b, t| b.iter(|| pollster::block_on(build_sa_gpu(t))),
        );
    }
    group.finish();
}
```

### 8.5 Browser-Based WebGPU Tests

Use `wasm-pack test --headless --chrome` with feature-gated test modules. These tests exercise the actual browser WebGPU path, which is critical because:
- wgpu's WebGPU backend behaves differently from native Vulkan/Metal
- Buffer mapping semantics differ (truly async in browser vs pseudo-async on native)
- Shader compilation may surface browser-specific WGSL validation issues

## 9. Dependencies

### 9.1 Rust Crates

```toml
[dependencies]
# Error handling
thiserror = "2"

# Serialization
bincode = "1"
serde = { version = "1", features = ["derive"] }

# GPU (optional, behind feature flag)
wgpu = { version = "24", optional = true }
bytemuck = { version = "1", features = ["derive"] }
pollster = { version = "0.4", optional = true }  # For blocking on async in native tests

# WASM (optional, behind feature flag)
wasm-bindgen = { version = "0.2", optional = true }
wasm-bindgen-futures = { version = "0.4", optional = true }
js-sys = { version = "0.3", optional = true }
web-sys = { version = "0.3", optional = true, features = ["console"] }

# Logging
log = "0.4"

[dev-dependencies]
# Testing
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }
tokio = { version = "1", features = ["rt", "macros"] }  # For async GPU tests
pollster = "0.4"
rand = "0.9"
```

### 9.2 Build Tooling

- `rustup target add wasm32-unknown-unknown`
- `cargo install wasm-pack`
- `cargo install wasm-bindgen-cli` (version must match `wasm-bindgen` crate)
- Node.js (for `web/` demo: `npm`, `npx`, TypeScript compiler)
- Chrome or Firefox Nightly (for browser WebGPU testing)

### 9.3 CI Pipeline

```yaml
# .github/workflows/ci.yml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - cargo test --features cpu

  test-gpu:
    runs-on: ubuntu-latest  # Needs GPU; use self-hosted runner or skip in CI
    steps:
      - cargo test --features gpu

  test-wasm:
    runs-on: ubuntu-latest
    steps:
      - wasm-pack build --target web --features wasm
      - wasm-pack test --headless --chrome --features wasm

  bench:
    runs-on: ubuntu-latest
    steps:
      - cargo bench --features cpu
```

## 10. Risks and Mitigations

### 10.1 WebGPU Browser Support

**Risk:** WebGPU is not yet universally supported. As of 2026, Chrome, Edge, and Firefox support it, but Safari support may be incomplete. Users on older browsers will have no GPU path.

**Mitigation:**
- Feature-detect WebGPU via `navigator.gpu` before attempting GPU build
- Graceful fallback to CPU path (`FmIndex::build_cpu()`) when GPU is unavailable
- Clear error messages indicating WebGPU is required for GPU acceleration
- The CPU path is always available as a baseline

### 10.2 GPU Memory Limits

**Risk:** WebGPU's `maxBufferSize` is typically 256MB-1GB depending on browser and hardware. The SA construction requires ~20-24 bytes per character, so:
- 10M bases -> ~240 MB (fits in most GPUs)
- 50M bases -> ~1.2 GB (may exceed limits on some hardware)
- 100M bases -> ~2.4 GB (exceeds most WebGPU limits)

**Mitigation:**
- Query device limits at initialization, compute maximum processable text length
- For texts exceeding GPU memory, fall back to CPU or implement chunk-based construction:
  - Divide text into overlapping chunks
  - Build partial SAs on GPU
  - Merge on CPU (O(n log n) merge)
- Report maximum supported text length to the user via API
- Optimize buffer usage: reuse buffers across pipeline stages (SA and BWT buffers can share memory since BWT is computed after SA is finalized)

### 10.3 Numerical Precision in Parallel Prefix Sums

**Risk:** Parallel prefix sum implementations can have off-by-one errors or boundary issues, especially at workgroup boundaries. A single incorrect prefix sum value corrupts the entire radix sort and cascades to an incorrect SA.

**Mitigation:**
- Exhaustive unit tests for prefix sum at various sizes (1, 2, 3, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, etc.)
- Test at exact workgroup boundaries and one-off boundaries
- Validate prefix sum against CPU reference for every test
- Use property test: `prefix_sum(data)[i] == data[0..=i].sum()` for all i
- Include a "validation mode" flag that checks intermediate GPU results against CPU during development

### 10.4 WASM Memory Model Constraints

**Risk:** WASM has a linear memory model with a default limit of ~4GB. The FM-index struct, BWT, Occ table, and sampled SA all live in WASM linear memory after GPU readback. For a 100M base text: BWT = 100MB, Occ checkpoints = ~30MB, SA samples = ~12MB = ~142MB. This is well within WASM limits, but combined with the text itself and intermediate buffers, memory pressure is real.

**Mitigation:**
- Drop GPU buffers immediately after readback (Rust's RAII handles this)
- Use `Vec::shrink_to_fit()` on readback buffers
- Consider streaming readback for very large results
- Document memory requirements in API docs

### 10.5 WGSL Shader Compatibility

**Risk:** Different WebGPU implementations may have slightly different WGSL validation rules or runtime behavior. Shader compilation errors at runtime would be fatal.

**Mitigation:**
- Use `naga` (wgpu's shader compiler) for offline WGSL validation during development
- Test shaders on all three backends (Vulkan, Metal, WebGPU/browser)
- Stick to core WGSL features; avoid extensions
- Include shader source as `include_str!()` in Rust, not as external files, to ensure they ship with the WASM binary

### 10.6 Async Complexity in WASM

**Risk:** All WebGPU operations are async in the browser. Buffer mapping requires `map_async` + `await`. The prefix doubling loop requires per-iteration GPU-to-CPU readback (convergence check), which means many async round-trips.

**Mitigation:**
- Design the build pipeline as fully async from the start (not bolted on later)
- Minimize convergence check reads: instead of reading back the full max rank, read a single u32 flag buffer
- Consider running a fixed number of iterations (e.g., ceil(log2(n))) without checking convergence, then verifying once at the end. For n <= 128M, this is at most 27 iterations, which is acceptable.
- Use `wasm-bindgen-futures` to bridge Rust futures to JS Promises

### 10.7 Performance Regression: GPU Slower Than CPU for Small Inputs

**Risk:** For small texts (< 10K bases), GPU overhead (buffer allocation, shader compilation, data transfer) may exceed the benefit of parallelism.

**Mitigation:**
- Add a size threshold: use CPU for texts below a configurable cutoff (default: 100K bases)
- Cache compiled compute pipelines in `GpuContext` so shader compilation is amortized
- Benchmark to determine the crossover point and set the threshold accordingly

## 11. File Path Summary

All paths are relative to `/home/sriramv/Projects/webgpu-fmidx/`:

**Core source files:**
- `/home/sriramv/Projects/webgpu-fmidx/Cargo.toml`
- `/home/sriramv/Projects/webgpu-fmidx/src/lib.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/error.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/alphabet.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/c_array.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/suffix_array/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/suffix_array/cpu.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/suffix_array/gpu.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/bwt/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/bwt/cpu.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/bwt/gpu.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/occ/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/occ/cpu.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/occ/gpu.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/fm_index/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/fm_index/query.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/fm_index/serialize.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/gpu/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/gpu/buffers.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/gpu/pipeline.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/gpu/prefix_sum.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/gpu/radix_sort.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/gpu/histogram.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/wasm/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/src/wasm/js_api.rs`

**Shaders:**
- `/home/sriramv/Projects/webgpu-fmidx/shaders/prefix_sum.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/radix_sort_count.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/radix_sort_scatter.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/sa_init_ranks.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/sa_compare_pairs.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/sa_update_ranks.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/bwt_gather.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/occ_scan.wgsl`
- `/home/sriramv/Projects/webgpu-fmidx/shaders/histogram.wgsl`

**Tests and benchmarks:**
- `/home/sriramv/Projects/webgpu-fmidx/tests/common/mod.rs`
- `/home/sriramv/Projects/webgpu-fmidx/tests/cpu_correctness.rs`
- `/home/sriramv/Projects/webgpu-fmidx/tests/gpu_correctness.rs`
- `/home/sriramv/Projects/webgpu-fmidx/tests/cpu_gpu_equivalence.rs`
- `/home/sriramv/Projects/webgpu-fmidx/tests/query_tests.rs`
- `/home/sriramv/Projects/webgpu-fmidx/tests/serialize_tests.rs`
- `/home/sriramv/Projects/webgpu-fmidx/benches/construction.rs`
- `/home/sriramv/Projects/webgpu-fmidx/benches/query.rs`

**Browser demo:**
- `/home/sriramv/Projects/webgpu-fmidx/web/index.html`
- `/home/sriramv/Projects/webgpu-fmidx/web/main.ts`
- `/home/sriramv/Projects/webgpu-fmidx/web/package.json`
- `/home/sriramv/Projects/webgpu-fmidx/web/tsconfig.json`

## 12. Success Criteria

- [ ] CPU FM-index construction produces correct suffix arrays for all test vectors
- [ ] CPU count() and locate() match brute-force search for all property tests
- [ ] GPU suffix array matches CPU suffix array for inputs up to 10M bases
- [ ] GPU BWT and Occ table match CPU equivalents
- [ ] Full GPU pipeline builds a valid FM-index (queries return correct results)
- [ ] GPU construction is faster than CPU for inputs > 100K bases
- [ ] WASM module builds and runs in Chrome with WebGPU enabled
- [ ] JS API can build an index from FASTA input and run queries
- [ ] Serialization roundtrip preserves index correctness
- [ ] Test coverage >= 80% on CPU path
- [ ] Benchmark suite demonstrates GPU speedup curve
- [ ] Graceful fallback when WebGPU is unavailable