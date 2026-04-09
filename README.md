# webgpu-fmidx

[![Built with Anthropic](https://img.shields.io/badge/Built%20with-Anthropic-191919?logo=anthropic&logoColor=white)](https://www.anthropic.com)

GPU-accelerated FM-index construction for DNA sequences, targeting WebGPU via `wgpu`.

Builds a full FM-index (suffix array → BWT → C array → Occ table) on the GPU using
compute shaders written in WGSL. Compiles to native (Vulkan / Metal / DX12) and to
WebAssembly (browser WebGPU).

## Features

- **GPU-accelerated construction** — radix sort, prefix sum, and gather shaders
- **CPU fallback** — identical results, no GPU required
- **WASM target** — runs entirely in the browser via `wasm-pack`
- **FM-index queries** — `count(pattern)` in O(m), `locate(pattern)` in O(m + occ·k)
- **Serialization** — binary round-trip with `bincode`
- **DNA alphabet** — A, C, G, T (+ sentinel `$`); rejects invalid characters early

## Quick Start

### Native (CPU only)

```bash
cargo build
cargo test
```

### Native (GPU — Vulkan / Metal / DX12)

```bash
cargo build --features gpu
cargo test --features gpu
```

### WebAssembly (browser WebGPU)

Prerequisites: `wasm-bindgen-cli`, `node`, `npm`.

```bash
# Install wasm-bindgen-cli (once, must match Cargo.toml version)
cargo install wasm-bindgen-cli --version 0.2.117

# 1. Build the WASM package
cargo build --target wasm32-unknown-unknown --features wasm --release
wasm-bindgen target/wasm32-unknown-unknown/release/webgpu_fmidx.wasm \
  --target web --out-dir web/pkg

# 2. Install JS dependencies
cd web && npm install

# 3. Start dev server
npm run dev
```

Or use the convenience script: `cd web && npm run build-wasm`.

Open `http://localhost:5173` in a WebGPU-capable browser (Chrome 113+, Edge 113+).

## Rust API

```rust
use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};

let sequences = vec![
    DnaSequence::from_str("ACGTACGTACGT")?,
    DnaSequence::from_str("TGCATGCATGCA")?,
];

// CPU build (sync)
let config = FmIndexConfig::default();
let index = FmIndex::build_cpu(&sequences, &config)?;

// GPU build (async, requires `gpu` feature)
#[cfg(feature = "gpu")]
let index = FmIndex::build(&sequences, &config).await?;

// Count occurrences
let count = index.count(&[1, 2, 3, 4]); // encoded ACGT

// Locate positions
let positions = index.locate(&[1, 2, 3, 4]);

// Serialize / deserialize
let bytes = index.to_bytes()?;
let restored = FmIndex::from_bytes(&bytes)?;
```

## JavaScript / TypeScript API

```typescript
import init, { FmIndexBuilder, FmIndexHandle } from "./pkg/webgpu_fmidx.js";

await init(); // load WASM

const builder = new FmIndexBuilder(32); // sa_sample_rate = 32
builder.add_fasta(`>seq1\nACGTACGT\n>seq2\nTGCATGCA`);

// GPU build (async)
const handle = await builder.build_gpu();
// or CPU: const handle = builder.build_cpu();

console.log(handle.count("ACGT"));       // number of occurrences
console.log(handle.locate("ACGT"));      // Uint32Array of positions
console.log(handle.text_len());          // total text length
console.log(handle.num_sequences());     // number of indexed sequences

// Serialize
const bytes = handle.to_bytes();         // Uint8Array
const restored = FmIndexHandle.from_bytes(bytes);
```

## Feature Flags

| Flag   | Enables                              | Implies  |
|--------|--------------------------------------|----------|
| `cpu`  | CPU implementations (default)        | —        |
| `gpu`  | GPU implementations via `wgpu`       | —        |
| `wasm` | WASM bindings via `wasm-bindgen`     | `gpu`    |

## Architecture

```
Construction pipeline
  ┌─────────────────────────────────┐
  │  Input: [DnaSequence]           │
  │  → concatenate + add sentinels  │
  │  → GPU: build suffix array      │  radix_sort.wgsl + sa_*.wgsl
  │  → GPU: derive BWT              │  bwt_gather.wgsl
  │  → CPU: compute C array         │  (trivial histogram, 5 values)
  │  → GPU: build Occ table         │  occ_scan.wgsl + prefix_sum.wgsl
  │  → sample SA at rate k          │
  └─────────────────────────────────┘

Query pipeline (CPU, O(m) count / O(m + occ·k) locate)
  backward_search → SA interval [lo, hi)
  locate: LF-walk from each interval position to nearest SA sample
```

## Browser Compatibility

| Browser       | Version | WebGPU | Notes                         |
|---------------|---------|--------|-------------------------------|
| Chrome        | 113+    | Yes    | Stable WebGPU support         |
| Edge          | 113+    | Yes    | Same as Chrome                |
| Firefox       | —       | Flag   | Behind `dom.webgpu.enabled`   |
| Safari        | 18+     | Yes    | macOS / iOS 18+               |

The CPU path (`builder.build_cpu()`) works in all browsers that support WASM.

## Benchmarks

Run with Criterion:

```bash
cargo bench --features gpu
```

Benchmarks cover:
- `build_cpu` / `build_gpu` at 1 K, 10 K, 100 K, 1 M characters
- `cpu_vs_gpu` — direct comparison at same sizes
- `query_count` / `query_locate` — query throughput at varying pattern lengths

## License

MIT
