# Contributing to haystackfm

Thanks for your interest in improving haystackfm! This document covers how to build, test,
and submit changes.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

```bash
git clone https://github.com/sriram98v/haystackfm.git
cd haystackfm
cargo build            # CPU only (default features)
```

The Minimum Supported Rust Version (MSRV) is **1.87**. CI and releases target 1.87+.

### With Nix (optional)

A flake provides a pinned toolchain plus the WASM, docs, and Vulkan tooling, so you don't
have to install `wasm-bindgen-cli`, `mdbook`, or Node by hand:

```bash
nix develop            # dev shell: cargo, clippy, rustfmt, rust-analyzer,
                       # wasm32 target, wasm-bindgen, wasm-pack, mdbook, node
nix build              # build the crate (default `cpu` features) and run its tests
nix flake check        # same, as a check
```

If `nix` reports that `nix-command` or `flakes` is disabled, either add
`experimental-features = nix-command flakes` to `~/.config/nix/nix.conf`, or prefix commands
with `nix --extra-experimental-features 'nix-command flakes' …`.

The flake builds the crate with default (`cpu`) features only. GPU and WASM builds stay
manual — GPU tests need a real device, and `wasm-bindgen` must version-match the crate. The
dev shell warns on entry if the two disagree.

## Feature flags

The crate is split into additive feature flags — build and test the ones your change touches:

| Flag | Enables |
|------|---------|
| `cpu` (default) | CPU construction and queries |
| `gpu` | WebGPU-accelerated construction/queries via `wgpu` |
| `wasm` | `wasm-bindgen` JS/TS bindings (implies `gpu`) |

```bash
cargo build --features gpu
cargo build --features wasm
```

## Before you open a PR

Run the same checks CI enforces:

```bash
cargo fmt --all -- --check          # formatting
cargo clippy --all-features -- -D warnings   # lints (must be clean)
cargo test                          # CPU tests
cargo test --features gpu           # GPU tests (requires a physical GPU/WebGPU device)
```

- **GPU tests** are gated behind `#[cfg(feature = "gpu")]` and need a real device; CI runs the
  CPU suite only. If your change touches a GPU path, run the GPU tests locally and say so in the PR.
- **Parity:** every GPU query path has a CPU ground-truth test (see `gpu_*_matches_cpu` tests and
  `tests/gpu_mem_parity.rs`). New GPU kernels must add an equivalent CPU-vs-GPU test.
- **Benchmarks** use Criterion and are not run by `cargo test`:
  ```bash
  cargo bench --features gpu --bench locate_bench
  ```

## WASM / browser build

```bash
wasm-pack build --target web --features wasm
```

The `web/` directory holds a demo harness for in-browser WebGPU validation.

## Commit & PR conventions

- Follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `perf:`,
  `refactor:`, `docs:`, `test:`, `chore:`, `ci:`.
- Keep PRs focused; describe *what* changed and *why*, plus how you tested it.
- Update `CHANGELOG.md` under the `[Unreleased]` heading for user-visible changes.
- All PRs require review before merge (see [CODEOWNERS](.github/CODEOWNERS)).

## Reporting bugs / requesting features

Open an issue using the templates under `.github/ISSUE_TEMPLATE/`. For security-sensitive
reports, follow [SECURITY.md](SECURITY.md) instead of filing a public issue.
