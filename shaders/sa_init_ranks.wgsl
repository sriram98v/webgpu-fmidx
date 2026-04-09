// Initialize suffix array and ranks for prefix doubling.
// SA[i] = i (identity permutation)
// ranks[i] = text[i] (character code as initial rank)

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> text: array<u32>;
@group(0) @binding(1) var<storage, read_write> sa: array<u32>;
@group(0) @binding(2) var<storage, read_write> ranks: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn init_ranks(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }
    sa[i] = i;
    ranks[i] = text[i];
}
