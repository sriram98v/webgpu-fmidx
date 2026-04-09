// Gather rank pairs for radix sort key formation.
// For each i: key_primary[i] = ranks[sa[i]], key_secondary[i] = ranks[(sa[i] + h) % n]

struct Params {
    n: u32,
    h: u32,
}

@group(0) @binding(0) var<storage, read> sa: array<u32>;
@group(0) @binding(1) var<storage, read> ranks: array<u32>;
@group(0) @binding(2) var<storage, read_write> key_primary: array<u32>;
@group(0) @binding(3) var<storage, read_write> key_secondary: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn gather_keys(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let sa_i = sa[i];
    key_primary[i] = ranks[sa_i];
    key_secondary[i] = ranks[(sa_i + params.h) % params.n];
}
