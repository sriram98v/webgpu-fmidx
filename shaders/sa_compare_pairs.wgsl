// Compare adjacent sorted suffixes by their rank pairs.
// flags[i] = 1 if (key_primary[i], key_secondary[i]) != (key_primary[i-1], key_secondary[i-1])
// flags[0] = 1 always (first element is always a new group)

struct Params {
    n: u32,
    h: u32,
}

@group(0) @binding(0) var<storage, read> sa: array<u32>;
@group(0) @binding(1) var<storage, read> ranks: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compare_pairs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }

    if (i == 0u) {
        flags[i] = 1u;
        return;
    }

    let curr = sa[i];
    let prev = sa[i - 1u];

    let curr_primary = ranks[curr];
    let prev_primary = ranks[prev];
    let curr_secondary = ranks[(curr + params.h) % params.n];
    let prev_secondary = ranks[(prev + params.h) % params.n];

    flags[i] = select(0u, 1u,
        curr_primary != prev_primary || curr_secondary != prev_secondary);
}
