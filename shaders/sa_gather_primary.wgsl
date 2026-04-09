// Gather primary keys: output[i] = ranks[indices[i]]
// Used to regather primary sort keys after sorting by secondary key.

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read>       indices: array<u32>;
@group(0) @binding(1) var<storage, read>       ranks:   array<u32>;
@group(0) @binding(2) var<storage, read_write> output:  array<u32>;
@group(0) @binding(3) var<uniform>             params:  Params;

@compute @workgroup_size(256)
fn gather_primary(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }
    output[i] = ranks[indices[i]];
}
