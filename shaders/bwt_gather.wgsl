// BWT gather: BWT[i] = text[(SA[i] - 1 + n) % n]
// Both text and BWT are stored as u32 arrays (one character per u32).

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read>       text:       array<u32>;
@group(0) @binding(1) var<storage, read>       sa:         array<u32>;
@group(0) @binding(2) var<storage, read_write> bwt:        array<u32>;
@group(0) @binding(3) var<uniform>             params:     Params;

@compute @workgroup_size(256)
fn bwt_gather(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let sa_i = sa[i];
    // Predecessor position: (SA[i] - 1 + n) % n
    let text_pos = select(sa_i - 1u, params.n - 1u, sa_i == 0u);
    bwt[i] = text[text_pos];
}
