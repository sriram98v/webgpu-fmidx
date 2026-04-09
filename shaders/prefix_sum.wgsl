// Blelloch-style exclusive prefix sum.
// Each workgroup processes BLOCK_SIZE (512) elements.
// For arrays larger than 512, a hierarchical approach is needed (managed from Rust side).

struct Params {
    n: u32,           // total number of elements
    block_offset: u32, // offset into the global array for this dispatch
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 512u; // 2 * WORKGROUP_SIZE

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) gid: vec3u,
) {
    let block_start = gid.x * BLOCK_SIZE;
    let idx0 = block_start + 2u * lid.x;
    let idx1 = block_start + 2u * lid.x + 1u;

    // Load into shared memory (zero-pad out-of-bounds)
    temp[2u * lid.x] = select(0u, data[idx0], idx0 < params.n);
    temp[2u * lid.x + 1u] = select(0u, data[idx1], idx1 < params.n);
    workgroupBarrier();

    // Up-sweep (reduce)
    var offset = 1u;
    for (var d = WORKGROUP_SIZE; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (lid.x < d) {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        offset <<= 1u;
    }

    // Save block sum and clear root
    if (lid.x == 0u) {
        block_sums[gid.x] = temp[BLOCK_SIZE - 1u];
        temp[BLOCK_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep
    for (var d = 1u; d < BLOCK_SIZE; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if (lid.x < d) {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Write back
    if (idx0 < params.n) {
        data[idx0] = temp[2u * lid.x];
    }
    if (idx1 < params.n) {
        data[idx1] = temp[2u * lid.x + 1u];
    }
}

// Add block sums to each element (second pass for hierarchical scan).
@compute @workgroup_size(256)
fn add_block_sums(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) gid: vec3u,
) {
    let block_start = gid.x * BLOCK_SIZE;
    let idx0 = block_start + 2u * lid.x;
    let idx1 = block_start + 2u * lid.x + 1u;
    let block_sum = block_sums[gid.x];

    if (idx0 < params.n) {
        data[idx0] += block_sum;
    }
    if (idx1 < params.n) {
        data[idx1] += block_sum;
    }
}
