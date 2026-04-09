// Update ranks from prefix-summed flags.
//
// Correct formula: new_rank[sa[i]] = inclusive_prefix_sum(flags)[i] - 1
//                                   = exclusive_prefix_sum(flags)[i] + flags[i] - 1
//
// Using exclusive PS alone (prefix_sums[i]) gives WRONG ranks because equal
// adjacent pairs (flags[i]=0) inherit the accumulated count from distinct pairs
// before them, making them appear in different rank groups.
//
// Example: flags=[1,1,1,0,0]
//   exclusive PS = [0,1,2,3,3]  ← 3 and 4 both get 3, but 2 gets 2 → wrong
//   inclusive PS = [1,2,3,3,3], inclusive-1 = [0,1,2,2,2]  ← correct

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> sa: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_sums: array<u32>;  // exclusive PS of flags
@group(0) @binding(2) var<storage, read_write> new_ranks: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> flags: array<u32>;  // original (pre-PS) flags

@compute @workgroup_size(256)
fn update_ranks(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.n) { return; }

    // inclusive_PS[i] - 1 = exclusive_PS[i] + flags[i] - 1
    new_ranks[sa[i]] = prefix_sums[i] + flags[i] - 1u;
}

// Parallel max reduction to find the maximum rank (for convergence check).
// Writes the max value to result[0].

@group(0) @binding(0) var<storage, read> reduce_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> reduce_params: Params;

var<workgroup> shared_max: array<u32, 256>;

@compute @workgroup_size(256)
fn parallel_max(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(global_invocation_id) gid: vec3u,
) {
    // Load
    shared_max[lid.x] = select(0u, reduce_data[gid.x], gid.x < reduce_params.n);
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + stride]);
        }
        workgroupBarrier();
    }

    // Write workgroup max to global result via atomic max
    if (lid.x == 0u) {
        atomicMax(&result[0], shared_max[0]);
    }
}
