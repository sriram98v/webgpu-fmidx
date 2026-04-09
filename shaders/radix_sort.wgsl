// GPU Radix Sort: count and scatter passes for 8-bit LSD radix sort.
// Radix = 256 (8-bit digits), 4 passes for 32-bit keys.
//
// TILE_SIZE = WORKGROUP_SIZE: each thread handles exactly one element.
// This enables a fully parallel scatter pass where all 256 threads
// simultaneously compute their scatter destinations.

struct Params {
    n: u32,
    digit_shift: u32,   // 0, 8, 16, or 24
    num_workgroups: u32,
    _pad: u32,
}

const WORKGROUP_SIZE: u32 = 256u;
const ITEMS_PER_THREAD: u32 = 1u;
const TILE_SIZE: u32 = 256u;  // WORKGROUP_SIZE * ITEMS_PER_THREAD
const RADIX: u32 = 256u;

// ---- Count pass: build per-workgroup digit histograms ----

@group(0) @binding(0) var<storage, read>       keys_in:    array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>; // [num_workgroups * 256]
@group(0) @binding(2) var<uniform>             params:     Params;

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn count_digits(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id)        gid: vec3u,
) {
    // Zero local histogram
    atomicStore(&local_hist[lid.x], 0u);
    workgroupBarrier();

    // Each thread counts one element
    let idx = gid.x * TILE_SIZE + lid.x;
    if (idx < params.n) {
        let digit = (keys_in[idx] >> params.digit_shift) & 0xFFu;
        atomicAdd(&local_hist[digit], 1u);
    }
    workgroupBarrier();

    // Write local histogram to global memory
    // Layout: histograms[digit * num_workgroups + workgroup_id]
    histograms[lid.x * params.num_workgroups + gid.x] = atomicLoad(&local_hist[lid.x]);
}

// ---- Scatter pass: fully parallel stable scatter ----
//
// Each thread handles exactly one element (TILE_SIZE = WORKGROUP_SIZE).
// Stability: element at tile-index t with digit d gets local rank =
//   count of same-digit elements at tile-indices 0..t-1.
// This is computed by each thread scanning shared_digit[0..lid.x).
// Wall-clock cost: O(TILE_SIZE) per workgroup (not O(TILE_SIZE) per thread,
// since all threads run in parallel).

@group(0) @binding(0) var<storage, read>       scatter_keys_in:  array<u32>;
@group(0) @binding(1) var<storage, read>       scatter_vals_in:  array<u32>;
@group(0) @binding(2) var<storage, read_write> scatter_keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> scatter_vals_out: array<u32>;
@group(0) @binding(4) var<storage, read>       global_offsets:   array<u32>; // prefix-summed histograms
@group(0) @binding(5) var<uniform>             scatter_params:   Params;

var<workgroup> local_offsets: array<u32, 256>;
var<workgroup> shared_digit:  array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id)        gid: vec3u,
) {
    // Load per-digit starting offsets for this workgroup into shared memory.
    local_offsets[lid.x] = global_offsets[lid.x * scatter_params.num_workgroups + gid.x];

    // Load this thread's digit (RADIX = sentinel for out-of-bounds).
    let idx = gid.x * TILE_SIZE + lid.x;
    var my_digit = RADIX;
    if (idx < scatter_params.n) {
        my_digit = (scatter_keys_in[idx] >> scatter_params.digit_shift) & 0xFFu;
    }
    shared_digit[lid.x] = my_digit;
    workgroupBarrier();

    // Each in-bounds thread computes its local rank within this workgroup:
    // count of elements with the same digit at tile-relative positions 0..lid.x-1.
    if (my_digit < RADIX) {
        var local_rank = 0u;
        for (var s = 0u; s < lid.x; s++) {
            if (shared_digit[s] == my_digit) {
                local_rank += 1u;
            }
        }
        let dest = local_offsets[my_digit] + local_rank;
        scatter_keys_out[dest] = scatter_keys_in[idx];
        scatter_vals_out[dest] = scatter_vals_in[idx];
    }
}
