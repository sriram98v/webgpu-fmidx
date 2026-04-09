use crate::gpu::prefix_sum::PrefixSumPipelines;
use crate::gpu::radix_sort::{RadixSortPipelines, SortResult};
use crate::gpu::GpuContext;

use super::SuffixArray;

const INIT_SHADER: &str = include_str!("../../shaders/sa_init_ranks.wgsl");
const GATHER_SHADER: &str = include_str!("../../shaders/sa_gather_keys.wgsl");
const GATHER_PRIMARY_SHADER: &str = include_str!("../../shaders/sa_gather_primary.wgsl");
const COMPARE_SHADER: &str = include_str!("../../shaders/sa_compare_pairs.wgsl");
const UPDATE_SHADER: &str = include_str!("../../shaders/sa_update_ranks.wgsl");

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InitParams {
    n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    n: u32,
    h: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompareParams {
    n: u32,
    h: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateParams {
    n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherPrimaryParams {
    n: u32,
}

/// Cached pipelines for GPU suffix array construction.
pub struct SaPipelines {
    init_pipeline: wgpu::ComputePipeline,
    gather_pipeline: wgpu::ComputePipeline,
    gather_primary_pipeline: wgpu::ComputePipeline,
    compare_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    max_pipeline: wgpu::ComputePipeline,
    prefix_sum: PrefixSumPipelines,
    radix_sort: RadixSortPipelines,
}

impl SaPipelines {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            init_pipeline: ctx.create_compute_pipeline("sa_init", INIT_SHADER, "init_ranks"),
            gather_pipeline: ctx.create_compute_pipeline("sa_gather", GATHER_SHADER, "gather_keys"),
            gather_primary_pipeline: ctx.create_compute_pipeline(
                "sa_gather_primary",
                GATHER_PRIMARY_SHADER,
                "gather_primary",
            ),
            compare_pipeline: ctx.create_compute_pipeline(
                "sa_compare",
                COMPARE_SHADER,
                "compare_pairs",
            ),
            update_pipeline: ctx.create_compute_pipeline(
                "sa_update",
                UPDATE_SHADER,
                "update_ranks",
            ),
            max_pipeline: ctx.create_compute_pipeline("sa_max", UPDATE_SHADER, "parallel_max"),
            prefix_sum: PrefixSumPipelines::new(ctx),
            radix_sort: RadixSortPipelines::new(ctx),
        }
    }

    /// Build the suffix array on the GPU using prefix doubling.
    pub async fn build_suffix_array(&self, ctx: &GpuContext, text: &[u8]) -> SuffixArray {
        let n = text.len() as u32;
        if n <= 1 {
            return SuffixArray {
                data: (0..n).collect(),
            };
        }

        let num_wg = GpuContext::workgroup_count(n, WG_SIZE);

        // Upload text as u32 array (one char per u32 for simplicity)
        let text_u32: Vec<u32> = text.iter().map(|&b| b as u32).collect();
        let text_buf = ctx.create_buffer_init("text", &text_u32);

        // Allocate buffers
        let sa_buf = ctx.create_buffer_empty("sa", n);
        let ranks_buf = ctx.create_buffer_empty("ranks", n);
        let new_ranks_buf = ctx.create_buffer_empty("new_ranks", n);
        let flags_buf = ctx.create_buffer_empty("flags", n);

        // Radix sort double buffers
        let keys_a_buf = ctx.create_buffer_empty("keys_a", n);
        let vals_a_buf = ctx.create_buffer_empty("vals_a", n);
        let keys_b_buf = ctx.create_buffer_empty("keys_b", n);
        let vals_b_buf = ctx.create_buffer_empty("vals_b", n);

        // Max reduction result
        let max_buf = ctx.create_buffer_init("max_result", &[0u32]);

        // Separate buffer for prefix sums (keeps flags_buf intact for update_ranks)
        let prefix_sums_buf = ctx.create_buffer_empty("prefix_sums", n);

        // Step 1: Initialize SA and ranks
        let init_params = InitParams { n };
        let init_params_buf = ctx.create_uniform_buffer("init_params", &init_params);
        let init_bg = ctx.create_bind_group(
            &self.init_pipeline,
            0,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: text_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ranks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: init_params_buf.as_entire_binding(),
                },
            ],
        );
        ctx.dispatch(&self.init_pipeline, &init_bg, (num_wg, 1, 1));

        // Step 2: Prefix doubling iterations
        let mut h = 1u32;
        let max_iterations = 32u32; // Safety limit (log2(4G) = 32)

        for _iter in 0..max_iterations {
            // 2a: Gather sort keys: key_primary[i] = ranks[sa[i]], key_secondary[i] = ranks[(sa[i]+h)%n]
            let gather_params = GatherParams { n, h };
            let gather_params_buf = ctx.create_uniform_buffer("gather_params", &gather_params);
            let gather_bg = ctx.create_bind_group(
                &self.gather_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ranks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: gather_params_buf.as_entire_binding(),
                    },
                ],
            );
            ctx.dispatch(&self.gather_pipeline, &gather_bg, (num_wg, 1, 1));

            // Copy SA to vals_a for radix sort (SA indices are the values)
            self.copy_buffer(ctx, &sa_buf, &vals_a_buf, n);

            // 2b: Stable sort by secondary key first, then primary key
            // Sort by secondary key (keys_b holds secondary keys, use keys_a as temp)
            // We need to sort vals_a by keys_b, then sort the result by keys_a.
            // Approach: pack both keys as secondary sort, then primary sort.

            // First sort: by secondary key (key_secondary is in keys_b)
            let result1 = self.radix_sort.sort(
                ctx,
                &self.prefix_sum,
                &keys_b_buf,
                &vals_a_buf,
                &keys_a_buf, // reuse keys_a as temp output for keys
                &vals_b_buf,
                n,
            );

            // After sort, result is either in (keys_b, vals_a) or (keys_a, vals_b)
            // We need the vals for next sort and the primary keys.
            // Regather primary keys in sorted-by-secondary order.

            let (sorted_vals_sec, _) = match result1 {
                SortResult::InA => (&vals_a_buf, &keys_b_buf),
                SortResult::InB => (&vals_b_buf, &keys_a_buf),
            };

            // Regather primary keys from the SA values sorted by secondary
            // key_primary_sorted[i] = ranks[sorted_vals[i]]
            self.gather_primary_keys(ctx, sorted_vals_sec, &ranks_buf, &keys_a_buf, n);

            // Now we need to copy sorted_vals_sec to vals_a if they're in vals_b
            if result1 == SortResult::InB {
                self.copy_buffer(ctx, &vals_b_buf, &vals_a_buf, n);
            }

            // Second sort: by primary key
            let result2 = self.radix_sort.sort(
                ctx,
                &self.prefix_sum,
                &keys_a_buf,
                &vals_a_buf,
                &keys_b_buf,
                &vals_b_buf,
                n,
            );

            // Copy sorted SA back
            let final_vals = match result2 {
                SortResult::InA => &vals_a_buf,
                SortResult::InB => &vals_b_buf,
            };
            self.copy_buffer(ctx, final_vals, &sa_buf, n);

            // 2c: Compare adjacent pairs
            let compare_params = CompareParams { n, h };
            let compare_params_buf = ctx.create_uniform_buffer("compare_params", &compare_params);
            let compare_bg = ctx.create_bind_group(
                &self.compare_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ranks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: flags_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: compare_params_buf.as_entire_binding(),
                    },
                ],
            );
            ctx.dispatch(&self.compare_pipeline, &compare_bg, (num_wg, 1, 1));

            // 2d: Copy flags → prefix_sums_buf, then exclusive-prefix-sum that copy.
            // flags_buf stays intact (original 0/1 flags needed by update_ranks).
            self.copy_buffer(ctx, &flags_buf, &prefix_sums_buf, n);
            self.prefix_sum
                .exclusive_prefix_sum(ctx, &prefix_sums_buf, n);

            // 2e: Update ranks: new_ranks[sa[i]] = prefix_sums[i] + flags[i] - 1
            //     (= inclusive prefix sum of flags, minus 1 → 0-based group index)
            let update_params = UpdateParams { n };
            let update_params_buf = ctx.create_uniform_buffer("update_params", &update_params);
            let update_bg = ctx.create_bind_group(
                &self.update_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prefix_sums_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: new_ranks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: update_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: flags_buf.as_entire_binding(),
                    },
                ],
            );
            ctx.dispatch(&self.update_pipeline, &update_bg, (num_wg, 1, 1));

            // 2f: Check convergence: is max(new_ranks) == n - 1?
            ctx.upload_to_buffer(&max_buf, &[0u32]);
            let max_params = UpdateParams { n };
            let max_params_buf = ctx.create_uniform_buffer("max_params", &max_params);
            let max_bg = ctx.create_bind_group(
                &self.max_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: new_ranks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: max_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: max_params_buf.as_entire_binding(),
                    },
                ],
            );
            ctx.dispatch(&self.max_pipeline, &max_bg, (num_wg, 1, 1));

            let max_rank = ctx.download_single(&max_buf).await;

            // Swap ranks and new_ranks (copy new_ranks -> ranks)
            self.copy_buffer(ctx, &new_ranks_buf, &ranks_buf, n);

            if max_rank >= n - 1 {
                break;
            }

            h *= 2;
            if h >= n {
                break;
            }
        }

        // Download final SA
        let sa_data = ctx.download_buffer(&sa_buf, n).await;
        SuffixArray { data: sa_data }
    }

    /// Copy one buffer to another via a command encoder.
    fn copy_buffer(&self, ctx: &GpuContext, src: &wgpu::Buffer, dst: &wgpu::Buffer, count: u32) {
        let size = count as u64 * 4;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Gather primary keys on the GPU: output[i] = ranks[indices[i]]
    fn gather_primary_keys(
        &self,
        ctx: &GpuContext,
        indices: &wgpu::Buffer,
        ranks: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) {
        let params = GatherPrimaryParams { n };
        let params_buf = ctx.create_uniform_buffer("gather_primary_params", &params);
        let bg = ctx.create_bind_group(
            &self.gather_primary_pipeline,
            0,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ranks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        );
        let num_wg = GpuContext::workgroup_count(n, WG_SIZE);
        ctx.dispatch(&self.gather_primary_pipeline, &bg, (num_wg, 1, 1));
    }
}
