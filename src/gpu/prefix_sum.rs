use super::GpuContext;

const SHADER_SOURCE: &str = include_str!("../../shaders/prefix_sum.wgsl");
const BLOCK_SIZE: u32 = 512; // 2 * WORKGROUP_SIZE (256)

/// Parameters uniform for prefix sum shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    block_offset: u32,
}

/// Cached compute pipelines for prefix sum.
pub struct PrefixSumPipelines {
    scan_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
}

impl PrefixSumPipelines {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            scan_pipeline: ctx.create_compute_pipeline(
                "prefix_sum_scan",
                SHADER_SOURCE,
                "scan_blocks",
            ),
            add_pipeline: ctx.create_compute_pipeline(
                "prefix_sum_add",
                SHADER_SOURCE,
                "add_block_sums",
            ),
        }
    }

    /// Perform an exclusive prefix sum on the data buffer (in-place).
    /// `n` is the number of u32 elements in the buffer.
    pub fn exclusive_prefix_sum(&self, ctx: &GpuContext, data_buf: &wgpu::Buffer, n: u32) {
        if n <= 1 {
            // 0 or 1 elements: exclusive prefix sum is all zeros
            if n == 1 {
                ctx.upload_to_buffer(data_buf, &[0]);
            }
            return;
        }

        let num_blocks = GpuContext::workgroup_count(n, BLOCK_SIZE);

        // Create block sums buffer
        let block_sums_buf = ctx.create_buffer_empty("block_sums", num_blocks.max(1));

        let params = Params { n, block_offset: 0 };
        let params_buf = ctx.create_uniform_buffer("prefix_sum_params", &params);

        // Pass 1: scan each block, write block sums
        let bind_group = ctx.create_bind_group(
            &self.scan_pipeline,
            0,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_sums_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        );

        ctx.dispatch(&self.scan_pipeline, &bind_group, (num_blocks, 1, 1));

        // If more than one block, recursively scan block sums, then add back
        if num_blocks > 1 {
            // Recursive prefix sum on block_sums
            self.exclusive_prefix_sum(ctx, &block_sums_buf, num_blocks);

            // Pass 2: add block sums to each element
            let add_bind_group = ctx.create_bind_group(
                &self.add_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: block_sums_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            );

            ctx.dispatch(&self.add_pipeline, &add_bind_group, (num_blocks, 1, 1));
        }
    }
}
