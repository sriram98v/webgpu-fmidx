use crate::fm_index::seq_id::SeqId;

/// A Maximal Exact Match with GPU-resolved reference positions.
///
/// Returned by [`crate::BidirFmIndex::find_smems_gpu`] and [`crate::BidirFmIndex::find_mems_gpu`].
/// Positions are resolved via the GPU SA resolve + reference boundary mapping passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemHit {
    /// Start position in the query (0-based, inclusive).
    pub query_start: u32,
    /// End position in the query (0-based, exclusive).
    pub query_end: u32,
    /// Number of occurrences in the reference (SA interval size).
    pub match_count: u32,
    /// Reference positions: `(sequence_id, offset_within_ref)`.
    /// At most `max_hits_per_mem` entries (default 1024).
    /// Same shape as [`Mem::positions`](crate::Mem::positions) on the CPU path, so CPU and
    /// GPU results can be consumed by the same code.
    pub positions: Vec<(SeqId, u32)>,
    /// True if the hit list was truncated due to `max_hits_per_mem`.
    pub truncated: bool,
}
