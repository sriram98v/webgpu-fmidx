pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

use crate::alphabet::ALPHABET_SIZE;

/// Block size for Occ table checkpoints.
pub const BLOCK_SIZE: u32 = 64;

/// Occ table: supports O(1) rank queries over the BWT.
///
/// Structure:
/// - Checkpoints every BLOCK_SIZE positions: cumulative count of each character.
/// - Bitvectors per block per character: which positions in the block have that character.
///
/// Rank query: Occ(c, i) = checkpoint[block][c] + popcount(bitvector[block][c] & mask(offset))
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OccTable {
    /// checkpoints[block][char] = cumulative count of char in bwt[0..block*BLOCK_SIZE)
    pub checkpoints: Vec<[u32; ALPHABET_SIZE]>,
    /// bitvectors[block][char] = 64-bit vector: bit j is set if bwt[block*BLOCK_SIZE + j] == char
    pub bitvectors: Vec<[u64; ALPHABET_SIZE]>,
    pub block_size: u32,
    pub text_len: u32,
}

impl OccTable {
    /// Rank query: count of character `c` in bwt[0..i).
    ///
    /// Occ(c, i) = number of occurrences of c in bwt[0], bwt[1], ..., bwt[i-1].
    pub fn rank(&self, c: u8, i: u32) -> u32 {
        if i == 0 {
            return 0;
        }
        let c_idx = c as usize;
        let block = ((i - 1) / self.block_size) as usize;
        let offset = (i - 1) % self.block_size;

        let checkpoint = self.checkpoints[block][c_idx];
        let bitvec = self.bitvectors[block][c_idx];

        // Count bits set in positions 0..=offset
        let mask = if offset == 63 {
            u64::MAX
        } else {
            (1u64 << (offset + 1)) - 1
        };

        checkpoint + (bitvec & mask).count_ones()
    }
}
