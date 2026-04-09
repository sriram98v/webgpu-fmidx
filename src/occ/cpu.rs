//! CPU implementation of the Occ table (rank data structure) over the BWT.

use super::{OccTable, BLOCK_SIZE};
use crate::alphabet::ALPHABET_SIZE;
use crate::bwt::Bwt;

/// Build the Occ table from the BWT on the CPU.
///
/// Creates checkpoint + bitvector structure for O(1) rank queries.
pub fn build_occ_table(bwt: &Bwt) -> OccTable {
    let n = bwt.len() as u32;
    let num_blocks = n.div_ceil(BLOCK_SIZE) as usize;

    let mut checkpoints = Vec::with_capacity(num_blocks);
    let mut bitvectors = Vec::with_capacity(num_blocks);

    let mut cumulative = [0u32; ALPHABET_SIZE];

    for block_idx in 0..num_blocks {
        let start = block_idx as u32 * BLOCK_SIZE;
        let end = std::cmp::min(start + BLOCK_SIZE, n);

        // Store checkpoint: cumulative counts up to (but not including) this block
        checkpoints.push(cumulative);

        // Build bitvectors for this block
        let mut block_bits = [0u64; ALPHABET_SIZE];
        for pos in start..end {
            let ch = bwt.data[pos as usize] as usize;
            let bit_pos = (pos - start) as u64;
            if ch < ALPHABET_SIZE {
                block_bits[ch] |= 1u64 << bit_pos;
                cumulative[ch] += 1;
            }
        }
        bitvectors.push(block_bits);
    }

    OccTable {
        checkpoints,
        bitvectors,
        block_size: BLOCK_SIZE,
        text_len: n,
    }
}

/// Naive rank computation for testing: count occurrences of c in bwt[0..i).
pub fn naive_rank(bwt: &[u8], c: u8, i: u32) -> u32 {
    bwt[..i as usize].iter().filter(|&&ch| ch == c).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;
    use crate::bwt::cpu::build_bwt;
    use crate::suffix_array::cpu::build_suffix_array;

    fn encode(s: &str) -> Vec<u8> {
        let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
        if v.last() != Some(&SENTINEL) {
            v.push(SENTINEL);
        }
        v
    }

    #[test]
    fn test_occ_matches_naive() {
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                let expected = naive_rank(&bwt.data, c, i);
                let actual = occ.rank(c, i);
                assert_eq!(
                    actual, expected,
                    "Occ({}, {}) = {} but expected {}",
                    c, i, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_occ_long_text() {
        // Text longer than one block (>64 chars)
        let s = "ACGT".repeat(20); // 80 chars + sentinel = 81
        let text = encode(&s);
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                let expected = naive_rank(&bwt.data, c, i);
                let actual = occ.rank(c, i);
                assert_eq!(actual, expected, "Occ({}, {}) mismatch", c, i);
            }
        }
    }

    #[test]
    fn test_occ_boundary_values() {
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        // rank(c, 0) should always be 0
        for c in 0..ALPHABET_SIZE as u8 {
            assert_eq!(occ.rank(c, 0), 0);
        }

        // rank(c, n) should equal total count of c in bwt
        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            let total = bwt.data.iter().filter(|&&ch| ch == c).count() as u32;
            assert_eq!(occ.rank(c, n), total);
        }
    }
}
