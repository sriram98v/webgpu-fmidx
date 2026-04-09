//! CPU implementation of suffix array construction (prefix-doubling algorithm).

use super::SuffixArray;

/// Build a suffix array using the prefix doubling algorithm.
///
/// Time: O(n log^2 n) with counting sort per iteration.
/// Space: O(n) auxiliary.
pub fn build_suffix_array(text: &[u8]) -> SuffixArray {
    let n = text.len();
    if n == 0 {
        return SuffixArray { data: vec![] };
    }
    if n == 1 {
        return SuffixArray { data: vec![0] };
    }

    let n32 = n as u32;

    // SA[i] = index of the i-th suffix in sorted order
    let mut sa: Vec<u32> = (0..n32).collect();

    // rank[i] = rank of suffix starting at position i
    let mut rank: Vec<u32> = vec![0; n];
    for i in 0..n {
        rank[i] = text[i] as u32;
    }

    let mut new_rank = vec![0u32; n];
    let mut h: u32 = 1;

    loop {
        // Sort SA by (rank[sa[i]], rank[(sa[i]+h) % n])
        // Use radix sort: first by secondary key, then by primary key (stable)
        let max_rank = *rank.iter().max().unwrap() + 1;

        // Sort by secondary key: rank[(sa[i] + h) % n]
        sa = counting_sort_by(&sa, max_rank as usize, |&idx| {
            rank[((idx as u64 + h as u64) % n as u64) as usize]
        });

        // Sort by primary key: rank[sa[i]] (stable sort preserves secondary ordering)
        sa = counting_sort_by(&sa, max_rank as usize, |&idx| rank[idx as usize]);

        // Compute new ranks based on sorted order
        new_rank[sa[0] as usize] = 0;
        for i in 1..n {
            let prev = sa[i - 1] as usize;
            let curr = sa[i] as usize;
            let prev_secondary = ((prev as u64 + h as u64) % n as u64) as usize;
            let curr_secondary = ((curr as u64 + h as u64) % n as u64) as usize;

            if rank[curr] == rank[prev] && rank[curr_secondary] == rank[prev_secondary] {
                new_rank[curr] = new_rank[prev];
            } else {
                new_rank[curr] = new_rank[prev] + 1;
            }
        }

        std::mem::swap(&mut rank, &mut new_rank);

        // Check if all ranks are unique
        let max_r = *rank.iter().max().unwrap();
        if max_r == n32 - 1 {
            break;
        }

        h *= 2;
        if h >= n32 {
            break;
        }
    }

    SuffixArray { data: sa }
}

/// Stable counting sort of `items` by the key extracted via `key_fn`.
/// `max_key` is an exclusive upper bound on key values.
fn counting_sort_by<F>(items: &[u32], max_key: usize, key_fn: F) -> Vec<u32>
where
    F: Fn(&u32) -> u32,
{
    let n = items.len();
    let mut counts = vec![0usize; max_key + 1];

    for item in items {
        counts[key_fn(item) as usize] += 1;
    }

    // Exclusive prefix sum -> starting positions
    let mut offsets = vec![0usize; max_key + 1];
    for i in 1..=max_key {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }

    let mut output = vec![0u32; n];
    for item in items {
        let key = key_fn(item) as usize;
        output[offsets[key]] = *item;
        offsets[key] += 1;
    }

    output
}

/// Simple O(n log n) SA construction using standard library sort.
/// Used as a reference for testing.
pub fn build_suffix_array_naive(text: &[u8]) -> SuffixArray {
    let n = text.len();
    let mut sa: Vec<u32> = (0..n as u32).collect();
    sa.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
    SuffixArray { data: sa }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;

    fn encode(s: &str) -> Vec<u8> {
        let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
        // Append sentinel if not present
        if v.last() != Some(&SENTINEL) {
            v.push(SENTINEL);
        }
        v
    }

    #[test]
    fn test_banana() {
        // "banana$" using our alphabet: b->1(A), a->1(A), n->3(G)...
        // Let's use a DNA-like string instead
        // Use "ACAC$"
        let text = encode("ACAC");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(
            sa.data, naive.data,
            "prefix doubling SA must match naive SA"
        );
    }

    #[test]
    fn test_single_char() {
        let text = encode("A");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_all_same() {
        let text = encode("AAAA");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_all_different() {
        let text = encode("ACGT");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_longer_sequence() {
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_sa_is_permutation() {
        let text = encode("ACGTTAGCCA");
        let sa = build_suffix_array(&text);
        let n = sa.len();
        let mut sorted = sa.data.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_sa_is_sorted() {
        let text = encode("ACGTTAGCCA");
        let sa = build_suffix_array(&text);
        for i in 1..sa.len() {
            let a = sa.data[i - 1] as usize;
            let b = sa.data[i] as usize;
            assert!(
                text[a..] < text[b..],
                "suffix at {} should be < suffix at {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_repetitive() {
        let text = encode("ACACACACAC");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }
}
