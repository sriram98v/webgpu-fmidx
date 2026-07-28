//! End-to-end check that `find_mems` enumerates occurrence-level MEMs.
//!
//! Two references share a query prefix, but only one extends through the whole query. Under
//! whole-set maximality the longer reference's occurrences veto every shorter interval and
//! `find_mems` collapses onto `find_smems`; per occurrence, four more MEMs are real.
//!
//! The brute-force enumerator here is deliberately independent of the index — it walks every
//! (query position, reference position) pair — so it is a genuine cross-check rather than a
//! restatement of the implementation.
//!
//! Run with `cargo run --example mem_bug_repro`. `find_mems` must equal the brute-force truth
//! set, and `missed by find_mems` must be empty.

use haystackfm::alphabet::encode_byte;
use haystackfm::{BidirFmIndex, DnaSequence, FmIndexConfig};

fn encode(s: &str) -> Vec<u8> {
    s.bytes().map(|b| encode_byte(b).unwrap()).collect()
}

/// Every occurrence-maximal exact match of `query` in `texts`, as query intervals.
fn brute_force_mems(query: &[u8], texts: &[Vec<u8>], min_len: usize) -> Vec<(usize, usize)> {
    let mut out = std::collections::BTreeSet::new();
    for t in texts {
        for i in 0..query.len() {
            for j in 0..t.len() {
                if query[i] != t[j] {
                    continue;
                }
                if i > 0 && j > 0 && query[i - 1] == t[j - 1] {
                    continue; // not left-maximal at this occurrence
                }
                let mut l = 0;
                while i + l < query.len() && j + l < t.len() && query[i + l] == t[j + l] {
                    l += 1;
                }
                if l >= min_len {
                    out.insert((i, i + l));
                }
            }
        }
    }
    out.into_iter().collect()
}

fn main() {
    let query = "ACGTACGTAC";
    let ref_a = "ACGTACGTAC"; // contains the whole query
    let ref_b = "ACGTACT"; // contains only query[0..6), flanked by 'T' != query[6]='G'
    let min_len = 2;

    let seqs = vec![
        DnaSequence::from_str_with_header(ref_a, "A").unwrap(),
        DnaSequence::from_str_with_header(ref_b, "B").unwrap(),
    ];
    let idx = BidirFmIndex::build_cpu(&seqs, &FmIndexConfig::default()).unwrap();

    let q = encode(query);
    let mems: Vec<(usize, usize)> = idx
        .find_mems(&q, min_len, false)
        .iter()
        .map(|m| (m.query_start, m.query_end))
        .collect();
    let smems: Vec<(usize, usize)> = idx
        .find_smems(&q, min_len, false)
        .iter()
        .map(|m| (m.query_start, m.query_end))
        .collect();
    let truth = brute_force_mems(&q, &[encode(ref_a), encode(ref_b)], min_len);

    println!("true MEMs (brute force) = {truth:?}");
    println!("find_mems               = {mems:?}");
    println!("find_smems              = {smems:?}");
    println!("find_mems == find_smems = {}", mems == smems);
    println!(
        "missed by find_mems     = {:?}",
        truth
            .iter()
            .filter(|x| !mems.contains(x))
            .collect::<Vec<_>>()
    );

    assert_eq!(
        mems, truth,
        "find_mems must equal the brute-force truth set"
    );
    assert_ne!(
        mems, smems,
        "find_mems must be strictly richer than find_smems"
    );
    println!("\nOK: find_mems matches brute force and is strictly richer than find_smems.");
}
