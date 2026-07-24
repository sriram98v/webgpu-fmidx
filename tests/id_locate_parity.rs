//! Parity between the `String`-header locate path and the integer-id locate path.
//!
//! The `_ids` variants must find exactly the same occurrences as their `String`
//! counterparts — only the sequence representation differs. These tests pin that
//! equivalence, plus the stability of ids across serialization.

mod common;

use common::encode_pattern;
use haystackfm::alphabet::DnaSequence;
use haystackfm::{BidirFmIndex, FmIndex, FmIndexConfig};

/// Multi-reference corpus with a region ("GATTACA") conserved across every sequence, so a
/// single seed resolves to many references — the case the id API exists for.
const REFS: &[(&str, &str)] = &[
    ("ref_alpha", "ACGTGATTACAACGTTAGC"),
    ("ref_beta", "TTGGCCAAGATTACATTGCA"),
    ("ref_gamma", "GATTACAGGGCCCTTTAAA"),
    ("ref_delta", "AACCGGTTGATTACACGCGCG"),
];

fn config(sa_sample_rate: u32) -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate,
        use_gpu: false,
        ..Default::default()
    }
}

fn sequences() -> Vec<DnaSequence> {
    REFS.iter()
        .map(|(header, seq)| DnaSequence::from_str_with_header(seq, header).unwrap())
        .collect()
}

fn fm_index(sa_sample_rate: u32) -> FmIndex {
    FmIndex::build_cpu(&sequences(), &config(sa_sample_rate)).unwrap()
}

fn bidir_index(sa_sample_rate: u32) -> BidirFmIndex {
    BidirFmIndex::build_cpu(&sequences(), &config(sa_sample_rate)).unwrap()
}

/// Patterns exercised against the multi-reference corpus: the conserved seed, substrings
/// unique to one reference, a single base occurring everywhere, and an absent pattern.
const PATTERNS: &[&str] = &[
    "GATTACA", "ACGT", "A", "C", "GG", "TTGCA", "CGCGCG", "TAGC", "GGGCCC", "TTTTTTTT",
];

/// `String` and id hits must line up 1:1, in the same order, with the id resolving to the
/// same header.
fn assert_pairs_agree(idx: &BidirFmIndex, strs: &[(String, u32)], ids: &[(u32, u32)]) {
    assert_eq!(
        strs.len(),
        ids.len(),
        "hit counts differ: {strs:?} vs {ids:?}"
    );
    for ((header, str_pos), &(seq_id, id_pos)) in strs.iter().zip(ids) {
        assert_eq!(*str_pos, id_pos, "position mismatch for header {header}");
        assert_eq!(
            idx.seq_header(seq_id as usize),
            Some(header.as_str()),
            "seq_id {seq_id} does not resolve to header {header}"
        );
    }
}

// ── locate ────────────────────────────────────────────────────────────────────

#[test]
fn locate_ids_matches_locate() {
    // Sampling rate ≥ 2 exercises the LF-walk to the nearest SA sample, which is where
    // multi-sentinel indexes have historically diverged from the full-SA path.
    for rate in [1, 2, 4, 8] {
        let idx = fm_index(rate);
        let headers = idx.seq_headers();
        for pattern in PATTERNS {
            let encoded = encode_pattern(pattern);
            let strs = idx.locate(&encoded);
            let ids = idx.locate_ids(&encoded);
            assert_eq!(
                strs.len(),
                ids.len(),
                "rate {rate}, pattern {pattern}: hit counts differ"
            );
            for ((header, str_pos), &(seq_id, id_pos)) in strs.iter().zip(&ids) {
                assert_eq!(*str_pos, id_pos, "rate {rate}, pattern {pattern}");
                assert_eq!(
                    &headers[seq_id as usize], header,
                    "rate {rate}, pattern {pattern}: id {seq_id} != header {header}"
                );
            }
        }
    }
}

#[test]
fn locate_ids_returns_in_range_ids() {
    let idx = fm_index(4);
    let n = idx.num_sequences();
    for pattern in PATTERNS {
        for (seq_id, pos) in idx.locate_ids(&encode_pattern(pattern)) {
            assert!(
                seq_id < n,
                "seq_id {seq_id} out of range (num_sequences {n})"
            );
            let seq_len = REFS[seq_id as usize].1.len() as u32;
            assert!(
                pos < seq_len,
                "pos {pos} outside sequence {seq_id} of length {seq_len}"
            );
        }
    }
}

#[test]
fn conserved_seed_hits_every_reference() {
    // Guards the premise of the feature: one seed, many references.
    let idx = fm_index(4);
    let ids = idx.locate_ids(&encode_pattern("GATTACA"));
    let mut hit_refs: Vec<u32> = ids.iter().map(|&(seq_id, _)| seq_id).collect();
    hit_refs.sort_unstable();
    hit_refs.dedup();
    assert_eq!(hit_refs, vec![0, 1, 2, 3]);
}

// ── locate_interval ───────────────────────────────────────────────────────────

#[test]
fn locate_interval_ids_matches_locate_interval() {
    for rate in [1, 4] {
        let idx = bidir_index(rate);
        for pattern in PATTERNS {
            let mut iv = Some(idx.full_interval());
            for &c in encode_pattern(pattern).iter() {
                iv = iv.and_then(|cur| idx.extend_right(cur, c));
                if iv.is_none() {
                    break;
                }
            }
            let Some(iv) = iv else { continue };
            assert_pairs_agree(
                &idx,
                &idx.locate_interval(&iv),
                &idx.locate_interval_ids(&iv),
            );
        }
    }
}

// ── find_smems / find_mems ────────────────────────────────────────────────────

/// Queries spanning reference boundaries and the conserved region.
const QUERIES: &[&str] = &[
    "ACGTGATTACAACGTTAGC",
    "GATTACAGGG",
    "TTGGCCAAGATTACATTGCA",
    "AACCGGTTGATTACACGCGCG",
    "ACGTACGTACGT",
    "GATTACA",
];

#[test]
fn find_smems_ids_matches_find_smems() {
    for rate in [1, 4] {
        let idx = bidir_index(rate);
        for query in QUERIES {
            let encoded = encode_pattern(query);
            for min_len in [3usize, 5, 7] {
                let strs = idx.find_smems(&encoded, min_len, true);
                let ids = idx.find_smems_ids(&encoded, min_len, true);
                assert_eq!(
                    strs.len(),
                    ids.len(),
                    "rate {rate}, query {query}, min_len {min_len}: SMEM counts differ"
                );
                for (s, i) in strs.iter().zip(&ids) {
                    assert_eq!((s.query_start, s.query_end), (i.query_start, i.query_end));
                    assert_eq!(s.match_count, i.match_count);
                    assert_pairs_agree(&idx, &s.positions, &i.positions);
                }
            }
        }
    }
}

#[test]
fn find_mems_ids_matches_find_mems() {
    for rate in [1, 4] {
        let idx = bidir_index(rate);
        for query in QUERIES {
            let encoded = encode_pattern(query);
            for min_len in [3usize, 5, 7] {
                let strs = idx.find_mems(&encoded, min_len, true);
                let ids = idx.find_mems_ids(&encoded, min_len, true);
                assert_eq!(
                    strs.len(),
                    ids.len(),
                    "rate {rate}, query {query}, min_len {min_len}: MEM counts differ"
                );
                for (s, i) in strs.iter().zip(&ids) {
                    assert_eq!((s.query_start, s.query_end), (i.query_start, i.query_end));
                    assert_eq!(s.match_count, i.match_count);
                    assert_pairs_agree(&idx, &s.positions, &i.positions);
                }
            }
        }
    }
}

#[test]
fn ids_variants_skip_positions_when_locate_is_false() {
    let idx = bidir_index(4);
    let encoded = encode_pattern("ACGTGATTACAACGTTAGC");

    let smems = idx.find_smems_ids(&encoded, 5, false);
    assert!(!smems.is_empty(), "expected SMEMs for the test query");
    assert!(smems.iter().all(|m| m.positions.is_empty()));
    // Matches are still reported, with their counts.
    assert!(smems.iter().all(|m| m.match_count > 0));

    let mems = idx.find_mems_ids(&encoded, 5, false);
    assert!(!mems.is_empty(), "expected MEMs for the test query");
    assert!(mems.iter().all(|m| m.positions.is_empty()));
}

#[test]
fn ids_variants_report_same_matches_regardless_of_locate() {
    let idx = bidir_index(4);
    let encoded = encode_pattern("TTGGCCAAGATTACATTGCA");
    let located = idx.find_smems_ids(&encoded, 5, true);
    let unlocated = idx.find_smems_ids(&encoded, 5, false);
    assert_eq!(located.len(), unlocated.len());
    for (a, b) in located.iter().zip(&unlocated) {
        assert_eq!((a.query_start, a.query_end), (b.query_start, b.query_end));
        assert_eq!(a.match_count, b.match_count);
    }
}

#[test]
fn mem_ids_len_matches_query_span() {
    let idx = bidir_index(4);
    for mem in idx.find_smems_ids(&encode_pattern("ACGTGATTACAACGTTAGC"), 5, false) {
        assert_eq!(mem.len(), mem.query_end - mem.query_start);
        assert!(!mem.is_empty());
    }
}

// ── accessors ─────────────────────────────────────────────────────────────────

#[test]
fn seq_header_and_seq_id_round_trip() {
    let idx = fm_index(4);
    assert_eq!(idx.num_sequences() as usize, REFS.len());
    assert_eq!(idx.seq_headers().len(), REFS.len());
    for (i, (header, _)) in REFS.iter().enumerate() {
        assert_eq!(idx.seq_header(i), Some(*header));
        assert_eq!(idx.seq_id(header), Some(i));
        assert_eq!(&idx.seq_headers()[i], header);
    }
}

#[test]
fn seq_accessors_reject_unknown_inputs() {
    let idx = fm_index(4);
    assert_eq!(idx.seq_header(REFS.len()), None);
    assert_eq!(idx.seq_header(usize::MAX), None);
    assert_eq!(idx.seq_id("no_such_header"), None);
    assert_eq!(idx.seq_id(""), None);
}

#[test]
fn bidir_accessors_delegate_to_forward_index() {
    let bidir = bidir_index(4);
    let fwd = fm_index(4);
    assert_eq!(bidir.seq_headers(), fwd.seq_headers());
    for (i, (header, _)) in REFS.iter().enumerate() {
        assert_eq!(bidir.seq_header(i), Some(*header));
        assert_eq!(bidir.seq_id(header), Some(i));
    }
    assert_eq!(bidir.seq_header(REFS.len()), None);
    assert_eq!(bidir.seq_id("no_such_header"), None);
}

// ── id stability across serialization ─────────────────────────────────────────

#[test]
fn ids_survive_serialization_round_trip() {
    let idx = fm_index(4);
    let restored = FmIndex::from_bytes(&idx.to_bytes().unwrap()).unwrap();

    assert_eq!(restored.seq_headers(), idx.seq_headers());
    for pattern in PATTERNS {
        let encoded = encode_pattern(pattern);
        assert_eq!(
            restored.locate_ids(&encoded),
            idx.locate_ids(&encoded),
            "locate_ids changed across serialization for {pattern}"
        );
    }
}

#[test]
fn bidir_ids_survive_serialization_round_trip() {
    let idx = bidir_index(4);
    let restored = BidirFmIndex::from_bytes(&idx.to_bytes().unwrap()).unwrap();

    assert_eq!(restored.seq_headers(), idx.seq_headers());
    let encoded = encode_pattern("ACGTGATTACAACGTTAGC");
    let before = idx.find_smems_ids(&encoded, 5, true);
    let after = restored.find_smems_ids(&encoded, 5, true);
    assert_eq!(before, after);
}
