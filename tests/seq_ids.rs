//! The [`SeqId`] query surface: queries report integer ids, and the header <-> id
//! accessors are exact inverses of each other.
//!
//! Correctness of the reported *locations* is checked against a brute-force search over
//! the same corpus, so these tests stand on their own rather than against another index
//! code path.

mod common;

use common::encode_pattern;
use haystackfm::alphabet::DnaSequence;
use haystackfm::error::FmIndexError;
use haystackfm::{BidirFmIndex, FmIndex, FmIndexConfig, SeqId};
use std::collections::HashSet;

/// Multi-reference corpus with a region ("GATTACA") conserved across every sequence, so a
/// single seed resolves to many references — the case the id API exists for.
const REFS: &[(&str, &str)] = &[
    ("ref_alpha", "ACGTGATTACAACGTTAGC"),
    ("ref_beta", "TTGGCCAAGATTACATTGCA"),
    ("ref_gamma", "GATTACAGGGCCCTTTAAA"),
    ("ref_delta", "AACCGGTTGATTACACGCGCG"),
];

/// Patterns exercised against the corpus: the conserved seed, substrings unique to one
/// reference, single bases occurring everywhere, and an absent pattern.
const PATTERNS: &[&str] = &[
    "GATTACA", "ACGT", "A", "C", "GG", "TTGCA", "CGCGCG", "TAGC", "GGGCCC", "TTTTTTTT",
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

/// Brute-force `(seq_id, offset)` occurrences of `pattern` across [`REFS`].
fn naive_hits(pattern: &str) -> HashSet<(SeqId, u32)> {
    let mut hits = HashSet::new();
    for (id, (_, text)) in REFS.iter().enumerate() {
        if pattern.is_empty() || pattern.len() > text.len() {
            continue;
        }
        for start in 0..=text.len() - pattern.len() {
            if &text[start..start + pattern.len()] == pattern {
                hits.insert((SeqId::new(id as u32), start as u32));
            }
        }
    }
    hits
}

// ── locate ────────────────────────────────────────────────────────────────────

#[test]
fn locate_reports_ids_matching_brute_force() {
    // Sampling rate >= 2 exercises the LF-walk to the nearest SA sample, which is where
    // multi-sentinel indexes have historically diverged from the full-SA path.
    for rate in [1, 2, 4, 8] {
        let idx = fm_index(rate);
        for pattern in PATTERNS {
            let got: HashSet<(SeqId, u32)> =
                idx.locate(&encode_pattern(pattern)).into_iter().collect();
            assert_eq!(
                got,
                naive_hits(pattern),
                "rate {rate}, pattern {pattern}: locate disagrees with brute force"
            );
        }
    }
}

#[test]
fn located_ids_resolve_to_the_containing_reference() {
    let idx = fm_index(4);
    for pattern in PATTERNS {
        for (id, pos) in idx.locate(&encode_pattern(pattern)) {
            let header = idx.seq_header(id).expect("located id must be in range");
            let (expected_header, text) = REFS[id.index()];
            assert_eq!(header, expected_header);
            let end = pos as usize + pattern.len();
            assert!(end <= text.len(), "hit {pos} runs past the end of {header}");
            assert_eq!(&text[pos as usize..end], *pattern);
        }
    }
}

#[test]
fn conserved_seed_hits_every_reference() {
    // Guards the premise of the feature: one seed, many references.
    let idx = fm_index(4);
    let mut hit_refs: Vec<SeqId> = idx
        .locate(&encode_pattern("GATTACA"))
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    hit_refs.sort_unstable();
    hit_refs.dedup();
    assert_eq!(
        hit_refs,
        (0..REFS.len() as u32).map(SeqId::new).collect::<Vec<_>>()
    );
}

// ── locate_interval / MEM / SMEM ──────────────────────────────────────────────

#[test]
fn locate_interval_reports_ids_matching_brute_force() {
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
            let expected = naive_hits(pattern);
            match iv {
                Some(iv) => {
                    let got: HashSet<(SeqId, u32)> = idx.locate_interval(&iv).into_iter().collect();
                    assert_eq!(got, expected, "rate {rate}, pattern {pattern}");
                }
                // Interval collapsed — the pattern must genuinely be absent.
                None => assert!(expected.is_empty(), "rate {rate}, pattern {pattern}"),
            }
        }
    }
}

/// Queries spanning reference boundaries and the conserved region.
const QUERIES: &[&str] = &[
    "ACGTGATTACAACGTTAGC",
    "GATTACAGGG",
    "TTGGCCAAGATTACATTGCA",
    "AACCGGTTGATTACACGCGCG",
    "ACGTACGTACGT",
    "GATTACA",
];

/// Every located MEM/SMEM position must be an id whose reference really contains the
/// matched query span at that offset.
#[test]
fn mem_positions_carry_ids_of_references_containing_the_match() {
    for rate in [1, 4] {
        let idx = bidir_index(rate);
        for query in QUERIES {
            let encoded = encode_pattern(query);
            for min_len in [3usize, 5, 7] {
                let smems = idx.find_smems(&encoded, min_len, true);
                let mems = idx.find_mems(&encoded, min_len, true);
                for m in smems.iter().chain(&mems) {
                    let matched = &query[m.query_start..m.query_end];
                    assert!(m.len() >= min_len);
                    for &(id, pos) in &m.positions {
                        let (header, text) = REFS[id.index()];
                        assert_eq!(idx.seq_header(id), Some(header));
                        let end = pos as usize + matched.len();
                        assert!(end <= text.len(), "{header}: match runs past end");
                        assert_eq!(
                            &text[pos as usize..end],
                            matched,
                            "rate {rate}, query {query}, min_len {min_len}: \
                             {header} does not contain the reported match at {pos}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mem_positions_are_empty_when_locate_is_false() {
    let idx = bidir_index(4);
    let encoded = encode_pattern("ACGTGATTACAACGTTAGC");

    let smems = idx.find_smems(&encoded, 5, false);
    assert!(!smems.is_empty(), "expected SMEMs for the test query");
    assert!(smems.iter().all(|m| m.positions.is_empty()));
    // Matches are still reported, with their counts.
    assert!(smems.iter().all(|m| m.match_count > 0));

    let mems = idx.find_mems(&encoded, 5, false);
    assert!(!mems.is_empty(), "expected MEMs for the test query");
    assert!(mems.iter().all(|m| m.positions.is_empty()));
}

// ── accessors ─────────────────────────────────────────────────────────────────

#[test]
fn seq_header_and_seq_id_are_exact_inverses() {
    let idx = fm_index(4);
    assert_eq!(idx.num_sequences() as usize, REFS.len());
    assert_eq!(idx.seq_headers().len(), REFS.len());
    for (i, (header, _)) in REFS.iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(idx.seq_header(id), Some(*header));
        assert_eq!(idx.seq_id(header), Some(id));
        assert_eq!(&idx.seq_headers()[id.index()], header);
    }
}

#[test]
fn seq_accessors_reject_unknown_inputs() {
    let idx = fm_index(4);
    assert_eq!(idx.seq_header(SeqId::new(REFS.len() as u32)), None);
    assert_eq!(idx.seq_header(SeqId::new(u32::MAX)), None);
    assert_eq!(idx.seq_id("no_such_header"), None);
    assert_eq!(idx.seq_id(""), None);
}

#[test]
fn bidir_accessors_delegate_to_forward_index() {
    let bidir = bidir_index(4);
    let fwd = fm_index(4);
    assert_eq!(bidir.seq_headers(), fwd.seq_headers());
    for (i, (header, _)) in REFS.iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(bidir.seq_header(id), Some(*header));
        assert_eq!(bidir.seq_id(header), Some(id));
    }
    assert_eq!(bidir.seq_header(SeqId::new(REFS.len() as u32)), None);
    assert_eq!(bidir.seq_id("no_such_header"), None);
}

#[test]
fn sequences_without_headers_get_distinct_generated_names() {
    // `DnaSequence::from_str` leaves the header empty; the builder fills in `seq_{i}`,
    // which must stay unique so the reverse lookup is still exact.
    let seqs: Vec<DnaSequence> = ["ACGT", "TTTT", "GGGG"]
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    let idx = FmIndex::build_cpu(&seqs, &config(1)).unwrap();
    for (i, expected) in ["seq_0", "seq_1", "seq_2"].iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(idx.seq_header(id), Some(*expected));
        assert_eq!(idx.seq_id(expected), Some(id));
    }
}

// ── duplicate headers ─────────────────────────────────────────────────────────

#[test]
fn duplicate_headers_are_rejected_at_build() {
    let seqs = vec![
        DnaSequence::from_str_with_header("ACGT", "chr1").unwrap(),
        DnaSequence::from_str_with_header("TTTT", "chr2").unwrap(),
        DnaSequence::from_str_with_header("GGGG", "chr1").unwrap(),
    ];
    let err = FmIndex::build_cpu(&seqs, &config(1)).unwrap_err();
    assert!(
        matches!(&err, FmIndexError::DuplicateHeader(h) if h == "chr1"),
        "unexpected error: {err}"
    );

    let err = BidirFmIndex::build_cpu(&seqs, &config(1)).unwrap_err();
    assert!(
        matches!(&err, FmIndexError::DuplicateHeader(h) if h == "chr1"),
        "unexpected error: {err}"
    );
}

#[test]
fn explicit_header_colliding_with_a_generated_name_is_rejected() {
    // Sequence 0 is explicitly named "seq_1"; sequence 1 has no header and would be
    // auto-named "seq_1" too.
    let seqs = vec![
        DnaSequence::from_str_with_header("ACGT", "seq_1").unwrap(),
        DnaSequence::from_str("TTTT").unwrap(),
    ];
    let err = FmIndex::build_cpu(&seqs, &config(1)).unwrap_err();
    assert!(
        matches!(&err, FmIndexError::DuplicateHeader(h) if h == "seq_1"),
        "unexpected error: {err}"
    );
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
            restored.locate(&encoded),
            idx.locate(&encoded),
            "locate changed across serialization for {pattern}"
        );
    }
}

#[test]
fn header_lookup_works_after_deserialization() {
    // The header -> id map is not serialized; `from_bytes` must rebuild it.
    let idx = fm_index(4);
    let restored = FmIndex::from_bytes(&idx.to_bytes().unwrap()).unwrap();
    for (i, (header, _)) in REFS.iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(restored.seq_id(header), Some(id));
        assert_eq!(restored.seq_header(id), Some(*header));
    }
    assert_eq!(restored.seq_id("no_such_header"), None);
}

#[test]
fn bidir_ids_survive_serialization_round_trip() {
    let idx = bidir_index(4);
    let restored = BidirFmIndex::from_bytes(&idx.to_bytes().unwrap()).unwrap();

    assert_eq!(restored.seq_headers(), idx.seq_headers());
    assert_eq!(restored.seq_id("ref_gamma"), Some(SeqId::new(2)));
    let encoded = encode_pattern("ACGTGATTACAACGTTAGC");
    assert_eq!(
        restored.find_smems(&encoded, 5, true),
        idx.find_smems(&encoded, 5, true)
    );
}
