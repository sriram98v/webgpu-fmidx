//! GPU MEM/SMEM parity tests.
//!
//! SMEM tests assert `find_smems_gpu` returns the same multiset of
//! `(query_start, query_end, match_count)` as the CPU `find_smems`.
//!
//! MEM tests assert against `legacy_mem_oracle`, **not** `find_mems`: the shader's MODE_MEM
//! path has not been ported to occurrence-level MEMs (issue 1 in `KNOWN-ISSUES.md`), so it
//! returns a strict subset of the CPU result. Point these back at `find_mems` as part of the
//! port.

#[cfg(feature = "gpu")]
mod legacy_mem_oracle;

#[cfg(feature = "gpu")]
mod tests {
    use crate::legacy_mem_oracle::{legacy_mems, LegacyMem};
    use haystackfm::alphabet::DnaSequence;
    use haystackfm::fm_index::smem::Mem;
    use haystackfm::{BidirFmIndex, FmIndexConfig, MemHit};
    use pollster::FutureExt as _;

    fn cpu_config() -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        }
    }

    fn build(seqs: &[&str]) -> BidirFmIndex {
        let dna: Vec<DnaSequence> = seqs
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        BidirFmIndex::build_cpu(&dna, &cpu_config()).unwrap()
    }

    fn seq(s: &str) -> DnaSequence {
        DnaSequence::from_str(s).unwrap()
    }

    fn cpu_key(m: &Mem) -> (usize, usize, u32) {
        (m.query_start, m.query_end, m.match_count)
    }

    fn gpu_key(m: &MemHit) -> (usize, usize, u32) {
        (m.query_start as usize, m.query_end as usize, m.match_count)
    }

    fn cpu_sorted(mems: &[Mem]) -> Vec<(usize, usize, u32)> {
        let mut keys: Vec<_> = mems.iter().map(cpu_key).collect();
        keys.sort();
        keys
    }

    fn gpu_sorted(mems: &[MemHit]) -> Vec<(usize, usize, u32)> {
        let mut keys: Vec<_> = mems.iter().map(gpu_key).collect();
        keys.sort();
        keys
    }

    /// Expected MEM keys for the GPU: the legacy whole-set algorithm the shader still runs.
    fn legacy_sorted(mems: &[LegacyMem]) -> Vec<(usize, usize, u32)> {
        let mut keys: Vec<_> = mems
            .iter()
            .map(|m| (m.query_start, m.query_end, m.match_count))
            .collect();
        keys.sort();
        keys
    }

    // Pass &[] for ref_boundaries — skips position resolution, tests MEM spans only.
    fn smems_gpu_sync(
        idx: &BidirFmIndex,
        queries: &[DnaSequence],
        min_len: usize,
    ) -> Vec<Vec<MemHit>> {
        idx.find_smems_gpu(queries, min_len, &[], 1024)
            .block_on()
            .unwrap()
    }

    fn mems_gpu_sync(
        idx: &BidirFmIndex,
        queries: &[DnaSequence],
        min_len: usize,
    ) -> Vec<Vec<MemHit>> {
        idx.find_mems_gpu(queries, min_len, &[], 1024)
            .block_on()
            .unwrap()
    }

    // ── SMEM parity tests ─────────────────────────────────────────────────────

    #[test]
    fn smem_single_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    #[test]
    fn smem_no_match() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("TTTT");
        let cpu = idx.find_smems(q.as_slice(), 4, false);
        let gpu = smems_gpu_sync(&idx, &[q], 4);
        assert!(cpu.is_empty());
        assert!(gpu[0].is_empty());
    }

    #[test]
    fn smem_min_len_filter() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("AC");
        let cpu = idx.find_smems(q.as_slice(), 3, false);
        let gpu = smems_gpu_sync(&idx, &[q], 3);
        assert!(cpu.is_empty());
        assert!(gpu[0].is_empty());
    }

    #[test]
    fn smem_single_char() {
        let idx = build(&["AAAACCCGGG"]);
        for ch in ["A", "C", "G"] {
            let q = seq(ch);
            let cpu = idx.find_smems(q.as_slice(), 1, false);
            let gpu = smems_gpu_sync(&idx, &[q.clone()], 1);
            assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]), "char={ch}");
        }
    }

    #[test]
    fn smem_multi_seq() {
        let idx = build(&["ACGT", "TGCA"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    #[test]
    fn smem_repeated_pattern() {
        let idx = build(&["ACGTACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    #[test]
    fn smem_longer_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGTACGT");
        let cpu = idx.find_smems(q.as_slice(), 2, false);
        let gpu = smems_gpu_sync(&idx, &[q], 2);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    // ── MEM parity tests ──────────────────────────────────────────────────────

    #[test]
    fn mem_single_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let expected = legacy_mems(&idx, q.as_slice(), 1, false);
        let gpu = mems_gpu_sync(&idx, &[q], 1);
        assert_eq!(legacy_sorted(&expected), gpu_sorted(&gpu[0]));
    }

    #[test]
    fn mem_no_match() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("TTTT");
        let expected = legacy_mems(&idx, q.as_slice(), 4, false);
        // The occurrence-level CPU path agrees there is nothing to find here.
        let cpu = idx.find_mems(q.as_slice(), 4, false);
        let gpu = mems_gpu_sync(&idx, &[q], 4);
        assert!(expected.is_empty());
        assert!(cpu.is_empty());
        assert!(gpu[0].is_empty());
    }

    #[test]
    fn mem_batch() {
        let idx = build(&["ACGTACGT"]);
        let queries = vec![seq("A"), seq("AC"), seq("ACG"), seq("ACGT")];
        let expected: Vec<Vec<LegacyMem>> = queries
            .iter()
            .map(|q| legacy_mems(&idx, q.as_slice(), 1, false))
            .collect();
        let gpu = mems_gpu_sync(&idx, &queries, 1);
        for (i, (c, g)) in expected.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(legacy_sorted(c), gpu_sorted(g), "query {i}");
        }
    }

    #[test]
    fn mem_multi_seq() {
        let idx = build(&["ACGT", "TGCA", "AAAA"]);
        let q = seq("ACGTA");
        let expected = legacy_mems(&idx, q.as_slice(), 1, false);
        let gpu = mems_gpu_sync(&idx, &[q], 1);
        assert_eq!(legacy_sorted(&expected), gpu_sorted(&gpu[0]));
    }

    /// Pins the known CPU/GPU MEM gap so the shader port has an explicit before/after:
    /// the GPU misses MEMs the occurrence-level CPU path finds. Delete with issue 1.
    #[test]
    fn mem_gpu_is_a_strict_subset_of_cpu_here() {
        let idx = build(&["ACGT", "TGCA", "AAAA"]);
        let q = seq("ACGTA");
        let cpu = cpu_sorted(&idx.find_mems(q.as_slice(), 1, false));
        let gpu = gpu_sorted(&mems_gpu_sync(&idx, &[q], 1)[0]);

        assert!(
            gpu.iter().all(|k| cpu.contains(k)),
            "GPU reported a MEM the CPU does not:\ncpu={cpu:?}\ngpu={gpu:?}"
        );
        assert!(
            gpu.len() < cpu.len(),
            "GPU MEM mode appears to have been ported — retarget these tests at find_mems \
             and close issue 1 in KNOWN-ISSUES.md\ncpu={cpu:?}\ngpu={gpu:?}"
        );
    }
}
