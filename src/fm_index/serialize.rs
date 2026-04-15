//! Serialization and deserialization of the FM-index to/from bytes via `bincode`.

use super::FmIndex;
use crate::bwt::Bwt;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::occ::OccTable;
use crate::suffix_array::SampledSuffixArray;

impl FmIndex {
    /// Serialize the FM-index to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let serializable = SerializableFmIndex {
            bwt: &self.bwt,
            c_array: &self.c_array,
            occ: &self.occ,
            sa_samples: &self.sa_samples,
            text_len: self.text_len,
            num_sequences: self.num_sequences,
            seq_boundaries: &self.seq_boundaries,
            seq_headers: &self.seq_headers,
        };
        bincode::serialize(&serializable).map_err(|e| FmIndexError::SerializeError(e.to_string()))
    }

    /// Deserialize an FM-index from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let deserialized: OwnedSerializableFmIndex = bincode::deserialize(data)
            .map_err(|e| FmIndexError::DeserializeError(e.to_string()))?;
        Ok(Self {
            bwt: deserialized.bwt,
            c_array: deserialized.c_array,
            occ: deserialized.occ,
            sa_samples: deserialized.sa_samples,
            text_len: deserialized.text_len,
            num_sequences: deserialized.num_sequences,
            seq_boundaries: deserialized.seq_boundaries,
            seq_headers: deserialized.seq_headers,
        })
    }
}

#[derive(serde::Serialize)]
struct SerializableFmIndex<'a> {
    bwt: &'a Bwt,
    c_array: &'a CArray,
    occ: &'a OccTable,
    sa_samples: &'a SampledSuffixArray,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: &'a [u32],
    seq_headers: &'a [String],
}

#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndex {
    bwt: Bwt,
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
}

#[cfg(test)]
mod tests {
    use crate::alphabet::*;
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn encode_pattern(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    #[test]
    fn test_serialize_roundtrip() {
        let seq = DnaSequence::from_str("ACGTACGTACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
        };
        let original = FmIndex::build_cpu(&[seq], &config).unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = FmIndex::from_bytes(&bytes).unwrap();

        // Verify the restored index produces the same results
        for pattern in &["A", "AC", "ACGT", "GT", "ACGTACGT"] {
            let p = encode_pattern(pattern);
            assert_eq!(
                original.count(&p),
                restored.count(&p),
                "count mismatch for '{}'",
                pattern
            );

            let mut orig_locs = original.locate(&p);
            let mut rest_locs = restored.locate(&p);
            orig_locs.sort_by_key(|(_, pos)| *pos);
            rest_locs.sort_by_key(|(_, pos)| *pos);
            assert_eq!(orig_locs, rest_locs, "locate mismatch for '{}'", pattern);
        }
    }

    #[test]
    fn test_serialize_multi_sequence() {
        let sequences = vec![
            DnaSequence::from_str("ACGT").unwrap(),
            DnaSequence::from_str("TGCA").unwrap(),
        ];
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
        };
        let original = FmIndex::build_cpu(&sequences, &config).unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = FmIndex::from_bytes(&bytes).unwrap();

        assert_eq!(original.text_len(), restored.text_len());
        assert_eq!(original.num_sequences(), restored.num_sequences());
    }
}
