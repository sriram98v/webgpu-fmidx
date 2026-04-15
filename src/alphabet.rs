use crate::error::FmIndexError;

/// Alphabet size: $, A, C, G, T
pub const ALPHABET_SIZE: usize = 5;

/// Sentinel character (lexicographically smallest)
pub const SENTINEL: u8 = 0;
/// Encoded value for adenine (A).
pub const A: u8 = 1;
/// Encoded value for cytosine (C).
pub const C: u8 = 2;
/// Encoded value for guanine (G).
pub const G: u8 = 3;
/// Encoded value for thymine (T).
pub const T: u8 = 4;

/// Encode a single ASCII DNA character to its alphabet index.
pub fn encode_char(ch: char) -> Option<u8> {
    match ch {
        '$' => Some(SENTINEL),
        'A' | 'a' => Some(A),
        'C' | 'c' => Some(C),
        'G' | 'g' => Some(G),
        'T' | 't' => Some(T),
        _ => None,
    }
}

/// Decode an alphabet index back to its ASCII character.
pub fn decode_char(code: u8) -> Option<char> {
    match code {
        SENTINEL => Some('$'),
        A => Some('A'),
        C => Some('C'),
        G => Some('G'),
        T => Some('T'),
        _ => None,
    }
}

/// A validated DNA sequence (A, C, G, T only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnaSequence {
    bases: Vec<u8>,
    /// FASTA header (without leading `>`). Empty string if not provided.
    header: String,
}

impl DnaSequence {
    /// Parse from a string of ACGT characters. Returns Err on invalid characters.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, FmIndexError> {
        if s.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        let mut bases = Vec::with_capacity(s.len());
        for (i, ch) in s.chars().enumerate() {
            match ch {
                'A' | 'a' => bases.push(A),
                'C' | 'c' => bases.push(C),
                'G' | 'g' => bases.push(G),
                'T' | 't' => bases.push(T),
                _ => return Err(FmIndexError::InvalidCharacter(ch, i)),
            }
        }
        Ok(Self { bases, header: String::new() })
    }

    /// Parse from a string of ACGT characters with a FASTA header.
    pub fn from_str_with_header(s: &str, header: &str) -> Result<Self, FmIndexError> {
        let mut seq = Self::from_str(s)?;
        seq.header = header.to_string();
        Ok(seq)
    }

    /// Create from pre-encoded bases (no validation).
    pub fn from_encoded(bases: Vec<u8>) -> Self {
        Self { bases, header: String::new() }
    }

    /// Returns the FASTA header (without `>`), or empty string if not set.
    pub fn header(&self) -> &str {
        &self.header
    }

    /// Returns the number of bases in the sequence.
    pub fn len(&self) -> usize {
        self.bases.len()
    }

    /// Returns `true` if the sequence contains no bases.
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Returns the underlying encoded bases as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.bases
    }
}

/// Concatenate multiple DNA sequences with $ separators into a single encoded text.
/// Result: s1 $ s2 $ ... $ sn $
/// Returns the concatenated text and the cumulative lengths (for mapping positions back).
pub fn concatenate_sequences(
    sequences: &[DnaSequence],
) -> Result<(Vec<u8>, Vec<u32>), FmIndexError> {
    let total_len: usize = sequences.iter().map(|s| s.len() + 1).sum();
    if total_len > u32::MAX as usize {
        return Err(FmIndexError::TextTooLarge(total_len));
    }

    let mut text = Vec::with_capacity(total_len);
    let mut cumulative_lengths = Vec::with_capacity(sequences.len());
    let mut offset = 0u32;

    for seq in sequences {
        text.extend_from_slice(seq.as_slice());
        text.push(SENTINEL);
        offset += seq.len() as u32 + 1;
        cumulative_lengths.push(offset);
    }

    Ok((text, cumulative_lengths))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        for ch in ['$', 'A', 'C', 'G', 'T'] {
            let code = encode_char(ch).unwrap();
            assert_eq!(decode_char(code).unwrap(), ch);
        }
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(encode_char('a'), Some(A));
        assert_eq!(encode_char('c'), Some(C));
        assert_eq!(encode_char('g'), Some(G));
        assert_eq!(encode_char('t'), Some(T));
    }

    #[test]
    fn test_invalid_char() {
        assert!(encode_char('X').is_none());
        assert!(encode_char('N').is_none());
    }

    #[test]
    fn test_dna_sequence_valid() {
        let seq = DnaSequence::from_str("ACGT").unwrap();
        assert_eq!(seq.as_slice(), &[A, C, G, T]);
        assert_eq!(seq.len(), 4);
    }

    #[test]
    fn test_dna_sequence_invalid() {
        assert!(DnaSequence::from_str("ACXGT").is_err());
    }

    #[test]
    fn test_dna_sequence_empty() {
        assert!(DnaSequence::from_str("").is_err());
    }

    #[test]
    fn test_concatenate() {
        let s1 = DnaSequence::from_str("ACG").unwrap();
        let s2 = DnaSequence::from_str("TT").unwrap();
        let (text, cum) = concatenate_sequences(&[s1, s2]).unwrap();
        assert_eq!(text, vec![A, C, G, SENTINEL, T, T, SENTINEL]);
        assert_eq!(cum, vec![4, 7]);
    }
}
