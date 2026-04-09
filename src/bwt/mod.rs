//! Burrows-Wheeler Transform (BWT) construction and representation.
//!
//! The BWT is a reversible permutation of the input text that groups similar
//! characters together, enabling efficient rank queries via the Occ table.

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Burrows-Wheeler Transform: a permutation of the input text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Bwt {
    pub data: Vec<u8>,
}

impl Bwt {
    /// Returns the length of the BWT (equals the length of the original text).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the BWT is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
