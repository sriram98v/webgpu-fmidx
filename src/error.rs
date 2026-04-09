use thiserror::Error;

/// Errors returned by FM-index construction and query operations.
#[derive(Debug, Error)]
pub enum FmIndexError {
    /// A character outside the DNA alphabet (A, C, G, T) was encountered.
    #[error("invalid DNA character '{0}' at position {1}")]
    InvalidCharacter(char, usize),

    /// An empty sequence was provided where a non-empty sequence is required.
    #[error("empty sequence")]
    EmptySequence,

    /// The combined text length exceeds the `u32::MAX` limit.
    #[error("text too large: {0} bytes exceeds u32::MAX")]
    TextTooLarge(usize),

    /// Deserialization from bytes failed.
    #[error("deserialization failed: {0}")]
    DeserializeError(String),

    /// Serialization to bytes failed.
    #[error("serialization failed: {0}")]
    SerializeError(String),

    /// A GPU operation failed (only available with the `gpu` feature).
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    GpuError(String),
}
