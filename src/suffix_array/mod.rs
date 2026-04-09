//! Suffix array construction and sampled suffix array for locate queries.

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Suffix array: SA[i] = starting position of the i-th lexicographically smallest suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuffixArray {
    pub data: Vec<u32>,
}

impl SuffixArray {
    /// Returns the number of entries in the suffix array (equals the text length).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the suffix array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Sampled suffix array for space-efficient locate queries.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SampledSuffixArray {
    pub samples: Vec<u32>,
    pub sample_rate: u32,
}

impl SampledSuffixArray {
    /// Build a sampled SA from a full SA.
    pub fn from_full(sa: &SuffixArray, sample_rate: u32) -> Self {
        let n = sa.len();
        let mut samples = vec![0u32; n];
        // samples[i] stores SA[i] only when SA[i] % sample_rate == 0.
        // We need a mapping: for position i in the BWT order,
        // if SA[i] is a multiple of sample_rate, store it directly.
        // For others, we walk via LF-mapping at query time.
        //
        // Actually, the standard approach: store SA[i] for all i where SA[i] % sample_rate == 0.
        // We need to know which BWT positions have sampled SA values.
        // Store a flat array: samples[i] = SA[i] if SA[i] % sample_rate == 0, else u32::MAX (unsampled).
        for (i, &sa_val) in sa.data.iter().enumerate() {
            samples[i] = if sa_val.is_multiple_of(sample_rate) {
                sa_val
            } else {
                u32::MAX
            };
        }
        Self {
            samples,
            sample_rate,
        }
    }

    /// Check if position i in the BWT has a sampled SA value.
    pub fn is_sampled(&self, i: u32) -> bool {
        self.samples[i as usize] != u32::MAX
    }

    /// Get the SA value for a sampled position.
    pub fn get(&self, i: u32) -> Option<u32> {
        let val = self.samples[i as usize];
        if val != u32::MAX {
            Some(val)
        } else {
            None
        }
    }
}
