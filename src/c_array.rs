use crate::alphabet::ALPHABET_SIZE;

/// C array: C[c] = number of characters in the text that are lexicographically smaller than c.
///
/// For DNA alphabet {$=0, A=1, C=2, G=3, T=4, N=5}:
/// C[0] = 0
/// C[1] = count($)
/// C[2] = count($) + count(A)
/// C[3] = count($) + count(A) + count(C)
/// C[4] = count($) + count(A) + count(C) + count(G)
/// C[5] = count($) + count(A) + count(C) + count(G) + count(T)
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CArray {
    pub data: [u32; ALPHABET_SIZE],
}

impl CArray {
    /// Build C array from BWT (or any text) by counting character frequencies.
    pub fn from_text(text: &[u8]) -> Self {
        let mut freq = [0u32; ALPHABET_SIZE];
        for &ch in text {
            if (ch as usize) < ALPHABET_SIZE {
                freq[ch as usize] += 1;
            }
        }

        // Exclusive prefix sum
        let mut data = [0u32; ALPHABET_SIZE];
        let mut sum = 0u32;
        for i in 0..ALPHABET_SIZE {
            data[i] = sum;
            sum += freq[i];
        }

        Self { data }
    }

    /// Get C[c]: number of characters smaller than c in the text.
    pub fn get(&self, c: u8) -> u32 {
        self.data[c as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;

    #[test]
    fn test_c_array_basic() {
        // text: A C G T $  => freq: $=1, A=1, C=1, G=1, T=1, N=0
        let text = vec![A, C, G, T, SENTINEL];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_c_array_repeated() {
        // text: A A C C $  => freq: $=1, A=2, C=2, G=0, T=0, N=0
        let text = vec![A, A, C, C, SENTINEL];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 1, 3, 5, 5, 5]);
    }

    #[test]
    fn test_c_array_empty() {
        let text: Vec<u8> = vec![];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_c_array_with_n() {
        // text: A N N $  => freq: $=1, A=1, C=0, G=0, T=0, N=2
        let text = vec![A, N, N, SENTINEL];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 1, 2, 2, 2, 2]);
    }
}
