//! Locality-Sensitive Hashing (LSH) — 256-bit SimHash fallback (4 seeds active).
//!
//! Used as a BM25 fallback bridge: when BM25 returns fewer than 2 candidates,
//! four active SimHash planes (`LSH_SEEDS[..4]`) find neurons within Hamming
//! distance ≤14 and inject them as overflow candidates. The remaining 12 seeds
//! are reserved. Because the seeds vary only the FNV-1a offset basis while
//! sharing the multiplier, the planes are correlated rather than fully
//! independent. False positives are filtered by downstream BM25 re-ranking.

use crate::types::TermFrequency;
use std::collections::HashMap;

/// 16 compile-time seeds for the SimHash ensemble (4 active, 12 reserved).
/// Derived from golden ratio (φ = 1.618…) bit patterns and prime multiples.
pub(super) const LSH_SEEDS: [u64; 16] = [
    0x9e3779b97f4a7c15, // golden ratio × 2^64
    0x6c62272e07bb0142, // FNV-1a basis
    0xd4e27153a6fb0c00,
    0xa3b195354a2b7d37,
    0x1b03738712fad5c9,
    0x5bf03635d3a99f43,
    0xcbf29ce484222325, // original FNV offset
    0x517cc1b727220a95,
    0x3a84f8a00be8cb24,
    0xf1d84f7032c88cf9,
    0x2ff9bcb7eedfbc29,
    0xb3a5c5eb2c9bbd93,
    0x8e2fcac9574ac83c,
    0xd8a4d8012b77b7b5,
    0x45291b48a2da8af2,
    0x71d93f1c7ab0ec25,
];

/// S-II (R16): Compute a 64-bit SimHash fingerprint from a term→weight map.
///
/// SimHash projects each term onto a 64-dimensional bit vector using a seeded
/// FNV-1a hash, then sums the weighted contributions per dimension. The final
/// bit is set when the sum is positive. Zero external dependencies.
///
/// Hamming distance between two SimHashes approximates cosine distance over
/// the original TF-IDF vectors; neurons within distance ≤12 bits are likely
/// semantically related.
pub(super) fn simhash_with_seed(term_freq: &HashMap<String, TermFrequency>, seed: u64) -> u64 {
    let mut v = [0.0f64; 64];
    for (term, weight) in term_freq {
        // FNV-1a seeded: XOR seed into the offset basis for independent hash family
        let mut h: u64 = seed;
        for byte in term.as_bytes() {
            h ^= *byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let w = weight.get() as f64;
        for bit in 0..64u32 {
            if (h >> bit) & 1 == 1 {
                v[bit as usize] += w;
            } else {
                v[bit as usize] -= w;
            }
        }
    }
    let mut fingerprint: u64 = 0;
    for bit in 0..64u32 {
        if v[bit as usize] > 0.0 {
            fingerprint |= 1u64 << bit;
        }
    }
    fingerprint
}

/// Compute four independent 64-bit SimHash fingerprints from a term→weight map
/// (256 effective bits).
///
/// # Ensemble design and limitations
/// Four independent 64-bit planes yield 256 effective bits, providing the same
/// Johnson-Lindenstrauss accuracy improvement over a single-seed SimHash while
/// eliminating 75% of the seed-iteration overhead of a 16-seed design.
///
/// **Limitation:** SimHash approximates cosine distance over TF-weighted term
/// vectors — it is a good proxy for lexical overlap but is not semantic.
/// Two documents with near-identical vocabularies but opposite meanings may get
/// a low Hamming distance. The ensemble is used **only** as a fallback bridge
/// when BM25 returns fewer than 2 candidates; it is never the primary retrieval
/// path, so false-positive candidates from LSH are filtered by downstream BM25
/// re-ranking before reaching the caller.
pub(super) fn simhash_256(term_freq: &HashMap<String, TermFrequency>) -> [u64; 4] {
    let mut fps = [0u64; 4];
    for (i, &seed) in LSH_SEEDS[..4].iter().enumerate() {
        fps[i] = simhash_with_seed(term_freq, seed);
    }
    fps
}

/// Popcount (Hamming weight) of the XOR of two 64-bit values — i.e., Hamming distance.
#[inline]
pub(super) fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}
