//! Branch probability estimation from PGO profile data.
//!
//! Provides branch weight annotations for the IR graph using measured
//! branch frequencies. Falls back to static heuristics when no profile
//! data is available.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
//! │ ProfileData │───▶│ BranchProbability │───▶│  Annotated CFG   │
//! │ (measured)  │    │   (estimation)    │    │  (weights/hints) │
//! └─────────────┘    └──────────────────┘    └──────────────────┘
//! ```

use rustc_hash::FxHashMap;
use std::fmt;

// =============================================================================
// Branch Probability
// =============================================================================

/// A branch probability expressed as a numerator over 2^32.
///
/// This fixed-point representation avoids floating-point arithmetic
/// in the critical path while providing adequate precision (1 part in ~4 billion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BranchProbability {
    /// Numerator (denominator is always 2^32).
    numer: u32,
}

impl BranchProbability {
    /// Denominator is always 2^32 represented as u64.
    const DENOM: u64 = 1u64 << 32;

    /// Always taken (probability = 1.0).
    pub const ALWAYS: Self = Self { numer: u32::MAX };
    /// Never taken (probability = 0.0).
    pub const NEVER: Self = Self { numer: 0 };
    /// Even odds (probability = 0.5).
    pub const EVEN: Self = Self {
        numer: u32::MAX / 2,
    };
    /// Likely taken (probability ≈ 0.9).
    pub const LIKELY: Self = Self {
        numer: (u32::MAX as u64 * 9 / 10) as u32,
    };
    /// Unlikely taken (probability ≈ 0.1).
    pub const UNLIKELY: Self = Self {
        numer: (u32::MAX as u64 / 10) as u32,
    };
    /// Very likely (probability ≈ 0.99).
    pub const VERY_LIKELY: Self = Self {
        numer: (u32::MAX as u64 * 99 / 100) as u32,
    };
    /// Very unlikely (probability ≈ 0.01).
    pub const VERY_UNLIKELY: Self = Self {
        numer: (u32::MAX as u64 / 100) as u32,
    };

    /// Create from a floating-point probability [0.0, 1.0].
    pub fn from_f64(p: f64) -> Self {
        let p = p.clamp(0.0, 1.0);
        Self {
            numer: (p * Self::DENOM as f64) as u32,
        }
    }

    /// Create from ratio (num/den).
    pub fn from_ratio(numerator: u64, denominator: u64) -> Self {
        if denominator == 0 {
            return Self::EVEN;
        }
        let scaled = (numerator as u128 * Self::DENOM as u128) / denominator as u128;
        Self {
            numer: scaled.min(u32::MAX as u128) as u32,
        }
    }

    /// Create from measured branch counts.
    pub fn from_counts(taken: u64, not_taken: u64) -> Self {
        let total = taken.saturating_add(not_taken);
        Self::from_ratio(taken, total)
    }

    /// Get as floating-point probability.
    pub fn as_f64(self) -> f64 {
        self.numer as f64 / Self::DENOM as f64
    }

    /// Get the complement (1 - p).
    pub fn complement(self) -> Self {
        Self {
            numer: u32::MAX - self.numer,
        }
    }

    /// Scale by a factor [0.0, 1.0].
    pub fn scale(self, factor: f64) -> Self {
        let factor = factor.clamp(0.0, 1.0);
        Self {
            numer: (self.numer as f64 * factor) as u32,
        }
    }

    /// Whether this probability is biased (away from 50/50).
    pub fn is_biased(self) -> bool {
        let diff = if self.numer > Self::EVEN.numer {
            self.numer - Self::EVEN.numer
        } else {
            Self::EVEN.numer - self.numer
        };
        // Consider biased if > 20% away from 50%
        diff > (u32::MAX / 5)
    }

    /// Whether this is a high probability (>= 0.8).
    pub fn is_likely(self) -> bool {
        self.numer >= (u32::MAX as u64 * 4 / 5) as u32
    }

    /// Whether this is a low probability (<= 0.2).
    pub fn is_unlikely(self) -> bool {
        self.numer <= (u32::MAX as u64 / 5) as u32
    }

    /// Raw numerator value.
    pub fn numerator(self) -> u32 {
        self.numer
    }
}

impl fmt::Display for BranchProbability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.as_f64())
    }
}

impl Default for BranchProbability {
    fn default() -> Self {
        Self::EVEN
    }
}

// =============================================================================
// Block Frequency
// =============================================================================

/// Estimated execution frequency relative to the function entry.
///
/// Entry block has frequency 1.0. Inner loops scale proportionally
/// to their estimated trip count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockFrequency {
    /// Frequency as a floating-point multiple of entry.
    freq: f64,
}

impl BlockFrequency {
    /// Entry block frequency (1.0).
    pub const ENTRY: Self = Self { freq: 1.0 };
    /// Cold block frequency (near-zero).
    pub const COLD: Self = Self { freq: 0.001 };

    /// Create from raw frequency value.
    pub fn new(freq: f64) -> Self {
        Self {
            freq: freq.max(0.0),
        }
    }

    /// Create from a loop with estimated trip count.
    pub fn for_loop(entry_freq: f64, trip_count: f64) -> Self {
        Self::new(entry_freq * trip_count)
    }

    /// Get the frequency value.
    pub fn value(self) -> f64 {
        self.freq
    }

    /// Whether this is a hot block (executed frequently).
    pub fn is_hot(self, threshold: f64) -> bool {
        self.freq >= threshold
    }

    /// Whether this is a cold block (rarely executed).
    pub fn is_cold(self) -> bool {
        self.freq < 0.01
    }

    /// Scale by a probability.
    pub fn scale(self, prob: BranchProbability) -> Self {
        Self::new(self.freq * prob.as_f64())
    }
}

impl Default for BlockFrequency {
    fn default() -> Self {
        Self::ENTRY
    }
}

impl fmt::Display for BlockFrequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}x", self.freq)
    }
}

// =============================================================================
// Static Heuristics
// =============================================================================

/// Static branch prediction heuristics for when no profile data is available.
///
/// Based on well-known heuristics from the literature (Ball & Larus, 1993)
/// and practical experience with Python programs.
pub struct StaticHeuristics;

impl StaticHeuristics {
    /// Exception/error path is unlikely.
    pub const EXCEPTION_UNLIKELY: BranchProbability = BranchProbability::VERY_UNLIKELY;

    /// Loop back-edges are likely taken (~90%).
    pub const LOOP_BRANCH: BranchProbability = BranchProbability {
        numer: (u32::MAX as u64 * 9 / 10) as u32,
    };

    /// Guard/check branches are likely to pass (~85%).
    pub const GUARD_LIKELY: BranchProbability = BranchProbability {
        numer: (u32::MAX as u64 * 85 / 100) as u32,
    };

    /// Pointer null checks — pointer is likely non-null (~95%).
    pub const NULL_CHECK_LIKELY: BranchProbability = BranchProbability {
        numer: (u32::MAX as u64 * 95 / 100) as u32,
    };

    /// Type check deopt — type check usually passes (~90%).
    pub const TYPE_CHECK_LIKELY: BranchProbability = BranchProbability {
        numer: (u32::MAX as u64 * 9 / 10) as u32,
    };

    /// Default unknown branch (50/50).
    pub const UNKNOWN: BranchProbability = BranchProbability::EVEN;

    /// Classify a branch by its pattern and return probability.
    pub fn classify(hint: BranchHint) -> BranchProbability {
        match hint {
            BranchHint::None => Self::UNKNOWN,
            BranchHint::LoopBack => Self::LOOP_BRANCH,
            BranchHint::Guard => Self::GUARD_LIKELY,
            BranchHint::Exception => Self::EXCEPTION_UNLIKELY,
            BranchHint::NullCheck => Self::NULL_CHECK_LIKELY,
            BranchHint::TypeCheck => Self::TYPE_CHECK_LIKELY,
            BranchHint::Likely => BranchProbability::LIKELY,
            BranchHint::Unlikely => BranchProbability::UNLIKELY,
        }
    }
}

/// Hint about the nature of a branch for static prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BranchHint {
    /// No hint available.
    None,
    /// Loop back-edge branch (likely taken).
    LoopBack,
    /// Guard/deopt branch (likely passes).
    Guard,
    /// Exception handling path (unlikely).
    Exception,
    /// Null pointer check (likely non-null).
    NullCheck,
    /// Type check (likely passes).
    TypeCheck,
    /// User-annotated likely.
    Likely,
    /// User-annotated unlikely.
    Unlikely,
}

// =============================================================================
// Branch Annotation Map
// =============================================================================

/// A map of branch bytecode offsets to their estimated probabilities.
///
/// This is the central data structure consumed by downstream passes
/// (code layout, register allocation hints, etc.).
#[derive(Debug, Clone)]
pub struct BranchAnnotations {
    /// Branch offset → probability of taken path.
    branches: FxHashMap<u32, BranchProbability>,
    /// Block ID → estimated frequency.
    block_freqs: FxHashMap<u32, BlockFrequency>,
    /// Whether annotations include profile-guided data.
    has_profile_data: bool,
}

impl BranchAnnotations {
    /// Create empty annotations.
    pub fn new() -> Self {
        Self {
            branches: FxHashMap::default(),
            block_freqs: FxHashMap::default(),
            has_profile_data: false,
        }
    }

    /// Set the branch probability for a given offset.
    pub fn set_branch(&mut self, offset: u32, prob: BranchProbability) {
        self.branches.insert(offset, prob);
    }

    /// Get the branch probability for a given offset.
    pub fn get_branch(&self, offset: u32) -> Option<BranchProbability> {
        self.branches.get(&offset).copied()
    }

    /// Get branch probability with fallback to static heuristic.
    pub fn get_branch_or_default(&self, offset: u32, hint: BranchHint) -> BranchProbability {
        self.branches
            .get(&offset)
            .copied()
            .unwrap_or_else(|| StaticHeuristics::classify(hint))
    }

    /// Set the block frequency for a given block ID.
    pub fn set_block_freq(&mut self, block_id: u32, freq: BlockFrequency) {
        self.block_freqs.insert(block_id, freq);
    }

    /// Get the block frequency for a given block ID.
    pub fn get_block_freq(&self, block_id: u32) -> Option<BlockFrequency> {
        self.block_freqs.get(&block_id).copied()
    }

    /// Check if a block is hot.
    pub fn is_block_hot(&self, block_id: u32, threshold: f64) -> bool {
        self.block_freqs
            .get(&block_id)
            .map_or(false, |f| f.is_hot(threshold))
    }

    /// Check if a block is cold.
    pub fn is_block_cold(&self, block_id: u32) -> bool {
        self.block_freqs
            .get(&block_id)
            .map_or(false, |f| f.is_cold())
    }

    /// Whether these annotations include profile-guided data.
    pub fn has_profile_data(&self) -> bool {
        self.has_profile_data
    }

    /// Mark as having profile-guided data.
    pub fn set_has_profile_data(&mut self, has: bool) {
        self.has_profile_data = has;
    }

    /// Number of annotated branches.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Number of annotated blocks.
    pub fn block_count(&self) -> usize {
        self.block_freqs.len()
    }

    /// Merge annotations from a profile, overwriting existing entries.
    pub fn merge_from_profile(&mut self, profile: &crate::runtime::profile_data::ProfileData) {
        self.has_profile_data = true;
        // Iterate profile branches and annotate
        for offset in profile.branch_offsets() {
            if let Some(bp) = profile.branch_at(offset) {
                let prob = BranchProbability::from_counts(bp.taken, bp.not_taken);
                self.branches.insert(offset, prob);
            }
        }
    }

    /// All annotated branch offsets.
    pub fn branch_offsets(&self) -> Vec<u32> {
        self.branches.keys().copied().collect()
    }

    /// Iterate over all branch annotations.
    pub fn iter_branches(&self) -> impl Iterator<Item = (u32, BranchProbability)> + '_ {
        self.branches.iter().map(|(&k, &v)| (k, v))
    }
}

impl Default for BranchAnnotations {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Branch Probability Pass (Pipeline Adapter)
// =============================================================================

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::operators::{ControlOp, Operator};

/// Pipeline-integrated branch probability annotation pass.
///
/// Walks the IR graph to identify branch nodes, then annotates each
/// with a probability derived from profile data (if available) or
/// static heuristics. Computed annotations are stored for downstream
/// passes (e.g., `HotColdPass`) to consume.
///
/// # Pipeline Phase
///
/// Runs in the `ProfileGuided` phase, after canonicalization has
/// normalized the IR and before local optimizations can leverage
/// the annotations.
pub struct BranchProbabilityPass {
    /// Computed annotations from the last run.
    annotations: BranchAnnotations,
    /// Number of branches annotated in the last run.
    branches_annotated: usize,
    /// Number of blocks with frequency estimates.
    blocks_estimated: usize,
}

impl BranchProbabilityPass {
    /// Create a new branch probability pass.
    pub fn new() -> Self {
        Self {
            annotations: BranchAnnotations::new(),
            branches_annotated: 0,
            blocks_estimated: 0,
        }
    }

    /// Get the computed branch annotations.
    pub fn annotations(&self) -> &BranchAnnotations {
        &self.annotations
    }

    /// Number of branches annotated in the last run.
    pub fn branches_annotated(&self) -> usize {
        self.branches_annotated
    }

    /// Number of blocks with frequency estimates.
    pub fn blocks_estimated(&self) -> usize {
        self.blocks_estimated
    }
}

impl Default for BranchProbabilityPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for BranchProbabilityPass {
    fn name(&self) -> &'static str {
        "BranchProbability"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        let mut annotations = BranchAnnotations::new();
        let mut changed = false;
        let mut branch_count = 0;
        let mut block_count = 0;

        // Walk the graph nodes looking for control-flow operations.
        // In a Sea-of-Nodes IR, branches are If nodes, loops are Loop nodes,
        // and merge points are Region nodes.
        for (id, node) in graph.iter() {
            let offset = id.index() as u32;

            match node.op {
                // Conditional branches get probability annotations
                Operator::Control(ControlOp::If) => {
                    // Default to even odds; profile merging will refine
                    annotations.set_branch(offset, BranchProbability::EVEN);
                    branch_count += 1;
                    changed = true;
                }
                // Loop headers get elevated frequency (estimated 10-trip count)
                Operator::Control(ControlOp::Loop) => {
                    annotations.set_block_freq(offset, BlockFrequency::for_loop(1.0, 10.0));
                    block_count += 1;
                    changed = true;
                }
                // Region (merge) nodes get entry-level frequency by default
                Operator::Control(ControlOp::Region) => {
                    annotations.set_block_freq(offset, BlockFrequency::ENTRY);
                    block_count += 1;
                }
                _ => {}
            }
        }

        self.annotations = annotations;
        self.branches_annotated = branch_count;
        self.blocks_estimated = block_count;
        changed
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BranchProbability Tests
    // =========================================================================

    #[test]
    fn test_probability_constants() {
        assert!((BranchProbability::ALWAYS.as_f64() - 1.0).abs() < 0.001);
        assert!((BranchProbability::NEVER.as_f64() - 0.0).abs() < 0.001);
        assert!((BranchProbability::EVEN.as_f64() - 0.5).abs() < 0.001);
        assert!((BranchProbability::LIKELY.as_f64() - 0.9).abs() < 0.01);
        assert!((BranchProbability::UNLIKELY.as_f64() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_probability_from_f64() {
        let p = BranchProbability::from_f64(0.75);
        assert!((p.as_f64() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_probability_from_f64_clamp() {
        let p1 = BranchProbability::from_f64(-0.5);
        assert!((p1.as_f64() - 0.0).abs() < 0.001);
        let p2 = BranchProbability::from_f64(1.5);
        assert!((p2.as_f64() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_probability_from_ratio() {
        let p = BranchProbability::from_ratio(3, 4);
        assert!((p.as_f64() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_probability_from_ratio_zero_denom() {
        let p = BranchProbability::from_ratio(3, 0);
        assert_eq!(p, BranchProbability::EVEN);
    }

    #[test]
    fn test_probability_from_counts() {
        let p = BranchProbability::from_counts(90, 10);
        assert!((p.as_f64() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_probability_from_counts_zero() {
        let p = BranchProbability::from_counts(0, 0);
        assert_eq!(p, BranchProbability::EVEN);
    }

    #[test]
    fn test_probability_complement() {
        let p = BranchProbability::from_f64(0.75);
        let c = p.complement();
        assert!((c.as_f64() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_probability_scale() {
        let p = BranchProbability::from_f64(0.8);
        let s = p.scale(0.5);
        assert!((s.as_f64() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_probability_is_biased() {
        assert!(BranchProbability::from_f64(0.9).is_biased());
        assert!(BranchProbability::from_f64(0.1).is_biased());
        assert!(!BranchProbability::from_f64(0.55).is_biased());
    }

    #[test]
    fn test_probability_is_likely() {
        assert!(BranchProbability::from_f64(0.85).is_likely());
        assert!(!BranchProbability::from_f64(0.5).is_likely());
    }

    #[test]
    fn test_probability_is_unlikely() {
        assert!(BranchProbability::from_f64(0.1).is_unlikely());
        assert!(!BranchProbability::from_f64(0.5).is_unlikely());
    }

    #[test]
    fn test_probability_display() {
        let p = BranchProbability::from_f64(0.9);
        let s = format!("{}", p);
        assert!(s.contains("0.9"));
    }

    #[test]
    fn test_probability_default() {
        assert_eq!(BranchProbability::default(), BranchProbability::EVEN);
    }

    #[test]
    fn test_probability_numerator() {
        let p = BranchProbability::ALWAYS;
        assert_eq!(p.numerator(), u32::MAX);
    }

    // =========================================================================
    // BlockFrequency Tests
    // =========================================================================

    #[test]
    fn test_block_frequency_entry() {
        assert!((BlockFrequency::ENTRY.value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_frequency_cold() {
        assert!(BlockFrequency::COLD.is_cold());
    }

    #[test]
    fn test_block_frequency_new() {
        let f = BlockFrequency::new(5.0);
        assert!((f.value() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_frequency_new_negative_clamped() {
        let f = BlockFrequency::new(-1.0);
        assert!((f.value() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_frequency_for_loop() {
        let f = BlockFrequency::for_loop(2.0, 100.0);
        assert!((f.value() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_frequency_is_hot() {
        let f = BlockFrequency::new(10.0);
        assert!(f.is_hot(5.0));
        assert!(!f.is_hot(15.0));
    }

    #[test]
    fn test_block_frequency_is_cold() {
        let f = BlockFrequency::new(0.005);
        assert!(f.is_cold());
    }

    #[test]
    fn test_block_frequency_scale() {
        let f = BlockFrequency::new(10.0);
        let scaled = f.scale(BranchProbability::from_f64(0.5));
        assert!((scaled.value() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_block_frequency_display() {
        let f = BlockFrequency::new(3.5);
        assert_eq!(format!("{}", f), "3.50x");
    }

    #[test]
    fn test_block_frequency_default() {
        let f = BlockFrequency::default();
        assert!((f.value() - 1.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // StaticHeuristics Tests
    // =========================================================================

    #[test]
    fn test_heuristic_exception() {
        let p = StaticHeuristics::classify(BranchHint::Exception);
        assert!(p.is_unlikely());
    }

    #[test]
    fn test_heuristic_loop_back() {
        let p = StaticHeuristics::classify(BranchHint::LoopBack);
        assert!(p.is_likely());
    }

    #[test]
    fn test_heuristic_guard() {
        let p = StaticHeuristics::classify(BranchHint::Guard);
        assert!(p.is_likely());
    }

    #[test]
    fn test_heuristic_null_check() {
        let p = StaticHeuristics::classify(BranchHint::NullCheck);
        assert!(p.is_likely());
    }

    #[test]
    fn test_heuristic_type_check() {
        let p = StaticHeuristics::classify(BranchHint::TypeCheck);
        assert!(p.is_likely());
    }

    #[test]
    fn test_heuristic_unknown() {
        let p = StaticHeuristics::classify(BranchHint::None);
        assert_eq!(p, BranchProbability::EVEN);
    }

    #[test]
    fn test_heuristic_likely_hint() {
        let p = StaticHeuristics::classify(BranchHint::Likely);
        assert!(p.is_likely());
    }

    #[test]
    fn test_heuristic_unlikely_hint() {
        let p = StaticHeuristics::classify(BranchHint::Unlikely);
        assert!(p.is_unlikely());
    }

    // =========================================================================
    // BranchAnnotations Tests
    // =========================================================================

    #[test]
    fn test_annotations_new() {
        let ann = BranchAnnotations::new();
        assert_eq!(ann.branch_count(), 0);
        assert_eq!(ann.block_count(), 0);
        assert!(!ann.has_profile_data());
    }

    #[test]
    fn test_annotations_default() {
        let ann = BranchAnnotations::default();
        assert_eq!(ann.branch_count(), 0);
    }

    #[test]
    fn test_annotations_set_get_branch() {
        let mut ann = BranchAnnotations::new();
        ann.set_branch(10, BranchProbability::LIKELY);
        assert_eq!(ann.get_branch(10), Some(BranchProbability::LIKELY));
        assert_eq!(ann.get_branch(20), None);
    }

    #[test]
    fn test_annotations_get_branch_or_default() {
        let mut ann = BranchAnnotations::new();
        ann.set_branch(10, BranchProbability::from_f64(0.95));

        // With recorded data
        let bp = ann.get_branch_or_default(10, BranchHint::None);
        assert!((bp.as_f64() - 0.95).abs() < 0.01);

        // Without recorded data, falls back to heuristic
        let bp2 = ann.get_branch_or_default(99, BranchHint::LoopBack);
        assert!(bp2.is_likely());
    }

    #[test]
    fn test_annotations_set_get_block_freq() {
        let mut ann = BranchAnnotations::new();
        ann.set_block_freq(1, BlockFrequency::new(10.0));
        assert!((ann.get_block_freq(1).unwrap().value() - 10.0).abs() < f64::EPSILON);
        assert!(ann.get_block_freq(999).is_none());
    }

    #[test]
    fn test_annotations_is_block_hot() {
        let mut ann = BranchAnnotations::new();
        ann.set_block_freq(1, BlockFrequency::new(10.0));
        assert!(ann.is_block_hot(1, 5.0));
        assert!(!ann.is_block_hot(1, 15.0));
        assert!(!ann.is_block_hot(999, 1.0));
    }

    #[test]
    fn test_annotations_is_block_cold() {
        let mut ann = BranchAnnotations::new();
        ann.set_block_freq(1, BlockFrequency::new(0.005));
        assert!(ann.is_block_cold(1));
        assert!(!ann.is_block_cold(999));
    }

    #[test]
    fn test_annotations_merge_from_profile() {
        let mut profile = crate::runtime::profile_data::ProfileData::new(1);
        profile.record_branch(10, true);
        profile.record_branch(10, true);
        profile.record_branch(10, false);

        let mut ann = BranchAnnotations::new();
        ann.merge_from_profile(&profile);

        assert!(ann.has_profile_data());
        assert_eq!(ann.branch_count(), 1);
        let bp = ann.get_branch(10).unwrap();
        // 2 taken, 1 not taken → ~0.667
        assert!((bp.as_f64() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_annotations_branch_offsets() {
        let mut ann = BranchAnnotations::new();
        ann.set_branch(10, BranchProbability::LIKELY);
        ann.set_branch(20, BranchProbability::UNLIKELY);
        let offsets = ann.branch_offsets();
        assert_eq!(offsets.len(), 2);
        assert!(offsets.contains(&10));
        assert!(offsets.contains(&20));
    }

    #[test]
    fn test_annotations_iter_branches() {
        let mut ann = BranchAnnotations::new();
        ann.set_branch(10, BranchProbability::LIKELY);
        ann.set_branch(20, BranchProbability::UNLIKELY);
        let items: Vec<_> = ann.iter_branches().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_annotations_profile_data_flag() {
        let mut ann = BranchAnnotations::new();
        assert!(!ann.has_profile_data());
        ann.set_has_profile_data(true);
        assert!(ann.has_profile_data());
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_probability_extreme_counts() {
        let p = BranchProbability::from_counts(u64::MAX / 2, u64::MAX / 2);
        assert!((p.as_f64() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_probability_one_count() {
        let p = BranchProbability::from_counts(1, 0);
        assert!((p.as_f64() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_frequency_very_high() {
        let f = BlockFrequency::new(1_000_000.0);
        assert!(f.is_hot(1.0));
        assert!(!f.is_cold());
    }

    #[test]
    fn test_probability_complement_round_trip() {
        let p = BranchProbability::from_f64(0.3);
        let c = p.complement();
        let cc = c.complement();
        // Note: there's a ±1 quantization error, so we check near-equality
        assert!((p.as_f64() - cc.as_f64()).abs() < 0.001);
    }

    #[test]
    fn test_multiple_profile_merge() {
        let mut profile1 = crate::runtime::profile_data::ProfileData::new(1);
        profile1.record_branch(10, true);
        profile1.record_branch(10, true);

        let mut profile2 = crate::runtime::profile_data::ProfileData::new(1);
        profile2.record_branch(10, false);
        profile2.record_branch(20, true);

        let mut ann = BranchAnnotations::new();
        ann.merge_from_profile(&profile1);
        ann.merge_from_profile(&profile2);

        // Profile2 overwrites profile1 for offset 10
        assert_eq!(ann.branch_count(), 2);
    }
}
