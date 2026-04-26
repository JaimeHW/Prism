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
    /// Optional profile data for PGO-guided annotation.
    profile_data: Option<crate::runtime::profile_data::ProfileData>,
}

impl BranchProbabilityPass {
    /// Create a new branch probability pass.
    pub fn new() -> Self {
        Self {
            annotations: BranchAnnotations::new(),
            branches_annotated: 0,
            blocks_estimated: 0,
            profile_data: None,
        }
    }

    /// Create with attached profile data for PGO.
    pub fn with_profile(profile: crate::runtime::profile_data::ProfileData) -> Self {
        Self {
            annotations: BranchAnnotations::new(),
            branches_annotated: 0,
            blocks_estimated: 0,
            profile_data: Some(profile),
        }
    }

    /// Inject profile data after construction.
    pub fn inject_profile(&mut self, profile: crate::runtime::profile_data::ProfileData) {
        self.profile_data = Some(profile);
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
                    // Default to even odds; profile merging below will refine
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

        // Merge PGO-measured probabilities, overriding static defaults
        if let Some(ref profile) = self.profile_data {
            annotations.merge_from_profile(profile);

            // Refine loop frequencies from measured iteration counts
            for (id, node) in graph.iter() {
                let offset = id.index() as u32;
                if matches!(node.op, Operator::Control(ControlOp::Loop)) {
                    let trip_count = profile.loop_count(offset);
                    if trip_count > 0 {
                        let exec_count = profile.execution_count().max(1);
                        let avg_trips = trip_count as f64 / exec_count as f64;
                        annotations.set_block_freq(
                            offset,
                            BlockFrequency::for_loop(1.0, avg_trips.max(1.0)),
                        );
                    }
                }
            }
        }

        self.annotations = annotations;
        self.branches_annotated = branch_count;
        self.blocks_estimated = block_count;
        changed
    }
}
