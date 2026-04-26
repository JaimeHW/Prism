//! Speculation bridge for PGO-guided JIT compilation.
//!
//! Bridges raw `ProfileData` into type-safe speculation decisions consumed
//! by the optimization pipeline and bytecode translator. Provides the
//! `SpeculationProvider` trait and a concrete `PgoSpeculationProvider`.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
//! │ ProfileData  │────▶│ PgoSpeculationProv.  │────▶│ Opt Pipeline │
//! │ (raw counts) │     │ (typed decisions)    │     │ (branch/call │
//! └──────────────┘     └─────────────────────┘     │  annotations)│
//!                                                   └──────────────┘
//! ```

use crate::opt::branch_probability::BranchProbability;
use crate::runtime::profile_data::{CallProfile, ProfileData, TypeProfile};

// =============================================================================
// PGO Branch Hint
// =============================================================================

/// PGO-derived branch probability hint for a single branch site.
///
/// Uses fixed-point representation (numerator over 2^32) for zero-allocation
/// branch probability queries in the compilation hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoBranchHint {
    /// Bytecode offset of the branch instruction.
    pub offset: u32,
    /// Probability of the taken path as fixed-point numerator (denom = 2^32).
    pub taken_numer: u32,
}

impl PgoBranchHint {
    /// Create a new branch hint from measured counts.
    #[inline]
    pub fn from_counts(offset: u32, taken: u64, not_taken: u64) -> Self {
        let prob = BranchProbability::from_counts(taken, not_taken);
        Self {
            offset,
            taken_numer: prob.numerator(),
        }
    }

    /// Create from a pre-computed probability.
    #[inline]
    pub fn from_probability(offset: u32, prob: BranchProbability) -> Self {
        Self {
            offset,
            taken_numer: prob.numerator(),
        }
    }

    /// Convert to `BranchProbability`.
    #[inline]
    pub fn as_branch_probability(self) -> BranchProbability {
        BranchProbability::from_ratio(self.taken_numer as u64, u32::MAX as u64)
    }

    /// Whether the branch is likely taken (>= 80%).
    #[inline]
    pub fn is_likely_taken(self) -> bool {
        self.taken_numer >= (u32::MAX as u64 * 4 / 5) as u32
    }

    /// Whether the branch is likely not taken (<= 20%).
    #[inline]
    pub fn is_unlikely_taken(self) -> bool {
        self.taken_numer <= (u32::MAX as u64 / 5) as u32
    }

    /// Whether the hint is biased away from 50/50.
    #[inline]
    pub fn is_biased(self) -> bool {
        let mid = u32::MAX / 2;
        let diff = if self.taken_numer > mid {
            self.taken_numer - mid
        } else {
            mid - self.taken_numer
        };
        diff > (u32::MAX / 5)
    }

    /// Taken probability as f64 [0.0, 1.0].
    #[inline]
    pub fn taken_f64(self) -> f64 {
        self.taken_numer as f64 / u32::MAX as f64
    }
}

// =============================================================================
// PGO Call Target
// =============================================================================

/// A single observed call target at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoCallTarget {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// Unique identifier of the callee code unit.
    pub target_id: u32,
    /// Call frequency (raw count).
    pub frequency: u64,
}

// =============================================================================
// Call Site Profile
// =============================================================================

/// Aggregated call-site profile from PGO data.
///
/// Contains all observed call targets for a single bytecode offset,
/// sorted by frequency (most frequent first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSiteProfile {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// Targets sorted by frequency (descending).
    pub targets: Vec<PgoCallTarget>,
    /// Total calls observed at this site.
    pub total_calls: u64,
}

/// Classification of a call site's polymorphism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallMorphism {
    /// No calls observed.
    Cold,
    /// Single target (>95% of calls).
    Monomorphic,
    /// 2-4 significant targets.
    Polymorphic,
    /// 5+ significant targets; too many to speculate on.
    Megamorphic,
}

impl CallSiteProfile {
    /// Create from a `CallProfile` at a given bytecode offset.
    pub fn from_call_profile(offset: u32, cp: &CallProfile) -> Self {
        let targets: Vec<PgoCallTarget> = cp
            .targets()
            .iter()
            .map(|t| PgoCallTarget {
                offset,
                target_id: t.target_id,
                frequency: t.count,
            })
            .collect();

        Self {
            offset,
            targets,
            total_calls: cp.total(),
        }
    }

    /// Classify the call site's morphism.
    #[inline]
    pub fn morphism(&self) -> CallMorphism {
        if self.total_calls == 0 {
            return CallMorphism::Cold;
        }

        let significant = self
            .targets
            .iter()
            .filter(|t| t.frequency as f64 / self.total_calls.max(1) as f64 > 0.01)
            .count();

        match significant {
            0 => CallMorphism::Cold,
            1 => {
                // Check >95% dominance for true monomorphism
                if let Some(primary) = self.targets.first() {
                    if primary.frequency as f64 / self.total_calls as f64 > 0.95 {
                        CallMorphism::Monomorphic
                    } else {
                        CallMorphism::Polymorphic
                    }
                } else {
                    CallMorphism::Cold
                }
            }
            2..=4 => CallMorphism::Polymorphic,
            _ => CallMorphism::Megamorphic,
        }
    }

    /// Get the dominant (most frequently called) target.
    #[inline]
    pub fn dominant_target(&self) -> Option<&PgoCallTarget> {
        self.targets.first()
    }

    /// Whether inline speculation is worthwhile for this site.
    #[inline]
    pub fn is_inline_candidate(&self) -> bool {
        matches!(self.morphism(), CallMorphism::Monomorphic)
    }
}

// =============================================================================
// Type Hint from PGO
// =============================================================================

/// PGO-derived type hint for an operation site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PgoTypeHint {
    /// Monomorphic: single type observed >95% of the time.
    Monomorphic {
        /// The dominant type ID.
        type_id: u8,
    },
    /// Polymorphic: 2-4 types with significant frequency.
    Polymorphic,
    /// Megamorphic: 5+ types; too many to speculate on.
    Megamorphic,
    /// No profile data available for this site.
    NoData,
}

impl PgoTypeHint {
    /// Derive from a `TypeProfile`.
    pub fn from_type_profile(tp: &TypeProfile) -> Self {
        if tp.total() < 10 {
            return Self::NoData;
        }
        if tp.is_monomorphic() {
            if let Some(entry) = tp.dominant_type() {
                return Self::Monomorphic {
                    type_id: entry.type_id,
                };
            }
        }
        if tp.is_megamorphic() {
            return Self::Megamorphic;
        }
        if tp.is_polymorphic() {
            return Self::Polymorphic;
        }
        Self::NoData
    }

    /// Whether speculation on a single type is safe.
    #[inline]
    pub fn is_specialize_candidate(self) -> bool {
        matches!(self, Self::Monomorphic { .. })
    }
}

// =============================================================================
// Speculation Provider Trait
// =============================================================================

/// Central interface for querying PGO speculation information during
/// JIT compilation.
///
/// This trait abstracts over the source of profile data, enabling
/// both real PGO-backed providers and mock providers for testing.
pub trait SpeculationProvider {
    /// Get a branch probability hint for the given bytecode offset.
    fn get_branch_hint(&self, offset: u32) -> Option<PgoBranchHint>;

    /// Get call site profile for the given bytecode offset.
    fn get_call_targets(&self, offset: u32) -> Option<CallSiteProfile>;

    /// Get a type hint for the given bytecode offset.
    fn get_type_hint(&self, offset: u32) -> PgoTypeHint;

    /// Whether any profile data is available.
    fn has_profile_data(&self) -> bool;

    /// Total execution count of the code unit.
    fn execution_count(&self) -> u64;
}

// =============================================================================
// PGO-Backed Speculation Provider
// =============================================================================

/// Concrete `SpeculationProvider` backed by `ProfileData`.
///
/// This is the primary implementation used during Tier 2 JIT compilation.
/// It wraps a `ProfileData` snapshot and translates raw profile counts
/// into typed speculation decisions.
#[derive(Debug, Clone)]
pub struct PgoSpeculationProvider {
    /// The underlying profile data.
    profile: ProfileData,
}

impl PgoSpeculationProvider {
    /// Create a new provider from profile data.
    #[inline]
    pub fn new(profile: ProfileData) -> Self {
        Self { profile }
    }

    /// Borrow the underlying profile data.
    #[inline]
    pub fn profile(&self) -> &ProfileData {
        &self.profile
    }
}

impl SpeculationProvider for PgoSpeculationProvider {
    #[inline]
    fn get_branch_hint(&self, offset: u32) -> Option<PgoBranchHint> {
        self.profile
            .branch_at(offset)
            .map(|bp| PgoBranchHint::from_counts(offset, bp.taken, bp.not_taken))
    }

    fn get_call_targets(&self, offset: u32) -> Option<CallSiteProfile> {
        self.profile
            .call_at(offset)
            .map(|cp| CallSiteProfile::from_call_profile(offset, cp))
    }

    fn get_type_hint(&self, offset: u32) -> PgoTypeHint {
        self.profile
            .type_at(offset)
            .map_or(PgoTypeHint::NoData, |tp| PgoTypeHint::from_type_profile(tp))
    }

    #[inline]
    fn has_profile_data(&self) -> bool {
        true
    }

    #[inline]
    fn execution_count(&self) -> u64 {
        self.profile.execution_count()
    }
}

// =============================================================================
// Empty Speculation Provider (no PGO data)
// =============================================================================

/// A no-op speculation provider used when no profile data is available.
///
/// Returns `None`/`NoData` for all queries. Used for baseline compilation
/// or when PGO is disabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSpeculation;

impl SpeculationProvider for NoSpeculation {
    #[inline]
    fn get_branch_hint(&self, _offset: u32) -> Option<PgoBranchHint> {
        None
    }

    #[inline]
    fn get_call_targets(&self, _offset: u32) -> Option<CallSiteProfile> {
        None
    }

    #[inline]
    fn get_type_hint(&self, _offset: u32) -> PgoTypeHint {
        PgoTypeHint::NoData
    }

    #[inline]
    fn has_profile_data(&self) -> bool {
        false
    }

    #[inline]
    fn execution_count(&self) -> u64 {
        0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
