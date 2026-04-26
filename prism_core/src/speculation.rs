//! Speculation hints for JIT compilation.
//!
//! This module defines type speculation hints that can be shared between
//! the VM (which collects type feedback) and the JIT (which uses it for
//! type-specialized code generation).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
//! │  Interpreter  │────▶│ ProfileCollector  │────▶│ SpeculationProvider │
//! │  (feedback)   │     │ (atomic counters) │     │ (query interface)  │
//! └──────────────┘     └──────────────────┘     └──────────────────┘
//!                                                       │
//!                              ┌────────────────────────┤
//!                              ▼                        ▼
//!                        ┌──────────┐          ┌──────────────┐
//!                        │TypeHints │          │BranchHints   │
//!                        └──────────┘          │CallTargets   │
//!                                              └──────────────┘
//! ```

// =============================================================================
// Type Hint
// =============================================================================

/// Type speculation hint for binary operations.
///
/// These hints are derived from runtime type feedback and inform the JIT
/// about observed operand types to enable speculative compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TypeHint {
    /// No type information available (cold or polymorphic site).
    #[default]
    None = 0,
    /// Both operands are integers.
    IntInt = 1,
    /// Both operands are floats.
    FloatFloat = 2,
    /// Left operand is int, right is float.
    IntFloat = 3,
    /// Left operand is float, right is int.
    FloatInt = 4,
    /// Both operands are strings.
    StrStr = 5,
    /// String and int (for repetition).
    StrInt = 6,
    /// Int and string (for repetition).
    IntStr = 7,
    /// Both operands are lists (concatenation).
    ListList = 8,
}

impl TypeHint {
    /// Check if this hint suggests integer operations.
    #[inline]
    pub const fn is_int(self) -> bool {
        matches!(self, Self::IntInt)
    }

    /// Check if this hint suggests float operations.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::FloatFloat | Self::IntFloat | Self::FloatInt)
    }

    /// Check if this hint suggests string operations.
    #[inline]
    pub const fn is_string(self) -> bool {
        matches!(self, Self::StrStr | Self::StrInt | Self::IntStr)
    }

    /// Check if this hint is valid (not None).
    #[inline]
    pub const fn is_valid(self) -> bool {
        !matches!(self, Self::None)
    }
}

// =============================================================================
// PGO Branch Hint
// =============================================================================

/// A branch prediction hint derived from profile-guided data.
///
/// Associates a bytecode offset with a measured branch-taken probability,
/// expressed as a fixed-point fraction (numerator over 2^32). This avoids
/// floating-point arithmetic on the critical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoBranchHint {
    /// Bytecode offset of the branch instruction.
    pub offset: u32,
    /// Probability that the branch is taken, as a numerator over 2^32.
    ///
    /// - `0` = never taken
    /// - `u32::MAX` = always taken
    /// - `u32::MAX / 2` ≈ 50/50
    pub taken_numer: u32,
}

impl PgoBranchHint {
    /// Denominator (2^32 represented as u64 for ratio calculations).
    const DENOM: u64 = 1u64 << 32;

    /// Create a new branch hint.
    #[inline]
    pub const fn new(offset: u32, taken_numer: u32) -> Self {
        Self {
            offset,
            taken_numer,
        }
    }

    /// Create from measured branch counts.
    ///
    /// Computes `taken / (taken + not_taken)` as a fixed-point fraction.
    #[inline]
    pub fn from_counts(offset: u32, taken: u64, not_taken: u64) -> Self {
        let total = taken.saturating_add(not_taken);
        let numer = if total == 0 {
            u32::MAX / 2 // Default to 50/50 for no data
        } else {
            ((taken as u128 * Self::DENOM as u128) / total as u128).min(u32::MAX as u128) as u32
        };
        Self {
            offset,
            taken_numer: numer,
        }
    }

    /// Create from a floating-point probability [0.0, 1.0].
    #[inline]
    pub fn from_f64(offset: u32, prob: f64) -> Self {
        let clamped = prob.clamp(0.0, 1.0);
        let numer = (clamped * Self::DENOM as f64) as u32;
        Self {
            offset,
            taken_numer: numer,
        }
    }

    /// Get the taken probability as a float.
    #[inline]
    pub fn taken_probability(&self) -> f64 {
        self.taken_numer as f64 / Self::DENOM as f64
    }

    /// Get the not-taken probability as a float.
    #[inline]
    pub fn not_taken_probability(&self) -> f64 {
        1.0 - self.taken_probability()
    }

    /// Whether the branch is biased (significantly away from 50/50).
    ///
    /// A branch is considered biased if its probability deviates more
    /// than 20 percentage points from 50%.
    #[inline]
    pub fn is_biased(&self) -> bool {
        let mid = u32::MAX / 2;
        let threshold = u32::MAX / 5; // ~20%
        self.taken_numer > mid.saturating_add(threshold)
            || self.taken_numer < mid.saturating_sub(threshold)
    }

    /// Whether the branch is likely taken (probability >= 80%).
    #[inline]
    pub fn is_likely_taken(&self) -> bool {
        self.taken_numer >= (u32::MAX as u64 * 4 / 5) as u32
    }

    /// Whether the branch is unlikely taken (probability <= 20%).
    #[inline]
    pub fn is_unlikely_taken(&self) -> bool {
        self.taken_numer <= (u32::MAX as u64 / 5) as u32
    }

    /// Complement: returns a hint with the opposite probability.
    #[inline]
    pub fn complement(&self) -> Self {
        Self {
            offset: self.offset,
            taken_numer: u32::MAX.wrapping_sub(self.taken_numer),
        }
    }
}

// =============================================================================
// PGO Call Target
// =============================================================================

/// A call target observed via profile-guided data.
///
/// Records which function was called at a particular bytecode offset
/// and how frequently. Used for PGO-guided inlining decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoCallTarget {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// Unique identifier for the called function's code unit.
    pub target_id: u32,
    /// Number of times this target was observed at this call site.
    pub frequency: u32,
}

impl PgoCallTarget {
    /// Create a new call target.
    #[inline]
    pub const fn new(offset: u32, target_id: u32, frequency: u32) -> Self {
        Self {
            offset,
            target_id,
            frequency,
        }
    }

    /// Whether this is a monomorphic call site (single target observed).
    ///
    /// This is a hint — the caller should check that there are no other
    /// targets at the same offset.
    #[inline]
    pub fn is_frequent(&self, threshold: u32) -> bool {
        self.frequency >= threshold
    }
}

// =============================================================================
// Call Site Profile
// =============================================================================

/// Aggregated call target profile for a single call site.
///
/// Summarizes all observed targets at a bytecode offset, including
/// polymorphism classification used for inlining heuristics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSiteProfile {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// All observed call targets, sorted by frequency (descending).
    pub targets: Vec<PgoCallTarget>,
    /// Total call count at this site.
    pub total_calls: u64,
}

impl CallSiteProfile {
    /// Create a new call site profile.
    pub fn new(offset: u32) -> Self {
        Self {
            offset,
            targets: Vec::new(),
            total_calls: 0,
        }
    }

    /// Add a target observation.
    pub fn add_target(&mut self, target_id: u32, frequency: u32) {
        self.targets
            .push(PgoCallTarget::new(self.offset, target_id, frequency));
        self.total_calls += frequency as u64;
        // Maintain descending frequency order
        self.targets
            .sort_unstable_by(|a, b| b.frequency.cmp(&a.frequency));
    }

    /// The morphism classification of this call site.
    #[inline]
    pub fn morphism(&self) -> CallMorphism {
        match self.targets.len() {
            0 => CallMorphism::Cold,
            1 => CallMorphism::Monomorphic,
            2..=4 => CallMorphism::Polymorphic,
            _ => CallMorphism::Megamorphic,
        }
    }

    /// The most frequently called target, if any.
    #[inline]
    pub fn dominant_target(&self) -> Option<&PgoCallTarget> {
        self.targets.first()
    }

    /// Whether a single target dominates (>= threshold% of calls).
    pub fn has_dominant_target(&self, threshold_pct: u32) -> bool {
        if let Some(target) = self.dominant_target() {
            if self.total_calls == 0 {
                return false;
            }
            let pct = (target.frequency as u64 * 100) / self.total_calls;
            pct >= threshold_pct as u64
        } else {
            false
        }
    }

    /// Number of distinct targets observed.
    #[inline]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }
}

/// Call site morphism classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallMorphism {
    /// No calls observed at this site.
    Cold,
    /// Single target — ideal for inlining.
    Monomorphic,
    /// Few targets (2-4) — may benefit from inline cache / dispatch table.
    Polymorphic,
    /// Many targets (5+) — unlikely to benefit from specialization.
    Megamorphic,
}

// =============================================================================
// Speculation Provider Trait
// =============================================================================

/// Provider of type speculation hints for JIT compilation.
///
/// This trait allows the JIT to query speculation information without
/// direct dependency on VM internals.
pub trait SpeculationProvider {
    /// Get the type hint for a specific bytecode site.
    ///
    /// # Arguments
    /// * `code_id` - Unique identifier for the compiled code unit
    /// * `bc_offset` - Bytecode offset within the code unit
    ///
    /// # Returns
    /// The observed type hint, or `TypeHint::None` if no information is available.
    fn get_type_hint(&self, code_id: u32, bc_offset: u32) -> TypeHint;

    /// Get the PGO branch hint for a specific branch site.
    ///
    /// Returns `None` if no profile data is available for this branch.
    fn get_branch_hint(&self, code_id: u32, bc_offset: u32) -> Option<PgoBranchHint> {
        let _ = (code_id, bc_offset);
        None
    }

    /// Get all branch hints for a code unit.
    ///
    /// Returns an empty vector if no profile data is available.
    fn get_all_branch_hints(&self, code_id: u32) -> Vec<PgoBranchHint> {
        let _ = code_id;
        Vec::new()
    }

    /// Get the call site profile for a specific call site.
    ///
    /// Returns `None` if no profile data is available for this call site.
    fn get_call_targets(&self, code_id: u32, bc_offset: u32) -> Option<CallSiteProfile> {
        let _ = (code_id, bc_offset);
        None
    }

    /// Whether profile-guided data is available for this code unit.
    fn has_profile_data(&self, code_id: u32) -> bool {
        let _ = code_id;
        false
    }

    /// Get the execution count for a code unit (0 if unknown).
    fn execution_count(&self, code_id: u32) -> u64 {
        let _ = code_id;
        0
    }
}

// =============================================================================
// No-Op Provider
// =============================================================================

/// No-op speculation provider that always returns None.
///
/// Useful for testing or when speculation data is not available.
#[derive(Debug, Default)]
pub struct NoSpeculation;

impl SpeculationProvider for NoSpeculation {
    #[inline]
    fn get_type_hint(&self, _code_id: u32, _bc_offset: u32) -> TypeHint {
        TypeHint::None
    }
}

// =============================================================================
// Static Speculation Provider
// =============================================================================

/// A simple speculation provider backed by pre-populated static data.
///
/// Useful for testing and benchmarking without a live interpreter.
#[derive(Debug, Default)]
pub struct StaticSpeculation {
    /// Type hints: (code_id, bc_offset) → TypeHint
    type_hints: Vec<(u32, u32, TypeHint)>,
    /// Branch hints: (code_id, bc_offset) → PgoBranchHint
    branch_hints: Vec<(u32, PgoBranchHint)>,
    /// Call targets: (code_id, bc_offset) → CallSiteProfile
    call_profiles: Vec<(u32, CallSiteProfile)>,
    /// Code units with profile data
    profiled_units: Vec<u32>,
    /// Execution counts: code_id → count
    exec_counts: Vec<(u32, u64)>,
}

impl StaticSpeculation {
    /// Create a new empty static speculation provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a type hint.
    pub fn add_type_hint(&mut self, code_id: u32, bc_offset: u32, hint: TypeHint) {
        self.type_hints.push((code_id, bc_offset, hint));
    }

    /// Add a branch hint.
    pub fn add_branch_hint(&mut self, code_id: u32, hint: PgoBranchHint) {
        if !self.profiled_units.contains(&code_id) {
            self.profiled_units.push(code_id);
        }
        self.branch_hints.push((code_id, hint));
    }

    /// Add a call site profile.
    pub fn add_call_profile(&mut self, code_id: u32, profile: CallSiteProfile) {
        if !self.profiled_units.contains(&code_id) {
            self.profiled_units.push(code_id);
        }
        self.call_profiles.push((code_id, profile));
    }

    /// Set the execution count for a code unit.
    pub fn set_execution_count(&mut self, code_id: u32, count: u64) {
        if !self.profiled_units.contains(&code_id) {
            self.profiled_units.push(code_id);
        }
        self.exec_counts.push((code_id, count));
    }
}

impl SpeculationProvider for StaticSpeculation {
    fn get_type_hint(&self, code_id: u32, bc_offset: u32) -> TypeHint {
        self.type_hints
            .iter()
            .find(|(cid, off, _)| *cid == code_id && *off == bc_offset)
            .map(|(_, _, hint)| *hint)
            .unwrap_or(TypeHint::None)
    }

    fn get_branch_hint(&self, code_id: u32, bc_offset: u32) -> Option<PgoBranchHint> {
        self.branch_hints
            .iter()
            .find(|(cid, hint)| *cid == code_id && hint.offset == bc_offset)
            .map(|(_, hint)| *hint)
    }

    fn get_all_branch_hints(&self, code_id: u32) -> Vec<PgoBranchHint> {
        self.branch_hints
            .iter()
            .filter(|(cid, _)| *cid == code_id)
            .map(|(_, hint)| *hint)
            .collect()
    }

    fn get_call_targets(&self, code_id: u32, bc_offset: u32) -> Option<CallSiteProfile> {
        self.call_profiles
            .iter()
            .find(|(cid, profile)| *cid == code_id && profile.offset == bc_offset)
            .map(|(_, profile)| profile.clone())
    }

    fn has_profile_data(&self, code_id: u32) -> bool {
        self.profiled_units.contains(&code_id)
    }

    fn execution_count(&self, code_id: u32) -> u64 {
        self.exec_counts
            .iter()
            .find(|(cid, _)| *cid == code_id)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
