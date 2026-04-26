//! Range Check Elimination Logic.
//!
//! This module implements the actual elimination of range checks based on
//! induction variable analysis. It determines which checks can be safely
//! removed based on proven invariants.
//!
//! # Elimination Strategies
//!
//! ## Lower Bound Elimination
//! A lower bound check `iv >= 0` can be eliminated if:
//! - `init >= 0` AND `direction == Increasing`
//! - This proves the IV never goes negative
//!
//! ## Upper Bound Elimination  
//! An upper bound check `iv < length` can be eliminated if:
//! - `init < length` AND `max_value < length`
//! - This requires computing the maximum value the IV can reach
//!
//! ## Widening Strategy
//! Instead of eliminating, we can "widen" the check:
//! - Move a single check to the loop preheader
//! - Check: `init >= 0 && (init + (trip_count - 1) * step) < length`
//! - This turns N checks into 1 check

use super::bounds::{BoundValue, RangeCheck, RangeCheckKind};
use super::induction::{InductionDirection, InductionInit, InductionStep, InductionVariable};
use crate::ir::node::NodeId;

// =============================================================================
// Elimination Decision
// =============================================================================

/// Decision about what to do with a range check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EliminationDecision {
    /// Keep the check as-is (cannot eliminate).
    Keep,

    /// Eliminate the check entirely (provably safe).
    Eliminate,

    /// Hoist the check to loop preheader (single check instead of N).
    Hoist(HoistInfo),

    /// Replace with a widened check in preheader.
    Widen(WidenInfo),
}

/// Information for hoisting a check to preheader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoistInfo {
    /// The original guard node.
    pub guard: NodeId,

    /// The induction variable.
    pub iv: NodeId,
}

/// Information for widening a check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WidenInfo {
    /// The original guard node.
    pub guard: NodeId,

    /// The induction variable.
    pub iv: NodeId,

    /// Minimum value to check.
    pub min_check: WidenBound,

    /// Maximum value to check.
    pub max_check: WidenBound,
}

/// A widened bound check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WidenBound {
    /// Check against a constant.
    Constant(i64),

    /// Check against a node value.
    Node(NodeId),

    /// Computed from IV properties.
    Computed {
        /// Base value (init or final).
        base: NodeId,
        /// Offset to add.
        offset: i64,
    },
}

// =============================================================================
// Elimination Analyzer
// =============================================================================

/// Analyzes range checks and determines elimination decisions.
#[derive(Debug)]
pub struct EliminationAnalyzer {
    /// Statistics.
    stats: EliminationStats,

    /// Enable aggressive elimination (may speculatively eliminate).
    aggressive: bool,
}

/// Statistics from elimination analysis.
#[derive(Debug, Clone, Default)]
pub struct EliminationStats {
    /// Number of checks analyzed.
    pub analyzed: usize,

    /// Number of checks that can be eliminated.
    pub eliminable: usize,

    /// Number of checks that can be hoisted.
    pub hoistable: usize,

    /// Number of checks that can be widened.
    pub widenable: usize,

    /// Number of checks that must be kept.
    pub kept: usize,
}

impl EliminationAnalyzer {
    /// Create a new elimination analyzer.
    #[inline]
    pub fn new() -> Self {
        Self {
            stats: EliminationStats::default(),
            aggressive: false,
        }
    }

    /// Create an aggressive analyzer.
    #[inline]
    pub fn aggressive() -> Self {
        Self {
            stats: EliminationStats::default(),
            aggressive: true,
        }
    }

    /// Get statistics.
    #[inline]
    pub fn stats(&self) -> &EliminationStats {
        &self.stats
    }

    /// Reset statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats = EliminationStats::default();
    }

    /// Analyze a range check and determine what to do with it.
    pub fn analyze(&mut self, check: &RangeCheck, iv: &InductionVariable) -> EliminationDecision {
        self.stats.analyzed += 1;

        let decision = match check.kind {
            RangeCheckKind::LowerBound => self.analyze_lower_bound(check, iv),
            RangeCheckKind::UpperBound => self.analyze_upper_bound(check, iv),
            RangeCheckKind::UpperBoundInclusive => self.analyze_upper_bound_inclusive(check, iv),
        };

        // Update stats
        match &decision {
            EliminationDecision::Keep => self.stats.kept += 1,
            EliminationDecision::Eliminate => self.stats.eliminable += 1,
            EliminationDecision::Hoist(_) => self.stats.hoistable += 1,
            EliminationDecision::Widen(_) => self.stats.widenable += 1,
        }

        decision
    }

    /// Analyze a lower bound check (iv >= 0).
    fn analyze_lower_bound(
        &self,
        check: &RangeCheck,
        iv: &InductionVariable,
    ) -> EliminationDecision {
        // Case 1: Constant init >= 0 and increasing direction
        // The IV starts non-negative and only increases, so always >= 0
        if let InductionInit::Constant(init) = iv.init {
            if init >= 0 && matches!(iv.direction, InductionDirection::Increasing) {
                return EliminationDecision::Eliminate;
            }
        }

        // Case 2: Constant init >= 0 and constant positive step
        // Even with unknown direction, if init >= 0 and step > 0, we're safe
        if let (InductionInit::Constant(init), InductionStep::Constant(step)) = (iv.init, iv.step) {
            if init >= 0 && step > 0 {
                return EliminationDecision::Eliminate;
            }
        }

        // Case 3: Non-constant init but known increasing from 0
        // This would need more complex analysis

        // Case 4: Aggressive mode - hoist the check if possible
        if self.aggressive {
            // Could hoist init >= 0 check to preheader
            return EliminationDecision::Hoist(HoistInfo {
                guard: check.guard,
                iv: check.induction_var,
            });
        }

        EliminationDecision::Keep
    }

    /// Analyze an upper bound check (iv < bound).
    fn analyze_upper_bound(
        &self,
        check: &RangeCheck,
        iv: &InductionVariable,
    ) -> EliminationDecision {
        // Case 1: Constant init, constant bound, and decreasing direction
        // If we start below bound and only decrease, we stay below
        if let (InductionInit::Constant(init), BoundValue::Constant(bound)) = (iv.init, check.bound)
        {
            if init < bound && matches!(iv.direction, InductionDirection::Decreasing) {
                return EliminationDecision::Eliminate;
            }
        }

        // Case 2: Widening opportunity
        // If we can compute the max value, we can widen
        if let Some(widen) = self.try_widen_upper_bound(check, iv) {
            return EliminationDecision::Widen(widen);
        }

        // Case 3: Aggressive hoisting
        if self.aggressive {
            return EliminationDecision::Hoist(HoistInfo {
                guard: check.guard,
                iv: check.induction_var,
            });
        }

        EliminationDecision::Keep
    }

    /// Analyze an upper bound inclusive check (iv <= bound).
    fn analyze_upper_bound_inclusive(
        &self,
        check: &RangeCheck,
        iv: &InductionVariable,
    ) -> EliminationDecision {
        // Similar to upper bound but with <= instead of <
        if let (InductionInit::Constant(init), BoundValue::Constant(bound)) = (iv.init, check.bound)
        {
            if init <= bound && matches!(iv.direction, InductionDirection::Decreasing) {
                return EliminationDecision::Eliminate;
            }
        }

        // For inclusive checks, widening is slightly different
        if let Some(widen) = self.try_widen_upper_bound_inclusive(check, iv) {
            return EliminationDecision::Widen(widen);
        }

        if self.aggressive {
            return EliminationDecision::Hoist(HoistInfo {
                guard: check.guard,
                iv: check.induction_var,
            });
        }

        EliminationDecision::Keep
    }

    /// Try to create a widened upper bound check.
    fn try_widen_upper_bound(
        &self,
        check: &RangeCheck,
        iv: &InductionVariable,
    ) -> Option<WidenInfo> {
        // For widening, we need to compute the maximum value the IV can reach
        // This requires knowing the loop trip count or exit condition

        // Simple case: constant init, constant step, increasing
        // Max value = init + (trip_count - 1) * step
        // But we don't have trip count here...

        // For now, only widen if we have aggressive mode and simple constants
        if !self.aggressive {
            return None;
        }

        match (iv.init, iv.step, iv.direction) {
            (
                InductionInit::Constant(init),
                InductionStep::Constant(step),
                InductionDirection::Increasing,
            ) => {
                // We could create a widened check, but need trip count
                // For now, create a placeholder
                Some(WidenInfo {
                    guard: check.guard,
                    iv: check.induction_var,
                    min_check: WidenBound::Constant(init),
                    max_check: WidenBound::Computed {
                        base: iv.phi,
                        offset: step, // This is incomplete, needs trip count
                    },
                })
            }
            _ => None,
        }
    }

    /// Try to create a widened upper bound inclusive check.
    fn try_widen_upper_bound_inclusive(
        &self,
        check: &RangeCheck,
        iv: &InductionVariable,
    ) -> Option<WidenInfo> {
        // Similar to above but for <= checks
        if !self.aggressive {
            return None;
        }

        match (iv.init, iv.step, iv.direction) {
            (
                InductionInit::Constant(init),
                InductionStep::Constant(step),
                InductionDirection::Increasing,
            ) => Some(WidenInfo {
                guard: check.guard,
                iv: check.induction_var,
                min_check: WidenBound::Constant(init),
                max_check: WidenBound::Computed {
                    base: iv.phi,
                    offset: step,
                },
            }),
            _ => None,
        }
    }
}

impl Default for EliminationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Elimination Result
// =============================================================================

/// Result of elimination analysis for all checks in a loop.
#[derive(Debug, Default)]
pub struct EliminationResult {
    /// Decisions for each check.
    decisions: Vec<(RangeCheck, EliminationDecision)>,
}

impl EliminationResult {
    /// Create new empty result.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            decisions: Vec::with_capacity(cap),
        }
    }

    /// Add a decision.
    #[inline]
    pub fn add(&mut self, check: RangeCheck, decision: EliminationDecision) {
        self.decisions.push((check, decision));
    }

    /// Get all decisions.
    #[inline]
    pub fn decisions(&self) -> &[(RangeCheck, EliminationDecision)] {
        &self.decisions
    }

    /// Iterate over eliminable checks.
    pub fn eliminable(&self) -> impl Iterator<Item = &RangeCheck> {
        self.decisions
            .iter()
            .filter_map(|(c, d)| matches!(d, EliminationDecision::Eliminate).then_some(c))
    }

    /// Iterate over hoistable checks.
    pub fn hoistable(&self) -> impl Iterator<Item = (&RangeCheck, &HoistInfo)> {
        self.decisions.iter().filter_map(|(c, d)| match d {
            EliminationDecision::Hoist(h) => Some((c, h)),
            _ => None,
        })
    }

    /// Iterate over widenable checks.
    pub fn widenable(&self) -> impl Iterator<Item = (&RangeCheck, &WidenInfo)> {
        self.decisions.iter().filter_map(|(c, d)| match d {
            EliminationDecision::Widen(w) => Some((c, w)),
            _ => None,
        })
    }

    /// Count eliminable checks.
    #[inline]
    pub fn count_eliminable(&self) -> usize {
        self.decisions
            .iter()
            .filter(|(_, d)| matches!(d, EliminationDecision::Eliminate))
            .count()
    }

    /// Count hoistable checks.
    #[inline]
    pub fn count_hoistable(&self) -> usize {
        self.decisions
            .iter()
            .filter(|(_, d)| matches!(d, EliminationDecision::Hoist(_)))
            .count()
    }

    /// Count widenable checks.
    #[inline]
    pub fn count_widenable(&self) -> usize {
        self.decisions
            .iter()
            .filter(|(_, d)| matches!(d, EliminationDecision::Widen(_)))
            .count()
    }

    /// Check if any checks were processed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Total number of decisions.
    #[inline]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
