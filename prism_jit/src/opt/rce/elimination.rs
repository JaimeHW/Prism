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
mod tests {
    use super::*;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    fn make_iv(init: i64, step: i64, direction: InductionDirection) -> InductionVariable {
        InductionVariable {
            phi: NodeId::new(0),
            init: InductionInit::Constant(init),
            step: InductionStep::Constant(step),
            direction,
            update_node: None,
        }
    }

    fn make_lower_check(iv_node: NodeId) -> RangeCheck {
        RangeCheck::new(
            NodeId::new(10),
            iv_node,
            NodeId::new(11),
            BoundValue::Constant(0),
            RangeCheckKind::LowerBound,
            0,
        )
    }

    fn make_upper_check(iv_node: NodeId, bound: i64) -> RangeCheck {
        RangeCheck::new(
            NodeId::new(10),
            iv_node,
            NodeId::new(11),
            BoundValue::Constant(bound),
            RangeCheckKind::UpperBound,
            0,
        )
    }

    // =========================================================================
    // EliminationAnalyzer Tests
    // =========================================================================

    #[test]
    fn test_analyzer_new() {
        let analyzer = EliminationAnalyzer::new();
        assert!(!analyzer.aggressive);
        assert_eq!(analyzer.stats().analyzed, 0);
    }

    #[test]
    fn test_analyzer_aggressive() {
        let analyzer = EliminationAnalyzer::aggressive();
        assert!(analyzer.aggressive);
    }

    #[test]
    fn test_lower_bound_eliminable_positive_init_increasing() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(0, 1, InductionDirection::Increasing);
        let check = make_lower_check(NodeId::new(0));

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Eliminate);
        assert_eq!(analyzer.stats().eliminable, 1);
    }

    #[test]
    fn test_lower_bound_eliminable_positive_init_positive_step() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(5, 2, InductionDirection::Increasing);
        let check = make_lower_check(NodeId::new(0));

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Eliminate);
    }

    #[test]
    fn test_lower_bound_kept_negative_init() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(-5, 1, InductionDirection::Increasing);
        let check = make_lower_check(NodeId::new(0));

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Keep);
        assert_eq!(analyzer.stats().kept, 1);
    }

    #[test]
    fn test_lower_bound_kept_decreasing() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(10, -1, InductionDirection::Decreasing);
        let check = make_lower_check(NodeId::new(0));

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Keep);
    }

    #[test]
    fn test_upper_bound_eliminable_decreasing_below() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(50, -1, InductionDirection::Decreasing);
        let check = make_upper_check(NodeId::new(0), 100);

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Eliminate);
    }

    #[test]
    fn test_upper_bound_kept_increasing_no_bound_info() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(0, 1, InductionDirection::Increasing);
        let check = make_upper_check(NodeId::new(0), 100);

        let decision = analyzer.analyze(&check, &iv);
        // Without trip count info, we can't eliminate
        assert_eq!(decision, EliminationDecision::Keep);
    }

    #[test]
    fn test_aggressive_hoists_lower_bound() {
        let mut analyzer = EliminationAnalyzer::aggressive();
        let iv = make_iv(-5, 1, InductionDirection::Increasing);
        let check = make_lower_check(NodeId::new(0));

        let decision = analyzer.analyze(&check, &iv);
        assert!(matches!(decision, EliminationDecision::Hoist(_)));
        assert_eq!(analyzer.stats().hoistable, 1);
    }

    #[test]
    fn test_aggressive_widens_upper_bound() {
        let mut analyzer = EliminationAnalyzer::aggressive();
        let iv = make_iv(0, 1, InductionDirection::Increasing);
        let check = make_upper_check(NodeId::new(0), 100);

        let decision = analyzer.analyze(&check, &iv);
        assert!(matches!(decision, EliminationDecision::Widen(_)));
        assert_eq!(analyzer.stats().widenable, 1);
    }

    #[test]
    fn test_stats_reset() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(0, 1, InductionDirection::Increasing);
        let check = make_lower_check(NodeId::new(0));

        analyzer.analyze(&check, &iv);
        assert_eq!(analyzer.stats().analyzed, 1);

        analyzer.reset_stats();
        assert_eq!(analyzer.stats().analyzed, 0);
    }

    // =========================================================================
    // EliminationResult Tests
    // =========================================================================

    #[test]
    fn test_result_new() {
        let result = EliminationResult::new();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_result_add() {
        let mut result = EliminationResult::new();
        let check = make_lower_check(NodeId::new(0));
        result.add(check, EliminationDecision::Eliminate);

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result.count_eliminable(), 1);
    }

    #[test]
    fn test_result_eliminable_iter() {
        let mut result = EliminationResult::new();
        result.add(
            make_lower_check(NodeId::new(0)),
            EliminationDecision::Eliminate,
        );
        result.add(
            make_upper_check(NodeId::new(0), 100),
            EliminationDecision::Keep,
        );
        result.add(
            make_lower_check(NodeId::new(1)),
            EliminationDecision::Eliminate,
        );

        let eliminable: Vec<_> = result.eliminable().collect();
        assert_eq!(eliminable.len(), 2);
    }

    #[test]
    fn test_result_hoistable_iter() {
        let mut result = EliminationResult::new();
        result.add(
            make_lower_check(NodeId::new(0)),
            EliminationDecision::Hoist(HoistInfo {
                guard: NodeId::new(10),
                iv: NodeId::new(0),
            }),
        );
        result.add(
            make_upper_check(NodeId::new(0), 100),
            EliminationDecision::Keep,
        );

        let hoistable: Vec<_> = result.hoistable().collect();
        assert_eq!(hoistable.len(), 1);
    }

    #[test]
    fn test_result_widenable_iter() {
        let mut result = EliminationResult::new();
        result.add(
            make_upper_check(NodeId::new(0), 100),
            EliminationDecision::Widen(WidenInfo {
                guard: NodeId::new(10),
                iv: NodeId::new(0),
                min_check: WidenBound::Constant(0),
                max_check: WidenBound::Constant(99),
            }),
        );

        let widenable: Vec<_> = result.widenable().collect();
        assert_eq!(widenable.len(), 1);
    }

    // =========================================================================
    // WidenBound Tests
    // =========================================================================

    #[test]
    fn test_widen_bound_constant() {
        let bound = WidenBound::Constant(100);
        assert!(matches!(bound, WidenBound::Constant(100)));
    }

    #[test]
    fn test_widen_bound_node() {
        let bound = WidenBound::Node(NodeId::new(5));
        assert!(matches!(bound, WidenBound::Node(_)));
    }

    #[test]
    fn test_widen_bound_computed() {
        let bound = WidenBound::Computed {
            base: NodeId::new(0),
            offset: 10,
        };
        assert!(matches!(bound, WidenBound::Computed { .. }));
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_zero_step_iv() {
        let mut analyzer = EliminationAnalyzer::new();
        let iv = make_iv(0, 0, InductionDirection::Unknown);
        let check = make_lower_check(NodeId::new(0));

        // Zero step with init >= 0 should still be eliminable for lower bound
        // Actually no, because direction is Unknown
        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Keep);
    }

    #[test]
    fn test_negative_step_positive_init_lower_bound() {
        let mut analyzer = EliminationAnalyzer::new();
        // Init = 100, step = -1 (decreasing)
        // Lower bound check NOT safe - will eventually go negative
        let iv = make_iv(100, -1, InductionDirection::Decreasing);
        let check = make_lower_check(NodeId::new(0));

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Keep);
    }

    #[test]
    fn test_upper_bound_at_exact_limit() {
        let mut analyzer = EliminationAnalyzer::new();
        // Init = 99, decreasing, bound = 100
        // 99 < 100 is true, and we're decreasing, so safe
        let iv = make_iv(99, -1, InductionDirection::Decreasing);
        let check = make_upper_check(NodeId::new(0), 100);

        let decision = analyzer.analyze(&check, &iv);
        assert_eq!(decision, EliminationDecision::Eliminate);
    }

    #[test]
    fn test_upper_bound_starting_at_limit() {
        let mut analyzer = EliminationAnalyzer::new();
        // Init = 100, decreasing, bound = 100
        // 100 < 100 is FALSE, so not safe
        let iv = make_iv(100, -1, InductionDirection::Decreasing);
        let check = make_upper_check(NodeId::new(0), 100);

        let decision = analyzer.analyze(&check, &iv);
        // init < bound check fails (100 < 100 is false)
        assert_eq!(decision, EliminationDecision::Keep);
    }
}
