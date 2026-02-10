//! Inlining Cost Model
//!
//! This module implements a sophisticated cost model for inlining decisions.
//! The cost model considers multiple factors to predict the benefit vs. cost
//! of inlining a particular call site.
//!
//! # Cost Factors
//!
//! - **Code Size**: Larger callees increase code cache pressure
//! - **Call Overhead**: Eliminated by inlining
//! - **Optimization Opportunities**: Cross-procedural opts enabled
//! - **Argument Passing**: Complex args may benefit from inlining
//! - **Loop Context**: Inlining in loops multiplies benefits
//!
//! # Benefit Factors
//!
//! - **Constant Propagation**: Known arguments can be folded
//! - **Dead Code Elimination**: Unused branches can be removed
//! - **Escape Analysis**: Allocations may become non-escaping
//! - **Type Specialization**: Known types enable fast paths

use super::{CallSite, CalleeInfo};

// =============================================================================
// Inline Cost
// =============================================================================

/// Represents the computed cost/benefit of inlining a call site.
#[derive(Debug, Clone, Copy)]
pub struct InlineCost {
    /// Raw cost score (negative = beneficial to inline).
    pub score: i32,
    /// Size cost component.
    pub size_cost: i32,
    /// Benefit from optimization opportunities.
    pub optimization_benefit: i32,
    /// Benefit from call elimination.
    pub call_elimination_benefit: i32,
    /// Benefit from loop context.
    pub loop_benefit: i32,
    /// Whether this is a forced inline.
    pub forced: bool,
    /// Whether this is blocked from inlining.
    pub blocked: bool,
}

impl InlineCost {
    /// Cost threshold below which we should inline.
    /// Negative scores indicate beneficial inlining.
    pub const INLINE_THRESHOLD: i32 = 50;

    /// Minimum cost for a forced inline (always inline).
    pub const FORCED_INLINE: Self = Self {
        score: i32::MIN,
        size_cost: 0,
        optimization_benefit: 0,
        call_elimination_benefit: 0,
        loop_benefit: 0,
        forced: true,
        blocked: false,
    };

    /// Maximum cost for blocked inlining (never inline).
    pub const BLOCKED: Self = Self {
        score: i32::MAX,
        size_cost: 0,
        optimization_benefit: 0,
        call_elimination_benefit: 0,
        loop_benefit: 0,
        forced: false,
        blocked: true,
    };

    /// Create a new inline cost with the given components.
    pub fn new(
        size_cost: i32,
        optimization_benefit: i32,
        call_elimination_benefit: i32,
        loop_benefit: i32,
    ) -> Self {
        let score = size_cost - optimization_benefit - call_elimination_benefit - loop_benefit;
        Self {
            score,
            size_cost,
            optimization_benefit,
            call_elimination_benefit,
            loop_benefit,
            forced: false,
            blocked: false,
        }
    }

    /// Check if inlining is beneficial based on the cost.
    #[inline]
    pub fn should_inline(&self) -> bool {
        if self.forced {
            return true;
        }
        if self.blocked {
            return false;
        }
        self.score < Self::INLINE_THRESHOLD
    }

    /// Get the net benefit (negative cost means benefit).
    #[inline]
    pub fn net_benefit(&self) -> i32 {
        -self.score
    }
}

impl Default for InlineCost {
    fn default() -> Self {
        Self {
            score: 0,
            size_cost: 0,
            optimization_benefit: 0,
            call_elimination_benefit: 0,
            loop_benefit: 0,
            forced: false,
            blocked: false,
        }
    }
}

// =============================================================================
// Cost Model Configuration
// =============================================================================

/// Configuration for the cost model.
#[derive(Debug, Clone)]
pub struct CostModelConfig {
    /// Cost per node in the callee.
    pub cost_per_node: i32,
    /// Fixed cost of a function call (benefit of eliminating).
    pub call_overhead: i32,
    /// Benefit per constant argument (enables constant prop).
    pub const_arg_benefit: i32,
    /// Benefit multiplier for loop nesting.
    pub loop_multiplier: i32,
    /// Benefit for eliminating allocation escape.
    pub escape_benefit: i32,
    /// Cost for recursive calls.
    pub recursion_penalty: i32,
    /// Benefit for small functions (< 10 nodes).
    pub small_function_bonus: i32,
    /// Benefit for leaf functions (no calls).
    pub leaf_function_bonus: i32,
    /// Cost for functions with exception handling.
    pub exception_cost: i32,
}

impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            cost_per_node: 3,
            call_overhead: 25,
            const_arg_benefit: 15,
            loop_multiplier: 20,
            escape_benefit: 30,
            recursion_penalty: 100,
            small_function_bonus: 20,
            leaf_function_bonus: 10,
            exception_cost: 20,
        }
    }
}

// =============================================================================
// Cost Model
// =============================================================================

/// The inlining cost model computes whether inlining is beneficial.
#[derive(Debug, Clone)]
pub struct InlineCostModel {
    /// Configuration.
    config: CostModelConfig,
}

impl InlineCostModel {
    /// Create a new cost model with default configuration.
    pub fn new() -> Self {
        Self {
            config: CostModelConfig::default(),
        }
    }

    /// Create a cost model with custom configuration.
    pub fn with_config(config: CostModelConfig) -> Self {
        Self { config }
    }

    /// Compute the inlining cost for a call site.
    pub fn compute_cost(&self, callee: &CalleeInfo, site: &CallSite) -> InlineCost {
        // Handle forced cases
        if callee.always_inline {
            return InlineCost::FORCED_INLINE;
        }
        if callee.never_inline {
            return InlineCost::BLOCKED;
        }

        // Compute size cost
        let size_cost = self.compute_size_cost(callee);

        // Compute optimization benefit
        let optimization_benefit = self.compute_optimization_benefit(callee, site);

        // Call elimination benefit
        let call_elimination_benefit = self.config.call_overhead;

        // Loop benefit
        let loop_benefit = self.compute_loop_benefit(site);

        InlineCost::new(
            size_cost,
            optimization_benefit,
            call_elimination_benefit,
            loop_benefit,
        )
    }

    /// Compute the code size cost of inlining.
    fn compute_size_cost(&self, callee: &CalleeInfo) -> i32 {
        let base_cost = (callee.size as i32) * self.config.cost_per_node;

        // Small function bonus
        let size_adjustment = if callee.size < 10 {
            -self.config.small_function_bonus
        } else if callee.size < 30 {
            -self.config.small_function_bonus / 2
        } else {
            0
        };

        // Penalty for functions with loops (they expand significantly)
        let loop_penalty = if callee.has_loops { 30 } else { 0 };

        base_cost + size_adjustment + loop_penalty
    }

    /// Compute optimization opportunities from inlining.
    fn compute_optimization_benefit(&self, callee: &CalleeInfo, site: &CallSite) -> i32 {
        let mut benefit = 0;

        // Benefit from constant arguments
        // In a real implementation, we'd analyze which args are constants
        // For now, estimate based on call site info
        let estimated_const_args = self.estimate_constant_args(site);
        benefit += estimated_const_args * self.config.const_arg_benefit;

        // Intrinsics have high benefit
        if callee.is_intrinsic {
            benefit += 50;
        }

        // Hot call sites benefit more from inlining
        if site.is_hot {
            benefit += 25;
        }

        // High call count indicates the function is worth optimizing
        if callee.call_count > 1000 {
            benefit += 20;
        } else if callee.call_count > 100 {
            benefit += 10;
        }

        benefit
    }

    /// Estimate how many arguments are likely constants.
    fn estimate_constant_args(&self, site: &CallSite) -> i32 {
        // In a complete implementation, we'd check if argument nodes
        // are constants. For now, assume 1 constant arg on average.
        (site.arguments.len() as i32).min(2)
    }

    /// Compute benefit from loop context.
    fn compute_loop_benefit(&self, site: &CallSite) -> i32 {
        if site.loop_depth == 0 {
            return 0;
        }

        // Benefit scales with loop depth (but diminishes)
        let base = self.config.loop_multiplier;
        let depth_factor = (site.loop_depth as i32).min(4);

        // First level of loop nesting: full multiplier
        // Each additional level: 50% of previous
        let mut benefit = 0;
        let mut current = base;
        for _ in 0..depth_factor {
            benefit += current;
            current /= 2;
        }

        benefit
    }
}

impl Default for InlineCostModel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::NodeId;
    use crate::ir::operators::CallKind;

    fn make_site(loop_depth: u32, is_hot: bool, arg_count: usize) -> CallSite {
        CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: None,
            loop_depth,
            is_hot,
            priority: 0,
            arguments: (0..arg_count).map(|i| NodeId::new(i as u32)).collect(),
            control_input: None,
        }
    }

    fn make_callee(size: usize, has_loops: bool) -> CalleeInfo {
        CalleeInfo {
            func_id: 1,
            size,
            param_count: 2,
            is_recursive: false,
            has_loops,
            always_inline: false,
            never_inline: false,
            call_count: 0,
            is_intrinsic: false,
        }
    }

    #[test]
    fn test_inline_cost_should_inline() {
        // Beneficial (negative score)
        let cost = InlineCost::new(10, 20, 25, 10);
        assert!(cost.should_inline());
        assert!(cost.net_benefit() > 0);

        // Not beneficial (high positive score)
        let cost = InlineCost::new(200, 10, 25, 0);
        assert!(!cost.should_inline());
    }

    #[test]
    fn test_inline_cost_forced() {
        assert!(InlineCost::FORCED_INLINE.should_inline());
        assert!(InlineCost::FORCED_INLINE.forced);
    }

    #[test]
    fn test_inline_cost_blocked() {
        assert!(!InlineCost::BLOCKED.should_inline());
        assert!(InlineCost::BLOCKED.blocked);
    }

    #[test]
    fn test_cost_model_small_function() {
        let model = InlineCostModel::new();
        let small_callee = make_callee(5, false);
        let site = make_site(0, false, 2);

        let cost = model.compute_cost(&small_callee, &site);

        // Small functions should have low or negative cost
        assert!(cost.size_cost < 50);
    }

    #[test]
    fn test_cost_model_large_function() {
        let model = InlineCostModel::new();
        let large_callee = make_callee(200, false);
        let site = make_site(0, false, 2);

        let cost = model.compute_cost(&large_callee, &site);

        // Large functions should have high cost
        assert!(cost.size_cost > 500);
        assert!(!cost.should_inline());
    }

    #[test]
    fn test_cost_model_loop_benefit() {
        let model = InlineCostModel::new();
        let callee = make_callee(20, false);

        let site_no_loop = make_site(0, false, 2);
        let site_in_loop = make_site(2, false, 2);

        let cost_no_loop = model.compute_cost(&callee, &site_no_loop);
        let cost_in_loop = model.compute_cost(&callee, &site_in_loop);

        // Inlining in loops should be more beneficial
        assert!(cost_in_loop.loop_benefit > cost_no_loop.loop_benefit);
        assert!(cost_in_loop.score < cost_no_loop.score);
    }

    #[test]
    fn test_cost_model_hot_site() {
        let model = InlineCostModel::new();
        let callee = make_callee(30, false);

        let cold_site = make_site(0, false, 2);
        let hot_site = make_site(0, true, 2);

        let cold_cost = model.compute_cost(&callee, &cold_site);
        let hot_cost = model.compute_cost(&callee, &hot_site);

        // Hot sites should have more optimization benefit
        assert!(hot_cost.optimization_benefit > cold_cost.optimization_benefit);
    }

    #[test]
    fn test_cost_model_always_inline() {
        let model = InlineCostModel::new();
        let mut callee = make_callee(500, true);
        callee.always_inline = true;

        let site = make_site(0, false, 2);
        let cost = model.compute_cost(&callee, &site);

        assert!(cost.forced);
        assert!(cost.should_inline());
    }

    #[test]
    fn test_cost_model_never_inline() {
        let model = InlineCostModel::new();
        let mut callee = make_callee(5, false);
        callee.never_inline = true;

        let site = make_site(0, false, 2);
        let cost = model.compute_cost(&callee, &site);

        assert!(cost.blocked);
        assert!(!cost.should_inline());
    }

    #[test]
    fn test_cost_model_intrinsic() {
        let model = InlineCostModel::new();
        let mut callee = make_callee(10, false);
        callee.is_intrinsic = true;

        let site = make_site(0, false, 2);
        let cost = model.compute_cost(&callee, &site);

        // Intrinsics should have high optimization benefit
        assert!(cost.optimization_benefit >= 50);
    }

    #[test]
    fn test_cost_model_loop_penalty() {
        let model = InlineCostModel::new();
        let callee_no_loop = make_callee(20, false);
        let callee_with_loop = make_callee(20, true);

        let site = make_site(0, false, 2);

        let cost_no_loop = model.compute_cost(&callee_no_loop, &site);
        let cost_with_loop = model.compute_cost(&callee_with_loop, &site);

        // Functions with loops should have higher size cost
        assert!(cost_with_loop.size_cost > cost_no_loop.size_cost);
    }

    #[test]
    fn test_cost_model_high_call_count() {
        let model = InlineCostModel::new();
        let mut cold_callee = make_callee(20, false);
        cold_callee.call_count = 10;

        let mut hot_callee = make_callee(20, false);
        hot_callee.call_count = 5000;

        let site = make_site(0, false, 2);

        let cold_cost = model.compute_cost(&cold_callee, &site);
        let hot_cost = model.compute_cost(&hot_callee, &site);

        // High call count should increase benefit
        assert!(hot_cost.optimization_benefit > cold_cost.optimization_benefit);
    }
}
