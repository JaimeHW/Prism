//! Cost Model and Heuristics for Loop Unrolling.
//!
//! This module implements the decision-making logic for loop unrolling,
//! using a comprehensive cost model to determine:
//!
//! - Whether to unroll at all
//! - What unrolling strategy to use
//! - What unroll factor is optimal
//!
//! # Cost Model
//!
//! The cost model considers:
//! - **Loop overhead**: Branch, induction variable update
//! - **Code size**: Impact on instruction cache
//! - **Register pressure**: Live values vs available registers
//! - **ILP**: Instruction-level parallelism gained
//!
//! # Heuristics
//!
//! Heuristics guide decisions when exact costs are unknown:
//! - Inner loops are prioritized over outer loops
//! - Shorter bodies favor higher unroll factors
//! - Loops with known trip counts favor full unrolling

use super::analysis::UnrollabilityAnalysis;
use super::{NoUnrollReason, RemainderStrategy, UnrollConfig, UnrollStrategy};

// =============================================================================
// Cost Model
// =============================================================================

/// Cost model for loop unrolling decisions.
#[derive(Debug, Clone)]
pub struct UnrollCostModel {
    /// Cost of a loop iteration (branch + update).
    pub loop_iteration_cost: f64,

    /// Cost per instruction (base code size).
    pub instruction_cost: f64,

    /// Penalty for instruction cache misses.
    pub icache_miss_penalty: f64,

    /// Penalty for register spills.
    pub register_spill_penalty: f64,

    /// Benefit from eliminating a branch.
    pub branch_elimination_benefit: f64,

    /// Benefit from increased ILP.
    pub ilp_benefit: f64,

    /// Benefit from enabling further optimizations.
    pub optimization_opportunity_benefit: f64,

    /// Target instruction cache size (in instructions).
    pub target_icache_size: usize,

    /// Target register count.
    pub target_register_count: usize,
}

impl Default for UnrollCostModel {
    fn default() -> Self {
        Self {
            loop_iteration_cost: 3.0,        // Branch + induction update
            instruction_cost: 0.1,           // Base cost per instruction
            icache_miss_penalty: 50.0,       // Cache miss is expensive
            register_spill_penalty: 5.0,     // Spill/reload cost
            branch_elimination_benefit: 2.0, // Fewer branches = better
            ilp_benefit: 0.5,                // Independent ops can overlap
            optimization_opportunity_benefit: 1.0,
            target_icache_size: 32768, // 32KB L1 icache
            target_register_count: 16, // x64 general purpose regs
        }
    }
}

impl UnrollCostModel {
    /// Create a model optimized for server workloads (larger caches).
    pub fn server() -> Self {
        Self {
            loop_iteration_cost: 3.0,
            instruction_cost: 0.05, // Lower cost, bigger caches
            icache_miss_penalty: 30.0,
            register_spill_penalty: 4.0,
            branch_elimination_benefit: 2.5,
            ilp_benefit: 0.7, // More execution units
            optimization_opportunity_benefit: 1.0,
            target_icache_size: 65536, // 64KB L1 icache
            target_register_count: 16,
        }
    }

    /// Create a model optimized for mobile/embedded (smaller caches).
    pub fn mobile() -> Self {
        Self {
            loop_iteration_cost: 2.5,
            instruction_cost: 0.2, // Higher cost, smaller caches
            icache_miss_penalty: 100.0,
            register_spill_penalty: 8.0,
            branch_elimination_benefit: 1.5,
            ilp_benefit: 0.3, // Fewer execution units
            optimization_opportunity_benefit: 0.8,
            target_icache_size: 16384, // 16KB L1 icache
            target_register_count: 14,
        }
    }

    /// Compute the cost of not unrolling.
    pub fn baseline_cost(&self, analysis: &UnrollabilityAnalysis) -> f64 {
        let trip = analysis.trip_count.upper_bound().unwrap_or(100) as f64;
        let body_cost = analysis.body_size as f64 * self.instruction_cost;
        let iteration_cost = self.loop_iteration_cost;

        trip * (body_cost + iteration_cost)
    }

    /// Compute the cost of full unrolling.
    pub fn full_unroll_cost(&self, analysis: &UnrollabilityAnalysis, trip: u32) -> f64 {
        let trip = trip as f64;
        let body_cost = analysis.body_size as f64 * self.instruction_cost;

        // Unrolled code size
        let code_size = analysis.body_size * trip as usize;

        // Base execution cost (no loop overhead)
        let mut cost = trip * body_cost;

        // Register pressure penalty
        let pressure = analysis.register_pressure * trip as usize / 2;
        if pressure > self.target_register_count {
            let spills = pressure - self.target_register_count;
            cost += spills as f64 * self.register_spill_penalty;
        }

        // ICache pressure
        if code_size > self.target_icache_size / 64 {
            let excess = code_size - self.target_icache_size / 64;
            cost += excess as f64 * self.icache_miss_penalty / 100.0;
        }

        // Benefit from branch elimination
        cost -= trip * self.branch_elimination_benefit;

        // Benefit from optimization opportunities
        cost -= (trip - 1.0) * self.optimization_opportunity_benefit * body_cost / 10.0;

        cost.max(0.0)
    }

    /// Compute the cost of partial unrolling.
    pub fn partial_unroll_cost(&self, analysis: &UnrollabilityAnalysis, factor: u32) -> f64 {
        let trip = analysis.trip_count.upper_bound().unwrap_or(100) as f64;
        let body_cost = analysis.body_size as f64 * self.instruction_cost;
        let factor = factor as f64;

        // Number of main loop iterations
        let main_iters = trip / factor;
        // Remainder iterations
        let remainder = trip % factor;

        // Main loop cost
        let main_loop_cost = main_iters * (body_cost * factor + self.loop_iteration_cost);

        // Remainder cost (still looped)
        let remainder_cost = remainder * (body_cost + self.loop_iteration_cost);

        // Register pressure penalty
        let pressure = analysis.register_pressure * factor as usize;
        let spill_cost = if pressure > self.target_register_count {
            let spills = pressure - self.target_register_count;
            main_iters * spills as f64 * self.register_spill_penalty
        } else {
            0.0
        };

        // ILP benefit
        let ilp_reduction = main_iters * (factor - 1.0) * self.ilp_benefit;

        main_loop_cost + remainder_cost + spill_cost - ilp_reduction
    }

    /// Compute benefit/cost ratio for unrolling.
    pub fn unroll_benefit(&self, analysis: &UnrollabilityAnalysis, factor: u32) -> f64 {
        let baseline = self.baseline_cost(analysis);
        let unrolled = if analysis.can_fully_unroll(factor, usize::MAX) {
            self.full_unroll_cost(analysis, factor)
        } else {
            self.partial_unroll_cost(analysis, factor)
        };

        // If unrolled cost is very low, that's highly beneficial
        if unrolled < 0.01 {
            // Return the absolute savings as the benefit
            baseline.max(1.0)
        } else {
            (baseline - unrolled) / unrolled
        }
    }
}

// =============================================================================
// Unroll Decision
// =============================================================================

/// Decision from the unroll heuristics.
#[derive(Debug, Clone)]
pub struct UnrollDecision {
    /// The chosen strategy.
    pub strategy: UnrollStrategy,
    /// Estimated speedup (ratio).
    pub estimated_speedup: f64,
    /// Estimated code growth (ratio).
    pub code_growth: f64,
    /// Confidence in the decision (0.0 - 1.0).
    pub confidence: f64,
}

impl UnrollDecision {
    /// Create a decision not to unroll.
    pub fn no_unroll(reason: NoUnrollReason) -> Self {
        Self {
            strategy: UnrollStrategy::NoUnroll { reason },
            estimated_speedup: 1.0,
            code_growth: 1.0,
            confidence: 1.0,
        }
    }
}

// =============================================================================
// Heuristics
// =============================================================================

/// Heuristics engine for unroll decisions.
pub struct UnrollHeuristics<'a> {
    config: &'a UnrollConfig,
    cost_model: &'a UnrollCostModel,
}

impl<'a> UnrollHeuristics<'a> {
    /// Create a new heuristics engine.
    pub fn new(config: &'a UnrollConfig, cost_model: &'a UnrollCostModel) -> Self {
        Self { config, cost_model }
    }

    /// Determine the best unrolling strategy for a loop.
    pub fn determine_strategy(&self, analysis: &UnrollabilityAnalysis) -> UnrollStrategy {
        // First, check if unrolling is even possible
        if let Some(reason) = self.check_hard_constraints(analysis) {
            return UnrollStrategy::NoUnroll { reason };
        }

        // Try full unroll first (most beneficial)
        if let Some(trip) = analysis.trip_count.as_constant() {
            if self.should_fully_unroll(analysis, trip as u32) {
                return UnrollStrategy::FullUnroll {
                    trip_count: trip as u32,
                };
            }
        }

        // Try partial unroll
        if let Some(factor) = self.best_partial_factor(analysis) {
            if factor > 1 {
                let remainder = self.choose_remainder_strategy(analysis, factor);
                return UnrollStrategy::PartialUnroll { factor, remainder };
            }
        }

        // Try runtime unroll if enabled
        if self.config.enable_runtime_unroll {
            if let Some(factor) = self.best_runtime_factor(analysis) {
                if factor > 1 {
                    let remainder = self.choose_remainder_strategy(analysis, factor);
                    return UnrollStrategy::RuntimeUnroll {
                        min_trip: self.config.min_runtime_trip,
                        factor,
                        remainder,
                    };
                }
            }
        }

        // Not profitable
        UnrollStrategy::NoUnroll {
            reason: NoUnrollReason::NotProfitable,
        }
    }

    /// Check hard constraints that prevent unrolling.
    fn check_hard_constraints(&self, analysis: &UnrollabilityAnalysis) -> Option<NoUnrollReason> {
        // Must be in canonical form
        if !analysis.is_canonical {
            return Some(NoUnrollReason::NotCanonical);
        }

        // Body too large
        if analysis.body_size > self.config.max_full_unroll_size {
            // Only a hard constraint for full unroll
            // Partial unroll may still be OK
        }

        // Complex control flow
        if analysis.has_early_exits && !analysis.has_single_exit {
            return Some(NoUnrollReason::ComplexControlFlow);
        }

        // Contains calls (usually unprofitable to unroll)
        if analysis.contains_calls {
            // Only block if aggressive is off
            if !self.config.aggressive_inner_loops {
                return Some(NoUnrollReason::ContainsCalls);
            }
        }

        // Nesting too deep
        if analysis.nesting_depth > 3 {
            return Some(NoUnrollReason::NestingTooDeep);
        }

        None
    }

    /// Check if full unrolling is beneficial.
    fn should_fully_unroll(&self, analysis: &UnrollabilityAnalysis, trip: u32) -> bool {
        // Check trip count limit
        if trip > self.config.max_full_unroll_trip {
            return false;
        }

        // Check size limit
        let unrolled_size = analysis.body_size * trip as usize;
        if unrolled_size > self.config.max_full_unroll_size {
            return false;
        }

        // Check code growth
        let growth = unrolled_size as f64 / analysis.body_size as f64;
        if growth > self.config.max_code_growth {
            return false;
        }

        // Cost model check
        let benefit = self.cost_model.unroll_benefit(analysis, trip);
        benefit > 0.1 // At least 10% improvement
    }

    /// Find the best partial unroll factor.
    fn best_partial_factor(&self, analysis: &UnrollabilityAnalysis) -> Option<u32> {
        if analysis.induction_vars.is_empty() {
            return None;
        }

        let mut best_factor = 1;
        let mut best_benefit = 0.0;

        // Try powers of 2 first (best for remainder handling)
        for factor in [2, 4, 8, 16, 32] {
            if factor > self.config.max_unroll_factor {
                break;
            }

            // Check code growth
            let growth = (analysis.body_size * factor as usize) as f64 / analysis.body_size as f64;
            if growth > self.config.max_code_growth {
                continue;
            }

            // Check register pressure
            let estimated_pressure = analysis.register_pressure * factor as usize;
            if estimated_pressure > self.config.register_pressure_limit * 2 {
                continue;
            }

            let benefit = self.cost_model.unroll_benefit(analysis, factor);
            if benefit > best_benefit {
                best_benefit = benefit;
                best_factor = factor;
            }
        }

        if best_factor > 1 && best_benefit > 0.05 {
            Some(best_factor)
        } else {
            None
        }
    }

    /// Find the best runtime unroll factor.
    fn best_runtime_factor(&self, analysis: &UnrollabilityAnalysis) -> Option<u32> {
        // For runtime unrolling, we're more conservative
        // since we're adding a trip count check
        let max_factor = self.config.default_unroll_factor.min(4);

        for factor in [2, 4] {
            if factor > max_factor {
                break;
            }

            let growth = (analysis.body_size * factor as usize) as f64 / analysis.body_size as f64;
            if growth <= self.config.max_code_growth / 2.0 {
                return Some(factor);
            }
        }

        None
    }

    /// Choose the remainder handling strategy.
    fn choose_remainder_strategy(
        &self,
        analysis: &UnrollabilityAnalysis,
        factor: u32,
    ) -> RemainderStrategy {
        if !self.config.enable_remainder {
            return RemainderStrategy::None;
        }

        // For small factors, unrolled remainder is often best
        if factor <= 4 && analysis.body_size <= 16 {
            return RemainderStrategy::UnrolledRemainder;
        }

        // For larger factors or bodies, use an epilog loop
        if analysis.body_size * factor as usize <= 64 {
            return RemainderStrategy::UnrolledRemainder;
        }

        RemainderStrategy::EpilogLoop
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::NodeId;
    use crate::opt::unroll::analysis::LoopTripCount;
    use rustc_hash::FxHashSet;

    fn make_analysis(
        trip: LoopTripCount,
        body_size: usize,
        canonical: bool,
    ) -> UnrollabilityAnalysis {
        UnrollabilityAnalysis {
            loop_idx: 0,
            trip_count: trip,
            body_size,
            has_single_entry: canonical,
            has_single_exit: canonical,
            contains_calls: false,
            has_memory_effects: false,
            has_early_exits: !canonical,
            nesting_depth: 0,
            induction_vars: if canonical {
                vec![NodeId::new(5)]
            } else {
                vec![]
            },
            register_pressure: 4,
            is_canonical: canonical,
            body_nodes: FxHashSet::default(),
        }
    }

    // =========================================================================
    // Cost Model Tests
    // =========================================================================

    #[test]
    fn test_cost_model_default() {
        let model = UnrollCostModel::default();
        assert_eq!(model.loop_iteration_cost, 3.0);
        assert_eq!(model.target_register_count, 16);
    }

    #[test]
    fn test_cost_model_server() {
        let model = UnrollCostModel::server();
        assert_eq!(model.target_icache_size, 65536);
    }

    #[test]
    fn test_cost_model_mobile() {
        let model = UnrollCostModel::mobile();
        assert_eq!(model.target_icache_size, 16384);
        assert!(model.instruction_cost > UnrollCostModel::default().instruction_cost);
    }

    #[test]
    fn test_cost_model_baseline_cost() {
        let model = UnrollCostModel::default();
        let analysis = make_analysis(LoopTripCount::Constant(10), 10, true);
        let cost = model.baseline_cost(&analysis);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_model_full_unroll_cost() {
        let model = UnrollCostModel::default();
        let analysis = make_analysis(LoopTripCount::Constant(4), 10, true);
        let cost = model.full_unroll_cost(&analysis, 4);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_cost_model_partial_unroll_cost() {
        let model = UnrollCostModel::default();
        let analysis = make_analysis(LoopTripCount::Constant(100), 10, true);
        let cost = model.partial_unroll_cost(&analysis, 4);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_model_unroll_benefit() {
        let model = UnrollCostModel::default();
        let analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
        let benefit = model.unroll_benefit(&analysis, 4);
        // Full unroll of small loop should be beneficial
        assert!(benefit > 0.0);
    }

    // =========================================================================
    // Unroll Decision Tests
    // =========================================================================

    #[test]
    fn test_unroll_decision_no_unroll() {
        let decision = UnrollDecision::no_unroll(NoUnrollReason::BodyTooLarge);
        assert!(matches!(decision.strategy, UnrollStrategy::NoUnroll { .. }));
        assert_eq!(decision.estimated_speedup, 1.0);
    }

    // =========================================================================
    // Heuristics Tests
    // =========================================================================

    #[test]
    fn test_heuristics_full_unroll_small_loop() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
        let strategy = heuristics.determine_strategy(&analysis);

        assert!(matches!(
            strategy,
            UnrollStrategy::FullUnroll { trip_count: 4 }
        ));
    }

    #[test]
    fn test_heuristics_no_unroll_non_canonical() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let analysis = make_analysis(LoopTripCount::Constant(4), 5, false);
        let strategy = heuristics.determine_strategy(&analysis);

        assert!(matches!(
            strategy,
            UnrollStrategy::NoUnroll {
                reason: NoUnrollReason::NotCanonical
            }
        ));
    }

    #[test]
    fn test_heuristics_partial_unroll_large_trip() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let analysis = make_analysis(LoopTripCount::Constant(1000), 10, true);
        let strategy = heuristics.determine_strategy(&analysis);

        assert!(matches!(
            strategy,
            UnrollStrategy::PartialUnroll { .. } | UnrollStrategy::NoUnroll { .. }
        ));
    }

    #[test]
    fn test_heuristics_no_unroll_too_large_trip() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let analysis = make_analysis(LoopTripCount::Constant(32), 100, true);
        let strategy = heuristics.determine_strategy(&analysis);

        // Body too large for full unroll, should try partial or skip
        assert!(!matches!(strategy, UnrollStrategy::FullUnroll { .. }));
    }

    #[test]
    fn test_heuristics_chooses_epilog_for_large() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let remainder = heuristics
            .choose_remainder_strategy(&make_analysis(LoopTripCount::Constant(100), 50, true), 8);

        assert_eq!(remainder, RemainderStrategy::EpilogLoop);
    }

    #[test]
    fn test_heuristics_chooses_unrolled_for_small() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let remainder = heuristics
            .choose_remainder_strategy(&make_analysis(LoopTripCount::Constant(100), 8, true), 4);

        assert_eq!(remainder, RemainderStrategy::UnrolledRemainder);
    }

    #[test]
    fn test_heuristics_no_calls_block() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let mut analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
        analysis.contains_calls = true;

        let strategy = heuristics.determine_strategy(&analysis);

        assert!(matches!(
            strategy,
            UnrollStrategy::NoUnroll {
                reason: NoUnrollReason::ContainsCalls
            }
        ));
    }

    #[test]
    fn test_heuristics_aggressive_allows_calls() {
        let config = UnrollConfig::aggressive();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let mut analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
        analysis.contains_calls = true;

        let strategy = heuristics.determine_strategy(&analysis);

        // Aggressive config allows calls
        assert!(!matches!(
            strategy,
            UnrollStrategy::NoUnroll {
                reason: NoUnrollReason::ContainsCalls
            }
        ));
    }

    #[test]
    fn test_heuristics_nesting_limit() {
        let config = UnrollConfig::default();
        let model = UnrollCostModel::default();
        let heuristics = UnrollHeuristics::new(&config, &model);

        let mut analysis = make_analysis(LoopTripCount::Constant(4), 5, true);
        analysis.nesting_depth = 5; // Too deep

        let strategy = heuristics.determine_strategy(&analysis);

        assert!(matches!(
            strategy,
            UnrollStrategy::NoUnroll {
                reason: NoUnrollReason::NestingTooDeep
            }
        ));
    }
}
