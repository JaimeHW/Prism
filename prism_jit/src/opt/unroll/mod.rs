//! Loop Unrolling Optimization Pass.
//!
//! This module implements a production-quality loop unrolling system for the
//! Sea-of-Nodes IR. Loop unrolling replicates loop bodies to reduce loop overhead
//! and enable further optimizations.
//!
//! # Unrolling Strategies
//!
//! 1. **Full Unroll**: Complete elimination of small, constant-trip loops
//! 2. **Partial Unroll**: Replicate body N times while keeping the loop
//! 3. **Runtime Unroll**: Add trip count check for different unroll paths
//!
//! # Architecture
//!
//! - `analysis.rs`: Trip count analysis and unrollability detection
//! - `heuristics.rs`: Cost model and unroll factor selection
//! - `transform.rs`: Body replication and phi resolution
//! - `remainder.rs`: Epilog/prologue generation for remainders
//!
//! # Algorithm
//!
//! 1. Identify loops via loop analysis
//! 2. Analyze each loop for trip count and unrollability
//! 3. Apply cost model to select unroll strategy
//! 4. Transform the loop according to the strategy
//! 5. Clean up: DCE, simplify, etc.
//!
//! # Performance Characteristics
//!
//! - O(n) analysis per loop where n = loop body size
//! - O(n * factor) transformation for partial unrolling
//! - Memory growth bounded by configurable limits

mod analysis;
mod heuristics;
mod remainder;
mod transform;

pub use analysis::{
    LoopTripCount, TripCountAnalyzer, UnrollabilityAnalysis, UnrollabilityAnalyzer,
};
pub use heuristics::{UnrollCostModel, UnrollDecision, UnrollHeuristics};
pub use remainder::{RemainderGenerator, RemainderResult, RemainderStrategy};
pub use transform::{LoopUnroller, UnrollTransform};

use crate::ir::cfg::{Cfg, DominatorTree, LoopAnalysis};
use crate::ir::graph::Graph;
use crate::opt::OptimizationPass;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the loop unrolling pass.
#[derive(Debug, Clone)]
pub struct UnrollConfig {
    /// Maximum trip count for full unrolling.
    pub max_full_unroll_trip: u32,

    /// Maximum body size (in nodes) for full unrolling.
    pub max_full_unroll_size: usize,

    /// Default partial unroll factor.
    pub default_unroll_factor: u32,

    /// Maximum partial unroll factor.
    pub max_unroll_factor: u32,

    /// Maximum code growth factor (relative to original size).
    pub max_code_growth: f64,

    /// Enable runtime trip count checks.
    pub enable_runtime_unroll: bool,

    /// Minimum trip count for runtime unrolling to be worthwhile.
    pub min_runtime_trip: u32,

    /// Enable remainder loop generation.
    pub enable_remainder: bool,

    /// Enable aggressive unrolling for inner loops.
    pub aggressive_inner_loops: bool,

    /// Register pressure limit (approximate).
    pub register_pressure_limit: usize,

    /// Enable SIMD vectorization hints.
    pub enable_vectorization_hints: bool,
}

impl Default for UnrollConfig {
    fn default() -> Self {
        Self {
            max_full_unroll_trip: 16,
            max_full_unroll_size: 64,
            default_unroll_factor: 4,
            max_unroll_factor: 16,
            max_code_growth: 4.0,
            enable_runtime_unroll: true,
            min_runtime_trip: 8,
            enable_remainder: true,
            aggressive_inner_loops: false,
            register_pressure_limit: 16,
            enable_vectorization_hints: true,
        }
    }
}

impl UnrollConfig {
    /// Create a conservative configuration (less aggressive unrolling).
    pub fn conservative() -> Self {
        Self {
            max_full_unroll_trip: 8,
            max_full_unroll_size: 32,
            default_unroll_factor: 2,
            max_unroll_factor: 4,
            max_code_growth: 2.0,
            enable_runtime_unroll: false,
            min_runtime_trip: 16,
            enable_remainder: true,
            aggressive_inner_loops: false,
            register_pressure_limit: 12,
            enable_vectorization_hints: false,
        }
    }

    /// Create an aggressive configuration (maximum unrolling).
    pub fn aggressive() -> Self {
        Self {
            max_full_unroll_trip: 32,
            max_full_unroll_size: 128,
            default_unroll_factor: 8,
            max_unroll_factor: 32,
            max_code_growth: 8.0,
            enable_runtime_unroll: true,
            min_runtime_trip: 4,
            enable_remainder: true,
            aggressive_inner_loops: true,
            register_pressure_limit: 24,
            enable_vectorization_hints: true,
        }
    }

    /// Create a configuration optimized for tier-1 JIT (fast compile).
    pub fn tier1() -> Self {
        Self {
            max_full_unroll_trip: 4,
            max_full_unroll_size: 16,
            default_unroll_factor: 2,
            max_unroll_factor: 2,
            max_code_growth: 1.5,
            enable_runtime_unroll: false,
            min_runtime_trip: 32,
            enable_remainder: false,
            aggressive_inner_loops: false,
            register_pressure_limit: 8,
            enable_vectorization_hints: false,
        }
    }

    /// Create a configuration optimized for tier-2 JIT (thorough optimization).
    pub fn tier2() -> Self {
        Self::default()
    }
}

// =============================================================================
// Unroll Strategy
// =============================================================================

/// Strategy for unrolling a specific loop.
#[derive(Debug, Clone, PartialEq)]
pub enum UnrollStrategy {
    /// Fully unroll: completely eliminate the loop.
    FullUnroll {
        /// Known trip count.
        trip_count: u32,
    },

    /// Partially unroll: replicate body while keeping loop structure.
    PartialUnroll {
        /// Unroll factor (number of body copies).
        factor: u32,
        /// How to handle remainder iterations.
        remainder: RemainderStrategy,
    },

    /// Runtime unroll: add trip count check for different paths.
    RuntimeUnroll {
        /// Minimum trip count to use unrolled path.
        min_trip: u32,
        /// Unroll factor for the fast path.
        factor: u32,
        /// Remainder handling for the fast path.
        remainder: RemainderStrategy,
    },

    /// No unrolling is beneficial.
    NoUnroll {
        /// Reason for not unrolling.
        reason: NoUnrollReason,
    },
}

/// Reasons why a loop cannot or should not be unrolled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoUnrollReason {
    /// Loop body is too large.
    BodyTooLarge,
    /// Trip count cannot be analyzed.
    UnknownTripCount,
    /// Loop has complex control flow (multiple exits, breaks).
    ComplexControlFlow,
    /// Loop contains calls to unknown functions.
    ContainsCalls,
    /// Loop has side effects that prevent unrolling.
    SideEffects,
    /// Register pressure would be too high.
    RegisterPressure,
    /// Code growth would exceed limits.
    CodeGrowthLimit,
    /// Loop is not in canonical form.
    NotCanonical,
    /// Loop nesting is too deep.
    NestingTooDeep,
    /// Cost model says not profitable.
    NotProfitable,
    /// Loop is already unrolled.
    AlreadyUnrolled,
}

impl std::fmt::Display for NoUnrollReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoUnrollReason::BodyTooLarge => write!(f, "loop body too large"),
            NoUnrollReason::UnknownTripCount => write!(f, "unknown trip count"),
            NoUnrollReason::ComplexControlFlow => write!(f, "complex control flow"),
            NoUnrollReason::ContainsCalls => write!(f, "contains function calls"),
            NoUnrollReason::SideEffects => write!(f, "has side effects"),
            NoUnrollReason::RegisterPressure => write!(f, "register pressure too high"),
            NoUnrollReason::CodeGrowthLimit => write!(f, "code growth limit exceeded"),
            NoUnrollReason::NotCanonical => write!(f, "not in canonical form"),
            NoUnrollReason::NestingTooDeep => write!(f, "nesting too deep"),
            NoUnrollReason::NotProfitable => write!(f, "not profitable"),
            NoUnrollReason::AlreadyUnrolled => write!(f, "already unrolled"),
        }
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics from the loop unrolling pass.
#[derive(Debug, Clone, Default)]
pub struct UnrollStats {
    /// Number of loops analyzed.
    pub loops_analyzed: usize,
    /// Number of loops fully unrolled.
    pub loops_fully_unrolled: usize,
    /// Number of loops partially unrolled.
    pub loops_partially_unrolled: usize,
    /// Number of loops with runtime unrolling.
    pub loops_runtime_unrolled: usize,
    /// Number of loops not unrolled.
    pub loops_not_unrolled: usize,
    /// Total nodes added by unrolling.
    pub nodes_added: usize,
    /// Total nodes removed by cleanup.
    pub nodes_removed: usize,
    /// Reasons for not unrolling (histogram).
    pub no_unroll_reasons: [usize; 11],
}

impl UnrollStats {
    /// Record a no-unroll decision.
    pub fn record_no_unroll(&mut self, reason: NoUnrollReason) {
        self.loops_not_unrolled += 1;
        let idx = reason as usize;
        if idx < self.no_unroll_reasons.len() {
            self.no_unroll_reasons[idx] += 1;
        }
    }

    /// Get total loops processed.
    pub fn total_loops(&self) -> usize {
        self.loops_fully_unrolled
            + self.loops_partially_unrolled
            + self.loops_runtime_unrolled
            + self.loops_not_unrolled
    }

    /// Get unroll success rate.
    pub fn success_rate(&self) -> f64 {
        let total = self.total_loops();
        if total == 0 {
            0.0
        } else {
            let unrolled = self.loops_fully_unrolled
                + self.loops_partially_unrolled
                + self.loops_runtime_unrolled;
            unrolled as f64 / total as f64
        }
    }
}

// =============================================================================
// Main Unroll Pass
// =============================================================================

/// The main loop unrolling optimization pass.
#[derive(Debug)]
pub struct Unroll {
    /// Configuration.
    config: UnrollConfig,
    /// Statistics from the last run.
    stats: UnrollStats,
    /// Cost model.
    cost_model: UnrollCostModel,
}

impl Unroll {
    /// Create a new unroll pass with default configuration.
    pub fn new() -> Self {
        Self {
            config: UnrollConfig::default(),
            stats: UnrollStats::default(),
            cost_model: UnrollCostModel::default(),
        }
    }

    /// Create an unroll pass with custom configuration.
    pub fn with_config(config: UnrollConfig) -> Self {
        Self {
            config,
            stats: UnrollStats::default(),
            cost_model: UnrollCostModel::default(),
        }
    }

    /// Create a conservative unroll pass.
    pub fn conservative() -> Self {
        Self::with_config(UnrollConfig::conservative())
    }

    /// Create an aggressive unroll pass.
    pub fn aggressive() -> Self {
        Self::with_config(UnrollConfig::aggressive())
    }

    /// Get the configuration.
    pub fn config(&self) -> &UnrollConfig {
        &self.config
    }

    /// Get statistics from the last run.
    pub fn stats(&self) -> &UnrollStats {
        &self.stats
    }

    /// Run the unroll pass on the graph.
    pub fn run_unroll(&mut self, graph: &mut Graph) -> bool {
        self.stats = UnrollStats::default();

        // Build analysis infrastructure
        let cfg = Cfg::build(graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        if loops.loops.is_empty() {
            return false;
        }

        let mut changed = false;

        // Process loops from innermost to outermost (reverse order)
        let loop_indices: Vec<usize> = (0..loops.loops.len()).rev().collect();

        for loop_idx in loop_indices {
            self.stats.loops_analyzed += 1;

            // Analyze unrollability
            let analyzer = UnrollabilityAnalyzer::new(graph, &loops, &cfg);
            let analysis = match analyzer.analyze(loop_idx) {
                Some(a) => a,
                None => {
                    self.stats.record_no_unroll(NoUnrollReason::NotCanonical);
                    continue;
                }
            };

            // Determine strategy
            let heuristics = UnrollHeuristics::new(&self.config, &self.cost_model);
            let strategy = heuristics.determine_strategy(&analysis);

            // Apply transformation - need to rebuild cfg as mutable
            let mut cfg_mut = Cfg::build(graph);

            match &strategy {
                UnrollStrategy::FullUnroll { trip_count } => {
                    let transform = UnrollTransform::new(graph, &mut cfg_mut, &analysis);
                    if transform.full_unroll(*trip_count) {
                        self.stats.loops_fully_unrolled += 1;
                        changed = true;
                    }
                }
                UnrollStrategy::PartialUnroll { factor, remainder } => {
                    let transform = UnrollTransform::new(graph, &mut cfg_mut, &analysis);
                    if transform.partial_unroll(*factor, *remainder) {
                        self.stats.loops_partially_unrolled += 1;
                        changed = true;
                    }
                }
                UnrollStrategy::RuntimeUnroll {
                    min_trip,
                    factor,
                    remainder,
                } => {
                    if self.config.enable_runtime_unroll {
                        let transform = UnrollTransform::new(graph, &mut cfg_mut, &analysis);
                        if transform.runtime_unroll(*min_trip, *factor, *remainder) {
                            self.stats.loops_runtime_unrolled += 1;
                            changed = true;
                        }
                    } else {
                        self.stats.record_no_unroll(NoUnrollReason::NotProfitable);
                    }
                }
                UnrollStrategy::NoUnroll { reason } => {
                    self.stats.record_no_unroll(*reason);
                }
            }
        }

        changed
    }
}

impl Default for Unroll {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Unroll {
    fn name(&self) -> &'static str {
        "Unroll"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_unroll(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_unroll_config_default() {
        let config = UnrollConfig::default();
        assert_eq!(config.max_full_unroll_trip, 16);
        assert_eq!(config.max_full_unroll_size, 64);
        assert_eq!(config.default_unroll_factor, 4);
        assert_eq!(config.max_unroll_factor, 16);
        assert!(config.enable_runtime_unroll);
        assert!(config.enable_remainder);
    }

    #[test]
    fn test_unroll_config_conservative() {
        let config = UnrollConfig::conservative();
        assert_eq!(config.max_full_unroll_trip, 8);
        assert_eq!(config.max_full_unroll_size, 32);
        assert_eq!(config.default_unroll_factor, 2);
        assert!(!config.enable_runtime_unroll);
    }

    #[test]
    fn test_unroll_config_aggressive() {
        let config = UnrollConfig::aggressive();
        assert_eq!(config.max_full_unroll_trip, 32);
        assert_eq!(config.max_full_unroll_size, 128);
        assert_eq!(config.default_unroll_factor, 8);
        assert!(config.aggressive_inner_loops);
    }

    #[test]
    fn test_unroll_config_tier1() {
        let config = UnrollConfig::tier1();
        assert_eq!(config.max_full_unroll_trip, 4);
        assert!(!config.enable_runtime_unroll);
        assert!(!config.enable_remainder);
    }

    #[test]
    fn test_unroll_config_tier2() {
        let config = UnrollConfig::tier2();
        assert_eq!(config.max_full_unroll_trip, 16);
        assert!(config.enable_runtime_unroll);
    }

    // =========================================================================
    // Strategy Tests
    // =========================================================================

    #[test]
    fn test_unroll_strategy_full() {
        let strategy = UnrollStrategy::FullUnroll { trip_count: 4 };
        assert!(matches!(strategy, UnrollStrategy::FullUnroll { .. }));
    }

    #[test]
    fn test_unroll_strategy_partial() {
        let strategy = UnrollStrategy::PartialUnroll {
            factor: 4,
            remainder: RemainderStrategy::EpilogLoop,
        };
        if let UnrollStrategy::PartialUnroll { factor, remainder } = strategy {
            assert_eq!(factor, 4);
            assert_eq!(remainder, RemainderStrategy::EpilogLoop);
        } else {
            panic!("Wrong strategy type");
        }
    }

    #[test]
    fn test_unroll_strategy_runtime() {
        let strategy = UnrollStrategy::RuntimeUnroll {
            min_trip: 8,
            factor: 4,
            remainder: RemainderStrategy::UnrolledRemainder,
        };
        if let UnrollStrategy::RuntimeUnroll {
            min_trip,
            factor,
            remainder,
        } = strategy
        {
            assert_eq!(min_trip, 8);
            assert_eq!(factor, 4);
            assert_eq!(remainder, RemainderStrategy::UnrolledRemainder);
        } else {
            panic!("Wrong strategy type");
        }
    }

    #[test]
    fn test_unroll_strategy_no_unroll() {
        let strategy = UnrollStrategy::NoUnroll {
            reason: NoUnrollReason::BodyTooLarge,
        };
        if let UnrollStrategy::NoUnroll { reason } = strategy {
            assert_eq!(reason, NoUnrollReason::BodyTooLarge);
        } else {
            panic!("Wrong strategy type");
        }
    }

    // =========================================================================
    // NoUnrollReason Tests
    // =========================================================================

    #[test]
    fn test_no_unroll_reason_display() {
        assert_eq!(
            NoUnrollReason::BodyTooLarge.to_string(),
            "loop body too large"
        );
        assert_eq!(
            NoUnrollReason::UnknownTripCount.to_string(),
            "unknown trip count"
        );
        assert_eq!(
            NoUnrollReason::ComplexControlFlow.to_string(),
            "complex control flow"
        );
        assert_eq!(
            NoUnrollReason::ContainsCalls.to_string(),
            "contains function calls"
        );
        assert_eq!(NoUnrollReason::SideEffects.to_string(), "has side effects");
        assert_eq!(
            NoUnrollReason::RegisterPressure.to_string(),
            "register pressure too high"
        );
        assert_eq!(
            NoUnrollReason::CodeGrowthLimit.to_string(),
            "code growth limit exceeded"
        );
        assert_eq!(
            NoUnrollReason::NotCanonical.to_string(),
            "not in canonical form"
        );
        assert_eq!(
            NoUnrollReason::NestingTooDeep.to_string(),
            "nesting too deep"
        );
        assert_eq!(NoUnrollReason::NotProfitable.to_string(), "not profitable");
        assert_eq!(
            NoUnrollReason::AlreadyUnrolled.to_string(),
            "already unrolled"
        );
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    #[test]
    fn test_unroll_stats_default() {
        let stats = UnrollStats::default();
        assert_eq!(stats.loops_analyzed, 0);
        assert_eq!(stats.loops_fully_unrolled, 0);
        assert_eq!(stats.loops_partially_unrolled, 0);
        assert_eq!(stats.loops_runtime_unrolled, 0);
        assert_eq!(stats.loops_not_unrolled, 0);
        assert_eq!(stats.nodes_added, 0);
        assert_eq!(stats.nodes_removed, 0);
    }

    #[test]
    fn test_unroll_stats_record_no_unroll() {
        let mut stats = UnrollStats::default();
        stats.record_no_unroll(NoUnrollReason::BodyTooLarge);
        assert_eq!(stats.loops_not_unrolled, 1);
        assert_eq!(
            stats.no_unroll_reasons[NoUnrollReason::BodyTooLarge as usize],
            1
        );
    }

    #[test]
    fn test_unroll_stats_total_loops() {
        let mut stats = UnrollStats::default();
        stats.loops_fully_unrolled = 2;
        stats.loops_partially_unrolled = 3;
        stats.loops_runtime_unrolled = 1;
        stats.loops_not_unrolled = 4;
        assert_eq!(stats.total_loops(), 10);
    }

    #[test]
    fn test_unroll_stats_success_rate() {
        let mut stats = UnrollStats::default();
        stats.loops_fully_unrolled = 3;
        stats.loops_partially_unrolled = 2;
        stats.loops_not_unrolled = 5;
        assert!((stats.success_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_unroll_stats_success_rate_empty() {
        let stats = UnrollStats::default();
        assert_eq!(stats.success_rate(), 0.0);
    }

    // =========================================================================
    // Unroll Pass Tests
    // =========================================================================

    #[test]
    fn test_unroll_new() {
        let unroll = Unroll::new();
        assert_eq!(unroll.config().max_full_unroll_trip, 16);
    }

    #[test]
    fn test_unroll_with_config() {
        let config = UnrollConfig::conservative();
        let unroll = Unroll::with_config(config.clone());
        assert_eq!(
            unroll.config().max_full_unroll_trip,
            config.max_full_unroll_trip
        );
    }

    #[test]
    fn test_unroll_conservative() {
        let unroll = Unroll::conservative();
        assert_eq!(unroll.config().max_full_unroll_trip, 8);
    }

    #[test]
    fn test_unroll_aggressive() {
        let unroll = Unroll::aggressive();
        assert_eq!(unroll.config().max_full_unroll_trip, 32);
    }

    #[test]
    fn test_unroll_name() {
        let unroll = Unroll::new();
        assert_eq!(unroll.name(), "Unroll");
    }

    #[test]
    fn test_unroll_no_loops() {
        use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

        let mut builder = GraphBuilder::new(4, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let mut unroll = Unroll::new();
        let changed = unroll.run(&mut graph);

        assert!(!changed);
        assert_eq!(unroll.stats().loops_analyzed, 0);
    }

    #[test]
    fn test_unroll_stats_after_run() {
        use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

        let mut builder = GraphBuilder::new(4, 0);
        let c = builder.const_int(42);
        builder.return_value(c);

        let mut graph = builder.finish();
        let mut unroll = Unroll::new();
        unroll.run(&mut graph);

        // Stats should be reset each run
        assert_eq!(unroll.stats().loops_analyzed, 0);
    }

    #[test]
    fn test_unroll_default_impl() {
        let unroll = Unroll::default();
        assert_eq!(unroll.config().max_full_unroll_trip, 16);
    }
}
