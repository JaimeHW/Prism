//! Range Check Elimination (RCE) optimization pass.
//!
//! This module provides a high-performance loop optimization that eliminates
//! redundant bounds checks by analyzing induction variables and proving that
//! checks are always satisfied.
//!
//! # Module Structure
//!
//! - [`induction`]: Induction variable detection and analysis
//! - [`bounds`]: Range check identification and classification
//! - [`elimination`]: Elimination decision logic
//! - [`trip_count`]: Loop iteration count analysis
//! - [`transform`]: Graph transformation for applying decisions
//!
//! # Algorithm Overview
//!
//! 1. **Loop Analysis**: Build CFG, dominator tree, and loop nest
//! 2. **IV Detection**: Find induction variables (LoopPhi + constant step)
//! 3. **Trip Count**: Compute loop iteration counts for widening
//! 4. **Check Collection**: Identify Guard(Bounds) nodes on IVs
//! 5. **Elimination Analysis**: Determine which checks can be eliminated
//! 6. **Transformation**: Apply decisions to modify IR graph
//!
//! # Performance Benefits
//!
//! - Removes N-1 bounds checks per loop iteration
//! - Enables vectorization by removing data dependencies
//! - Reduces branch misprediction overhead
//! - Improves instruction cache utilization
//!
//! # Example
//!
//! Before RCE:
//! ```text
//! for i in range(0, n):
//!     guard(i >= 0)        # Checked every iteration
//!     guard(i < len(arr))  # Checked every iteration
//!     x = arr[i]
//! ```
//!
//! After RCE:
//! ```text
//! guard(n > 0 => n-1 < len(arr))  # Single hoisted check
//! for i in range(0, n):
//!     x = arr[i]                   # No bounds checks!
//! ```

pub mod bounds;
pub mod elimination;
pub mod guard_insert;
pub mod induction;
pub mod transform;
pub mod trip_count;

// Re-export primary types
pub use bounds::{
    BoundValue, RangeCheck, RangeCheckCollection, RangeCheckDetector, RangeCheckKind,
};
pub use elimination::{
    EliminationAnalyzer, EliminationDecision, EliminationResult, EliminationStats, HoistInfo,
    WidenInfo,
};
pub use guard_insert::{GuardInserter, InsertionResult};
pub use induction::{
    InductionAnalysis, InductionDetector, InductionDirection, InductionInit, InductionStep,
    InductionVariable,
};
pub use transform::{
    PreheaderUtils, RceTransformContext, RceTransformer, TransformationResult, WidenBounds,
    WidenCalculator,
};
pub use trip_count::{
    MaxIVValue, SymbolicTripCount, TripCount, TripCountAnalyzer, TripCountCache, TripCountValue,
};

use crate::ir::cfg::{Cfg, DominatorTree, LoopAnalysis};
use crate::ir::graph::Graph;

use super::OptimizationPass;

// =============================================================================
// RCE Pass
// =============================================================================

/// Range Check Elimination optimization pass.
///
/// This pass eliminates redundant bounds checks in loops by analyzing
/// induction variables and proving that checks are always satisfied.
#[derive(Debug)]
pub struct RangeCheckElimination {
    /// Enable aggressive optimization.
    aggressive: bool,

    /// Statistics from the last run.
    stats: RceStats,
}

/// Statistics from RCE.
#[derive(Debug, Clone, Default)]
pub struct RceStats {
    /// Number of loops analyzed.
    pub loops_analyzed: usize,

    /// Number of induction variables found.
    pub induction_vars_found: usize,

    /// Number of range checks found.
    pub range_checks_found: usize,

    /// Number of checks eliminated.
    pub checks_eliminated: usize,

    /// Number of checks hoisted.
    pub checks_hoisted: usize,

    /// Number of checks widened.
    pub checks_widened: usize,
}

impl RangeCheckElimination {
    /// Create a new RCE pass.
    #[inline]
    pub fn new() -> Self {
        Self {
            aggressive: false,
            stats: RceStats::default(),
        }
    }

    /// Create an aggressive RCE pass.
    #[inline]
    pub fn aggressive() -> Self {
        Self {
            aggressive: true,
            stats: RceStats::default(),
        }
    }

    /// Get statistics from the last run.
    #[inline]
    pub fn stats(&self) -> &RceStats {
        &self.stats
    }

    /// Get number of checks eliminated.
    #[inline]
    pub fn checks_eliminated(&self) -> usize {
        self.stats.checks_eliminated
    }

    /// Get number of checks hoisted.
    #[inline]
    pub fn checks_hoisted(&self) -> usize {
        self.stats.checks_hoisted
    }

    /// Get number of induction variables found.
    #[inline]
    pub fn induction_vars_found(&self) -> usize {
        self.stats.induction_vars_found
    }

    /// Run the RCE optimization.
    pub fn run_rce(&mut self, graph: &mut Graph) -> bool {
        // Reset stats
        self.stats = RceStats::default();

        // Build required analyses
        let cfg = Cfg::build(graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        if loops.loops.is_empty() {
            return false;
        }

        // Phase 1: Collect induction variables for all loops
        let iv_detector = InductionDetector::new(graph);
        let mut iv_analysis = InductionAnalysis::with_capacity(loops.loops.len());

        for loop_info in &loops.loops {
            let ivs = iv_detector.find_induction_variables(loop_info);
            self.stats.induction_vars_found += ivs.len();
            iv_analysis.add_loop(ivs);
        }

        if iv_analysis.total() == 0 {
            return false;
        }

        // Phase 2: Collect range checks for all loops
        let check_detector = RangeCheckDetector::new(graph);
        let mut all_checks = RangeCheckCollection::new();

        for (loop_idx, loop_info) in loops.loops.iter().enumerate() {
            let checks = check_detector.find_range_checks(loop_info, loop_idx, &iv_analysis);
            self.stats.range_checks_found += checks.len();
            all_checks.add_all(checks);
        }

        if all_checks.is_empty() {
            return false;
        }

        // Phase 3: Analyze elimination opportunities
        let mut analyzer = if self.aggressive {
            EliminationAnalyzer::aggressive()
        } else {
            EliminationAnalyzer::new()
        };

        let mut result = EliminationResult::with_capacity(all_checks.len());

        for check in all_checks.all() {
            if let Some(iv) = iv_analysis.get_iv(check.loop_idx, check.induction_var) {
                let decision = analyzer.analyze(check, iv);
                result.add(check.clone(), decision);
            }
        }

        // Phase 4: Apply transformations
        let mut transformer = RceTransformer::new(graph);
        let mut ctx = RceTransformContext::new(&cfg, &loops, &iv_analysis);
        let transform_result = transformer.apply_with_context(&result, &mut ctx);

        // Update stats from transformation
        self.stats.checks_eliminated = result.count_eliminable();
        self.stats.checks_hoisted = result.count_hoistable();
        self.stats.checks_widened = result.count_widenable();
        self.stats.loops_analyzed = loops.loops.len();

        transform_result.has_changes()
    }
}

impl Default for RangeCheckElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for RangeCheckElimination {
    fn name(&self) -> &'static str {
        "rce"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_rce(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
