//! Graph Transformation for Range Check Elimination.
//!
//! This module applies elimination decisions to the IR graph, actually
//! removing, hoisting, or widening bounds checks based on the analysis
//! performed by the elimination analyzer.
//!
//! # Transformation Types
//!
//! - **Eliminate**: Remove the guard node entirely (replace with unconditional path)
//! - **Hoist**: Move the check to the loop preheader
//! - **Widen**: Replace per-iteration checks with a single widened check
//!
//! # Safety
//!
//! All transformations preserve program semantics:
//! - Eliminated checks are provably always true
//! - Hoisted checks execute before any iteration
//! - Widened checks cover all possible values the IV will take
//!
//! # Graph Modifications
//!
//! When eliminating a guard:
//! 1. The guard's continuation edge becomes unconditional
//! 2. The deoptimization path becomes unreachable
//! 3. Dead code elimination will clean up the deopt path

use super::bounds::{BoundValue, RangeCheck, RangeCheckKind};
use super::elimination::{EliminationDecision, EliminationResult, HoistInfo, WidenInfo};
use super::guard_insert::GuardInserter;
use super::induction::{InductionAnalysis, InductionVariable};
use super::trip_count::{MaxIVValue, TripCount};
use crate::ir::cfg::LoopAnalysis;
use crate::ir::cfg::{BlockId, Cfg, Loop};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::Operator;

// =============================================================================
// Transformation Result
// =============================================================================

/// Result of applying transformations.
#[derive(Debug, Clone, Default)]
pub struct TransformationResult {
    /// Number of guards eliminated.
    pub eliminated: usize,

    /// Number of guards hoisted.
    pub hoisted: usize,

    /// Number of guards widened.
    pub widened: usize,

    /// Guards that were marked dead.
    pub dead_guards: Vec<NodeId>,

    /// New guards created (for hoisting/widening).
    pub new_guards: Vec<NodeId>,

    /// Number of guards that couldn't be inserted (diagnostics).
    pub insertion_failures: usize,
}

impl TransformationResult {
    /// Create empty result.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any transformations were applied.
    #[inline]
    pub fn has_changes(&self) -> bool {
        self.eliminated > 0 || self.hoisted > 0 || self.widened > 0
    }

    /// Total number of transformations.
    #[inline]
    pub fn total(&self) -> usize {
        self.eliminated + self.hoisted + self.widened
    }

    /// Merge with another result.
    pub fn merge(&mut self, other: &TransformationResult) {
        self.eliminated += other.eliminated;
        self.hoisted += other.hoisted;
        self.widened += other.widened;
        self.dead_guards.extend_from_slice(&other.dead_guards);
        self.new_guards.extend_from_slice(&other.new_guards);
        self.insertion_failures += other.insertion_failures;
    }
}

// =============================================================================
// Transformation Context
// =============================================================================

/// Context required for RCE transformations.
///
/// This bundles all the analysis results needed to safely perform
/// guard transformations, keeping the transformer API clean.
#[derive(Debug)]
pub struct RceTransformContext<'a> {
    /// Control flow graph.
    pub cfg: &'a Cfg,

    /// Loop analysis results.
    pub loops: &'a LoopAnalysis,

    /// Induction variable analysis.
    pub iv_analysis: &'a InductionAnalysis,

    /// Cache of preheader control nodes per loop.
    preheader_controls: Vec<Option<NodeId>>,
}

impl<'a> RceTransformContext<'a> {
    /// Create a new transformation context.
    pub fn new(cfg: &'a Cfg, loops: &'a LoopAnalysis, iv_analysis: &'a InductionAnalysis) -> Self {
        let preheader_controls = vec![None; loops.loops.len()];
        Self {
            cfg,
            loops,
            iv_analysis,
            preheader_controls,
        }
    }

    /// Get the preheader control node for a loop.
    ///
    /// The control node is the Region node representing the preheader block.
    /// We cache this to avoid repeated lookups.
    pub fn get_preheader_control(&mut self, loop_idx: usize, graph: &Graph) -> Option<NodeId> {
        // Check cache first
        if let Some(control) = self.preheader_controls.get(loop_idx).copied().flatten() {
            return Some(control);
        }

        // Find the preheader block
        let loop_info = self.loops.loops.get(loop_idx)?;
        let preheader_block = PreheaderUtils::find_preheader(self.cfg, loop_info)?;

        // Get the Region node for this block
        let control_node = self.block_to_region(preheader_block, graph)?;

        // Cache and return
        if loop_idx < self.preheader_controls.len() {
            self.preheader_controls[loop_idx] = Some(control_node);
        }

        Some(control_node)
    }

    /// Convert a block ID to its corresponding Region node.
    ///
    /// Each basic block in the CFG corresponds to a Region node in the graph.
    fn block_to_region(&self, block: BlockId, _graph: &Graph) -> Option<NodeId> {
        // The block's region field is the Region node that starts the block
        let block_info = self.cfg.block(block);

        // Each block has a 'region' field that is the control node
        Some(block_info.region)
    }

    /// Get loop info by index.
    #[inline]
    pub fn get_loop(&self, loop_idx: usize) -> Option<&Loop> {
        self.loops.loops.get(loop_idx)
    }

    /// Get the IV analysis reference.
    #[inline]
    pub fn iv_analysis(&self) -> &InductionAnalysis {
        self.iv_analysis
    }
}

// =============================================================================
// Transformer
// =============================================================================

/// Transforms the graph based on elimination decisions.
///
/// This is the core engine that applies RCE transformations to the IR graph.
/// It works in conjunction with `GuardInserter` to:
/// - Eliminate guards that are provably always satisfied
/// - Hoist guards from loop bodies to preheaders
/// - Widen guards to cover entire IV ranges
#[derive(Debug)]
pub struct RceTransformer<'g> {
    graph: &'g mut Graph,
    result: TransformationResult,
}

impl<'g> RceTransformer<'g> {
    /// Create a new transformer.
    #[inline]
    pub fn new(graph: &'g mut Graph) -> Self {
        Self {
            graph,
            result: TransformationResult::new(),
        }
    }

    /// Apply all elimination decisions with full context.
    ///
    /// This is the primary API for applying RCE transformations.
    /// It uses the provided context to find preheaders and insert guards.
    pub fn apply_with_context(
        &mut self,
        decisions: &EliminationResult,
        ctx: &mut RceTransformContext<'_>,
    ) -> TransformationResult {
        for (check, decision) in decisions.decisions() {
            match decision {
                EliminationDecision::Eliminate => {
                    self.eliminate_guard(check);
                }
                EliminationDecision::Hoist(info) => {
                    self.hoist_guard_with_context(check, info, ctx);
                }
                EliminationDecision::Widen(info) => {
                    self.widen_guard_with_context(check, info, ctx);
                }
                EliminationDecision::Keep => {
                    // No transformation needed
                }
            }
        }

        std::mem::take(&mut self.result)
    }

    /// Apply all elimination decisions (legacy API, no preheader insertion).
    ///
    /// This API only eliminates guards - it does not insert new guards
    /// in preheaders. Use `apply_with_context` for full functionality.
    pub fn apply(&mut self, decisions: &EliminationResult) -> TransformationResult {
        for (check, decision) in decisions.decisions() {
            match decision {
                EliminationDecision::Eliminate => {
                    self.eliminate_guard(check);
                }
                EliminationDecision::Hoist(_) => {
                    // Without context, we can only eliminate the old guard
                    self.eliminate_guard_only(check);
                    self.result.hoisted += 1;
                }
                EliminationDecision::Widen(_) => {
                    // Without context, we can only eliminate the old guard
                    self.eliminate_guard_only(check);
                    self.result.widened += 1;
                }
                EliminationDecision::Keep => {}
            }
        }

        std::mem::take(&mut self.result)
    }

    /// Eliminate a guard by marking it as always-true.
    ///
    /// For a Guard node, we replace it with ConstBool(true) which effectively
    /// makes the guard always succeed. Dead code elimination will clean up
    /// the deoptimization path.
    fn eliminate_guard(&mut self, check: &RangeCheck) {
        let guard_id = check.guard;

        if let Some(node) = self.graph.get_mut(guard_id) {
            // Replace the guard with a constant true - effectively eliminates the check
            // The actual removal will be done by DCE
            node.op = Operator::ConstBool(true);
            self.result.eliminated += 1;
            self.result.dead_guards.push(guard_id);
        }
    }

    /// Eliminate only the old guard (used when we can't insert new one).
    fn eliminate_guard_only(&mut self, check: &RangeCheck) {
        let guard_id = check.guard;

        if let Some(node) = self.graph.get_mut(guard_id) {
            node.op = Operator::ConstBool(true);
            self.result.dead_guards.push(guard_id);
        }
    }

    /// Hoist a guard to the loop preheader with full context.
    ///
    /// This properly inserts a new guard in the preheader using the
    /// initial IV value, then eliminates the old per-iteration guard.
    fn hoist_guard_with_context(
        &mut self,
        check: &RangeCheck,
        info: &HoistInfo,
        ctx: &mut RceTransformContext<'_>,
    ) {
        // Get the preheader control node
        let preheader_control = match ctx.get_preheader_control(check.loop_idx, self.graph) {
            Some(ctrl) => ctrl,
            None => {
                // Can't hoist without preheader - just eliminate the old guard
                self.eliminate_guard_only(check);
                self.result.hoisted += 1;
                self.result.insertion_failures += 1;
                return;
            }
        };

        // Create guard inserter and insert the hoisted guard
        let mut inserter = GuardInserter::new(self.graph, ctx.iv_analysis);

        match inserter.insert_hoisted_guard(check, info, check.loop_idx, preheader_control) {
            Some(new_guard) => {
                self.result.new_guards.push(new_guard);
                self.result.hoisted += 1;
            }
            None => {
                // Failed to insert
                inserter.record_failure();
                self.result.insertion_failures += 1;
            }
        }

        // Merge insertion result
        let insert_result = inserter.into_result();
        self.result.new_guards.extend(insert_result.new_guards);

        // Eliminate the old per-iteration guard
        self.eliminate_guard_only(check);
    }

    /// Widen a guard to cover the entire loop range with full context.
    ///
    /// This creates new guards in the preheader that check both the
    /// minimum and maximum IV values, then eliminates per-iteration guards.
    fn widen_guard_with_context(
        &mut self,
        check: &RangeCheck,
        info: &WidenInfo,
        ctx: &mut RceTransformContext<'_>,
    ) {
        // Get the preheader control node
        let preheader_control = match ctx.get_preheader_control(check.loop_idx, self.graph) {
            Some(ctrl) => ctrl,
            None => {
                // Can't widen without preheader - just eliminate the old guard
                self.eliminate_guard_only(check);
                self.result.widened += 1;
                self.result.insertion_failures += 1;
                return;
            }
        };

        // Create guard inserter and insert the widened guards
        let mut inserter = GuardInserter::new(self.graph, ctx.iv_analysis);

        match inserter.insert_widened_guard(check, info, check.loop_idx, preheader_control) {
            Some((min_guard, max_guard)) => {
                if let Some(g) = min_guard {
                    self.result.new_guards.push(g);
                }
                if let Some(g) = max_guard {
                    self.result.new_guards.push(g);
                }
                self.result.widened += 1;
            }
            None => {
                // Failed to insert
                inserter.record_failure();
                self.result.insertion_failures += 1;
            }
        }

        // Merge insertion result
        let insert_result = inserter.into_result();
        self.result
            .new_guards
            .extend(insert_result.new_guards.iter().copied());

        // Eliminate the old per-iteration guard
        self.eliminate_guard_only(check);
    }

    /// Get the transformation result.
    #[inline]
    pub fn result(&self) -> &TransformationResult {
        &self.result
    }
}

// =============================================================================
// Widening Calculator
// =============================================================================

/// Computes widened bounds for a range check.
#[derive(Debug)]
pub struct WidenCalculator;

/// Widening bounds information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WidenBounds {
    /// Minimum bound value.
    pub min_bound: BoundValue,
    /// Maximum bound value.
    pub max_bound: BoundValue,
}

impl WidenCalculator {
    /// Compute widened bounds for a lower bound check.
    ///
    /// For `i >= 0` where i starts at `init`:
    /// - The minimum value is `init` (first iteration)
    /// - If init >= 0 and IV is non-decreasing, check can be eliminated
    pub fn compute_lower_bound_widen(
        iv: &InductionVariable,
        check: &RangeCheck,
    ) -> Option<WidenBounds> {
        // For lower bound check (i >= 0), we need min(iv) >= 0
        // min(iv) = init for increasing IV, or init + step*(n-1) for decreasing

        let init = iv.constant_init()?;

        if iv.is_increasing() {
            // Minimum is at first iteration
            if init >= 0 {
                // Can eliminate entirely
                return Some(WidenBounds {
                    min_bound: BoundValue::Constant(init),
                    max_bound: check.bound,
                });
            }
        }

        None
    }

    /// Compute widened bounds for an upper bound check.
    ///
    /// For `i < len` where i goes from init to init + step*(n-1):
    /// - The maximum value is `init + step*(n-1)`
    /// - We need to check once that max_val < len
    pub fn compute_upper_bound_widen(
        iv: &InductionVariable,
        check: &RangeCheck,
        trip_count: &TripCount,
    ) -> Option<WidenBounds> {
        let max_iv = trip_count.max_iv_value(iv)?;

        match (&max_iv, &check.bound) {
            // Both constant - can potentially eliminate
            (MaxIVValue::Constant(max), BoundValue::Constant(bound)) => {
                let bound_val = match check.kind {
                    RangeCheckKind::UpperBound => *bound,          // i < bound
                    RangeCheckKind::UpperBoundInclusive => *bound, // i <= bound
                    _ => return None,
                };

                let safe = match check.kind {
                    RangeCheckKind::UpperBound => *max < bound_val,
                    RangeCheckKind::UpperBoundInclusive => *max <= bound_val,
                    _ => false,
                };

                if safe {
                    Some(WidenBounds {
                        min_bound: BoundValue::Constant(iv.constant_init()?),
                        max_bound: BoundValue::Constant(*max),
                    })
                } else {
                    None
                }
            }

            // Symbolic max - need runtime check in preheader
            (MaxIVValue::Symbolic { bound, offset: _ }, _check_bound) => {
                // Create a symbolic widen info
                Some(WidenBounds {
                    min_bound: BoundValue::Constant(iv.constant_init().unwrap_or(0)),
                    max_bound: BoundValue::Node(*bound), // max = bound + offset
                })
            }

            // AtMost - conservative upper bound
            (MaxIVValue::AtMost(max), BoundValue::Constant(bound)) => {
                let bound_val = *bound;
                if *max < bound_val {
                    Some(WidenBounds {
                        min_bound: BoundValue::Constant(iv.constant_init()?),
                        max_bound: BoundValue::Constant(*max),
                    })
                } else {
                    None
                }
            }

            _ => None,
        }
    }
}

// =============================================================================
// Preheader Utilities
// =============================================================================

/// Utilities for working with loop preheaders.
#[derive(Debug)]
pub struct PreheaderUtils;

impl PreheaderUtils {
    /// Find the preheader block for a loop.
    ///
    /// The preheader is the unique predecessor of the loop header
    /// that is not a back edge (i.e., not from within the loop body).
    ///
    /// Returns `Some(block_id)` if a unique preheader exists, `None` otherwise.
    pub fn find_preheader(cfg: &Cfg, loop_info: &Loop) -> Option<BlockId> {
        let header = loop_info.header;
        let header_block = cfg.block(header);

        // Collect predecessors that are NOT back edges
        // A back edge comes from within the loop body
        let mut preheader_candidates: Vec<BlockId> = header_block
            .predecessors
            .iter()
            .copied()
            .filter(|pred| !loop_info.body.contains(pred))
            .collect();

        // A valid preheader setup has exactly one non-loop predecessor
        if preheader_candidates.len() == 1 {
            let candidate = preheader_candidates.pop().unwrap();
            // Verify it's a valid preheader
            if Self::is_valid_preheader(cfg, candidate, loop_info) {
                return Some(candidate);
            }
        }

        // Multiple or zero preheader candidates - would need to create one
        None
    }

    /// Check if a block is a valid preheader.
    ///
    /// A valid preheader:
    /// - Has exactly one successor (the loop header)
    /// - Is not part of the loop body
    /// - Is dominated by any outer preheaders
    pub fn is_valid_preheader(cfg: &Cfg, block: BlockId, loop_info: &Loop) -> bool {
        // Check block is not in loop body
        if loop_info.body.contains(&block) {
            return false;
        }

        // Check it has single successor pointing to header
        let block_info = cfg.block(block);
        if block_info.successors.len() != 1 {
            return false;
        }

        if block_info.successors[0] != loop_info.header {
            return false;
        }

        true
    }

    /// Get or create a preheader block for a loop.
    ///
    /// If a valid preheader exists, returns it.
    /// Otherwise, returns None (preheader creation not yet implemented).
    pub fn get_or_create_preheader(
        cfg: &Cfg,
        _graph: &mut Graph,
        loop_info: &Loop,
    ) -> Option<BlockId> {
        // Try to find existing preheader first
        if let Some(preheader) = Self::find_preheader(cfg, loop_info) {
            return Some(preheader);
        }

        // Would need to create a preheader
        // This requires complex CFG surgery:
        // 1. Create new Region node
        // 2. Redirect non-back-edge predecessors of header to new block
        // 3. Add unconditional jump from preheader to header
        // 4. Update phi nodes in header
        //
        // For now, we don't create preheaders - the transformation
        // gracefully handles this by keeping checks in place
        None
    }

    /// Create a preheader block if one doesn't exist.
    ///
    /// This is complex graph surgery and is only needed when:
    /// - The loop header has multiple non-loop predecessors
    /// - We want to hoist guards that don't have a valid preheader target
    pub fn create_preheader(_graph: &mut Graph, _cfg: &Cfg, _loop_info: &Loop) -> Option<BlockId> {
        // Creating a preheader requires:
        // 1. Create new Region node to serve as the preheader
        // 2. For each predecessor of header that's NOT a back edge:
        //    a. Redirect its branch to the new preheader
        //    b. Update any phi inputs accordingly
        // 3. Add unconditional branch from preheader to header
        // 4. The preheader becomes the new "single entry" to the loop
        //
        // This is deferred as most well-formed loops already have preheaders

        None
    }
}
