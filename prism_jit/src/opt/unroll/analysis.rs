//! Loop Trip Count and Unrollability Analysis.
//!
//! This module provides the analysis infrastructure for loop unrolling:
//!
//! - **Trip count analysis**: Determine loop iteration counts
//! - **Unrollability checks**: Verify loop structure requirements
//! - **Cost estimation**: Predict code growth and register pressure
//!
//! # Trip Count Categories
//!
//! - **Constant**: Known at compile time (e.g., `for i in range(10)`)
//! - **Symbolic**: Computable from parameters (e.g., `for i in range(n)`)
//! - **Runtime**: Requires dynamic check (e.g., data-dependent bounds)
//!
//! # Canonicality Requirements
//!
//! For unrolling, loops must have:
//! - Single entry (header)
//! - Single back edge source
//! - Identified induction variable
//! - Computable bounds

use crate::ir::cfg::{Cfg, DominatorTree, LoopAnalysis};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, CmpOp, MemoryOp, Operator};

use rustc_hash::FxHashSet;

// =============================================================================
// Trip Count
// =============================================================================

/// Loop trip count classification.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopTripCount {
    /// Exactly known trip count.
    Constant(i64),

    /// Trip count is a function of a parameter.
    Parameter {
        /// Parameter index.
        param: usize,
        /// Offset from parameter.
        offset: i64,
        /// Scale factor.
        scale: i64,
    },

    /// Trip count depends on symbolic value.
    Symbolic {
        /// Node producing the trip count.
        node: NodeId,
    },

    /// Bounded but not exact.
    Bounded {
        /// Lower bound (may be 0).
        min: u32,
        /// Upper bound.
        max: u32,
    },

    /// Unknown trip count.
    Unknown,
}

impl LoopTripCount {
    /// Get constant trip count if known.
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            LoopTripCount::Constant(n) => Some(*n),
            _ => None,
        }
    }

    /// Check if trip count is compile-time constant.
    pub fn is_constant(&self) -> bool {
        matches!(self, LoopTripCount::Constant(_))
    }

    /// Get upper bound estimate.
    pub fn upper_bound(&self) -> Option<u32> {
        match self {
            LoopTripCount::Constant(n) => Some(*n as u32),
            LoopTripCount::Bounded { max, .. } => Some(*max),
            LoopTripCount::Parameter { .. } => Some(1000), // Heuristic
            _ => None,
        }
    }

    /// Get lower bound estimate.
    pub fn lower_bound(&self) -> Option<u32> {
        match self {
            LoopTripCount::Constant(n) => Some(*n as u32),
            LoopTripCount::Bounded { min, .. } => Some(*min),
            _ => Some(0),
        }
    }
}

impl Default for LoopTripCount {
    fn default() -> Self {
        LoopTripCount::Unknown
    }
}

// =============================================================================
// Unrollability Analysis
// =============================================================================

/// Full unrollability analysis for a loop.
#[derive(Debug)]
pub struct UnrollabilityAnalysis {
    /// Loop index in LoopAnalysis.
    pub loop_idx: usize,

    /// Analyzed trip count.
    pub trip_count: LoopTripCount,

    /// Number of nodes in the loop body.
    pub body_size: usize,

    /// Whether the loop has a single entry point.
    pub has_single_entry: bool,

    /// Whether the loop has a single exit point.
    pub has_single_exit: bool,

    /// Whether the loop body contains function calls.
    pub contains_calls: bool,

    /// Whether the loop has memory side effects.
    pub has_memory_effects: bool,

    /// Whether the loop has early exits (break/continue).
    pub has_early_exits: bool,

    /// Nesting depth (0 = outermost).
    pub nesting_depth: u32,

    /// Identified induction variables.
    pub induction_vars: Vec<NodeId>,

    /// Estimated register pressure.
    pub register_pressure: usize,

    /// Whether the loop is in canonical form.
    pub is_canonical: bool,

    /// Nodes in the loop body.
    pub body_nodes: FxHashSet<NodeId>,
}

impl UnrollabilityAnalysis {
    /// Check if the loop can be fully unrolled with given constraints.
    pub fn can_fully_unroll(&self, max_trip: u32, max_size: usize) -> bool {
        if !self.is_canonical {
            return false;
        }

        match &self.trip_count {
            LoopTripCount::Constant(n) => {
                let n = *n as u32;
                n <= max_trip && self.body_size * n as usize <= max_size
            }
            _ => false,
        }
    }

    /// Check if partial unrolling is beneficial.
    pub fn can_partial_unroll(&self, min_trip: u32) -> bool {
        if !self.is_canonical || self.induction_vars.is_empty() {
            return false;
        }

        match &self.trip_count {
            LoopTripCount::Constant(n) => *n as u32 >= min_trip,
            LoopTripCount::Bounded { min, .. } => *min >= min_trip,
            LoopTripCount::Parameter { .. } => true,
            _ => false,
        }
    }
}

// =============================================================================
// Trip Count Analyzer
// =============================================================================

/// Analyzes loop trip counts.
#[allow(dead_code)]
pub struct TripCountAnalyzer<'a> {
    graph: &'a Graph,
    loops: &'a LoopAnalysis,
    cfg: &'a Cfg,
}

impl<'a> TripCountAnalyzer<'a> {
    /// Create a new trip count analyzer.
    pub fn new(graph: &'a Graph, loops: &'a LoopAnalysis, cfg: &'a Cfg) -> Self {
        Self { graph, loops, cfg }
    }

    /// Analyze the trip count of a loop.
    pub fn analyze(&self, loop_idx: usize) -> LoopTripCount {
        if loop_idx >= self.loops.loops.len() {
            return LoopTripCount::Unknown;
        }

        let loop_info = &self.loops.loops[loop_idx];

        // Find the loop condition
        if let Some(condition) = self.find_loop_condition(loop_info.header) {
            // Try to extract bounds from the condition
            if let Some(trip) = self.extract_trip_count(condition) {
                return trip;
            }
        }

        // If we have back edges but no clear condition, try induction analysis
        if !loop_info.back_edges.is_empty() {
            if let Some(trip) = self.analyze_induction_variable(loop_idx) {
                return trip;
            }
        }

        LoopTripCount::Unknown
    }

    /// Find the loop condition node.
    fn find_loop_condition(&self, header_block: crate::ir::cfg::BlockId) -> Option<NodeId> {
        // Get the header region node
        let header_region = self.cfg.block(header_block).region;

        // Look for a Branch node that uses the header's control
        for use_id in self.graph.uses(header_region).iter() {
            if let Some(node) = self.graph.get(*use_id) {
                if let Operator::Control(crate::ir::operators::ControlOp::If) = &node.op {
                    // The condition is the first data input
                    if let Some(cond_id) = node.inputs.get(1) {
                        return Some(cond_id);
                    }
                }
            }
        }

        None
    }

    /// Extract trip count from a comparison condition.
    fn extract_trip_count(&self, condition: NodeId) -> Option<LoopTripCount> {
        let cond_node = self.graph.get(condition)?;

        // Handle comparison operators (IntCmp or GenericCmp)
        let cmp_op = match &cond_node.op {
            Operator::IntCmp(op) => Some(*op),
            Operator::GenericCmp(op) => Some(*op),
            _ => None,
        }?;

        let left = cond_node.inputs.get(0)?;
        let right = cond_node.inputs.get(1)?;

        // Check for i < N pattern
        if matches!(cmp_op, CmpOp::Lt | CmpOp::Le) {
            // Right side is the limit
            if let Some(limit_node) = self.graph.get(right) {
                if let Operator::ConstInt(limit) = &limit_node.op {
                    // Check if left is an induction variable starting at 0
                    if let Some(start) = self.find_induction_start(left) {
                        let trip = match cmp_op {
                            CmpOp::Lt => *limit - start,
                            CmpOp::Le => *limit - start + 1,
                            _ => return None,
                        };
                        if trip > 0 {
                            return Some(LoopTripCount::Constant(trip));
                        }
                    }
                } else if let Operator::Parameter(idx) = &limit_node.op {
                    return Some(LoopTripCount::Parameter {
                        param: *idx as usize,
                        offset: 0,
                        scale: 1,
                    });
                }
            }
        }

        None
    }

    /// Find the starting value of an induction variable.
    fn find_induction_start(&self, iv_node: NodeId) -> Option<i64> {
        let node = self.graph.get(iv_node)?;

        // If it's a LoopPhi, the first input (after control) is the initial value
        if let Operator::LoopPhi = &node.op {
            let init = node.inputs.get(1)?;
            if let Some(init_node) = self.graph.get(init) {
                if let Operator::ConstInt(n) = &init_node.op {
                    return Some(*n);
                }
            }
        }

        None
    }

    /// Analyze loop using induction variable detection.
    fn analyze_induction_variable(&self, loop_idx: usize) -> Option<LoopTripCount> {
        let loop_info = &self.loops.loops[loop_idx];

        // Get the header region
        let header_region = self.cfg.block(loop_info.header).region;

        // Find LoopPhi nodes in the header (induction variables)
        for use_id in self.graph.uses(header_region).iter() {
            if let Some(node) = self.graph.get(*use_id) {
                if let Operator::LoopPhi = &node.op {
                    // Found an induction variable
                    // Try to determine its range
                    if let Some(init) = node.inputs.get(1) {
                        if let Some(init_node) = self.graph.get(init) {
                            if let Operator::ConstInt(start) = &init_node.op {
                                // Check for bounded iteration pattern
                                return Some(LoopTripCount::Bounded {
                                    min: 0,
                                    max: 1000 - *start as u32,
                                });
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

// =============================================================================
// Unrollability Analyzer
// =============================================================================

/// Analyzes loops for unrollability.
pub struct UnrollabilityAnalyzer<'a> {
    graph: &'a Graph,
    loops: &'a LoopAnalysis,
    cfg: &'a Cfg,
}

impl<'a> UnrollabilityAnalyzer<'a> {
    /// Create a new unrollability analyzer.
    pub fn new(graph: &'a Graph, loops: &'a LoopAnalysis, cfg: &'a Cfg) -> Self {
        Self { graph, loops, cfg }
    }

    /// Analyze a specific loop.
    pub fn analyze(&self, loop_idx: usize) -> Option<UnrollabilityAnalysis> {
        if loop_idx >= self.loops.loops.len() {
            return None;
        }

        let loop_info = &self.loops.loops[loop_idx];

        // Compute trip count
        let trip_analyzer = TripCountAnalyzer::new(self.graph, self.loops, self.cfg);
        let trip_count = trip_analyzer.analyze(loop_idx);

        // Collect body nodes
        let body_nodes = self.collect_body_nodes(loop_idx);
        let body_size = body_nodes.len();

        // Analyze loop properties
        let contains_calls = self.check_contains_calls(&body_nodes);
        let has_memory_effects = self.check_memory_effects(&body_nodes);
        let has_early_exits = self.check_early_exits(loop_idx);

        // Find induction variables
        let induction_vars = self.find_induction_variables(loop_idx);

        // Compute register pressure estimate
        let register_pressure = self.estimate_register_pressure(&body_nodes);

        // Check canonicality
        let is_canonical = self.is_canonical(loop_idx);

        Some(UnrollabilityAnalysis {
            loop_idx,
            trip_count,
            body_size,
            has_single_entry: true, // Natural loops always have single entry
            has_single_exit: loop_info.back_edges.len() == 1,
            contains_calls,
            has_memory_effects,
            has_early_exits,
            nesting_depth: loop_info.depth - 1, // Subtract 1 for 0-indexed depth
            induction_vars,
            register_pressure,
            is_canonical,
            body_nodes,
        })
    }

    /// Collect all nodes in the loop body.
    fn collect_body_nodes(&self, loop_idx: usize) -> FxHashSet<NodeId> {
        let loop_info = &self.loops.loops[loop_idx];
        let mut nodes = FxHashSet::default();

        // For each block in the loop body
        for &block in &loop_info.body {
            let region = self.cfg.block(block).region;

            // Add the region node
            nodes.insert(region);

            // Add all nodes that use this region's control
            for &use_id in self.graph.uses(region) {
                nodes.insert(use_id);
            }
        }

        nodes
    }

    /// Check if the loop body contains function calls.
    fn check_contains_calls(&self, body_nodes: &FxHashSet<NodeId>) -> bool {
        for &node_id in body_nodes {
            if let Some(node) = self.graph.get(node_id) {
                if matches!(&node.op, Operator::Call(_)) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the loop has memory side effects.
    fn check_memory_effects(&self, body_nodes: &FxHashSet<NodeId>) -> bool {
        for &node_id in body_nodes {
            if let Some(node) = self.graph.get(node_id) {
                // Check for store operations
                if let Operator::Memory(mem_op) = &node.op {
                    if matches!(mem_op, MemoryOp::StoreField | MemoryOp::StoreElement) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check if the loop has early exits.
    fn check_early_exits(&self, loop_idx: usize) -> bool {
        let loop_info = &self.loops.loops[loop_idx];

        // An early exit is an edge from inside the loop to outside
        for &block in &loop_info.body {
            let bb = self.cfg.block(block);
            for &succ in &bb.successors {
                if !loop_info.body.contains(&succ) && succ != loop_info.header {
                    return true;
                }
            }
        }

        false
    }

    /// Find induction variables in the loop.
    fn find_induction_variables(&self, loop_idx: usize) -> Vec<NodeId> {
        let loop_info = &self.loops.loops[loop_idx];
        let mut ivs = Vec::new();

        // Get the header region
        let header_region = self.cfg.block(loop_info.header).region;

        // Find LoopPhi nodes
        for &use_id in self.graph.uses(header_region) {
            if let Some(node) = self.graph.get(use_id) {
                if let Operator::LoopPhi = &node.op {
                    // Check if this is a linear induction variable
                    if self.is_linear_iv(use_id, loop_idx) {
                        ivs.push(use_id);
                    }
                }
            }
        }

        ivs
    }

    /// Check if a phi is a linear induction variable.
    fn is_linear_iv(&self, phi: NodeId, _loop_idx: usize) -> bool {
        // The second input (recurrence) should be phi + constant or phi * constant + constant
        let phi_node = match self.graph.get(phi) {
            Some(n) => n,
            None => return false,
        };

        if let Some(recurrence) = phi_node.inputs.get(2) {
            if let Some(rec_node) = self.graph.get(recurrence) {
                // Check for phi + constant pattern
                if let Operator::IntOp(ArithOp::Add) = &rec_node.op {
                    if let (Some(lhs), Some(rhs)) = (rec_node.inputs.get(0), rec_node.inputs.get(1))
                    {
                        // One operand should be the phi, other should be constant
                        if (lhs == phi && self.is_constant(rhs))
                            || (rhs == phi && self.is_constant(lhs))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Check if a node produces a constant value.
    fn is_constant(&self, node_id: NodeId) -> bool {
        if let Some(node) = self.graph.get(node_id) {
            matches!(&node.op, Operator::ConstInt(_) | Operator::ConstFloat(_))
        } else {
            false
        }
    }

    /// Estimate register pressure from loop body.
    fn estimate_register_pressure(&self, body_nodes: &FxHashSet<NodeId>) -> usize {
        // Simple heuristic: count nodes that produce values
        let mut live_values = 0;

        for &node_id in body_nodes {
            if let Some(node) = self.graph.get(node_id) {
                // Non-control nodes produce values
                if !matches!(&node.op, Operator::Control(_)) {
                    live_values += 1;
                }
            }
        }

        // Approximate: assume 1/3 of values are live at any point
        live_values / 3 + 1
    }

    /// Check if loop is in canonical form.
    fn is_canonical(&self, loop_idx: usize) -> bool {
        let loop_info = &self.loops.loops[loop_idx];

        // Must have exactly one back edge
        if loop_info.back_edges.len() != 1 {
            return false;
        }

        // Header must have a preheader (single non-back-edge predecessor)
        let header_block = self.cfg.block(loop_info.header);
        let non_back_preds: Vec<_> = header_block
            .predecessors
            .iter()
            .filter(|&&pred| !loop_info.back_edges.contains(&pred))
            .collect();

        if non_back_preds.len() != 1 {
            return false;
        }

        // Must have at least one induction variable
        !self.find_induction_variables(loop_idx).is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::DominatorTree;

    // =========================================================================
    // LoopTripCount Tests
    // =========================================================================

    #[test]
    fn test_trip_count_constant() {
        let tc = LoopTripCount::Constant(10);
        assert!(tc.is_constant());
        assert_eq!(tc.as_constant(), Some(10));
        assert_eq!(tc.upper_bound(), Some(10));
        assert_eq!(tc.lower_bound(), Some(10));
    }

    #[test]
    fn test_trip_count_bounded() {
        let tc = LoopTripCount::Bounded { min: 5, max: 100 };
        assert!(!tc.is_constant());
        assert_eq!(tc.as_constant(), None);
        assert_eq!(tc.upper_bound(), Some(100));
        assert_eq!(tc.lower_bound(), Some(5));
    }

    #[test]
    fn test_trip_count_parameter() {
        let tc = LoopTripCount::Parameter {
            param: 0,
            offset: 0,
            scale: 1,
        };
        assert!(!tc.is_constant());
        assert_eq!(tc.upper_bound(), Some(1000)); // Heuristic
    }

    #[test]
    fn test_trip_count_unknown() {
        let tc = LoopTripCount::Unknown;
        assert!(!tc.is_constant());
        assert_eq!(tc.as_constant(), None);
        assert!(tc.upper_bound().is_none());
        assert_eq!(tc.lower_bound(), Some(0));
    }

    #[test]
    fn test_trip_count_default() {
        let tc = LoopTripCount::default();
        assert_eq!(tc, LoopTripCount::Unknown);
    }

    // =========================================================================
    // UnrollabilityAnalysis Tests
    // =========================================================================

    fn make_canonical_analysis() -> UnrollabilityAnalysis {
        UnrollabilityAnalysis {
            loop_idx: 0,
            trip_count: LoopTripCount::Constant(4),
            body_size: 10,
            has_single_entry: true,
            has_single_exit: true,
            contains_calls: false,
            has_memory_effects: false,
            has_early_exits: false,
            nesting_depth: 0,
            induction_vars: vec![NodeId::new(5)],
            register_pressure: 4,
            is_canonical: true,
            body_nodes: FxHashSet::default(),
        }
    }

    #[test]
    fn test_can_fully_unroll_yes() {
        let analysis = make_canonical_analysis();
        assert!(analysis.can_fully_unroll(16, 100));
    }

    #[test]
    fn test_can_fully_unroll_trip_too_large() {
        let analysis = make_canonical_analysis();
        assert!(!analysis.can_fully_unroll(2, 100)); // max_trip = 2, trip = 4
    }

    #[test]
    fn test_can_fully_unroll_size_too_large() {
        let analysis = make_canonical_analysis();
        assert!(!analysis.can_fully_unroll(16, 20)); // 4 * 10 = 40 > 20
    }

    #[test]
    fn test_can_fully_unroll_not_canonical() {
        let mut analysis = make_canonical_analysis();
        analysis.is_canonical = false;
        assert!(!analysis.can_fully_unroll(16, 100));
    }

    #[test]
    fn test_can_partial_unroll_yes() {
        let analysis = make_canonical_analysis();
        assert!(analysis.can_partial_unroll(2));
    }

    #[test]
    fn test_can_partial_unroll_trip_too_small() {
        let analysis = make_canonical_analysis();
        assert!(!analysis.can_partial_unroll(10)); // trip = 4 < 10
    }

    #[test]
    fn test_can_partial_unroll_no_iv() {
        let mut analysis = make_canonical_analysis();
        analysis.induction_vars.clear();
        assert!(!analysis.can_partial_unroll(2));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_analyzer_empty_loops() {
        use crate::ir::builder::GraphBuilder;

        let builder = GraphBuilder::new(4, 0);
        let graph = builder.finish();

        let cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        let analyzer = UnrollabilityAnalyzer::new(&graph, &loops, &cfg);
        assert!(analyzer.analyze(0).is_none()); // No loops
    }

    #[test]
    fn test_trip_count_analyzer_no_loops() {
        use crate::ir::builder::GraphBuilder;

        let builder = GraphBuilder::new(4, 0);
        let graph = builder.finish();

        let cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        let analyzer = TripCountAnalyzer::new(&graph, &loops, &cfg);
        assert_eq!(analyzer.analyze(0), LoopTripCount::Unknown);
    }
}
