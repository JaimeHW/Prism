//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP is a powerful dataflow optimization that combines:
//! - **Constant Propagation**: Track and fold compile-time constants
//! - **Unreachable Code Elimination**: Detect and remove dead branches
//!
//! # Algorithm
//!
//! Uses a sparse, SSA-based approach:
//! 1. Initialize lattice values (Undef for all, except constants)
//! 2. Process worklist until fixed point
//! 3. Transform graph: replace constants, remove unreachable code
//!
//! # Lattice Structure
//!
//! ```text
//!       ⊤ (Overdefined) - Value varies at runtime
//!           |
//!    Constant(v) - Value is known constant v
//!           |
//!       ⊥ (Undef) - Value undefined
//! ```
//!
//! # Performance
//!
//! - Time: O(n × h) where n = nodes, h = lattice height (3)
//! - Space: O(n) for value storage
//!
//! # Example
//!
//! ```text
//! // Before SCCP:
//! x = 10
//! y = 20
//! z = x + y  // z = 30 (folded)
//! if true:
//!     return z
//! else:
//!     return 0  // unreachable
//!
//! // After SCCP:
//! return 30
//! ```

pub mod evaluation;
pub mod lattice;
pub mod solver;

pub use evaluation::ConstEvaluator;
pub use lattice::{Constant, LatticeValue};
pub use solver::{SccpSolver, SolverResult, SolverStats};

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeFlags, NodeId};
use crate::ir::operators::Operator;
use crate::opt::OptimizationPass;

// =============================================================================
// SCCP Configuration
// =============================================================================

/// Configuration for the SCCP pass.
#[derive(Debug, Clone)]
pub struct SccpConfig {
    /// Enable aggressive dead code elimination.
    pub eliminate_dead_code: bool,
    /// Enable constant folding in expressions.
    pub fold_constants: bool,
    /// Maximum iterations for fixed-point (safety limit).
    pub max_iterations: usize,
}

impl Default for SccpConfig {
    fn default() -> Self {
        Self {
            eliminate_dead_code: true,
            fold_constants: true,
            max_iterations: 1000,
        }
    }
}

impl SccpConfig {
    /// Create aggressive configuration.
    pub fn aggressive() -> Self {
        Self {
            eliminate_dead_code: true,
            fold_constants: true,
            max_iterations: 10000,
        }
    }

    /// Create conservative configuration.
    pub fn conservative() -> Self {
        Self {
            eliminate_dead_code: false,
            fold_constants: true,
            max_iterations: 500,
        }
    }
}

// =============================================================================
// SCCP Pass
// =============================================================================

/// Sparse Conditional Constant Propagation optimization pass.
#[derive(Debug)]
pub struct Sccp {
    /// Configuration.
    config: SccpConfig,
    /// Statistics from last run.
    stats: SccpStats,
}

/// Statistics from SCCP.
#[derive(Debug, Clone, Default)]
pub struct SccpStats {
    /// Number of constants folded.
    pub constants_folded: usize,
    /// Number of dead nodes removed.
    pub dead_nodes_removed: usize,
    /// Number of branches simplified.
    pub branches_simplified: usize,
    /// Number of phis removed.
    pub phis_removed: usize,
}

impl Sccp {
    /// Create a new SCCP pass with default configuration.
    pub fn new() -> Self {
        Self {
            config: SccpConfig::default(),
            stats: SccpStats::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: SccpConfig) -> Self {
        Self {
            config,
            stats: SccpStats::default(),
        }
    }

    /// Get statistics from the last run.
    pub fn stats(&self) -> &SccpStats {
        &self.stats
    }

    /// Run SCCP on the graph.
    pub fn run_sccp(&mut self, graph: &mut Graph) -> bool {
        self.stats = SccpStats::default();

        // Phase 1: Analysis - compute lattice values
        let solver = SccpSolver::new(graph);
        let result = solver.solve();

        // Phase 2: Transform - apply results to graph
        let mut changed = false;

        if self.config.fold_constants {
            changed |= self.fold_constants(graph, &result);
        }

        if self.config.eliminate_dead_code {
            changed |= self.eliminate_unreachable(graph, &result);
        }

        changed
    }

    /// Fold constant expressions.
    fn fold_constants(&mut self, graph: &mut Graph, result: &SolverResult) -> bool {
        let mut changed = false;
        let mut replacements: Vec<(NodeId, Operator)> = Vec::new();

        // Find nodes to replace
        for (node_id, constant) in result.constants() {
            // Skip nodes that are already constants
            if let Some(node) = graph.get(node_id) {
                if Self::is_constant_op(&node.op) {
                    continue;
                }

                // Convert lattice constant to IR operator
                if let Some(ir_op) = self.to_ir_operator(constant) {
                    replacements.push((node_id, ir_op));
                }
            }
        }

        // Apply replacements
        for (node_id, ir_op) in replacements {
            if let Some(node) = graph.get_mut(node_id) {
                node.op = ir_op;
                node.inputs = InputList::Empty;
                self.stats.constants_folded += 1;
                changed = true;
            }
        }

        changed
    }

    /// Check if an operator is a constant.
    fn is_constant_op(op: &Operator) -> bool {
        matches!(
            op,
            Operator::ConstInt(_)
                | Operator::ConstFloat(_)
                | Operator::ConstBool(_)
                | Operator::ConstNone
        )
    }

    /// Eliminate unreachable code.
    fn eliminate_unreachable(&mut self, graph: &mut Graph, result: &SolverResult) -> bool {
        let mut changed = false;
        let mut to_remove: Vec<NodeId> = Vec::new();

        // Find unreachable nodes
        for (node_id, node) in graph.iter() {
            if !result.is_reachable(node_id) && !node.flags.contains(NodeFlags::PINNED) {
                // Don't remove Start or End nodes
                if !matches!(
                    node.op,
                    Operator::Control(crate::ir::operators::ControlOp::Start)
                        | Operator::Control(crate::ir::operators::ControlOp::End)
                ) {
                    to_remove.push(node_id);
                }
            }
        }

        // Mark nodes as dead (actual removal is DCE's job)
        for node_id in to_remove {
            if let Some(node) = graph.get_mut(node_id) {
                node.flags |= NodeFlags::DEAD;
                self.stats.dead_nodes_removed += 1;
                changed = true;
            }
        }

        changed
    }

    /// Convert a lattice constant to an IR operator.
    fn to_ir_operator(&self, constant: &Constant) -> Option<Operator> {
        match constant {
            Constant::Int(v) => Some(Operator::ConstInt(*v)),
            Constant::Float(v) => Some(Operator::ConstFloat(v.to_bits())),
            Constant::Bool(v) => Some(Operator::ConstBool(*v)),
            Constant::None => Some(Operator::ConstNone),
            // Other types are not directly representable
            _ => None,
        }
    }
}

impl Default for Sccp {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Sccp {
    fn name(&self) -> &'static str {
        "sccp"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_sccp(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = SccpConfig::default();
        assert!(config.eliminate_dead_code);
        assert!(config.fold_constants);
    }

    #[test]
    fn test_config_aggressive() {
        let config = SccpConfig::aggressive();
        assert!(config.eliminate_dead_code);
        assert!(config.max_iterations > 1000);
    }

    #[test]
    fn test_config_conservative() {
        let config = SccpConfig::conservative();
        assert!(!config.eliminate_dead_code);
        assert!(config.fold_constants);
    }

    // =========================================================================
    // Basic SCCP Tests
    // =========================================================================

    #[test]
    fn test_sccp_empty_graph() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let mut sccp = Sccp::new();
        let changed = sccp.run(&mut graph);

        assert!(!changed);
    }

    #[test]
    fn test_sccp_constant_folding() {
        let mut builder = GraphBuilder::new(8, 0);
        // x = 10 + 20
        let c10 = builder.const_int(10);
        let c20 = builder.const_int(20);
        let sum = builder.int_add(c10, c20);
        builder.return_value(sum);

        let mut graph = builder.finish();

        // SCCP should fold 10 + 20 = 30
        let mut sccp = Sccp::new();
        let changed = sccp.run(&mut graph);

        assert!(changed);
        assert!(sccp.stats().constants_folded > 0);

        // The sum node should now be a constant
        if let Some(node) = graph.get(sum) {
            match &node.op {
                Operator::ConstInt(30) => (),
                other => panic!("Expected ConstInt(30), got {:?}", other),
            }
        }
    }

    #[test]
    fn test_sccp_chained_constants() {
        let mut builder = GraphBuilder::new(12, 0);
        // a = 2 * 3 = 6
        // b = a + 4 = 10
        // c = b * 2 = 20
        let c2 = builder.const_int(2);
        let c3 = builder.const_int(3);
        let a = builder.int_mul(c2, c3);
        let c4 = builder.const_int(4);
        let b = builder.int_add(a, c4);
        let c2_2 = builder.const_int(2);
        let c = builder.int_mul(b, c2_2);
        builder.return_value(c);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        // All intermediate computations should be folded
        if let Some(node) = graph.get(c) {
            assert!(matches!(&node.op, Operator::ConstInt(20)));
        }
    }

    #[test]
    fn test_sccp_preserves_parameters() {
        let mut builder = GraphBuilder::new(8, 1);
        // result = param0 + 10
        let p0 = builder.parameter(0).unwrap();
        let c10 = builder.const_int(10);
        let sum = builder.int_add(p0, c10);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        // Sum cannot be folded (parameter is unknown)
        if let Some(node) = graph.get(sum) {
            assert!(!Sccp::is_constant_op(&node.op));
        }
    }

    #[test]
    fn test_sccp_comparison_folding() {
        let mut builder = GraphBuilder::new(8, 0);
        // x = 5 < 10 (= true)
        let c5 = builder.const_int(5);
        let c10 = builder.const_int(10);
        let cmp = builder.int_lt(c5, c10);
        builder.return_value(cmp);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        // Comparison should be folded to true
        if let Some(node) = graph.get(cmp) {
            assert!(matches!(&node.op, Operator::ConstBool(true)));
        }
    }

    #[test]
    fn test_sccp_subtraction_folding() {
        let mut builder = GraphBuilder::new(8, 0);
        let c100 = builder.const_int(100);
        let c42 = builder.const_int(42);
        let diff = builder.int_sub(c100, c42);
        builder.return_value(diff);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(diff) {
            assert!(matches!(&node.op, Operator::ConstInt(58)));
        }
    }

    #[test]
    fn test_sccp_division_folding() {
        let mut builder = GraphBuilder::new(8, 0);
        let c100 = builder.const_int(100);
        let c5 = builder.const_int(5);
        let quot = builder.int_div(c100, c5);
        builder.return_value(quot);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(quot) {
            assert!(matches!(&node.op, Operator::ConstInt(20)));
        }
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    #[test]
    fn test_sccp_stats() {
        let mut builder = GraphBuilder::new(8, 0);
        let c1 = builder.const_int(1);
        let c2 = builder.const_int(2);
        let c3 = builder.const_int(3);
        let sum1 = builder.int_add(c1, c2);
        let sum2 = builder.int_add(sum1, c3);
        builder.return_value(sum2);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        let stats = sccp.stats();
        assert!(stats.constants_folded >= 2);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_sccp_zero_operations() {
        let mut builder = GraphBuilder::new(8, 0);
        // x * 0 = 0
        let c42 = builder.const_int(42);
        let c0 = builder.const_int(0);
        let mul = builder.int_mul(c42, c0);
        builder.return_value(mul);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(mul) {
            assert!(matches!(&node.op, Operator::ConstInt(0)));
        }
    }

    #[test]
    fn test_sccp_identity_operations() {
        let mut builder = GraphBuilder::new(8, 0);
        // x + 0 = x
        let c42 = builder.const_int(42);
        let c0 = builder.const_int(0);
        let sum = builder.int_add(c42, c0);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(sum) {
            assert!(matches!(&node.op, Operator::ConstInt(42)));
        }
    }

    #[test]
    fn test_sccp_negative_numbers() {
        let mut builder = GraphBuilder::new(8, 0);
        let c_neg5 = builder.const_int(-5);
        let c_neg3 = builder.const_int(-3);
        let sum = builder.int_add(c_neg5, c_neg3);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(sum) {
            assert!(matches!(&node.op, Operator::ConstInt(-8)));
        }
    }

    // =========================================================================
    // Pass Interface Tests
    // =========================================================================

    #[test]
    fn test_sccp_pass_name() {
        let sccp = Sccp::new();
        assert_eq!(sccp.name(), "sccp");
    }

    #[test]
    fn test_sccp_default() {
        let sccp = Sccp::default();
        assert_eq!(sccp.name(), "sccp");
    }

    #[test]
    fn test_sccp_with_config() {
        let config = SccpConfig::aggressive();
        let sccp = Sccp::with_config(config);
        assert_eq!(sccp.name(), "sccp");
    }

    // =========================================================================
    // Float Tests
    // =========================================================================

    #[test]
    fn test_sccp_float_arithmetic() {
        let mut builder = GraphBuilder::new(8, 0);
        let f1 = builder.const_float(2.5);
        let f2 = builder.const_float(3.5);
        let sum = builder.float_add(f1, f2);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(sum) {
            match &node.op {
                Operator::ConstFloat(bits) => {
                    let v = f64::from_bits(*bits);
                    assert!((v - 6.0).abs() < f64::EPSILON);
                }
                other => panic!("Expected ConstFloat, got {:?}", other),
            }
        }
    }

    // =========================================================================
    // Boolean Tests
    // =========================================================================

    #[test]
    fn test_sccp_equality_comparison() {
        let mut builder = GraphBuilder::new(8, 0);
        let c5 = builder.const_int(5);
        let c5_2 = builder.const_int(5);
        let eq = builder.int_eq(c5, c5_2);
        builder.return_value(eq);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(eq) {
            assert!(matches!(&node.op, Operator::ConstBool(true)));
        }
    }

    #[test]
    fn test_sccp_inequality_comparison() {
        let mut builder = GraphBuilder::new(8, 0);
        let c5 = builder.const_int(5);
        let c10 = builder.const_int(10);
        let ne = builder.int_ne(c5, c10);
        builder.return_value(ne);

        let mut graph = builder.finish();
        let mut sccp = Sccp::new();
        sccp.run(&mut graph);

        if let Some(node) = graph.get(ne) {
            assert!(matches!(&node.op, Operator::ConstBool(true)));
        }
    }
}
