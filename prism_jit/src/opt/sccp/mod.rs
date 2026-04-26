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
mod tests;
