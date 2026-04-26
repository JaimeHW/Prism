//! Anticipation Analysis for PRE.
//!
//! An expression is **anticipated** at a point if it will definitely
//! be computed on every path from that point to the program exit.
//!
//! Uses backward data flow analysis.

use rustc_hash::FxHashSet;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, Operator};

use super::{ExprId, ExpressionTable};

// =============================================================================
// Anticipation Analysis
// =============================================================================

/// Result of anticipation analysis.
#[derive(Debug)]
pub struct AnticipationAnalysis {
    /// Expressions anticipated at each node.
    antic_at: Vec<FxHashSet<ExprId>>,
    /// Number of nodes analyzed.
    #[allow(dead_code)]
    node_count: usize,
    /// Empty set for out-of-bounds access.
    empty: FxHashSet<ExprId>,
}

impl AnticipationAnalysis {
    /// Compute anticipation for the graph.
    pub fn compute(graph: &Graph, expr_table: &ExpressionTable) -> Self {
        let node_count = graph.len();
        let mut analysis = Self {
            antic_at: vec![FxHashSet::default(); node_count],
            node_count,
            empty: FxHashSet::default(),
        };

        analysis.analyze(graph, expr_table);
        analysis
    }

    /// Run backward dataflow analysis.
    fn analyze(&mut self, graph: &Graph, expr_table: &ExpressionTable) {
        // Initialize: expressions are anticipated where they're defined
        for i in 0..graph.len() {
            let node_id = NodeId::new(i as u32);
            if let Some(expr_id) = expr_table.get_expr_id(node_id) {
                self.antic_at[i].insert(expr_id);
            }
        }

        // Backward propagation until fixpoint
        let mut changed = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iteration < MAX_ITERATIONS {
            changed = false;
            iteration += 1;

            // Process nodes in reverse order (rough backward traversal)
            for i in (0..graph.len()).rev() {
                let node_id = NodeId::new(i as u32);

                // Get uses (successors in data flow sense)
                let uses = graph.uses(node_id);
                if uses.is_empty() {
                    continue;
                }

                // Anticipation propagates backward:
                // If an expression is anticipated at ALL successors, it's anticipated here
                let old_size = self.antic_at[i].len();

                // Get intersection of successor anticipation sets
                let mut first = true;
                let mut intersection = FxHashSet::default();

                for &succ_id in uses {
                    let succ_idx = succ_id.index() as usize;
                    if succ_idx < self.antic_at.len() {
                        if first {
                            intersection = self.antic_at[succ_idx].clone();
                            first = false;
                        } else {
                            intersection.retain(|e| self.antic_at[succ_idx].contains(e));
                        }
                    }
                }

                // Check for control flow that blocks anticipation
                if let Some(node) = graph.get(node_id) {
                    if Self::blocks_anticipation(&node.op) {
                        // Control flow nodes don't propagate anticipation
                        intersection.clear();
                    }
                }

                // Merge intersection into current set
                for expr_id in intersection {
                    if self.antic_at[i].insert(expr_id) {
                        changed = true;
                    }
                }

                if self.antic_at[i].len() != old_size {
                    changed = true;
                }
            }
        }
    }

    /// Check if an operator blocks anticipation propagation.
    fn blocks_anticipation(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Control(ControlOp::If)
                | Operator::Control(ControlOp::Throw)
                | Operator::Control(ControlOp::Deopt)
        )
    }

    /// Check if an expression is anticipated at a node.
    pub fn is_anticipated(&self, node: NodeId, expr: ExprId) -> bool {
        let idx = node.index() as usize;
        if idx < self.antic_at.len() {
            self.antic_at[idx].contains(&expr)
        } else {
            false
        }
    }

    /// Get all anticipated expressions at a node.
    pub fn anticipated_at(&self, node: NodeId) -> &FxHashSet<ExprId> {
        let idx = node.index() as usize;
        if idx < self.antic_at.len() {
            &self.antic_at[idx]
        } else {
            &self.empty
        }
    }

    /// Get the number of nodes with anticipated expressions.
    pub fn nodes_with_anticipation(&self) -> usize {
        self.antic_at.iter().filter(|s| !s.is_empty()).count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
