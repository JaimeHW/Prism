//! Comparison patterns for instruction combining.
//!
//! Patterns include:
//! - x == x -> true
//! - x != x -> false
//! - x < x -> false
//! - x <= x -> true
//! - x > x -> false
//! - x >= x -> true

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CmpOp, Operator};

use super::PatternMatch;

// =============================================================================
// Comparison Patterns
// =============================================================================

/// Comparison pattern matcher.
pub struct ComparisonPatterns;

impl ComparisonPatterns {
    /// Try to match a comparison pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::IntCmp(cmp_op)
            | Operator::FloatCmp(cmp_op)
            | Operator::GenericCmp(cmp_op) => Self::try_cmp(graph, node, *cmp_op),
            _ => None,
        }
    }

    /// Try to match comparison patterns.
    fn try_cmp(graph: &Graph, node: NodeId, op: CmpOp) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get() since as_slice() only works for Many variant
        let lhs = n.inputs.get(0)?;
        let rhs = n.inputs.get(1)?;

        // Check for comparing with self
        if lhs == rhs {
            return Self::try_self_comparison(node, op);
        }

        // Check for comparison with constants
        Self::try_constant_comparison(graph, node, lhs, rhs, op)
    }

    /// Handle x cmp x cases.
    fn try_self_comparison(node: NodeId, op: CmpOp) -> Option<PatternMatch> {
        match op {
            CmpOp::Eq => Some(PatternMatch::new(node, "eq_self_true")),
            CmpOp::Ne => Some(PatternMatch::new(node, "ne_self_false")),
            CmpOp::Lt | CmpOp::Gt => Some(PatternMatch::new(node, "lt_gt_self_false")),
            CmpOp::Le | CmpOp::Ge => Some(PatternMatch::new(node, "le_ge_self_true")),
            _ => None,
        }
    }

    /// Try constant comparison patterns.
    fn try_constant_comparison(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        _op: CmpOp,
    ) -> Option<PatternMatch> {
        let lhs_const = Self::get_int_constant(graph, lhs);
        let rhs_const = Self::get_int_constant(graph, rhs);

        if let (Some(_l), Some(_r)) = (lhs_const, rhs_const) {
            return Some(PatternMatch::new(node, "const_fold"));
        }

        None
    }

    /// Get integer constant value if node is a constant.
    fn get_int_constant(graph: &Graph, node: NodeId) -> Option<i64> {
        if let Some(n) = graph.get(node) {
            if let Operator::ConstInt(v) = n.op {
                return Some(v);
            }
        }
        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
