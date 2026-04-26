//! Arithmetic patterns for instruction combining.
//!
//! Patterns include:
//! - x + 0 -> x
//! - x - 0 -> x
//! - x * 0 -> 0
//! - x * 1 -> x
//! - x - x -> 0
//! - x / 1 -> x
//! - 0 / x -> 0 (where x != 0)

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, Operator};

use super::PatternMatch;

// =============================================================================
// Arithmetic Patterns
// =============================================================================

/// Arithmetic pattern matcher.
pub struct ArithmeticPatterns;

impl ArithmeticPatterns {
    /// Try to match an arithmetic pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::IntOp(arith_op)
            | Operator::FloatOp(arith_op)
            | Operator::GenericOp(arith_op) => Self::try_arith_op(graph, node, *arith_op),
            _ => None,
        }
    }

    /// Try to match arithmetic operation patterns.
    fn try_arith_op(graph: &Graph, node: NodeId, op: ArithOp) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get(0) and get(1) since as_slice() only works for Many variant
        let lhs = n.inputs.get(0)?;
        let rhs = n.inputs.get(1)?;

        match op {
            ArithOp::Add => Self::try_add_patterns(graph, node, lhs, rhs),
            ArithOp::Sub => Self::try_sub_patterns(graph, node, lhs, rhs),
            ArithOp::Mul => Self::try_mul_patterns(graph, node, lhs, rhs),
            ArithOp::TrueDiv | ArithOp::FloorDiv => Self::try_div_patterns(graph, node, lhs, rhs),
            _ => None,
        }
    }

    /// Add patterns: x + 0 -> x, 0 + x -> x
    fn try_add_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "add_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "add_zero_left"));
        }
        None
    }

    /// Sub patterns: x - 0 -> x
    fn try_sub_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "sub_zero"));
        }
        None
    }

    /// Mul patterns: x * 0 -> 0, x * 1 -> x, 0 * x -> 0, 1 * x -> x
    fn try_mul_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, rhs, "mul_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, lhs, "mul_zero_left"));
        }
        if Self::is_one(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "mul_one_right"));
        }
        if Self::is_one(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "mul_one_left"));
        }
        None
    }

    /// Div patterns: x / 1 -> x, 0 / x -> 0
    fn try_div_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_one(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "div_one"));
        }
        if Self::is_zero(graph, lhs) && !Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "zero_div"));
        }
        None
    }

    /// Check if a node is a zero constant.
    fn is_zero(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            match n.op {
                Operator::ConstInt(v) => v == 0,
                Operator::ConstFloat(bits) => f64::from_bits(bits) == 0.0,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a node is a one constant.
    fn is_one(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            match n.op {
                Operator::ConstInt(v) => v == 1,
                Operator::ConstFloat(bits) => f64::from_bits(bits) == 1.0,
                _ => false,
            }
        } else {
            false
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
