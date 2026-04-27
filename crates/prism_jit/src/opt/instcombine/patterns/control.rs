//! Control flow patterns for instruction combining.
//!
//! Patterns include:
//! - Branch on constant condition -> unconditional

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, Operator};

use super::PatternMatch;

// =============================================================================
// Control Patterns
// =============================================================================

/// Control flow pattern matcher.
pub struct ControlPatterns;

impl ControlPatterns {
    /// Try to match a control flow pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::Control(ctrl_op) => Self::try_control(graph, node, *ctrl_op),
            _ => None,
        }
    }

    /// Try to match control operation patterns.
    fn try_control(graph: &Graph, node: NodeId, op: ControlOp) -> Option<PatternMatch> {
        match op {
            ControlOp::If => Self::try_if_patterns(graph, node),
            _ => None,
        }
    }

    /// Try if/branch patterns.
    fn try_if_patterns(graph: &Graph, node: NodeId) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get() since as_slice() only works for Many variant
        let condition = n.inputs.get(0)?;

        // Branch on constant condition
        if let Some(cond_val) = Self::get_bool_constant(graph, condition) {
            if cond_val {
                return Some(PatternMatch::new(node, "branch_true"));
            } else {
                return Some(PatternMatch::new(node, "branch_false"));
            }
        }

        None
    }

    /// Get boolean constant value if node is a constant.
    fn get_bool_constant(graph: &Graph, node: NodeId) -> Option<bool> {
        if let Some(n) = graph.get(node) {
            match n.op {
                Operator::ConstBool(v) => Some(v),
                Operator::ConstInt(v) => Some(v != 0),
                _ => None,
            }
        } else {
            None
        }
    }
}
