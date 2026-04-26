//! Memory patterns for instruction combining.
//!
//! Patterns include:
//! - Load after store to same location -> use stored value
//! - Store after store to same location -> eliminate first store
//! - Load of constant -> fold to constant
//!
//! Note: These patterns require sophisticated alias analysis for full
//! implementation. This module provides the infrastructure for such
//! patterns but the actual matching is conservative.

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};

use super::PatternMatch;

// =============================================================================
// Memory Patterns
// =============================================================================

/// Memory pattern matcher.
pub struct MemoryPatterns;

impl MemoryPatterns {
    /// Try to match a memory pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::Memory(mem_op) => Self::try_memory(graph, node, *mem_op),
            _ => None,
        }
    }

    /// Try to match memory operation patterns.
    fn try_memory(_graph: &Graph, _node: NodeId, _op: MemoryOp) -> Option<PatternMatch> {
        // Memory patterns require alias analysis which is done
        // separately in the DSE pass. InstCombine handles simpler
        // algebraic patterns; memory optimization is deferred to DSE.
        None
    }

    /// Check if a node is a fresh allocation.
    #[allow(dead_code)]
    fn is_fresh_alloc(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            matches!(
                n.op,
                Operator::Memory(MemoryOp::Alloc) | Operator::Memory(MemoryOp::AllocArray)
            )
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
