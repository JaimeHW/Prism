//! Dead Code Elimination (DCE) optimization pass.
//!
//! DCE removes nodes that are not used by any live computation.
//! A node is live if:
//!
//! 1. It has side effects (stores, calls, control flow)
//! 2. It is used by another live node
//!
//! # Algorithm
//!
//! 1. Mark all nodes with side effects as live
//! 2. Propagate liveness backwards through use-def chains
//! 3. Remove all nodes that are not marked as live

use super::OptimizationPass;
use crate::ir::arena::BitSet;
use crate::ir::graph::Graph;
use crate::ir::operators::{ControlOp, Operator};

// =============================================================================
// DCE Pass
// =============================================================================

/// Dead Code Elimination pass.
pub struct Dce {
    /// Number of nodes removed.
    removed: usize,
}

impl Dce {
    /// Create a new DCE pass.
    pub fn new() -> Self {
        Dce { removed: 0 }
    }

    /// Get the number of removed nodes.
    pub fn removed(&self) -> usize {
        self.removed
    }

    /// Check if an operator has side effects (cannot be removed).
    fn has_side_effects(op: &Operator) -> bool {
        match op {
            // Control flow is always live
            Operator::Control(ControlOp::Return)
            | Operator::Control(ControlOp::Throw)
            | Operator::Control(ControlOp::If)
            | Operator::Control(ControlOp::Loop)
            | Operator::Control(ControlOp::Region)
            | Operator::Control(ControlOp::Start)
            | Operator::Control(ControlOp::End)
            | Operator::Control(ControlOp::Deopt) => true,

            // Calls have side effects
            Operator::Call(_) => true,

            // Stores have side effects
            Operator::SetItem | Operator::SetAttr => true,

            // Memory operations that mutate
            Operator::Memory(_) => true,

            // Vector memory operations have side effects
            Operator::VectorMemory(..) => true,

            // Guards must be preserved
            Operator::Guard(_) => true,

            // Container construction can have side effects
            Operator::BuildList(_) | Operator::BuildTuple(_) | Operator::BuildDict(_) => true,

            // Iteration has side effects
            Operator::GetIter | Operator::IterNext => true,

            // Pure operations (can be removed if unused)
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::Parameter(_)
            | Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::GenericOp(_)
            | Operator::MulHigh
            | Operator::MulHighSigned
            | Operator::IntCmp(_)
            | Operator::FloatCmp(_)
            | Operator::GenericCmp(_)
            | Operator::Bitwise(_)
            | Operator::LogicalNot
            | Operator::Phi
            | Operator::LoopPhi
            | Operator::GetItem
            | Operator::GetAttr
            | Operator::Len
            | Operator::TypeCheck
            | Operator::Box
            | Operator::Unbox
            | Operator::Projection(_) => false,

            // Pure vector operations (can be removed if unused)
            Operator::VectorArith(..)
            | Operator::VectorFma(_)
            | Operator::VectorBroadcast(_)
            | Operator::VectorExtract(..)
            | Operator::VectorInsert(..)
            | Operator::VectorShuffle(..)
            | Operator::VectorHadd(_)
            | Operator::VectorCmp(..)
            | Operator::VectorBlend(_)
            | Operator::VectorSplat(..) => false,
        }
    }
}

impl Default for Dce {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Dce {
    fn name(&self) -> &'static str {
        "DCE"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.removed = 0;

        let node_count = graph.len();
        let mut live = BitSet::with_capacity(node_count);
        let mut worklist = Vec::new();

        // Phase 1: Mark nodes with side effects as live
        for (id, node) in graph.iter() {
            if Self::has_side_effects(&node.op) || node.is_dead() {
                if !node.is_dead() {
                    live.insert(id.as_usize());
                    worklist.push(id);
                }
            }
        }

        // Phase 2: Propagate liveness backwards through inputs
        while let Some(id) = worklist.pop() {
            let node = graph.node(id);
            for input in node.inputs.iter() {
                if input.is_valid() && !live.contains(input.as_usize()) {
                    live.insert(input.as_usize());
                    worklist.push(input);
                }
            }
        }

        // Phase 3: Collect dead nodes
        let mut dead_nodes = Vec::new();
        for (id, node) in graph.iter() {
            if !live.contains(id.as_usize()) && !node.is_dead() {
                dead_nodes.push(id);
            }
        }

        // Phase 4: Kill dead nodes
        self.removed = dead_nodes.len();
        for id in dead_nodes {
            graph.kill(id);
        }

        self.removed > 0
    }
}
