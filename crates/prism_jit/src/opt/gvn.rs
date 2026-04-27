//! Global Value Numbering (GVN) optimization pass.
//!
//! GVN eliminates redundant computations by finding nodes that compute
//! the same value and replacing all uses with a single canonical computation.
//!
//! # Algorithm
//!
//! 1. For each node, compute a hash based on its operator and input IDs
//! 2. If a node with the same hash exists, compare structurally
//! 3. If structurally identical, replace all uses of the duplicate
//!
//! # Complexity
//!
//! O(n) time where n is the number of nodes, assuming good hash distribution.

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::Operator;

use std::collections::HashMap;

// =============================================================================
// Value Numbering Key
// =============================================================================

/// A key for value numbering that captures the essential properties of a node.
#[derive(Clone, PartialEq, Eq, Hash)]
struct GvnKey {
    /// The operator.
    op: Operator,
    /// Input node IDs (as raw u32 values for hashing).
    inputs: Vec<u32>,
}

impl GvnKey {
    /// Create a GVN key for a node.
    fn new(op: Operator, inputs: &[NodeId]) -> Self {
        GvnKey {
            op,
            inputs: inputs.iter().map(|id| id.index()).collect(),
        }
    }
}

// =============================================================================
// GVN Pass
// =============================================================================

/// Global Value Numbering pass.
pub struct Gvn {
    /// Number of nodes deduplicated.
    deduplicated: usize,
}

impl Gvn {
    /// Create a new GVN pass.
    pub fn new() -> Self {
        Gvn { deduplicated: 0 }
    }

    /// Get the number of deduplicated nodes.
    pub fn deduplicated(&self) -> usize {
        self.deduplicated
    }

    /// Check if an operator is eligible for GVN.
    fn is_gvn_eligible(op: &Operator) -> bool {
        match op {
            // Constants are always eligible (pure, no side effects)
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone => true,

            // Pure arithmetic operations
            Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::GenericOp(_)
            | Operator::MulHigh
            | Operator::MulHighSigned => true,

            // Pure comparisons
            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => true,

            // Bitwise operations
            Operator::Bitwise(_) => true,

            // Logical not
            Operator::LogicalNot => true,

            // Box/unbox are pure
            Operator::Box | Operator::Unbox => true,

            // Projections are pure
            Operator::Projection(_) => true,

            // Pure vector operations (no memory access)
            Operator::VectorArith(..)
            | Operator::VectorFma(_)
            | Operator::VectorBroadcast(_)
            | Operator::VectorExtract(..)
            | Operator::VectorInsert(..)
            | Operator::VectorShuffle(..)
            | Operator::VectorHadd(_)
            | Operator::VectorCmp(..)
            | Operator::VectorBlend(_)
            | Operator::VectorSplat(..) => true,

            // These have side effects or are context-dependent
            Operator::Parameter(_)
            | Operator::Phi
            | Operator::LoopPhi
            | Operator::Control(_)
            | Operator::Guard(_)
            | Operator::Call(_)
            | Operator::Memory(_)
            | Operator::VectorMemory(..)
            | Operator::GetItem
            | Operator::SetItem
            | Operator::GetAttr
            | Operator::SetAttr
            | Operator::GetIter
            | Operator::IterNext
            | Operator::Len
            | Operator::TypeCheck
            | Operator::BuildList(_)
            | Operator::BuildTuple(_)
            | Operator::BuildDict(_) => false,
        }
    }
}

impl Default for Gvn {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Gvn {
    fn name(&self) -> &'static str {
        "GVN"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.deduplicated = 0;

        // Map from GVN key to canonical node ID
        let mut value_table: HashMap<GvnKey, NodeId> = HashMap::new();

        // Map from original node to replacement
        let mut replacements: HashMap<NodeId, NodeId> = HashMap::new();

        // First pass: build value table and find duplicates
        for (id, node) in graph.iter() {
            // Skip nodes not eligible for GVN
            if !Self::is_gvn_eligible(&node.op) {
                continue;
            }

            // Create key from operator and inputs
            let inputs: Vec<NodeId> = node.inputs.iter().collect();
            let key = GvnKey::new(node.op, &inputs);

            // Check if we've seen this computation before
            if let Some(&canonical) = value_table.get(&key) {
                // Found a duplicate - record replacement
                if canonical != id {
                    replacements.insert(id, canonical);
                    self.deduplicated += 1;
                }
            } else {
                // New computation - add to table
                value_table.insert(key, id);
            }
        }

        // Second pass: apply replacements
        if !replacements.is_empty() {
            for (old, new) in &replacements {
                graph.replace_all_uses(*old, *new);
            }
        }

        self.deduplicated > 0
    }
}
