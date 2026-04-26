//! Shared Transform Utilities for Tail Call Optimization
//!
//! Common transformation utilities used by self-recursion, mutual recursion,
//! and sibling call optimization.

use rustc_hash::FxHashMap;

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{CallKind, ControlOp, Operator};

// =============================================================================
// Statistics
// =============================================================================

/// Statistics about tail call optimization.
#[derive(Debug, Clone, Default)]
pub struct TcoStats {
    /// Total tail calls found.
    pub tail_calls_found: usize,
    /// Self-tail-calls transformed to loops.
    pub self_recursion_transformed: usize,
    /// Mutual recursion sets optimized.
    pub mutual_recursion_optimized: usize,
    /// Sibling calls transformed to jumps.
    pub sibling_calls_optimized: usize,
    /// Tail calls that couldn't be optimized.
    pub not_optimized: usize,
}

impl TcoStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total optimizations performed.
    pub fn total_optimized(&self) -> usize {
        self.self_recursion_transformed
            + self.mutual_recursion_optimized
            + self.sibling_calls_optimized
    }

    /// Merge another stats into this one.
    pub fn merge(&mut self, other: &TcoStats) {
        self.tail_calls_found += other.tail_calls_found;
        self.self_recursion_transformed += other.self_recursion_transformed;
        self.mutual_recursion_optimized += other.mutual_recursion_optimized;
        self.sibling_calls_optimized += other.sibling_calls_optimized;
        self.not_optimized += other.not_optimized;
    }
}

// =============================================================================
// Value Remapping
// =============================================================================

/// Tracks value remapping during transformation.
#[derive(Debug, Clone, Default)]
pub struct ValueRemap {
    /// Old node -> new node mapping.
    map: FxHashMap<NodeId, NodeId>,
}

impl ValueRemap {
    /// Create a new empty remap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping.
    pub fn add(&mut self, old: NodeId, new: NodeId) {
        self.map.insert(old, new);
    }

    /// Get the remapped node, or original if not remapped.
    pub fn get(&self, node: NodeId) -> NodeId {
        self.map.get(&node).copied().unwrap_or(node)
    }

    /// Check if a node is remapped.
    pub fn contains(&self, node: NodeId) -> bool {
        self.map.contains_key(&node)
    }

    /// Apply remapping to a slice of nodes.
    pub fn remap_slice(&self, nodes: &mut [NodeId]) {
        for node in nodes {
            *node = self.get(*node);
        }
    }

    /// Get number of mappings.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// =============================================================================
// Phi Construction
// =============================================================================

/// Helper for constructing phi nodes.
pub struct PhiBuilder {
    /// Region/header node.
    region: NodeId,
    /// Incoming values.
    values: Vec<NodeId>,
}

impl PhiBuilder {
    /// Create a new phi builder.
    pub fn new(region: NodeId) -> Self {
        Self {
            region,
            values: Vec::new(),
        }
    }

    /// Add an incoming value.
    pub fn add_incoming(&mut self, value: NodeId) -> &mut Self {
        self.values.push(value);
        self
    }

    /// Build a Phi node.
    pub fn build_phi(self, graph: &mut Graph) -> NodeId {
        let inputs = self.make_inputs();
        graph.add_node(Operator::Phi, inputs)
    }

    /// Build a LoopPhi node.
    pub fn build_loop_phi(self, graph: &mut Graph) -> NodeId {
        let inputs = self.make_inputs();
        graph.add_node(Operator::LoopPhi, inputs)
    }

    /// Create the input list.
    fn make_inputs(self) -> InputList {
        match self.values.len() {
            0 => InputList::Single(self.region),
            1 => InputList::Pair(self.region, self.values[0]),
            2 => InputList::Triple(self.region, self.values[0], self.values[1]),
            3 => InputList::Quad(self.region, self.values[0], self.values[1], self.values[2]),
            _ => {
                let mut all = vec![self.region];
                all.extend(self.values);
                InputList::Many(all)
            }
        }
    }
}

// =============================================================================
// Control Flow Helpers
// =============================================================================

/// Create a loop header node.
pub fn create_loop_header(graph: &mut Graph) -> NodeId {
    graph.add_node(Operator::Control(ControlOp::Loop), InputList::Empty)
}

/// Create a region (control merge) node.
pub fn create_region(graph: &mut Graph, predecessors: &[NodeId]) -> NodeId {
    let inputs = match predecessors.len() {
        0 => InputList::Empty,
        1 => InputList::Single(predecessors[0]),
        2 => InputList::Pair(predecessors[0], predecessors[1]),
        3 => InputList::Triple(predecessors[0], predecessors[1], predecessors[2]),
        4 => InputList::Quad(
            predecessors[0],
            predecessors[1],
            predecessors[2],
            predecessors[3],
        ),
        _ => InputList::Many(predecessors.to_vec()),
    };
    graph.add_node(Operator::Control(ControlOp::Region), inputs)
}

// =============================================================================
// Argument Handling
// =============================================================================

/// Extract arguments from a call node.
pub fn extract_call_args(graph: &Graph, call_node: NodeId) -> Vec<NodeId> {
    graph
        .get(call_node)
        .map(|n| {
            let mut args = Vec::with_capacity(n.inputs.len());
            for i in 0..n.inputs.len() {
                if let Some(id) = n.inputs.get(i) {
                    args.push(id);
                }
            }
            args
        })
        .unwrap_or_default()
}

/// Count arguments in a call.
pub fn count_call_args(graph: &Graph, call_node: NodeId) -> usize {
    graph.get(call_node).map(|n| n.inputs.len()).unwrap_or(0)
}
