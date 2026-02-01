//! Sea-of-Nodes graph structure.
//!
//! The graph provides:
//! - **Arena-based storage**: Efficient node allocation and traversal
//! - **Use-def chains**: Fast lookup of node users for optimization
//! - **Control flow structure**: Start, end, and region tracking
//! - **Incremental modification**: Add/remove nodes without full rebuild
//!
//! # Design Principles
//!
//! - **Unified representation**: Both data and control flow are edges
//! - **Explicit types**: Each node has a known output type
//! - **Minimal pointers**: Uses indices (NodeId) instead of Rc/Arc

use super::arena::{Arena, SecondaryMap};
use super::node::{InputList, Node, NodeId};
use super::operators::{ControlOp, Operator};
use super::types::ValueType;

// =============================================================================
// Graph Structure
// =============================================================================

/// A Sea-of-Nodes graph.
///
/// The graph owns all nodes and maintains use-def chains for efficient
/// optimization passes.
#[derive(Clone)]
pub struct Graph {
    /// Arena for node storage.
    nodes: Arena<Node>,

    /// Use chains: for each node, which nodes use its output.
    uses: SecondaryMap<Node, Vec<NodeId>>,

    /// The start node (control entry).
    pub start: NodeId,

    /// The end node (control exit).
    pub end: NodeId,

    /// Next bytecode offset to assign.
    next_bc_offset: u32,
}

impl Graph {
    /// Create a new empty graph with start and end nodes.
    pub fn new() -> Self {
        let mut nodes = Arena::with_capacity(256);
        let uses = SecondaryMap::new();

        // Create start node
        let start = nodes.alloc(Node::with_type(
            Operator::Control(ControlOp::Start),
            InputList::Empty,
            ValueType::Control,
        ));

        // Create end node with start as control input
        let end = nodes.alloc(Node::with_type(
            Operator::Control(ControlOp::End),
            InputList::Single(start),
            ValueType::Control,
        ));

        let mut graph = Graph {
            nodes,
            uses,
            start,
            end,
            next_bc_offset: 0,
        };

        // Register use of start by end
        graph.add_use(start, end);

        graph
    }

    /// Create a graph with pre-allocated capacity.
    pub fn with_capacity(node_capacity: usize) -> Self {
        let mut graph = Self::new();
        graph.nodes.reserve(node_capacity);
        graph
    }

    // =========================================================================
    // Node Access
    // =========================================================================

    /// Get a reference to a node.
    #[inline]
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// Get a mutable reference to a node.
    #[inline]
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    /// Get a node by ID (optional).
    #[inline]
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get a mutable node by ID (optional).
    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Get the number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty (only start/end).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 2
    }

    // =========================================================================
    // Node Creation
    // =========================================================================

    /// Add a new node to the graph.
    pub fn add_node(&mut self, op: Operator, inputs: InputList) -> NodeId {
        let ty = self.infer_type(&op, &inputs);
        let mut node = Node::with_type(op, inputs.clone(), ty);
        node.bc_offset = self.next_bc_offset;

        let id = self.nodes.alloc(node);

        // Register uses
        for input_id in inputs.iter() {
            self.add_use(input_id, id);
        }

        id
    }

    /// Add a node with explicit type.
    pub fn add_node_with_type(&mut self, op: Operator, inputs: InputList, ty: ValueType) -> NodeId {
        let mut node = Node::with_type(op, inputs.clone(), ty);
        node.bc_offset = self.next_bc_offset;

        let id = self.nodes.alloc(node);

        for input_id in inputs.iter() {
            self.add_use(input_id, id);
        }

        id
    }

    /// Set the current bytecode offset for new nodes.
    pub fn set_bc_offset(&mut self, offset: u32) {
        self.next_bc_offset = offset;
    }

    // =========================================================================
    // Use-Def Chains
    // =========================================================================

    /// Get all uses of a node (nodes that have this node as input).
    pub fn uses(&self, id: NodeId) -> &[NodeId] {
        self.uses.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get the number of uses.
    pub fn use_count(&self, id: NodeId) -> usize {
        self.uses.get(id).map(|v| v.len()).unwrap_or(0)
    }

    /// Add a use relationship: `user` uses `def`.
    fn add_use(&mut self, def: NodeId, user: NodeId) {
        self.uses.resize(def.as_usize() + 1);
        if let Some(uses) = self.uses.get_mut(def) {
            uses.push(user);
        } else {
            self.uses.set(def, vec![user]);
        }
    }

    /// Remove a use relationship.
    fn remove_use(&mut self, def: NodeId, user: NodeId) {
        if let Some(uses) = self.uses.get_mut(def) {
            if let Some(pos) = uses.iter().position(|&u| u == user) {
                uses.swap_remove(pos);
            }
        }
    }

    // =========================================================================
    // Node Modification
    // =========================================================================

    /// Replace a node's input at the given index.
    pub fn replace_input(&mut self, node: NodeId, index: usize, new_input: NodeId) {
        let old_input = self.nodes[node].inputs.get(index);

        // Remove old use
        if let Some(old) = old_input {
            self.remove_use(old, node);
        }

        // Update input
        self.nodes[node].inputs.set(index, new_input);

        // Add new use
        self.add_use(new_input, node);
    }

    /// Replace all uses of `old` with `new`.
    pub fn replace_all_uses(&mut self, old: NodeId, new: NodeId) {
        // Clone uses list to avoid borrow issues
        let users: Vec<NodeId> = self.uses(old).to_vec();

        // Collect all updates first to avoid borrowing issues
        let mut updates: Vec<(NodeId, usize)> = Vec::new();

        for user in &users {
            // Find which input(s) reference old
            let node = &self.nodes[*user];
            for i in 0..node.inputs.len() {
                if node.inputs.get(i) == Some(old) {
                    updates.push((*user, i));
                }
            }
        }

        // Apply updates
        for (user, i) in updates {
            self.nodes[user].inputs.set(i, new);
            self.add_use(new, user);
        }

        // Clear old uses
        if let Some(uses) = self.uses.get_mut(old) {
            uses.clear();
        }
    }

    /// Mark a node as dead (will be removed by DCE).
    pub fn kill(&mut self, id: NodeId) {
        self.nodes[id].mark_dead();

        // Remove from inputs' use lists
        let inputs: Vec<NodeId> = self.nodes[id].inputs.iter().collect();
        for input in inputs {
            self.remove_use(input, id);
        }
    }

    // =========================================================================
    // Type Inference
    // =========================================================================

    /// Infer the result type for an operator given its inputs.
    fn infer_type(&self, op: &Operator, inputs: &InputList) -> ValueType {
        let input_types: Vec<ValueType> = inputs
            .iter()
            .filter_map(|id| self.get(id).map(|n| n.ty))
            .collect();

        op.result_type(&input_types)
    }

    /// Recompute the type of a node based on its inputs.
    pub fn recompute_type(&mut self, id: NodeId) {
        let inputs = self.nodes[id].inputs.clone();
        let ty = self.infer_type(&self.nodes[id].op, &inputs);
        self.nodes[id].ty = ty;
    }

    // =========================================================================
    // Constants
    // =========================================================================

    /// Create an integer constant.
    pub fn const_int(&mut self, value: i64) -> NodeId {
        self.add_node(Operator::ConstInt(value), InputList::Empty)
    }

    /// Create a float constant.
    pub fn const_float(&mut self, value: f64) -> NodeId {
        self.add_node(Operator::ConstFloat(value.to_bits()), InputList::Empty)
    }

    /// Create a boolean constant.
    pub fn const_bool(&mut self, value: bool) -> NodeId {
        self.add_node(Operator::ConstBool(value), InputList::Empty)
    }

    /// Create a None constant.
    pub fn const_none(&mut self) -> NodeId {
        self.add_node(Operator::ConstNone, InputList::Empty)
    }

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    /// Create an integer add node.
    pub fn int_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        use super::operators::ArithOp;
        self.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(lhs, rhs))
    }

    /// Create an integer subtract node.
    pub fn int_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        use super::operators::ArithOp;
        self.add_node(Operator::IntOp(ArithOp::Sub), InputList::Pair(lhs, rhs))
    }

    /// Create an integer multiply node.
    pub fn int_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        use super::operators::ArithOp;
        self.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(lhs, rhs))
    }

    /// Create a float add node.
    pub fn float_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        use super::operators::ArithOp;
        self.add_node(Operator::FloatOp(ArithOp::Add), InputList::Pair(lhs, rhs))
    }

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    /// Create an integer less-than comparison.
    pub fn int_lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        use super::operators::CmpOp;
        self.add_node(Operator::IntCmp(CmpOp::Lt), InputList::Pair(lhs, rhs))
    }

    /// Create an integer equality comparison.
    pub fn int_eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        use super::operators::CmpOp;
        self.add_node(Operator::IntCmp(CmpOp::Eq), InputList::Pair(lhs, rhs))
    }

    // =========================================================================
    // Control Flow
    // =========================================================================

    /// Create a region (control merge) node.
    pub fn region(&mut self, preds: &[NodeId]) -> NodeId {
        self.add_node_with_type(
            Operator::Control(ControlOp::Region),
            InputList::from_slice(preds),
            ValueType::Control,
        )
    }

    /// Create a Phi node for value merging.
    pub fn phi(&mut self, region: NodeId, values: &[NodeId], ty: ValueType) -> NodeId {
        let mut inputs = vec![region];
        inputs.extend_from_slice(values);
        self.add_node_with_type(Operator::Phi, InputList::from_slice(&inputs), ty)
    }

    /// Create a return node.
    pub fn return_value(&mut self, control: NodeId, value: NodeId) -> NodeId {
        self.add_node_with_type(
            Operator::Control(ControlOp::Return),
            InputList::Pair(control, value),
            ValueType::Control,
        )
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterate over all nodes with their IDs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter()
    }

    /// Iterate over all node IDs.
    pub fn ids(&self) -> impl Iterator<Item = NodeId> {
        self.nodes.ids()
    }

    /// Iterate over all nodes mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (NodeId, &mut Node)> {
        self.nodes.iter_mut()
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Verify graph consistency (for debugging).
    #[cfg(debug_assertions)]
    pub fn verify(&self) -> Result<(), String> {
        // Check all inputs are valid
        for (id, node) in self.iter() {
            for input in node.inputs.iter() {
                if input.as_usize() >= self.nodes.len() {
                    return Err(format!("Node {:?} has invalid input {:?}", id, input));
                }
            }
        }

        // Check start has no inputs
        if !self.nodes[self.start].inputs.is_empty() {
            return Err("Start node should have no inputs".into());
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    pub fn verify(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph ({} nodes):", self.nodes.len())?;
        for (id, node) in self.iter() {
            writeln!(f, "  {:?}: {:?}", id, node)?;
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let g = Graph::new();
        assert!(g.len() >= 2); // start + end
        assert!(g.get(g.start).is_some());
        assert!(g.get(g.end).is_some());
    }

    #[test]
    fn test_add_constant() {
        let mut g = Graph::new();

        let c1 = g.const_int(42);
        let c2 = g.const_int(10);

        assert_eq!(g.node(c1).as_int(), Some(42));
        assert_eq!(g.node(c2).as_int(), Some(10));
    }

    #[test]
    fn test_add_arithmetic() {
        let mut g = Graph::new();

        let a = g.const_int(5);
        let b = g.const_int(3);
        let sum = g.int_add(a, b);

        assert_eq!(g.node(sum).inputs.len(), 2);
        assert_eq!(g.node(sum).inputs.get(0), Some(a));
        assert_eq!(g.node(sum).inputs.get(1), Some(b));
    }

    #[test]
    fn test_use_chains() {
        let mut g = Graph::new();

        let c = g.const_int(5);
        let _add1 = g.int_add(c, c);
        let _add2 = g.int_add(c, c);

        // c should have 4 uses (twice in each add)
        assert_eq!(g.use_count(c), 4);
    }

    #[test]
    fn test_replace_all_uses() {
        let mut g = Graph::new();

        let c1 = g.const_int(5);
        let c2 = g.const_int(10);
        let add = g.int_add(c1, c1);

        g.replace_all_uses(c1, c2);

        // add should now use c2
        assert_eq!(g.node(add).inputs.get(0), Some(c2));
        assert_eq!(g.node(add).inputs.get(1), Some(c2));
    }

    #[test]
    fn test_phi_node() {
        let mut g = Graph::new();

        let region = g.region(&[g.start]);
        let v1 = g.const_int(1);
        let v2 = g.const_int(2);
        let phi = g.phi(region, &[v1, v2], ValueType::Int64);

        assert!(g.node(phi).is_phi());
        assert_eq!(g.node(phi).inputs.len(), 3); // region + 2 values
    }

    #[test]
    fn test_graph_verify() {
        let g = Graph::new();
        assert!(g.verify().is_ok());
    }
}
