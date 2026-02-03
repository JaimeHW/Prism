//! IR node definitions for the Sea-of-Nodes IR.
//!
//! Sea-of-Nodes represents both data flow and control flow uniformly as edges
//! between nodes. This provides:
//! - **Unified representation**: No separate CFG structure
//! - **Natural scheduling flexibility**: Nodes can float to optimal positions
//! - **Efficient optimization**: Use-def chains enable fast dead code elimination
//!
//! # Node Structure
//!
//! Each node has:
//! - **Operator**: What the node computes
//! - **Inputs**: Data and control dependencies (use-def edges)
//! - **Output type**: Inferred or declared result type
//! - **Position**: Bytecode offset (for deoptimization)
//!
//! # Control vs Data Nodes
//!
//! - **Control nodes**: Start, End, Region, If, Loop - define execution order
//! - **Data nodes**: Arithmetic, Memory, Phi - compute values
//! - **Guard nodes**: Type checks that may deoptimize

use super::arena::Id;
use super::operators::Operator;
use super::types::ValueType;

// =============================================================================
// Node ID Type Alias
// =============================================================================

/// Unique identifier for a node in the graph.
pub type NodeId = Id<Node>;

// =============================================================================
// Input List
// =============================================================================

/// Maximum number of inline inputs before spilling to heap.
const INLINE_INPUTS: usize = 4;

/// Compact input list optimized for small node arity.
///
/// Most nodes have 0-4 inputs, so we store them inline to avoid allocation.
/// Larger input lists (e.g., Phi with many predecessors) use a Vec.
#[derive(Clone)]
pub enum InputList {
    /// No inputs.
    Empty,
    /// Single input (very common).
    Single(NodeId),
    /// Two inputs (binary ops).
    Pair(NodeId, NodeId),
    /// Three inputs.
    Triple(NodeId, NodeId, NodeId),
    /// Four inputs (inline limit).
    Quad(NodeId, NodeId, NodeId, NodeId),
    /// Many inputs (heap allocated).
    Many(Vec<NodeId>),
}

impl InputList {
    /// Create empty input list.
    pub const fn empty() -> Self {
        InputList::Empty
    }

    /// Create from a slice.
    pub fn from_slice(inputs: &[NodeId]) -> Self {
        match inputs.len() {
            0 => InputList::Empty,
            1 => InputList::Single(inputs[0]),
            2 => InputList::Pair(inputs[0], inputs[1]),
            3 => InputList::Triple(inputs[0], inputs[1], inputs[2]),
            4 => InputList::Quad(inputs[0], inputs[1], inputs[2], inputs[3]),
            _ => InputList::Many(inputs.to_vec()),
        }
    }

    /// Get the number of inputs.
    pub fn len(&self) -> usize {
        match self {
            InputList::Empty => 0,
            InputList::Single(_) => 1,
            InputList::Pair(..) => 2,
            InputList::Triple(..) => 3,
            InputList::Quad(..) => 4,
            InputList::Many(v) => v.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        matches!(self, InputList::Empty)
    }

    /// Get input at index.
    pub fn get(&self, index: usize) -> Option<NodeId> {
        match self {
            InputList::Empty => None,
            InputList::Single(a) => {
                if index == 0 {
                    Some(*a)
                } else {
                    None
                }
            }
            InputList::Pair(a, b) => match index {
                0 => Some(*a),
                1 => Some(*b),
                _ => None,
            },
            InputList::Triple(a, b, c) => match index {
                0 => Some(*a),
                1 => Some(*b),
                2 => Some(*c),
                _ => None,
            },
            InputList::Quad(a, b, c, d) => match index {
                0 => Some(*a),
                1 => Some(*b),
                2 => Some(*c),
                3 => Some(*d),
                _ => None,
            },
            InputList::Many(v) => v.get(index).copied(),
        }
    }

    /// Set input at index.
    pub fn set(&mut self, index: usize, value: NodeId) {
        match self {
            InputList::Single(a) if index == 0 => *a = value,
            InputList::Pair(a, b) => match index {
                0 => *a = value,
                1 => *b = value,
                _ => {}
            },
            InputList::Triple(a, b, c) => match index {
                0 => *a = value,
                1 => *b = value,
                2 => *c = value,
                _ => {}
            },
            InputList::Quad(a, b, c, d) => match index {
                0 => *a = value,
                1 => *b = value,
                2 => *c = value,
                3 => *d = value,
                _ => {}
            },
            InputList::Many(v) => {
                if index < v.len() {
                    v[index] = value;
                }
            }
            _ => {}
        }
    }

    /// Push a new input.
    pub fn push(&mut self, value: NodeId) {
        *self = match std::mem::take(self) {
            InputList::Empty => InputList::Single(value),
            InputList::Single(a) => InputList::Pair(a, value),
            InputList::Pair(a, b) => InputList::Triple(a, b, value),
            InputList::Triple(a, b, c) => InputList::Quad(a, b, c, value),
            InputList::Quad(a, b, c, d) => InputList::Many(vec![a, b, c, d, value]),
            InputList::Many(mut v) => {
                v.push(value);
                InputList::Many(v)
            }
        };
    }

    /// Iterate over inputs.
    pub fn iter(&self) -> InputIter<'_> {
        InputIter {
            list: self,
            index: 0,
        }
    }

    /// Get as slice (for Many variant).
    pub fn as_slice(&self) -> &[NodeId] {
        match self {
            InputList::Many(v) => v.as_slice(),
            _ => &[], // Less efficient for inline, but rarely needed
        }
    }

    /// Convert to Vec.
    pub fn to_vec(&self) -> Vec<NodeId> {
        self.iter().collect()
    }
}

impl Default for InputList {
    fn default() -> Self {
        InputList::Empty
    }
}

impl std::fmt::Debug for InputList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, id) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", id)?;
        }
        write!(f, "]")
    }
}

/// Iterator over input list.
pub struct InputIter<'a> {
    list: &'a InputList,
    index: usize,
}

impl<'a> Iterator for InputIter<'a> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.list.get(self.index);
        self.index += 1;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for InputIter<'_> {}

// =============================================================================
// Node
// =============================================================================

/// A node in the Sea-of-Nodes IR graph.
///
/// Nodes represent operations with inputs (dependencies) and a result type.
/// The graph structure is defined by the input edges.
#[derive(Clone)]
pub struct Node {
    /// The operation this node performs.
    pub op: Operator,

    /// Input nodes (dependencies).
    pub inputs: InputList,

    /// Result type (inferred or declared).
    pub ty: ValueType,

    /// Bytecode offset (for deoptimization and debugging).
    pub bc_offset: u32,

    /// Use count (number of nodes that use this node's output).
    /// This is maintained lazily and may be stale.
    pub use_count: u32,

    /// Flags for various node properties.
    pub flags: NodeFlags,
}

impl Node {
    /// Create a new node with the given operator and inputs.
    pub fn new(op: Operator, inputs: InputList) -> Self {
        let ty = op.result_type(&[]);
        Node {
            op,
            inputs,
            ty,
            bc_offset: 0,
            use_count: 0,
            flags: NodeFlags::empty(),
        }
    }

    /// Create a new node with operator, inputs, and type.
    pub fn with_type(op: Operator, inputs: InputList, ty: ValueType) -> Self {
        Node {
            op,
            inputs,
            ty,
            bc_offset: 0,
            use_count: 0,
            flags: NodeFlags::empty(),
        }
    }

    /// Create a constant integer node.
    pub fn const_int(value: i64) -> Self {
        Node::new(Operator::ConstInt(value), InputList::Empty)
    }

    /// Create a constant float node.
    pub fn const_float(value: f64) -> Self {
        Node::new(Operator::ConstFloat(value.to_bits()), InputList::Empty)
    }

    /// Create a constant bool node.
    pub fn const_bool(value: bool) -> Self {
        Node::new(Operator::ConstBool(value), InputList::Empty)
    }

    /// Create a ConstNone node.
    pub fn const_none() -> Self {
        Node::new(Operator::ConstNone, InputList::Empty)
    }

    /// Create a parameter node.
    pub fn parameter(index: u16) -> Self {
        Node::new(Operator::Parameter(index), InputList::Empty)
    }

    /// Get the first input (control input for many nodes).
    pub fn control_input(&self) -> Option<NodeId> {
        self.inputs.get(0)
    }

    /// Get the first data input.
    pub fn data_input(&self, index: usize) -> Option<NodeId> {
        self.inputs.get(index)
    }

    /// Check if this node is a constant.
    pub fn is_constant(&self) -> bool {
        matches!(
            self.op,
            Operator::ConstInt(_)
                | Operator::ConstFloat(_)
                | Operator::ConstBool(_)
                | Operator::ConstNone
        )
    }

    /// Check if this node is a control node.
    pub fn is_control(&self) -> bool {
        matches!(self.op, Operator::Control(_))
    }

    /// Check if this node is a Phi.
    pub fn is_phi(&self) -> bool {
        matches!(self.op, Operator::Phi | Operator::LoopPhi)
    }

    /// Check if this node is pure (no side effects).
    pub fn is_pure(&self) -> bool {
        self.op.is_pure()
    }

    /// Check if this node has been marked dead.
    pub fn is_dead(&self) -> bool {
        self.flags.contains(NodeFlags::DEAD)
    }

    /// Mark this node as dead.
    pub fn mark_dead(&mut self) {
        self.flags.insert(NodeFlags::DEAD);
    }

    /// Get as integer constant if this is one.
    pub fn as_int(&self) -> Option<i64> {
        match self.op {
            Operator::ConstInt(v) => Some(v),
            _ => None,
        }
    }

    /// Get as float constant if this is one.
    pub fn as_float(&self) -> Option<f64> {
        match self.op {
            Operator::ConstFloat(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    /// Get as bool constant if this is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self.op {
            Operator::ConstBool(v) => Some(v),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.op)?;
        if !self.inputs.is_empty() {
            write!(f, " {:?}", self.inputs)?;
        }
        write!(f, " : {:?}", self.ty)
    }
}

// =============================================================================
// Node Flags
// =============================================================================

bitflags::bitflags! {
    /// Flags for node properties.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct NodeFlags: u8 {
        /// Node has been marked dead (will be removed).
        const DEAD = 0b0000_0001;
        /// Node is pinned (cannot be moved by scheduler).
        const PINNED = 0b0000_0010;
        /// Node is in a loop.
        const IN_LOOP = 0b0000_0100;
        /// Node has been visited (for traversal).
        const VISITED = 0b0000_1000;
        /// Node is loop-invariant.
        const LOOP_INVARIANT = 0b0001_0000;
        /// Node has been hoisted out of a loop by LICM.
        const HOISTED = 0b0010_0000;
    }
}

impl Default for NodeFlags {
    fn default() -> Self {
        NodeFlags::empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operators::ArithOp;

    #[test]
    fn test_input_list_empty() {
        let list = InputList::empty();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
    }

    #[test]
    fn test_input_list_single() {
        let id = NodeId::new(42);
        let list = InputList::Single(id);
        assert_eq!(list.len(), 1);
        assert_eq!(list.get(0), Some(id));
        assert_eq!(list.get(1), None);
    }

    #[test]
    fn test_input_list_from_slice() {
        let ids: Vec<NodeId> = (0..5).map(|i| NodeId::new(i)).collect();
        let list = InputList::from_slice(&ids);
        assert_eq!(list.len(), 5);

        for (i, id) in list.iter().enumerate() {
            assert_eq!(id.index() as usize, i);
        }
    }

    #[test]
    fn test_input_list_push() {
        let mut list = InputList::empty();

        list.push(NodeId::new(1));
        assert_eq!(list.len(), 1);

        list.push(NodeId::new(2));
        assert_eq!(list.len(), 2);

        list.push(NodeId::new(3));
        list.push(NodeId::new(4));
        list.push(NodeId::new(5));
        assert_eq!(list.len(), 5);

        // Should now be Many variant
        matches!(list, InputList::Many(_));
    }

    #[test]
    fn test_node_const_int() {
        let node = Node::const_int(42);
        assert!(node.is_constant());
        assert_eq!(node.as_int(), Some(42));
        assert_eq!(node.ty, ValueType::Int64);
    }

    #[test]
    fn test_node_const_float() {
        let node = Node::const_float(3.14);
        assert!(node.is_constant());
        assert_eq!(node.as_float(), Some(3.14));
        assert_eq!(node.ty, ValueType::Float64);
    }

    #[test]
    fn test_node_arith() {
        let node = Node::new(
            Operator::IntOp(ArithOp::Add),
            InputList::Pair(NodeId::new(0), NodeId::new(1)),
        );
        assert!(!node.is_constant());
        assert!(node.is_pure());
        assert_eq!(node.inputs.len(), 2);
    }

    #[test]
    fn test_node_flags() {
        let mut node = Node::const_int(0);
        assert!(!node.is_dead());

        node.mark_dead();
        assert!(node.is_dead());
    }
}
