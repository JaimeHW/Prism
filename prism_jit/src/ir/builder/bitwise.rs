use super::GraphBuilder;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{BitwiseOp, Operator};

/// Builder trait for bitwise operations.
pub trait BitwiseBuilder {
    /// Bitwise AND: a & b
    fn bitwise_and(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    /// Bitwise OR: a | b
    fn bitwise_or(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    /// Bitwise XOR: a ^ b
    fn bitwise_xor(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    /// Left shift: a << b
    fn bitwise_shl(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    /// Right shift: a >> b
    fn bitwise_shr(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    /// Bitwise NOT: ~a
    fn bitwise_not(&mut self, value: NodeId) -> NodeId;
}

impl BitwiseBuilder for GraphBuilder {
    fn bitwise_and(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::Bitwise(BitwiseOp::And), InputList::Pair(lhs, rhs))
    }

    fn bitwise_or(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::Bitwise(BitwiseOp::Or), InputList::Pair(lhs, rhs))
    }

    fn bitwise_xor(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::Bitwise(BitwiseOp::Xor), InputList::Pair(lhs, rhs))
    }

    fn bitwise_shl(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::Bitwise(BitwiseOp::Shl), InputList::Pair(lhs, rhs))
    }

    fn bitwise_shr(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::Bitwise(BitwiseOp::Shr), InputList::Pair(lhs, rhs))
    }

    fn bitwise_not(&mut self, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::Bitwise(BitwiseOp::Not), InputList::Single(value))
    }
}
