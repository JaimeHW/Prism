use super::GraphBuilder;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ArithOp, CmpOp, Operator};

/// Builder trait for arithmetic and comparison operations.
pub trait ArithmeticBuilder {
    // Constants
    fn const_int(&mut self, value: i64) -> NodeId;
    fn const_float(&mut self, value: f64) -> NodeId;
    fn const_bool(&mut self, value: bool) -> NodeId;
    fn const_none(&mut self) -> NodeId;

    // Integer Arithmetic
    fn int_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_mod(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_neg(&mut self, value: NodeId) -> NodeId;

    // Float Arithmetic
    fn float_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn float_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn float_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn float_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn float_neg(&mut self, value: NodeId) -> NodeId;

    // Comparisons
    fn int_lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_le(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_ne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_gt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn int_ge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    // Generic Arithmetic
    fn generic_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    // Comparisons
    fn generic_lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_le(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_ne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_gt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn generic_ge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
}

impl ArithmeticBuilder for GraphBuilder {
    // Constants
    fn const_int(&mut self, value: i64) -> NodeId {
        self.graph.const_int(value)
    }

    fn const_float(&mut self, value: f64) -> NodeId {
        self.graph.const_float(value)
    }

    fn const_bool(&mut self, value: bool) -> NodeId {
        self.graph.const_bool(value)
    }

    fn const_none(&mut self) -> NodeId {
        self.graph.const_none()
    }

    // Integer Arithmetic
    fn int_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_add(lhs, rhs)
    }

    fn int_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_sub(lhs, rhs)
    }

    fn int_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_mul(lhs, rhs)
    }

    fn int_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.add_node(
            Operator::IntOp(ArithOp::FloorDiv),
            InputList::Pair(lhs, rhs),
        )
    }

    fn int_mod(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntOp(ArithOp::Mod), InputList::Pair(lhs, rhs))
    }

    fn int_neg(&mut self, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntOp(ArithOp::Neg), InputList::Single(value))
    }

    // Float Arithmetic
    fn float_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.float_add(lhs, rhs)
    }

    fn float_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::FloatOp(ArithOp::Sub), InputList::Pair(lhs, rhs))
    }

    fn float_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::FloatOp(ArithOp::Mul), InputList::Pair(lhs, rhs))
    }

    fn float_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.add_node(
            Operator::FloatOp(ArithOp::TrueDiv),
            InputList::Pair(lhs, rhs),
        )
    }

    fn float_neg(&mut self, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::FloatOp(ArithOp::Neg), InputList::Single(value))
    }

    // Comparisons
    fn int_lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_lt(lhs, rhs)
    }

    fn int_le(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Le), InputList::Pair(lhs, rhs))
    }

    fn int_eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_eq(lhs, rhs)
    }

    fn int_ne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Ne), InputList::Pair(lhs, rhs))
    }

    fn int_gt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Gt), InputList::Pair(lhs, rhs))
    }

    fn int_ge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Ge), InputList::Pair(lhs, rhs))
    }

    // Generic Arithmetic
    fn generic_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericOp(ArithOp::Add), InputList::Pair(lhs, rhs))
    }

    fn generic_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericOp(ArithOp::Sub), InputList::Pair(lhs, rhs))
    }

    fn generic_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericOp(ArithOp::Mul), InputList::Pair(lhs, rhs))
    }

    // Generic Comparisons
    fn generic_lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericCmp(CmpOp::Lt), InputList::Pair(lhs, rhs))
    }

    fn generic_le(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericCmp(CmpOp::Le), InputList::Pair(lhs, rhs))
    }

    fn generic_eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericCmp(CmpOp::Eq), InputList::Pair(lhs, rhs))
    }

    fn generic_ne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericCmp(CmpOp::Ne), InputList::Pair(lhs, rhs))
    }

    fn generic_gt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericCmp(CmpOp::Gt), InputList::Pair(lhs, rhs))
    }

    fn generic_ge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GenericCmp(CmpOp::Ge), InputList::Pair(lhs, rhs))
    }
}
