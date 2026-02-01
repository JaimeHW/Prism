//! Sea-of-Nodes IR node definitions.

/// A unique node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// IR node representing an operation.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Node operation.
    pub op: NodeOp,
    /// Input nodes (data dependencies).
    pub inputs: Vec<NodeId>,
}

/// Node operations.
#[derive(Debug, Clone)]
pub enum NodeOp {
    /// Start node.
    Start,
    /// End/return node.
    End,
    /// Integer constant.
    ConstInt(i64),
    /// Float constant.
    ConstFloat(f64),
    /// Parameter.
    Param(u32),
    /// Add operation.
    Add,
    /// Subtract operation.
    Sub,
    /// Multiply operation.
    Mul,
    /// Divide operation.
    Div,
    /// Phi node (SSA merge).
    Phi,
    /// Region (control merge).
    Region,
    /// If branch.
    If,
    /// Call.
    Call,
}
