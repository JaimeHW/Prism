//! Sea-of-Nodes Intermediate Representation.
//!
//! This module provides a professional-grade IR for the JIT compiler:
//!
//! # Core Components
//!
//! - **Types** (`types.rs`): Value type lattice for type inference
//! - **Operators** (`operators.rs`): Comprehensive operator definitions
//! - **Arena** (`arena.rs`): Efficient memory allocation
//! - **Node** (`node.rs`): IR node definitions
//! - **Graph** (`graph.rs`): Sea-of-Nodes graph structure
//! - **CFG** (`cfg.rs`): Control flow graph and dominators
//! - **Builder** (`builder.rs`): Bytecode to IR translation
//!
//! # Design Principles
//!
//! - **Arena allocation**: O(1) node creation, cache-friendly traversal
//! - **Use-def chains**: Fast optimization passes
//! - **Type lattice**: Enables type specialization and guard elimination
//! - **Unified control/data**: Sea-of-Nodes representation

pub mod arena;
pub mod builder;
pub mod cfg;
pub mod graph;
pub mod node;
pub mod operators;
pub mod types;

// Re-export commonly used types
pub use arena::{Arena, BitSet, Id, SecondaryMap};
pub use builder::GraphBuilder;
pub use cfg::{BasicBlock, BlockId, Cfg, DominatorTree, Loop, LoopAnalysis};
pub use graph::Graph;
pub use node::{InputList, Node, NodeFlags, NodeId};
pub use operators::{
    ArithOp, BitwiseOp, CallKind, CmpOp, ControlOp, GuardKind, MemoryOp, Operator,
};
pub use types::{TypeTuple, ValueType};
