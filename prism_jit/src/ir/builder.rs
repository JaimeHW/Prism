//! IR Graph Builder - Bytecode to Sea-of-Nodes translation.
//!
//! The builder translates bytecode into SSA-form Sea-of-Nodes IR:
//!
//! - **SSA construction**: Phi nodes at merge points
//! - **Type inference**: Using profiler feedback
//! - **Guard insertion**: For speculative optimization
//!
//! # Translation Strategy
//!
//! 1. **Abstract interpretation**: Track register values as NodeIds
//! 2. **Control flow merging**: Insert Phi nodes via dominance frontier
//! 3. **Type specialization**: Use profiler data to insert guards

use super::graph::Graph;
use super::node::{InputList, NodeId};
use super::operators::{ArithOp, CmpOp, ControlOp, Operator};
use super::types::ValueType;

use std::collections::HashMap;

// =============================================================================
// Register State
// =============================================================================

/// State of bytecode registers during IR construction.
#[derive(Clone)]
struct RegisterState {
    /// Current value for each register.
    values: Vec<NodeId>,

    /// Current control dependency.
    control: NodeId,
}

impl RegisterState {
    /// Create a new register state.
    fn new(num_registers: usize, control: NodeId) -> Self {
        RegisterState {
            values: vec![NodeId::INVALID; num_registers],
            control,
        }
    }

    /// Get a register value.
    fn get(&self, reg: u16) -> NodeId {
        self.values
            .get(reg as usize)
            .copied()
            .unwrap_or(NodeId::INVALID)
    }

    /// Set a register value.
    fn set(&mut self, reg: u16, value: NodeId) {
        if (reg as usize) < self.values.len() {
            self.values[reg as usize] = value;
        }
    }
}

// =============================================================================
// Graph Builder
// =============================================================================

/// Builder for constructing IR graphs from bytecode.
///
/// The builder maintains abstract state during translation,
/// creating Phi nodes at control merge points.
pub struct GraphBuilder {
    /// The graph being built.
    graph: Graph,

    /// Current register state.
    state: RegisterState,

    /// States at each bytecode offset (for jumps).
    states_at_offset: HashMap<u32, RegisterState>,

    /// Number of bytecode registers.
    num_registers: usize,

    /// Parameter nodes.
    parameters: Vec<NodeId>,
}

impl GraphBuilder {
    /// Create a new graph builder.
    pub fn new(num_registers: usize, num_parameters: usize) -> Self {
        let mut graph = Graph::new();
        let start = graph.start;

        // Create parameter nodes
        let mut parameters = Vec::with_capacity(num_parameters);
        for i in 0..num_parameters {
            let param = graph.add_node(Operator::Parameter(i as u16), InputList::Single(start));
            parameters.push(param);
        }

        let state = RegisterState::new(num_registers, start);

        GraphBuilder {
            graph,
            state,
            states_at_offset: HashMap::new(),
            num_registers,
            parameters,
        }
    }

    /// Get the constructed graph.
    pub fn finish(self) -> Graph {
        self.graph
    }

    /// Get a reference to the graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get a mutable reference to the graph.
    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    // =========================================================================
    // State Management
    // =========================================================================

    /// Set the current bytecode offset.
    pub fn set_bc_offset(&mut self, offset: u32) {
        self.graph.set_bc_offset(offset);
    }

    /// Get a parameter node.
    pub fn parameter(&self, index: usize) -> Option<NodeId> {
        self.parameters.get(index).copied()
    }

    /// Get the current control node.
    pub fn control(&self) -> NodeId {
        self.state.control
    }

    /// Set the current control node.
    pub fn set_control(&mut self, control: NodeId) {
        self.state.control = control;
    }

    /// Get a register value.
    pub fn get_register(&self, reg: u16) -> NodeId {
        self.state.get(reg)
    }

    /// Set a register value.
    pub fn set_register(&mut self, reg: u16, value: NodeId) {
        self.state.set(reg, value);
    }

    /// Save state at a bytecode offset (for forward jumps).
    pub fn save_state(&mut self, offset: u32) {
        self.states_at_offset.insert(offset, self.state.clone());
    }

    /// Merge with state at a bytecode offset.
    pub fn merge_state(&mut self, offset: u32) {
        if let Some(other) = self.states_at_offset.get(&offset).cloned() {
            self.merge_with_state(&other);
        }
    }

    /// Merge current state with another state.
    fn merge_with_state(&mut self, other: &RegisterState) {
        // Create region for merge
        let region = self.graph.region(&[self.state.control, other.control]);

        // Create Phi nodes for differing registers
        for i in 0..self.num_registers {
            let my_val = self.state.values[i];
            let other_val = other.values[i];

            if my_val != other_val && my_val.is_valid() && other_val.is_valid() {
                // Get the type from the graph
                let ty = self.graph.node(my_val).ty;
                let phi = self.graph.phi(region, &[my_val, other_val], ty);
                self.state.values[i] = phi;
            }
        }

        self.state.control = region;
    }

    // =========================================================================
    // Constants
    // =========================================================================

    /// Create an integer constant.
    pub fn const_int(&mut self, value: i64) -> NodeId {
        self.graph.const_int(value)
    }

    /// Create a float constant.
    pub fn const_float(&mut self, value: f64) -> NodeId {
        self.graph.const_float(value)
    }

    /// Create a boolean constant.
    pub fn const_bool(&mut self, value: bool) -> NodeId {
        self.graph.const_bool(value)
    }

    /// Create a None constant.
    pub fn const_none(&mut self) -> NodeId {
        self.graph.const_none()
    }

    // =========================================================================
    // Integer Arithmetic
    // =========================================================================

    /// Integer addition.
    pub fn int_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_add(lhs, rhs)
    }

    /// Integer subtraction.
    pub fn int_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_sub(lhs, rhs)
    }

    /// Integer multiplication.
    pub fn int_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_mul(lhs, rhs)
    }

    /// Integer division.
    pub fn int_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.add_node(
            Operator::IntOp(ArithOp::FloorDiv),
            InputList::Pair(lhs, rhs),
        )
    }

    /// Integer modulo.
    pub fn int_mod(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntOp(ArithOp::Mod), InputList::Pair(lhs, rhs))
    }

    /// Integer negation.
    pub fn int_neg(&mut self, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntOp(ArithOp::Neg), InputList::Single(value))
    }

    // =========================================================================
    // Float Arithmetic
    // =========================================================================

    /// Float addition.
    pub fn float_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.float_add(lhs, rhs)
    }

    /// Float subtraction.
    pub fn float_sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::FloatOp(ArithOp::Sub), InputList::Pair(lhs, rhs))
    }

    /// Float multiplication.
    pub fn float_mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::FloatOp(ArithOp::Mul), InputList::Pair(lhs, rhs))
    }

    /// Float division.
    pub fn float_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.add_node(
            Operator::FloatOp(ArithOp::TrueDiv),
            InputList::Pair(lhs, rhs),
        )
    }

    /// Float negation.
    pub fn float_neg(&mut self, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::FloatOp(ArithOp::Neg), InputList::Single(value))
    }

    // =========================================================================
    // Comparisons
    // =========================================================================

    /// Integer less-than.
    pub fn int_lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_lt(lhs, rhs)
    }

    /// Integer less-than-or-equal.
    pub fn int_le(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Le), InputList::Pair(lhs, rhs))
    }

    /// Integer equal.
    pub fn int_eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph.int_eq(lhs, rhs)
    }

    /// Integer not-equal.
    pub fn int_ne(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Ne), InputList::Pair(lhs, rhs))
    }

    /// Integer greater-than.
    pub fn int_gt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Gt), InputList::Pair(lhs, rhs))
    }

    /// Integer greater-than-or-equal.
    pub fn int_ge(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IntCmp(CmpOp::Ge), InputList::Pair(lhs, rhs))
    }

    // =========================================================================
    // Control Flow
    // =========================================================================

    /// Create a region (control merge).
    pub fn region(&mut self, preds: &[NodeId]) -> NodeId {
        let region = self.graph.region(preds);
        self.state.control = region;
        region
    }

    /// Create an if branch.
    pub fn branch(&mut self, condition: NodeId) -> NodeId {
        self.graph.add_node_with_type(
            Operator::Control(ControlOp::If),
            InputList::Pair(self.state.control, condition),
            ValueType::Control,
        )
    }

    /// Create a loop header.
    pub fn loop_header(&mut self, entry_control: NodeId) -> NodeId {
        let loop_node = self.graph.add_node_with_type(
            Operator::Control(ControlOp::Loop),
            InputList::Single(entry_control),
            ValueType::Control,
        );
        self.state.control = loop_node;
        loop_node
    }

    /// Create a return node.
    pub fn return_value(&mut self, value: NodeId) -> NodeId {
        self.graph.return_value(self.state.control, value)
    }

    /// Create a return None node.
    pub fn return_none(&mut self) -> NodeId {
        let none = self.const_none();
        self.return_value(none)
    }

    // =========================================================================
    // Phi Nodes
    // =========================================================================

    /// Create a Phi node.
    pub fn phi(&mut self, region: NodeId, values: &[NodeId], ty: ValueType) -> NodeId {
        self.graph.phi(region, values, ty)
    }

    /// Create a LoopPhi node.
    pub fn loop_phi(&mut self, loop_header: NodeId, initial: NodeId, ty: ValueType) -> NodeId {
        self.graph
            .add_node_with_type(Operator::LoopPhi, InputList::Pair(loop_header, initial), ty)
    }

    /// Update a LoopPhi's back value.
    pub fn set_loop_phi_back(&mut self, phi: NodeId, back_value: NodeId) {
        let node = self.graph.node_mut(phi);
        node.inputs.push(back_value);
    }

    // =========================================================================
    // Memory Operations
    // =========================================================================

    /// Get item: obj[key].
    pub fn get_item(&mut self, obj: NodeId, key: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GetItem, InputList::Pair(obj, key))
    }

    /// Set item: obj[key] = value.
    pub fn set_item(&mut self, obj: NodeId, key: NodeId, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::SetItem, InputList::Triple(obj, key, value))
    }

    /// Get attribute: obj.attr.
    pub fn get_attr(&mut self, obj: NodeId, _attr_index: u16) -> NodeId {
        // TODO: Store attr index in operator
        self.graph
            .add_node(Operator::GetAttr, InputList::Single(obj))
    }

    /// Set attribute: obj.attr = value.
    pub fn set_attr(&mut self, obj: NodeId, _attr_index: u16, value: NodeId) -> NodeId {
        // TODO: Store attr index in operator
        self.graph
            .add_node(Operator::SetAttr, InputList::Pair(obj, value))
    }

    // =========================================================================
    // Container Operations
    // =========================================================================

    /// Build a list from elements.
    pub fn build_list(&mut self, elements: &[NodeId]) -> NodeId {
        self.graph.add_node_with_type(
            Operator::BuildList(elements.len() as u16),
            InputList::from_slice(elements),
            ValueType::List,
        )
    }

    /// Build a tuple from elements.
    pub fn build_tuple(&mut self, elements: &[NodeId]) -> NodeId {
        self.graph.add_node_with_type(
            Operator::BuildTuple(elements.len() as u16),
            InputList::from_slice(elements),
            ValueType::Tuple,
        )
    }

    /// Get iterator.
    pub fn get_iter(&mut self, iterable: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GetIter, InputList::Single(iterable))
    }

    /// Get next from iterator.
    pub fn iter_next(&mut self, iterator: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IterNext, InputList::Single(iterator))
    }

    /// Get length.
    pub fn len(&mut self, obj: NodeId) -> NodeId {
        self.graph
            .add_node_with_type(Operator::Len, InputList::Single(obj), ValueType::Int64)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let mut builder = GraphBuilder::new(4, 2);

        // Get parameters
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Add them
        let sum = builder.int_add(p0, p1);

        // Return
        let _ret = builder.return_value(sum);

        let graph = builder.finish();
        assert!(graph.len() > 4); // start, end, 2 params, add, return
    }

    #[test]
    fn test_builder_constants() {
        let mut builder = GraphBuilder::new(4, 0);

        let a = builder.const_int(10);
        let b = builder.const_int(20);
        let sum = builder.int_add(a, b);

        builder.set_register(0, sum);

        assert_eq!(builder.get_register(0), sum);
    }

    #[test]
    fn test_builder_comparison() {
        let mut builder = GraphBuilder::new(4, 0);

        let a = builder.const_int(5);
        let b = builder.const_int(10);

        let lt = builder.int_lt(a, b);
        let eq = builder.int_eq(a, b);

        let graph = builder.finish();

        assert_eq!(graph.node(lt).ty, ValueType::Bool);
        assert_eq!(graph.node(eq).ty, ValueType::Bool);
    }

    #[test]
    fn test_builder_loop() {
        let mut builder = GraphBuilder::new(4, 0);

        let entry = builder.control();
        let loop_head = builder.loop_header(entry);

        // Create loop phi for counter
        let initial = builder.const_int(0);
        let counter = builder.loop_phi(loop_head, initial, ValueType::Int64);

        // Increment
        let one = builder.const_int(1);
        let next = builder.int_add(counter, one);

        // Update back edge
        builder.set_loop_phi_back(counter, next);

        let graph = builder.finish();
        assert!(graph.node(counter).is_phi());
    }

    #[test]
    fn test_builder_register_state() {
        let mut builder = GraphBuilder::new(8, 0);

        let c1 = builder.const_int(42);
        let c2 = builder.const_int(100);

        builder.set_register(0, c1);
        builder.set_register(3, c2);

        assert_eq!(builder.get_register(0), c1);
        assert_eq!(builder.get_register(3), c2);
        assert!(!builder.get_register(1).is_valid()); // Uninitialized
    }
}
