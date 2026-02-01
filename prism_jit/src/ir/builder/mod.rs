//! IR Graph Builder module.
//!
//! Refactored from monolithic builder.rs.

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::Operator;
use std::collections::HashMap;

pub mod arithmetic;
pub mod containers;
pub mod control;
pub mod guards;
pub mod objects;
pub mod translator;

#[cfg(test)]
mod tests;

pub use arithmetic::ArithmeticBuilder;
pub use containers::ContainerBuilder;
pub use control::ControlBuilder;
pub use guards::GuardBuilder;
pub use objects::ObjectBuilder;

// =============================================================================
// Register State
// =============================================================================

/// State of bytecode registers during IR construction.
#[derive(Clone, Debug)]
pub struct RegisterState {
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
pub struct GraphBuilder {
    /// The graph being built.
    pub(crate) graph: Graph,

    /// Current register state.
    pub(crate) state: RegisterState,

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
    /// Save state at a bytecode offset (for forward jumps).
    pub fn save_state(&mut self, offset: u32) {
        if self.states_at_offset.contains_key(&offset) {
            // Merge with existing state
            let incoming = self.state.clone();
            let accumulator = self.states_at_offset.remove(&offset).unwrap();

            // Swap in accumulator to merge into it
            let old_state = std::mem::replace(&mut self.state, accumulator);
            self.merge_with_state(&incoming);

            // Store result and restore original state
            self.states_at_offset.insert(offset, self.state.clone());
            self.state = old_state;
        } else {
            self.states_at_offset.insert(offset, self.state.clone());
        }
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
}
