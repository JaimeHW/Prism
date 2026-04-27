use super::GraphBuilder;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ControlOp, Operator};
use crate::ir::types::ValueType;

/// Builder trait for control flow operations.
pub trait ControlBuilder {
    // Basic Control Flow
    fn region(&mut self, preds: &[NodeId]) -> NodeId;
    fn branch(&mut self, condition: NodeId) -> NodeId;
    fn loop_header(&mut self, entry_control: NodeId) -> NodeId;
    fn return_value(&mut self, value: NodeId) -> NodeId;
    fn return_none(&mut self) -> NodeId;

    // Phi Nodes
    fn phi(&mut self, region: NodeId, values: &[NodeId], ty: ValueType) -> NodeId;
    fn loop_phi(&mut self, loop_header: NodeId, initial: NodeId, ty: ValueType) -> NodeId;
    fn set_loop_phi_back(&mut self, phi: NodeId, back_value: NodeId);

    // High-level Translation
    fn translate_jump(&mut self, target: u32);
    fn translate_branch(&mut self, condition: NodeId, true_target: u32, false_target: u32);
}

impl ControlBuilder for GraphBuilder {
    fn region(&mut self, preds: &[NodeId]) -> NodeId {
        let region = self.graph.region(preds);
        self.state.control = region;
        region
    }

    fn branch(&mut self, condition: NodeId) -> NodeId {
        self.graph.add_node_with_type(
            Operator::Control(ControlOp::If),
            InputList::Pair(self.state.control, condition),
            ValueType::Control,
        )
    }

    fn loop_header(&mut self, entry_control: NodeId) -> NodeId {
        let loop_node = self.graph.add_node_with_type(
            Operator::Control(ControlOp::Loop),
            InputList::Single(entry_control),
            ValueType::Control,
        );
        self.state.control = loop_node;
        loop_node
    }

    fn return_value(&mut self, value: NodeId) -> NodeId {
        self.graph.return_value(self.state.control, value)
    }

    fn return_none(&mut self) -> NodeId {
        let none = self.graph.const_none();
        self.return_value(none)
    }

    fn phi(&mut self, region: NodeId, values: &[NodeId], ty: ValueType) -> NodeId {
        self.graph.phi(region, values, ty)
    }

    fn loop_phi(&mut self, loop_header: NodeId, initial: NodeId, ty: ValueType) -> NodeId {
        self.graph
            .add_node_with_type(Operator::LoopPhi, InputList::Pair(loop_header, initial), ty)
    }

    fn set_loop_phi_back(&mut self, phi: NodeId, back_value: NodeId) {
        let node = self.graph.node_mut(phi);
        node.inputs.push(back_value);
    }

    fn translate_jump(&mut self, target: u32) {
        // Unconditional jump just saves state to target.
        self.save_state(target);
    }

    fn translate_branch(&mut self, condition: NodeId, true_target: u32, false_target: u32) {
        let if_node = self.branch(condition);

        // True path (Project 0)
        let true_proj = self.graph.add_node_with_type(
            Operator::Projection(0),
            InputList::Single(if_node),
            ValueType::Control,
        );

        // Save true path state
        self.state.control = true_proj;
        self.save_state(true_target);

        // False path (Project 1)
        let false_proj = self.graph.add_node_with_type(
            Operator::Projection(1),
            InputList::Single(if_node),
            ValueType::Control,
        );

        // Save false path state
        self.state.control = false_proj;
        self.save_state(false_target);

        // Restore to true path (fallthrough assumption by translator)
        self.state.control = true_proj;
    }
}
