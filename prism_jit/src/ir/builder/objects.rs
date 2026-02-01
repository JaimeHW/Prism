use super::GraphBuilder;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::Operator;

/// Builder trait for object operations.
pub trait ObjectBuilder {
    fn get_item(&mut self, obj: NodeId, key: NodeId) -> NodeId;
    fn set_item(&mut self, obj: NodeId, key: NodeId, value: NodeId) -> NodeId;
    fn get_attr(&mut self, obj: NodeId, name: NodeId) -> NodeId;
    fn set_attr(&mut self, obj: NodeId, name: NodeId, value: NodeId) -> NodeId;
    fn call(&mut self, function: NodeId, args: &[NodeId]) -> NodeId;
    fn call_method(&mut self, method_name: NodeId, obj: NodeId, args: &[NodeId]) -> NodeId;
}

impl ObjectBuilder for GraphBuilder {
    fn get_item(&mut self, obj: NodeId, key: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GetItem, InputList::Pair(obj, key))
    }

    fn set_item(&mut self, obj: NodeId, key: NodeId, value: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::SetItem, InputList::Triple(obj, key, value))
    }

    fn get_attr(&mut self, obj: NodeId, name: NodeId) -> NodeId {
        self.graph.add_node_with_type(
            Operator::GetAttr,
            InputList::Pair(obj, name),
            crate::ir::types::ValueType::Top,
        )
    }

    fn set_attr(&mut self, obj: NodeId, name: NodeId, value: NodeId) -> NodeId {
        self.graph.add_node_with_type(
            Operator::SetAttr,
            InputList::Triple(obj, name, value),
            crate::ir::types::ValueType::Top,
        )
    }

    fn call(&mut self, function: NodeId, args: &[NodeId]) -> NodeId {
        // Prepend function to args for the Call node inputs?
        // Usually Call node takes inputs: [function, arg0, arg1...]
        // Operator::Call(Direct)
        let mut inputs = Vec::with_capacity(args.len() + 1);
        inputs.push(function);
        inputs.extend_from_slice(args);

        self.graph.add_node_with_type(
            Operator::Call(crate::ir::operators::CallKind::Direct),
            InputList::from_slice(&inputs),
            crate::ir::types::ValueType::Top,
        )
    }

    fn call_method(&mut self, method_name: NodeId, obj: NodeId, args: &[NodeId]) -> NodeId {
        // Implementation:
        let method = self.get_attr(obj, method_name);
        self.call(method, args)
    }
}
