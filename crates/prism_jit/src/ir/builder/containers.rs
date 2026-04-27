use super::GraphBuilder;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::Operator;
use crate::ir::types::ValueType;

/// Builder trait for container operations.
pub trait ContainerBuilder {
    fn build_list(&mut self, elements: &[NodeId]) -> NodeId;
    fn build_tuple(&mut self, elements: &[NodeId]) -> NodeId;

    // Iteration
    fn get_iter(&mut self, iterable: NodeId) -> NodeId;
    fn iter_next(&mut self, iterator: NodeId) -> NodeId;
    fn len(&mut self, obj: NodeId) -> NodeId;
}

impl ContainerBuilder for GraphBuilder {
    fn build_list(&mut self, elements: &[NodeId]) -> NodeId {
        self.graph.add_node_with_type(
            Operator::BuildList(elements.len() as u16),
            InputList::from_slice(elements),
            ValueType::List,
        )
    }

    fn build_tuple(&mut self, elements: &[NodeId]) -> NodeId {
        self.graph.add_node_with_type(
            Operator::BuildTuple(elements.len() as u16),
            InputList::from_slice(elements),
            ValueType::Tuple,
        )
    }

    fn get_iter(&mut self, iterable: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::GetIter, InputList::Single(iterable))
    }

    fn iter_next(&mut self, iterator: NodeId) -> NodeId {
        self.graph
            .add_node(Operator::IterNext, InputList::Single(iterator))
    }

    fn len(&mut self, obj: NodeId) -> NodeId {
        self.graph
            .add_node_with_type(Operator::Len, InputList::Single(obj), ValueType::Int64)
    }
}
