use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
use crate::ir::operators::ArithOp;

#[test]
fn test_clone_result_map_id() {
    let mut id_map = FxHashMap::default();
    id_map.insert(NodeId::new(0), NodeId::new(100));
    id_map.insert(NodeId::new(1), NodeId::new(101));

    let result = CloneResult {
        id_map,
        cloned_start: NodeId::new(100),
        cloned_returns: vec![],
        nodes_cloned: 2,
    };

    assert_eq!(result.map_id(NodeId::new(0)), Some(NodeId::new(100)));
    assert_eq!(result.map_id(NodeId::new(1)), Some(NodeId::new(101)));
    assert_eq!(result.map_id(NodeId::new(2)), None);
}

#[test]
fn test_clone_simple_graph() {
    // Create a simple source graph: add two parameters
    let mut source_builder = GraphBuilder::new(8, 2);
    let p0 = source_builder.parameter(0).unwrap();
    let p1 = source_builder.parameter(1).unwrap();
    let sum = source_builder.int_add(p0, p1);
    source_builder.return_value(sum);
    let source = source_builder.finish();

    // Create target graph
    let mut target = Graph::new();
    let arg0 = target.const_int(10);
    let arg1 = target.const_int(20);

    // Clone source into target
    let cloner = GraphCloner::new(&source)
        .with_arguments(&[arg0, arg1])
        .with_control_input(target.start);

    let result = cloner.clone_into(&mut target);

    // Should have cloned nodes (excluding start and parameters)
    assert!(result.nodes_cloned > 0);

    // The sum node should be mapped
    assert!(!result.id_map.is_empty());
}

#[test]
fn test_clone_with_argument_substitution() {
    // Create source graph with parameters
    let mut source_builder = GraphBuilder::new(8, 2);
    let p0 = source_builder.parameter(0).unwrap();
    let p1 = source_builder.parameter(1).unwrap();
    let _result = source_builder.int_add(p0, p1);
    let source = source_builder.finish();

    // Create target
    let mut target = Graph::new();
    let const_a = target.const_int(5);
    let const_b = target.const_int(10);

    // Clone with argument substitution
    let cloner = GraphCloner::new(&source)
        .with_argument(0, const_a)
        .with_argument(1, const_b);

    let result = cloner.clone_into(&mut target);

    // Parameters should map to our constants
    // Find the parameter nodes in source
    for (old_id, _) in source.iter() {
        if let Some(node) = source.get(old_id) {
            if let Operator::Parameter(0) = node.op {
                // This should map to const_a
                if let Some(mapped) = result.map_id(old_id) {
                    assert_eq!(mapped, const_a);
                }
            }
            if let Operator::Parameter(1) = node.op {
                // This should map to const_b
                if let Some(mapped) = result.map_id(old_id) {
                    assert_eq!(mapped, const_b);
                }
            }
        }
    }
}

#[test]
fn test_clone_control_input() {
    let mut source_builder = GraphBuilder::new(4, 1);
    let p0 = source_builder.parameter(0).unwrap();
    source_builder.return_value(p0);
    let source = source_builder.finish();

    let mut target = Graph::new();
    let region = target.region(&[target.start]);

    let cloner = GraphCloner::new(&source).with_control_input(region);

    let result = cloner.clone_into(&mut target);

    // The start node should map to our region
    assert_eq!(result.map_id(source.start), Some(region));
}

#[test]
fn test_clone_return_tracking() {
    let mut source_builder = GraphBuilder::new(4, 1);
    let p0 = source_builder.parameter(0).unwrap();
    source_builder.return_value(p0);
    let source = source_builder.finish();

    let mut target = Graph::new();
    let arg = target.const_int(42);

    let cloner = GraphCloner::new(&source).with_arguments(&[arg]);

    let result = cloner.clone_into(&mut target);

    // Should have captured the return node
    assert!(!result.cloned_returns.is_empty());
}

#[test]
fn test_clone_preserves_types() {
    let mut source = Graph::new();
    let int_const = source.const_int(42);
    let float_const = source.const_float(3.125);

    let int_type = source.node(int_const).ty;
    let float_type = source.node(float_const).ty;

    let mut target = Graph::new();
    let cloner = GraphCloner::new(&source);
    let result = cloner.clone_into(&mut target);

    // Check types are preserved
    if let Some(new_int) = result.map_id(int_const) {
        assert_eq!(target.node(new_int).ty, int_type);
    }
    if let Some(new_float) = result.map_id(float_const) {
        assert_eq!(target.node(new_float).ty, float_type);
    }
}

#[test]
fn test_clone_node_function() {
    let source = Graph::new();
    let int_node = Node::const_int(42);

    let mut target = Graph::new();
    let new_id = clone_node(&int_node, &mut target, |id| id);

    assert_eq!(target.node(new_id).as_int(), Some(42));
}
