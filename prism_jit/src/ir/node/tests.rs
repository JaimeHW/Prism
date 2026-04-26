use super::*;
use crate::ir::operators::ArithOp;

#[test]
fn test_input_list_empty() {
    let list = InputList::empty();
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());
}

#[test]
fn test_input_list_single() {
    let id = NodeId::new(42);
    let list = InputList::Single(id);
    assert_eq!(list.len(), 1);
    assert_eq!(list.get(0), Some(id));
    assert_eq!(list.get(1), None);
}

#[test]
fn test_input_list_from_slice() {
    let ids: Vec<NodeId> = (0..5).map(|i| NodeId::new(i)).collect();
    let list = InputList::from_slice(&ids);
    assert_eq!(list.len(), 5);

    for (i, id) in list.iter().enumerate() {
        assert_eq!(id.index() as usize, i);
    }
}

#[test]
fn test_input_list_push() {
    let mut list = InputList::empty();

    list.push(NodeId::new(1));
    assert_eq!(list.len(), 1);

    list.push(NodeId::new(2));
    assert_eq!(list.len(), 2);

    list.push(NodeId::new(3));
    list.push(NodeId::new(4));
    list.push(NodeId::new(5));
    assert_eq!(list.len(), 5);

    // Should now be Many variant
    matches!(list, InputList::Many(_));
}

#[test]
fn test_node_const_int() {
    let node = Node::const_int(42);
    assert!(node.is_constant());
    assert_eq!(node.as_int(), Some(42));
    assert_eq!(node.ty, ValueType::Int64);
}

#[test]
fn test_node_const_float() {
    let node = Node::const_float(3.125);
    assert!(node.is_constant());
    assert_eq!(node.as_float(), Some(3.125));
    assert_eq!(node.ty, ValueType::Float64);
}

#[test]
fn test_node_arith() {
    let node = Node::new(
        Operator::IntOp(ArithOp::Add),
        InputList::Pair(NodeId::new(0), NodeId::new(1)),
    );
    assert!(!node.is_constant());
    assert!(node.is_pure());
    assert_eq!(node.inputs.len(), 2);
}

#[test]
fn test_node_flags() {
    let mut node = Node::const_int(0);
    assert!(!node.is_dead());

    node.mark_dead();
    assert!(node.is_dead());
}
