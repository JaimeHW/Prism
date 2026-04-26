use super::*;
use crate::ir::node::InputList;

#[test]
fn test_branch_true() {
    let mut graph = Graph::new();
    let cond = graph.add_node(Operator::ConstBool(true), InputList::Empty);
    let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

    let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "branch_true");
}

#[test]
fn test_branch_false() {
    let mut graph = Graph::new();
    let cond = graph.add_node(Operator::ConstBool(false), InputList::Empty);
    let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

    let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "branch_false");
}

#[test]
fn test_branch_int_true() {
    let mut graph = Graph::new();
    let cond = graph.const_int(1);
    let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

    let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "branch_true");
}

#[test]
fn test_branch_int_false() {
    let mut graph = Graph::new();
    let cond = graph.const_int(0);
    let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

    let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "branch_false");
}

#[test]
fn test_no_match_variable_condition() {
    let mut graph = Graph::new();
    // Non-constant condition (an allocation)
    let alloc = graph.add_node(
        Operator::Memory(crate::ir::operators::MemoryOp::Alloc),
        InputList::Empty,
    );
    let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(alloc));

    let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
    assert!(m.is_none());
}

#[test]
fn test_get_bool_constant_true() {
    let mut graph = Graph::new();
    let c = graph.add_node(Operator::ConstBool(true), InputList::Empty);

    assert_eq!(ControlPatterns::get_bool_constant(&graph, c), Some(true));
}

#[test]
fn test_get_bool_constant_false() {
    let mut graph = Graph::new();
    let c = graph.add_node(Operator::ConstBool(false), InputList::Empty);

    assert_eq!(ControlPatterns::get_bool_constant(&graph, c), Some(false));
}

#[test]
fn test_get_bool_constant_int() {
    let mut graph = Graph::new();
    let c = graph.const_int(5);

    assert_eq!(ControlPatterns::get_bool_constant(&graph, c), Some(true));
}
