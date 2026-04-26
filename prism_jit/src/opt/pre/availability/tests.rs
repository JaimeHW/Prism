use super::*;
use crate::ir::node::InputList;
use crate::ir::operators::ArithOp;

#[test]
fn test_availability_empty_graph() {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    assert_eq!(avail.nodes_with_availability(), 0);
}

#[test]
fn test_availability_single_constant() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    // The constant should be available where it's defined
    let expr_id = expr_table.get_expr_id(c).unwrap();
    assert!(avail.is_available(c, expr_id));
}

#[test]
fn test_availability_arithmetic() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);
    let sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    // Sum should be available at its definition
    let sum_expr = expr_table.get_expr_id(sum).unwrap();
    assert!(avail.is_available(sum, sum_expr));
}

#[test]
fn test_availability_propagates_forward() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);
    let sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    // Constants should be available at the sum node
    let a_expr = expr_table.get_expr_id(a).unwrap();
    let b_expr = expr_table.get_expr_id(b).unwrap();

    assert!(avail.is_available(sum, a_expr));
    assert!(avail.is_available(sum, b_expr));
}

#[test]
fn test_availability_nodes_with_availability() {
    let mut graph = Graph::new();
    graph.const_int(1);
    graph.const_int(2);

    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    assert!(avail.nodes_with_availability() >= 2);
}

#[test]
fn test_availability_at_empty() {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    let set = avail.available_at(NodeId::new(0));
    assert!(set.is_empty());
}

#[test]
fn test_availability_out_of_bounds() {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    assert!(!avail.is_available(NodeId::new(1000), ExprId::new(0)));
}

#[test]
fn test_kills_availability_store() {
    assert!(AvailabilityAnalysis::kills_availability(&Operator::Memory(
        MemoryOp::StoreField
    )));
}

#[test]
fn test_kills_availability_throw() {
    assert!(AvailabilityAnalysis::kills_availability(
        &Operator::Control(ControlOp::Throw)
    ));
}

#[test]
fn test_kills_availability_arithmetic() {
    assert!(!AvailabilityAnalysis::kills_availability(&Operator::IntOp(
        ArithOp::Add
    )));
}

#[test]
fn test_kills_availability_load() {
    // Loads don't kill availability
    assert!(!AvailabilityAnalysis::kills_availability(
        &Operator::Memory(MemoryOp::LoadField)
    ));
}

#[test]
fn test_available_at_returns_set() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    let expr_table = ExpressionTable::build(&graph);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

    let set = avail.available_at(c);
    assert!(!set.is_empty());
}
