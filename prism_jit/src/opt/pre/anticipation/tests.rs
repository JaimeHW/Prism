use super::*;
use crate::ir::node::InputList;
use crate::ir::operators::ArithOp;

#[test]
fn test_anticipation_empty_graph() {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    assert_eq!(antic.nodes_with_anticipation(), 0);
}

#[test]
fn test_anticipation_single_constant() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    // The constant should be anticipated where it's defined
    let expr_id = expr_table.get_expr_id(c).unwrap();
    assert!(antic.is_anticipated(c, expr_id));
}

#[test]
fn test_anticipation_arithmetic() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);
    let sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    // Sum should be anticipated at its definition
    let sum_expr = expr_table.get_expr_id(sum).unwrap();
    assert!(antic.is_anticipated(sum, sum_expr));
}

#[test]
fn test_anticipation_not_anticipated_wrong_node() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);

    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    // Expression for 'a' should not be anticipated at 'b'
    let a_expr = expr_table.get_expr_id(a).unwrap();
    assert!(!antic.is_anticipated(b, a_expr));
}

#[test]
fn test_anticipation_nodes_with_anticipation() {
    let mut graph = Graph::new();
    graph.const_int(1);
    graph.const_int(2);

    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    assert!(antic.nodes_with_anticipation() >= 2);
}

#[test]
fn test_anticipation_at_empty() {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    let set = antic.anticipated_at(NodeId::new(0));
    assert!(set.is_empty());
}

#[test]
fn test_anticipation_out_of_bounds() {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    // Out of bounds node should return empty
    assert!(!antic.is_anticipated(NodeId::new(1000), ExprId::new(0)));
}

#[test]
fn test_blocks_anticipation_if() {
    assert!(AnticipationAnalysis::blocks_anticipation(
        &Operator::Control(ControlOp::If)
    ));
}

#[test]
fn test_blocks_anticipation_throw() {
    assert!(AnticipationAnalysis::blocks_anticipation(
        &Operator::Control(ControlOp::Throw)
    ));
}

#[test]
fn test_blocks_anticipation_arithmetic() {
    assert!(!AnticipationAnalysis::blocks_anticipation(
        &Operator::IntOp(ArithOp::Add)
    ));
}

#[test]
fn test_anticipated_at_returns_set() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);

    let set = antic.anticipated_at(c);
    assert!(!set.is_empty());
}
