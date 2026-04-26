use super::*;
use crate::ir::node::InputList;

#[test]
fn test_eq_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let eq = graph.add_node(Operator::IntCmp(CmpOp::Eq), InputList::Pair(x, x));

    let m = ComparisonPatterns::try_match(&graph, eq, &Operator::IntCmp(CmpOp::Eq));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "eq_self_true");
}

#[test]
fn test_ne_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let ne = graph.add_node(Operator::IntCmp(CmpOp::Ne), InputList::Pair(x, x));

    let m = ComparisonPatterns::try_match(&graph, ne, &Operator::IntCmp(CmpOp::Ne));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "ne_self_false");
}

#[test]
fn test_lt_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let lt = graph.add_node(Operator::IntCmp(CmpOp::Lt), InputList::Pair(x, x));

    let m = ComparisonPatterns::try_match(&graph, lt, &Operator::IntCmp(CmpOp::Lt));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "lt_gt_self_false");
}

#[test]
fn test_le_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let le = graph.add_node(Operator::IntCmp(CmpOp::Le), InputList::Pair(x, x));

    let m = ComparisonPatterns::try_match(&graph, le, &Operator::IntCmp(CmpOp::Le));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "le_ge_self_true");
}

#[test]
fn test_gt_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let gt = graph.add_node(Operator::IntCmp(CmpOp::Gt), InputList::Pair(x, x));

    let m = ComparisonPatterns::try_match(&graph, gt, &Operator::IntCmp(CmpOp::Gt));
    assert!(m.is_some());
}

#[test]
fn test_ge_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let ge = graph.add_node(Operator::IntCmp(CmpOp::Ge), InputList::Pair(x, x));

    let m = ComparisonPatterns::try_match(&graph, ge, &Operator::IntCmp(CmpOp::Ge));
    assert!(m.is_some());
}

#[test]
fn test_const_comparison() {
    let mut graph = Graph::new();
    let a = graph.const_int(5);
    let b = graph.const_int(10);
    let lt = graph.add_node(Operator::IntCmp(CmpOp::Lt), InputList::Pair(a, b));

    let m = ComparisonPatterns::try_match(&graph, lt, &Operator::IntCmp(CmpOp::Lt));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "const_fold");
}

#[test]
fn test_no_match_different_vars() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let y = graph.const_int(43);
    let eq = graph.add_node(Operator::IntCmp(CmpOp::Eq), InputList::Pair(x, y));

    let m = ComparisonPatterns::try_match(&graph, eq, &Operator::IntCmp(CmpOp::Eq));
    assert!(m.is_some());
}

#[test]
fn test_get_int_constant() {
    let mut graph = Graph::new();
    let c = graph.const_int(42);

    let val = ComparisonPatterns::get_int_constant(&graph, c);
    assert_eq!(val, Some(42));
}

#[test]
fn test_get_int_constant_not_const() {
    let graph = Graph::new();
    let val = ComparisonPatterns::get_int_constant(&graph, NodeId::new(1));
    assert_eq!(val, None);
}
