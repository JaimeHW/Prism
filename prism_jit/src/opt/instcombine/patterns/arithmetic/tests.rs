use super::*;
use crate::ir::node::InputList;

#[test]
fn test_add_zero_right() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(x, zero));

    let m = ArithmeticPatterns::try_match(&graph, add, &Operator::IntOp(ArithOp::Add));
    assert!(m.is_some());
    let m = m.unwrap();
    assert_eq!(m.replacement(), Some(x));
    assert_eq!(m.pattern_name(), "add_zero_right");
}

#[test]
fn test_add_zero_left() {
    let mut graph = Graph::new();
    let zero = graph.const_int(0);
    let x = graph.const_int(42);
    let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(zero, x));

    let m = ArithmeticPatterns::try_match(&graph, add, &Operator::IntOp(ArithOp::Add));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(x));
}

#[test]
fn test_sub_zero() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let sub = graph.add_node(Operator::IntOp(ArithOp::Sub), InputList::Pair(x, zero));

    let m = ArithmeticPatterns::try_match(&graph, sub, &Operator::IntOp(ArithOp::Sub));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "sub_zero");
}

#[test]
fn test_mul_zero() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let mul = graph.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(x, zero));

    let m = ArithmeticPatterns::try_match(&graph, mul, &Operator::IntOp(ArithOp::Mul));
    assert!(m.is_some());
    let m = m.unwrap();
    assert_eq!(m.replacement(), Some(zero));
}

#[test]
fn test_mul_one() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let one = graph.const_int(1);
    let mul = graph.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(x, one));

    let m = ArithmeticPatterns::try_match(&graph, mul, &Operator::IntOp(ArithOp::Mul));
    assert!(m.is_some());
    let m = m.unwrap();
    assert_eq!(m.replacement(), Some(x));
}

#[test]
fn test_div_one() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let one = graph.const_int(1);
    let div = graph.add_node(Operator::IntOp(ArithOp::TrueDiv), InputList::Pair(x, one));

    let m = ArithmeticPatterns::try_match(&graph, div, &Operator::IntOp(ArithOp::TrueDiv));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(x));
}

#[test]
fn test_zero_div() {
    let mut graph = Graph::new();
    let zero = graph.const_int(0);
    let x = graph.const_int(42);
    let div = graph.add_node(Operator::IntOp(ArithOp::TrueDiv), InputList::Pair(zero, x));

    let m = ArithmeticPatterns::try_match(&graph, div, &Operator::IntOp(ArithOp::TrueDiv));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(zero));
}

#[test]
fn test_no_match() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let y = graph.const_int(10);
    let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(x, y));

    let m = ArithmeticPatterns::try_match(&graph, add, &Operator::IntOp(ArithOp::Add));
    assert!(m.is_none());
}

#[test]
fn test_is_zero_int() {
    let mut graph = Graph::new();
    let zero = graph.const_int(0);
    let one = graph.const_int(1);

    assert!(ArithmeticPatterns::is_zero(&graph, zero));
    assert!(!ArithmeticPatterns::is_zero(&graph, one));
}

#[test]
fn test_is_one_int() {
    let mut graph = Graph::new();
    let zero = graph.const_int(0);
    let one = graph.const_int(1);

    assert!(!ArithmeticPatterns::is_one(&graph, zero));
    assert!(ArithmeticPatterns::is_one(&graph, one));
}
