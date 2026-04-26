use super::*;
use crate::ir::node::InputList;

#[test]
fn test_and_zero() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let and = graph.add_node(Operator::Bitwise(BitwiseOp::And), InputList::Pair(x, zero));

    let m = BitwisePatterns::try_match(&graph, and, &Operator::Bitwise(BitwiseOp::And));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(zero));
}

#[test]
fn test_and_all_ones() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let all_ones = graph.const_int(-1);
    let and = graph.add_node(
        Operator::Bitwise(BitwiseOp::And),
        InputList::Pair(x, all_ones),
    );

    let m = BitwisePatterns::try_match(&graph, and, &Operator::Bitwise(BitwiseOp::And));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(x));
}

#[test]
fn test_and_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let and = graph.add_node(Operator::Bitwise(BitwiseOp::And), InputList::Pair(x, x));

    let m = BitwisePatterns::try_match(&graph, and, &Operator::Bitwise(BitwiseOp::And));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "and_self");
}

#[test]
fn test_or_zero() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let or = graph.add_node(Operator::Bitwise(BitwiseOp::Or), InputList::Pair(x, zero));

    let m = BitwisePatterns::try_match(&graph, or, &Operator::Bitwise(BitwiseOp::Or));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(x));
}

#[test]
fn test_or_all_ones() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let all_ones = graph.const_int(-1);
    let or = graph.add_node(
        Operator::Bitwise(BitwiseOp::Or),
        InputList::Pair(x, all_ones),
    );

    let m = BitwisePatterns::try_match(&graph, or, &Operator::Bitwise(BitwiseOp::Or));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(all_ones));
}

#[test]
fn test_or_self() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let or = graph.add_node(Operator::Bitwise(BitwiseOp::Or), InputList::Pair(x, x));

    let m = BitwisePatterns::try_match(&graph, or, &Operator::Bitwise(BitwiseOp::Or));
    assert!(m.is_some());
    assert_eq!(m.unwrap().pattern_name(), "or_self");
}

#[test]
fn test_xor_zero() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let xor = graph.add_node(Operator::Bitwise(BitwiseOp::Xor), InputList::Pair(x, zero));

    let m = BitwisePatterns::try_match(&graph, xor, &Operator::Bitwise(BitwiseOp::Xor));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(x));
}

#[test]
fn test_shift_zero() {
    let mut graph = Graph::new();
    let x = graph.const_int(42);
    let zero = graph.const_int(0);
    let shl = graph.add_node(Operator::Bitwise(BitwiseOp::Shl), InputList::Pair(x, zero));

    let m = BitwisePatterns::try_match(&graph, shl, &Operator::Bitwise(BitwiseOp::Shl));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(x));
}

#[test]
fn test_zero_shift() {
    let mut graph = Graph::new();
    let zero = graph.const_int(0);
    let x = graph.const_int(5);
    let shl = graph.add_node(Operator::Bitwise(BitwiseOp::Shl), InputList::Pair(zero, x));

    let m = BitwisePatterns::try_match(&graph, shl, &Operator::Bitwise(BitwiseOp::Shl));
    assert!(m.is_some());
    assert_eq!(m.unwrap().replacement(), Some(zero));
}

#[test]
fn test_is_zero() {
    let mut graph = Graph::new();
    let zero = graph.const_int(0);
    let one = graph.const_int(1);

    assert!(BitwisePatterns::is_zero(&graph, zero));
    assert!(!BitwisePatterns::is_zero(&graph, one));
}

#[test]
fn test_is_all_ones() {
    let mut graph = Graph::new();
    let all_ones = graph.const_int(-1);
    let zero = graph.const_int(0);

    assert!(BitwisePatterns::is_all_ones(&graph, all_ones));
    assert!(!BitwisePatterns::is_all_ones(&graph, zero));
}
