use super::*;
use crate::match_compiler::matrix::PatternRow;
use crate::match_compiler::pattern::{FlatPattern, LiteralValue};

fn literal_row(val: i64, action: usize) -> PatternRow {
    PatternRow {
        patterns: vec![FlatPattern::Literal(LiteralValue::Int(val))],
        bindings: vec![],
        guard: None,
        action,
    }
}

fn wildcard_row(action: usize) -> PatternRow {
    PatternRow {
        patterns: vec![FlatPattern::Wildcard],
        bindings: vec![],
        guard: None,
        action,
    }
}

#[test]
fn test_compile_empty() {
    let matrix = PatternMatrix::new(vec![]);
    let tree = compile(&matrix);
    assert!(matches!(tree, DecisionTree::Fail));
}

#[test]
fn test_compile_single_wildcard() {
    let matrix = PatternMatrix::new(vec![wildcard_row(0)]);
    let tree = compile(&matrix);

    if let DecisionTree::Leaf { action, guard, .. } = tree {
        assert_eq!(action, 0);
        assert!(guard.is_none());
    } else {
        panic!("Expected Leaf node");
    }
}

#[test]
fn test_compile_two_literals() {
    let matrix = PatternMatrix::new(vec![literal_row(1, 0), literal_row(2, 1)]);
    let tree = compile(&matrix);

    if let DecisionTree::Switch { cases, default, .. } = tree {
        assert_eq!(cases.len(), 2);
        assert!(default.is_none());
    } else {
        panic!("Expected Switch node");
    }
}

#[test]
fn test_compile_literal_with_default() {
    let matrix = PatternMatrix::new(vec![literal_row(1, 0), wildcard_row(1)]);
    let tree = compile(&matrix);

    if let DecisionTree::Switch { cases, default, .. } = tree {
        assert_eq!(cases.len(), 1);
        assert!(default.is_some());
    } else {
        panic!("Expected Switch node");
    }
}

#[test]
fn test_tree_node_count() {
    let matrix = PatternMatrix::new(vec![literal_row(1, 0), literal_row(2, 1), wildcard_row(2)]);
    let tree = compile(&matrix);

    // Should have: 1 switch + 2 leaf cases + 1 default leaf
    assert!(tree.node_count() >= 4);
}

#[test]
fn test_tree_max_depth() {
    let matrix = PatternMatrix::new(vec![wildcard_row(0)]);
    let tree = compile(&matrix);

    // Single leaf = depth 1
    assert_eq!(tree.max_depth(), 1);
}
