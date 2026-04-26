use super::*;
use crate::ir::node::InputList;
use crate::ir::operators::ArithOp;

// =========================================================================
// ExprId Tests
// =========================================================================

#[test]
fn test_expr_id_new() {
    let id = ExprId::new(42);
    assert_eq!(id.raw(), 42);
}

// =========================================================================
// Expression Tests
// =========================================================================

#[test]
fn test_expression_new() {
    let expr = Expression::new(Operator::ConstInt(42), vec![]);
    assert_eq!(expr.op, Operator::ConstInt(42));
    assert!(expr.inputs.is_empty());
}

#[test]
fn test_expression_with_inputs() {
    let inputs = vec![ExprId::new(0), ExprId::new(1)];
    let expr = Expression::new(Operator::IntOp(ArithOp::Add), inputs.clone());
    assert_eq!(expr.inputs, inputs);
}

// =========================================================================
// ExpressionTable Tests
// =========================================================================

#[test]
fn test_expression_table_new() {
    let table = ExpressionTable::new();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

#[test]
fn test_expression_table_build_empty() {
    let graph = Graph::new();
    let table = ExpressionTable::build(&graph);
    // Start/End nodes are not numberable
    assert!(table.is_empty());
}

#[test]
fn test_expression_table_build_constants() {
    let mut graph = Graph::new();
    graph.const_int(42);
    graph.const_int(42); // Duplicate
    graph.const_int(100);

    let table = ExpressionTable::build(&graph);
    // Should have 2 unique expressions
    assert_eq!(table.len(), 2);
}

#[test]
fn test_expression_table_build_arithmetic() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);
    let _sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

    let table = ExpressionTable::build(&graph);
    // 2 constants + 1 add
    assert_eq!(table.len(), 3);
}

#[test]
fn test_expression_table_deduplicates() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);
    let sum1 = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));
    let sum2 = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

    let table = ExpressionTable::build(&graph);

    // Both sums should have same expression ID
    let expr1 = table.get_expr_id(sum1);
    let expr2 = table.get_expr_id(sum2);
    assert_eq!(expr1, expr2);
}

#[test]
fn test_expression_table_get_nodes() {
    let mut graph = Graph::new();
    let a = graph.const_int(1);
    let b = graph.const_int(2);
    let sum1 = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));
    let sum2 = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

    let table = ExpressionTable::build(&graph);

    let expr_id = table.get_expr_id(sum1).unwrap();
    let nodes = table.get_nodes(expr_id);
    assert!(nodes.contains(&sum1));
    assert!(nodes.contains(&sum2));
}

#[test]
fn test_is_numberable_constants() {
    assert!(ExpressionTable::is_numberable(&Operator::ConstInt(42)));
    assert!(ExpressionTable::is_numberable(&Operator::ConstFloat(0)));
    assert!(ExpressionTable::is_numberable(&Operator::ConstBool(true)));
    assert!(ExpressionTable::is_numberable(&Operator::ConstNone));
}

#[test]
fn test_is_numberable_arithmetic() {
    assert!(ExpressionTable::is_numberable(&Operator::IntOp(
        ArithOp::Add
    )));
    assert!(ExpressionTable::is_numberable(&Operator::FloatOp(
        ArithOp::Mul
    )));
}

#[test]
fn test_is_numberable_control() {
    use crate::ir::operators::ControlOp;
    assert!(!ExpressionTable::is_numberable(&Operator::Control(
        ControlOp::Return
    )));
}

// =========================================================================
// PreStats Tests
// =========================================================================

#[test]
fn test_pre_stats_default() {
    let stats = PreStats::default();
    assert_eq!(stats.expressions_eliminated, 0);
    assert_eq!(stats.expressions_inserted, 0);
}

#[test]
fn test_pre_stats_net_reduction_positive() {
    let stats = PreStats {
        expressions_eliminated: 5,
        expressions_inserted: 2,
        expressions_analyzed: 10,
    };
    assert_eq!(stats.net_reduction(), 3);
}

#[test]
fn test_pre_stats_net_reduction_negative() {
    let stats = PreStats {
        expressions_eliminated: 2,
        expressions_inserted: 5,
        expressions_analyzed: 10,
    };
    assert_eq!(stats.net_reduction(), -3);
}

#[test]
fn test_pre_stats_merge() {
    let mut stats1 = PreStats {
        expressions_eliminated: 3,
        expressions_inserted: 1,
        expressions_analyzed: 10,
    };
    let stats2 = PreStats {
        expressions_eliminated: 2,
        expressions_inserted: 2,
        expressions_analyzed: 5,
    };
    stats1.merge(&stats2);
    assert_eq!(stats1.expressions_eliminated, 5);
    assert_eq!(stats1.expressions_inserted, 3);
    assert_eq!(stats1.expressions_analyzed, 15);
}

// =========================================================================
// PreConfig Tests
// =========================================================================

#[test]
fn test_pre_config_default() {
    let config = PreConfig::default();
    assert_eq!(config.max_code_growth, 1.5);
    assert_eq!(config.min_frequency_ratio, 0.5);
    assert_eq!(config.max_expr_size, 100);
}

// =========================================================================
// Pre Pass Tests
// =========================================================================

#[test]
fn test_pre_new() {
    let pre = Pre::new();
    assert_eq!(pre.stats().expressions_eliminated, 0);
}

#[test]
fn test_pre_with_config() {
    let config = PreConfig {
        max_code_growth: 2.0,
        min_frequency_ratio: 0.3,
        max_expr_size: 50,
    };
    let pre = Pre::with_config(config);
    assert_eq!(pre.config().max_code_growth, 2.0);
}

#[test]
fn test_pre_name() {
    let pre = Pre::new();
    assert_eq!(pre.name(), "pre");
}

#[test]
fn test_pre_default() {
    let pre = Pre::default();
    assert_eq!(pre.stats().expressions_analyzed, 0);
}

#[test]
fn test_pre_empty_graph() {
    let mut graph = Graph::new();
    let mut pre = Pre::new();
    let changed = pre.run(&mut graph);
    assert!(!changed);
}

#[test]
fn test_pre_stats_after_run() {
    let mut graph = Graph::new();
    graph.const_int(42);

    let mut pre = Pre::new();
    pre.run(&mut graph);

    assert!(pre.stats().expressions_analyzed >= 1);
}
