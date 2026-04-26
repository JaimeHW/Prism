use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

// =========================================================================
// Configuration Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = SccpConfig::default();
    assert!(config.eliminate_dead_code);
    assert!(config.fold_constants);
}

#[test]
fn test_config_aggressive() {
    let config = SccpConfig::aggressive();
    assert!(config.eliminate_dead_code);
    assert!(config.max_iterations > 1000);
}

#[test]
fn test_config_conservative() {
    let config = SccpConfig::conservative();
    assert!(!config.eliminate_dead_code);
    assert!(config.fold_constants);
}

// =========================================================================
// Basic SCCP Tests
// =========================================================================

#[test]
fn test_sccp_empty_graph() {
    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();

    let mut sccp = Sccp::new();
    let changed = sccp.run(&mut graph);

    assert!(!changed);
}

#[test]
fn test_sccp_constant_folding() {
    let mut builder = GraphBuilder::new(8, 0);
    // x = 10 + 20
    let c10 = builder.const_int(10);
    let c20 = builder.const_int(20);
    let sum = builder.int_add(c10, c20);
    builder.return_value(sum);

    let mut graph = builder.finish();

    // SCCP should fold 10 + 20 = 30
    let mut sccp = Sccp::new();
    let changed = sccp.run(&mut graph);

    assert!(changed);
    assert!(sccp.stats().constants_folded > 0);

    // The sum node should now be a constant
    if let Some(node) = graph.get(sum) {
        match &node.op {
            Operator::ConstInt(30) => (),
            other => panic!("Expected ConstInt(30), got {:?}", other),
        }
    }
}

#[test]
fn test_sccp_chained_constants() {
    let mut builder = GraphBuilder::new(12, 0);
    // a = 2 * 3 = 6
    // b = a + 4 = 10
    // c = b * 2 = 20
    let c2 = builder.const_int(2);
    let c3 = builder.const_int(3);
    let a = builder.int_mul(c2, c3);
    let c4 = builder.const_int(4);
    let b = builder.int_add(a, c4);
    let c2_2 = builder.const_int(2);
    let c = builder.int_mul(b, c2_2);
    builder.return_value(c);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    // All intermediate computations should be folded
    if let Some(node) = graph.get(c) {
        assert!(matches!(&node.op, Operator::ConstInt(20)));
    }
}

#[test]
fn test_sccp_preserves_parameters() {
    let mut builder = GraphBuilder::new(8, 1);
    // result = param0 + 10
    let p0 = builder.parameter(0).unwrap();
    let c10 = builder.const_int(10);
    let sum = builder.int_add(p0, c10);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    // Sum cannot be folded (parameter is unknown)
    if let Some(node) = graph.get(sum) {
        assert!(!Sccp::is_constant_op(&node.op));
    }
}

#[test]
fn test_sccp_comparison_folding() {
    let mut builder = GraphBuilder::new(8, 0);
    // x = 5 < 10 (= true)
    let c5 = builder.const_int(5);
    let c10 = builder.const_int(10);
    let cmp = builder.int_lt(c5, c10);
    builder.return_value(cmp);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    // Comparison should be folded to true
    if let Some(node) = graph.get(cmp) {
        assert!(matches!(&node.op, Operator::ConstBool(true)));
    }
}

#[test]
fn test_sccp_subtraction_folding() {
    let mut builder = GraphBuilder::new(8, 0);
    let c100 = builder.const_int(100);
    let c42 = builder.const_int(42);
    let diff = builder.int_sub(c100, c42);
    builder.return_value(diff);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(diff) {
        assert!(matches!(&node.op, Operator::ConstInt(58)));
    }
}

#[test]
fn test_sccp_division_folding() {
    let mut builder = GraphBuilder::new(8, 0);
    let c100 = builder.const_int(100);
    let c5 = builder.const_int(5);
    let quot = builder.int_div(c100, c5);
    builder.return_value(quot);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(quot) {
        assert!(matches!(&node.op, Operator::ConstInt(20)));
    }
}

// =========================================================================
// Statistics Tests
// =========================================================================

#[test]
fn test_sccp_stats() {
    let mut builder = GraphBuilder::new(8, 0);
    let c1 = builder.const_int(1);
    let c2 = builder.const_int(2);
    let c3 = builder.const_int(3);
    let sum1 = builder.int_add(c1, c2);
    let sum2 = builder.int_add(sum1, c3);
    builder.return_value(sum2);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    let stats = sccp.stats();
    assert!(stats.constants_folded >= 2);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_sccp_zero_operations() {
    let mut builder = GraphBuilder::new(8, 0);
    // x * 0 = 0
    let c42 = builder.const_int(42);
    let c0 = builder.const_int(0);
    let mul = builder.int_mul(c42, c0);
    builder.return_value(mul);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(mul) {
        assert!(matches!(&node.op, Operator::ConstInt(0)));
    }
}

#[test]
fn test_sccp_identity_operations() {
    let mut builder = GraphBuilder::new(8, 0);
    // x + 0 = x
    let c42 = builder.const_int(42);
    let c0 = builder.const_int(0);
    let sum = builder.int_add(c42, c0);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(sum) {
        assert!(matches!(&node.op, Operator::ConstInt(42)));
    }
}

#[test]
fn test_sccp_negative_numbers() {
    let mut builder = GraphBuilder::new(8, 0);
    let c_neg5 = builder.const_int(-5);
    let c_neg3 = builder.const_int(-3);
    let sum = builder.int_add(c_neg5, c_neg3);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(sum) {
        assert!(matches!(&node.op, Operator::ConstInt(-8)));
    }
}

// =========================================================================
// Pass Interface Tests
// =========================================================================

#[test]
fn test_sccp_pass_name() {
    let sccp = Sccp::new();
    assert_eq!(sccp.name(), "sccp");
}

#[test]
fn test_sccp_default() {
    let sccp = Sccp::default();
    assert_eq!(sccp.name(), "sccp");
}

#[test]
fn test_sccp_with_config() {
    let config = SccpConfig::aggressive();
    let sccp = Sccp::with_config(config);
    assert_eq!(sccp.name(), "sccp");
}

// =========================================================================
// Float Tests
// =========================================================================

#[test]
fn test_sccp_float_arithmetic() {
    let mut builder = GraphBuilder::new(8, 0);
    let f1 = builder.const_float(2.5);
    let f2 = builder.const_float(3.5);
    let sum = builder.float_add(f1, f2);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(sum) {
        match &node.op {
            Operator::ConstFloat(bits) => {
                let v = f64::from_bits(*bits);
                assert!((v - 6.0).abs() < f64::EPSILON);
            }
            other => panic!("Expected ConstFloat, got {:?}", other),
        }
    }
}

// =========================================================================
// Boolean Tests
// =========================================================================

#[test]
fn test_sccp_equality_comparison() {
    let mut builder = GraphBuilder::new(8, 0);
    let c5 = builder.const_int(5);
    let c5_2 = builder.const_int(5);
    let eq = builder.int_eq(c5, c5_2);
    builder.return_value(eq);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(eq) {
        assert!(matches!(&node.op, Operator::ConstBool(true)));
    }
}

#[test]
fn test_sccp_inequality_comparison() {
    let mut builder = GraphBuilder::new(8, 0);
    let c5 = builder.const_int(5);
    let c10 = builder.const_int(10);
    let ne = builder.int_ne(c5, c10);
    builder.return_value(ne);

    let mut graph = builder.finish();
    let mut sccp = Sccp::new();
    sccp.run(&mut graph);

    if let Some(node) = graph.get(ne) {
        assert!(matches!(&node.op, Operator::ConstBool(true)));
    }
}
