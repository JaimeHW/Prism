use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

// =========================================================================
// EdgeExecutability Tests
// =========================================================================

#[test]
fn test_edge_executability_new() {
    let e = EdgeExecutability::new();
    assert_eq!(e.edge_count(), 0);
}

#[test]
fn test_edge_executability_mark() {
    let mut e = EdgeExecutability::new();
    let n1 = NodeId::new(1);
    let n2 = NodeId::new(2);

    assert!(e.mark_executable(n1, n2));
    assert!(e.is_edge_executable(n1, n2));
    assert!(e.is_reachable(n2));

    // Marking again returns false
    assert!(!e.mark_executable(n1, n2));
}

#[test]
fn test_edge_executability_reachable() {
    let mut e = EdgeExecutability::new();
    let n1 = NodeId::new(1);

    assert!(e.mark_reachable(n1));
    assert!(e.is_reachable(n1));
    assert!(!e.mark_reachable(n1));
}

// =========================================================================
// SolverStats Tests
// =========================================================================

#[test]
fn test_solver_stats_default() {
    let s = SolverStats::default();
    assert_eq!(s.nodes_visited, 0);
    assert_eq!(s.value_changes, 0);
    assert_eq!(s.constants_found, 0);
    assert_eq!(s.unreachable_nodes, 0);
}

// =========================================================================
// Basic Solver Tests
// =========================================================================

#[test]
fn test_solver_empty_graph() {
    let builder = GraphBuilder::new(0, 0);
    let graph = builder.finish();

    let solver = SccpSolver::new(&graph);
    let result = solver.solve();

    assert_eq!(result.stats.constants_found, 0);
}

#[test]
fn test_solver_simple_constant() {
    let mut builder = GraphBuilder::new(8, 0);
    // x = 10 + 32
    let c10 = builder.const_int(10);
    let c32 = builder.const_int(32);
    let sum = builder.int_add(c10, c32);
    builder.return_value(sum);

    let graph = builder.finish();
    let solver = SccpSolver::new(&graph);
    let result = solver.solve();

    // The sum should be constant 42
    assert!(result.is_constant(sum));
    match result.constant_value(sum) {
        Some(Constant::Int(42)) => (),
        other => panic!("Expected Int(42), got {:?}", other),
    }
}

#[test]
fn test_solver_constant_chain() {
    let mut builder = GraphBuilder::new(12, 0);
    // a = 2
    // b = 3
    // c = a * b  (= 6)
    // d = c + 1  (= 7)
    let a = builder.const_int(2);
    let b = builder.const_int(3);
    let c = builder.int_mul(a, b);
    let c1 = builder.const_int(1);
    let d = builder.int_add(c, c1);
    builder.return_value(d);

    let graph = builder.finish();
    let solver = SccpSolver::new(&graph);
    let result = solver.solve();

    assert_eq!(result.constant_value(c), Some(&Constant::Int(6)));
    assert_eq!(result.constant_value(d), Some(&Constant::Int(7)));
}

#[test]
fn test_solver_with_parameter() {
    let mut builder = GraphBuilder::new(8, 1);
    // x = param0 + 10
    let p0 = builder.parameter(0).unwrap();
    let c10 = builder.const_int(10);
    let sum = builder.int_add(p0, c10);
    builder.return_value(sum);

    let graph = builder.finish();
    let solver = SccpSolver::new(&graph);
    let result = solver.solve();

    // sum cannot be constant (depends on parameter)
    assert!(!result.is_constant(sum));
}

#[test]
fn test_solver_result_constants_iter() {
    let mut builder = GraphBuilder::new(8, 0);
    let c1 = builder.const_int(1);
    let c2 = builder.const_int(2);
    let sum = builder.int_add(c1, c2);
    builder.return_value(sum);

    let graph = builder.finish();
    let result = SccpSolver::new(&graph).solve();

    let constants: Vec<_> = result.constants().collect();
    assert!(constants.len() >= 3); // c1, c2, sum
}

// =========================================================================
// SolverResult Tests
// =========================================================================

#[test]
fn test_solver_result_is_reachable() {
    let mut builder = GraphBuilder::new(4, 0);
    let c = builder.const_int(42);
    builder.return_value(c);

    let graph = builder.finish();
    let result = SccpSolver::new(&graph).solve();

    // Constants are reachable via initialization
    assert!(result.is_constant(c));
}
