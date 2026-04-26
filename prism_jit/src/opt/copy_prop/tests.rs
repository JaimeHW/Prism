use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
use crate::ir::types::ValueType;

/// Helper to create a phi node directly in the graph
fn make_phi(graph: &mut Graph, region: NodeId, values: &[NodeId]) -> NodeId {
    let mut inputs = InputList::default();
    inputs.push(region);
    for &v in values {
        inputs.push(v);
    }
    graph.add_node_with_type(Operator::Phi, inputs, ValueType::Top)
}

/// Helper to create a loop phi node
fn make_loop_phi(
    graph: &mut Graph,
    loop_header: NodeId,
    initial: NodeId,
    back_edge: Option<NodeId>,
) -> NodeId {
    let mut inputs = InputList::default();
    inputs.push(loop_header);
    inputs.push(initial);
    if let Some(back) = back_edge {
        inputs.push(back);
    }
    graph.add_node_with_type(Operator::LoopPhi, inputs, ValueType::Int64)
}

// =========================================================================
// CopyPropStats Tests
// =========================================================================

#[test]
fn test_copy_prop_stats_default() {
    let stats = CopyPropStats::default();
    assert_eq!(stats.copies_found, 0);
    assert_eq!(stats.uses_rewritten, 0);
    assert_eq!(stats.copies_eliminated, 0);
    assert_eq!(stats.phis_simplified, 0);
}

// =========================================================================
// CopyProp Construction Tests
// =========================================================================

#[test]
fn test_copy_prop_new() {
    let pass = CopyProp::new();
    assert_eq!(pass.copies_found(), 0);
    assert_eq!(pass.uses_rewritten(), 0);
    assert!(!pass.aggressive);
}

#[test]
fn test_copy_prop_aggressive() {
    let pass = CopyProp::aggressive();
    assert!(pass.aggressive);
}

#[test]
fn test_copy_prop_name() {
    let pass = CopyProp::new();
    assert_eq!(pass.name(), "copy_prop");
}

// =========================================================================
// Empty Graph Tests
// =========================================================================

#[test]
fn test_copy_prop_empty_graph() {
    let builder = GraphBuilder::new(0, 0);
    let mut graph = builder.finish();

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(!changed);
    assert_eq!(pass.copies_found(), 0);
}

#[test]
fn test_copy_prop_no_copies() {
    let mut builder = GraphBuilder::new(4, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let initial_size = graph.len();

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(!changed);
    assert_eq!(graph.len(), initial_size);
}

// =========================================================================
// Single Input Phi Tests
// =========================================================================

#[test]
fn test_copy_prop_single_input_phi() {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let region = builder.control();

    let mut graph = builder.finish();

    // Create a phi with single value input
    let phi = make_phi(&mut graph, region, &[p0]);

    // Create an add using the phi
    let add = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(phi, phi),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().phis_simplified, 1);

    // Verify the add now uses p0 directly
    let add_node = graph.get(add).unwrap();
    assert!(add_node.inputs.iter().all(|i| i == p0));
}

// =========================================================================
// Uniform Phi Tests
// =========================================================================

#[test]
fn test_copy_prop_uniform_phi() {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let region = builder.control();

    let mut graph = builder.finish();

    // Phi where all value inputs are the same
    let phi = make_phi(&mut graph, region, &[p0, p0, p0]);

    // Use it
    let add = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(phi, phi),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().phis_simplified, 1);

    let add_node = graph.get(add).unwrap();
    assert!(add_node.inputs.iter().all(|i| i == p0));
}

// =========================================================================
// Transitive Chain Tests
// =========================================================================

#[test]
fn test_copy_prop_transitive_chain() {
    let mut builder = GraphBuilder::new(12, 2);
    let p0 = builder.parameter(0).unwrap();
    let region = builder.control();

    let mut graph = builder.finish();

    // Chain: phi1 = p0, phi2 = phi1, phi3 = phi2
    let phi1 = make_phi(&mut graph, region, &[p0]);
    let phi2 = make_phi(&mut graph, region, &[phi1]);
    let phi3 = make_phi(&mut graph, region, &[phi2]);

    // Use phi3
    let add = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(phi3, phi3),
        ValueType::Int64,
    );

    let mut pass = CopyProp::aggressive();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().copies_found, 3);

    // In aggressive mode, add should use p0
    let add_node = graph.get(add).unwrap();
    assert!(add_node.inputs.iter().all(|i| i == p0));
}

// =========================================================================
// LoopPhi Tests
// =========================================================================

#[test]
fn test_copy_prop_trivial_loop_phi() {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let entry = builder.control();
    let loop_head = builder.loop_header(entry);

    let mut graph = builder.finish();

    // Loop phi with only initial value
    let loop_phi = make_loop_phi(&mut graph, loop_head, p0, None);

    // Use it
    let add = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(loop_phi, loop_phi),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().phis_simplified, 1);

    let add_node = graph.get(add).unwrap();
    assert!(add_node.inputs.iter().all(|i| i == p0));
}

#[test]
fn test_copy_prop_loop_phi_same_back_edge() {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let entry = builder.control();
    let loop_head = builder.loop_header(entry);

    let mut graph = builder.finish();

    // Loop phi where initial == back edge
    let loop_phi = make_loop_phi(&mut graph, loop_head, p0, Some(p0));

    let add = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(loop_phi, loop_phi),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);

    let add_node = graph.get(add).unwrap();
    assert!(add_node.inputs.iter().all(|i| i == p0));
}

// =========================================================================
// Multiple Copies Tests
// =========================================================================

#[test]
fn test_copy_prop_multiple_independent_copies() {
    let mut builder = GraphBuilder::new(12, 4);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let region = builder.control();

    let mut graph = builder.finish();

    let phi_a = make_phi(&mut graph, region, &[p0]);
    let phi_b = make_phi(&mut graph, region, &[p1]);

    let sum = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(phi_a, phi_b),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().copies_found, 2);

    let sum_node = graph.get(sum).unwrap();
    let inputs: Vec<_> = sum_node.inputs.iter().collect();
    assert!(inputs.contains(&p0));
    assert!(inputs.contains(&p1));
}

// =========================================================================
// Preserves Non-Copies Tests
// =========================================================================

#[test]
fn test_copy_prop_preserves_non_copies() {
    let mut builder = GraphBuilder::new(12, 4);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let region = builder.control();

    let mut graph = builder.finish();

    // Real phi with different value inputs (not a copy)
    let real_phi = make_phi(&mut graph, region, &[p0, p1]);

    // Copy phi
    let copy_phi = make_phi(&mut graph, region, &[p0]);

    let sum = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(real_phi, copy_phi),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().copies_found, 1);

    let sum_node = graph.get(sum).unwrap();
    let inputs: Vec<_> = sum_node.inputs.iter().collect();
    assert!(inputs.contains(&real_phi));
    assert!(inputs.contains(&p0));
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_copy_prop_idempotent() {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();
    let region = builder.control();

    let mut graph = builder.finish();

    let phi = make_phi(&mut graph, region, &[p0]);
    graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(phi, phi),
        ValueType::Int64,
    );

    // First run
    let mut pass = CopyProp::new();
    let changed1 = pass.run(&mut graph);
    assert!(changed1);

    // Second run should be no-op
    let mut pass2 = CopyProp::new();
    let changed2 = pass2.run(&mut graph);
    assert!(!changed2);
}

// =========================================================================
// Projection Tests
// =========================================================================

#[test]
fn test_copy_prop_projection_single_input() {
    let mut builder = GraphBuilder::new(8, 2);
    let p0 = builder.parameter(0).unwrap();

    let mut graph = builder.finish();

    // Projection(0) with single input
    let projected = graph.add_node_with_type(
        Operator::Projection(0),
        InputList::Single(p0),
        ValueType::Int64,
    );

    let result = graph.add_node_with_type(
        Operator::IntOp(crate::ir::operators::ArithOp::Add),
        InputList::Pair(projected, projected),
        ValueType::Int64,
    );

    let mut pass = CopyProp::new();
    let changed = pass.run(&mut graph);

    assert!(changed);
    assert_eq!(pass.stats().copies_found, 1);

    let result_node = graph.get(result).unwrap();
    assert!(result_node.inputs.iter().all(|i| i == p0));
}
