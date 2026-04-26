use super::*;
use crate::ir::builder::GraphBuilder;
use crate::opt::unroll::analysis::LoopTripCount;
use rustc_hash::FxHashSet;

fn make_test_analysis() -> UnrollabilityAnalysis {
    UnrollabilityAnalysis {
        loop_idx: 0,
        trip_count: LoopTripCount::Constant(4),
        body_size: 5,
        has_single_entry: true,
        has_single_exit: true,
        contains_calls: false,
        has_memory_effects: false,
        has_early_exits: false,
        nesting_depth: 0,
        induction_vars: vec![NodeId::new(2)],
        register_pressure: 3,
        is_canonical: true,
        body_nodes: FxHashSet::default(),
    }
}

// =========================================================================
// LoopUnroller Tests
// =========================================================================

#[test]
fn test_loop_unroller_new() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let unroller = LoopUnroller::new(&mut graph, &mut cfg);
    assert_eq!(unroller.nodes_added(), 0);
    assert_eq!(unroller.nodes_removed(), 0);
}

#[test]
fn test_loop_unroller_full_unroll_zero_trip() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let analysis = make_test_analysis();
    let mut unroller = LoopUnroller::new(&mut graph, &mut cfg);

    let result = unroller.full_unroll(&analysis, 0);
    assert!(result); // Zero-trip elimination is a valid transformation
}

#[test]
fn test_loop_unroller_full_unroll_empty_body() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let analysis = make_test_analysis();
    let mut unroller = LoopUnroller::new(&mut graph, &mut cfg);

    let result = unroller.full_unroll(&analysis, 4);
    assert!(!result); // Empty body returns false
}

#[test]
fn test_loop_unroller_partial_unroll_factor_1() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let analysis = make_test_analysis();
    let mut unroller = LoopUnroller::new(&mut graph, &mut cfg);

    let result = unroller.partial_unroll(&analysis, 1, RemainderStrategy::None);
    assert!(!result); // Factor 1 is not worth it
}

#[test]
fn test_loop_unroller_clone_body() {
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder};

    let mut builder = GraphBuilder::new(4, 1);
    let p0 = builder.parameter(0).unwrap();
    let c1 = builder.const_int(1);
    let sum = builder.int_add(p0, c1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let body_nodes = vec![p0, c1, sum];
    let mut unroller = LoopUnroller::new(&mut graph, &mut cfg);

    let clone_map = unroller.clone_body(&body_nodes);
    assert_eq!(clone_map.len(), 3);
    assert!(unroller.nodes_added() >= 3);
}

// =========================================================================
// UnrollTransform Tests
// =========================================================================

#[test]
fn test_unroll_transform_full() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let analysis = make_test_analysis();
    let transform = UnrollTransform::new(&mut graph, &mut cfg, &analysis);

    // This will return false because body_nodes is empty
    let result = transform.full_unroll(4);
    assert!(!result);
}

#[test]
fn test_unroll_transform_partial() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let analysis = make_test_analysis();
    let transform = UnrollTransform::new(&mut graph, &mut cfg, &analysis);

    let result = transform.partial_unroll(4, RemainderStrategy::None);
    assert!(!result); // Empty body
}

#[test]
fn test_unroll_transform_runtime() {
    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let analysis = make_test_analysis();
    let transform = UnrollTransform::new(&mut graph, &mut cfg, &analysis);

    let result = transform.runtime_unroll(8, 4, RemainderStrategy::EpilogLoop);
    assert!(!result); // Empty body
}

// =========================================================================
// Node Copying Tests
// =========================================================================

#[test]
fn test_clone_preserves_operator() {
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder};

    let mut builder = GraphBuilder::new(4, 1);
    let p0 = builder.parameter(0).unwrap();
    let c1 = builder.const_int(42);
    let sum = builder.int_add(p0, c1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    // Get original ops
    let orig_const_op = graph.get(c1).unwrap().op.clone();
    let orig_add_op = graph.get(sum).unwrap().op.clone();

    let body_nodes = vec![c1, sum];
    let mut unroller = LoopUnroller::new(&mut graph, &mut cfg);
    let clone_map = unroller.clone_body(&body_nodes);

    // Check cloned operators match
    let cloned_c1 = clone_map.get(&c1).unwrap();
    let cloned_sum = clone_map.get(&sum).unwrap();

    assert_eq!(graph.get(*cloned_c1).unwrap().op, orig_const_op);
    assert_eq!(graph.get(*cloned_sum).unwrap().op, orig_add_op);
}

#[test]
fn test_clone_updates_inputs() {
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder};

    let mut builder = GraphBuilder::new(4, 1);
    let _p0 = builder.parameter(0).unwrap();
    let c1 = builder.const_int(1);
    let c2 = builder.const_int(2);
    let sum = builder.int_add(c1, c2);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    // Clone only c1, c2, and sum (not p0)
    let body_nodes = vec![c1, c2, sum];
    let mut unroller = LoopUnroller::new(&mut graph, &mut cfg);
    let clone_map = unroller.clone_body(&body_nodes);

    let cloned_sum = clone_map.get(&sum).unwrap();
    let cloned_c1 = clone_map.get(&c1).unwrap();
    let cloned_c2 = clone_map.get(&c2).unwrap();

    // Cloned sum should use cloned c1 and c2
    let sum_node = graph.get(*cloned_sum).unwrap();
    assert!(sum_node.inputs.iter().any(|x| x == *cloned_c1));
    assert!(sum_node.inputs.iter().any(|x| x == *cloned_c2));
}
