use super::*;
use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

#[test]
fn test_licm_new() {
    let licm = Licm::new();
    assert_eq!(licm.hoisted(), 0);
    assert_eq!(licm.loops_processed(), 0);
    assert!(!licm.aggressive);
}

#[test]
fn test_licm_aggressive() {
    let licm = Licm::aggressive();
    assert!(licm.aggressive);
}

#[test]
fn test_licm_name() {
    let licm = Licm::new();
    assert_eq!(licm.name(), "licm");
}

#[test]
fn test_licm_no_loops() {
    let mut builder = GraphBuilder::new(2, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    builder.return_value(sum);

    let mut graph = builder.finish();
    let mut licm = Licm::new();

    let changed = licm.run(&mut graph);
    assert!(!changed);
    assert_eq!(licm.hoisted(), 0);
}

#[test]
fn test_topological_sort_empty() {
    let builder = GraphBuilder::new(0, 0);
    let graph = builder.finish();

    let sorted = Licm::topological_sort(&graph, vec![]);
    assert!(sorted.is_empty());
}

#[test]
fn test_topological_sort_single() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let graph = builder.finish();

    let sorted = Licm::topological_sort(&graph, vec![p0]);
    assert_eq!(sorted.len(), 1);
    assert_eq!(sorted[0], p0);
}

#[test]
fn test_topological_sort_chain() {
    let mut builder = GraphBuilder::new(3, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let graph = builder.finish();

    // Sort should put p0, p1 before sum
    let sorted = Licm::topological_sort(&graph, vec![sum, p0, p1]);
    assert_eq!(sorted.len(), 3);
    // sum should be last (depends on p0, p1)
    assert_eq!(sorted[2], sum);
}

#[test]
fn test_licm_stats_default() {
    let stats = LicmStats::default();
    assert_eq!(stats.nodes_hoisted, 0);
    assert_eq!(stats.loops_analyzed, 0);
}

// -------------------------------------------------------------------------
// hoist_node Tests
// -------------------------------------------------------------------------

#[test]
fn test_hoist_node_sets_flags() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let mut graph = builder.finish();

    // Initially no flags
    assert!(!graph.node(sum).flags.contains(NodeFlags::LOOP_INVARIANT));
    assert!(!graph.node(sum).flags.contains(NodeFlags::HOISTED));

    let mut licm = Licm::new();
    let header = BlockId::new(0);
    let hoisted = licm.hoist_node(&mut graph, sum, header);

    assert!(hoisted);
    assert!(graph.node(sum).flags.contains(NodeFlags::LOOP_INVARIANT));
    assert!(graph.node(sum).flags.contains(NodeFlags::HOISTED));
}

#[test]
fn test_hoist_node_removes_in_loop_flag() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let mut graph = builder.finish();

    // Simulate node being inside a loop
    graph.node_mut(sum).flags.insert(NodeFlags::IN_LOOP);
    assert!(graph.node(sum).flags.contains(NodeFlags::IN_LOOP));

    let mut licm = Licm::new();
    let header = BlockId::new(0);
    let hoisted = licm.hoist_node(&mut graph, sum, header);

    assert!(hoisted);
    assert!(!graph.node(sum).flags.contains(NodeFlags::IN_LOOP));
}

#[test]
fn test_hoist_node_skips_already_hoisted() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let mut graph = builder.finish();

    // Mark as already hoisted
    graph.node_mut(sum).flags.insert(NodeFlags::HOISTED);

    let mut licm = Licm::new();
    let header = BlockId::new(0);
    let hoisted = licm.hoist_node(&mut graph, sum, header);

    // Should return false since already hoisted
    assert!(!hoisted);
}

#[test]
fn test_hoist_node_skips_pinned() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let mut graph = builder.finish();

    // Mark as pinned (cannot be moved)
    graph.node_mut(sum).flags.insert(NodeFlags::PINNED);

    let mut licm = Licm::new();
    let header = BlockId::new(0);
    let hoisted = licm.hoist_node(&mut graph, sum, header);

    // Should return false since pinned
    assert!(!hoisted);
    assert!(!graph.node(sum).flags.contains(NodeFlags::HOISTED));
}

#[test]
fn test_hoist_node_skips_dead() {
    let mut builder = GraphBuilder::new(2, 2);
    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let mut graph = builder.finish();

    // Mark as dead
    graph.node_mut(sum).flags.insert(NodeFlags::DEAD);

    let mut licm = Licm::new();
    let header = BlockId::new(0);
    let hoisted = licm.hoist_node(&mut graph, sum, header);

    // Should return false since dead
    assert!(!hoisted);
}
