use super::*;

// =========================================================================
// TcoStats Tests
// =========================================================================

#[test]
fn test_stats_new() {
    let stats = TcoStats::new();
    assert_eq!(stats.tail_calls_found, 0);
    assert_eq!(stats.total_optimized(), 0);
}

#[test]
fn test_stats_total() {
    let mut stats = TcoStats::new();
    stats.self_recursion_transformed = 2;
    stats.mutual_recursion_optimized = 1;
    stats.sibling_calls_optimized = 3;
    assert_eq!(stats.total_optimized(), 6);
}

#[test]
fn test_stats_merge() {
    let mut a = TcoStats::new();
    a.self_recursion_transformed = 2;

    let mut b = TcoStats::new();
    b.sibling_calls_optimized = 3;

    a.merge(&b);
    assert_eq!(a.self_recursion_transformed, 2);
    assert_eq!(a.sibling_calls_optimized, 3);
}

// =========================================================================
// ValueRemap Tests
// =========================================================================

#[test]
fn test_remap_new() {
    let remap = ValueRemap::new();
    assert!(remap.is_empty());
}

#[test]
fn test_remap_add_get() {
    let mut remap = ValueRemap::new();
    let old = NodeId::new(1);
    let new = NodeId::new(2);

    remap.add(old, new);

    assert_eq!(remap.get(old), new);
    assert!(remap.contains(old));
}

#[test]
fn test_remap_get_unmapped() {
    let remap = ValueRemap::new();
    let node = NodeId::new(1);

    assert_eq!(remap.get(node), node);
}

#[test]
fn test_remap_slice() {
    let mut remap = ValueRemap::new();
    remap.add(NodeId::new(1), NodeId::new(10));
    remap.add(NodeId::new(2), NodeId::new(20));

    let mut slice = vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)];
    remap.remap_slice(&mut slice);

    assert_eq!(slice[0], NodeId::new(10));
    assert_eq!(slice[1], NodeId::new(20));
    assert_eq!(slice[2], NodeId::new(3));
}

// =========================================================================
// PhiBuilder Tests
// =========================================================================

#[test]
fn test_phi_builder_new() {
    let region = NodeId::new(1);
    let builder = PhiBuilder::new(region);
    assert!(builder.values.is_empty());
}

#[test]
fn test_phi_builder_add_incoming() {
    let region = NodeId::new(1);
    let mut builder = PhiBuilder::new(region);

    builder.add_incoming(NodeId::new(2));
    builder.add_incoming(NodeId::new(3));

    assert_eq!(builder.values.len(), 2);
}

#[test]
fn test_phi_builder_build() {
    let mut graph = Graph::new();
    let region = create_loop_header(&mut graph);

    let value = graph.const_int(42);

    let mut builder = PhiBuilder::new(region);
    builder.add_incoming(value);

    let phi = builder.build_phi(&mut graph);

    let phi_node = graph.node(phi);
    assert!(matches!(phi_node.op, Operator::Phi));
}

// =========================================================================
// Control Flow Helper Tests
// =========================================================================

#[test]
fn test_create_loop_header() {
    let mut graph = Graph::new();
    let header = create_loop_header(&mut graph);

    let node = graph.node(header);
    assert!(matches!(node.op, Operator::Control(ControlOp::Loop)));
}

#[test]
fn test_create_region() {
    let mut graph = Graph::new();
    let pred = create_loop_header(&mut graph);
    let region = create_region(&mut graph, &[pred]);

    let node = graph.node(region);
    assert!(matches!(node.op, Operator::Control(ControlOp::Region)));
}

// =========================================================================
// Argument Handling Tests
// =========================================================================

#[test]
fn test_extract_call_args_empty() {
    let mut graph = Graph::new();
    let call = graph.add_node(Operator::Call(CallKind::Direct), InputList::Empty);

    let args = extract_call_args(&graph, call);
    assert!(args.is_empty());
}

#[test]
fn test_count_call_args() {
    let mut graph = Graph::new();
    let arg1 = graph.const_int(1);
    let arg2 = graph.const_int(2);

    let call = graph.add_node(
        Operator::Call(CallKind::Direct),
        InputList::Pair(arg1, arg2),
    );

    assert_eq!(count_call_args(&graph, call), 2);
}

#[test]
fn test_count_call_args_none() {
    let graph = Graph::new();
    assert_eq!(count_call_args(&graph, NodeId::new(999)), 0);
}
