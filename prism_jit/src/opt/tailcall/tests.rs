use super::*;
use crate::ir::graph::Graph;
use crate::ir::node::InputList;
use crate::ir::operators::{CallKind, ControlOp, Operator};

fn make_simple_tail_call_graph() -> (Graph, NodeId, Vec<NodeId>) {
    let mut graph = Graph::new();

    let param = graph.add_node(Operator::Parameter(0), InputList::Empty);
    let call = graph.add_node(Operator::Call(CallKind::Direct), InputList::Single(param));
    graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::Single(call),
    );

    (graph, call, vec![param])
}

// =========================================================================
// TcoConfig Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = TcoConfig::default();
    assert!(config.enable_self_recursion);
    assert!(config.enable_mutual_recursion);
    assert!(config.enable_sibling_calls);
}

#[test]
fn test_config_self_only() {
    let config = TcoConfig::self_only();
    assert!(config.enable_self_recursion);
    assert!(!config.enable_mutual_recursion);
    assert!(!config.enable_sibling_calls);
}

#[test]
fn test_config_conservative() {
    let config = TcoConfig::conservative();
    assert!(config.preserve_debug_info);
    assert_eq!(config.max_self_params, 8);
}

#[test]
fn test_config_aggressive() {
    let config = TcoConfig::aggressive();
    assert!(!config.preserve_debug_info);
    assert_eq!(config.max_self_params, 32);
}

// =========================================================================
// TailCallOpt Tests
// =========================================================================

#[test]
fn test_tco_new() {
    let tco = TailCallOpt::new();
    assert_eq!(tco.stats().tail_calls_found, 0);
}

#[test]
fn test_tco_with_config() {
    let config = TcoConfig::conservative();
    let tco = TailCallOpt::with_config(config);
    assert!(tco.config.preserve_debug_info);
}

#[test]
fn test_tco_optimize_empty() {
    let mut graph = Graph::new();
    let mut tco = TailCallOpt::new();

    let stats = tco.optimize(&mut graph, vec![]);

    assert_eq!(stats.tail_calls_found, 0);
    assert_eq!(stats.total_optimized(), 0);
}

#[test]
fn test_tco_detect_tail_call() {
    let (mut graph, _call, params) = make_simple_tail_call_graph();
    let mut tco = TailCallOpt::new();

    let stats = tco.optimize(&mut graph, params);

    assert!(stats.tail_calls_found > 0);
}

// =========================================================================
// OptimizationPass Tests
// =========================================================================

#[test]
fn test_pass_name() {
    let tco = TailCallOpt::new();
    assert_eq!(tco.name(), "tailcall");
}

#[test]
fn test_pass_run_empty() {
    let mut graph = Graph::new();
    let mut tco = TailCallOpt::new();

    let changed = tco.run(&mut graph);
    assert!(!changed);
}

#[test]
fn test_pass_run_with_call() {
    let (mut graph, _, _) = make_simple_tail_call_graph();
    let mut tco = TailCallOpt::new();

    let _changed = tco.run(&mut graph);
    assert!(tco.stats().tail_calls_found > 0);
}
