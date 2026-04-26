use super::*;
use crate::ir::graph::Graph;
use crate::ir::node::InputList;
use crate::ir::operators::{CallKind, ControlOp, Operator};
use crate::opt::tailcall::detection::TailCallStatus;

fn make_sibling_call_graph() -> (Graph, NodeId, NodeId) {
    let mut graph = Graph::new();
    let call = graph.add_node(Operator::Call(CallKind::Direct), InputList::Empty);
    let ret = graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::Single(call),
    );
    (graph, call, ret)
}

fn make_tail_call_info(call: NodeId, ret: NodeId, is_self: bool) -> TailCallInfo {
    TailCallInfo {
        call_node: call,
        is_self_call: is_self,
        status: TailCallStatus::TailPosition,
        return_node: Some(ret),
        arg_count: 0,
    }
}

// =========================================================================
// SiblingCallInfo Tests
// =========================================================================

#[test]
fn test_sibling_info_self_call_skipped() {
    let (_graph, call, ret) = make_sibling_call_graph();
    let tc = make_tail_call_info(call, ret, true);
    let frame = FrameInfo::new(64, 2);

    let info = SiblingCallInfo::analyze(tc, &frame, None);
    assert!(!info.can_optimize());
}

#[test]
fn test_sibling_info_eligible() {
    let (_graph, call, ret) = make_sibling_call_graph();
    let tc = make_tail_call_info(call, ret, false);
    let caller_frame = FrameInfo::new(64, 2);
    let callee_frame = FrameInfo::new(64, 2);

    let info = SiblingCallInfo::analyze(tc, &caller_frame, Some(&callee_frame));
    assert!(info.can_optimize());
}

// =========================================================================
// Config Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = SiblingConfig::default();
    assert_eq!(config.max_stack_adjustment, 128);
    assert!(!config.optimize_indirect);
}

// =========================================================================
// Optimizer Tests
// =========================================================================

#[test]
fn test_optimizer_analyze_empty() {
    let caller = FrameInfo::new(64, 2);
    let optimizer = SiblingCallOptimizer::new(caller);

    let candidates = optimizer.analyze_candidates(&[]);
    assert!(candidates.is_empty());
}

#[test]
fn test_optimizer_skip_self_calls() {
    let (_graph, call, ret) = make_sibling_call_graph();
    let tc = make_tail_call_info(call, ret, true);

    let caller = FrameInfo::new(64, 2);
    let optimizer = SiblingCallOptimizer::new(caller);

    let candidates = optimizer.analyze_candidates(&[tc]);
    assert!(candidates.is_empty());
}

#[test]
fn test_optimizer_optimize() {
    let (mut graph, call, ret) = make_sibling_call_graph();
    let tc = make_tail_call_info(call, ret, false);

    let caller = FrameInfo::new(64, 2);
    let callee = FrameInfo::new(64, 2);

    let mut optimizer = SiblingCallOptimizer::new(caller);
    optimizer.register_callee(call, callee);

    let result = optimizer.optimize(&mut graph, &[tc]);
    assert_eq!(result.calls_optimized, 1);
}

// =========================================================================
// Convenience Function Tests
// =========================================================================

#[test]
fn test_find_sibling_calls() {
    let call = NodeId::new(1);
    let ret = NodeId::new(2);

    let tc1 = make_tail_call_info(call, ret, false);
    let tc2 = make_tail_call_info(call, ret, true);
    let tc3 = make_tail_call_info(call, ret, false);
    let calls = [tc1, tc2, tc3];

    let siblings = find_sibling_calls(&calls);
    assert_eq!(siblings.len(), 2);
}

#[test]
fn test_count_sibling_calls() {
    let call = NodeId::new(1);
    let ret = NodeId::new(2);

    let tc1 = make_tail_call_info(call, ret, false);
    let tc2 = make_tail_call_info(call, ret, true);
    let calls = [tc1, tc2];

    assert_eq!(count_sibling_calls(&calls), 1);
}
