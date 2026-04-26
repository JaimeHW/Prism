use super::*;

fn make_call_node(graph: &mut Graph) -> NodeId {
    graph.add_node(Operator::Call(CallKind::Direct), InputList::Empty)
}

fn make_return_node(graph: &mut Graph, value: NodeId) -> NodeId {
    graph.add_node(
        Operator::Control(ControlOp::Return),
        InputList::Single(value),
    )
}

// =========================================================================
// TailCallStatus Tests
// =========================================================================

#[test]
fn test_status_is_optimizable() {
    assert!(TailCallStatus::TailPosition.is_optimizable());
    assert!(!TailCallStatus::ResultUsed.is_optimizable());
    assert!(!TailCallStatus::InExceptionHandler.is_optimizable());
}

// =========================================================================
// TailCallInfo Tests
// =========================================================================

#[test]
fn test_info_can_optimize() {
    let mut info = TailCallInfo::new(NodeId::new(1));
    info.status = TailCallStatus::TailPosition;
    assert!(info.can_optimize());

    info.status = TailCallStatus::ResultUsed;
    assert!(!info.can_optimize());
}

#[test]
fn test_info_is_self_tail_call() {
    let mut info = TailCallInfo::new(NodeId::new(1));
    info.is_self_call = true;
    info.status = TailCallStatus::TailPosition;
    assert!(info.is_self_tail_call());

    info.is_self_call = false;
    assert!(!info.is_self_tail_call());
}

// =========================================================================
// DetectionContext Tests
// =========================================================================

#[test]
fn test_context_new() {
    let ctx = DetectionContext::new();
    assert!(!ctx.preserve_debug);
}

#[test]
fn test_context_with_debug() {
    let ctx = DetectionContext::new().with_debug(true);
    assert!(ctx.preserve_debug);
}

// =========================================================================
// Simple Tail Position Tests
// =========================================================================

#[test]
fn test_direct_return_is_tail() {
    let mut graph = Graph::new();
    let call = make_call_node(&mut graph);
    let _ret = make_return_node(&mut graph, call);

    let ctx = DetectionContext::new();
    let detector = TailCallDetector::new(ctx);
    let info = detector.find_tail_calls(&graph);

    assert_eq!(info.len(), 1);
    assert_eq!(info[0].status, TailCallStatus::TailPosition);
}

#[test]
fn test_result_used_not_tail() {
    let mut graph = Graph::new();
    let call = make_call_node(&mut graph);

    // Use call result in an addition
    use crate::ir::operators::ArithOp;
    let const_1 = graph.const_int(1);
    let add = graph.add_node(
        Operator::IntOp(ArithOp::Add),
        InputList::Pair(call, const_1),
    );
    let _ret = make_return_node(&mut graph, add);

    let ctx = DetectionContext::new();
    let detector = TailCallDetector::new(ctx);
    let info = detector.find_tail_calls(&graph);

    assert_eq!(info.len(), 1);
    assert_eq!(info[0].status, TailCallStatus::ResultUsed);
}

// =========================================================================
// Exception Handler Tests
// =========================================================================

#[test]
fn test_in_exception_handler() {
    let mut graph = Graph::new();
    let call = make_call_node(&mut graph);
    let _ret = make_return_node(&mut graph, call);

    let mut ctx = DetectionContext::new();
    ctx.in_exception_region.insert(call);

    let detector = TailCallDetector::new(ctx);
    let info = detector.find_tail_calls(&graph);

    assert_eq!(info.len(), 1);
    assert_eq!(info[0].status, TailCallStatus::InExceptionHandler);
}

// =========================================================================
// Debug Frame Tests
// =========================================================================

#[test]
fn test_debug_frame_required() {
    let mut graph = Graph::new();
    let call = make_call_node(&mut graph);
    let _ret = make_return_node(&mut graph, call);

    let ctx = DetectionContext::new().with_debug(true);

    let detector = TailCallDetector::new(ctx);
    let info = detector.find_tail_calls(&graph);

    assert_eq!(info.len(), 1);
    assert_eq!(info[0].status, TailCallStatus::DebugFrameRequired);
}

// =========================================================================
// Return Node Finding Tests
// =========================================================================

#[test]
fn test_find_return_node() {
    let mut graph = Graph::new();
    let call = make_call_node(&mut graph);
    let ret = make_return_node(&mut graph, call);

    let ctx = DetectionContext::new();
    let detector = TailCallDetector::new(ctx);
    let info = detector.find_tail_calls(&graph);

    assert_eq!(info.len(), 1);
    assert_eq!(info[0].return_node, Some(ret));
}

// =========================================================================
// Convenience Function Tests
// =========================================================================

#[test]
fn test_find_tail_calls_fn() {
    let mut graph = Graph::new();
    let call = make_call_node(&mut graph);
    let _ret = make_return_node(&mut graph, call);

    let calls = find_tail_calls(&graph);
    assert_eq!(calls.len(), 1);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_no_calls_in_graph() {
    let graph = Graph::new();
    let calls = find_tail_calls(&graph);
    assert!(calls.is_empty());
}

#[test]
fn test_call_with_no_uses() {
    let mut graph = Graph::new();
    let _call = make_call_node(&mut graph);
    // Call with no uses - result discarded

    let calls = find_tail_calls(&graph);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].status, TailCallStatus::ResultUsed);
}
