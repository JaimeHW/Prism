use super::*;
use crate::opt::tailcall::detection::TailCallStatus;

fn make_self_tail_call() -> TailCallInfo {
    TailCallInfo {
        call_node: NodeId::new(1),
        is_self_call: true,
        status: TailCallStatus::TailPosition,
        return_node: None,
        arg_count: 2,
    }
}

fn make_non_self_tail_call() -> TailCallInfo {
    TailCallInfo {
        call_node: NodeId::new(2),
        is_self_call: false,
        status: TailCallStatus::TailPosition,
        return_node: None,
        arg_count: 2,
    }
}

// =========================================================================
// TransformResult Tests
// =========================================================================

#[test]
fn test_result_success() {
    let result = TransformResult::success(2, Some(NodeId::new(1)));
    assert!(result.success);
    assert_eq!(result.calls_transformed, 2);
}

#[test]
fn test_result_failure() {
    let result = TransformResult::failure("test error");
    assert!(!result.success);
    assert!(result.error.is_some());
}

#[test]
fn test_result_no_candidates() {
    let result = TransformResult::no_candidates();
    assert!(result.success);
    assert_eq!(result.calls_transformed, 0);
}

// =========================================================================
// Config Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = SelfRecursionConfig::default();
    assert_eq!(config.max_params, 16);
    assert!(!config.unroll_first);
}

// =========================================================================
// Transformer Tests
// =========================================================================

#[test]
fn test_transform_no_candidates() {
    let mut graph = Graph::new();
    let params = vec![];
    let frame = FrameInfo::new(64, 0);

    let calls = vec![make_non_self_tail_call()];
    let transformer = SelfRecursionTransformer::new(params, frame);
    let result = transformer.transform(&mut graph, &calls);

    assert!(result.success);
    assert_eq!(result.calls_transformed, 0);
}

#[test]
fn test_transform_self_call() {
    let mut graph = Graph::new();
    let params = vec![NodeId::new(100), NodeId::new(101)];
    let frame = FrameInfo::new(64, 2);

    let calls = vec![make_self_tail_call()];
    let transformer = SelfRecursionTransformer::new(params, frame);
    let result = transformer.transform(&mut graph, &calls);

    assert!(result.success);
    assert_eq!(result.calls_transformed, 1);
    assert!(result.loop_header.is_some());
}

#[test]
fn test_transform_too_many_params() {
    let mut graph = Graph::new();
    let params: Vec<NodeId> = (0..20).map(|i| NodeId::new(i)).collect();
    let frame = FrameInfo::new(64, 20);

    let config = SelfRecursionConfig {
        max_params: 8,
        ..Default::default()
    };

    let calls = vec![make_self_tail_call()];
    let transformer = SelfRecursionTransformer::with_config(params, frame, config);
    let result = transformer.transform(&mut graph, &calls);

    assert!(!result.success);
}

// =========================================================================
// Convenience Function Tests
// =========================================================================

#[test]
fn test_count_self_tail_calls() {
    let calls = vec![
        make_self_tail_call(),
        make_non_self_tail_call(),
        make_self_tail_call(),
    ];
    assert_eq!(count_self_tail_calls(&calls), 2);
}

#[test]
fn test_has_self_tail_calls() {
    let with_self = vec![make_self_tail_call()];
    let without_self = vec![make_non_self_tail_call()];

    assert!(has_self_tail_calls(&with_self));
    assert!(!has_self_tail_calls(&without_self));
}
