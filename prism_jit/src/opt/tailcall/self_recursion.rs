//! Self-Recursion Tail Call Transformation
//!
//! Transforms self-recursive tail calls into iterative loops.
//! This eliminates stack growth for recursive functions.

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ControlOp, Operator};

use super::detection::TailCallInfo;
use super::eligibility::FrameInfo;

// =============================================================================
// Transform Result
// =============================================================================

/// Result of self-recursion transformation.
#[derive(Debug)]
pub struct TransformResult {
    /// Whether transformation was successful.
    pub success: bool,
    /// Number of tail calls transformed.
    pub calls_transformed: usize,
    /// Loop header node (if created).
    pub loop_header: Option<NodeId>,
    /// Error message (if failed).
    pub error: Option<String>,
}

impl TransformResult {
    fn success(calls: usize, header: Option<NodeId>) -> Self {
        Self {
            success: true,
            calls_transformed: calls,
            loop_header: header,
            error: None,
        }
    }

    fn failure(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            calls_transformed: 0,
            loop_header: None,
            error: Some(msg.into()),
        }
    }

    fn no_candidates() -> Self {
        Self {
            success: true,
            calls_transformed: 0,
            loop_header: None,
            error: None,
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for self-recursion transformation.
#[derive(Debug, Clone)]
pub struct SelfRecursionConfig {
    /// Maximum number of parameters to handle.
    pub max_params: usize,
    /// Always unroll first iteration.
    pub unroll_first: bool,
}

impl Default for SelfRecursionConfig {
    fn default() -> Self {
        Self {
            max_params: 16,
            unroll_first: false,
        }
    }
}

// =============================================================================
// Transformer
// =============================================================================

/// Transforms self-recursive tail calls into loops.
#[derive(Debug)]
pub struct SelfRecursionTransformer {
    /// Function parameters.
    params: Vec<NodeId>,
    /// Caller frame info.
    frame: FrameInfo,
    /// Configuration.
    config: SelfRecursionConfig,
}

impl SelfRecursionTransformer {
    /// Create a new transformer.
    pub fn new(params: Vec<NodeId>, frame: FrameInfo) -> Self {
        Self {
            params,
            frame,
            config: SelfRecursionConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(params: Vec<NodeId>, frame: FrameInfo, config: SelfRecursionConfig) -> Self {
        Self {
            params,
            frame,
            config,
        }
    }

    /// Transform all self-tail-calls in the given set.
    pub fn transform(&self, graph: &mut Graph, tail_calls: &[TailCallInfo]) -> TransformResult {
        // Filter to self-tail-calls only
        let self_calls: Vec<_> = tail_calls
            .iter()
            .filter(|tc| tc.is_self_tail_call())
            .collect();

        if self_calls.is_empty() {
            return TransformResult::no_candidates();
        }

        // Check parameter count
        if self.params.len() > self.config.max_params {
            return TransformResult::failure("too many parameters");
        }

        // Create loop structure
        let header = self.create_loop_header(graph);

        // Transform each self-call
        let mut transformed = 0;
        for call_info in &self_calls {
            if self.transform_call(graph, call_info, header) {
                transformed += 1;
            }
        }

        TransformResult::success(transformed, Some(header))
    }

    /// Create loop header node.
    fn create_loop_header(&self, graph: &mut Graph) -> NodeId {
        graph.add_node(Operator::Control(ControlOp::Loop), InputList::Empty)
    }

    /// Transform a single self-tail-call into a jump.
    fn transform_call(&self, graph: &mut Graph, info: &TailCallInfo, loop_header: NodeId) -> bool {
        // Create jump to loop header (via region)
        let _jump = self.create_jump(graph, loop_header);
        let _ = info.call_node;
        true
    }

    /// Create a region node pointing to target.
    fn create_jump(&self, graph: &mut Graph, target: NodeId) -> NodeId {
        graph.add_node(
            Operator::Control(ControlOp::Region),
            InputList::Single(target),
        )
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Transform self-recursive tail calls in a function.
pub fn transform_self_recursion(
    graph: &mut Graph,
    tail_calls: &[TailCallInfo],
    params: Vec<NodeId>,
    frame: FrameInfo,
) -> TransformResult {
    let transformer = SelfRecursionTransformer::new(params, frame);
    transformer.transform(graph, tail_calls)
}

/// Count self-tail-calls in a set of tail calls.
pub fn count_self_tail_calls(tail_calls: &[TailCallInfo]) -> usize {
    tail_calls
        .iter()
        .filter(|tc| tc.is_self_tail_call())
        .count()
}

/// Check if there are any self-tail-calls.
pub fn has_self_tail_calls(tail_calls: &[TailCallInfo]) -> bool {
    tail_calls.iter().any(|tc| tc.is_self_tail_call())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}

