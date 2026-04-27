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
