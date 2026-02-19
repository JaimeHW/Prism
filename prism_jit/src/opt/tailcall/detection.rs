//! Tail Position Detection for Tail Call Optimization
//!
//! This module analyzes call sites to determine if they are in tail position,
//! meaning the call's result is immediately returned without further processing.

use rustc_hash::FxHashSet;

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{CallKind, ControlOp, Operator};

// =============================================================================
// Tail Call Classification
// =============================================================================

/// Classification of a call site's tail position status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TailCallStatus {
    /// Call is in proper tail position.
    TailPosition,
    /// Result is used after the call (not tail position).
    ResultUsed,
    /// Call is wrapped in an exception handler.
    InExceptionHandler,
    /// Must preserve stack frame for debugging.
    DebugFrameRequired,
    /// Unknown - analysis incomplete.
    Unknown,
}

impl TailCallStatus {
    /// Check if this status allows tail call optimization.
    #[inline]
    pub fn is_optimizable(self) -> bool {
        self == TailCallStatus::TailPosition
    }
}

// =============================================================================
// Tail Call Info
// =============================================================================

/// Information about a potential tail call.
#[derive(Debug, Clone)]
pub struct TailCallInfo {
    /// The call node ID.
    pub call_node: NodeId,
    /// Whether this is a self-recursive call.
    pub is_self_call: bool,
    /// Tail position status.
    pub status: TailCallStatus,
    /// The return node that this call flows to.
    pub return_node: Option<NodeId>,
    /// Number of arguments.
    pub arg_count: usize,
}

impl TailCallInfo {
    /// Create new tail call info.
    pub fn new(call_node: NodeId) -> Self {
        Self {
            call_node,
            is_self_call: false,
            status: TailCallStatus::Unknown,
            return_node: None,
            arg_count: 0,
        }
    }

    /// Check if this call can be optimized to a jump.
    pub fn can_optimize(&self) -> bool {
        self.status.is_optimizable()
    }

    /// Check if this is a self-tail-call (can become a loop).
    pub fn is_self_tail_call(&self) -> bool {
        self.is_self_call && self.status.is_optimizable()
    }
}

// =============================================================================
// Detection Context
// =============================================================================

/// Context for tail call detection within a function.
#[derive(Debug)]
pub struct DetectionContext {
    /// Whether debug info preservation is required.
    pub preserve_debug: bool,
    /// Set of nodes known to be in exception handling regions.
    pub in_exception_region: FxHashSet<NodeId>,
}

impl DetectionContext {
    /// Create a new detection context.
    pub fn new() -> Self {
        Self {
            preserve_debug: false,
            in_exception_region: FxHashSet::default(),
        }
    }

    /// Enable debug frame preservation.
    pub fn with_debug(mut self, preserve: bool) -> Self {
        self.preserve_debug = preserve;
        self
    }
}

impl Default for DetectionContext {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tail Call Detector
// =============================================================================

/// Analyzes a function graph to find tail calls.
#[derive(Debug)]
pub struct TailCallDetector {
    /// Detection context.
    context: DetectionContext,
}

impl TailCallDetector {
    /// Create a new detector with context.
    pub fn new(context: DetectionContext) -> Self {
        Self { context }
    }

    /// Analyze the graph and find all tail calls.
    pub fn find_tail_calls(&self, graph: &Graph) -> Vec<TailCallInfo> {
        let mut results = Vec::new();

        // Iterate through nodes by collecting call nodes first
        let call_nodes: Vec<NodeId> = self.collect_call_nodes(graph);

        for node_id in call_nodes {
            let info = self.analyze_call_site(graph, node_id);
            results.push(info);
        }

        results
    }

    /// Collect all call nodes from the graph.
    fn collect_call_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        let mut calls = Vec::new();
        let len = graph.len();

        // Visit each node in graph using direct index
        for i in 0..len {
            let node_id = NodeId::new(i as u32);
            if let Some(node) = graph.get(node_id) {
                if matches!(node.op, Operator::Call(_)) {
                    calls.push(node_id);
                }
            }
        }

        calls
    }

    /// Check if a specific call is in tail position.
    pub fn is_tail_call(&self, graph: &Graph, call_node: NodeId) -> bool {
        if let Some(node) = graph.get(call_node) {
            if matches!(node.op, Operator::Call(_)) {
                let info = self.analyze_call_site(graph, call_node);
                return info.status.is_optimizable();
            }
        }
        false
    }

    /// Analyze a single call site.
    fn analyze_call_site(&self, graph: &Graph, call_node: NodeId) -> TailCallInfo {
        let mut info = TailCallInfo::new(call_node);

        // Count arguments
        if let Some(node) = graph.get(call_node) {
            info.arg_count = node.inputs.len().saturating_sub(1);
        }

        // Check for exception handlers
        if self.context.in_exception_region.contains(&call_node) {
            info.status = TailCallStatus::InExceptionHandler;
            return info;
        }

        // Check for debug frame requirement
        if self.context.preserve_debug {
            info.status = TailCallStatus::DebugFrameRequired;
            return info;
        }

        // Trace result usage to determine tail position
        let mut visited = FxHashSet::default();
        info.status = self.trace_uses_to_return(graph, call_node, &mut visited);

        // Find return node if in tail position
        if info.status == TailCallStatus::TailPosition {
            info.return_node = self.find_return_node(graph, call_node);
        }

        info
    }

    /// Recursively trace uses to find return path.
    fn trace_uses_to_return(
        &self,
        graph: &Graph,
        node_id: NodeId,
        visited: &mut FxHashSet<NodeId>,
    ) -> TailCallStatus {
        if !visited.insert(node_id) {
            return TailCallStatus::Unknown;
        }

        // Check uses of this node
        let uses = graph.uses(node_id);

        // If no uses, result is discarded
        if uses.is_empty() {
            return TailCallStatus::ResultUsed;
        }

        let mut has_return_path = false;
        let mut has_non_return_path = false;

        for &use_id in uses {
            let Some(use_node) = graph.get(use_id) else {
                continue;
            };

            match &use_node.op {
                Operator::Control(ControlOp::Return) => {
                    has_return_path = true;
                }
                Operator::Projection(_) => {
                    let status = self.trace_uses_to_return(graph, use_id, visited);
                    match status {
                        TailCallStatus::TailPosition => has_return_path = true,
                        TailCallStatus::Unknown => {}
                        _ => has_non_return_path = true,
                    }
                }
                _ => {
                    has_non_return_path = true;
                }
            }
        }

        if has_non_return_path {
            TailCallStatus::ResultUsed
        } else if has_return_path {
            TailCallStatus::TailPosition
        } else {
            TailCallStatus::Unknown
        }
    }

    /// Find the return node that this call flows to.
    fn find_return_node(&self, graph: &Graph, call_node: NodeId) -> Option<NodeId> {
        let mut visited = FxHashSet::default();
        self.find_return_recursive(graph, call_node, &mut visited)
    }

    fn find_return_recursive(
        &self,
        graph: &Graph,
        node_id: NodeId,
        visited: &mut FxHashSet<NodeId>,
    ) -> Option<NodeId> {
        if !visited.insert(node_id) {
            return None;
        }

        for &use_id in graph.uses(node_id) {
            let Some(use_node) = graph.get(use_id) else {
                continue;
            };

            if matches!(use_node.op, Operator::Control(ControlOp::Return)) {
                return Some(use_id);
            }

            if matches!(use_node.op, Operator::Projection(_)) {
                if let Some(ret) = self.find_return_recursive(graph, use_id, visited) {
                    return Some(ret);
                }
            }
        }

        None
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Find all tail calls in a function.
pub fn find_tail_calls(graph: &Graph) -> Vec<TailCallInfo> {
    let ctx = DetectionContext::new();
    let detector = TailCallDetector::new(ctx);
    detector.find_tail_calls(graph)
}

/// Find only self-tail-calls (for loop conversion).
pub fn find_self_tail_calls(graph: &Graph) -> Vec<TailCallInfo> {
    find_tail_calls(graph)
        .into_iter()
        .filter(|info| info.is_self_tail_call())
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
