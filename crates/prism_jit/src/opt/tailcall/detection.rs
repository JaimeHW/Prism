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
