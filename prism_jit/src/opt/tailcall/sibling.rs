//! Sibling Call Optimization
//!
//! Optimizes tail calls to other functions (not self) by converting
//! them to jumps when calling conventions are compatible.

use rustc_hash::FxHashMap;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CallKind, Operator};

use super::detection::TailCallInfo;
use super::eligibility::{Eligibility, EligibilityAnalyzer, FrameInfo};

// =============================================================================
// Sibling Call Info
// =============================================================================

/// Information about a sibling call candidate.
#[derive(Debug, Clone)]
pub struct SiblingCallInfo {
    /// The tail call info.
    pub tail_call: TailCallInfo,
    /// Eligibility status.
    pub eligibility: Eligibility,
    /// Whether to use jump optimization.
    pub use_jump: bool,
    /// Required stack adjustment (bytes).
    pub stack_adjustment: i32,
}

impl SiblingCallInfo {
    /// Create from tail call info with analysis.
    pub fn analyze(
        tail_call: TailCallInfo,
        caller_frame: &FrameInfo,
        callee_frame: Option<&FrameInfo>,
    ) -> Self {
        // Skip self calls - handled by self_recursion module
        if tail_call.is_self_call {
            return Self {
                tail_call,
                eligibility: Eligibility::NotTailPosition,
                use_jump: false,
                stack_adjustment: 0,
            };
        }

        let analyzer = EligibilityAnalyzer::new(caller_frame.clone());
        let eligibility = analyzer.analyze(&tail_call, callee_frame);

        let (use_jump, stack_adj) = if eligibility.is_eligible() {
            let adj = callee_frame
                .map(|f| f.size as i32 - caller_frame.size as i32)
                .unwrap_or(0);
            (true, adj)
        } else {
            (false, 0)
        };

        Self {
            tail_call,
            eligibility,
            use_jump,
            stack_adjustment: stack_adj,
        }
    }

    /// Check if this call can be optimized.
    pub fn can_optimize(&self) -> bool {
        self.eligibility.is_eligible() && self.use_jump && !self.tail_call.is_self_call
    }
}

// =============================================================================
// Sibling Call Optimizer
// =============================================================================

/// Configuration for sibling call optimization.
#[derive(Debug, Clone)]
pub struct SiblingConfig {
    /// Maximum stack adjustment allowed.
    pub max_stack_adjustment: usize,
    /// Whether to optimize indirect calls.
    pub optimize_indirect: bool,
}

impl Default for SiblingConfig {
    fn default() -> Self {
        Self {
            max_stack_adjustment: 128,
            optimize_indirect: false,
        }
    }
}

/// Result of sibling call optimization.
#[derive(Debug)]
pub struct SiblingOptResult {
    /// Number of calls optimized.
    pub calls_optimized: usize,
    /// Calls that couldn't be optimized.
    pub skipped_calls: usize,
    /// Reasons for skipping.
    pub skip_reasons: Vec<String>,
}

impl SiblingOptResult {
    fn new() -> Self {
        Self {
            calls_optimized: 0,
            skipped_calls: 0,
            skip_reasons: Vec::new(),
        }
    }
}

/// Optimizes sibling tail calls.
#[derive(Debug)]
pub struct SiblingCallOptimizer {
    /// Configuration.
    config: SiblingConfig,
    /// Caller's frame info.
    caller_frame: FrameInfo,
    /// Known callee frames.
    callee_frames: FxHashMap<NodeId, FrameInfo>,
}

impl SiblingCallOptimizer {
    /// Create a new optimizer.
    pub fn new(caller_frame: FrameInfo) -> Self {
        Self::with_config(caller_frame, SiblingConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(caller_frame: FrameInfo, config: SiblingConfig) -> Self {
        Self {
            config,
            caller_frame,
            callee_frames: FxHashMap::default(),
        }
    }

    /// Register a callee's frame info.
    pub fn register_callee(&mut self, call_node: NodeId, frame: FrameInfo) {
        self.callee_frames.insert(call_node, frame);
    }

    /// Analyze all sibling call candidates.
    pub fn analyze_candidates(&self, tail_calls: &[TailCallInfo]) -> Vec<SiblingCallInfo> {
        tail_calls
            .iter()
            .filter(|tc| !tc.is_self_call)
            .map(|tc| {
                let callee_frame = self.callee_frames.get(&tc.call_node);
                SiblingCallInfo::analyze(tc.clone(), &self.caller_frame, callee_frame)
            })
            .collect()
    }

    /// Optimize eligible sibling calls.
    pub fn optimize(&self, graph: &mut Graph, tail_calls: &[TailCallInfo]) -> SiblingOptResult {
        let candidates = self.analyze_candidates(tail_calls);
        let mut result = SiblingOptResult::new();

        for info in candidates {
            if !info.can_optimize() {
                result.skipped_calls += 1;
                result
                    .skip_reasons
                    .push(info.eligibility.description().to_string());
                continue;
            }

            if info.stack_adjustment.unsigned_abs() as usize > self.config.max_stack_adjustment {
                result.skipped_calls += 1;
                result
                    .skip_reasons
                    .push("stack adjustment too large".to_string());
                continue;
            }

            if self.transform_to_tail_call(graph, &info) {
                result.calls_optimized += 1;
            } else {
                result.skipped_calls += 1;
                result
                    .skip_reasons
                    .push("transformation failed".to_string());
            }
        }

        result
    }

    /// Transform a call to a tail call.
    fn transform_to_tail_call(&self, graph: &mut Graph, info: &SiblingCallInfo) -> bool {
        if let Some(node) = graph.get_mut(info.tail_call.call_node) {
            // Change Direct call to Tail call
            node.op = Operator::Call(CallKind::Tail);
            true
        } else {
            false
        }
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Find all sibling call candidates.
pub fn find_sibling_calls(tail_calls: &[TailCallInfo]) -> Vec<&TailCallInfo> {
    tail_calls.iter().filter(|tc| !tc.is_self_call).collect()
}

/// Count sibling calls.
pub fn count_sibling_calls(tail_calls: &[TailCallInfo]) -> usize {
    find_sibling_calls(tail_calls).len()
}

/// Optimize sibling calls in a function.
pub fn optimize_sibling_calls(
    graph: &mut Graph,
    tail_calls: &[TailCallInfo],
    caller_frame: FrameInfo,
) -> SiblingOptResult {
    let optimizer = SiblingCallOptimizer::new(caller_frame);
    optimizer.optimize(graph, tail_calls)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
