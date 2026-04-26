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
mod tests;
