//! Tail Call Optimization Pass
//!
//! This module provides comprehensive tail call optimization (TCO) for the
//! Sea-of-Nodes IR. TCO eliminates stack growth for tail-recursive calls
//! by transforming them into jumps or loops.
//!
//! # Optimization Types
//!
//! 1. **Self-Recursion**: Tail calls to the same function become loops
//! 2. **Mutual Recursion**: Tail calls between related functions use trampolines
//! 3. **Sibling Calls**: Tail calls to other functions become jumps

pub mod detection;
pub mod eligibility;
pub mod mutual;
pub mod self_recursion;
pub mod sibling;
pub mod transform;

pub use detection::{
    DetectionContext, TailCallDetector, TailCallInfo, TailCallStatus, find_self_tail_calls,
    find_tail_calls,
};
pub use eligibility::{
    CallingConvention, Eligibility, EligibilityAnalyzer, EligibilityConfig, FrameInfo,
    analyze_batch, filter_eligible,
};
pub use mutual::{
    CallGraph, MutualRecursionConfig, MutualRecursionTransformer, MutualTransformResult, SccInfo,
    Trampoline, TrampolineState, build_call_graph, find_mutual_recursion,
};
pub use self_recursion::{
    SelfRecursionConfig, SelfRecursionTransformer, TransformResult, count_self_tail_calls,
    has_self_tail_calls, transform_self_recursion,
};
pub use sibling::{
    SiblingCallInfo, SiblingCallOptimizer, SiblingConfig, SiblingOptResult, count_sibling_calls,
    find_sibling_calls, optimize_sibling_calls,
};
pub use transform::{
    PhiBuilder, TcoStats, ValueRemap, count_call_args, create_loop_header, create_region,
    extract_call_args,
};

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::opt::OptimizationPass;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for tail call optimization.
#[derive(Debug, Clone)]
pub struct TcoConfig {
    /// Enable self-recursion optimization.
    pub enable_self_recursion: bool,
    /// Enable mutual recursion optimization.
    pub enable_mutual_recursion: bool,
    /// Enable sibling call optimization.
    pub enable_sibling_calls: bool,
    /// Maximum parameters for self-recursion.
    pub max_self_params: usize,
    /// Maximum SCC size for mutual recursion.
    pub max_mutual_scc: usize,
    /// Preserve debug info.
    pub preserve_debug_info: bool,
}

impl Default for TcoConfig {
    fn default() -> Self {
        Self {
            enable_self_recursion: true,
            enable_mutual_recursion: true,
            enable_sibling_calls: true,
            max_self_params: 16,
            max_mutual_scc: 8,
            preserve_debug_info: false,
        }
    }
}

impl TcoConfig {
    /// Configuration that only optimizes self-recursion.
    pub fn self_only() -> Self {
        Self {
            enable_mutual_recursion: false,
            enable_sibling_calls: false,
            ..Self::default()
        }
    }

    /// Conservative configuration.
    pub fn conservative() -> Self {
        Self {
            max_self_params: 8,
            max_mutual_scc: 4,
            preserve_debug_info: true,
            ..Self::default()
        }
    }

    /// Aggressive configuration.
    pub fn aggressive() -> Self {
        Self {
            max_self_params: 32,
            max_mutual_scc: 16,
            preserve_debug_info: false,
            ..Self::default()
        }
    }
}

// =============================================================================
// Main Pass
// =============================================================================

/// Tail Call Optimization pass.
#[derive(Debug)]
pub struct TailCallOpt {
    /// Configuration.
    config: TcoConfig,
    /// Statistics from last run.
    stats: TcoStats,
}

impl TailCallOpt {
    /// Create a new TCO pass with default configuration.
    pub fn new() -> Self {
        Self::with_config(TcoConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: TcoConfig) -> Self {
        Self {
            config,
            stats: TcoStats::new(),
        }
    }

    /// Get statistics from the last optimization run.
    pub fn stats(&self) -> &TcoStats {
        &self.stats
    }

    /// Optimize a function for tail calls.
    pub fn optimize(&mut self, graph: &mut Graph, params: Vec<NodeId>) -> TcoStats {
        self.stats = TcoStats::new();

        // 1. Detect all tail calls
        let ctx = DetectionContext::new().with_debug(self.config.preserve_debug_info);
        let detector = TailCallDetector::new(ctx);
        let tail_calls = detector.find_tail_calls(graph);
        self.stats.tail_calls_found = tail_calls.len();

        if tail_calls.is_empty() {
            return self.stats.clone();
        }

        // 2. Self-recursion optimization
        if self.config.enable_self_recursion {
            self.optimize_self_recursion(graph, &tail_calls, &params);
        }

        // 3. Mutual recursion optimization
        if self.config.enable_mutual_recursion {
            self.optimize_mutual_recursion(&tail_calls);
        }

        // 4. Sibling call optimization
        if self.config.enable_sibling_calls {
            self.optimize_sibling_calls(graph, &tail_calls);
        }

        self.stats.clone()
    }

    /// Optimize self-recursive tail calls.
    fn optimize_self_recursion(
        &mut self,
        graph: &mut Graph,
        tail_calls: &[TailCallInfo],
        params: &[NodeId],
    ) {
        let config = SelfRecursionConfig {
            max_params: self.config.max_self_params,
            ..Default::default()
        };

        let frame = FrameInfo::new(64, params.len());
        let transformer = SelfRecursionTransformer::with_config(params.to_vec(), frame, config);
        let result = transformer.transform(graph, tail_calls);
        self.stats.self_recursion_transformed = result.calls_transformed;
    }

    /// Optimize mutually recursive tail calls.
    fn optimize_mutual_recursion(&mut self, tail_calls: &[TailCallInfo]) {
        let call_graph = build_call_graph(0, tail_calls);
        let config = MutualRecursionConfig {
            max_scc_size: self.config.max_mutual_scc,
            ..Default::default()
        };

        let transformer = MutualRecursionTransformer::with_config(config);
        let sccs = transformer.analyze(&call_graph);

        for scc in &sccs {
            if scc.is_mutual_recursion() {
                self.stats.mutual_recursion_optimized += 1;
            }
        }
    }

    /// Optimize sibling tail calls.
    fn optimize_sibling_calls(&mut self, graph: &mut Graph, tail_calls: &[TailCallInfo]) {
        let frame = FrameInfo::new(64, 2);
        let optimizer = SiblingCallOptimizer::new(frame);
        let result = optimizer.optimize(graph, tail_calls);
        self.stats.sibling_calls_optimized = result.calls_optimized;
        self.stats.not_optimized += result.skipped_calls;
    }
}

impl Default for TailCallOpt {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for TailCallOpt {
    fn name(&self) -> &'static str {
        "tailcall"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        let tail_calls = find_tail_calls(graph);
        if tail_calls.is_empty() {
            return false;
        }

        self.stats.tail_calls_found = tail_calls.len();

        if self.config.enable_sibling_calls {
            let frame = FrameInfo::new(64, 2);
            let optimizer = SiblingCallOptimizer::new(frame);
            let result = optimizer.optimize(graph, &tail_calls);
            if result.calls_optimized > 0 {
                self.stats.sibling_calls_optimized = result.calls_optimized;
                return true;
            }
        }

        false
    }
}
