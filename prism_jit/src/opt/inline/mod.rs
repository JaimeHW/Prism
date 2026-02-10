//! Function Inlining Optimization Pass
//!
//! This module implements a production-quality function inlining system for the
//! Sea-of-Nodes IR. Inlining replaces call sites with the body of the called
//! function, eliminating call overhead and enabling cross-procedural optimization.
//!
//! # Architecture
//!
//! The inlining system consists of several components:
//!
//! - **Cost Model** (`cost.rs`): Heuristics for deciding what to inline
//! - **Graph Cloning** (`clone.rs`): Deep copying of callee graphs with ID remapping
//! - **Transform** (`transform.rs`): The actual inlining transformation
//! - **Speculative Inlining** (`speculative.rs`): Guard-based inlining for polymorphic sites
//! - **Callee Provider** (`callee.rs`): Interface for retrieving callee graphs
//!
//! # Inlining Process
//!
//! 1. **Discovery**: Find all call sites in the graph
//! 2. **Prioritization**: Rank call sites by expected benefit
//! 3. **Decision**: Apply cost model to filter candidates
//! 4. **Transformation**: Clone callee and integrate into caller
//! 5. **Cleanup**: Run DCE/GVN on inlined code
//!
//! # Performance Characteristics
//!
//! - O(n) call site discovery where n = number of nodes
//! - O(m) per inline where m = callee size
//! - Graph growth bounded by configurable factor
//! - Cycle detection prevents infinite recursion

mod callee;
mod clone;
mod cost;
mod hotness;
mod invalidation;
mod jit_provider;
mod speculative;
mod transform;

pub use callee::{CalleeGraph, CalleeProvider, CalleeRegistry, InlineHint};
pub use clone::GraphCloner;
pub use cost::{InlineCost, InlineCostModel};
pub use hotness::{HotnessConfig, HotnessLevel, HotnessTracker};
pub use invalidation::{
    InvalidationEvent, InvalidationReason, InvalidationRegistry, InvalidationSummary,
};
pub use jit_provider::{
    CompilationState, CompilationTier, DeoptReason, JitCalleeProvider, JitProviderConfig,
    ProviderSummary,
};
pub use speculative::{SpeculativeInliner, TypeGuardInfo};
pub use transform::InlineTransform;

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CallKind, Operator};
use std::collections::HashSet;
use std::sync::Arc;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the inlining heuristics.
#[derive(Debug, Clone)]
pub struct InlineConfig {
    /// Maximum callee size (in nodes) to inline.
    pub max_callee_size: usize,
    /// Maximum inlining depth (for recursive inlines).
    pub max_depth: usize,
    /// Maximum total code growth factor.
    pub max_growth_factor: f64,
    /// Maximum absolute graph size after inlining.
    pub max_total_size: usize,
    /// Bonus for call sites in hot loops (added to priority).
    pub hot_call_bonus: i32,
    /// Penalty for calls with many arguments.
    pub arg_penalty_per_arg: i32,
    /// Always inline functions marked as always-inline.
    pub respect_always_inline: bool,
    /// Never inline functions marked as never-inline.
    pub respect_never_inline: bool,
    /// Enable speculative inlining with type guards.
    pub enable_speculative: bool,
    /// Minimum call count for speculative inlining.
    pub speculative_min_count: u32,
    /// Maximum polymorphism for speculative inlining.
    pub speculative_max_targets: usize,
}

impl Default for InlineConfig {
    fn default() -> Self {
        Self {
            max_callee_size: 100,
            max_depth: 4,
            max_growth_factor: 2.0,
            max_total_size: 10000,
            hot_call_bonus: 50,
            arg_penalty_per_arg: 5,
            respect_always_inline: true,
            respect_never_inline: true,
            enable_speculative: true,
            speculative_min_count: 100,
            speculative_max_targets: 4,
        }
    }
}

impl InlineConfig {
    /// Create a conservative inlining configuration for faster compilation.
    pub fn conservative() -> Self {
        Self {
            max_callee_size: 30,
            max_depth: 2,
            max_growth_factor: 1.2,
            max_total_size: 5000,
            enable_speculative: false,
            ..Default::default()
        }
    }

    /// Create an aggressive inlining configuration for maximum optimization.
    pub fn aggressive() -> Self {
        Self {
            max_callee_size: 500,
            max_depth: 8,
            max_growth_factor: 5.0,
            max_total_size: 50000,
            hot_call_bonus: 100,
            enable_speculative: true,
            speculative_max_targets: 8,
            ..Default::default()
        }
    }

    /// Create configuration optimized for tier-1 JIT (fast compile, limited inline).
    pub fn tier1() -> Self {
        Self {
            max_callee_size: 20,
            max_depth: 1,
            max_growth_factor: 1.1,
            max_total_size: 2000,
            enable_speculative: false,
            ..Default::default()
        }
    }

    /// Create configuration optimized for tier-2 JIT (thorough optimization).
    pub fn tier2() -> Self {
        Self::default()
    }
}

// =============================================================================
// Callee Information
// =============================================================================

/// Information about a potential inline candidate.
#[derive(Debug, Clone)]
pub struct CalleeInfo {
    /// Unique function identifier.
    pub func_id: u64,
    /// Number of nodes in the callee graph.
    pub size: usize,
    /// Number of parameters.
    pub param_count: usize,
    /// Whether the function is recursive.
    pub is_recursive: bool,
    /// Whether the function contains loops.
    pub has_loops: bool,
    /// Whether marked as always-inline.
    pub always_inline: bool,
    /// Whether marked as never-inline.
    pub never_inline: bool,
    /// Estimated execution frequency (from profiling).
    pub call_count: u32,
    /// Whether this is a known builtin that can be intrinsified.
    pub is_intrinsic: bool,
}

impl Default for CalleeInfo {
    fn default() -> Self {
        Self {
            func_id: 0,
            size: 0,
            param_count: 0,
            is_recursive: false,
            has_loops: false,
            always_inline: false,
            never_inline: false,
            call_count: 0,
            is_intrinsic: false,
        }
    }
}

impl CalleeInfo {
    /// Create a new CalleeInfo from a callee graph.
    pub fn from_graph(func_id: u64, graph: &Graph, param_count: usize) -> Self {
        let size = graph.len();
        let has_loops = Self::detect_loops(graph);

        Self {
            func_id,
            size,
            param_count,
            is_recursive: false,
            has_loops,
            always_inline: false,
            never_inline: false,
            call_count: 0,
            is_intrinsic: false,
        }
    }

    /// Detect if the graph contains loops.
    fn detect_loops(graph: &Graph) -> bool {
        use crate::ir::operators::ControlOp;

        for (_, node) in graph.iter() {
            if let Operator::Control(ControlOp::Loop) = node.op {
                return true;
            }
            if let Operator::LoopPhi = node.op {
                return true;
            }
        }
        false
    }
}

// =============================================================================
// Call Site Information
// =============================================================================

/// Information about a call site in the graph.
#[derive(Debug, Clone)]
pub struct CallSite {
    /// Node ID of the call instruction.
    pub call_node: NodeId,
    /// Kind of call (direct, method, etc.).
    pub call_kind: CallKind,
    /// Callee information (if available).
    pub callee: Option<CalleeInfo>,
    /// Loop nesting depth at the call site.
    pub loop_depth: u32,
    /// Whether profiling indicates this is a hot call site.
    pub is_hot: bool,
    /// Computed inlining priority (higher = more beneficial).
    pub priority: i32,
    /// Arguments passed to the call.
    pub arguments: Vec<NodeId>,
    /// Control input to the call.
    pub control_input: Option<NodeId>,
}

impl CallSite {
    /// Create a new call site from a call node.
    pub fn from_node(graph: &Graph, node_id: NodeId) -> Option<Self> {
        let node = graph.get(node_id)?;

        let call_kind = match node.op {
            Operator::Call(kind) => kind,
            _ => return None,
        };

        // Extract arguments (skip control input at index 0, and callee at index 1)
        let arguments: Vec<NodeId> = node.inputs.iter().skip(2).collect();
        let control_input = node.inputs.get(0);

        Some(Self {
            call_node: node_id,
            call_kind,
            callee: None,
            loop_depth: 0,
            is_hot: false,
            priority: 0,
            arguments,
            control_input,
        })
    }
}

// =============================================================================
// Inlining Statistics
// =============================================================================

/// Statistics from the inlining pass.
#[derive(Debug, Clone, Default)]
pub struct InlineStats {
    /// Number of call sites examined.
    pub sites_examined: usize,
    /// Number of call sites inlined.
    pub sites_inlined: usize,
    /// Number of speculative inlines.
    pub speculative_inlines: usize,
    /// Total nodes added by inlining.
    pub nodes_added: usize,
    /// Total nodes before inlining.
    pub initial_size: usize,
    /// Total nodes after inlining.
    pub final_size: usize,
    /// Number of call sites rejected due to size.
    pub rejected_size: usize,
    /// Number of call sites rejected due to depth.
    pub rejected_depth: usize,
    /// Number of call sites rejected due to recursion.
    pub rejected_recursive: usize,
}

// =============================================================================
// Inlining Pass
// =============================================================================

/// The main inlining optimization pass.
#[derive(Debug)]
pub struct Inline {
    /// Configuration.
    config: InlineConfig,
    /// Callee graph provider.
    callee_provider: Option<Arc<dyn CalleeProvider>>,
    /// Statistics from the last run.
    stats: InlineStats,
    /// Current inlining depth.
    current_depth: usize,
    /// Functions currently being inlined (for cycle detection).
    inline_stack: Vec<u64>,
    /// Cost model for inlining decisions.
    cost_model: InlineCostModel,
}

impl Inline {
    /// Create a new inlining pass with default configuration.
    pub fn new() -> Self {
        Self {
            config: InlineConfig::default(),
            callee_provider: None,
            stats: InlineStats::default(),
            current_depth: 0,
            inline_stack: Vec::new(),
            cost_model: InlineCostModel::new(),
        }
    }

    /// Create inlining pass with custom configuration.
    pub fn with_config(config: InlineConfig) -> Self {
        Self {
            config,
            callee_provider: None,
            stats: InlineStats::default(),
            current_depth: 0,
            inline_stack: Vec::new(),
            cost_model: InlineCostModel::new(),
        }
    }

    /// Set the callee provider for retrieving function graphs.
    pub fn with_callee_provider(mut self, provider: Arc<dyn CalleeProvider>) -> Self {
        self.callee_provider = Some(provider);
        self
    }

    /// Get statistics from the last inlining run.
    pub fn stats(&self) -> &InlineStats {
        &self.stats
    }

    /// Get the number of functions inlined.
    #[inline]
    pub fn inlined(&self) -> usize {
        self.stats.sites_inlined
    }

    /// Get total code growth in nodes.
    #[inline]
    pub fn growth(&self) -> usize {
        self.stats.nodes_added
    }

    /// Run the inlining pass on a graph.
    fn run_inline(&mut self, graph: &mut Graph) -> bool {
        self.stats = InlineStats::default();
        self.stats.initial_size = graph.len();

        // Find all call sites
        let call_sites = self.find_call_sites(graph);
        self.stats.sites_examined = call_sites.len();

        if call_sites.is_empty() {
            self.stats.final_size = graph.len();
            return false;
        }

        // Sort by priority (highest first)
        let mut sorted_sites = call_sites;
        sorted_sites.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Calculate size limits
        let max_size_from_growth =
            (self.stats.initial_size as f64 * self.config.max_growth_factor) as usize;
        let max_size = max_size_from_growth.min(self.config.max_total_size);

        let mut changed = false;
        let mut inlined_nodes: HashSet<NodeId> = HashSet::new();

        for site in sorted_sites {
            // Skip if already inlined (node might have been replaced)
            if inlined_nodes.contains(&site.call_node) {
                continue;
            }

            // Check size limit
            if graph.len() >= max_size {
                break;
            }

            // Check depth limit
            if self.current_depth >= self.config.max_depth {
                self.stats.rejected_depth += 1;
                continue;
            }

            // Check if we should inline this site
            if let Some(decision) = self.should_inline(&site) {
                if decision {
                    if self.inline_call_site(graph, &site) {
                        self.stats.sites_inlined += 1;
                        inlined_nodes.insert(site.call_node);
                        changed = true;
                    }
                }
            }
        }

        self.stats.final_size = graph.len();
        self.stats.nodes_added = self
            .stats
            .final_size
            .saturating_sub(self.stats.initial_size);

        changed
    }

    /// Find all call sites in the graph.
    fn find_call_sites(&self, graph: &Graph) -> Vec<CallSite> {
        let mut sites = Vec::new();

        for (node_id, node) in graph.iter() {
            if let Operator::Call(call_kind) = &node.op {
                if let Some(mut site) = CallSite::from_node(graph, node_id) {
                    // Try to get callee information
                    if let Some(provider) = &self.callee_provider {
                        // Extract function ID from call (implementation-specific)
                        let func_id = self.extract_func_id(graph, node_id);
                        if let Some(id) = func_id {
                            if let Some(callee_graph) = provider.get_graph(id) {
                                let param_count = site.arguments.len();
                                site.callee = Some(CalleeInfo::from_graph(
                                    id,
                                    &callee_graph.graph,
                                    param_count,
                                ));
                            }
                        }
                    }

                    // Compute priority
                    site.priority = self.compute_priority(
                        *call_kind,
                        site.loop_depth,
                        site.is_hot,
                        &site.callee,
                    );

                    sites.push(site);
                }
            }
        }

        sites
    }

    /// Extract function ID from a call node.
    fn extract_func_id(&self, graph: &Graph, call_node: NodeId) -> Option<u64> {
        let node = graph.get(call_node)?;

        // The callee is typically the second input (after control)
        let callee_id = node.inputs.get(1)?;
        let callee_node = graph.get(callee_id)?;

        // If callee is a constant (function reference), extract its ID
        if let Operator::ConstInt(id) = callee_node.op {
            return Some(id as u64);
        }

        None
    }

    /// Compute inlining priority for a call site.
    fn compute_priority(
        &self,
        _call_kind: CallKind,
        loop_depth: u32,
        is_hot: bool,
        callee: &Option<CalleeInfo>,
    ) -> i32 {
        let mut priority: i32 = 0;

        // Hot call sites get bonus
        if is_hot {
            priority += self.config.hot_call_bonus;
        }

        // Loop depth significantly increases priority
        priority += (loop_depth as i32) * 20;

        // Callee-specific adjustments
        if let Some(info) = callee {
            // Smaller functions are better candidates
            if info.size < 10 {
                priority += 30;
            } else if info.size < 30 {
                priority += 15;
            }

            // High call count increases priority
            if info.call_count > 1000 {
                priority += 25;
            } else if info.call_count > 100 {
                priority += 10;
            }

            // Intrinsics get high priority
            if info.is_intrinsic {
                priority += 50;
            }

            // Always-inline gets maximum priority
            if info.always_inline {
                priority += 1000;
            }

            // Penalize functions with loops (they expand code a lot)
            if info.has_loops {
                priority -= 20;
            }

            // Penalize by argument count
            priority -= (info.param_count as i32) * self.config.arg_penalty_per_arg;
        }

        priority
    }

    /// Decide whether to inline a call site.
    fn should_inline(&mut self, site: &CallSite) -> Option<bool> {
        let callee = site.callee.as_ref()?;

        // Never-inline check
        if self.config.respect_never_inline && callee.never_inline {
            return Some(false);
        }

        // Always-inline check
        if self.config.respect_always_inline && callee.always_inline {
            return Some(true);
        }

        // Size check
        if callee.size > self.config.max_callee_size {
            self.stats.rejected_size += 1;
            return Some(false);
        }

        // Recursion check
        if self.inline_stack.contains(&callee.func_id) {
            self.stats.rejected_recursive += 1;
            return Some(false);
        }

        // Apply cost model
        let cost = self.cost_model.compute_cost(callee, site);
        Some(cost.should_inline())
    }

    /// Inline a single call site.
    fn inline_call_site(&mut self, graph: &mut Graph, site: &CallSite) -> bool {
        let callee = match &site.callee {
            Some(c) => c,
            None => return false,
        };

        // Get the callee graph
        let callee_graph = match &self.callee_provider {
            Some(provider) => match provider.get_graph(callee.func_id) {
                Some(g) => g,
                None => return false,
            },
            None => return false,
        };

        // Push to inline stack for cycle detection
        self.inline_stack.push(callee.func_id);
        self.current_depth += 1;

        // Perform the actual inlining transformation
        let result = InlineTransform::inline(graph, site, &callee_graph.graph);

        // Pop from inline stack
        self.inline_stack.pop();
        self.current_depth -= 1;

        result.is_ok()
    }
}

impl Default for Inline {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Inline {
    fn name(&self) -> &'static str {
        "inline"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_inline(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ControlBuilder, GraphBuilder};

    #[test]
    fn test_inline_config_default() {
        let config = InlineConfig::default();
        assert_eq!(config.max_callee_size, 100);
        assert_eq!(config.max_depth, 4);
        assert!(config.enable_speculative);
    }

    #[test]
    fn test_inline_config_conservative() {
        let config = InlineConfig::conservative();
        assert!(config.max_callee_size < InlineConfig::default().max_callee_size);
        assert!(config.max_depth < InlineConfig::default().max_depth);
        assert!(!config.enable_speculative);
    }

    #[test]
    fn test_inline_config_aggressive() {
        let config = InlineConfig::aggressive();
        assert!(config.max_callee_size > InlineConfig::default().max_callee_size);
        assert!(config.max_depth > InlineConfig::default().max_depth);
    }

    #[test]
    fn test_inline_config_tier1() {
        let config = InlineConfig::tier1();
        assert_eq!(config.max_depth, 1);
        assert!(!config.enable_speculative);
    }

    #[test]
    fn test_inline_new() {
        let inline = Inline::new();
        assert_eq!(inline.inlined(), 0);
        assert_eq!(inline.growth(), 0);
    }

    #[test]
    fn test_inline_name() {
        let inline = Inline::new();
        assert_eq!(inline.name(), "inline");
    }

    #[test]
    fn test_inline_no_calls() {
        let mut builder = GraphBuilder::new(2, 1);
        let p0 = builder.parameter(0).unwrap();
        builder.return_value(p0);

        let mut graph = builder.finish();
        let mut inline = Inline::new();

        let changed = inline.run(&mut graph);
        assert!(!changed);
        assert_eq!(inline.inlined(), 0);
    }

    #[test]
    fn test_callee_info_default() {
        let info = CalleeInfo::default();
        assert_eq!(info.size, 0);
        assert!(!info.is_recursive);
        assert!(!info.always_inline);
        assert!(!info.has_loops);
    }

    #[test]
    fn test_inline_stats_default() {
        let stats = InlineStats::default();
        assert_eq!(stats.sites_examined, 0);
        assert_eq!(stats.sites_inlined, 0);
        assert_eq!(stats.nodes_added, 0);
    }

    #[test]
    fn test_priority_calculation() {
        let inline = Inline::new();

        // Base priority
        let priority = inline.compute_priority(CallKind::Direct, 0, false, &None);
        assert_eq!(priority, 0);

        // Hot call bonus
        let priority_hot = inline.compute_priority(CallKind::Direct, 0, true, &None);
        assert_eq!(priority_hot, inline.config.hot_call_bonus);

        // Loop depth bonus
        let priority_loop = inline.compute_priority(CallKind::Direct, 2, false, &None);
        assert_eq!(priority_loop, 40); // 2 * 20

        // Small callee bonus
        let small_callee = CalleeInfo {
            size: 5,
            ..Default::default()
        };
        let priority_small =
            inline.compute_priority(CallKind::Direct, 0, false, &Some(small_callee));
        assert!(priority_small > 0);
    }

    #[test]
    fn test_should_inline_never_inline() {
        let mut inline = Inline::new();
        let callee = CalleeInfo {
            never_inline: true,
            size: 10,
            ..Default::default()
        };
        let site = CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: Some(callee),
            loop_depth: 0,
            is_hot: false,
            priority: 100,
            arguments: vec![],
            control_input: None,
        };

        assert_eq!(inline.should_inline(&site), Some(false));
    }

    #[test]
    fn test_should_inline_always_inline() {
        let mut inline = Inline::new();
        let callee = CalleeInfo {
            always_inline: true,
            size: 1000, // Would normally be rejected
            ..Default::default()
        };
        let site = CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: Some(callee),
            loop_depth: 0,
            is_hot: false,
            priority: 100,
            arguments: vec![],
            control_input: None,
        };

        assert_eq!(inline.should_inline(&site), Some(true));
    }

    #[test]
    fn test_should_inline_too_large() {
        let mut inline = Inline::new();
        let callee = CalleeInfo {
            size: 1000, // Exceeds max_callee_size
            ..Default::default()
        };
        let site = CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: Some(callee),
            loop_depth: 0,
            is_hot: false,
            priority: 100,
            arguments: vec![],
            control_input: None,
        };

        assert_eq!(inline.should_inline(&site), Some(false));
        assert_eq!(inline.stats.rejected_size, 1);
    }
}
