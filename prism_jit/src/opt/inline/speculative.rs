//! Speculative Inlining with Type Guards
//!
//! This module implements speculative (guarded) inlining for polymorphic call sites.
//! In dynamic languages like Python, call sites are often polymorphic (multiple receiver
//! types) but tend to have a dominant type at runtime. Speculative inlining:
//!
//! 1. Uses profiling data to identify the most common receiver types
//! 2. Inserts type guards that check if the actual type matches expected
//! 3. Inlines the specialized implementation for the fast path
//! 4. Falls back to megamorphic dispatch on guard failure
//!
//! # Architecture
//!
//! ```text
//! Before:                          After (Monomorphic):
//!
//! ┌─────────────┐                  ┌─────────────┐
//! │  receiver   │                  │  receiver   │
//! └──────┬──────┘                  └──────┬──────┘
//!        │                                │
//! ┌──────▼──────┐                  ┌──────▼──────┐
//! │ method call │       →          │ type guard  │
//! │   obj.foo() │                  │ obj.type == │
//! └──────┬──────┘                  │   Expected? │
//!        │                         └──────┬──────┘
//!        │                           ┌────┴────┐
//!        │                           │         │
//!        │                      ┌────▼───┐ ┌───▼────┐
//!        │                      │ INLINE │ │ DEOPT  │
//!        │                      │ fast   │ │ slow   │
//!        │                      │ path   │ │ path   │
//!        │                      └────┬───┘ └───┬────┘
//!        │                           │         │
//!        │                      ┌────▼─────────▼────┐
//!        │                      │       merge       │
//! ┌──────▼──────┐               └─────────┬─────────┘
//! │  continue   │                         │
//! └─────────────┘               ┌─────────▼─────────┐
//!                               │     continue      │
//!                               └───────────────────┘
//! ```
//!
//! # Polymorphism Levels
//!
//! - **Monomorphic**: Single type seen - one inline with guard
//! - **Polymorphic**: 2-4 types seen - cascading guards (if-else chain)
//! - **Megamorphic**: >4 types seen - no inlining, use IC/dispatch table
//!
//! # Deoptimization
//!
//! When a type guard fails, we need to deoptimize:
//! - Continue to the slow path for this call
//! - Optionally record the new type for recompilation
//! - Eventually transition to megamorphic if too many types

use super::callee::{CalleeGraph, CalleeProvider};
use super::clone::GraphCloner;
use super::transform::{InlineError, InlineResult};
use super::{CallSite, CalleeInfo, InlineConfig};
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{CmpOp, ControlOp, GuardKind, Operator};
use crate::ir::types::ValueType;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// =============================================================================
// Type Constants
// =============================================================================

/// Maximum number of types to track per call site before going megamorphic.
pub const MAX_POLYMORPHIC_TYPES: usize = 4;

/// Minimum call count before considering speculative inlining.
pub const MIN_CALL_COUNT_FOR_SPECULATION: u32 = 100;

/// Minimum percentage of calls a type must represent to be inlined.
pub const MIN_TYPE_PERCENTAGE: f64 = 0.05; // 5%

/// Threshold for considering a site monomorphic (single dominant type).
pub const MONOMORPHIC_THRESHOLD: f64 = 0.95; // 95% same type

// =============================================================================
// Type Guard Information
// =============================================================================

/// Information about a type guard used for speculative inlining.
#[derive(Debug, Clone)]
pub struct TypeGuardInfo {
    /// The expected type ID (shape/class pointer).
    pub expected_type: TypeId,
    /// The guard condition node in the IR.
    pub guard_node: Option<NodeId>,
    /// The true projection (fast path).
    pub true_branch: Option<NodeId>,
    /// The false projection (slow path/next guard).
    pub false_branch: Option<NodeId>,
    /// Profiling hit count for this type.
    pub hit_count: u64,
    /// Total calls seen at this site.
    pub total_calls: u64,
}

impl TypeGuardInfo {
    /// Create a new type guard info.
    pub fn new(expected_type: TypeId) -> Self {
        Self {
            expected_type,
            guard_node: None,
            true_branch: None,
            false_branch: None,
            hit_count: 0,
            total_calls: 0,
        }
    }

    /// Create with profiling data.
    pub fn with_profile(expected_type: TypeId, hit_count: u64, total_calls: u64) -> Self {
        Self {
            expected_type,
            guard_node: None,
            true_branch: None,
            false_branch: None,
            hit_count,
            total_calls,
        }
    }

    /// Get the hit rate for this type (0.0 to 1.0).
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.total_calls as f64
        }
    }

    /// Check if this type is the dominant type (monomorphic).
    #[inline]
    pub fn is_monomorphic(&self) -> bool {
        self.hit_rate() >= MONOMORPHIC_THRESHOLD
    }

    /// Check if this type is worth inlining.
    #[inline]
    pub fn is_worth_inlining(&self) -> bool {
        self.hit_rate() >= MIN_TYPE_PERCENTAGE
    }
}

// =============================================================================
// Type Identifier
// =============================================================================

/// A unique identifier for a type (shape/class).
///
/// In Prism, this corresponds to a Shape pointer or class ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub u64);

impl TypeId {
    /// Unknown/any type.
    pub const UNKNOWN: TypeId = TypeId(0);

    /// Integer type.
    pub const INT: TypeId = TypeId(1);

    /// Float type.
    pub const FLOAT: TypeId = TypeId(2);

    /// String type.
    pub const STRING: TypeId = TypeId(3);

    /// List type.
    pub const LIST: TypeId = TypeId(4);

    /// Dict type.
    pub const DICT: TypeId = TypeId(5);

    /// None type.
    pub const NONE: TypeId = TypeId(6);

    /// Bool type.
    pub const BOOL: TypeId = TypeId(7);

    /// Tuple type.
    pub const TUPLE: TypeId = TypeId(8);

    /// First user-defined type ID.
    pub const USER_TYPE_START: u64 = 1000;

    /// Create a new type ID.
    #[inline]
    pub const fn new(id: u64) -> Self {
        TypeId(id)
    }

    /// Check if this is a known primitive type.
    #[inline]
    pub fn is_primitive(&self) -> bool {
        self.0 > 0 && self.0 < Self::USER_TYPE_START
    }

    /// Check if this is a user-defined type.
    #[inline]
    pub fn is_user_type(&self) -> bool {
        self.0 >= Self::USER_TYPE_START
    }

    /// Get the raw ID value.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for TypeId {
    fn from(id: u64) -> Self {
        TypeId(id)
    }
}

// =============================================================================
// Speculative Target
// =============================================================================

/// A potential inline target for speculative inlining.
#[derive(Debug, Clone)]
pub struct SpeculativeTarget {
    /// The receiver type this target handles.
    pub receiver_type: TypeId,
    /// The function to inline for this type.
    pub target_func: u64,
    /// Callee information.
    pub callee_info: CalleeInfo,
    /// Number of times this target was invoked.
    pub invocation_count: u64,
    /// Whether this is a megamorphic fallback.
    pub is_fallback: bool,
}

impl SpeculativeTarget {
    /// Create a new speculative target.
    pub fn new(receiver_type: TypeId, target_func: u64, callee_info: CalleeInfo) -> Self {
        Self {
            receiver_type,
            target_func,
            callee_info,
            invocation_count: 0,
            is_fallback: false,
        }
    }

    /// Create a megamorphic fallback target.
    pub fn fallback(target_func: u64) -> Self {
        Self {
            receiver_type: TypeId::UNKNOWN,
            target_func,
            callee_info: CalleeInfo::default(),
            invocation_count: 0,
            is_fallback: true,
        }
    }

    /// Set the invocation count.
    pub fn with_count(mut self, count: u64) -> Self {
        self.invocation_count = count;
        self
    }
}

// =============================================================================
// Type Profile
// =============================================================================

/// Profiling data for a single call site.
#[derive(Debug)]
pub struct TypeProfile {
    /// Bytecode offset of the call site.
    pub bc_offset: u32,
    /// Total number of calls observed.
    total_calls: AtomicU64,
    /// Type counts: TypeId -> count.
    type_counts: RwLock<FxHashMap<TypeId, u64>>,
    /// Cached sorted targets (invalidated on updates).
    cached_targets: RwLock<Option<Vec<(TypeId, u64)>>>,
}

impl TypeProfile {
    /// Create a new empty type profile.
    pub fn new(bc_offset: u32) -> Self {
        Self {
            bc_offset,
            total_calls: AtomicU64::new(0),
            type_counts: RwLock::new(FxHashMap::default()),
            cached_targets: RwLock::new(None),
        }
    }

    /// Record a call with the given receiver type.
    pub fn record(&self, receiver_type: TypeId) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);

        // Invalidate cache
        *self.cached_targets.write() = None;

        // Update type count
        let mut counts = self.type_counts.write();
        *counts.entry(receiver_type).or_insert(0) += 1;
    }

    /// Get the total number of calls.
    #[inline]
    pub fn total_calls(&self) -> u64 {
        self.total_calls.load(Ordering::Relaxed)
    }

    /// Get the count for a specific type.
    pub fn type_count(&self, type_id: TypeId) -> u64 {
        self.type_counts.read().get(&type_id).copied().unwrap_or(0)
    }

    /// Get the number of distinct types seen.
    pub fn type_diversity(&self) -> usize {
        self.type_counts.read().len()
    }

    /// Check if this site is monomorphic.
    pub fn is_monomorphic(&self) -> bool {
        let total = self.total_calls();
        if total < MIN_CALL_COUNT_FOR_SPECULATION as u64 {
            return false;
        }

        let counts = self.type_counts.read();
        if counts.len() != 1 {
            return false;
        }

        if let Some((_, &count)) = counts.iter().next() {
            let hit_rate = count as f64 / total as f64;
            return hit_rate >= MONOMORPHIC_THRESHOLD;
        }

        false
    }

    /// Check if this site is polymorphic (2-4 types).
    pub fn is_polymorphic(&self) -> bool {
        let diversity = self.type_diversity();
        diversity >= 2 && diversity <= MAX_POLYMORPHIC_TYPES
    }

    /// Check if this site is megamorphic (too many types).
    pub fn is_megamorphic(&self) -> bool {
        self.type_diversity() > MAX_POLYMORPHIC_TYPES
    }

    /// Get the dominant types sorted by frequency.
    pub fn get_dominant_types(&self, max_types: usize) -> Vec<(TypeId, u64)> {
        // Check cache first
        if let Some(ref cached) = *self.cached_targets.read() {
            return cached.iter().take(max_types).cloned().collect();
        }

        // Compute sorted types
        let counts = self.type_counts.read();
        let mut types: Vec<(TypeId, u64)> = counts.iter().map(|(&t, &c)| (t, c)).collect();

        // Sort by count descending
        types.sort_by(|a, b| b.1.cmp(&a.1));

        // Cache the result
        *self.cached_targets.write() = Some(types.clone());

        types.into_iter().take(max_types).collect()
    }

    /// Get type guard information for speculative inlining.
    pub fn get_guard_info(&self) -> Vec<TypeGuardInfo> {
        let total = self.total_calls();
        let dominant = self.get_dominant_types(MAX_POLYMORPHIC_TYPES);

        dominant
            .into_iter()
            .filter(|(_, count)| {
                let rate = *count as f64 / total as f64;
                rate >= MIN_TYPE_PERCENTAGE
            })
            .map(|(type_id, count)| TypeGuardInfo::with_profile(type_id, count, total))
            .collect()
    }
}

impl Clone for TypeProfile {
    fn clone(&self) -> Self {
        let type_counts = self.type_counts.read();
        let new_counts: FxHashMap<TypeId, u64> = type_counts.clone();

        Self {
            bc_offset: self.bc_offset,
            total_calls: AtomicU64::new(self.total_calls.load(Ordering::Relaxed)),
            type_counts: RwLock::new(new_counts),
            cached_targets: RwLock::new(None),
        }
    }
}

// =============================================================================
// Type Profile Registry
// =============================================================================

/// Registry of type profiles for all call sites.
#[derive(Debug)]
pub struct TypeProfileRegistry {
    /// Profiles indexed by (function_id, bc_offset).
    profiles: RwLock<FxHashMap<(u64, u32), Arc<TypeProfile>>>,
}

impl TypeProfileRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            profiles: RwLock::new(FxHashMap::default()),
        }
    }

    /// Get or create a profile for a call site.
    pub fn get_or_create(&self, func_id: u64, bc_offset: u32) -> Arc<TypeProfile> {
        let key = (func_id, bc_offset);

        // Fast path: read lock
        {
            let profiles = self.profiles.read();
            if let Some(profile) = profiles.get(&key) {
                return Arc::clone(profile);
            }
        }

        // Slow path: write lock
        let mut profiles = self.profiles.write();
        profiles
            .entry(key)
            .or_insert_with(|| Arc::new(TypeProfile::new(bc_offset)))
            .clone()
    }

    /// Get a profile if it exists.
    pub fn get(&self, func_id: u64, bc_offset: u32) -> Option<Arc<TypeProfile>> {
        self.profiles.read().get(&(func_id, bc_offset)).cloned()
    }

    /// Clear all profiles.
    pub fn clear(&self) {
        self.profiles.write().clear();
    }

    /// Get all profiles for a function.
    pub fn profiles_for_function(&self, func_id: u64) -> Vec<Arc<TypeProfile>> {
        self.profiles
            .read()
            .iter()
            .filter(|((fid, _), _)| *fid == func_id)
            .map(|(_, p)| Arc::clone(p))
            .collect()
    }
}

impl Default for TypeProfileRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Speculative Inlining Statistics
// =============================================================================

/// Statistics from speculative inlining.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Number of monomorphic inlines.
    pub monomorphic_inlines: usize,
    /// Number of polymorphic inlines (with multiple guards).
    pub polymorphic_inlines: usize,
    /// Number of sites skipped due to megamorphism.
    pub megamorphic_skipped: usize,
    /// Number of type guards inserted.
    pub guards_inserted: usize,
    /// Number of deopt paths created.
    pub deopt_paths: usize,
    /// Total nodes added.
    pub nodes_added: usize,
}

// =============================================================================
// Speculative Inliner
// =============================================================================

/// The speculative inliner performs type-guarded inlining.
#[derive(Debug)]
pub struct SpeculativeInliner {
    /// Configuration.
    config: InlineConfig,
    /// Type profile registry.
    profiles: Arc<TypeProfileRegistry>,
    /// Callee provider.
    callee_provider: Option<Arc<dyn CalleeProvider>>,
    /// Statistics.
    stats: SpeculativeStats,
}

impl SpeculativeInliner {
    /// Create a new speculative inliner.
    pub fn new(config: InlineConfig, profiles: Arc<TypeProfileRegistry>) -> Self {
        Self {
            config,
            profiles,
            callee_provider: None,
            stats: SpeculativeStats::default(),
        }
    }

    /// Set the callee provider.
    pub fn with_callee_provider(mut self, provider: Arc<dyn CalleeProvider>) -> Self {
        self.callee_provider = Some(provider);
        self
    }

    /// Get statistics from the last run.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Attempt speculative inlining at a call site.
    pub fn try_speculative_inline(
        &mut self,
        graph: &mut Graph,
        site: &CallSite,
        func_id: u64,
    ) -> InlineResult<SpeculativeInlineInfo> {
        // Get type profile for this site
        let bc_offset = graph.get(site.call_node).map(|n| n.bc_offset).unwrap_or(0);

        let profile = match self.profiles.get(func_id, bc_offset) {
            Some(p) => p,
            None => {
                return Err(InlineError::InvalidTransformation(
                    "No type profile available".into(),
                ))
            }
        };

        // Check minimum call count
        if profile.total_calls() < self.config.speculative_min_count as u64 {
            return Err(InlineError::InvalidTransformation(
                "Insufficient call count for speculation".into(),
            ));
        }

        // Choose inlining strategy based on polymorphism
        if profile.is_megamorphic() {
            self.stats.megamorphic_skipped += 1;
            return Err(InlineError::InvalidTransformation(
                "Site is megamorphic".into(),
            ));
        }

        if profile.is_monomorphic() {
            self.inline_monomorphic(graph, site, &profile)
        } else {
            self.inline_polymorphic(graph, site, &profile)
        }
    }

    /// Perform monomorphic inlining with a single type guard.
    fn inline_monomorphic(
        &mut self,
        graph: &mut Graph,
        site: &CallSite,
        profile: &TypeProfile,
    ) -> InlineResult<SpeculativeInlineInfo> {
        let guard_infos = profile.get_guard_info();
        if guard_infos.is_empty() {
            return Err(InlineError::InvalidTransformation(
                "No suitable types for speculation".into(),
            ));
        }

        let primary_type = guard_infos[0].expected_type;

        // Get the callee for this type
        let callee_graph = self.get_callee_for_type(site, primary_type)?;

        // Insert type guard and inline
        let info = self.insert_guarded_inline(graph, site, &callee_graph, primary_type)?;

        self.stats.monomorphic_inlines += 1;
        self.stats.guards_inserted += 1;
        self.stats.deopt_paths += 1;
        self.stats.nodes_added += info.nodes_added;

        Ok(info)
    }

    /// Perform polymorphic inlining with cascading type guards.
    fn inline_polymorphic(
        &mut self,
        graph: &mut Graph,
        site: &CallSite,
        profile: &TypeProfile,
    ) -> InlineResult<SpeculativeInlineInfo> {
        let guard_infos = profile.get_guard_info();
        if guard_infos.is_empty() {
            return Err(InlineError::InvalidTransformation(
                "No suitable types for speculation".into(),
            ));
        }

        let max_targets = self.config.speculative_max_targets.min(guard_infos.len());
        let targets: Vec<_> = guard_infos.into_iter().take(max_targets).collect();

        // Build cascading guards
        let info = self.insert_cascading_guards(graph, site, &targets)?;

        self.stats.polymorphic_inlines += 1;
        self.stats.guards_inserted += targets.len();
        self.stats.deopt_paths += 1;
        self.stats.nodes_added += info.nodes_added;

        Ok(info)
    }

    /// Get the callee graph for a specific receiver type.
    fn get_callee_for_type(
        &self,
        site: &CallSite,
        _type_id: TypeId,
    ) -> InlineResult<Arc<CalleeGraph>> {
        let provider = self
            .callee_provider
            .as_ref()
            .ok_or(InlineError::InvalidTransformation(
                "No callee provider".into(),
            ))?;

        // In a real implementation, we'd look up the method for this specific type.
        // For now, use the call site's callee if available.
        let func_id = site
            .callee
            .as_ref()
            .map(|c| c.func_id)
            .ok_or(InlineError::InvalidTransformation("No callee info".into()))?;

        provider
            .get_graph(func_id)
            .ok_or(InlineError::InvalidTransformation(
                "Callee graph not found".into(),
            ))
    }

    /// Insert a single guarded inline.
    fn insert_guarded_inline(
        &self,
        graph: &mut Graph,
        site: &CallSite,
        callee: &CalleeGraph,
        expected_type: TypeId,
    ) -> InlineResult<SpeculativeInlineInfo> {
        // Get the receiver (first argument for method calls)
        let receiver = site
            .arguments
            .first()
            .copied()
            .ok_or(InlineError::InvalidTransformation("No receiver".into()))?;

        // Get control input
        let call_node = graph
            .get(site.call_node)
            .ok_or(InlineError::InvalidCallSite)?;
        let control_input = site
            .control_input
            .or_else(|| call_node.inputs.get(0))
            .ok_or(InlineError::ControlFlowError)?;

        // Create type check: receiver.type == expected_type
        // We use a Guard(GuardKind::Type) for type checking
        let type_const = graph.const_int(expected_type.raw() as i64);

        // Emit a type guard: checks if receiver has the expected type
        // The guard produces a boolean result
        let receiver_type_node = self.emit_get_type(graph, receiver);
        let type_check = graph.add_node_with_type(
            Operator::IntCmp(CmpOp::Eq),
            InputList::Pair(receiver_type_node, type_const),
            ValueType::Bool,
        );

        // Create if node for guard
        let if_node = graph.add_node_with_type(
            Operator::Control(ControlOp::If),
            InputList::Pair(control_input, type_check),
            ValueType::Control,
        );

        // True projection (fast path - type matches)
        let true_proj = graph.add_node_with_type(
            Operator::Projection(0),
            InputList::Single(if_node),
            ValueType::Control,
        );

        // False projection (slow path - deoptimize)
        let false_proj = graph.add_node_with_type(
            Operator::Projection(1),
            InputList::Single(if_node),
            ValueType::Control,
        );

        // Inline the callee on the fast path
        let clone_result = GraphCloner::new(&callee.graph)
            .with_arguments(&site.arguments)
            .with_control_input(true_proj)
            .clone_into(graph);

        // Handle returns from inlined code
        let (fast_result, fast_exit) = handle_returns(graph, &clone_result.cloned_returns)?;

        // Create slow path: keep original call
        let slow_result = self.create_deopt_call(graph, site, false_proj)?;

        // Merge the two paths
        let merge = graph.region(&[fast_exit, slow_result.control]);

        // Create phi for result
        let final_result = match (fast_result, slow_result.value) {
            (Some(fast), Some(slow)) => {
                let result_type = graph.get(fast).map(|n| n.ty).unwrap_or(ValueType::Top);
                Some(graph.phi(merge, &[fast, slow], result_type))
            }
            (Some(r), None) | (None, Some(r)) => Some(r),
            (None, None) => None,
        };

        // Replace uses of original call
        if let Some(result) = final_result {
            graph.replace_all_uses(site.call_node, result);
        }

        // Reconnect control
        reconnect_control(graph, site.call_node, merge)?;

        // Kill original call
        graph.kill(site.call_node);

        Ok(SpeculativeInlineInfo {
            nodes_added: clone_result.nodes_cloned + 8, // Clone + guard structure
            guards_inserted: 1,
            result_node: final_result,
            exit_control: merge,
            guard_info: TypeGuardInfo {
                expected_type,
                guard_node: Some(if_node),
                true_branch: Some(true_proj),
                false_branch: Some(false_proj),
                hit_count: 0,
                total_calls: 0,
            },
        })
    }

    /// Insert cascading type guards for polymorphic sites.
    fn insert_cascading_guards(
        &self,
        graph: &mut Graph,
        site: &CallSite,
        targets: &[TypeGuardInfo],
    ) -> InlineResult<SpeculativeInlineInfo> {
        if targets.is_empty() {
            return Err(InlineError::InvalidTransformation("No targets".into()));
        }

        // Get the receiver
        let receiver = site
            .arguments
            .first()
            .copied()
            .ok_or(InlineError::InvalidTransformation("No receiver".into()))?;

        // Get control input
        let call_node = graph
            .get(site.call_node)
            .ok_or(InlineError::InvalidCallSite)?;
        let mut current_control = site
            .control_input
            .or_else(|| call_node.inputs.get(0))
            .ok_or(InlineError::ControlFlowError)?;

        // Get receiver type once
        let receiver_type = self.emit_get_type(graph, receiver);

        let mut inline_exits: Vec<NodeId> = Vec::new();
        let mut inline_results: Vec<Option<NodeId>> = Vec::new();
        let mut total_nodes = 0;

        // Create cascading guards: if (type == T1) { inline1 } else if (type == T2) { inline2 } ...
        for (i, target) in targets.iter().enumerate() {
            let is_last = i == targets.len() - 1;

            // Create type check
            let type_const = graph.const_int(target.expected_type.raw() as i64);
            let type_check = graph.add_node_with_type(
                Operator::IntCmp(CmpOp::Eq),
                InputList::Pair(receiver_type, type_const),
                ValueType::Bool,
            );

            // Create if node
            let if_node = graph.add_node_with_type(
                Operator::Control(ControlOp::If),
                InputList::Pair(current_control, type_check),
                ValueType::Control,
            );

            // True projection - inline this target
            let true_proj = graph.add_node_with_type(
                Operator::Projection(0),
                InputList::Single(if_node),
                ValueType::Control,
            );

            // False projection - continue to next guard or deopt
            let false_proj = graph.add_node_with_type(
                Operator::Projection(1),
                InputList::Single(if_node),
                ValueType::Control,
            );

            // Get callee and inline
            if let Ok(callee) = self.get_callee_for_type(site, target.expected_type) {
                let clone_result = GraphCloner::new(&callee.graph)
                    .with_arguments(&site.arguments)
                    .with_control_input(true_proj)
                    .clone_into(graph);

                if let Ok((result, exit)) = handle_returns(graph, &clone_result.cloned_returns) {
                    inline_exits.push(exit);
                    inline_results.push(result);
                    total_nodes += clone_result.nodes_cloned;
                }
            }

            if is_last {
                // Last target - false branch goes to deopt
                let deopt = self.create_deopt_call(graph, site, false_proj)?;
                inline_exits.push(deopt.control);
                inline_results.push(deopt.value);
            } else {
                // Not last - continue to next guard
                current_control = false_proj;
            }
        }

        // Merge all paths
        let merge = graph.region(&inline_exits);

        // Create phi for results
        let valid_results: Vec<NodeId> = inline_results.iter().filter_map(|r| *r).collect();
        let final_result = if valid_results.len() == inline_exits.len() && !valid_results.is_empty()
        {
            let result_type = graph
                .get(valid_results[0])
                .map(|n| n.ty)
                .unwrap_or(ValueType::Top);
            Some(graph.phi(merge, &valid_results, result_type))
        } else if !valid_results.is_empty() {
            Some(valid_results[0])
        } else {
            None
        };

        // Replace uses of original call
        if let Some(result) = final_result {
            graph.replace_all_uses(site.call_node, result);
        }

        // Reconnect control
        reconnect_control(graph, site.call_node, merge)?;

        // Kill original call
        graph.kill(site.call_node);

        Ok(SpeculativeInlineInfo {
            nodes_added: total_nodes + targets.len() * 5 + 2,
            guards_inserted: targets.len(),
            result_node: final_result,
            exit_control: merge,
            guard_info: targets[0].clone(),
        })
    }

    /// Emit IR to get the type of a value.
    ///
    /// This loads the object's hidden class / shape pointer for type comparison.
    /// The Guard(GuardKind::Type) is used for the actual type check.
    fn emit_get_type(&self, graph: &mut Graph, value: NodeId) -> NodeId {
        // In the real implementation, this would emit:
        // - Load the object's shape/class pointer from the header
        // - Or use inline cache to get cached type
        //
        // For now, emit a GetAttr that loads the internal __class__ pointer.
        // This will be lowered to an efficient header load in codegen.
        graph.add_node_with_type(
            Operator::Guard(GuardKind::Type),
            InputList::Single(value),
            ValueType::Int64,
        )
    }

    /// Create a deoptimization call for the slow path.
    fn create_deopt_call(
        &self,
        graph: &mut Graph,
        site: &CallSite,
        control: NodeId,
    ) -> InlineResult<DeoptCall> {
        // For the slow path, we either:
        // 1. Call the original function through megamorphic dispatch
        // 2. Deoptimize to the interpreter
        //
        // Here we create a new call node with the new control input

        let original = graph
            .get(site.call_node)
            .ok_or(InlineError::InvalidCallSite)?;

        // Build new inputs: new control + original callee + args
        let mut new_inputs = vec![control];
        for input in original.inputs.iter().skip(1) {
            new_inputs.push(input);
        }

        let call =
            graph.add_node_with_type(original.op, InputList::from_slice(&new_inputs), original.ty);

        Ok(DeoptCall {
            call,
            control: call,
            value: Some(call),
        })
    }
}

// =============================================================================
// Helper Functions (shared with transform.rs)
// =============================================================================

/// Handle returns from inlined code by creating merge points.
pub fn handle_returns(
    graph: &mut Graph,
    returns: &[NodeId],
) -> InlineResult<(Option<NodeId>, NodeId)> {
    match returns.len() {
        0 => Err(InlineError::NoReturns),
        1 => {
            let ret_node = graph.get(returns[0]).ok_or(InlineError::MalformedCallee)?;
            let return_value = ret_node.inputs.get(1);
            let callee_control = ret_node
                .inputs
                .get(0)
                .ok_or(InlineError::ControlFlowError)?;
            let exit_region = graph.region(&[callee_control]);
            Ok((return_value, exit_region))
        }
        _ => merge_returns(graph, returns),
    }
}

/// Merge multiple return paths into a single control/value.
fn merge_returns(graph: &mut Graph, returns: &[NodeId]) -> InlineResult<(Option<NodeId>, NodeId)> {
    let mut control_inputs: Vec<NodeId> = Vec::with_capacity(returns.len());
    let mut value_inputs: Vec<NodeId> = Vec::with_capacity(returns.len());
    let mut return_type = ValueType::None;

    for &ret_id in returns {
        let ret_node = graph.get(ret_id).ok_or(InlineError::MalformedCallee)?;

        if let Some(ctrl) = ret_node.inputs.get(0) {
            control_inputs.push(ctrl);
        }
        if let Some(val) = ret_node.inputs.get(1) {
            value_inputs.push(val);
            let val_type = graph.get(val).map(|n| n.ty).unwrap_or(ValueType::None);
            if return_type == ValueType::None && val_type != ValueType::None {
                return_type = val_type;
            }
        }
    }

    if control_inputs.is_empty() {
        return Err(InlineError::ControlFlowError);
    }

    let merge_region = graph.region(&control_inputs);

    let result_node = if !value_inputs.is_empty() && value_inputs.len() == control_inputs.len() {
        Some(graph.phi(merge_region, &value_inputs, return_type))
    } else {
        None
    };

    Ok((result_node, merge_region))
}

/// Reconnect control flow from call node to exit control.
pub fn reconnect_control(
    graph: &mut Graph,
    call_node: NodeId,
    exit_control: NodeId,
) -> InlineResult<()> {
    let users: Vec<NodeId> = graph.uses(call_node).to_vec();

    for user_id in users {
        let user = graph.get(user_id).ok_or(InlineError::ControlFlowError)?;

        if user.is_control() || matches!(user.op, Operator::Phi | Operator::LoopPhi) {
            let inputs = user.inputs.to_vec();
            for (i, &input) in inputs.iter().enumerate() {
                if input == call_node {
                    graph.replace_input(user_id, i, exit_control);
                }
            }
        }
    }

    Ok(())
}

/// Information about a deoptimization call.
struct DeoptCall {
    /// The call node.
    #[allow(dead_code)]
    call: NodeId,
    /// Control output.
    control: NodeId,
    /// Value output.
    value: Option<NodeId>,
}

// =============================================================================
// Speculative Inline Info
// =============================================================================

/// Information about a completed speculative inline.
#[derive(Debug, Clone)]
pub struct SpeculativeInlineInfo {
    /// Number of nodes added.
    pub nodes_added: usize,
    /// Number of type guards inserted.
    pub guards_inserted: usize,
    /// The result node (if any).
    pub result_node: Option<NodeId>,
    /// The exit control node.
    pub exit_control: NodeId,
    /// Information about the primary guard.
    pub guard_info: TypeGuardInfo,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
    use crate::ir::operators::CallKind;

    // -------------------------------------------------------------------------
    // TypeId Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_id_primitives() {
        assert!(TypeId::INT.is_primitive());
        assert!(TypeId::FLOAT.is_primitive());
        assert!(TypeId::STRING.is_primitive());
        assert!(TypeId::LIST.is_primitive());
        assert!(TypeId::DICT.is_primitive());
        assert!(TypeId::NONE.is_primitive());
        assert!(TypeId::BOOL.is_primitive());
        assert!(TypeId::TUPLE.is_primitive());

        assert!(!TypeId::UNKNOWN.is_primitive());
    }

    #[test]
    fn test_type_id_user_types() {
        let user_type = TypeId::new(TypeId::USER_TYPE_START);
        assert!(user_type.is_user_type());
        assert!(!user_type.is_primitive());

        let user_type2 = TypeId::new(TypeId::USER_TYPE_START + 100);
        assert!(user_type2.is_user_type());
    }

    #[test]
    fn test_type_id_from_u64() {
        let type_id: TypeId = 42u64.into();
        assert_eq!(type_id.raw(), 42);
    }

    // -------------------------------------------------------------------------
    // TypeGuardInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_guard_info_new() {
        let guard = TypeGuardInfo::new(TypeId::INT);
        assert_eq!(guard.expected_type, TypeId::INT);
        assert!(guard.guard_node.is_none());
        assert_eq!(guard.hit_count, 0);
        assert_eq!(guard.total_calls, 0);
    }

    #[test]
    fn test_type_guard_info_with_profile() {
        let guard = TypeGuardInfo::with_profile(TypeId::STRING, 950, 1000);
        assert_eq!(guard.expected_type, TypeId::STRING);
        assert_eq!(guard.hit_count, 950);
        assert_eq!(guard.total_calls, 1000);
    }

    #[test]
    fn test_type_guard_hit_rate() {
        let guard = TypeGuardInfo::with_profile(TypeId::INT, 750, 1000);
        assert!((guard.hit_rate() - 0.75).abs() < 0.001);

        let empty_guard = TypeGuardInfo::new(TypeId::INT);
        assert_eq!(empty_guard.hit_rate(), 0.0);
    }

    #[test]
    fn test_type_guard_is_monomorphic() {
        let mono = TypeGuardInfo::with_profile(TypeId::INT, 960, 1000);
        assert!(mono.is_monomorphic());

        let poly = TypeGuardInfo::with_profile(TypeId::INT, 800, 1000);
        assert!(!poly.is_monomorphic());
    }

    #[test]
    fn test_type_guard_is_worth_inlining() {
        let worthy = TypeGuardInfo::with_profile(TypeId::INT, 100, 1000); // 10%
        assert!(worthy.is_worth_inlining());

        let unworthy = TypeGuardInfo::with_profile(TypeId::INT, 40, 1000); // 4%
        assert!(!unworthy.is_worth_inlining());
    }

    // -------------------------------------------------------------------------
    // SpeculativeTarget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_speculative_target_new() {
        let callee = CalleeInfo::default();
        let target = SpeculativeTarget::new(TypeId::INT, 42, callee);

        assert_eq!(target.receiver_type, TypeId::INT);
        assert_eq!(target.target_func, 42);
        assert!(!target.is_fallback);
    }

    #[test]
    fn test_speculative_target_fallback() {
        let target = SpeculativeTarget::fallback(99);

        assert_eq!(target.receiver_type, TypeId::UNKNOWN);
        assert_eq!(target.target_func, 99);
        assert!(target.is_fallback);
    }

    #[test]
    fn test_speculative_target_with_count() {
        let callee = CalleeInfo::default();
        let target = SpeculativeTarget::new(TypeId::INT, 42, callee).with_count(1000);

        assert_eq!(target.invocation_count, 1000);
    }

    // -------------------------------------------------------------------------
    // TypeProfile Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_profile_new() {
        let profile = TypeProfile::new(100);
        assert_eq!(profile.bc_offset, 100);
        assert_eq!(profile.total_calls(), 0);
        assert_eq!(profile.type_diversity(), 0);
    }

    #[test]
    fn test_type_profile_record() {
        let profile = TypeProfile::new(0);

        profile.record(TypeId::INT);
        profile.record(TypeId::INT);
        profile.record(TypeId::FLOAT);

        assert_eq!(profile.total_calls(), 3);
        assert_eq!(profile.type_count(TypeId::INT), 2);
        assert_eq!(profile.type_count(TypeId::FLOAT), 1);
        assert_eq!(profile.type_diversity(), 2);
    }

    #[test]
    fn test_type_profile_monomorphic() {
        let profile = TypeProfile::new(0);

        // Record enough calls to pass minimum threshold
        for _ in 0..MIN_CALL_COUNT_FOR_SPECULATION + 10 {
            profile.record(TypeId::INT);
        }

        assert!(profile.is_monomorphic());
        assert!(!profile.is_polymorphic());
        assert!(!profile.is_megamorphic());
    }

    #[test]
    fn test_type_profile_polymorphic() {
        let profile = TypeProfile::new(0);

        // Record 3 different types
        for _ in 0..50 {
            profile.record(TypeId::INT);
            profile.record(TypeId::FLOAT);
            profile.record(TypeId::STRING);
        }

        assert!(!profile.is_monomorphic());
        assert!(profile.is_polymorphic());
        assert!(!profile.is_megamorphic());
    }

    #[test]
    fn test_type_profile_megamorphic() {
        let profile = TypeProfile::new(0);

        // Record more than MAX_POLYMORPHIC_TYPES types
        for i in 0..10 {
            profile.record(TypeId::new(i));
        }

        assert!(profile.is_megamorphic());
    }

    #[test]
    fn test_type_profile_dominant_types() {
        let profile = TypeProfile::new(0);

        // Record with different frequencies
        for _ in 0..100 {
            profile.record(TypeId::INT);
        }
        for _ in 0..50 {
            profile.record(TypeId::FLOAT);
        }
        for _ in 0..25 {
            profile.record(TypeId::STRING);
        }

        let dominant = profile.get_dominant_types(2);
        assert_eq!(dominant.len(), 2);
        assert_eq!(dominant[0].0, TypeId::INT);
        assert_eq!(dominant[0].1, 100);
        assert_eq!(dominant[1].0, TypeId::FLOAT);
        assert_eq!(dominant[1].1, 50);
    }

    #[test]
    fn test_type_profile_get_guard_info() {
        let profile = TypeProfile::new(0);

        for _ in 0..100 {
            profile.record(TypeId::INT);
        }
        for _ in 0..50 {
            profile.record(TypeId::FLOAT);
        }
        // This one is below MIN_TYPE_PERCENTAGE
        for _ in 0..5 {
            profile.record(TypeId::STRING);
        }

        let guards = profile.get_guard_info();

        // STRING should be filtered out (5/155 < 5%)
        assert_eq!(guards.len(), 2);
        assert!(guards.iter().any(|g| g.expected_type == TypeId::INT));
        assert!(guards.iter().any(|g| g.expected_type == TypeId::FLOAT));
    }

    #[test]
    fn test_type_profile_clone() {
        let profile = TypeProfile::new(42);
        profile.record(TypeId::INT);
        profile.record(TypeId::INT);
        profile.record(TypeId::FLOAT);

        let cloned = profile.clone();

        assert_eq!(cloned.bc_offset, 42);
        assert_eq!(cloned.total_calls(), 3);
        assert_eq!(cloned.type_count(TypeId::INT), 2);
        assert_eq!(cloned.type_count(TypeId::FLOAT), 1);
    }

    // -------------------------------------------------------------------------
    // TypeProfileRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_profile_registry_new() {
        let registry = TypeProfileRegistry::new();
        assert!(registry.get(0, 0).is_none());
    }

    #[test]
    fn test_profile_registry_get_or_create() {
        let registry = TypeProfileRegistry::new();

        let profile1 = registry.get_or_create(1, 100);
        let profile2 = registry.get_or_create(1, 100);

        // Should return the same profile
        assert_eq!(profile1.bc_offset, profile2.bc_offset);

        // Record on one should be visible on the other
        profile1.record(TypeId::INT);
        assert_eq!(profile2.type_count(TypeId::INT), 1);
    }

    #[test]
    fn test_profile_registry_different_sites() {
        let registry = TypeProfileRegistry::new();

        let profile1 = registry.get_or_create(1, 100);
        let profile2 = registry.get_or_create(1, 200);
        let profile3 = registry.get_or_create(2, 100);

        profile1.record(TypeId::INT);
        profile2.record(TypeId::FLOAT);
        profile3.record(TypeId::STRING);

        assert_eq!(profile1.type_count(TypeId::INT), 1);
        assert_eq!(profile2.type_count(TypeId::FLOAT), 1);
        assert_eq!(profile3.type_count(TypeId::STRING), 1);

        assert_eq!(profile1.type_count(TypeId::FLOAT), 0);
    }

    #[test]
    fn test_profile_registry_clear() {
        let registry = TypeProfileRegistry::new();

        registry.get_or_create(1, 100);
        registry.get_or_create(2, 200);

        registry.clear();

        assert!(registry.get(1, 100).is_none());
        assert!(registry.get(2, 200).is_none());
    }

    #[test]
    fn test_profile_registry_profiles_for_function() {
        let registry = TypeProfileRegistry::new();

        registry.get_or_create(1, 100);
        registry.get_or_create(1, 200);
        registry.get_or_create(1, 300);
        registry.get_or_create(2, 100);

        let func1_profiles = registry.profiles_for_function(1);
        assert_eq!(func1_profiles.len(), 3);

        let func2_profiles = registry.profiles_for_function(2);
        assert_eq!(func2_profiles.len(), 1);
    }

    // -------------------------------------------------------------------------
    // SpeculativeStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.monomorphic_inlines, 0);
        assert_eq!(stats.polymorphic_inlines, 0);
        assert_eq!(stats.megamorphic_skipped, 0);
        assert_eq!(stats.guards_inserted, 0);
        assert_eq!(stats.deopt_paths, 0);
        assert_eq!(stats.nodes_added, 0);
    }

    // -------------------------------------------------------------------------
    // SpeculativeInliner Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_speculative_inliner_new() {
        let config = InlineConfig::default();
        let profiles = Arc::new(TypeProfileRegistry::new());
        let inliner = SpeculativeInliner::new(config, profiles);

        let stats = inliner.stats();
        assert_eq!(stats.monomorphic_inlines, 0);
    }

    #[test]
    fn test_speculative_inliner_no_profile() {
        let config = InlineConfig::default();
        let profiles = Arc::new(TypeProfileRegistry::new());
        let mut inliner = SpeculativeInliner::new(config, profiles);

        let mut graph = Graph::new();
        let site = CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: None,
            loop_depth: 0,
            is_hot: false,
            priority: 0,
            arguments: vec![],
            control_input: None,
        };

        let result = inliner.try_speculative_inline(&mut graph, &site, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_speculative_inliner_insufficient_calls() {
        let config = InlineConfig::default();
        let profiles = Arc::new(TypeProfileRegistry::new());

        // Record fewer calls than minimum
        let profile = profiles.get_or_create(1, 0);
        for _ in 0..10 {
            profile.record(TypeId::INT);
        }

        let mut inliner = SpeculativeInliner::new(config, Arc::clone(&profiles));

        let mut graph = Graph::new();
        let site = CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: None,
            loop_depth: 0,
            is_hot: false,
            priority: 0,
            arguments: vec![],
            control_input: None,
        };

        let result = inliner.try_speculative_inline(&mut graph, &site, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_speculative_inliner_megamorphic_skip() {
        let config = InlineConfig::default();
        let profiles = Arc::new(TypeProfileRegistry::new());

        // Record many different types to make it megamorphic
        let profile = profiles.get_or_create(1, 0);
        for i in 0..20 {
            for _ in 0..10 {
                profile.record(TypeId::new(i));
            }
        }

        let mut inliner = SpeculativeInliner::new(config, Arc::clone(&profiles));

        let mut graph = Graph::new();
        let site = CallSite {
            call_node: NodeId::new(0),
            call_kind: CallKind::Direct,
            callee: None,
            loop_depth: 0,
            is_hot: false,
            priority: 0,
            arguments: vec![],
            control_input: None,
        };

        let result = inliner.try_speculative_inline(&mut graph, &site, 1);
        assert!(result.is_err());
        assert_eq!(inliner.stats().megamorphic_skipped, 1);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[allow(dead_code)]
    fn make_simple_callee() -> Graph {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);
        builder.finish()
    }

    #[test]
    fn test_guard_info_serialization() {
        let guard = TypeGuardInfo {
            expected_type: TypeId::INT,
            guard_node: Some(NodeId::new(10)),
            true_branch: Some(NodeId::new(11)),
            false_branch: Some(NodeId::new(12)),
            hit_count: 1000,
            total_calls: 1100,
        };

        assert_eq!(guard.expected_type, TypeId::INT);
        assert!(guard.guard_node.is_some());
        assert!((guard.hit_rate() - 0.909).abs() < 0.01);
    }

    #[test]
    fn test_speculative_inline_info() {
        let info = SpeculativeInlineInfo {
            nodes_added: 50,
            guards_inserted: 2,
            result_node: Some(NodeId::new(100)),
            exit_control: NodeId::new(101),
            guard_info: TypeGuardInfo::new(TypeId::INT),
        };

        assert_eq!(info.nodes_added, 50);
        assert_eq!(info.guards_inserted, 2);
        assert!(info.result_node.is_some());
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_profile_single_call() {
        let profile = TypeProfile::new(0);
        profile.record(TypeId::INT);

        assert_eq!(profile.total_calls(), 1);
        assert!(!profile.is_monomorphic()); // Below minimum threshold
    }

    #[test]
    fn test_dominant_types_empty() {
        let profile = TypeProfile::new(0);
        let dominant = profile.get_dominant_types(5);
        assert!(dominant.is_empty());
    }

    #[test]
    fn test_dominant_types_caching() {
        let profile = TypeProfile::new(0);
        for _ in 0..100 {
            profile.record(TypeId::INT);
        }

        // First call computes and caches
        let dominant1 = profile.get_dominant_types(2);

        // Second call should return cached value
        let dominant2 = profile.get_dominant_types(2);

        assert_eq!(dominant1, dominant2);
    }

    #[test]
    fn test_guard_info_edge_cases() {
        // Zero total calls
        let guard = TypeGuardInfo::with_profile(TypeId::INT, 0, 0);
        assert_eq!(guard.hit_rate(), 0.0);
        assert!(!guard.is_monomorphic());
        assert!(!guard.is_worth_inlining());

        // All hits
        let guard = TypeGuardInfo::with_profile(TypeId::INT, 1000, 1000);
        assert_eq!(guard.hit_rate(), 1.0);
        assert!(guard.is_monomorphic());
        assert!(guard.is_worth_inlining());
    }
}
