//! Callee Graph Provider Interface
//!
//! This module defines the interface for retrieving callee function graphs
//! during inlining. The provider abstraction allows different sources of
//! callee information:
//!
//! - **Compiled Functions**: Graphs from previously compiled functions
//! - **Intrinsics**: Built-in function implementations
//! - **Runtime Info**: Type-specialized variants
//!
//! # Thread Safety
//!
//! The `CalleeProvider` trait is `Send + Sync` to allow sharing across
//! threads in a concurrent JIT compiler.

use crate::ir::graph::Graph;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::Arc;

// =============================================================================
// Callee Graph
// =============================================================================

/// A callee graph with metadata.
#[derive(Debug, Clone)]
pub struct CalleeGraph {
    /// The IR graph for this function.
    pub graph: Graph,
    /// Number of parameters.
    pub param_count: usize,
    /// Whether this function has been marked for inlining.
    pub inline_hint: InlineHint,
    /// Estimated execution cost.
    pub cost_estimate: u32,
    /// Whether this is a known intrinsic.
    pub is_intrinsic: bool,
}

impl CalleeGraph {
    /// Create a new callee graph.
    pub fn new(graph: Graph, param_count: usize) -> Self {
        Self {
            graph,
            param_count,
            inline_hint: InlineHint::Default,
            cost_estimate: 0,
            is_intrinsic: false,
        }
    }

    /// Set the inline hint.
    pub fn with_hint(mut self, hint: InlineHint) -> Self {
        self.inline_hint = hint;
        self
    }

    /// Mark as intrinsic.
    pub fn as_intrinsic(mut self) -> Self {
        self.is_intrinsic = true;
        self
    }

    /// Set cost estimate.
    pub fn with_cost(mut self, cost: u32) -> Self {
        self.cost_estimate = cost;
        self
    }
}

/// Inlining hint from source annotations or heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineHint {
    /// No specific hint - use cost model.
    Default,
    /// Function should always be inlined.
    Always,
    /// Function should never be inlined.
    Never,
    /// Function is a good inline candidate (from profiling).
    Hot,
    /// Function is rarely called - prefer not to inline.
    Cold,
}

impl Default for InlineHint {
    fn default() -> Self {
        Self::Default
    }
}

// =============================================================================
// Callee Provider Trait
// =============================================================================

/// Trait for providing callee graphs during inlining.
///
/// Implementations may cache graphs, load from compiled code, or
/// synthesize graphs for intrinsics.
pub trait CalleeProvider: Send + Sync + std::fmt::Debug {
    /// Get the graph for a function by ID.
    fn get_graph(&self, func_id: u64) -> Option<Arc<CalleeGraph>>;

    /// Check if a function exists without retrieving its graph.
    fn has_function(&self, func_id: u64) -> bool {
        self.get_graph(func_id).is_some()
    }

    /// Get the parameter count for a function.
    fn param_count(&self, func_id: u64) -> Option<usize> {
        self.get_graph(func_id).map(|g| g.param_count)
    }

    /// Get the inline hint for a function.
    fn inline_hint(&self, func_id: u64) -> InlineHint {
        self.get_graph(func_id)
            .map(|g| g.inline_hint)
            .unwrap_or(InlineHint::Default)
    }

    /// Check if a function is an intrinsic.
    fn is_intrinsic(&self, func_id: u64) -> bool {
        self.get_graph(func_id)
            .map(|g| g.is_intrinsic)
            .unwrap_or(false)
    }
}

// =============================================================================
// Callee Registry
// =============================================================================

/// A thread-safe registry of callee graphs.
#[derive(Debug)]
pub struct CalleeRegistry {
    /// Stored callee graphs.
    graphs: RwLock<FxHashMap<u64, Arc<CalleeGraph>>>,
}

impl CalleeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            graphs: RwLock::new(FxHashMap::default()),
        }
    }

    /// Register a callee graph.
    pub fn register(&self, func_id: u64, graph: CalleeGraph) {
        let mut graphs = self.graphs.write();
        graphs.insert(func_id, Arc::new(graph));
    }

    /// Register a callee graph (Arc version).
    pub fn register_arc(&self, func_id: u64, graph: Arc<CalleeGraph>) {
        let mut graphs = self.graphs.write();
        graphs.insert(func_id, graph);
    }

    /// Remove a callee graph.
    pub fn unregister(&self, func_id: u64) -> Option<Arc<CalleeGraph>> {
        let mut graphs = self.graphs.write();
        graphs.remove(&func_id)
    }

    /// Clear all registered graphs.
    pub fn clear(&self) {
        let mut graphs = self.graphs.write();
        graphs.clear();
    }

    /// Get the number of registered functions.
    pub fn len(&self) -> usize {
        self.graphs.read().len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.graphs.read().is_empty()
    }
}

impl Default for CalleeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CalleeProvider for CalleeRegistry {
    fn get_graph(&self, func_id: u64) -> Option<Arc<CalleeGraph>> {
        let graphs = self.graphs.read();
        graphs.get(&func_id).cloned()
    }

    fn has_function(&self, func_id: u64) -> bool {
        let graphs = self.graphs.read();
        graphs.contains_key(&func_id)
    }
}

// =============================================================================
// Intrinsic Provider
// =============================================================================

/// Provides IR graphs for intrinsic functions.
///
/// Intrinsics are built-in functions with hand-crafted IR that is
/// highly optimized for specific operations.
#[derive(Debug)]
pub struct IntrinsicProvider {
    /// Registry of intrinsic graphs.
    intrinsics: FxHashMap<u64, Arc<CalleeGraph>>,
}

impl IntrinsicProvider {
    /// Create a new intrinsic provider with standard intrinsics.
    pub fn new() -> Self {
        let mut provider = Self {
            intrinsics: FxHashMap::default(),
        };
        provider.register_standard_intrinsics();
        provider
    }

    /// Register standard intrinsic functions.
    fn register_standard_intrinsics(&mut self) {
        // These would be populated with hand-crafted IR for common operations
        // For now, we just define the structure

        // Example intrinsic IDs (would be coordinated with runtime)
        // INTRINSIC_LEN = 1
        // INTRINSIC_ABS = 2
        // INTRINSIC_MIN = 3
        // INTRINSIC_MAX = 4
    }

    /// Register an intrinsic.
    pub fn register(&mut self, func_id: u64, graph: CalleeGraph) {
        self.intrinsics
            .insert(func_id, Arc::new(graph.as_intrinsic()));
    }
}

impl Default for IntrinsicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CalleeProvider for IntrinsicProvider {
    fn get_graph(&self, func_id: u64) -> Option<Arc<CalleeGraph>> {
        self.intrinsics.get(&func_id).cloned()
    }

    fn has_function(&self, func_id: u64) -> bool {
        self.intrinsics.contains_key(&func_id)
    }

    fn is_intrinsic(&self, func_id: u64) -> bool {
        self.intrinsics.contains_key(&func_id)
    }
}

// =============================================================================
// Composite Provider
// =============================================================================

/// A callee provider that chains multiple providers.
///
/// Searches providers in order until a graph is found.
#[derive(Debug)]
pub struct CompositeProvider {
    providers: Vec<Arc<dyn CalleeProvider>>,
}

impl CompositeProvider {
    /// Create a new composite provider.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Add a provider to the chain.
    pub fn add_provider(mut self, provider: Arc<dyn CalleeProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Add a provider to the chain (mutable version).
    pub fn push(&mut self, provider: Arc<dyn CalleeProvider>) {
        self.providers.push(provider);
    }
}

impl Default for CompositeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CalleeProvider for CompositeProvider {
    fn get_graph(&self, func_id: u64) -> Option<Arc<CalleeGraph>> {
        for provider in &self.providers {
            if let Some(graph) = provider.get_graph(func_id) {
                return Some(graph);
            }
        }
        None
    }

    fn has_function(&self, func_id: u64) -> bool {
        self.providers.iter().any(|p| p.has_function(func_id))
    }

    fn is_intrinsic(&self, func_id: u64) -> bool {
        self.providers.iter().any(|p| p.is_intrinsic(func_id))
    }
}
