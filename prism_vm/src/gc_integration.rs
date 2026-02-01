//! GC integration layer for the Virtual Machine.
//!
//! This module provides the bridge between the VM and the garbage collector,
//! including:
//!
//! - `ManagedHeap`: Wrapper providing allocation and collection for the VM
//! - `RootProvider`: Interface for walking stack roots during collection
//! - Safe point detection and collection triggering
//!
//! # Architecture
//!
//! The GC integration follows a conservative approach:
//!
//! 1. **Allocation**: Objects are allocated through `ManagedHeap::alloc()`
//!    which may trigger collection if heap is full
//! 2. **Collection**: The collector walks roots provided by the VM's frame stack
//! 3. **Write Barriers**: Container mutations notify the GC of old→young refs
//!
//! # Example
//!
//! ```ignore
//! use prism_vm::gc_integration::ManagedHeap;
//! use prism_runtime::RuntimeObjectTracer;
//! use prism_gc::config::GcConfig;
//!
//! let mut heap = ManagedHeap::new(GcConfig::default());
//! let tracer = RuntimeObjectTracer;
//!
//! // Allocate an object
//! let ptr = heap.alloc::<ListObject>(ListObject::new());
//!
//! // Collection happens automatically when needed, or manually trigger:
//! heap.collect_minor(&tracer);
//! ```

use prism_core::Value;
use prism_gc::{
    GcStats, ObjectTracer,
    collector::{CollectionResult, Collector},
    config::GcConfig,
    heap::GcHeap,
    roots::RootSet,
};

// =============================================================================
// ManagedHeap - VM's interface to the GC
// =============================================================================

/// Managed heap wrapper for VM integration.
///
/// Provides a high-level interface to the garbage collector,
/// handling allocation, collection triggers, and root management.
pub struct ManagedHeap {
    /// Underlying GC heap
    heap: GcHeap,
    /// Collector for running GC
    collector: Collector,
    /// Root set tracking
    roots: RootSet,
    /// Configuration
    config: GcConfig,
}

impl ManagedHeap {
    /// Create a new managed heap with the given configuration.
    pub fn new(config: GcConfig) -> Self {
        Self {
            heap: GcHeap::new(config.clone()),
            collector: Collector::new(),
            roots: RootSet::new(),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GcConfig::default())
    }

    /// Get the underlying heap for direct access.
    #[inline]
    pub fn heap(&self) -> &GcHeap {
        &self.heap
    }

    /// Get mutable access to the heap.
    #[inline]
    pub fn heap_mut(&mut self) -> &mut GcHeap {
        &mut self.heap
    }

    /// Get access to the root set.
    #[inline]
    pub fn roots(&self) -> &RootSet {
        &self.roots
    }

    /// Get mutable access to the root set.
    #[inline]
    pub fn roots_mut(&mut self) -> &mut RootSet {
        &mut self.roots
    }

    /// Get GC statistics.
    pub fn stats(&self) -> &GcStats {
        self.heap.stats()
    }

    /// Get configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }

    /// Trigger a minor collection with object tracing.
    ///
    /// This collects only the nursery (young generation),
    /// which is typically fast (< 1ms for small heaps).
    ///
    /// # Arguments
    /// - `tracer`: Object tracer for type-aware tracing (e.g., `RuntimeObjectTracer`)
    pub fn collect_minor<T: ObjectTracer>(&mut self, tracer: &T) -> CollectionResult {
        self.collector
            .collect_minor(&mut self.heap, &self.roots, tracer)
    }

    /// Trigger a major collection with object tracing.
    ///
    /// This performs a full mark-sweep of the entire heap.
    /// Use sparingly as it has higher latency.
    ///
    /// # Arguments
    /// - `tracer`: Object tracer for type-aware tracing (e.g., `RuntimeObjectTracer`)
    pub fn collect_major<T: ObjectTracer>(&mut self, tracer: &T) -> CollectionResult {
        self.collector
            .collect_major(&mut self.heap, &self.roots, tracer)
    }

    /// Let the collector decide which type of collection to run.
    ///
    /// The collector uses heuristics based on:
    /// - Nursery fill level
    /// - Time since last collection
    /// - Heap occupancy ratio
    ///
    /// # Arguments
    /// - `tracer`: Object tracer for type-aware tracing (e.g., `RuntimeObjectTracer`)
    pub fn collect_auto<T: ObjectTracer>(&mut self, tracer: &T) -> CollectionResult {
        self.collector.collect(&mut self.heap, &self.roots, tracer)
    }

    /// Trigger a minor collection without object tracing.
    ///
    /// This is useful for tests or when the runtime is unavailable.
    /// Only root objects will be processed; their children won't be traced.
    pub fn collect_minor_roots_only(&mut self) -> CollectionResult {
        self.collector
            .collect_minor_roots_only(&mut self.heap, &self.roots)
    }

    /// Trigger a major collection without object tracing.
    ///
    /// This is useful for tests or when the runtime is unavailable.
    pub fn collect_major_roots_only(&mut self) -> CollectionResult {
        self.collector
            .collect_major_roots_only(&mut self.heap, &self.roots)
    }

    /// Check if minor collection should be triggered.
    ///
    /// Called at safe points (function calls, loop back-edges).
    #[inline]
    pub fn should_minor_collect(&self) -> bool {
        self.heap.should_minor_collect()
    }

    /// Check if major collection should be triggered.
    #[inline]
    pub fn should_major_collect(&self) -> bool {
        self.heap.should_major_collect()
    }

    /// Get the promotion age threshold.
    #[inline]
    pub fn promotion_age(&self) -> u8 {
        self.collector.promotion_age()
    }
}

impl Default for ManagedHeap {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// =============================================================================
// FrameRoots - Walking frame stack for roots
// =============================================================================

/// Trait for types that can provide GC roots.
///
/// The VM implements this to allow the collector to walk the frame stack.
pub trait RootProvider {
    /// Visit all root values.
    ///
    /// Called during the marking phase of garbage collection.
    /// The implementation must call `visitor` for each Value
    /// that could be a live object reference.
    fn visit_roots<F>(&self, visitor: F)
    where
        F: FnMut(Value);
}

/// Stack roots collected from VM frames.
///
/// During collection, the VM provides this to allow the GC
/// to scan all live stack values.
pub struct StackRoots {
    /// Collected root values from all frames
    values: Vec<Value>,
}

impl StackRoots {
    /// Create an empty root set.
    pub fn new() -> Self {
        Self {
            values: Vec::with_capacity(256),
        }
    }

    /// Clear and prepare for reuse.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Add a root value.
    #[inline]
    pub fn add(&mut self, value: Value) {
        // Only add object references
        if value.as_object_ptr().is_some() {
            self.values.push(value);
        }
    }

    /// Add multiple root values.
    pub fn add_slice(&mut self, values: &[Value]) {
        for &value in values {
            self.add(value);
        }
    }

    /// Get all collected roots.
    pub fn as_slice(&self) -> &[Value] {
        &self.values
    }

    /// Number of roots.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Default for StackRoots {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GC Safe Points
// =============================================================================

/// Safe point types for GC.
///
/// Safe points are locations in execution where it's safe to pause
/// for garbage collection without corrupting state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafePoint {
    /// At function call boundary
    FunctionCall,
    /// At loop back-edge
    LoopBackEdge,
    /// Explicit allocation site
    Allocation,
    /// User-requested collection
    Explicit,
}

/// Check if we're at a safe point for collection.
///
/// Returns true if the VM is in a state where GC can safely run.
#[inline]
pub const fn is_safe_point() -> bool {
    // Currently we're always at a safe point between bytecode instructions
    true
}

// =============================================================================
// Allocation helpers
// =============================================================================

/// Allocation result from managed heap.
#[derive(Debug)]
pub enum AllocResult<T> {
    /// Allocation succeeded
    Ok(T),
    /// Out of memory - collection may help
    NeedsCollection,
    /// Permanent failure
    OutOfMemory,
}

impl<T> AllocResult<T> {
    /// Convert to Option, discarding error information.
    pub fn ok(self) -> Option<T> {
        match self {
            AllocResult::Ok(val) => Some(val),
            _ => None,
        }
    }

    /// Check if allocation succeeded.
    pub fn is_ok(&self) -> bool {
        matches!(self, AllocResult::Ok(_))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gc::NoopObjectTracer;

    #[test]
    fn test_managed_heap_creation() {
        let heap = ManagedHeap::new(GcConfig::default());
        // Fresh heap should have no roots registered
        assert_eq!(heap.roots().handle_count(), 0);
        assert_eq!(heap.roots().global_count(), 0);
    }

    #[test]
    fn test_managed_heap_defaults() {
        let heap = ManagedHeap::with_defaults();
        let config = heap.config();
        assert!(config.nursery_size > 0);
    }

    #[test]
    fn test_collect_minor_with_tracer() {
        let mut heap = ManagedHeap::with_defaults();
        let result = heap.collect_minor(&NoopObjectTracer);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_major_with_tracer() {
        let mut heap = ManagedHeap::with_defaults();
        let result = heap.collect_major(&NoopObjectTracer);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_auto_with_tracer() {
        let mut heap = ManagedHeap::with_defaults();
        let result = heap.collect_auto(&NoopObjectTracer);
        // Empty heap should do minor collection
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_roots_only() {
        let mut heap = ManagedHeap::with_defaults();
        let minor_result = heap.collect_minor_roots_only();
        let major_result = heap.collect_major_roots_only();
        assert_eq!(minor_result.bytes_freed, 0);
        assert_eq!(major_result.bytes_freed, 0);
    }

    #[test]
    fn test_stack_roots() {
        let mut roots = StackRoots::new();
        assert!(roots.is_empty());

        // Adding primitives doesn't add roots
        roots.add(Value::int(42).unwrap());
        roots.add(Value::bool(true));
        roots.add(Value::none());

        // Primitives are not object references
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn test_stack_roots_clear() {
        let mut roots = StackRoots::new();
        roots.add(Value::none());
        roots.clear();
        assert!(roots.is_empty());
    }

    #[test]
    fn test_safe_point() {
        assert!(is_safe_point());
    }

    #[test]
    fn test_alloc_result() {
        let result: AllocResult<i32> = AllocResult::Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(42));

        let result: AllocResult<i32> = AllocResult::NeedsCollection;
        assert!(!result.is_ok());
        assert_eq!(result.ok(), None);
    }

    #[test]
    fn test_should_collect_initially_false() {
        let heap = ManagedHeap::with_defaults();
        // Empty heap shouldn't need collection
        assert!(!heap.should_minor_collect());
    }

    #[test]
    fn test_promotion_age() {
        let heap = ManagedHeap::with_defaults();
        assert_eq!(heap.promotion_age(), 2);
    }
}
