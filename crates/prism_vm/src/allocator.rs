//! High-performance GC allocator for VM object allocation.
//!
//! This module provides a typed allocation API that bridges the gap between
//! the low-level `GcHeap` and the high-level VM operations. It ensures:
//!
//! - **Type Safety**: Allocations are properly typed and sized
//! - **Zero-Copy Initialization**: Objects are constructed directly in GC memory
//! - **Trace Enforcement**: Only `Trace`-implementing types can be allocated
//! - **Cache-Friendly Access**: Allocator is designed for hot-path inline expansion
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       VM Operations                              │
//! │  (containers.rs, calls.rs, objects.rs, subscript.rs)            │
//! └────────────────────────────┬────────────────────────────────────┘
//!                              │ alloc<T>() / alloc_value<T>()
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       GcAllocator                                │
//! │  - Typed allocation with automatic sizing                       │
//! │  - Direct Value construction                                    │
//! │  - OOM handling with AllocResult                                │
//! └────────────────────────────┬────────────────────────────────────┘
//!                              │ heap.alloc(size)
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       GcHeap                                     │
//! │  - Nursery (bump allocation)                                    │
//! │  - Old space (block allocation)                                 │
//! │  - Large object space                                           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! The allocator is designed for maximum performance:
//!
//! - **Inline Allocation**: All methods are `#[inline]` for hot-path optimization
//! - **No Branching**: Common path has minimal branches
//! - **Cache Locality**: Objects are allocated in contiguous nursery space
//! - **Zero Overhead**: Thin wrapper with no runtime checks beyond OOM

use prism_core::Value;
use prism_gc::GcHeap;
use prism_gc::trace::Trace;

use std::alloc::Layout;
use std::ptr::NonNull;

// =============================================================================
// GC Allocator
// =============================================================================

/// High-performance typed GC allocator.
///
/// Provides zero-cost typed allocation over the underlying `GcHeap`.
/// All allocations are properly sized and aligned for the target type.
///
/// # Usage
///
/// ```ignore
/// let allocator = GcAllocator::new(&heap);
///
/// // Allocate a list
/// let list_ptr = allocator.alloc(ListObject::new())?;
///
/// // Allocate and get Value directly
/// let list_val = allocator.alloc_value(ListObject::new())?;
/// ```
///
/// # Thread Safety
///
/// `GcAllocator` borrows the heap and is not `Send` or `Sync`.
/// Each thread should use its own allocator instance.
pub struct GcAllocator<'h> {
    /// Reference to the GC heap for allocation.
    heap: &'h GcHeap,
}

impl<'h> GcAllocator<'h> {
    /// Create a new allocator for the given heap.
    ///
    /// # Performance
    ///
    /// This is a zero-cost operation that just stores a reference.
    #[inline]
    pub const fn new(heap: &'h GcHeap) -> Self {
        Self { heap }
    }

    /// Allocate and initialize a GC-managed object.
    ///
    /// Returns a raw pointer to the allocated object, or `None` if
    /// the nursery is full and collection is needed.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The type to allocate. Must implement `Trace` for GC integration.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid until the next GC collection that
    /// determines the object is unreachable. Callers must ensure the
    /// object is rooted before the next GC safe point.
    ///
    /// # Performance
    ///
    /// - Fast path: Single bump-pointer increment + object initialization
    /// - Inlined for zero call overhead on hot paths
    #[inline]
    pub fn alloc<T: Trace>(&self, value: T) -> Option<*mut T> {
        // Allocate raw memory from heap
        let ptr = self.heap.alloc_layout(Layout::new::<T>())?;

        // Initialize the object in-place
        let typed_ptr = ptr.as_ptr() as *mut T;
        unsafe {
            std::ptr::write(typed_ptr, value);
        }

        Some(typed_ptr)
    }

    /// Allocate an object and return it as a Value.
    ///
    /// Convenience method that combines allocation with Value construction.
    /// This is the primary allocation method for VM operations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let val = allocator.alloc_value(ListObject::from_slice(&[
    ///     Value::int(1).unwrap(),
    ///     Value::int(2).unwrap(),
    /// ]))?;
    /// vm.set_register(dst, val);
    /// ```
    ///
    /// # Performance
    ///
    /// Single inline call with no additional overhead over `alloc()`.
    #[inline]
    pub fn alloc_value<T: Trace>(&self, value: T) -> Option<Value> {
        let ptr = self.alloc(value)?;
        Some(Value::object_ptr(ptr as *const ()))
    }

    /// Try to allocate, with explicit OOM handling.
    ///
    /// Returns `AllocResult` for cases where the caller needs to
    /// distinguish between "needs collection" and "out of memory".
    #[inline]
    pub fn try_alloc<T: Trace>(&self, value: T) -> AllocResult<*mut T> {
        match self.alloc(value) {
            Some(ptr) => AllocResult::Ok(ptr),
            None => {
                // Check if this was nursery-full or true OOM
                if self.heap.should_minor_collect() {
                    AllocResult::NeedsCollection
                } else {
                    AllocResult::OutOfMemory
                }
            }
        }
    }

    /// Allocate with explicit size and alignment.
    ///
    /// Low-level allocation for variable-sized objects or custom layouts.
    /// The caller is responsible for initialization.
    ///
    /// # Safety
    ///
    /// - The returned memory is uninitialized
    /// - Must be initialized before next GC
    /// - Size must be accurate for the object
    #[inline]
    pub fn alloc_raw(&self, size: usize) -> Option<NonNull<u8>> {
        self.heap.alloc(size)
    }

    /// Allocate directly in the old generation.
    ///
    /// Use for long-lived objects that should skip the nursery.
    /// This avoids copying during minor collections.
    ///
    /// # Use Cases
    ///
    /// - Module-level objects
    /// - Cached/interned objects
    /// - Objects known to be long-lived
    #[inline]
    pub fn alloc_tenured<T: Trace>(&self, value: T) -> Option<*mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.heap.alloc_tenured_layout(layout)?;

        let typed_ptr = ptr.as_ptr() as *mut T;
        unsafe {
            std::ptr::write(typed_ptr, value);
        }

        Some(typed_ptr)
    }

    /// Check if the allocator can satisfy an allocation of the given size.
    ///
    /// Non-allocating check for pre-flight verification.
    #[inline]
    pub fn can_alloc(&self, size: usize) -> bool {
        // Check if nursery has space
        !self.heap.should_minor_collect() || size >= self.heap.config().large_object_threshold
    }

    /// Get heap statistics for monitoring.
    #[inline]
    pub fn stats(&self) -> &prism_gc::GcStats {
        self.heap.stats()
    }
}

// =============================================================================
// Allocation Results
// =============================================================================

/// Result type for allocation operations with explicit error cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocResult<T> {
    /// Allocation succeeded.
    Ok(T),
    /// Nursery full - minor collection may free space.
    NeedsCollection,
    /// True out-of-memory condition.
    OutOfMemory,
}

impl<T> AllocResult<T> {
    /// Convert to Option, discarding error information.
    #[inline]
    pub fn ok(self) -> Option<T> {
        match self {
            AllocResult::Ok(val) => Some(val),
            _ => None,
        }
    }

    /// Check if allocation succeeded.
    #[inline]
    pub fn is_ok(&self) -> bool {
        matches!(self, AllocResult::Ok(_))
    }

    /// Check if collection is needed.
    #[inline]
    pub fn needs_collection(&self) -> bool {
        matches!(self, AllocResult::NeedsCollection)
    }

    /// Map the success value.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> AllocResult<U> {
        match self {
            AllocResult::Ok(val) => AllocResult::Ok(f(val)),
            AllocResult::NeedsCollection => AllocResult::NeedsCollection,
            AllocResult::OutOfMemory => AllocResult::OutOfMemory,
        }
    }

    /// Unwrap or panic with message.
    #[inline]
    pub fn expect(self, msg: &str) -> T {
        match self {
            AllocResult::Ok(val) => val,
            AllocResult::NeedsCollection => panic!("{}: needs collection", msg),
            AllocResult::OutOfMemory => panic!("{}: out of memory", msg),
        }
    }

    /// Unwrap or panic.
    #[inline]
    pub fn unwrap(self) -> T {
        self.expect("allocation failed")
    }
}

// =============================================================================
// Convenience Traits
// =============================================================================

/// Extension trait for allocating from any heap reference.
pub trait HeapAllocExt {
    /// Create an allocator for this heap.
    fn allocator(&self) -> GcAllocator<'_>;
}

impl HeapAllocExt for GcHeap {
    #[inline]
    fn allocator(&self) -> GcAllocator<'_> {
        GcAllocator::new(self)
    }
}
