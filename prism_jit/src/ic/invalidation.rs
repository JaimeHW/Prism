//! IC Invalidation System
//!
//! Manages shape-based invalidation of inline caches. When a shape transition
//! occurs that could affect cached slot offsets, all dependent ICs must be
//! invalidated.
//!
//! # Architecture
//!
//! - **ShapeVersion**: Monotonic epoch counter for tracking shape changes
//! - **IcDependency**: Links an IC site to the shapes it depends on
//! - **IcInvalidator**: Coordinates batch invalidation across IC managers
//!
//! # Invalidation Strategy
//!
//! Rather than eagerly invalidating ICs on every shape change, we use lazy
//! invalidation:
//!
//! 1. Shape changes bump the global shape version
//! 2. ICs record their creation version
//! 3. On IC access, if version mismatch → invalidate and re-populate
//!
//! This amortizes invalidation cost across accesses.

use prism_runtime::object::shape::ShapeId;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Shape Version
// =============================================================================

/// Global shape version counter.
///
/// Incremented on any shape transition that could affect IC validity.
/// ICs record their creation version and check against global on access.
static GLOBAL_SHAPE_VERSION: AtomicU64 = AtomicU64::new(0);

/// A shape version stamp.
///
/// Used to detect stale ICs that need invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ShapeVersion(u64);

impl ShapeVersion {
    /// Create a new version from a raw value.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the current global shape version.
    #[inline]
    pub fn current() -> Self {
        Self(GLOBAL_SHAPE_VERSION.load(Ordering::Acquire))
    }

    /// Increment the global shape version and return the new value.
    ///
    /// Call this when a shape transition occurs.
    #[inline]
    pub fn bump() -> Self {
        let new = GLOBAL_SHAPE_VERSION.fetch_add(1, Ordering::AcqRel) + 1;
        Self(new)
    }

    /// Get the raw version value.
    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Check if this version is stale compared to the current global version.
    #[inline]
    pub fn is_stale(self) -> bool {
        self.0 < GLOBAL_SHAPE_VERSION.load(Ordering::Acquire)
    }

    /// Check if this version matches the current global version.
    #[inline]
    pub fn is_current(self) -> bool {
        self.0 == GLOBAL_SHAPE_VERSION.load(Ordering::Acquire)
    }
}

impl Default for ShapeVersion {
    #[inline]
    fn default() -> Self {
        Self::current()
    }
}

// =============================================================================
// IC Dependency
// =============================================================================

/// Describes a dependency between an IC site and shapes.
///
/// When any of the watched shapes transitions, the IC must be invalidated.
#[derive(Debug, Clone)]
pub struct IcDependency {
    /// The shape this IC depends on.
    pub shape_id: ShapeId,

    /// The IC manager owning this dependency.
    /// Weak reference to avoid cycles (manager owns dependencies).
    pub manager_id: u64,

    /// Index of the IC site within the manager.
    pub site_index: u32,

    /// Version when this dependency was created.
    pub version: ShapeVersion,
}

impl IcDependency {
    /// Create a new dependency.
    #[inline]
    pub fn new(shape_id: ShapeId, manager_id: u64, site_index: u32) -> Self {
        Self {
            shape_id,
            manager_id,
            site_index,
            version: ShapeVersion::current(),
        }
    }

    /// Check if this dependency is stale.
    #[inline]
    pub fn is_stale(&self) -> bool {
        self.version.is_stale()
    }
}

// =============================================================================
// Invalidation Event
// =============================================================================

/// An event that triggers IC invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationReason {
    /// A new shape was created (transition from existing shape).
    ShapeTransition,
    /// A property was deleted.
    PropertyDeletion,
    /// An accessor (getter/setter) was installed.
    AccessorInstalled,
    /// The prototype chain was modified.
    PrototypeChange,
    /// Manual invalidation requested.
    Manual,
}

/// An invalidation event.
#[derive(Debug, Clone)]
pub struct InvalidationEvent {
    /// The shape that changed.
    pub shape_id: ShapeId,
    /// Reason for invalidation.
    pub reason: InvalidationReason,
    /// New version after this event.
    pub new_version: ShapeVersion,
}

impl InvalidationEvent {
    /// Create a new event and bump the global version.
    #[inline]
    pub fn new(shape_id: ShapeId, reason: InvalidationReason) -> Self {
        Self {
            shape_id,
            reason,
            new_version: ShapeVersion::bump(),
        }
    }
}

// =============================================================================
// IC Invalidator
// =============================================================================

/// Callback type for IC invalidation.
pub type InvalidationCallback = Box<dyn Fn(&InvalidationEvent) + Send + Sync>;

/// Coordinates IC invalidation across the runtime.
///
/// Maintains a registry of dependencies and handles batch invalidation.
/// Thread-safe via internal locking.
#[derive(Debug)]
pub struct IcInvalidator {
    /// Dependencies indexed by shape ID.
    /// Multiple ICs can depend on the same shape.
    dependencies: RwLock<HashMap<ShapeId, Vec<IcDependency>>>,

    /// Total number of dependencies tracked.
    dependency_count: AtomicU64,

    /// Total invalidation events processed.
    invalidation_count: AtomicU64,
}

impl IcInvalidator {
    /// Create a new invalidator.
    #[inline]
    pub fn new() -> Self {
        Self {
            dependencies: RwLock::new(HashMap::new()),
            dependency_count: AtomicU64::new(0),
            invalidation_count: AtomicU64::new(0),
        }
    }

    /// Register a dependency between an IC site and a shape.
    pub fn register_dependency(&self, dep: IcDependency) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        deps.entry(dep.shape_id).or_insert_with(Vec::new).push(dep);
        self.dependency_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Register multiple dependencies atomically.
    pub fn register_dependencies(&self, deps: impl IntoIterator<Item = IcDependency>) {
        let mut guard = self.dependencies.write().expect("Lock poisoned");
        let mut count = 0u64;
        for dep in deps {
            guard.entry(dep.shape_id).or_insert_with(Vec::new).push(dep);
            count += 1;
        }
        self.dependency_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Get all dependencies for a shape.
    pub fn get_dependencies(&self, shape_id: ShapeId) -> Vec<IcDependency> {
        let deps = self.dependencies.read().expect("Lock poisoned");
        deps.get(&shape_id).cloned().unwrap_or_default()
    }

    /// Remove dependencies for a specific manager.
    ///
    /// Call this when an IcManager is destroyed.
    pub fn remove_manager_dependencies(&self, manager_id: u64) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        let mut removed = 0u64;

        deps.retain(|_, v| {
            let before = v.len();
            v.retain(|d| d.manager_id != manager_id);
            removed += (before - v.len()) as u64;
            !v.is_empty()
        });

        if removed > 0 {
            // Saturating sub to handle race conditions
            let current = self.dependency_count.load(Ordering::Relaxed);
            self.dependency_count
                .store(current.saturating_sub(removed), Ordering::Relaxed);
        }
    }

    /// Invalidate all ICs depending on a shape.
    ///
    /// Returns the number of ICs invalidated.
    pub fn invalidate_shape(&self, event: InvalidationEvent) -> usize {
        self.invalidation_count.fetch_add(1, Ordering::Relaxed);

        let deps = self.dependencies.read().expect("Lock poisoned");
        let dependents = deps.get(&event.shape_id);

        if let Some(deps) = dependents {
            deps.len()
        } else {
            0
        }
    }

    /// Invalidate all ICs depending on any of the given shapes.
    ///
    /// Returns total number of ICs affected.
    pub fn invalidate_shapes(&self, shape_ids: &[ShapeId], reason: InvalidationReason) -> usize {
        if shape_ids.is_empty() {
            return 0;
        }

        let new_version = ShapeVersion::bump();
        let deps = self.dependencies.read().expect("Lock poisoned");
        let mut total = 0;

        for &shape_id in shape_ids {
            if let Some(shape_deps) = deps.get(&shape_id) {
                total += shape_deps.len();
            }
        }

        if total > 0 {
            self.invalidation_count.fetch_add(1, Ordering::Relaxed);
        }

        total
    }

    /// Prune stale dependencies.
    ///
    /// Removes dependencies whose version is older than `min_version`.
    /// Call periodically to prevent unbounded growth.
    pub fn prune_stale(&self, min_version: ShapeVersion) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        let mut removed = 0u64;

        deps.retain(|_, v| {
            let before = v.len();
            v.retain(|d| d.version >= min_version);
            removed += (before - v.len()) as u64;
            !v.is_empty()
        });

        if removed > 0 {
            let current = self.dependency_count.load(Ordering::Relaxed);
            self.dependency_count
                .store(current.saturating_sub(removed), Ordering::Relaxed);
        }
    }

    /// Clear all dependencies.
    pub fn clear(&self) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        deps.clear();
        self.dependency_count.store(0, Ordering::Relaxed);
    }

    /// Get total number of tracked dependencies.
    #[inline]
    pub fn dependency_count(&self) -> u64 {
        self.dependency_count.load(Ordering::Relaxed)
    }

    /// Get total number of invalidation events.
    #[inline]
    pub fn invalidation_count(&self) -> u64 {
        self.invalidation_count.load(Ordering::Relaxed)
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> InvalidatorStats {
        let deps = self.dependencies.read().expect("Lock poisoned");
        InvalidatorStats {
            dependency_count: self.dependency_count.load(Ordering::Relaxed),
            unique_shapes: deps.len() as u64,
            invalidation_count: self.invalidation_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for IcInvalidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the invalidator.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvalidatorStats {
    /// Total dependencies tracked.
    pub dependency_count: u64,
    /// Number of unique shapes with dependencies.
    pub unique_shapes: u64,
    /// Total invalidation events processed.
    pub invalidation_count: u64,
}

// =============================================================================
// Global Invalidator
// =============================================================================

use std::sync::OnceLock;

/// Global invalidator instance.
static GLOBAL_INVALIDATOR: OnceLock<IcInvalidator> = OnceLock::new();

/// Get the global IC invalidator.
#[inline]
pub fn global_invalidator() -> &'static IcInvalidator {
    GLOBAL_INVALIDATOR.get_or_init(IcInvalidator::new)
}

/// Initialize the global invalidator (call at startup).
pub fn init_invalidator() {
    let _ = global_invalidator();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
