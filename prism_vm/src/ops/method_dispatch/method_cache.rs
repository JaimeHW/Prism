//! Class-level method cache for O(1) method resolution.
//!
//! Python's attribute lookup involves MRO traversal which is O(n) in class
//! hierarchy depth. This cache memoizes resolved methods per type+name pair,
//! dramatically accelerating repeated method calls on the same types.
//!
//! # Cache Hierarchy
//!
//! 1. **Inline Cache (IC)** - Per call-site, handles monomorphic cases
//! 2. **Method Cache (this)** - Global, handles polymorphic cases
//! 3. **MRO Traversal** - Full lookup, populates caches
//!
//! # Invalidation
//!
//! The cache must be invalidated when:
//! - A class's `__dict__` is modified
//! - A class gains/loses a base class
//! - A descriptor is added/removed
//!
//! # Thread Safety
//!
//! Uses `RwLock` for concurrent read access with exclusive write access.
//! Read path is lock-free after first access via `RwLock::read()`.

use parking_lot::RwLock;
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Cache Key and Entry
// =============================================================================

/// Cache key: (type_id, method_name_interned_ptr)
///
/// Using the interned string pointer as key allows O(1) equality check
/// instead of string comparison.
type CacheKey = (TypeId, u64);

const METHOD_CACHE_SHARDS: usize = 32;

/// Cached method resolution result.
///
/// Contains all information needed to invoke the method without
/// repeating the resolution process.
#[derive(Clone, Copy, Debug)]
pub struct CachedMethod {
    /// The resolved method/function value (pointer to FunctionObject, etc.)
    pub method: Value,

    /// Whether this is a data descriptor requiring `__get__` call.
    ///
    /// Data descriptors (with `__set__` or `__delete__`) take precedence
    /// over instance attributes and require runtime binding.
    pub is_descriptor: bool,

    /// Slot index in type object for direct access.
    ///
    /// Some methods (like `__init__`, `__repr__`) have dedicated slots
    /// in the type object for O(1) access without hash lookup.
    pub slot: Option<u16>,
}

impl CachedMethod {
    /// Create a simple cached method (not a descriptor, no slot)
    #[inline]
    pub fn simple(method: Value) -> Self {
        Self {
            method,
            is_descriptor: false,
            slot: None,
        }
    }

    /// Create a cached descriptor method
    #[inline]
    pub fn descriptor(method: Value) -> Self {
        Self {
            method,
            is_descriptor: true,
            slot: None,
        }
    }

    /// Create a cached method with slot optimization
    #[inline]
    pub fn with_slot(method: Value, slot: u16) -> Self {
        Self {
            method,
            is_descriptor: false,
            slot: Some(slot),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct VersionedCachedMethod {
    version: u64,
    cached: CachedMethod,
}

// =============================================================================
// Method Cache
// =============================================================================

/// Global method resolution cache.
///
/// Thread-safe cache that memoizes method resolution results across all
/// call sites. This complements per-site inline caches by handling cases
/// where the same type+method is called from multiple locations.
pub struct MethodCache {
    /// The actual cache storage
    cache: Box<[RwLock<FxHashMap<CacheKey, VersionedCachedMethod>>]>,

    /// Cache hit counter (for profiling and tuning)
    hits: AtomicU64,

    /// Cache miss counter
    misses: AtomicU64,

    /// Invalidation counter (tracks class mutations)
    invalidations: AtomicU64,
}

impl MethodCache {
    /// Create a new empty method cache.
    pub fn new() -> Self {
        let mut cache = Vec::with_capacity(METHOD_CACHE_SHARDS);
        cache.resize_with(METHOD_CACHE_SHARDS, || RwLock::new(FxHashMap::default()));
        Self {
            cache: cache.into_boxed_slice(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    #[inline]
    fn shard(
        &self,
        type_id: TypeId,
        name_ptr: u64,
    ) -> &RwLock<FxHashMap<CacheKey, VersionedCachedMethod>> {
        let hash = (type_id.raw() as usize).wrapping_mul(1_114_111usize) ^ (name_ptr as usize);
        &self.cache[hash % METHOD_CACHE_SHARDS]
    }

    /// Look up a cached method resolution.
    ///
    /// # Arguments
    ///
    /// * `type_id` - The TypeId of the object's type
    /// * `name_ptr` - Interned string pointer for the method name
    ///
    /// # Returns
    ///
    /// `Some(CachedMethod)` if found, `None` if cache miss.
    ///
    /// # Performance
    ///
    /// - Cache hit: ~10 cycles (RwLock read + hash lookup)
    /// - Cache miss: Updates miss counter only
    #[inline]
    pub fn get(&self, type_id: TypeId, name_ptr: u64, version: u64) -> Option<CachedMethod> {
        let guard = self.shard(type_id, name_ptr).read();
        let result = guard
            .get(&(type_id, name_ptr))
            .and_then(|entry| (entry.version == version).then_some(entry.cached));

        if let Some(cached) = result {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(cached);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        None
    }

    /// Insert a method resolution into the cache.
    ///
    /// # Arguments
    ///
    /// * `type_id` - The TypeId of the object's type
    /// * `name_ptr` - Interned string pointer for the method name
    /// * `method` - The resolved method information
    ///
    /// # Thread Safety
    ///
    /// Uses write lock; blocks concurrent reads during insert.
    /// Insert is relatively rare (only on cache miss), so contention is low.
    #[inline]
    pub fn insert(&self, type_id: TypeId, name_ptr: u64, version: u64, method: CachedMethod) {
        self.shard(type_id, name_ptr).write().insert(
            (type_id, name_ptr),
            VersionedCachedMethod {
                version,
                cached: method,
            },
        );
    }

    /// Invalidate all cache entries for a specific type.
    ///
    /// Must be called when:
    /// - Type's `__dict__` is modified
    /// - Type gains/loses base classes
    /// - Descriptor is added/removed from type hierarchy
    ///
    /// # Performance
    ///
    /// Published class-version tags make destructive sweeps unnecessary here.
    pub fn invalidate_type(&self, _type_id: TypeId) {
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Invalidate cache entries for a type and all of its subclasses.
    ///
    /// This preserves correctness when a class dictionary changes, since any
    /// subclass may inherit the mutated attribute through its MRO.
    pub fn invalidate_type_hierarchy(&self, _type_id: TypeId) {
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Invalidate the entire cache.
    ///
    /// Called during:
    /// - Interpreter shutdown
    /// - After significant type system changes
    /// - Manual cache clear for testing
    pub fn clear(&self) {
        for shard in self.cache.iter() {
            shard.write().clear();
        }
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get cache statistics for profiling.
    ///
    /// # Returns
    ///
    /// Tuple of (hits, misses, invalidations)
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.invalidations.load(Ordering::Relaxed),
        )
    }

    /// Calculate hit rate as a percentage.
    ///
    /// Returns 0.0 if no lookups have been performed.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Get current cache size (number of entries).
    pub fn len(&self) -> usize {
        self.cache.iter().map(|shard| shard.read().len()).sum()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MethodCache {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Global Singleton
// =============================================================================

use std::sync::OnceLock;

/// Global method cache singleton.
///
/// Initialized lazily on first access. Thread-safe via internal RwLock.
static METHOD_CACHE: OnceLock<MethodCache> = OnceLock::new();

/// Get reference to the global method cache.
///
/// # Example
///
/// ```ignore
/// use prism_vm::ops::method_dispatch::method_cache;
///
/// let cache = method_cache();
/// if let Some(cached) = cache.get(type_id, name_ptr) {
///     // Use cached method
/// }
/// ```
#[inline]
pub fn method_cache() -> &'static MethodCache {
    METHOD_CACHE.get_or_init(MethodCache::new)
}
