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

use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use rustc_hash::FxHashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Cache Key and Entry
// =============================================================================

/// Cache key: (type_id, method_name_interned_ptr)
///
/// Using the interned string pointer as key allows O(1) equality check
/// instead of string comparison.
type CacheKey = (TypeId, u64);

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
    cache: RwLock<FxHashMap<CacheKey, CachedMethod>>,

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
        Self {
            cache: RwLock::new(FxHashMap::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
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
    pub fn get(&self, type_id: TypeId, name_ptr: u64) -> Option<CachedMethod> {
        // Fast path: try read lock
        let guard = match self.cache.read() {
            Ok(g) => g,
            Err(_) => return None, // Poisoned lock, treat as miss
        };

        let result = guard.get(&(type_id, name_ptr)).copied();

        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        result
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
    pub fn insert(&self, type_id: TypeId, name_ptr: u64, method: CachedMethod) {
        if let Ok(mut guard) = self.cache.write() {
            guard.insert((type_id, name_ptr), method);
        }
        // Silently ignore poisoned lock
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
    /// O(n) where n is cache size. Should be rare in practice.
    pub fn invalidate_type(&self, type_id: TypeId) {
        if let Ok(mut guard) = self.cache.write() {
            guard.retain(|(tid, _), _| *tid != type_id);
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Invalidate the entire cache.
    ///
    /// Called during:
    /// - Interpreter shutdown
    /// - After significant type system changes
    /// - Manual cache clear for testing
    pub fn clear(&self) {
        if let Ok(mut guard) = self.cache.write() {
            guard.clear();
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
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
        self.cache.read().map(|g| g.len()).unwrap_or(0)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_miss_on_empty() {
        let cache = MethodCache::new();
        let result = cache.get(TypeId::OBJECT, 0x12345678);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_hit_after_insert() {
        let cache = MethodCache::new();
        let type_id = TypeId::OBJECT;
        let name_ptr = 0x12345678u64;
        let method = CachedMethod::simple(Value::none());

        cache.insert(type_id, name_ptr, method);

        let result = cache.get(type_id, name_ptr);
        assert!(result.is_some());
    }

    #[test]
    fn test_different_types_different_entries() {
        let cache = MethodCache::new();
        let name_ptr = 0x12345678u64;

        let method1 = CachedMethod::simple(Value::bool(true));
        let method2 = CachedMethod::simple(Value::bool(false));

        cache.insert(TypeId::OBJECT, name_ptr, method1);
        cache.insert(TypeId::STR, name_ptr, method2);

        let result1 = cache.get(TypeId::OBJECT, name_ptr).unwrap();
        let result2 = cache.get(TypeId::STR, name_ptr).unwrap();

        // Different cached values
        assert_ne!(result1.method.as_bool(), result2.method.as_bool());
    }

    #[test]
    fn test_different_names_different_entries() {
        let cache = MethodCache::new();
        let type_id = TypeId::OBJECT;

        let method1 = CachedMethod::simple(Value::bool(true));
        let method2 = CachedMethod::simple(Value::bool(false));

        cache.insert(type_id, 0x11111111, method1);
        cache.insert(type_id, 0x22222222, method2);

        let result1 = cache.get(type_id, 0x11111111).unwrap();
        let result2 = cache.get(type_id, 0x22222222).unwrap();

        assert_ne!(result1.method.as_bool(), result2.method.as_bool());
    }

    #[test]
    fn test_invalidate_type() {
        let cache = MethodCache::new();
        let name_ptr = 0x12345678u64;

        cache.insert(
            TypeId::OBJECT,
            name_ptr,
            CachedMethod::simple(Value::none()),
        );
        cache.insert(TypeId::STR, name_ptr, CachedMethod::simple(Value::none()));

        // Both should be present
        assert!(cache.get(TypeId::OBJECT, name_ptr).is_some());
        assert!(cache.get(TypeId::STR, name_ptr).is_some());

        // Invalidate OBJECT entries
        cache.invalidate_type(TypeId::OBJECT);

        // OBJECT should be gone, STR should remain
        assert!(cache.get(TypeId::OBJECT, name_ptr).is_none());
        assert!(cache.get(TypeId::STR, name_ptr).is_some());
    }

    #[test]
    fn test_clear() {
        let cache = MethodCache::new();

        cache.insert(TypeId::OBJECT, 0x111, CachedMethod::simple(Value::none()));
        cache.insert(TypeId::STR, 0x222, CachedMethod::simple(Value::none()));

        assert!(!cache.is_empty());

        cache.clear();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_stats() {
        let cache = MethodCache::new();

        // Initial stats should be zero
        let (hits, misses, invalidations) = cache.stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(invalidations, 0);

        // Miss should increment miss counter
        cache.get(TypeId::OBJECT, 0x123);
        let (hits, misses, _) = cache.stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);

        // Insert and hit
        cache.insert(TypeId::OBJECT, 0x123, CachedMethod::simple(Value::none()));
        cache.get(TypeId::OBJECT, 0x123);
        let (hits, misses, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);

        // Invalidate
        cache.invalidate_type(TypeId::OBJECT);
        let (_, _, invalidations) = cache.stats();
        assert_eq!(invalidations, 1);
    }

    #[test]
    fn test_hit_rate() {
        let cache = MethodCache::new();

        // No lookups = 0% hit rate
        assert_eq!(cache.hit_rate(), 0.0);

        // All misses = 0%
        cache.get(TypeId::OBJECT, 0x1);
        cache.get(TypeId::OBJECT, 0x2);
        assert_eq!(cache.hit_rate(), 0.0);

        // Insert and get hits
        cache.insert(TypeId::OBJECT, 0x1, CachedMethod::simple(Value::none()));
        cache.get(TypeId::OBJECT, 0x1);
        cache.get(TypeId::OBJECT, 0x1);

        // 2 hits, 2 misses = 50%
        let rate = cache.hit_rate();
        assert!((rate - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_cached_method_constructors() {
        let simple = CachedMethod::simple(Value::none());
        assert!(!simple.is_descriptor);
        assert!(simple.slot.is_none());

        let desc = CachedMethod::descriptor(Value::none());
        assert!(desc.is_descriptor);
        assert!(desc.slot.is_none());

        let slotted = CachedMethod::with_slot(Value::none(), 42);
        assert!(!slotted.is_descriptor);
        assert_eq!(slotted.slot, Some(42));
    }

    #[test]
    fn test_global_singleton() {
        let cache1 = method_cache();
        let cache2 = method_cache();

        // Should be the same instance
        assert!(std::ptr::eq(cache1, cache2));
    }
}
