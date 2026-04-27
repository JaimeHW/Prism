//! Small Integer Cache for Python semantics.
//!
//! Python caches small integers in the range [-5, 256] so that:
//! - `a = 5; b = 5; a is b` returns True
//! - Common integers are pre-computed for performance
//!
//! # Performance Benefits
//!
//! 1. **Identity semantics**: Python's `is` operator works correctly for cached integers
//! 2. **Zero computation**: Cached integers avoid bit manipulation at runtime
//! 3. **Cache-friendly**: Hot integers are stored contiguously
//!
//! # Implementation
//!
//! We use a const-initialized static array containing pre-computed Value bitpatterns.
//! The cache is accessed via `SmallIntCache::get(i)` which performs a single bounds
//! check and array lookup.

use crate::Value;

/// Minimum cached small integer (inclusive).
pub const SMALL_INT_CACHE_MIN: i64 = -5;

/// Maximum cached small integer (inclusive).
pub const SMALL_INT_CACHE_MAX: i64 = 256;

/// Total number of cached small integers.
pub const SMALL_INT_CACHE_SIZE: usize = (SMALL_INT_CACHE_MAX - SMALL_INT_CACHE_MIN + 1) as usize;

/// Static array of pre-computed small integer Values.
///
/// Index 0 = -5, Index 5 = 0, Index 261 = 256
static SMALL_INT_CACHE: [Value; SMALL_INT_CACHE_SIZE] = {
    let mut cache = [Value::none(); SMALL_INT_CACHE_SIZE];
    let mut i = 0;
    while i < SMALL_INT_CACHE_SIZE {
        let val = SMALL_INT_CACHE_MIN + i as i64;
        // Safe because all values in [-5, 256] fit in small int range
        cache[i] = Value::int_unchecked(val);
        i += 1;
    }
    cache
};

/// Small integer cache for fast integer Value creation.
///
/// Provides O(1) lookup for integers in the range [-5, 256].
pub struct SmallIntCache;

impl SmallIntCache {
    /// Get a cached small integer, or None if out of range.
    ///
    /// # Performance
    ///
    /// This is a single bounds check + array index, approximately:
    /// - 2-3 instructions on the hot path
    /// - Always returns a reference to static memory (no allocation)
    ///
    /// # Example
    ///
    /// ```
    /// use prism_core::small_int_cache::SmallIntCache;
    ///
    /// // Cached (fast path)
    /// let five = SmallIntCache::get(5);
    /// assert!(five.is_some());
    ///
    /// // Not cached (returns None)
    /// let big = SmallIntCache::get(1000);
    /// assert!(big.is_none());
    /// ```
    #[inline]
    pub fn get(value: i64) -> Option<Value> {
        if value >= SMALL_INT_CACHE_MIN && value <= SMALL_INT_CACHE_MAX {
            let index = (value - SMALL_INT_CACHE_MIN) as usize;
            Some(SMALL_INT_CACHE[index])
        } else {
            None
        }
    }

    /// Get a cached small integer, panicking if out of range.
    ///
    /// # Panics
    ///
    /// Panics if value is not in [-5, 256].
    #[inline]
    pub fn get_unchecked(value: i64) -> Value {
        debug_assert!(
            value >= SMALL_INT_CACHE_MIN && value <= SMALL_INT_CACHE_MAX,
            "Integer {} is outside small int cache range [{}, {}]",
            value,
            SMALL_INT_CACHE_MIN,
            SMALL_INT_CACHE_MAX
        );
        let index = (value - SMALL_INT_CACHE_MIN) as usize;
        SMALL_INT_CACHE[index]
    }

    /// Check if a value is in the cached range.
    #[inline]
    pub const fn is_cached(value: i64) -> bool {
        value >= SMALL_INT_CACHE_MIN && value <= SMALL_INT_CACHE_MAX
    }

    /// Get the cache index for a value (assuming it's in range).
    #[inline]
    pub const fn index_of(value: i64) -> usize {
        debug_assert!(Self::is_cached(value));
        (value - SMALL_INT_CACHE_MIN) as usize
    }

    /// Get the minimum cached value.
    #[inline]
    pub const fn min() -> i64 {
        SMALL_INT_CACHE_MIN
    }

    /// Get the maximum cached value.
    #[inline]
    pub const fn max() -> i64 {
        SMALL_INT_CACHE_MAX
    }

    /// Get the number of cached values.
    #[inline]
    pub const fn size() -> usize {
        SMALL_INT_CACHE_SIZE
    }

    /// Get a direct reference to the cache array (for JIT integration).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of the program.
    /// Only access indices in [0, SMALL_INT_CACHE_SIZE).
    #[inline]
    pub fn cache_ptr() -> *const Value {
        SMALL_INT_CACHE.as_ptr()
    }
}
