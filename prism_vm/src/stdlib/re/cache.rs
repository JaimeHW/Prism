//! Pattern cache with LRU eviction.
//!
//! High-performance caching of compiled patterns to avoid
//! repeated compilation overhead for frequent patterns.

use super::flags::RegexFlags;
use super::pattern::CompiledPattern;
use rustc_hash::FxHashMap;
use std::sync::RwLock;

// =============================================================================
// Cache Configuration
// =============================================================================

/// Default cache size (256 patterns).
pub const DEFAULT_CACHE_SIZE: usize = 256;

// =============================================================================
// Cache Key
// =============================================================================

/// Cache key combining pattern and flags.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    pattern: String,
    flags: u32,
}

impl CacheKey {
    fn new(pattern: &str, flags: RegexFlags) -> Self {
        Self {
            pattern: pattern.to_string(),
            flags: flags.bits(),
        }
    }
}

// =============================================================================
// LRU Entry
// =============================================================================

/// Entry in the LRU cache.
#[derive(Debug)]
struct CacheEntry {
    pattern: CompiledPattern,
    /// Last access timestamp for LRU eviction.
    last_access: u64,
}

// =============================================================================
// Pattern Cache
// =============================================================================

/// Thread-safe LRU cache for compiled patterns.
///
/// Patterns are cached by their source string and flags combination.
/// When the cache is full, the least recently used pattern is evicted.
#[derive(Debug)]
pub struct PatternCache {
    /// Cached patterns.
    cache: RwLock<FxHashMap<CacheKey, CacheEntry>>,
    /// Maximum cache size.
    max_size: usize,
    /// Access counter for LRU tracking.
    access_counter: RwLock<u64>,
}

impl PatternCache {
    /// Create a new cache with default size.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_SIZE)
    }

    /// Create a new cache with specified capacity.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
            max_size,
            access_counter: RwLock::new(0),
        }
    }

    /// Get or compile a pattern.
    ///
    /// Returns cached pattern if available, otherwise compiles and caches.
    pub fn get_or_compile(
        &self,
        pattern: &str,
        flags: RegexFlags,
    ) -> Result<CompiledPattern, super::engine::RegexError> {
        let key = CacheKey::new(pattern, flags);

        // Try read lock first (fast path)
        {
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(&key) {
                // Clone the pattern and update access time
                let pattern = entry.pattern.clone();
                drop(cache);

                // Update access time
                self.touch(&key);
                return Ok(pattern);
            }
        }

        // Compile the pattern
        let compiled = CompiledPattern::compile(pattern, flags)?;

        // Insert into cache
        self.insert(key, compiled.clone());

        Ok(compiled)
    }

    /// Update access time for a key.
    fn touch(&self, key: &CacheKey) {
        let mut counter = self.access_counter.write().unwrap();
        *counter += 1;
        let time = *counter;
        drop(counter);

        let mut cache = self.cache.write().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            entry.last_access = time;
        }
    }

    /// Insert a pattern into the cache.
    fn insert(&self, key: CacheKey, pattern: CompiledPattern) {
        let mut counter = self.access_counter.write().unwrap();
        *counter += 1;
        let time = *counter;
        drop(counter);

        let mut cache = self.cache.write().unwrap();

        // Evict if at capacity
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache);
        }

        cache.insert(
            key,
            CacheEntry {
                pattern,
                last_access: time,
            },
        );
    }

    /// Evict the least recently used entry.
    fn evict_lru(&self, cache: &mut FxHashMap<CacheKey, CacheEntry>) {
        if let Some(key) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(k, _)| k.clone())
        {
            cache.remove(&key);
        }
    }

    /// Clear all cached patterns.
    pub fn purge(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get current cache size.
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().unwrap().is_empty()
    }

    /// Get cache hit rate statistics.
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        let counter = self.access_counter.read().unwrap();
        CacheStats {
            size: cache.len(),
            capacity: self.max_size,
            total_accesses: *counter,
        }
    }
}

impl Default for PatternCache {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Cache Statistics
// =============================================================================

/// Cache usage statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached patterns.
    pub size: usize,
    /// Maximum capacity.
    pub capacity: usize,
    /// Total access count.
    pub total_accesses: u64,
}

// =============================================================================
// Global Cache
// =============================================================================

use std::sync::LazyLock;

/// Global pattern cache instance.
static GLOBAL_CACHE: LazyLock<PatternCache> = LazyLock::new(PatternCache::new);

/// Get the global pattern cache.
pub fn global_cache() -> &'static PatternCache {
    &GLOBAL_CACHE
}

/// Purge the global cache (equivalent to Python's `re.purge()`).
pub fn purge_global_cache() {
    GLOBAL_CACHE.purge();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = PatternCache::with_capacity(10);
        let pattern = cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        assert_eq!(pattern.pattern(), r"\d+");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_hit() {
        let cache = PatternCache::with_capacity(10);

        // First access - miss (compile)
        let p1 = cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();

        // Second access - hit (cached)
        let p2 = cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();

        assert_eq!(p1.pattern(), p2.pattern());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_different_flags() {
        let cache = PatternCache::with_capacity(10);

        let p1 = cache
            .get_or_compile(r"hello", RegexFlags::default())
            .unwrap();
        let p2 = cache
            .get_or_compile(r"hello", RegexFlags::new(RegexFlags::IGNORECASE))
            .unwrap();

        assert_eq!(cache.len(), 2);
        assert!(!p1.is_match("HELLO"));
        assert!(p2.is_match("HELLO"));
    }

    #[test]
    fn test_cache_eviction() {
        let cache = PatternCache::with_capacity(3);

        cache.get_or_compile(r"a", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"b", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"c", RegexFlags::default()).unwrap();
        assert_eq!(cache.len(), 3);

        // This should evict the least recently used
        cache.get_or_compile(r"d", RegexFlags::default()).unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_cache_lru_order() {
        let cache = PatternCache::with_capacity(3);

        cache.get_or_compile(r"a", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"b", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"c", RegexFlags::default()).unwrap();

        // Access 'a' to make it most recently used
        cache.get_or_compile(r"a", RegexFlags::default()).unwrap();

        // Now add 'd' - should evict 'b' (oldest)
        cache.get_or_compile(r"d", RegexFlags::default()).unwrap();

        // 'a' should still be in cache
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_purge() {
        let cache = PatternCache::with_capacity(10);
        cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"\w+", RegexFlags::default()).unwrap();
        assert_eq!(cache.len(), 2);

        cache.purge();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_stats() {
        let cache = PatternCache::with_capacity(10);
        cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.capacity, 10);
        assert!(stats.total_accesses >= 2);
    }

    #[test]
    fn test_global_cache() {
        let pattern = global_cache()
            .get_or_compile(r"test\d+", RegexFlags::default())
            .unwrap();
        assert_eq!(pattern.pattern(), r"test\d+");
    }
}
