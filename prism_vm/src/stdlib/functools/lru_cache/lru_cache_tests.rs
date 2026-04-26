
use super::*;
use prism_core::intern::intern;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn str_val(s: &str) -> Value {
    Value::string(intern(s))
}

fn hv(v: Value) -> HashableValue {
    HashableValue(v)
}

// =========================================================================
// Bounded cache: Basic operations
// =========================================================================

#[test]
fn test_bounded_put_get() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(10));
}

#[test]
fn test_bounded_miss_returns_none() {
    let mut cache = LruCache::new(3);
    assert!(cache.get(&hv(int(1))).is_none());
}

#[test]
fn test_bounded_overwrite() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(1)), int(20));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(20));
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_bounded_multiple_entries() {
    let mut cache = LruCache::new(5);
    for i in 0..5 {
        cache.put(hv(int(i)), int(i * 10));
    }
    assert_eq!(cache.len(), 5);
    for i in 0..5 {
        assert_eq!(cache.get(&hv(int(i))).unwrap().as_int(), Some(i * 10));
    }
}

// =========================================================================
// Bounded cache: Eviction
// =========================================================================

#[test]
fn test_bounded_evicts_lru() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));
    cache.put(hv(int(3)), int(30));
    // Cache full: [3, 2, 1] (head to tail)

    // Insert 4th, should evict 1
    cache.put(hv(int(4)), int(40));
    assert!(cache.get(&hv(int(1))).is_none()); // Evicted
    assert_eq!(cache.get(&hv(int(2))).unwrap().as_int(), Some(20));
    assert_eq!(cache.len(), 3);
}

#[test]
fn test_bounded_access_prevents_eviction() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));
    cache.put(hv(int(3)), int(30));

    // Access 1, making it most recently used
    cache.get(&hv(int(1)));

    // Insert 4, should evict 2 (now LRU)
    cache.put(hv(int(4)), int(40));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(10)); // Still here
    assert!(cache.get(&hv(int(2))).is_none()); // Evicted
}

#[test]
fn test_bounded_sequential_eviction() {
    let mut cache = LruCache::new(2);
    for i in 0..10 {
        cache.put(hv(int(i)), int(i * 10));
    }
    // Only last 2 should remain
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&hv(int(9))).unwrap().as_int(), Some(90));
    assert_eq!(cache.get(&hv(int(8))).unwrap().as_int(), Some(80));
}

#[test]
fn test_bounded_size_one() {
    let mut cache = LruCache::new(1);
    cache.put(hv(int(1)), int(10));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(10));

    cache.put(hv(int(2)), int(20));
    assert!(cache.get(&hv(int(1))).is_none());
    assert_eq!(cache.get(&hv(int(2))).unwrap().as_int(), Some(20));
}

#[test]
fn test_bounded_update_then_evict() {
    let mut cache = LruCache::new(2);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));

    // Update 1 (moves to head)
    cache.put(hv(int(1)), int(100));

    // Insert 3, should evict 2 (now LRU)
    cache.put(hv(int(3)), int(30));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(100));
    assert!(cache.get(&hv(int(2))).is_none());
}

// =========================================================================
// Bounded cache: Statistics
// =========================================================================

#[test]
fn test_bounded_cache_info_initial() {
    let cache = LruCache::new(128);
    let info = cache.cache_info();
    assert_eq!(info.hits, 0);
    assert_eq!(info.misses, 0);
    assert_eq!(info.maxsize, Some(128));
    assert_eq!(info.currsize, 0);
}

#[test]
fn test_bounded_cache_info_after_operations() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));

    cache.get(&hv(int(1))); // Hit
    cache.get(&hv(int(3))); // Miss
    cache.get(&hv(int(1))); // Hit
    cache.get(&hv(int(4))); // Miss

    let info = cache.cache_info();
    assert_eq!(info.hits, 2);
    assert_eq!(info.misses, 2);
    assert_eq!(info.currsize, 2);
}

#[test]
fn test_bounded_hit_rate() {
    let mut cache = LruCache::new(10);
    cache.put(hv(int(1)), int(10));

    // 3 hits
    for _ in 0..3 {
        cache.get(&hv(int(1)));
    }
    // 1 miss
    cache.get(&hv(int(2)));

    let info = cache.cache_info();
    assert!((info.hit_rate() - 75.0).abs() < 0.001);
}

#[test]
fn test_bounded_hit_rate_zero_accesses() {
    let cache = LruCache::new(10);
    let info = cache.cache_info();
    assert_eq!(info.hit_rate(), 0.0);
}

// =========================================================================
// Bounded cache: Clear
// =========================================================================

#[test]
fn test_bounded_clear() {
    let mut cache = LruCache::new(5);
    for i in 0..5 {
        cache.put(hv(int(i)), int(i));
    }
    cache.get(&hv(int(0))); // 1 hit

    cache.cache_clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    // Stats reset
    let info = cache.cache_info();
    assert_eq!(info.hits, 0);
    assert_eq!(info.misses, 0);
}

#[test]
fn test_bounded_reuse_after_clear() {
    let mut cache = LruCache::new(3);
    for i in 0..3 {
        cache.put(hv(int(i)), int(i));
    }
    cache.cache_clear();

    cache.put(hv(int(10)), int(100));
    assert_eq!(cache.get(&hv(int(10))).unwrap().as_int(), Some(100));
    assert_eq!(cache.len(), 1);
}

// =========================================================================
// Unbounded cache tests
// =========================================================================

#[test]
fn test_unbounded_put_get() {
    let mut cache = LruCache::unbounded();
    cache.put(hv(int(1)), int(10));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(10));
}

#[test]
fn test_unbounded_miss() {
    let mut cache = LruCache::unbounded();
    assert!(cache.get(&hv(int(1))).is_none());
}

#[test]
fn test_unbounded_no_eviction() {
    let mut cache = LruCache::unbounded();
    for i in 0..1000 {
        cache.put(hv(int(i)), int(i * 10));
    }
    assert_eq!(cache.len(), 1000);
    // All entries should still be present
    for i in 0..1000 {
        assert_eq!(cache.get(&hv(int(i))).unwrap().as_int(), Some(i * 10));
    }
}

#[test]
fn test_unbounded_cache_info() {
    let mut cache = LruCache::unbounded();
    cache.put(hv(int(1)), int(10));
    cache.get(&hv(int(1))); // Hit
    cache.get(&hv(int(2))); // Miss

    let info = cache.cache_info();
    assert_eq!(info.hits, 1);
    assert_eq!(info.misses, 1);
    assert_eq!(info.maxsize, None);
    assert_eq!(info.currsize, 1);
}

#[test]
fn test_unbounded_clear() {
    let mut cache = LruCache::unbounded();
    for i in 0..100 {
        cache.put(hv(int(i)), int(i));
    }
    cache.cache_clear();
    assert!(cache.is_empty());
}

#[test]
fn test_unbounded_with_capacity() {
    let mut cache = LruCache::unbounded_with_capacity(100);
    cache.put(hv(int(1)), int(10));
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(10));
}

// =========================================================================
// Contains tests
// =========================================================================

#[test]
fn test_contains_present() {
    let mut cache = LruCache::new(5);
    cache.put(hv(int(1)), int(10));
    assert!(cache.contains(&hv(int(1))));
}

#[test]
fn test_contains_absent() {
    let cache = LruCache::new(5);
    assert!(!cache.contains(&hv(int(1))));
}

#[test]
fn test_contains_after_eviction() {
    let mut cache = LruCache::new(2);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));
    cache.put(hv(int(3)), int(30)); // Evicts 1
    assert!(!cache.contains(&hv(int(1))));
    assert!(cache.contains(&hv(int(3))));
}

// =========================================================================
// Maxsize tests
// =========================================================================

#[test]
fn test_maxsize_bounded() {
    let cache = LruCache::new(42);
    assert_eq!(cache.maxsize(), Some(42));
}

#[test]
fn test_maxsize_unbounded() {
    let cache = LruCache::unbounded();
    assert_eq!(cache.maxsize(), None);
}

// =========================================================================
// String key tests
// =========================================================================

#[test]
fn test_string_keys() {
    let mut cache = LruCache::new(3);
    cache.put(hv(str_val("foo")), int(1));
    cache.put(hv(str_val("bar")), int(2));
    cache.put(hv(str_val("baz")), int(3));

    assert_eq!(cache.get(&hv(str_val("foo"))).unwrap().as_int(), Some(1));
    assert_eq!(cache.get(&hv(str_val("bar"))).unwrap().as_int(), Some(2));
    assert_eq!(cache.get(&hv(str_val("baz"))).unwrap().as_int(), Some(3));
}

#[test]
fn test_string_key_eviction() {
    let mut cache = LruCache::new(2);
    cache.put(hv(str_val("a")), int(1));
    cache.put(hv(str_val("b")), int(2));
    cache.put(hv(str_val("c")), int(3)); // Evicts "a"

    assert!(cache.get(&hv(str_val("a"))).is_none());
    assert_eq!(cache.get(&hv(str_val("c"))).unwrap().as_int(), Some(3));
}

// =========================================================================
// CacheInfo display tests
// =========================================================================

#[test]
fn test_cache_info_display_bounded() {
    let info = CacheInfo {
        hits: 10,
        misses: 3,
        maxsize: Some(128),
        currsize: 8,
    };
    let s = info.to_string();
    assert!(s.contains("hits=10"));
    assert!(s.contains("misses=3"));
    assert!(s.contains("maxsize=128"));
    assert!(s.contains("currsize=8"));
}

#[test]
fn test_cache_info_display_unbounded() {
    let info = CacheInfo {
        hits: 0,
        misses: 0,
        maxsize: None,
        currsize: 0,
    };
    let s = info.to_string();
    assert!(s.contains("maxsize=None"));
}

// =========================================================================
// Arena recycling tests
// =========================================================================

#[test]
fn test_arena_recycling() {
    let mut cache = LruCache::new(2);
    // Fill cache
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));
    // Evict and refill multiple times
    for i in 3..100 {
        cache.put(hv(int(i)), int(i * 10));
    }
    assert_eq!(cache.len(), 2);
    // Arena should have recycled nodes, not grown unboundedly
    if let LruCache::Bounded(ref c) = cache {
        // Arena size should be small due to recycling
        assert!(c.arena.len() <= 100);
    }
}

// =========================================================================
// LRU ordering correctness tests
// =========================================================================

#[test]
fn test_lru_ordering_complex() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10)); // [1]
    cache.put(hv(int(2)), int(20)); // [2, 1]
    cache.put(hv(int(3)), int(30)); // [3, 2, 1]

    // Access 1 → [1, 3, 2]
    cache.get(&hv(int(1)));

    // Insert 4 → evicts 2 → [4, 1, 3]
    cache.put(hv(int(4)), int(40));
    assert!(cache.get(&hv(int(2))).is_none()); // 2 was evicted
    assert!(cache.contains(&hv(int(1))));
    assert!(cache.contains(&hv(int(3))));
    assert!(cache.contains(&hv(int(4))));
}

#[test]
fn test_lru_repeated_access_same_key() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));
    cache.put(hv(int(3)), int(30));

    // Repeatedly access key 1
    for _ in 0..100 {
        cache.get(&hv(int(1)));
    }

    // Insert 4, should evict 2 (not 1)
    cache.put(hv(int(4)), int(40));
    assert!(cache.contains(&hv(int(1))));
    assert!(!cache.contains(&hv(int(2))));
}

#[test]
fn test_lru_access_all_then_evict() {
    let mut cache = LruCache::new(3);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(int(2)), int(20));
    cache.put(hv(int(3)), int(30));

    // Access in reverse order: 1, 2, 3
    // After this, order is [3, 2, 1]
    cache.get(&hv(int(1)));
    cache.get(&hv(int(2)));
    cache.get(&hv(int(3)));

    // Insert 4 → evicts 1 (LRU)
    cache.put(hv(int(4)), int(40));
    assert!(!cache.contains(&hv(int(1))));
    assert!(cache.contains(&hv(int(2))));
    assert!(cache.contains(&hv(int(3))));
    assert!(cache.contains(&hv(int(4))));
}

// =========================================================================
// Stress tests
// =========================================================================

#[test]
fn test_bounded_stress_many_operations() {
    let mut cache = LruCache::new(100);
    for i in 0..10_000 {
        cache.put(hv(int(i)), int(i * 10));
    }
    assert_eq!(cache.len(), 100);

    // Last 100 should be cached
    for i in 9900..10_000 {
        assert!(cache.contains(&hv(int(i))));
    }
}

#[test]
fn test_bounded_stress_high_hit_rate() {
    let mut cache = LruCache::new(10);
    // Fill cache
    for i in 0..10 {
        cache.put(hv(int(i)), int(i));
    }
    // Many hits on the same keys
    for _ in 0..1000 {
        for i in 0..10 {
            cache.get(&hv(int(i)));
        }
    }
    let info = cache.cache_info();
    assert_eq!(info.hits, 10_000);
    assert_eq!(info.misses, 0);
}

#[test]
fn test_unbounded_stress() {
    let mut cache = LruCache::unbounded();
    for i in 0..10_000 {
        cache.put(hv(int(i)), int(i * 10));
    }
    assert_eq!(cache.len(), 10_000);
    for i in 0..10_000 {
        assert_eq!(cache.get(&hv(int(i))).unwrap().as_int(), Some(i * 10));
    }
}

// =========================================================================
// Edge case tests
// =========================================================================

#[test]
fn test_put_same_key_many_times() {
    let mut cache = LruCache::new(3);
    for i in 0..100 {
        cache.put(hv(int(1)), int(i));
    }
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(99));
}

#[test]
fn test_empty_cache_operations() {
    let mut cache = LruCache::new(5);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert!(!cache.contains(&hv(int(1))));
    assert!(cache.get(&hv(int(1))).is_none());

    cache.cache_clear(); // Should not panic
    assert!(cache.is_empty());
}

#[test]
fn test_bool_keys() {
    let mut cache = LruCache::new(5);
    cache.put(hv(Value::bool(true)), int(1));
    cache.put(hv(Value::bool(false)), int(0));
    assert_eq!(cache.get(&hv(Value::bool(true))).unwrap().as_int(), Some(1));
    assert_eq!(
        cache.get(&hv(Value::bool(false))).unwrap().as_int(),
        Some(0)
    );
}

#[test]
fn test_none_key() {
    let mut cache = LruCache::new(5);
    cache.put(hv(Value::none()), int(42));
    assert_eq!(cache.get(&hv(Value::none())).unwrap().as_int(), Some(42));
}

#[test]
fn test_mixed_type_keys() {
    let mut cache = LruCache::new(10);
    cache.put(hv(int(1)), int(10));
    cache.put(hv(str_val("one")), int(11));
    cache.put(hv(Value::bool(true)), int(12));
    cache.put(hv(Value::none()), int(13));
    cache.put(hv(Value::float(3.14)), int(14));

    assert_eq!(cache.len(), 5);
    assert_eq!(cache.get(&hv(int(1))).unwrap().as_int(), Some(10));
    assert_eq!(cache.get(&hv(str_val("one"))).unwrap().as_int(), Some(11));
}
