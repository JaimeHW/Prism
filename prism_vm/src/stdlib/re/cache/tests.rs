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
