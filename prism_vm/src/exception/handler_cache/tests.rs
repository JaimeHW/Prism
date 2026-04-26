use super::*;

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Creation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_new_is_empty() {
    let cache = InlineHandlerCache::new();
    assert!(cache.is_empty());
    assert!(!cache.is_valid());
}

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Size Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_size() {
    // Must fit in 8 bytes for Frame padding efficiency
    assert_eq!(std::mem::size_of::<InlineHandlerCache>(), 8);
}

#[test]
fn test_cache_alignment() {
    // 4-byte alignment for efficient access
    assert_eq!(std::mem::align_of::<InlineHandlerCache>(), 4);
}

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Lookup Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_try_get_empty() {
    let mut cache = InlineHandlerCache::new();
    assert_eq!(cache.try_get(100), None);
}

#[test]
fn test_cache_try_get_after_record() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);

    assert_eq!(cache.try_get(100), Some(5));
}

#[test]
fn test_cache_try_get_wrong_pc() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);

    assert_eq!(cache.try_get(200), None);
}

#[test]
fn test_cache_try_get_increments_hit_count() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);

    assert_eq!(cache.hit_count(), 0);
    cache.try_get(100);
    assert_eq!(cache.hit_count(), 1);
    cache.try_get(100);
    assert_eq!(cache.hit_count(), 2);
}

#[test]
fn test_cache_hit_count_saturates() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    cache.hit_count = MAX_HIT_COUNT;

    cache.try_get(100);
    assert_eq!(cache.hit_count(), MAX_HIT_COUNT);
}

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Record Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_record_sets_valid() {
    let mut cache = InlineHandlerCache::new();
    assert!(!cache.is_valid());

    cache.record(100, 5);
    assert!(cache.is_valid());
}

#[test]
fn test_cache_record_overwrites() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    cache.record(200, 10);

    assert_eq!(cache.try_get(100), None);
    assert_eq!(cache.try_get(200), Some(10));
}

#[test]
fn test_cache_record_resets_hit_count() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    cache.try_get(100);
    cache.try_get(100);
    assert_eq!(cache.hit_count(), 2);

    cache.record(200, 10);
    assert_eq!(cache.hit_count(), 0);
}

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Miss Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_record_miss() {
    let mut cache = InlineHandlerCache::new();
    cache.record_miss(100);

    assert!(!cache.is_empty());
    assert!(!cache.is_valid());
    assert_eq!(cache.try_get(100), None);
}

#[test]
fn test_cache_record_miss_prevents_lookup() {
    let mut cache = InlineHandlerCache::new();
    cache.record_miss(100);

    // Even though PC matches, handler is NO_CACHED_HANDLER
    assert_eq!(cache.try_get(100), None);
}

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Invalidate Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_invalidate() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    assert!(cache.is_valid());

    cache.invalidate();
    assert!(cache.is_empty());
    assert!(!cache.is_valid());
}

#[test]
fn test_cache_invalidate_clears_hit_count() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    cache.try_get(100);
    cache.try_get(100);

    cache.invalidate();
    assert_eq!(cache.hit_count(), 0);
}

// ════════════════════════════════════════════════════════════════════════
// InlineHandlerCache Accessor Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_cached_pc_empty() {
    let cache = InlineHandlerCache::new();
    assert_eq!(cache.cached_pc(), None);
}

#[test]
fn test_cache_cached_pc_after_record() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    assert_eq!(cache.cached_pc(), Some(100));
}

#[test]
fn test_cache_cached_handler_empty() {
    let cache = InlineHandlerCache::new();
    assert_eq!(cache.cached_handler(), None);
}

#[test]
fn test_cache_cached_handler_after_record() {
    let mut cache = InlineHandlerCache::new();
    cache.record(100, 5);
    assert_eq!(cache.cached_handler(), Some(5));
}

// ════════════════════════════════════════════════════════════════════════
// HandlerCacheStats Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_stats_new() {
    let stats = HandlerCacheStats::new();
    assert_eq!(stats.lookups, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_stats_record_hit() {
    let mut stats = HandlerCacheStats::new();
    stats.record_hit();

    assert_eq!(stats.lookups, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_stats_record_miss() {
    let mut stats = HandlerCacheStats::new();
    stats.record_miss();

    assert_eq!(stats.lookups, 1);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 1);
}

#[test]
fn test_stats_hit_rate_zero() {
    let stats = HandlerCacheStats::new();
    assert_eq!(stats.hit_rate(), 0.0);
}

#[test]
fn test_stats_hit_rate_hundred() {
    let mut stats = HandlerCacheStats::new();
    stats.record_hit();
    stats.record_hit();

    assert_eq!(stats.hit_rate(), 100.0);
}

#[test]
fn test_stats_hit_rate_fifty() {
    let mut stats = HandlerCacheStats::new();
    stats.record_hit();
    stats.record_miss();

    assert_eq!(stats.hit_rate(), 50.0);
}

#[test]
fn test_stats_merge() {
    let mut stats1 = HandlerCacheStats::new();
    stats1.record_hit();
    stats1.record_miss();

    let mut stats2 = HandlerCacheStats::new();
    stats2.record_hit();
    stats2.record_hit();

    stats1.merge(&stats2);

    assert_eq!(stats1.lookups, 4);
    assert_eq!(stats1.hits, 3);
    assert_eq!(stats1.misses, 1);
}

#[test]
fn test_stats_reset() {
    let mut stats = HandlerCacheStats::new();
    stats.record_hit();
    stats.record_miss();

    stats.reset();

    assert_eq!(stats.lookups, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

// ════════════════════════════════════════════════════════════════════════
// MultiLevelCache Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_level_new() {
    let cache = MultiLevelCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.level_count(), 0);
}

#[test]
fn test_multi_level_push_level() {
    let mut cache = MultiLevelCache::new();
    cache.push_level();

    assert!(!cache.is_empty());
    assert_eq!(cache.level_count(), 1);
}

#[test]
fn test_multi_level_pop_level() {
    let mut cache = MultiLevelCache::new();
    cache.push_level();
    cache.pop_level();

    assert!(cache.is_empty());
    assert_eq!(cache.level_count(), 0);
}

#[test]
fn test_multi_level_pop_empty() {
    let mut cache = MultiLevelCache::new();
    cache.pop_level(); // Should not panic
    assert!(cache.is_empty());
}

#[test]
fn test_multi_level_max_levels() {
    let mut cache = MultiLevelCache::new();

    for _ in 0..10 {
        cache.push_level();
    }

    assert_eq!(cache.level_count(), MultiLevelCache::MAX_LEVELS);
}

#[test]
fn test_multi_level_record_and_get() {
    let mut cache = MultiLevelCache::new();
    cache.push_level();
    cache.record(100, 5);

    assert_eq!(cache.try_get(100), Some(5));
}

#[test]
fn test_multi_level_nested_lookup() {
    let mut cache = MultiLevelCache::new();

    cache.push_level();
    cache.record(100, 1);

    cache.push_level();
    cache.record(200, 2);

    // Should find both
    assert_eq!(cache.try_get(200), Some(2));
    assert_eq!(cache.try_get(100), Some(1));
}

#[test]
fn test_multi_level_pop_preserves_outer() {
    let mut cache = MultiLevelCache::new();

    cache.push_level();
    cache.record(100, 1);

    cache.push_level();
    cache.record(200, 2);

    cache.pop_level();

    assert_eq!(cache.try_get(100), Some(1));
    assert_eq!(cache.try_get(200), None);
}

#[test]
fn test_multi_level_invalidate_all() {
    let mut cache = MultiLevelCache::new();

    cache.push_level();
    cache.record(100, 1);
    cache.push_level();
    cache.record(200, 2);

    cache.invalidate_all();

    assert!(cache.is_empty());
    assert_eq!(cache.try_get(100), None);
    assert_eq!(cache.try_get(200), None);
}

#[test]
fn test_multi_level_size() {
    // 4 caches * 8 bytes + 1 byte count (padded to 4)
    let size = std::mem::size_of::<MultiLevelCache>();
    assert!(
        size <= 40,
        "MultiLevelCache should be <= 40 bytes, got {}",
        size
    );
}
