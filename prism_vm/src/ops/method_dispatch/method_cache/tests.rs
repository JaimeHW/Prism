use super::*;

#[test]
fn test_cache_miss_on_empty() {
    let cache = MethodCache::new();
    let result = cache.get(TypeId::OBJECT, 0x12345678, 0);
    assert!(result.is_none());
}

#[test]
fn test_cache_hit_after_insert() {
    let cache = MethodCache::new();
    let type_id = TypeId::OBJECT;
    let name_ptr = 0x12345678u64;
    let method = CachedMethod::simple(Value::none());

    cache.insert(type_id, name_ptr, 0, method);

    let result = cache.get(type_id, name_ptr, 0);
    assert!(result.is_some());
}

#[test]
fn test_different_types_different_entries() {
    let cache = MethodCache::new();
    let name_ptr = 0x12345678u64;

    let method1 = CachedMethod::simple(Value::bool(true));
    let method2 = CachedMethod::simple(Value::bool(false));

    cache.insert(TypeId::OBJECT, name_ptr, 0, method1);
    cache.insert(TypeId::STR, name_ptr, 0, method2);

    let result1 = cache.get(TypeId::OBJECT, name_ptr, 0).unwrap();
    let result2 = cache.get(TypeId::STR, name_ptr, 0).unwrap();

    // Different cached values
    assert_ne!(result1.method.as_bool(), result2.method.as_bool());
}

#[test]
fn test_different_names_different_entries() {
    let cache = MethodCache::new();
    let type_id = TypeId::OBJECT;

    let method1 = CachedMethod::simple(Value::bool(true));
    let method2 = CachedMethod::simple(Value::bool(false));

    cache.insert(type_id, 0x11111111, 0, method1);
    cache.insert(type_id, 0x22222222, 0, method2);

    let result1 = cache.get(type_id, 0x11111111, 0).unwrap();
    let result2 = cache.get(type_id, 0x22222222, 0).unwrap();

    assert_ne!(result1.method.as_bool(), result2.method.as_bool());
}

#[test]
fn test_invalidate_type() {
    let cache = MethodCache::new();
    let name_ptr = 0x12345678u64;

    cache.insert(
        TypeId::OBJECT,
        name_ptr,
        7,
        CachedMethod::simple(Value::none()),
    );
    cache.insert(
        TypeId::STR,
        name_ptr,
        3,
        CachedMethod::simple(Value::none()),
    );

    // Both should be present
    assert!(cache.get(TypeId::OBJECT, name_ptr, 7).is_some());
    assert!(cache.get(TypeId::STR, name_ptr, 3).is_some());

    // Invalidate OBJECT entries
    cache.invalidate_type(TypeId::OBJECT);

    // Version tags, not destructive removal, enforce correctness now.
    assert!(cache.get(TypeId::OBJECT, name_ptr, 8).is_none());
    assert!(cache.get(TypeId::STR, name_ptr, 3).is_some());
}

#[test]
fn test_version_mismatch_invalidates_hierarchy_entries_without_scanning() {
    let cache = MethodCache::new();
    let name_ptr = 0x12345678u64;

    cache.insert(
        TypeId::INT,
        name_ptr,
        11,
        CachedMethod::simple(Value::none()),
    );
    cache.insert(
        TypeId::BOOL,
        name_ptr,
        11,
        CachedMethod::simple(Value::none()),
    );
    cache.insert(
        TypeId::FLOAT,
        name_ptr,
        4,
        CachedMethod::simple(Value::none()),
    );

    cache.invalidate_type_hierarchy(TypeId::INT);

    assert!(cache.get(TypeId::INT, name_ptr, 12).is_none());
    assert!(cache.get(TypeId::BOOL, name_ptr, 12).is_none());
    assert!(cache.get(TypeId::FLOAT, name_ptr, 4).is_some());
}

#[test]
fn test_insert_replaces_same_key_when_version_changes() {
    let cache = MethodCache::new();
    let name_ptr = 0x12345678u64;

    cache.insert(
        TypeId::OBJECT,
        name_ptr,
        1,
        CachedMethod::simple(Value::bool(true)),
    );
    cache.insert(
        TypeId::OBJECT,
        name_ptr,
        2,
        CachedMethod::simple(Value::bool(false)),
    );

    assert!(cache.get(TypeId::OBJECT, name_ptr, 1).is_none());
    let current = cache
        .get(TypeId::OBJECT, name_ptr, 2)
        .expect("current version should remain cached");
    assert_eq!(current.method.as_bool(), Some(false));
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_clear() {
    let cache = MethodCache::new();

    cache.insert(
        TypeId::OBJECT,
        0x111,
        0,
        CachedMethod::simple(Value::none()),
    );
    cache.insert(TypeId::STR, 0x222, 0, CachedMethod::simple(Value::none()));

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
    cache.get(TypeId::OBJECT, 0x123, 0);
    let (hits, misses, _) = cache.stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 1);

    // Insert and hit
    cache.insert(
        TypeId::OBJECT,
        0x123,
        0,
        CachedMethod::simple(Value::none()),
    );
    cache.get(TypeId::OBJECT, 0x123, 0);
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
    cache.get(TypeId::OBJECT, 0x1, 0);
    cache.get(TypeId::OBJECT, 0x2, 0);
    assert_eq!(cache.hit_rate(), 0.0);

    // Insert and get hits
    cache.insert(TypeId::OBJECT, 0x1, 0, CachedMethod::simple(Value::none()));
    cache.get(TypeId::OBJECT, 0x1, 0);
    cache.get(TypeId::OBJECT, 0x1, 0);

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
