use super::*;

#[test]
fn test_cache_constants() {
    assert_eq!(SMALL_INT_CACHE_MIN, -5);
    assert_eq!(SMALL_INT_CACHE_MAX, 256);
    assert_eq!(SMALL_INT_CACHE_SIZE, 262);
}

#[test]
fn test_cache_zero() {
    let cached = SmallIntCache::get(0).unwrap();
    assert!(cached.is_int());
    assert_eq!(cached.as_int(), Some(0));
}

#[test]
fn test_cache_positive() {
    for i in 0..=256 {
        let cached = SmallIntCache::get(i).expect(&format!("Should cache {}", i));
        assert_eq!(cached.as_int(), Some(i), "Mismatch for {}", i);
    }
}

#[test]
fn test_cache_negative() {
    for i in -5..0 {
        let cached = SmallIntCache::get(i).expect(&format!("Should cache {}", i));
        assert_eq!(cached.as_int(), Some(i), "Mismatch for {}", i);
    }
}

#[test]
fn test_cache_boundaries() {
    // Min boundary
    assert!(SmallIntCache::get(-5).is_some());
    assert!(SmallIntCache::get(-6).is_none());

    // Max boundary
    assert!(SmallIntCache::get(256).is_some());
    assert!(SmallIntCache::get(257).is_none());
}

#[test]
fn test_cache_miss() {
    assert!(SmallIntCache::get(-100).is_none());
    assert!(SmallIntCache::get(1000).is_none());
    assert!(SmallIntCache::get(i64::MAX).is_none());
    assert!(SmallIntCache::get(i64::MIN).is_none());
}

#[test]
fn test_is_cached() {
    assert!(SmallIntCache::is_cached(0));
    assert!(SmallIntCache::is_cached(-5));
    assert!(SmallIntCache::is_cached(256));
    assert!(!SmallIntCache::is_cached(-6));
    assert!(!SmallIntCache::is_cached(257));
}

#[test]
fn test_index_of() {
    assert_eq!(SmallIntCache::index_of(-5), 0);
    assert_eq!(SmallIntCache::index_of(0), 5);
    assert_eq!(SmallIntCache::index_of(1), 6);
    assert_eq!(SmallIntCache::index_of(256), 261);
}

#[test]
fn test_get_unchecked() {
    for i in -5..=256 {
        let cached = SmallIntCache::get_unchecked(i);
        assert_eq!(cached.as_int(), Some(i));
    }
}

#[test]
fn test_cache_identity() {
    // Verify that cached values have identical bit patterns
    let a = SmallIntCache::get(42).unwrap();
    let b = SmallIntCache::get(42).unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn test_cache_vs_direct() {
    // Cached values should equal directly constructed values
    for i in -5..=256 {
        let cached = SmallIntCache::get(i).unwrap();
        let direct = Value::int(i).unwrap();
        assert_eq!(cached, direct, "Mismatch for {}", i);
        assert_eq!(
            cached.to_bits(),
            direct.to_bits(),
            "Bits mismatch for {}",
            i
        );
    }
}

#[test]
fn test_cache_ptr() {
    let ptr = SmallIntCache::cache_ptr();
    assert!(!ptr.is_null());

    // Verify we can read through the pointer
    unsafe {
        let zero = *ptr.add(5); // index 5 = value 0
        assert_eq!(zero.as_int(), Some(0));
    }
}

#[test]
fn test_cache_static_initialization() {
    // Verify all cached values are correctly initialized
    for i in 0..SMALL_INT_CACHE_SIZE {
        let expected_value = SMALL_INT_CACHE_MIN + i as i64;
        assert_eq!(
            SMALL_INT_CACHE[i].as_int(),
            Some(expected_value),
            "Index {} should contain {}",
            i,
            expected_value
        );
    }
}

#[test]
fn test_python_common_values() {
    // Verify commonly used Python integers are cached
    let common_values = [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // small numbers
        100, 200, 255, 256, // byte-related
    ];

    for &val in &common_values {
        assert!(
            SmallIntCache::get(val).is_some(),
            "Common value {} should be cached",
            val
        );
    }
}

#[test]
fn test_loop_counter_values() {
    // Loop counters and common ranges should be cached
    for i in 0..100 {
        assert!(
            SmallIntCache::get(i).is_some(),
            "Loop counter value {} should be cached",
            i
        );
    }
}

// =========================================================================
// Performance-focused tests
// =========================================================================

#[test]
fn test_cache_lookup_equivalent_to_construction() {
    // Both paths should produce identical results
    for i in -5..=256 {
        let from_cache = SmallIntCache::get(i).unwrap();
        let from_constructor = Value::int(i).unwrap();

        // Value equality
        assert_eq!(from_cache, from_constructor);

        // Bit-level equality (important for identity semantics)
        assert_eq!(from_cache.to_bits(), from_constructor.to_bits());
    }
}

#[test]
fn test_helpers() {
    assert_eq!(SmallIntCache::min(), -5);
    assert_eq!(SmallIntCache::max(), 256);
    assert_eq!(SmallIntCache::size(), 262);
}
