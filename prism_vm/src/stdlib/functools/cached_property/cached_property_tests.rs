
use super::*;
use prism_core::intern::intern;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn str_val(s: &str) -> Value {
    Value::string(intern(s))
}

// =========================================================================
// Construction tests
// =========================================================================

#[test]
fn test_new() {
    let cp = CachedProperty::new(int(42));
    assert_eq!(cp.func().as_int(), Some(42));
    assert!(!cp.is_cached());
    assert!(cp.attr_name().is_none());
}

#[test]
fn test_with_name() {
    let cp = CachedProperty::with_name(int(0), "my_prop".into());
    assert_eq!(cp.attr_name(), Some("my_prop"));
}

#[test]
fn test_with_doc() {
    let cp = CachedProperty::with_doc(int(0), "prop".into(), str_val("A property."));
    assert!(cp.doc().is_some());
}

// =========================================================================
// Caching behavior tests
// =========================================================================

#[test]
fn test_get_or_compute_first_call() {
    let mut cp = CachedProperty::new(int(0));

    let result = cp.get_or_compute(|_func| int(42));
    assert_eq!(result.as_int(), Some(42));
    assert!(cp.is_cached());
}

#[test]
fn test_get_or_compute_returns_cached() {
    let mut cp = CachedProperty::new(int(0));

    // First call
    cp.get_or_compute(|_| int(42));

    // Second call — should NOT call the closure
    let mut called = false;
    let result = cp.get_or_compute(|_| {
        called = true;
        int(999)
    });

    // Should return cached value, not new value
    assert_eq!(result.as_int(), Some(42));
    assert!(!called);
}

#[test]
fn test_get_cached_none() {
    let cp = CachedProperty::new(int(0));
    assert!(cp.get_cached().is_none());
}

#[test]
fn test_get_cached_some() {
    let mut cp = CachedProperty::new(int(0));
    cp.get_or_compute(|_| int(100));
    assert_eq!(cp.get_cached().unwrap().as_int(), Some(100));
}

// =========================================================================
// Invalidation tests
// =========================================================================

#[test]
fn test_invalidate() {
    let mut cp = CachedProperty::new(int(0));
    cp.get_or_compute(|_| int(42));
    assert!(cp.is_cached());

    cp.invalidate();
    assert!(!cp.is_cached());
}

#[test]
fn test_invalidate_recomputes() {
    let mut cp = CachedProperty::new(int(0));
    let mut counter = 0;

    // First computation
    cp.get_or_compute(|_| {
        counter += 1;
        int(counter)
    });
    assert_eq!(cp.get_cached().unwrap().as_int(), Some(1));

    // Invalidate
    cp.invalidate();

    // Recompute
    cp.get_or_compute(|_| {
        counter += 1;
        int(counter)
    });
    assert_eq!(cp.get_cached().unwrap().as_int(), Some(2));
}

#[test]
fn test_invalidate_uncached_noop() {
    let mut cp = CachedProperty::new(int(0));
    cp.invalidate(); // Should not panic
    assert!(!cp.is_cached());
}

// =========================================================================
// Direct set tests
// =========================================================================

#[test]
fn test_set_cached() {
    let mut cp = CachedProperty::new(int(0));
    cp.set_cached(int(99));
    assert!(cp.is_cached());
    assert_eq!(cp.get_cached().unwrap().as_int(), Some(99));
}

#[test]
fn test_set_cached_overwrites_computed() {
    let mut cp = CachedProperty::new(int(0));
    cp.get_or_compute(|_| int(42));
    cp.set_cached(int(99));
    assert_eq!(cp.get_cached().unwrap().as_int(), Some(99));
}

// =========================================================================
// set_name tests
// =========================================================================

#[test]
fn test_set_name() {
    let mut cp = CachedProperty::new(int(0));
    assert!(cp.attr_name().is_none());
    cp.set_name("my_property".into());
    assert_eq!(cp.attr_name(), Some("my_property"));
}

// =========================================================================
// Clone tests
// =========================================================================

#[test]
fn test_clone_uncached() {
    let cp = CachedProperty::with_name(int(0), "prop".into());
    let clone = cp.clone();
    assert!(!clone.is_cached());
    assert_eq!(clone.attr_name(), Some("prop"));
}

#[test]
fn test_clone_cached() {
    let mut cp = CachedProperty::new(int(0));
    cp.get_or_compute(|_| int(42));

    let clone = cp.clone();
    assert!(clone.is_cached());
    assert_eq!(clone.get_cached().unwrap().as_int(), Some(42));
}

// =========================================================================
// Real-world pattern tests
// =========================================================================

#[test]
fn test_expensive_computation_cached() {
    let mut cp = CachedProperty::new(int(0));
    let mut call_count = 0;

    // Simulate expensive computation
    for _ in 0..100 {
        cp.get_or_compute(|_| {
            call_count += 1;
            // "Expensive" computation
            int(42 * 42)
        });
    }

    // Should only compute once
    assert_eq!(call_count, 1);
    assert_eq!(cp.get_cached().unwrap().as_int(), Some(1764));
}

#[test]
fn test_cached_property_with_string_value() {
    let mut cp = CachedProperty::new(int(0));
    let result = cp.get_or_compute(|_| str_val("computed_value"));
    assert!(result.is_string());
}

#[test]
fn test_cached_property_with_none_value() {
    let mut cp = CachedProperty::new(int(0));
    let result = cp.get_or_compute(|_| Value::none());
    assert!(result.is_none());
    // None should still count as cached
    assert!(cp.is_cached());
}

#[test]
fn test_cached_property_with_bool_value() {
    let mut cp = CachedProperty::new(int(0));
    let result = cp.get_or_compute(|_| Value::bool(true));
    assert_eq!(result.as_bool(), Some(true));
}
