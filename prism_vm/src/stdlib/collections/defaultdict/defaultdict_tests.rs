
use super::*;
use prism_core::intern::intern;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn str_val(s: &str) -> Value {
    Value::string(intern(s))
}

// =========================================================================
// Construction Tests
// =========================================================================

#[test]
fn test_new_creates_empty() {
    let d = DefaultDict::new();
    assert!(d.is_empty());
    assert_eq!(d.len(), 0);
}

#[test]
fn test_with_factory() {
    let d = DefaultDict::with_factory(DefaultFactory::Int);
    assert_eq!(*d.default_factory(), DefaultFactory::Int);
}

// =========================================================================
// Get/Set Tests
// =========================================================================

#[test]
fn test_set_and_get() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));
    assert_eq!(d.get(&int(1)).and_then(|v| v.as_int()), Some(100));
}

#[test]
fn test_get_missing_no_factory() {
    let d = DefaultDict::new();
    assert_eq!(d.get(&int(99)), None);
}

#[test]
fn test_get_or_insert_no_factory_returns_none() {
    let mut d = DefaultDict::new();
    assert_eq!(d.get_or_insert(&int(1)), None);
}

#[test]
fn test_get_or_insert_with_int_factory() {
    let mut d = DefaultDict::with_factory(DefaultFactory::Int);
    let val = d.get_or_insert(&int(1)).unwrap();
    assert_eq!(val.as_int(), Some(0));
    // Should be inserted now
    assert!(d.contains(&int(1)));
}

#[test]
fn test_get_or_insert_with_float_factory() {
    let mut d = DefaultDict::with_factory(DefaultFactory::Float);
    let val = d.get_or_insert(&int(1)).unwrap();
    assert_eq!(val.as_float(), Some(0.0));
}

#[test]
fn test_get_or_insert_with_bool_factory() {
    let mut d = DefaultDict::with_factory(DefaultFactory::Bool);
    let val = d.get_or_insert(&int(1)).unwrap();
    assert_eq!(val.as_bool(), Some(false));
}

#[test]
fn test_get_returns_existing() {
    let mut d = DefaultDict::with_factory(DefaultFactory::Int);
    d.set(int(1), int(42));

    let val = d.get_or_insert(&int(1)).unwrap();
    assert_eq!(val.as_int(), Some(42));
}

// =========================================================================
// Contains/Remove Tests
// =========================================================================

#[test]
fn test_contains() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));

    assert!(d.contains(&int(1)));
    assert!(!d.contains(&int(99)));
}

#[test]
fn test_remove() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));

    let removed = d.remove(&int(1));
    assert_eq!(removed.and_then(|v| v.as_int()), Some(100));
    assert!(!d.contains(&int(1)));
}

#[test]
fn test_remove_missing() {
    let mut d = DefaultDict::new();
    assert_eq!(d.remove(&int(99)), None);
}

#[test]
fn test_clear() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));
    d.set(int(2), int(200));
    d.clear();

    assert!(d.is_empty());
}

// =========================================================================
// Factory Change Tests
// =========================================================================

#[test]
fn test_set_default_factory() {
    let mut d = DefaultDict::new();
    assert_eq!(*d.default_factory(), DefaultFactory::None);

    d.set_default_factory(DefaultFactory::Float);
    assert_eq!(*d.default_factory(), DefaultFactory::Float);
}

// =========================================================================
// Iterator Tests
// =========================================================================

#[test]
fn test_keys_iterator() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));
    d.set(int(2), int(200));

    let keys: Vec<_> = d.keys().collect();
    assert_eq!(keys.len(), 2);
}

#[test]
fn test_values_iterator() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));
    d.set(int(2), int(200));

    let sum: i64 = d.values().filter_map(|v| v.as_int()).sum();
    assert_eq!(sum, 300);
}

#[test]
fn test_iter() {
    let mut d = DefaultDict::new();
    d.set(int(1), int(100));

    let pairs: Vec<_> = d.iter().collect();
    assert_eq!(pairs.len(), 1);
}

// =========================================================================
// String Key Tests
// =========================================================================

#[test]
fn test_string_keys() {
    let mut d = DefaultDict::with_factory(DefaultFactory::Int);
    d.set(str_val("key1"), int(1));

    assert_eq!(d.get(&str_val("key1")).and_then(|v| v.as_int()), Some(1));

    let val = d.get_or_insert(&str_val("key2")).unwrap();
    assert_eq!(val.as_int(), Some(0));
}

// =========================================================================
// Counting Pattern Tests
// =========================================================================

#[test]
fn test_counting_pattern() {
    // Simulating: defaultdict(int)
    let mut d = DefaultDict::with_factory(DefaultFactory::Int);

    // Count occurrences
    let items = vec![int(1), int(2), int(1), int(3), int(1), int(2)];
    for item in items {
        // Get current count
        let count = d.get_or_insert(&item).unwrap().as_int().unwrap();
        // Increment
        d.set(item, Value::int_unchecked(count + 1));
    }

    assert_eq!(d.get(&int(1)).and_then(|v| v.as_int()), Some(3));
    assert_eq!(d.get(&int(2)).and_then(|v| v.as_int()), Some(2));
    assert_eq!(d.get(&int(3)).and_then(|v| v.as_int()), Some(1));
}

// =========================================================================
// Stress Tests
// =========================================================================

#[test]
fn test_stress_many_keys() {
    let mut d = DefaultDict::with_factory(DefaultFactory::Int);

    for i in 0..1000 {
        d.get_or_insert(&int(i));
    }

    assert_eq!(d.len(), 1000);

    for i in 0..1000 {
        assert!(d.contains(&int(i)));
    }
}
