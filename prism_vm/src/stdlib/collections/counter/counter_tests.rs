
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
fn test_new_creates_empty_counter() {
    let c = Counter::new();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
}

#[test]
fn test_from_iter() {
    let c = Counter::from_iter(vec![int(1), int(2), int(1)]);
    assert_eq!(c.get(&int(1)), 2);
    assert_eq!(c.get(&int(2)), 1);
}

#[test]
fn test_from_pairs() {
    let c = Counter::from_pairs(vec![(int(1), 5), (int(2), 3)]);
    assert_eq!(c.get(&int(1)), 5);
    assert_eq!(c.get(&int(2)), 3);
}

// =========================================================================
// Core Operation Tests
// =========================================================================

#[test]
fn test_get_returns_zero_for_missing() {
    let c = Counter::new();
    assert_eq!(c.get(&int(99)), 0);
}

#[test]
fn test_increment() {
    let mut c = Counter::new();
    c.increment(int(1));
    assert_eq!(c.get(&int(1)), 1);
    c.increment(int(1));
    assert_eq!(c.get(&int(1)), 2);
}

#[test]
fn test_set() {
    let mut c = Counter::new();
    c.set(int(1), 10);
    assert_eq!(c.get(&int(1)), 10);
}

#[test]
fn test_set_zero_removes() {
    let mut c = Counter::new();
    c.set(int(1), 5);
    c.set(int(1), 0);
    assert!(!c.contains(&int(1)));
}

#[test]
fn test_add_count() {
    let mut c = Counter::new();
    c.add_count(int(1), 5);
    c.add_count(int(1), 3);
    assert_eq!(c.get(&int(1)), 8);
}

#[test]
fn test_add_count_negative() {
    let mut c = Counter::new();
    c.add_count(int(1), 5);
    c.add_count(int(1), -3);
    assert_eq!(c.get(&int(1)), 2);
}

#[test]
fn test_update() {
    let mut c = Counter::new();
    c.update(vec![int(1), int(1), int(2)]);
    assert_eq!(c.get(&int(1)), 2);
    assert_eq!(c.get(&int(2)), 1);
}

#[test]
fn test_subtract() {
    let mut c = Counter::from_iter(vec![int(1), int(1), int(1)]);
    c.subtract(vec![int(1)]);
    assert_eq!(c.get(&int(1)), 2);
}

// =========================================================================
// Query Operation Tests
// =========================================================================

#[test]
fn test_total() {
    let c = Counter::from_pairs(vec![(int(1), 5), (int(2), 3)]);
    assert_eq!(c.total(), 8);
}

#[test]
fn test_len() {
    let c = Counter::from_iter(vec![int(1), int(2), int(1)]);
    assert_eq!(c.len(), 2);
}

#[test]
fn test_contains() {
    let c = Counter::from_iter(vec![int(1)]);
    assert!(c.contains(&int(1)));
    assert!(!c.contains(&int(99)));
}

#[test]
fn test_remove() {
    let mut c = Counter::from_pairs(vec![(int(1), 5)]);
    let removed = c.remove(&int(1));
    assert_eq!(removed, Some(5));
    assert!(!c.contains(&int(1)));
}

#[test]
fn test_clear() {
    let mut c = Counter::from_iter(vec![int(1), int(2), int(3)]);
    c.clear();
    assert!(c.is_empty());
}

// =========================================================================
// Most Common Tests
// =========================================================================

#[test]
fn test_most_common() {
    let c = Counter::from_pairs(vec![
        (str_val("a"), 5),
        (str_val("b"), 3),
        (str_val("c"), 8),
    ]);

    let mc = c.most_common();
    assert_eq!(mc.len(), 3);
    assert_eq!(mc[0].1, 8); // 'c' has highest count
    assert_eq!(mc[1].1, 5);
    assert_eq!(mc[2].1, 3);
}

#[test]
fn test_most_common_n() {
    let c = Counter::from_pairs(vec![(int(1), 10), (int(2), 5), (int(3), 3), (int(4), 1)]);

    let mc = c.most_common_n(2);
    assert_eq!(mc.len(), 2);
    assert_eq!(mc[0].1, 10);
    assert_eq!(mc[1].1, 5);
}

#[test]
fn test_elements_iterator() {
    let c = Counter::from_pairs(vec![(int(1), 2), (int(2), 3)]);
    let elements: Vec<_> = c.elements().collect();
    assert_eq!(elements.len(), 5);
}

// =========================================================================
// Set-like Operation Tests
// =========================================================================

#[test]
fn test_add_counters() {
    let c1 = Counter::from_pairs(vec![(int(1), 3), (int(2), 2)]);
    let c2 = Counter::from_pairs(vec![(int(1), 1), (int(3), 5)]);

    let sum = c1.add(&c2);
    assert_eq!(sum.get(&int(1)), 4);
    assert_eq!(sum.get(&int(2)), 2);
    assert_eq!(sum.get(&int(3)), 5);
}

#[test]
fn test_sub_counters() {
    let c1 = Counter::from_pairs(vec![(int(1), 5), (int(2), 3)]);
    let c2 = Counter::from_pairs(vec![(int(1), 2), (int(2), 1)]);

    let diff = c1.sub(&c2);
    assert_eq!(diff.get(&int(1)), 3);
    assert_eq!(diff.get(&int(2)), 2);
}

#[test]
fn test_intersection() {
    let c1 = Counter::from_pairs(vec![(int(1), 3), (int(2), 5)]);
    let c2 = Counter::from_pairs(vec![(int(1), 5), (int(2), 2)]);

    let intersect = c1.intersection(&c2);
    assert_eq!(intersect.get(&int(1)), 3); // min(3, 5)
    assert_eq!(intersect.get(&int(2)), 2); // min(5, 2)
}

#[test]
fn test_union() {
    let c1 = Counter::from_pairs(vec![(int(1), 3), (int(2), 5)]);
    let c2 = Counter::from_pairs(vec![(int(1), 5), (int(3), 4)]);

    let un = c1.union(&c2);
    assert_eq!(un.get(&int(1)), 5); // max(3, 5)
    assert_eq!(un.get(&int(2)), 5);
    assert_eq!(un.get(&int(3)), 4);
}

#[test]
fn test_positive() {
    let mut c = Counter::new();
    c.set(int(1), 5);
    c.set(int(2), -3);
    c.set(int(3), 0);

    c.positive();
    assert!(c.contains(&int(1)));
    assert!(!c.contains(&int(2)));
    assert!(!c.contains(&int(3)));
}

// =========================================================================
// Equality Tests
// =========================================================================

#[test]
fn test_equality() {
    let c1 = Counter::from_pairs(vec![(int(1), 5), (int(2), 3)]);
    let c2 = Counter::from_pairs(vec![(int(2), 3), (int(1), 5)]);
    assert_eq!(c1, c2);
}

#[test]
fn test_inequality_different_counts() {
    let c1 = Counter::from_pairs(vec![(int(1), 5)]);
    let c2 = Counter::from_pairs(vec![(int(1), 3)]);
    assert_ne!(c1, c2);
}

// =========================================================================
// String Value Tests
// =========================================================================

#[test]
fn test_string_counts() {
    let mut c = Counter::new();
    c.increment(str_val("hello"));
    c.increment(str_val("world"));
    c.increment(str_val("hello"));

    assert_eq!(c.get(&str_val("hello")), 2);
    assert_eq!(c.get(&str_val("world")), 1);
}

// =========================================================================
// Iterator Tests
// =========================================================================

#[test]
fn test_keys_iterator() {
    let c = Counter::from_iter(vec![int(1), int(2), int(3)]);
    let keys: Vec<_> = c.keys().collect();
    assert_eq!(keys.len(), 3);
}

#[test]
fn test_values_iterator() {
    let c = Counter::from_pairs(vec![(int(1), 5), (int(2), 3)]);
    let sum: i64 = c.values().sum();
    assert_eq!(sum, 8);
}

#[test]
fn test_iter() {
    let c = Counter::from_pairs(vec![(int(1), 5)]);
    let pairs: Vec<_> = c.iter().collect();
    assert_eq!(pairs.len(), 1);
    assert_eq!(*pairs[0].1, 5);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_none_values() {
    let mut c = Counter::new();
    c.increment(Value::none());
    c.increment(Value::none());
    assert_eq!(c.get(&Value::none()), 2);
}

#[test]
fn test_bool_values() {
    let mut c = Counter::new();
    c.increment(Value::bool(true));
    c.increment(Value::bool(false));
    c.increment(Value::bool(true));

    assert_eq!(c.get(&Value::bool(true)), 2);
    assert_eq!(c.get(&Value::bool(false)), 1);
}

#[test]
fn test_float_values() {
    let mut c = Counter::new();
    c.increment(Value::float(3.14));
    c.increment(Value::float(2.71));
    c.increment(Value::float(3.14));

    assert_eq!(c.get(&Value::float(3.14)), 2);
}

#[test]
fn test_negative_counts() {
    let mut c = Counter::from_pairs(vec![(int(1), 5)]);
    c.add_count(int(1), -10);
    assert_eq!(c.get(&int(1)), -5);
}

// =========================================================================
// Stress Tests
// =========================================================================

#[test]
fn test_stress_many_elements() {
    let mut c = Counter::new();
    for i in 0..1000 {
        c.increment(int(i % 100));
    }

    assert_eq!(c.len(), 100);
    assert_eq!(c.total(), 1000);

    for i in 0..100 {
        assert_eq!(c.get(&int(i)), 10);
    }
}

#[test]
fn test_stress_most_common() {
    let mut c = Counter::new();
    for i in 0..1000 {
        c.add_count(int(i), i);
    }

    let mc = c.most_common_n(10);
    assert_eq!(mc.len(), 10);
    assert_eq!(mc[0].1, 999);
}
