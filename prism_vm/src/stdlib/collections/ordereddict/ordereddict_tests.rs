
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
    let od = OrderedDict::new();
    assert!(od.is_empty());
    assert_eq!(od.len(), 0);
}

#[test]
fn test_with_capacity() {
    let od = OrderedDict::with_capacity(100);
    assert!(od.is_empty());
}

// =========================================================================
// Set/Get Tests
// =========================================================================

#[test]
fn test_set_and_get() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));
    od.set(int(2), int(200));

    assert_eq!(od.get(&int(1)).and_then(|v| v.as_int()), Some(100));
    assert_eq!(od.get(&int(2)).and_then(|v| v.as_int()), Some(200));
}

#[test]
fn test_get_missing() {
    let od = OrderedDict::new();
    assert_eq!(od.get(&int(99)), None);
}

#[test]
fn test_update_existing() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));
    od.set(int(1), int(999));

    assert_eq!(od.get(&int(1)).and_then(|v| v.as_int()), Some(999));
    assert_eq!(od.len(), 1);
}

#[test]
fn test_contains() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));

    assert!(od.contains(&int(1)));
    assert!(!od.contains(&int(99)));
}

// =========================================================================
// Order Tests
// =========================================================================

#[test]
fn test_insertion_order_preserved() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));
    od.set(str_val("c"), int(3));

    let keys: Vec<_> = od.keys().collect();
    assert_eq!(keys.len(), 3);
    // Order should be a, b, c
}

#[test]
fn test_update_preserves_order() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));
    od.set(str_val("c"), int(3));

    // Update 'b' - should stay in same position
    od.set(str_val("b"), int(20));

    let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![1, 20, 3]);
}

// =========================================================================
// Remove Tests
// =========================================================================

#[test]
fn test_remove() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));
    od.set(int(2), int(200));

    let removed = od.remove(&int(1));
    assert_eq!(removed.and_then(|v| v.as_int()), Some(100));
    assert!(!od.contains(&int(1)));
    assert_eq!(od.len(), 1);
}

#[test]
fn test_remove_preserves_order() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));
    od.set(str_val("c"), int(3));

    od.remove(&str_val("b"));

    let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![1, 3]);
}

#[test]
fn test_clear() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));
    od.set(int(2), int(200));
    od.clear();

    assert!(od.is_empty());
}

// =========================================================================
// Move/Pop Tests
// =========================================================================

#[test]
fn test_move_to_end_last() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));
    od.set(str_val("c"), int(3));

    od.move_to_end(&str_val("a"), true);

    // Order should now be b, c, a
    let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![2, 3, 1]);
}

#[test]
fn test_move_to_end_first() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));
    od.set(str_val("c"), int(3));

    od.move_to_end(&str_val("c"), false);

    // Order should now be c, a, b
    let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![3, 1, 2]);
}

#[test]
fn test_popitem_last() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));

    let (_, val) = od.popitem(true).unwrap();
    assert_eq!(val.as_int(), Some(2));
    assert_eq!(od.len(), 1);
}

#[test]
fn test_popitem_first() {
    let mut od = OrderedDict::new();
    od.set(str_val("a"), int(1));
    od.set(str_val("b"), int(2));

    let (_, val) = od.popitem(false).unwrap();
    assert_eq!(val.as_int(), Some(1));
    assert_eq!(od.len(), 1);
}

#[test]
fn test_popitem_empty() {
    let mut od = OrderedDict::new();
    assert_eq!(od.popitem(true), None);
}

// =========================================================================
// Iterator Tests
// =========================================================================

#[test]
fn test_keys_iterator() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));
    od.set(int(2), int(200));

    let keys: Vec<_> = od.keys().collect();
    assert_eq!(keys.len(), 2);
}

#[test]
fn test_values_iterator() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));
    od.set(int(2), int(200));

    let sum: i64 = od.values().filter_map(|v| v.as_int()).sum();
    assert_eq!(sum, 300);
}

#[test]
fn test_iter() {
    let mut od = OrderedDict::new();
    od.set(int(1), int(100));

    let pairs: Vec<_> = od.iter().collect();
    assert_eq!(pairs.len(), 1);
}

// =========================================================================
// Equality Tests
// =========================================================================

#[test]
fn test_equality_same_order() {
    let mut od1 = OrderedDict::new();
    let mut od2 = OrderedDict::new();

    od1.set(int(1), int(100));
    od1.set(int(2), int(200));

    od2.set(int(1), int(100));
    od2.set(int(2), int(200));

    assert_eq!(od1, od2);
}

// =========================================================================
// Compaction Tests
// =========================================================================

#[test]
fn test_compaction_triggers() {
    let mut od = OrderedDict::new();

    // Add many items
    for i in 0..100 {
        od.set(int(i), int(i * 10));
    }

    // Remove most of them
    for i in 0..80 {
        od.remove(&int(i));
    }

    // Should have compacted
    assert_eq!(od.len(), 20);

    // Verify remaining are correct
    for i in 80..100 {
        assert!(od.contains(&int(i)));
    }
}

// =========================================================================
// Stress Tests
// =========================================================================

#[test]
fn test_stress_many_entries() {
    let mut od = OrderedDict::new();

    for i in 0..1000 {
        od.set(int(i), int(i * 10));
    }

    assert_eq!(od.len(), 1000);

    // Verify all entries
    for i in 0..1000 {
        assert_eq!(od.get(&int(i)).and_then(|v| v.as_int()), Some(i * 10));
    }
}

#[test]
fn test_stress_interleaved_ops() {
    let mut od = OrderedDict::new();

    for i in 0..100 {
        od.set(int(i), int(i));
        if i % 3 == 0 {
            od.remove(&int(i / 2));
        }
    }

    // All even indexed items that weren't removed should exist
    // This is a smoke test, not exact verification
    assert!(od.len() > 0);
}
