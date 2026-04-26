
use super::*;
use prism_core::intern::intern;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn str_val(s: &str) -> Value {
    Value::string(intern(s))
}

// =========================================================================
// CmpKey basic tests
// =========================================================================

#[test]
fn test_cmp_key_equal() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let k1 = CmpKey::new(int(5), int_cmp);
    let k2 = CmpKey::new(int(5), int_cmp);
    assert_eq!(k1, k2);
}

#[test]
fn test_cmp_key_less() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let k1 = CmpKey::new(int(3), int_cmp);
    let k2 = CmpKey::new(int(5), int_cmp);
    assert!(k1 < k2);
}

#[test]
fn test_cmp_key_greater() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let k1 = CmpKey::new(int(7), int_cmp);
    let k2 = CmpKey::new(int(3), int_cmp);
    assert!(k1 > k2);
}

#[test]
fn test_cmp_key_ordering() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let k1 = CmpKey::new(int(1), int_cmp);
    let k2 = CmpKey::new(int(2), int_cmp);
    assert_eq!(k1.cmp(&k2), Ordering::Less);
    assert_eq!(k2.cmp(&k1), Ordering::Greater);
}

#[test]
fn test_cmp_key_value_access() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let key = CmpKey::new(int(42), int_cmp);
    assert_eq!(key.value().as_int(), Some(42));
}

#[test]
fn test_cmp_key_into_value() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let key = CmpKey::new(int(42), int_cmp);
    let val = key.into_value();
    assert_eq!(val.as_int(), Some(42));
}

// =========================================================================
// CmpKey sorting tests
// =========================================================================

#[test]
fn test_cmp_key_sort_ascending() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let mut keys: Vec<CmpKey<_>> = vec![int(3), int(1), int(4), int(1), int(5)]
        .into_iter()
        .map(|v| CmpKey::new(v, int_cmp))
        .collect();
    keys.sort();

    let sorted: Vec<i64> = keys.iter().map(|k| k.value().as_int().unwrap()).collect();
    assert_eq!(sorted, vec![1, 1, 3, 4, 5]);
}

#[test]
fn test_cmp_key_sort_descending() {
    let reverse_cmp = |a: &Value, b: &Value| b.as_int().unwrap() - a.as_int().unwrap();
    let mut keys: Vec<CmpKey<_>> = vec![int(3), int(1), int(4)]
        .into_iter()
        .map(|v| CmpKey::new(v, reverse_cmp))
        .collect();
    keys.sort();

    let sorted: Vec<i64> = keys.iter().map(|k| k.value().as_int().unwrap()).collect();
    assert_eq!(sorted, vec![4, 3, 1]);
}

#[test]
fn test_cmp_key_sort_absolute_value() {
    let abs_cmp = |a: &Value, b: &Value| a.as_int().unwrap().abs() - b.as_int().unwrap().abs();
    let mut keys: Vec<CmpKey<_>> = vec![int(-3), int(1), int(-2), int(4)]
        .into_iter()
        .map(|v| CmpKey::new(v, abs_cmp))
        .collect();
    keys.sort();

    let sorted: Vec<i64> = keys.iter().map(|k| k.value().as_int().unwrap()).collect();
    assert_eq!(sorted, vec![1, -2, -3, 4]);
}

// =========================================================================
// cmp_to_key factory tests
// =========================================================================

#[test]
fn test_cmp_to_key_factory() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let key_fn = cmp_to_key(int_cmp);

    let k1 = key_fn(int(3));
    let k2 = key_fn(int(5));
    assert!(k1 < k2);
}

#[test]
fn test_cmp_to_key_sort() {
    let int_cmp = |a: &Value, b: &Value| a.as_int().unwrap() - b.as_int().unwrap();
    let key_fn = cmp_to_key(int_cmp);

    let mut keys: Vec<_> = vec![int(5), int(2), int(8), int(1)]
        .into_iter()
        .map(key_fn)
        .collect();
    keys.sort();

    let sorted: Vec<i64> = keys.iter().map(|k| k.value().as_int().unwrap()).collect();
    assert_eq!(sorted, vec![1, 2, 5, 8]);
}

// =========================================================================
// sort_with_cmp tests
// =========================================================================

#[test]
fn test_sort_with_cmp_ascending() {
    let mut values = vec![int(3), int(1), int(4), int(1), int(5)];
    sort_with_cmp(&mut values, |a, b| {
        a.as_int().unwrap() - b.as_int().unwrap()
    });
    let sorted: Vec<i64> = values.iter().filter_map(|v| v.as_int()).collect();
    assert_eq!(sorted, vec![1, 1, 3, 4, 5]);
}

#[test]
fn test_sort_with_cmp_descending() {
    let mut values = vec![int(3), int(1), int(4)];
    sort_with_cmp(&mut values, |a, b| {
        b.as_int().unwrap() - a.as_int().unwrap()
    });
    let sorted: Vec<i64> = values.iter().filter_map(|v| v.as_int()).collect();
    assert_eq!(sorted, vec![4, 3, 1]);
}

#[test]
fn test_sort_with_cmp_empty() {
    let mut values: Vec<Value> = vec![];
    sort_with_cmp(&mut values, |a, b| {
        a.as_int().unwrap() - b.as_int().unwrap()
    });
    assert!(values.is_empty());
}

#[test]
fn test_sort_with_cmp_single() {
    let mut values = vec![int(42)];
    sort_with_cmp(&mut values, |a, b| {
        a.as_int().unwrap() - b.as_int().unwrap()
    });
    assert_eq!(values[0].as_int(), Some(42));
}

#[test]
fn test_sort_with_cmp_already_sorted() {
    let mut values = vec![int(1), int(2), int(3), int(4)];
    sort_with_cmp(&mut values, |a, b| {
        a.as_int().unwrap() - b.as_int().unwrap()
    });
    let sorted: Vec<i64> = values.iter().filter_map(|v| v.as_int()).collect();
    assert_eq!(sorted, vec![1, 2, 3, 4]);
}

#[test]
fn test_sort_with_cmp_reverse_sorted() {
    let mut values = vec![int(4), int(3), int(2), int(1)];
    sort_with_cmp(&mut values, |a, b| {
        a.as_int().unwrap() - b.as_int().unwrap()
    });
    let sorted: Vec<i64> = values.iter().filter_map(|v| v.as_int()).collect();
    assert_eq!(sorted, vec![1, 2, 3, 4]);
}

// =========================================================================
// TotalOrdering from __lt__ tests
// =========================================================================

#[test]
fn test_total_ordering_from_lt_all_ops() {
    let ordering = TotalOrdering::new(ComparisonBase::Lt);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let lt = |a: &Value, b: &Value| a.as_int().unwrap() < b.as_int().unwrap();

    // 3 < 5
    assert!(ordering.lt(&int(3), &int(5), &eq, &lt));
    assert!(ordering.le(&int(3), &int(5), &eq, &lt));
    assert!(!ordering.gt(&int(3), &int(5), &eq, &lt));
    assert!(!ordering.ge(&int(3), &int(5), &eq, &lt));
}

#[test]
fn test_total_ordering_from_lt_equal() {
    let ordering = TotalOrdering::new(ComparisonBase::Lt);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let lt = |a: &Value, b: &Value| a.as_int().unwrap() < b.as_int().unwrap();

    // 5 == 5
    assert!(!ordering.lt(&int(5), &int(5), &eq, &lt));
    assert!(ordering.le(&int(5), &int(5), &eq, &lt));
    assert!(!ordering.gt(&int(5), &int(5), &eq, &lt));
    assert!(ordering.ge(&int(5), &int(5), &eq, &lt));
}

#[test]
fn test_total_ordering_from_lt_greater() {
    let ordering = TotalOrdering::new(ComparisonBase::Lt);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let lt = |a: &Value, b: &Value| a.as_int().unwrap() < b.as_int().unwrap();

    // 7 > 2
    assert!(!ordering.lt(&int(7), &int(2), &eq, &lt));
    assert!(!ordering.le(&int(7), &int(2), &eq, &lt));
    assert!(ordering.gt(&int(7), &int(2), &eq, &lt));
    assert!(ordering.ge(&int(7), &int(2), &eq, &lt));
}

// =========================================================================
// TotalOrdering from __le__ tests
// =========================================================================

#[test]
fn test_total_ordering_from_le_all_ops() {
    let ordering = TotalOrdering::new(ComparisonBase::Le);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let le = |a: &Value, b: &Value| a.as_int().unwrap() <= b.as_int().unwrap();

    // 3 < 5
    assert!(ordering.lt(&int(3), &int(5), &eq, &le));
    assert!(ordering.le(&int(3), &int(5), &eq, &le));
    assert!(!ordering.gt(&int(3), &int(5), &eq, &le));
    assert!(!ordering.ge(&int(3), &int(5), &eq, &le));
}

#[test]
fn test_total_ordering_from_le_equal() {
    let ordering = TotalOrdering::new(ComparisonBase::Le);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let le = |a: &Value, b: &Value| a.as_int().unwrap() <= b.as_int().unwrap();

    assert!(!ordering.lt(&int(5), &int(5), &eq, &le));
    assert!(ordering.le(&int(5), &int(5), &eq, &le));
    assert!(!ordering.gt(&int(5), &int(5), &eq, &le));
    assert!(ordering.ge(&int(5), &int(5), &eq, &le));
}

// =========================================================================
// TotalOrdering from __gt__ tests
// =========================================================================

#[test]
fn test_total_ordering_from_gt_all_ops() {
    let ordering = TotalOrdering::new(ComparisonBase::Gt);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let gt = |a: &Value, b: &Value| a.as_int().unwrap() > b.as_int().unwrap();

    // 7 > 3
    assert!(!ordering.lt(&int(7), &int(3), &eq, &gt));
    assert!(!ordering.le(&int(7), &int(3), &eq, &gt));
    assert!(ordering.gt(&int(7), &int(3), &eq, &gt));
    assert!(ordering.ge(&int(7), &int(3), &eq, &gt));
}

#[test]
fn test_total_ordering_from_gt_less() {
    let ordering = TotalOrdering::new(ComparisonBase::Gt);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let gt = |a: &Value, b: &Value| a.as_int().unwrap() > b.as_int().unwrap();

    assert!(ordering.lt(&int(2), &int(5), &eq, &gt));
    assert!(ordering.le(&int(2), &int(5), &eq, &gt));
    assert!(!ordering.gt(&int(2), &int(5), &eq, &gt));
    assert!(!ordering.ge(&int(2), &int(5), &eq, &gt));
}

// =========================================================================
// TotalOrdering from __ge__ tests
// =========================================================================

#[test]
fn test_total_ordering_from_ge_all_ops() {
    let ordering = TotalOrdering::new(ComparisonBase::Ge);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let ge = |a: &Value, b: &Value| a.as_int().unwrap() >= b.as_int().unwrap();

    // 7 > 3
    assert!(!ordering.lt(&int(7), &int(3), &eq, &ge));
    assert!(!ordering.le(&int(7), &int(3), &eq, &ge));
    assert!(ordering.gt(&int(7), &int(3), &eq, &ge));
    assert!(ordering.ge(&int(7), &int(3), &eq, &ge));
}

#[test]
fn test_total_ordering_from_ge_equal() {
    let ordering = TotalOrdering::new(ComparisonBase::Ge);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let ge = |a: &Value, b: &Value| a.as_int().unwrap() >= b.as_int().unwrap();

    assert!(!ordering.lt(&int(5), &int(5), &eq, &ge));
    assert!(ordering.le(&int(5), &int(5), &eq, &ge));
    assert!(!ordering.gt(&int(5), &int(5), &eq, &ge));
    assert!(ordering.ge(&int(5), &int(5), &eq, &ge));
}

// =========================================================================
// derive_all tests
// =========================================================================

#[test]
fn test_derive_all_from_lt() {
    let ordering = TotalOrdering::new(ComparisonBase::Lt);
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
    let lt = |a: &Value, b: &Value| a.as_int().unwrap() < b.as_int().unwrap();

    let (lt_r, le_r, gt_r, ge_r) = ordering.derive_all(&int(3), &int(5), &eq, &lt);
    assert!(lt_r);
    assert!(le_r);
    assert!(!gt_r);
    assert!(!ge_r);

    let (lt_r, le_r, gt_r, ge_r) = ordering.derive_all(&int(5), &int(5), &eq, &lt);
    assert!(!lt_r);
    assert!(le_r);
    assert!(!gt_r);
    assert!(ge_r);
}

// =========================================================================
// TotalOrdering completeness tests (all 4 bases × 3 orderings)
// =========================================================================

#[test]
fn test_total_ordering_consistency_all_bases() {
    // For every comparison base, verify all derivations are consistent
    let bases = [
        ComparisonBase::Lt,
        ComparisonBase::Le,
        ComparisonBase::Gt,
        ComparisonBase::Ge,
    ];
    let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();

    for base in &bases {
        let ordering = TotalOrdering::new(*base);
        let base_fn: Box<dyn Fn(&Value, &Value) -> bool> = match base {
            ComparisonBase::Lt => {
                Box::new(|a: &Value, b: &Value| a.as_int().unwrap() < b.as_int().unwrap())
            }
            ComparisonBase::Le => {
                Box::new(|a: &Value, b: &Value| a.as_int().unwrap() <= b.as_int().unwrap())
            }
            ComparisonBase::Gt => {
                Box::new(|a: &Value, b: &Value| a.as_int().unwrap() > b.as_int().unwrap())
            }
            ComparisonBase::Ge => {
                Box::new(|a: &Value, b: &Value| a.as_int().unwrap() >= b.as_int().unwrap())
            }
        };

        // Test all orderings: a < b, a == b, a > b
        for (a, b) in [(3, 5), (5, 5), (7, 5)] {
            let av = int(a);
            let bv = int(b);

            let lt = ordering.lt(&av, &bv, &eq, &*base_fn);
            let le = ordering.le(&av, &bv, &eq, &*base_fn);
            let gt = ordering.gt(&av, &bv, &eq, &*base_fn);
            let ge = ordering.ge(&av, &bv, &eq, &*base_fn);

            // Consistency checks
            assert_eq!(lt, a < b, "lt failed for {:?} with a={}, b={}", base, a, b);
            assert_eq!(le, a <= b, "le failed for {:?} with a={}, b={}", base, a, b);
            assert_eq!(gt, a > b, "gt failed for {:?} with a={}, b={}", base, a, b);
            assert_eq!(ge, a >= b, "ge failed for {:?} with a={}, b={}", base, a, b);

            // Logical invariants
            assert_eq!(lt, !ge, "lt != !ge for {:?}", base);
            assert_eq!(le, !gt, "le != !gt for {:?}", base);
            assert_eq!(le, lt || eq(&av, &bv), "le != (lt || eq) for {:?}", base);
            assert_eq!(ge, gt || eq(&av, &bv), "ge != (gt || eq) for {:?}", base);
        }
    }
}

// =========================================================================
// Stress tests
// =========================================================================

#[test]
fn test_sort_with_cmp_large() {
    let mut values: Vec<Value> = (0..1000).rev().map(|i| int(i)).collect();
    sort_with_cmp(&mut values, |a, b| {
        a.as_int().unwrap() - b.as_int().unwrap()
    });

    for (i, v) in values.iter().enumerate() {
        assert_eq!(v.as_int(), Some(i as i64));
    }
}

#[test]
fn test_cmp_key_sort_stability() {
    // Verify stable sort: elements with equal keys maintain original order
    let remainder_cmp =
        |a: &Value, b: &Value| (a.as_int().unwrap() % 3) - (b.as_int().unwrap() % 3);

    let mut keys: Vec<CmpKey<_>> = vec![int(1), int(4), int(7), int(2), int(5), int(8)]
        .into_iter()
        .map(|v| CmpKey::new(v, remainder_cmp))
        .collect();
    keys.sort();

    // All have remainder 1, then 2, then 0
    // Within each group, original order is preserved
    let sorted: Vec<i64> = keys.iter().map(|k| k.value().as_int().unwrap()).collect();
    // remainder 0: none in this set, remainder 1: [1, 4, 7], remainder 2: [2, 5, 8]
    assert_eq!(sorted, vec![1, 4, 7, 2, 5, 8]);
}
