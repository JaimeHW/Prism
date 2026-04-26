use super::*;

#[test]
fn test_poly_entry_new() {
    let entry = PolyIcEntry::new(ShapeId(42), 100u32);
    assert_eq!(entry.shape_id, ShapeId(42));
    assert_eq!(entry.handler, 100);
    assert!(!entry.is_empty());
}

#[test]
fn test_poly_entry_empty() {
    let entry: PolyIcEntry<u32> = PolyIcEntry::empty();
    assert!(entry.is_empty());
    assert_eq!(entry.handler, 0);
}

#[test]
fn test_poly_entry_matches() {
    let entry = PolyIcEntry::new(ShapeId(10), 5u32);
    assert!(entry.matches(ShapeId(10)));
    assert!(!entry.matches(ShapeId(11)));
}

#[test]
fn test_poly_entry_touch() {
    let mut entry = PolyIcEntry::new(ShapeId(1), 1u32);
    assert_eq!(entry.access_count, 0);
    entry.touch();
    assert_eq!(entry.access_count, 1);
}

#[test]
fn test_poly_ic_new() {
    let ic: PolyIc<u32> = PolyIc::new();
    assert!(ic.is_empty());
    assert!(!ic.is_full());
}

#[test]
fn test_poly_ic_insert_and_lookup() {
    let mut ic: PolyIc<u32> = PolyIc::new();

    assert!(ic.try_insert(ShapeId(1), 100));
    assert_eq!(ic.len(), 1);

    let result = ic.lookup(ShapeId(1));
    assert_eq!(result, Some(100));
}

#[test]
fn test_poly_ic_lookup_miss() {
    let mut ic: PolyIc<u32> = PolyIc::new();
    ic.try_insert(ShapeId(1), 100);

    assert!(ic.lookup(ShapeId(2)).is_none());
}

#[test]
fn test_poly_ic_multiple_entries() {
    let mut ic: PolyIc<u32> = PolyIc::new();

    ic.try_insert(ShapeId(1), 10);
    ic.try_insert(ShapeId(2), 20);
    ic.try_insert(ShapeId(3), 30);
    ic.try_insert(ShapeId(4), 40);

    assert_eq!(ic.len(), 4);
    assert!(ic.is_full());

    assert_eq!(ic.lookup(ShapeId(1)), Some(10));
    assert_eq!(ic.lookup(ShapeId(4)), Some(40));
}

#[test]
fn test_poly_ic_insert_full() {
    let mut ic: PolyIc<u32> = PolyIc::new();

    for i in 1..=POLY_IC_ENTRIES as u32 {
        assert!(ic.try_insert(ShapeId(i), i * 10));
    }

    assert!(!ic.try_insert(ShapeId(99), 990));
}

#[test]
fn test_poly_ic_update_existing() {
    let mut ic: PolyIc<u32> = PolyIc::new();

    ic.try_insert(ShapeId(1), 100);
    assert_eq!(ic.lookup(ShapeId(1)), Some(100));

    // Update same key
    ic.try_insert(ShapeId(1), 200);
    assert_eq!(ic.lookup(ShapeId(1)), Some(200));
    assert_eq!(ic.len(), 1); // Still only 1 entry
}

#[test]
fn test_poly_ic_replace_lru() {
    let mut ic: PolyIc<u32> = PolyIc::new();

    // Fill cache
    for i in 1..=POLY_IC_ENTRIES as u32 {
        ic.try_insert(ShapeId(i), i * 10);
    }

    // Touch all except first
    for i in 2..=POLY_IC_ENTRIES as u32 {
        ic.lookup_and_touch(ShapeId(i));
        ic.lookup_and_touch(ShapeId(i));
    }

    // Replace LRU
    ic.replace_lru(ShapeId(99), 990);

    // First should be replaced
    assert!(ic.lookup(ShapeId(1)).is_none());
    assert_eq!(ic.lookup(ShapeId(99)), Some(990));
}

#[test]
fn test_poly_ic_clear() {
    let mut ic: PolyIc<u32> = PolyIc::new();
    ic.try_insert(ShapeId(1), 10);
    ic.try_insert(ShapeId(2), 20);

    ic.clear();

    assert!(ic.is_empty());
    assert!(ic.lookup(ShapeId(1)).is_none());
}

#[test]
fn test_poly_ic_record_miss() {
    let mut ic: PolyIc<u32> = PolyIc::new();

    assert_eq!(ic.miss_count(), 0);
    ic.record_miss();
    assert_eq!(ic.miss_count(), 1);
    ic.record_miss();
    assert_eq!(ic.miss_count(), 2);
}

#[test]
fn test_poly_ic_iter() {
    let mut ic: PolyIc<u32> = PolyIc::new();
    ic.try_insert(ShapeId(1), 10);
    ic.try_insert(ShapeId(2), 20);

    let entries: Vec<_> = ic.iter().collect();
    assert_eq!(entries.len(), 2);
}

#[test]
fn test_poly_ic_with_pointer_type() {
    // Test with a pointer-like type
    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    struct FnPtr(usize);

    let mut ic: PolyIc<FnPtr> = PolyIc::new();
    ic.try_insert(ShapeId(1), FnPtr(0x1000));
    ic.try_insert(ShapeId(2), FnPtr(0x2000));

    assert_eq!(ic.lookup(ShapeId(1)), Some(FnPtr(0x1000)));
    assert_eq!(ic.lookup(ShapeId(2)), Some(FnPtr(0x2000)));
}
