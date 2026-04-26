use super::*;

#[test]
fn test_remembered_set_creation() {
    let rs = RememberedSet::new();
    assert!(rs.is_empty());
    assert_eq!(rs.len(), 0);
}

#[test]
fn test_insert_single() {
    let rs = RememberedSet::new();
    let ptr = 0x1000 as *const ();
    rs.insert(ptr);
    assert_eq!(rs.len(), 1);
    assert!(!rs.is_empty());
}

#[test]
fn test_insert_multiple() {
    let rs = RememberedSet::new();
    for i in 0..100 {
        rs.insert((0x1000 + i * 8) as *const ());
    }
    assert_eq!(rs.len(), 100);
}

#[test]
fn test_drain_returns_all_entries() {
    let rs = RememberedSet::new();
    for i in 0..10 {
        rs.insert((0x1000 + i * 64) as *const ());
    }

    let entries = rs.drain();
    assert_eq!(entries.len(), 10);
    assert!(rs.is_empty());
}

#[test]
fn test_drain_deduplicates() {
    let rs = RememberedSet::new();
    let ptr = 0x2000 as *const ();

    // Insert the same pointer multiple times
    for _ in 0..50 {
        rs.insert(ptr);
    }

    let entries = rs.drain();
    // After dedup, should be exactly 1
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].holder, 0x2000);
}

#[test]
fn test_drain_preserves_unique_entries() {
    let rs = RememberedSet::new();

    // Insert 5 unique + duplicates
    for i in 0..5 {
        let ptr = (0x1000 + i * 512) as *const ();
        rs.insert(ptr);
        rs.insert(ptr); // duplicate
    }

    let entries = rs.drain();
    assert_eq!(entries.len(), 5);
}

#[test]
fn test_drain_sorted_order() {
    let rs = RememberedSet::new();

    // Insert in reverse order
    rs.insert(0x3000 as *const ());
    rs.insert(0x1000 as *const ());
    rs.insert(0x2000 as *const ());

    let entries = rs.drain();
    assert_eq!(entries[0].holder, 0x1000);
    assert_eq!(entries[1].holder, 0x2000);
    assert_eq!(entries[2].holder, 0x3000);
}

#[test]
fn test_clear() {
    let rs = RememberedSet::new();
    for i in 0..20 {
        rs.insert((0x1000 + i * 8) as *const ());
    }
    assert_eq!(rs.len(), 20);

    rs.clear();
    assert!(rs.is_empty());
    assert_eq!(rs.len(), 0);
}

#[test]
fn test_should_compress() {
    let rs = RememberedSet::new();
    assert!(!rs.should_compress());

    // Fill to capacity
    for i in 0..BUFFER_CAPACITY {
        rs.insert((0x1000 + i * 8) as *const ());
    }
    assert!(rs.should_compress());
}

#[test]
fn test_drain_then_insert() {
    let rs = RememberedSet::new();
    rs.insert(0x1000 as *const ());
    rs.insert(0x2000 as *const ());

    let entries = rs.drain();
    assert_eq!(entries.len(), 2);

    // Insert after drain
    rs.insert(0x3000 as *const ());
    assert_eq!(rs.len(), 1);

    let entries2 = rs.drain();
    assert_eq!(entries2.len(), 1);
    assert_eq!(entries2[0].holder, 0x3000);
}

#[test]
fn test_concurrent_insert_simulation() {
    // Simulate what happens when multiple threads insert
    let rs = RememberedSet::new();

    // Main thread inserts
    for i in 0..50 {
        rs.insert((0x1000 + i * 8) as *const ());
    }

    assert_eq!(rs.len(), 50);
    let entries = rs.drain();
    assert_eq!(entries.len(), 50);
}

#[test]
fn test_overflow_during_drain() {
    let rs = RememberedSet::new();

    // Insert before drain
    rs.insert(0x1000 as *const ());
    rs.insert(0x2000 as *const ());

    // Simulate draining flag being set
    rs.draining.store(true, Ordering::Release);

    // These should go to overflow
    rs.insert(0x3000 as *const ());
    rs.insert(0x4000 as *const ());

    // Reset draining
    rs.draining.store(false, Ordering::Release);

    // Drain should get all entries
    let entries = rs.drain();
    assert_eq!(entries.len(), 4);
}

#[test]
fn test_large_volume() {
    let rs = RememberedSet::new();

    // Insert many entries
    for i in 0..10_000 {
        rs.insert((0x10000 + i * 8) as *const ());
    }

    assert_eq!(rs.len(), 10_000);
    let entries = rs.drain();
    assert_eq!(entries.len(), 10_000);
    assert!(rs.is_empty());
}
